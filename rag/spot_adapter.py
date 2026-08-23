"""
Adapter for the native Spot project pipelines.

Ragora and the Spot project both use top-level module names such as
``config`` and ``pipeline``. This adapter loads Spot's modules in a short
isolated import window so the shared UI can delegate to Spot without copying
Spot retrieval logic into Ragora's Clearpath pipeline.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional
from urllib.parse import quote

import numpy as np
from langchain_core.prompts import PromptTemplate

from config.settings import (
    BASE_DIR,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    OLLAMA_MODEL,
    QDRANT_HOST,
    QDRANT_PORT,
    SPOT_MONGODB_COLLECTION,
    SPOT_MONGODB_DB_NAME,
    SPOT_MONGODB_URI,
    SPOT_QDRANT_COLLECTION,
)
from rag.context_safety import build_untrusted_context_block, sanitize_retrieved_text
from rag.model_utils import strip_thinking, with_no_think
from rag.spot_query_rewriter import SpotQueryRewriter


SPOT_QA_PROMPT = PromptTemplate(
    input_variables=["answer_language", "context", "question"],
    template="""Sen Ragora icinde Spot proje belgelerine bagli teknik asistansin.

Kurallar:
- Cevabi yalnizca verilen Spot baglamina dayanarak ver.
- Cevap dili zorunlu olarak {answer_language}.
- Cevap dili Turkish ise yalnizca Turkiye Turkcesi kullan; Ingilizce, Cince veya baska dilde cumle yazma.
- Dokumanlarda olmayan bilgiyi uydurma.
- Deger, birim, kosul ve uyari varsa aynen belirt.
- Soru genel Spot robotuyla ilgiliyse Spot robot kilavuzundaki degeri onceliklendir.
- Soru Spot Arm, gripper, end-effector veya kol diyorsa Spot Arm degerlerini kullan.
- Aksesuar degeri ile robot degeri farkliysa bunu ayirarak soyle.
- Yeterli bilgi yoksa bunu acikca soyle.

Spot belge baglami:
{context}

Soru: {question}

Cevap:""",
)


class SpotAdapterError(RuntimeError):
    """Raised when the native Spot adapter cannot be initialized."""


class SpotRAGAdapter:
    """Thin adapter around ``spot-rag-project-son/spot-rag`` retrieval code."""

    _SPOT_MODULE_ROOTS = (
        "chunkers",
        "config",
        "embedders",
        "parsers",
        "pipeline",
        "stores",
        "utils",
    )

    def __init__(self, llm=None, top_k: int = 5, score_threshold: float = 0.25):
        self.spot_root = BASE_DIR / "spot-rag-project-son" / "spot-rag"
        self.llm = llm
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._retriever = None
        self._visual_retriever = None
        self._visual_image_index = None
        self.query_rewriter = SpotQueryRewriter(llm=llm)

    def is_available(self) -> bool:
        return (self.spot_root / "pipeline" / "retriever.py").exists()

    def retrieve_context(self, query: str) -> Dict:
        chunks = self.retrieve(query)
        if not chunks:
            return {"context": "", "sources": [], "chunks": []}

        context_parts = []
        sources = []
        for index, chunk in enumerate(chunks, 1):
            section = " > ".join(getattr(chunk, "breadcrumb", []) or []) or "Genel"
            pages = getattr(chunk, "page_range", []) or []
            page_text = ""
            if len(pages) >= 2:
                page_text = f"s.{pages[0]}" if pages[0] == pages[1] else f"s.{pages[0]}-{pages[1]}"
            safe_text = sanitize_retrieved_text(getattr(chunk, "raw_text", "") or "")
            context_parts.append(
                f"Spot Dokuman {index} | Baslik: {section} | Sayfa: {page_text} "
                f"| Benzerlik: {float(getattr(chunk, 'score', 0.0)):.3f}\n{safe_text}"
            )
            sources.append(
                {
                    "source": getattr(chunk, "doc_name", "") or "Spot dokumani",
                    "source_url": self._pdf_source_url(chunk),
                    "category": "spot",
                    "section_title": section,
                    "similarity": self._display_score(index, len(chunks)),
                    "retrieval_score": float(getattr(chunk, "score", 0.0)),
                    "score_display": self._score_display(chunk),
                    "content_preview": safe_text[:200] + "...",
                    "images": [],
                    "page_range": pages,
                }
            )

        context = build_untrusted_context_block(
            context_parts,
            max_chars=MAX_CONTEXT_CHARS,
            label="Spot Technical Context",
        )
        return {"context": context, "sources": sources, "chunks": chunks}

    def retrieve(self, query: str) -> List:
        retriever = self._get_retriever()
        merged = {}
        for index, retrieval_query in enumerate(self._build_retrieval_queries(query)):
            # Spot documents are mostly English. auto_lang=True can detect Turkish
            # questions as "tr" and filter out all English chunks.
            for chunk in retriever.retrieve(query=retrieval_query, auto_lang=False):
                current = merged.get(chunk.chunk_id)
                adjusted_score = self._rerank_score(query, chunk, float(chunk.score), index)
                if current is None or adjusted_score > current[0]:
                    merged[chunk.chunk_id] = (adjusted_score, chunk)

        ranked = sorted(merged.values(), key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk in ranked[: self.top_k]:
            chunk.score = score
            results.append(chunk)
        return results

    def generate_answer(self, query: str) -> Dict:
        if not self.llm:
            return {
                "answer": "LLM servisi aktif degil. Lutfen Ollama'yi baslatin.",
                "sources": [],
                "error": True,
                "route": "spot",
            }

        retrieval = self.retrieve_context(query)
        if not retrieval["context"]:
            return {
                "answer": "Bu sorunun cevabini Spot dokumanlarinda bulamadim. Lutfen sorunuzu yeniden formule etmeyi deneyin.",
                "sources": [],
                "error": False,
                "route": "spot",
            }

        try:
            prompt = SPOT_QA_PROMPT.format(
                answer_language=self.query_rewriter.answer_language(query),
                context=retrieval["context"],
                question=query,
            )
            answer = strip_thinking(self.llm.invoke(with_no_think(prompt))).strip()
            return {
                "answer": answer,
                "sources": retrieval["sources"],
                "error": False,
                "route": "spot",
            }
        except Exception as exc:
            return {
                "answer": f"Spot cevabi uretilirken bir hata olustu: {exc}",
                "sources": retrieval["sources"],
                "error": True,
                "route": "spot",
            }

    def analyze_image_with_context(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        question: str = "",
    ) -> Dict:
        if not self.llm:
            return {
                "answer": "LLM servisi aktif degil. Lutfen Ollama'yi baslatin.",
                "sources": [],
                "similar_images": [],
                "error": True,
                "route": "spot",
            }

        try:
            visual = self._get_visual_retriever()
            image_input = image_bytes if image_bytes is not None else image_path
            hybrid = visual.retrieve_by_image(image_input=image_input, query_text=question)
            payload = hybrid.format_for_llm(include_images=False)
            context = build_untrusted_context_block(
                [sanitize_retrieved_text(payload.get("text_context", ""))],
                max_chars=MAX_CONTEXT_CHARS,
                label="Spot Visual Context",
            )
            sources = self._sources_from_spot_chunks(hybrid.text_chunks)
            similar_images = self._similar_images_from_visuals(hybrid.visual_chunks)

            if not context.strip():
                return self.generate_answer(question)

            prompt = SPOT_QA_PROMPT.format(
                answer_language=self.query_rewriter.answer_language(question),
                context=context,
                question=question,
            )
            answer = strip_thinking(self.llm.invoke(with_no_think(prompt))).strip()
            return {
                "answer": answer,
                "sources": sources,
                "similar_images": similar_images,
                "error": False,
                "route": "spot",
            }
        except Exception:
            # If Spot visual collections are not indexed yet, text Spot RAG is
            # still the correct fallback for a Spot question.
            return self.generate_answer(question)

    def probe_image(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None, question: str = "") -> Dict:
        """Return lightweight Spot visual evidence for visual routing."""
        try:
            visual = self._get_visual_retriever()
            image_input = image_bytes if image_bytes is not None else image_path
            hybrid = visual.retrieve_by_image(image_input=image_input, query_text=question)
            image_matches = self._search_spot_visual_images(
                image_path=image_path,
                image_bytes=image_bytes,
                limit=3,
            )
            best_visual = max((float(getattr(item, "score", 0.0)) for item in hybrid.visual_chunks), default=0.0)
            best_text = max((float(getattr(item, "score", 0.0)) for item in hybrid.text_chunks), default=0.0)
            best_image = max((float(item.get("score", 0.0)) for item in image_matches), default=0.0)
            return {
                "route": "spot",
                "has_evidence": bool(hybrid.visual_chunks or hybrid.text_chunks or image_matches),
                "best_visual_score": best_visual,
                "best_text_score": best_text,
                "best_image_score": best_image,
                "visual_count": len(hybrid.visual_chunks),
                "text_count": len(hybrid.text_chunks),
                "image_match_count": len(image_matches),
                "image_matches": image_matches,
            }
        except Exception as exc:
            return {
                "route": "spot",
                "has_evidence": False,
                "best_visual_score": 0.0,
                "best_text_score": 0.0,
                "best_image_score": 0.0,
                "visual_count": 0,
                "text_count": 0,
                "image_match_count": 0,
                "image_matches": [],
                "error": str(exc),
            }

    def _search_spot_visual_images(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        limit: int = 3,
    ) -> List[Dict]:
        visual = self._get_visual_retriever()
        query_vec = (
            visual._clip.embed_image_bytes(image_bytes)
            if image_bytes is not None
            else visual._clip.embed_image_path(image_path)
        )
        candidates = self._get_visual_image_index()
        scored = []
        for item in candidates:
            score = float(np.dot(query_vec, item["vector"]))
            scored.append({**item["metadata"], "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def _get_visual_image_index(self) -> List[Dict]:
        if self._visual_image_index is not None:
            return self._visual_image_index

        visual = self._get_visual_retriever()
        docs = list(
            visual._vis_mongo._col.find(
                {"image_b64": {"$type": "string", "$ne": ""}},
                {
                    "_id": 0,
                    "element_id": 1,
                    "doc_name": 1,
                    "page": 1,
                    "figure_caption": 1,
                    "visual_type": 1,
                    "image_b64": 1,
                },
            )
        )
        index = []
        for doc in docs:
            try:
                vec = visual._clip.embed_image_b64(doc["image_b64"])
            except Exception:
                continue
            index.append(
                {
                    "vector": vec,
                    "metadata": {
                        "element_id": doc.get("element_id", ""),
                        "doc_name": doc.get("doc_name", ""),
                        "page": doc.get("page", 0),
                        "caption": doc.get("figure_caption", ""),
                        "visual_type": doc.get("visual_type", ""),
                    },
                }
            )

        self._visual_image_index = index
        return self._visual_image_index

    def _sources_from_spot_chunks(self, chunks: List) -> List[Dict]:
        sources = []
        total = len(chunks or [])
        for index, chunk in enumerate(chunks or [], 1):
            safe_text = sanitize_retrieved_text(getattr(chunk, "raw_text", "") or "")
            sources.append(
                {
                    "source": getattr(chunk, "doc_name", "") or "Spot dokumani",
                    "source_url": self._pdf_source_url(chunk),
                    "category": "spot",
                    "section_title": " > ".join(getattr(chunk, "breadcrumb", []) or []),
                    "similarity": self._display_score(index, total),
                    "retrieval_score": float(getattr(chunk, "score", 0.0)),
                    "score_display": self._score_display(chunk),
                    "content_preview": safe_text[:200] + "...",
                    "images": [],
                    "page_range": getattr(chunk, "page_range", []),
                }
            )
        return sources

    def _pdf_source_url(self, chunk) -> str:
        doc_name = getattr(chunk, "doc_name", "") or ""
        if not doc_name:
            return ""
        pages = getattr(chunk, "page_range", []) or []
        page_fragment = f"#page={pages[0]}" if pages else ""
        return f"/app/static/spot-pdfs/{quote(f'{doc_name}.pdf')}{page_fragment}"

    def _display_score(self, index: int, total: int) -> float:
        if total <= 1:
            return 1.0
        return max(0.72, 0.98 - (index - 1) * 0.06)

    def _score_display(self, chunk) -> str:
        return f"Uyusma skoru: {float(getattr(chunk, 'score', 0.0)):.4f}"

    def _build_retrieval_queries(self, query: str) -> List[str]:
        return self.query_rewriter.build_retrieval_queries(query)

    def _rerank_score(self, query: str, chunk, base_score: float, query_index: int) -> float:
        score = base_score + (0.002 / (query_index + 1))
        doc_name = getattr(chunk, "doc_name", "") or ""
        text = self.query_rewriter.normalize_text(
            f"{doc_name} {' '.join(getattr(chunk, 'breadcrumb', []) or [])} {getattr(chunk, 'raw_text', '')}"
        )

        for rank, prefix in enumerate(self.query_rewriter.preferred_doc_prefixes(query)):
            if doc_name.startswith(prefix):
                score += 0.018 - min(rank, 3) * 0.004
                break

        if not self.query_rewriter.query_mentions_arm(query) and doc_name.startswith("spot-arm-user-manual-en"):
            score -= 0.012

        for keyword in self.query_rewriter.answer_keywords(query):
            if keyword in text:
                score += 0.004

        if "table of contents" in text:
            score -= 0.02

        return score

    def _similar_images_from_visuals(self, visuals: List) -> List[Dict]:
        items = []
        for visual in visuals or []:
            image_b64 = getattr(visual, "image_b64", "") or ""
            items.append(
                {
                    "image_url": f"data:image/jpeg;base64,{image_b64}" if image_b64 else "",
                    "category": "spot",
                    "similarity": float(getattr(visual, "score", 0.0)),
                    "caption": getattr(visual, "figure_caption", "") or getattr(visual, "doc_name", ""),
                }
            )
        return items

    def _get_retriever(self):
        if self._retriever is None:
            self._ensure_available()
            self._configure_spot_environment()
            with self._isolated_spot_imports():
                from pipeline.retriever import Retriever

                self._retriever = Retriever(
                    top_k=self.top_k,
                    use_hybrid=True,
                    score_threshold=self.score_threshold,
                )
        return self._retriever

    def _get_visual_retriever(self):
        if self._visual_retriever is None:
            self._ensure_available()
            self._configure_spot_environment()
            with self._isolated_spot_imports():
                from pipeline.visual_retriever import VisualRetriever

                self._visual_retriever = VisualRetriever(
                    top_k_text=self.top_k,
                    top_k_visual=3,
                )
        return self._visual_retriever

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise SpotAdapterError(f"Spot project not found at {self.spot_root}")

    def _configure_spot_environment(self) -> None:
        os.environ.setdefault("MONGO_URI", SPOT_MONGODB_URI)
        os.environ.setdefault("MONGO_DB", SPOT_MONGODB_DB_NAME)
        os.environ.setdefault("MONGO_COLLECTION", SPOT_MONGODB_COLLECTION)
        os.environ.setdefault("QDRANT_HOST", QDRANT_HOST)
        os.environ.setdefault("QDRANT_PORT", str(QDRANT_PORT))
        os.environ.setdefault("QDRANT_COLLECTION", SPOT_QDRANT_COLLECTION)
        os.environ.setdefault("VISUAL_MONGO_COLLECTION", "spot_visual_collection")
        os.environ.setdefault("VISUAL_QDRANT_COLLECTION", "spot_visual_chunks")
        os.environ.setdefault("EMBED_MODEL", EMBEDDING_MODEL)
        os.environ.setdefault("EMBED_DEVICE", "cpu")
        os.environ.setdefault("OLLAMA_MODEL", OLLAMA_MODEL)

    @contextmanager
    def _isolated_spot_imports(self):
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if self._is_spot_module_name(name)
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)

        root_str = str(self.spot_root)
        original_sys_path = list(sys.path)
        sys.path = [path for path in sys.path if path != root_str]
        sys.path.insert(0, root_str)

        try:
            yield
        finally:
            sys.path = original_sys_path
            for name in [name for name in sys.modules if self._is_spot_module_name(name)]:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def _is_spot_module_name(self, name: str) -> bool:
        return any(name == root or name.startswith(f"{root}.") for root in self._SPOT_MODULE_ROOTS)
