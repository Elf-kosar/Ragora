"""
Question router/orchestrator for the shared Ragora UI.

Clearpath questions stay on Ragora-Web's existing pipeline. Spot questions are
delegated to the native Spot project adapter. Comparative questions retrieve
context from both sides and produce one answer.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from config.settings import CATEGORIES, MAX_CONTEXT_CHARS
from rag.context_safety import build_untrusted_context_block, sanitize_retrieved_text
from rag.model_utils import strip_thinking, with_no_think
from rag.rag_pipeline import RAGSystem
from rag.spot_adapter import SpotRAGAdapter


COMPARISON_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Sen Ragora'nin ortak teknik asistani olarak cevap veriyorsun.

Kurallar:
- Clearpath ve Spot baglamlarini ayri kaynaklar olarak ele al.
- Karsilastirma isteniyorsa farklari ve ortak noktalari acik belirt.
- Baglamda olmayan bilgiyi uydurma.
- Soru Turkce ise Turkce cevap ver.

Birlesik teknik baglam:
{context}

Soru: {question}

Cevap:""",
)


class UnifiedRAGRouter:
    """Routes user questions to Clearpath, Spot, or both."""

    SPOT_TERMS = (
        "spot",
        "boston dynamics",
        "spot arm",
        "spot dock",
        "spot cam",
        "spot camera",
        "estop",
        "e stop",
        "e-stop",
    )
    COMPARISON_TERMS = (
        "karsilastir",
        "karsilastirma",
        "kiyasla",
        "fark",
        "farki",
        "benzer",
        "versus",
        " vs ",
        "compare",
        "comparison",
        "difference",
        "similar",
    )

    def __init__(self):
        self.clearpath_rag = RAGSystem()
        self.spot_rag = SpotRAGAdapter(llm=self.clearpath_rag.llm)
        self.llm = self.clearpath_rag.llm
        self.clearpath_categories = {
            key
            for key, meta in CATEGORIES.items()
            if meta.get("project") != "spot" and key != "spot"
        }
        self.clearpath_aliases = self._build_clearpath_aliases()

    def route_question(self, query: str, category: Optional[str] = None) -> str:
        if category == "spot":
            return "spot"
        if category in self.clearpath_categories:
            return "clearpath"

        has_spot = self._has_spot_signal(query)
        clearpath_hits = self._infer_clearpath_categories(query)
        has_compare = self._has_comparison_signal(query)

        if has_spot and clearpath_hits and has_compare:
            return "both"
        if has_spot:
            return "spot"
        return "clearpath"

    def generate_answer(self, query: str, category: Optional[str] = None) -> Dict:
        route = self.route_question(query, category)
        if route == "spot":
            return self.spot_rag.generate_answer(query)
        if route == "both":
            return self._generate_comparison_answer(query)

        clearpath_category = category if category in self.clearpath_categories else None
        result = self.clearpath_rag.generate_answer(query, category=clearpath_category)
        result["route"] = "clearpath"
        return result

    def analyze_image_with_context(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        question: str = "",
        category: Optional[str] = None,
    ) -> Dict:
        route = self.route_question(question, category)
        if route == "spot":
            return self.spot_rag.analyze_image_with_context(
                image_path=image_path,
                image_bytes=image_bytes,
                question=question,
            )
        if category is None:
            spot_probe = self.spot_rag.probe_image(
                image_path=image_path,
                image_bytes=image_bytes,
                question=question,
            )
            if self._is_confident_spot_visual_match(spot_probe):
                return self.spot_rag.analyze_image_with_context(
                    image_path=image_path,
                    image_bytes=image_bytes,
                    question=question,
                )
        return {
            "answer": "",
            "sources": [],
            "similar_images": [],
            "error": True,
            "route": route,
            "delegate_to_clearpath_vision": True,
        }

    def close(self):
        self.clearpath_rag.close()

    def _generate_comparison_answer(self, query: str) -> Dict:
        if not self.llm:
            return {
                "answer": "LLM servisi aktif degil. Lutfen Ollama'yi baslatin.",
                "sources": [],
                "error": True,
                "route": "both",
            }

        contexts = []
        sources = []

        for category in self._infer_clearpath_categories(query) or [None]:
            docs = self.clearpath_rag.retrieve_relevant_documents(query, category=category)
            clear_context, clear_sources = self._format_clearpath_context(docs)
            if clear_context:
                label = CATEGORIES.get(category or "", {}).get("name", "Clearpath")
                contexts.append(f"[Clearpath / {label}]\n{clear_context}")
                sources.extend(clear_sources)

        spot = self.spot_rag.retrieve_context(query)
        if spot["context"]:
            contexts.append(f"[Spot]\n{spot['context']}")
            sources.extend(spot["sources"])

        if not contexts:
            return {
                "answer": "Bu sorunun cevabini mevcut dokumanlarda bulamadim. Lutfen sorunuzu yeniden formule etmeyi deneyin.",
                "sources": [],
                "error": False,
                "route": "both",
            }

        context = build_untrusted_context_block(
            contexts,
            max_chars=MAX_CONTEXT_CHARS,
            label="Unified Technical Context",
        )

        try:
            prompt = COMPARISON_PROMPT.format(context=context, question=query)
            answer = strip_thinking(self.llm.invoke(with_no_think(prompt))).strip()
            return {
                "answer": answer,
                "sources": sources,
                "error": False,
                "route": "both",
            }
        except Exception as exc:
            return {
                "answer": f"Karsilastirmali cevap uretilirken bir hata olustu: {exc}",
                "sources": sources,
                "error": True,
                "route": "both",
            }

    def _format_clearpath_context(self, docs: List[Tuple[Dict, float]]) -> Tuple[str, List[Dict]]:
        context_parts = []
        sources = []
        for index, (doc, score) in enumerate(docs or [], 1):
            safe_text = sanitize_retrieved_text(doc.get("content", ""))
            context_parts.append(
                f"Clearpath Dokuman {index} | Kategori: {doc.get('category', 'Bilinmiyor')} "
                f"| Baslik: {doc.get('section_title', '') or 'Bilinmiyor'} "
                f"| Benzerlik: {float(score):.3f}\n{safe_text}"
            )
            sources.append(
                {
                    "source": doc.get("source", "Bilinmiyor"),
                    "category": doc.get("category", "Bilinmiyor"),
                    "section_title": doc.get("section_title", ""),
                    "similarity": float(score),
                    "content_preview": safe_text[:200] + "...",
                    "images": doc.get("images", []),
                }
            )
        return "\n\n---\n\n".join(context_parts), sources

    def _build_clearpath_aliases(self) -> Dict[str, List[str]]:
        aliases = {}
        for key, meta in CATEGORIES.items():
            if key == "spot" or meta.get("project") == "spot":
                continue
            name = str(meta.get("name", "")).lower()
            pieces = {key.replace("_", " "), key}
            pieces.update(token for token in re.split(r"[^a-zA-Z0-9]+", name) if token)
            aliases[key] = sorted(pieces, key=len, reverse=True)
        return aliases

    def _infer_clearpath_categories(self, query: str) -> List[str]:
        normalized = self._normalize(query)
        hits = []
        for category, aliases in self.clearpath_aliases.items():
            if any(self._term_in_text(alias, normalized) for alias in aliases):
                hits.append(category)
        return hits

    def _has_spot_signal(self, query: str) -> bool:
        normalized = self._normalize(query)
        return any(self._term_in_text(term, normalized) for term in self.SPOT_TERMS)

    def _has_comparison_signal(self, query: str) -> bool:
        normalized = f" {self._normalize(query)} "
        return any(term in normalized for term in self.COMPARISON_TERMS)

    def _is_confident_spot_visual_match(self, probe: Dict) -> bool:
        if not probe.get("has_evidence"):
            return False
        best_image = float(probe.get("best_image_score", 0.0))
        best_visual = float(probe.get("best_visual_score", 0.0))
        best_text = float(probe.get("best_text_score", 0.0))
        visual_count = int(probe.get("visual_count", 0))
        text_count = int(probe.get("text_count", 0))
        return (
            best_image >= 0.86
            or best_visual >= 0.24
            or (visual_count >= 2 and text_count >= 1 and best_text >= 0.35)
        )

    def _term_in_text(self, term: str, text: str) -> bool:
        term = self._normalize(term)
        if not term:
            return False
        if " " in term or "-" in term:
            return term in text
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.casefold().replace("ı", "i")
