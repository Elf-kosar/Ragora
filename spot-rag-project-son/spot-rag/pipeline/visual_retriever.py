"""
Visual Retriever — görsel sorgu ile hem metin hem görsel chunk'ları getirir.

3 Sorgu Modu
------------

1. Metin sorgu → metin chunk'lar + görsel chunk'lar
   "E-Stop nerede?"
   → BGE embed → Qdrant metin search
   → CLIP text embed → VisualQdrant search
   → Her ikisi RRF ile birleştirilir

2. Görsel sorgu → benzer görseller + ilgili metin chunk'lar
   [kullanıcı görsel gönderir]
   → CLIP image embed → VisualQdrant search (görsel benzerlik)
   → En yakın görsel sayfanın breadcrumb'ı ile metin Qdrant search
   → Her ikisi RRF ile birleştirilir

3. Görsel + metin sorgu (en güçlü)
   "Bu görseldeki E-Stop düğmesi nasıl kullanılır?"
   → CLIP image embed → görsel benzerlik arama
   → Metin sorgusu → BGE embed → metin arama
   → Üçlü RRF birleştirme

Dönen Sonuç
-----------
HybridResult:
  - text_chunks: list[RetrievedChunk]   ← ilgili metin/tablo chunk'lar
  - visual_chunks: list[RetrievedVisual] ← ilgili görsel chunk'lar (image_b64 dahil)
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import settings
from embedders.clip_embedder import CLIPEmbedder
from embedders.local_embedder import LocalEmbedder
from pipeline.retriever import RetrievedChunk, Retriever
from utils.lang_utils import detect_lang_from_text
from stores.mongo_store import MongoChunkStore
from stores.qdrant_store import QdrantVectorStore
from stores.visual_store import VisualMongoStore, VisualQdrantStore


@dataclass
class RetrievedVisual:
    """Görsel retrieval sonucu."""
    element_id: str
    score: float
    page: int
    doc_name: str
    figure_caption: str
    vlm_description: str
    image_b64: str           # base64 JPEG — doğrudan multimodal LLM'e verilebilir
    visual_type: str
    breadcrumb: list[str]
    source: str              # "image_query" | "text_query" | "page_match"


@dataclass
class HybridResult:
    """
    Metin + görsel birleşik retrieval sonucu.
    """
    query_text: str
    query_image_b64: Optional[str]       # sorgu görseli (varsa)
    text_chunks: list[RetrievedChunk]    # ilgili metin chunk'lar
    visual_chunks: list[RetrievedVisual] # ilgili görsel chunk'lar

    def has_visuals(self) -> bool:
        return len(self.visual_chunks) > 0

    def format_for_llm(self, include_images: bool = True) -> dict:
        """
        Multimodal LLM'e gönderilecek payload oluşturur.

        Returns
        -------
        dict:
          "text_context": str   ← metin chunk'ları birleştirilmiş
          "images": [           ← görsel base64'ler (multimodal LLM için)
            {"b64": ..., "caption": ..., "description": ..., "page": ...}
          ]
        """
        # Metin context
        text_parts = []
        for i, chunk in enumerate(self.text_chunks, 1):
            section = " > ".join(chunk.breadcrumb) if chunk.breadcrumb else "Genel"
            pages = f"s.{chunk.page_range[0]}"
            if chunk.page_range[0] != chunk.page_range[1]:
                pages = f"s.{chunk.page_range[0]}-{chunk.page_range[1]}"
            header = f"[{i}] {section} ({pages})"
            if chunk.chunk_type == "table":
                header += " [TABLO]"
            text_parts.append(f"{header}\n{chunk.raw_text}")

        text_context = "\n\n---\n\n".join(text_parts)

        # Görsel payload
        images = []
        if include_images:
            for vis in self.visual_chunks:
                images.append({
                    "b64": vis.image_b64,
                    "caption": vis.figure_caption,
                    "description": vis.vlm_description,
                    "page": vis.page,
                    "section": " > ".join(vis.breadcrumb),
                    "type": vis.visual_type,
                })

        return {
            "text_context": text_context,
            "images": images,
        }


class VisualRetriever:
    """
    Metin ve/veya görsel sorgu ile hibrit retrieval.

    Parameters
    ----------
    top_k_text : int
        Kaç metin chunk döndürülsün.
    top_k_visual : int
        Kaç görsel chunk döndürülsün.
    """

    def __init__(
        self,
        top_k_text: int = 5,
        top_k_visual: int = 3,
        score_threshold_text: float = 0.3,
        score_threshold_visual: float = 0.2,
    ):
        self.top_k_text = top_k_text
        self.top_k_visual = top_k_visual

        # Metin retrieval (mevcut sistem)
        self._text_retriever = Retriever(
            top_k=top_k_text,
            use_hybrid=True,
            score_threshold=score_threshold_text,
        )

        # CLIP embedder (görsel + metin cross-modal)
        self._clip = CLIPEmbedder(
            model_name=settings.clip_model,
            device=settings.embed_device,
        )

        # Görsel store'lar
        self._vis_mongo = VisualMongoStore(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
            collection_name=settings.visual_mongo_collection,
        )
        self._vis_qdrant = VisualQdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.visual_qdrant_collection,
            vector_dim=self._clip.dimension,
        )

        # Text Qdrant (görsel sayfa → ilgili metin için)
        self._text_qdrant = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection,
            vector_dim=self._text_retriever.embedder.dimension,
        )
        self._text_mongo = MongoChunkStore(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
            collection_name=settings.mongo_collection,
        )

    # ─────────────────────────────────────────────────────────────────
    # Ana retrieval metotları
    # ─────────────────────────────────────────────────────────────────

    def retrieve_by_text(
        self,
        query: str,
        filter_docs: list[str] | None = None,
        filter_langs: list[str] | None = None,
        auto_lang: bool = True,
    ) -> HybridResult:
        """
        Metin sorgusu → metin chunk'lar + ilgili görsel chunk'lar.

        "E-Stop nasıl kullanılır?" gibi normal RAG sorguları için.
        """
        logger.debug(f"Text query: '{query[:60]}...'")

        # 1. Metin retrieval (mevcut BGE + Qdrant)
        # auto_lang devre dışı — multilingual-e5-large Türkçe sorguyu İngilizce
        # chunk'larla çapraz dil olarak eşleştirir, dil filtresi sonuçları keser
        text_chunks = self._text_retriever.retrieve(
            query=query,
            filter_docs=filter_docs,
            filter_langs=filter_langs,
            auto_lang=False,
        )

        # RRF skorları (0.01-0.04) hardcoded sibling/table skorlarıyla (0.5-0.75)
        # uyumsuz — primary chunk'lar merge sonrası top_k dışına düşüyor.
        # Normalize: en iyi chunk → 1.0, en kötü → 0.55 (sibling 0.5 üstü).
        if text_chunks:
            max_s = max(c.score for c in text_chunks) or 1.0
            for c in text_chunks:
                c.score = 0.55 + 0.45 * (c.score / max_s)

        # 2. Aynı sorgu ile görsel retrieval (CLIP text embed)
        clip_vec = self._clip.embed_texts([query])[0]
        visual_chunks = self._search_visuals(
            query_vec=clip_vec,
            top_k=self.top_k_visual,
            filter_docs=filter_docs,
            source="text_query",
        )

        # 3. Metin chunk sayfalarından görsel ara (sayfa bazlı eşleştirme)
        page_visuals = self._find_visuals_for_pages(
            text_chunks=text_chunks,
            filter_docs=filter_docs,
        )

        # 4. Görsel sonuçları birleştir (dedup)
        visual_chunks = self._merge_visuals(visual_chunks, page_visuals)

        # 5. CLIP'in bulduğu görsellerin sayfalarından metin chunk'ları da getir
        # (örn: clearance diyagramı s.26 → o sayfanın metin chunk'ı)
        visual_page_text = self._find_text_chunks_for_visual_pages(visual_chunks, filter_docs)
        text_chunks = self._merge_text_chunks(text_chunks, visual_page_text)

        # 6. Bulunan chunk'ların sayfasındaki kardeş chunk'ları da ekle
        sibling_chunks = self._fetch_sibling_chunks(text_chunks)
        text_chunks = self._merge_text_chunks(text_chunks, sibling_chunks)

        # 7. Alakalı dokümanların tablo chunk'larını zorunlu ekle
        # (tablo markdown embedding semantik aramada zayıf → top-k'ya girmeyebilir)
        table_chunks = self._fetch_table_chunks_from_docs(text_chunks, query)
        if table_chunks:
            seen_ids = {c.chunk_id for c in text_chunks}
            for tc in table_chunks:
                if tc.chunk_id not in seen_ids:
                    text_chunks.append(tc)

        return HybridResult(
            query_text=query,
            query_image_b64=None,
            text_chunks=text_chunks,
            visual_chunks=visual_chunks,
        )

    def retrieve_by_image(
        self,
        image_input: str | bytes | Path,
        query_text: str = "",
        filter_docs: list[str] | None = None,
    ) -> HybridResult:
        """
        Görsel sorgu → benzer görsel chunk'lar + ilgili metin chunk'lar.

        Kullanıcı bir görsel gönderdiğinde: "Bu nedir?" veya
        "Bu görseldeki parçanın bakımını nasıl yaparım?"

        Parameters
        ----------
        image_input : str | bytes | Path
            Base64 string, ham bytes, veya dosya yolu.
        query_text : str
            İsteğe bağlı metin açıklaması (varsa daha iyi sonuç).
        """
        logger.debug("Image query başlatılıyor...")

        # Girişi normalize et
        image_b64 = self._normalize_image_input(image_input)

        # 1. CLIP image embed
        image_vec = self._clip.embed_image_b64(image_b64)

        # 2. Görsel benzerlik arama
        visual_chunks = self._search_visuals(
            query_vec=image_vec,
            top_k=self.top_k_visual,
            filter_docs=filter_docs,
            source="image_query",
        )

        # 3. En yakın görsel sayfaların breadcrumb'ı ile metin arama
        text_chunks = self._find_text_for_visuals(
            visual_chunks=visual_chunks,
            query_text=query_text,
            filter_docs=filter_docs,
        )

        # 4. Metin sorgusu da varsa birleştir (opsiyonel)
        if query_text:
            extra_text = self._text_retriever.retrieve(
                query=query_text,
                filter_docs=filter_docs,
                auto_lang=False,
            )
            text_chunks = self._merge_text_chunks(text_chunks, extra_text)

        # 5. Bulunan chunk'ların sayfasındaki kardeş chunk'ları da ekle
        sibling_chunks = self._fetch_sibling_chunks(text_chunks)
        text_chunks = self._merge_text_chunks(text_chunks, sibling_chunks)

        return HybridResult(
            query_text=query_text,
            query_image_b64=image_b64,
            text_chunks=text_chunks,
            visual_chunks=visual_chunks,
        )

    def retrieve_by_image_and_text(
        self,
        image_input: str | bytes | Path,
        query_text: str,
        filter_docs: list[str] | None = None,
    ) -> HybridResult:
        """
        Görsel + metin birleşik sorgu (en güçlü mod).

        "Bu görseldeki E-Stop nasıl kullanılır?" gibi sorgular için.
        Görsel ile PDF'deki benzer sayfayı bul, metin ile de ilgili
        bölümleri getir.
        """
        return self.retrieve_by_image(
            image_input=image_input,
            query_text=query_text,
            filter_docs=filter_docs,
        )

    # ─────────────────────────────────────────────────────────────────
    # Görsel arama
    # ─────────────────────────────────────────────────────────────────

    def _search_visuals(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filter_docs: list[str] | None,
        source: str,
    ) -> list[RetrievedVisual]:
        """CLIP vektörü ile görsel Qdrant'ta arama yapar."""
        hits = self._vis_qdrant.search(
            query_vector=query_vec,
            top_k=top_k,
            filter_docs=filter_docs,
        )
        return self._hydrate_visuals(hits, source=source)

    def _hydrate_visuals(
        self, hits: list[dict], source: str
    ) -> list[RetrievedVisual]:
        """Qdrant hit'lerini MongoDB'den tam veriye dönüştürür."""
        if not hits:
            return []

        element_ids = [h["element_id"] for h in hits]
        score_map = {h["element_id"]: h["score"] for h in hits}

        docs = self._vis_mongo.get_by_ids(element_ids)

        results = []
        for doc in docs:
            results.append(
                RetrievedVisual(
                    element_id=doc["element_id"],
                    score=score_map.get(doc["element_id"], 0.0),
                    page=doc.get("page", 0),
                    doc_name=doc.get("doc_name", ""),
                    figure_caption=doc.get("figure_caption", ""),
                    vlm_description=doc.get("vlm_description", ""),
                    image_b64=doc.get("image_b64", ""),
                    visual_type=doc.get("visual_type", "unknown"),
                    breadcrumb=doc.get("breadcrumb", []),
                    source=source,
                )
            )
        return results

    # ─────────────────────────────────────────────────────────────────
    # Sayfa bazlı metin-görsel eşleştirme
    # ─────────────────────────────────────────────────────────────────

    def _find_visuals_for_pages(
        self,
        text_chunks: list[RetrievedChunk],
        filter_docs: list[str] | None,
    ) -> list[RetrievedVisual]:
        """
        Metin chunk sayfalarında görsel var mı diye bakar.

        Örn: "1.5.1 Spot Anatomy" bölümünün 9. sayfasında diyagram var,
        metin chunk bu sayfayı içeriyorsa o diyagramı da getir.
        """
        results = []
        seen_pages: set[tuple[str, int]] = set()
        seen_element_ids: set[str] = set()

        # Sadece semantik arama sonuçlarından görsel üret
        # table_fetch / page_sibling chunk'ları alakasız sayfalardan görsel getirir
        visual_source_chunks = [
            c for c in text_chunks
            if c.source in ("hybrid", "semantic")
        ] or text_chunks

        # Görsel için tek doküman kullan — en yüksek skorlu chunk'ın dökümanı
        if visual_source_chunks:
            top_chunk = max(visual_source_chunks, key=lambda c: c.score)
            top_visual_doc = top_chunk.doc_name
            visual_source_chunks = [c for c in visual_source_chunks if c.doc_name == top_visual_doc]

        for chunk in visual_source_chunks:
            doc_name = chunk.doc_name
            for page in range(chunk.page_range[0], chunk.page_range[1] + 1):
                key = (doc_name, page)
                if key in seen_pages:
                    continue
                seen_pages.add(key)

                visual_docs = self._vis_mongo.get_by_page(doc_name, page)
                for visual_doc in visual_docs:
                    eid = visual_doc["element_id"]
                    if eid in seen_element_ids:
                        continue
                    seen_element_ids.add(eid)
                    results.append(
                        RetrievedVisual(
                            element_id=eid,
                            score=0.75,  # sayfa eşleşmesi için sabit skor
                            page=visual_doc["page"],
                            doc_name=visual_doc["doc_name"],
                            figure_caption=visual_doc.get("figure_caption", ""),
                            vlm_description=visual_doc.get("vlm_description", ""),
                            image_b64=visual_doc.get("image_b64", ""),
                            visual_type=visual_doc.get("visual_type", "unknown"),
                            breadcrumb=visual_doc.get("breadcrumb", []),
                            source="page_match",
                        )
                    )

        return results

    def _find_text_for_visuals(
        self,
        visual_chunks: list[RetrievedVisual],
        query_text: str,
        filter_docs: list[str] | None,
    ) -> list[RetrievedChunk]:
        """
        Görsel chunk'ların sayfalarındaki metin chunk'ları bulur.
        Görsel sorgu yapılırken ilgili metin bağlamını getirmek için.
        """
        if not visual_chunks:
            return []

        # En yakın görselin breadcrumb'ı ile metin arama
        top_visual = visual_chunks[0]
        breadcrumb_query = " ".join(top_visual.breadcrumb)

        # breadcrumb + query_text birleşimi ile arama
        combined_query = f"{breadcrumb_query} {query_text}".strip()
        if not combined_query:
            combined_query = top_visual.figure_caption or top_visual.vlm_description[:200]

        if combined_query:
            return self._text_retriever.retrieve(
                query=combined_query,
                filter_docs=filter_docs,
                auto_lang=False,
            )
        return []

    # ─────────────────────────────────────────────────────────────────
    # Dedup / Merge
    # ─────────────────────────────────────────────────────────────────

    def _merge_visuals(
        self,
        primary: list[RetrievedVisual],
        secondary: list[RetrievedVisual],
    ) -> list[RetrievedVisual]:
        """Görsel listelerini birleştirir, tekrarları kaldırır."""
        seen: set[str] = set()
        merged = []
        for vis in primary + secondary:
            if vis.element_id not in seen:
                seen.add(vis.element_id)
                merged.append(vis)
        # Skor sırasına göre sırala
        merged.sort(key=lambda v: v.score, reverse=True)
        return merged[:max(self.top_k_visual, 6)]

    def _find_text_chunks_for_visual_pages(
        self,
        visual_chunks: list[RetrievedVisual],
        filter_docs: list[str] | None,
    ) -> list[RetrievedChunk]:
        """
        CLIP'in bulduğu görsellerin sayfalarındaki metin chunk'larını getirir.
        Metin araması kaçıran ama görsel aramasının bulduğu sayfaları yakalar.
        """
        seen_pages: set[tuple[str, int]] = set()
        results = []
        for vis in visual_chunks[:3]:  # top-3 görsel yeterli
            key = (vis.doc_name, vis.page)
            if key in seen_pages:
                continue
            seen_pages.add(key)
            page_docs = self._text_mongo.get_chunks_by_page(vis.doc_name, vis.page)
            for doc in page_docs:
                if filter_docs and doc.get("doc_name") not in filter_docs:
                    continue
                results.append(RetrievedChunk(
                    chunk_id=doc["chunk_id"],
                    score=0.6,
                    raw_text=doc.get("raw_text", ""),
                    text=doc.get("text", ""),
                    breadcrumb=doc.get("breadcrumb", []),
                    page_range=doc.get("page_range", [vis.page, vis.page]),
                    chunk_type=doc.get("chunk_type", "text"),
                    doc_name=doc.get("doc_name", ""),
                    lang=doc.get("lang", "en"),
                    source="visual_page",
                ))
        return results

    def _fetch_table_chunks_from_docs(
        self, chunks: list[RetrievedChunk], query_text: str = ""
    ) -> list[RetrievedChunk]:
        """
        Hybrid/semantic retrieval'ın en çok döndürdüğü tek dokümanın
        tablo chunk'larını getirir (max 8 tablo).

        Tablo içeriği sorgu keyword'leriyle eşleşiyorsa score=0.75 (görüntülenir),
        eşleşmiyorsa score=0.30 (LLM bağlamında kalır ama Kaynak Tablolar'da
        render_inline_tables min_score=0.35 eşiği altında kalarak görünmez).
        """
        from collections import Counter
        hybrid_chunks = [c for c in chunks if c.source in ("hybrid", "semantic")]
        if not hybrid_chunks:
            return []

        doc_counts = Counter(c.doc_name for c in hybrid_chunks if c.doc_name)
        if not doc_counts:
            return []
        top_docs = [doc for doc, _ in doc_counts.most_common(2)]

        # 5+ harfli kelime kökleri (ilk 5 karakter) — basit stemming
        # Doküman isimlerinden gelen generik kelimeler her tabloda geçer —
        # tablo eşleştirmesinde false positive yaratmamak için çıkarılır
        import re as _re
        _DOC_STEMS = {
            "spot", "spots", "stati", "dock", "arm", "cam", "power", "suppl", "manua",
            # Çok genel terimler — her tabloda geçer, skor kirliliği yaratır
            "speci", "requi", "infor", "descri", "detai", "gener", "overv",
        }
        # Noktalama işaretlerini temizle, sonra stem al
        clean_words = [_re.sub(r"[^\w]", "", w).lower() for w in query_text.split()]
        all_stems = {w[:5] for w in clean_words if len(w) >= 5}
        query_stems = all_stems - _DOC_STEMS or all_stems  # fallback: stem yoksa tümü kullan

        existing_ids = {c.chunk_id for c in chunks}
        results = []
        for top_doc in top_docs:
            for doc in self._text_mongo.get_table_chunks_by_doc(top_doc)[:6]:
                if doc["chunk_id"] in existing_ids:
                    continue
                if query_stems:
                    table_text = (
                        doc.get("raw_text", "") + " " + " ".join(doc.get("breadcrumb", []))
                    ).lower()
                    match_count = sum(
                        1 for stem in query_stems
                        if _re.search(r"\b" + _re.escape(stem), table_text)
                    )
                    required = min(2, len(query_stems))
                    score = 0.75 if match_count >= required else 0.30
                else:
                    score = 0.65
                existing_ids.add(doc["chunk_id"])
                results.append(RetrievedChunk(
                    chunk_id=doc["chunk_id"],
                    score=score,
                    raw_text=doc.get("raw_text", ""),
                    text=doc.get("text", ""),
                    breadcrumb=doc.get("breadcrumb", []),
                    page_range=doc.get("page_range", [0, 0]),
                    chunk_type="table",
                    doc_name=doc.get("doc_name", ""),
                    lang=doc.get("lang", "en"),
                    source="table_fetch",
                ))
        return results

    def _fetch_sibling_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        En alakalı top-3 chunk'ın sayfasındaki diğer chunk'ları getirir.
        Tüm chunk'lara uygulanırsa alakasız sayfalardan gürültü gelir.
        """
        # Her unique doc'tan en yüksek skorlu chunk + global top-3 için sibling çek
        seen_docs: set[str] = set()
        expanded: list[RetrievedChunk] = []
        for chunk in chunks[:3]:
            expanded.append(chunk)
        for chunk in chunks:
            if chunk.doc_name not in seen_docs:
                seen_docs.add(chunk.doc_name)
                if chunk not in expanded:
                    expanded.append(chunk)

        seen_pages: set[tuple[str, int]] = set()
        siblings = []
        for chunk in expanded:
            for page in range(chunk.page_range[0], chunk.page_range[1] + 1):
                key = (chunk.doc_name, page)
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                page_docs = self._text_mongo.get_chunks_by_page(chunk.doc_name, page)
                for doc in page_docs:
                    siblings.append(RetrievedChunk(
                        chunk_id=doc["chunk_id"],
                        score=0.5,
                        raw_text=doc.get("raw_text", ""),
                        text=doc.get("text", ""),
                        breadcrumb=doc.get("breadcrumb", []),
                        page_range=doc.get("page_range", [page, page]),
                        chunk_type=doc.get("chunk_type", "text"),
                        doc_name=doc.get("doc_name", ""),
                        lang=doc.get("lang", "en"),
                        source="page_sibling",
                    ))
        return siblings

    def _merge_text_chunks(
        self,
        primary: list[RetrievedChunk],
        secondary: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Metin chunk listelerini birleştirir, tekrarları kaldırır."""
        seen: set[str] = set()
        merged = []
        for chunk in primary + secondary:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)
        merged.sort(key=lambda c: c.score, reverse=True)
        return merged[: self.top_k_text]

    # ─────────────────────────────────────────────────────────────────
    # Görsel input normalize
    # ─────────────────────────────────────────────────────────────────

    def _normalize_image_input(
        self, image_input: str | bytes | Path
    ) -> str:
        """
        Farklı görsel input türlerini base64 string'e dönüştürür.
        """
        if isinstance(image_input, str):
            # Zaten base64 mi?
            try:
                base64.b64decode(image_input[:100])
                return image_input
            except Exception:
                # Dosya yolu olabilir
                return self._normalize_image_input(Path(image_input))

        if isinstance(image_input, Path):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        if isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")

        raise ValueError(f"Desteklenmeyen görsel input türü: {type(image_input)}")
