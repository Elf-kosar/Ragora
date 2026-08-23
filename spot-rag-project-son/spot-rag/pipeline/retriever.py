"""
Retriever — RAG pipeline'ının sorgulama katmanı.

Hybrid Search Stratejisi
------------------------
1. Qdrant semantic search  → vektör benzerliği
2. MongoDB full-text search → keyword eşleşmesi (isteğe bağlı)
3. RRF (Reciprocal Rank Fusion) ile birleştirme

Bu yaklaşım:
- "E-stop nasıl kullanılır?" → semantic search güçlü
- "SPOT-ENT-001 hata kodu" → keyword search güçlü
- İkisi birlikte: her iki durumu da yakalar
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config.settings import settings
from utils.lang_utils import detect_lang_from_text
from embedders.local_embedder import LocalEmbedder
from stores.mongo_store import MongoChunkStore
from stores.qdrant_store import QdrantVectorStore


@dataclass
class RetrievedChunk:
    """Retrieval sonucu."""
    chunk_id: str
    score: float          # birleştirilmiş skor
    raw_text: str
    text: str             # prefix dahil
    breadcrumb: list[str]
    page_range: list[int]
    chunk_type: str
    doc_name: str
    lang: str = "en"
    source: str = "semantic"


class Retriever:
    """
    Sorgulara karşı ilgili chunk'ları getirir.

    Parameters
    ----------
    top_k : int
        Kaç chunk döndürülsün.
    use_hybrid : bool
        True: semantic + keyword fusion
        False: sadece semantic
    """

    def __init__(
        self,
        top_k: int = 5,
        use_hybrid: bool = True,
        score_threshold: float = 0.3,
    ):
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self.score_threshold = score_threshold

        self.embedder = LocalEmbedder(
            model_name=settings.embed_model,
            device=settings.embed_device,
            batch_size=1,  # query için 1 yeterli
        )
        self.qdrant = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection,
            vector_dim=self.embedder.dimension,
        )
        self.mongo = MongoChunkStore(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
            collection_name=settings.mongo_collection,
        )

    def retrieve(
        self,
        query: str,
        filter_chunk_type: str | None = None,
        filter_docs: list[str] | None = None,
        filter_langs: list[str] | None = None,
        auto_lang: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Ana retrieval metodu.

        Parameters
        ----------
        query : str
            Kullanıcı sorusu.
        filter_chunk_type : str, optional
            Sadece belirli chunk türlerini getir.
        filter_docs : list[str], optional
            Sadece bu dokümanlardan ara. None → tüm dokümanlar.
        filter_langs : list[str], optional
            Dil filtresi. None + auto_lang=True → sorgu dilini otomatik tespit et.
        auto_lang : bool
            True → sorgu dili otomatik tespit edilir ve o dildeki
            dokümanlarda arama yapılır. filter_langs verilmişse devre dışı.

        Returns
        -------
        list[RetrievedChunk]
        """
        # Dil tespiti: filter_langs verilmemişse sorgu dilinden otomatik belirle
        if filter_langs is None and auto_lang:
            detected = detect_lang_from_text(query)
            filter_langs = [detected]
            logger.debug(f"Sorgu dili tespit edildi: '{detected}' → bu dildeki dokümanlarda aranıyor")

        # 1. Query embed
        query_vec = self.embedder.embed_query(query)

        # 2. Semantic search
        semantic_hits = self.qdrant.search(
            query_vector=query_vec,
            top_k=self.top_k * 2 if self.use_hybrid else self.top_k,
            score_threshold=self.score_threshold,
            filter_chunk_type=filter_chunk_type,
            filter_docs=filter_docs,
            filter_langs=filter_langs,
        )

        if not self.use_hybrid:
            return self._hydrate(semantic_hits, source="semantic")

        # 3. Keyword search (hybrid)
        keyword_hits = self.mongo.full_text_search(query, limit=self.top_k, filter_langs=filter_langs)

        # 4. RRF fusion
        fused = self._rrf_fusion(semantic_hits, keyword_hits)

        return self._hydrate(fused[: self.top_k], source="hybrid")

    def _rrf_fusion(
        self,
        semantic_hits: list[dict],
        keyword_hits: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        score_rrf(d) = Σ 1/(k + rank_i(d))

        k=60 standart değer (Cormack et al. 2009).
        """
        scores: dict[str, float] = {}

        for rank, hit in enumerate(semantic_hits):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

        for rank, doc in enumerate(keyword_hits):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)

        # Tüm chunk_id'leri topla
        all_ids = list(scores.keys())

        # Skor sırasına göre sırala
        sorted_ids = sorted(all_ids, key=lambda x: scores[x], reverse=True)

        return [{"chunk_id": cid, "score": scores[cid], "payload": {}} for cid in sorted_ids]

    def _hydrate(
        self, hits: list[dict], source: str
    ) -> list[RetrievedChunk]:
        """
        chunk_id listesini MongoDB'den tam dokümanlara dönüştürür.
        """
        if not hits:
            return []

        chunk_ids = [h["chunk_id"] for h in hits]
        score_map = {h["chunk_id"]: h["score"] for h in hits}

        docs = self.mongo.get_by_chunk_ids(chunk_ids)

        results = []
        for doc in docs:
            results.append(
                RetrievedChunk(
                    chunk_id=doc["chunk_id"],
                    score=score_map.get(doc["chunk_id"], 0.0),
                    raw_text=doc.get("raw_text", ""),
                    text=doc.get("text", ""),
                    breadcrumb=doc.get("breadcrumb", []),
                    page_range=doc.get("page_range", [0, 0]),
                    chunk_type=doc.get("chunk_type", "text"),
                    doc_name=doc.get("doc_name", ""),
                    lang=doc.get("lang", "en"),
                    source=source,
                )
            )

        return results

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Chunk'ları LLM prompt'una uygun context string'e dönüştürür.

        Her chunk için:
        - Hangi bölümde olduğu (breadcrumb)
        - Sayfa numarası
        - İçerik
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            section = " > ".join(chunk.breadcrumb) if chunk.breadcrumb else "Genel"
            pages = f"s.{chunk.page_range[0]}"
            if chunk.page_range[0] != chunk.page_range[1]:
                pages = f"s.{chunk.page_range[0]}-{chunk.page_range[1]}"

            header = f"[{i}] {section} ({pages})"
            if chunk.chunk_type == "table":
                header += " [TABLO]"

            parts.append(f"{header}\n{chunk.raw_text}")

        return "\n\n---\n\n".join(parts)
