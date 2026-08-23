"""
Qdrant vektör store.

Rol: Embedding vektörlerini ve chunk_id'leri saklar.
Semantic search sonucunda chunk_id'ler döner,
MongoDB'den tam metin alınır.

Collection şeması
-----------------
Point:
  id: UUID string (chunk_id)
  vector: float32[1024]  (bge-large boyutu)
  payload:
    doc_name: str
    page_start: int
    page_end: int
    chunk_type: str
    breadcrumb_str: str   ← filtered search için
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from chunkers.semantic_chunker import Chunk


class QdrantVectorStore:
    """
    Qdrant'a vektör yazar ve semantik arama yapar.

    Parameters
    ----------
    host : str
    port : int
    collection_name : str
    vector_dim : int
        Embedding modeli boyutu (bge-large=1024, bge-base=768).
    """

    BATCH_SIZE = 256  # Qdrant upsert batch boyutu

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "spot_chunks",
        vector_dim: int = 1024,
    ):
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection_name
        self._dim = vector_dim
        self._ensure_collection()
        logger.info(
            f"Qdrant bağlandı: {host}:{port}/{collection_name} (dim={vector_dim})"
        )

    def _ensure_collection(self) -> None:
        """Collection yoksa oluşturur."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Qdrant collection oluşturuldu: {self._collection}")
        else:
            logger.debug(f"Qdrant collection mevcut: {self._collection}")

    def upsert_embeddings(
        self, chunks: list[Chunk], embeddings: np.ndarray
    ) -> None:
        """
        Chunk'ları ve embedding vektörlerini Qdrant'a yazar.

        Parameters
        ----------
        chunks : list[Chunk]
            Chunk metadata'ları (payload için).
        embeddings : np.ndarray
            Shape: (len(chunks), dim)
        """
        assert len(chunks) == len(embeddings), "Chunk ve embedding sayısı eşleşmiyor"

        points = []
        for chunk, vec in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,  # UUID string
                    vector=vec.tolist(),
                    payload={
                        "doc_name": chunk.doc_name,
                        "page_start": chunk.page_range[0],
                        "page_end": chunk.page_range[1],
                        "chunk_type": chunk.chunk_type,
                        "breadcrumb_str": " > ".join(chunk.breadcrumb),
                        "lang": getattr(chunk, "lang", "en"),
                    },
                )
            )

        # Batch upsert
        for i in tqdm(
            range(0, len(points), self.BATCH_SIZE),
            desc="Qdrant upsert",
            unit="batch",
        ):
            batch = points[i : i + self.BATCH_SIZE]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        logger.success(f"Qdrant: {len(points)} vektör upsert edildi")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.3,
        filter_docs: list[str] | None = None,
        filter_chunk_type: str | None = None,
        filter_langs: list[str] | None = None,
    ) -> list[dict]:
        """
        Semantik arama yapar.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape: (dim,)
        top_k : int
            Döndürülecek maksimum sonuç sayısı.
        score_threshold : float
            Bu değerin altındaki sonuçları at.
        filter_docs : list[str], optional
            Sadece bu dokümanlardan ara. None → tümünde ara.
        filter_langs : list[str], optional
            Sadece bu dillerde ara. Örn: ["de", "en"] veya ["tr"]
            None → tüm diller.
        filter_chunk_type : str, optional
            "text" | "table" | "list"

        Returns
        -------
        list[dict]
            [{"chunk_id": ..., "score": ..., "payload": ...}, ...]
        """
        from qdrant_client.models import MatchAny

        # Filtre oluştur
        conditions = []
        if filter_docs:
            if len(filter_docs) == 1:
                conditions.append(
                    FieldCondition(key="doc_name", match=MatchValue(value=filter_docs[0]))
                )
            else:
                conditions.append(
                    FieldCondition(key="doc_name", match=MatchAny(any=filter_docs))
                )
        if filter_chunk_type:
            conditions.append(
                FieldCondition(
                    key="chunk_type", match=MatchValue(value=filter_chunk_type)
                )
            )
        if filter_langs:
            if len(filter_langs) == 1:
                conditions.append(
                    FieldCondition(key="lang", match=MatchValue(value=filter_langs[0]))
                )
            else:
                conditions.append(
                    FieldCondition(key="lang", match=MatchAny(any=filter_langs))
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        from qdrant_client.models import Query

        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "chunk_id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in result.points
        ]

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        """
        Belirtilen chunk_id'lere sahip point'leri siler.
        Doküman yeniden indexleme veya silme sırasında kullanılır.
        """
        if not chunk_ids:
            return
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=chunk_ids),
        )
        logger.info(f"Qdrant: {len(chunk_ids)} vektör silindi")

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)
        logger.warning(f"Qdrant collection silindi: {self._collection}")
        self._ensure_collection()
