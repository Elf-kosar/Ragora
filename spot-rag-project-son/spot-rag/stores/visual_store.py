"""
Visual Store — görsel chunk'lar için ayrı depolama.

Mimari
------
Görsel chunk'lar metin chunk'lardan AYRI collection'larda tutulur:

MongoDB  → spot_rag.visual_chunks
  - element_id, page, figure_caption, vlm_description,
    image_b64 (büyük alan), breadcrumb, doc_name, visual_type
  - image_b64 ayrı GridFS'e taşınabilir (>16MB PDF'ler için)

Qdrant   → spot_visual_chunks (ayrı collection, CLIP dim)
  - vector: CLIP embedding (768 dim, metin tarafı)
  - payload: element_id, page, doc_name, visual_type, has_vlm

Neden ayrı collection?
- Metin ve görsel boyutları farklı (BGE=1024, CLIP=768)
- Qdrant collection başına tek bir vektör boyutu destekler
- Görsel filtreleri (visual_type, has_vlm) metin arama ile karışmaz
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from parsers.visual_parser import VisualElement


class VisualMongoStore:
    """
    Görsel chunk metadatasını ve base64 görüntülerini MongoDB'de saklar.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "spot_rag",
        collection_name: str = "visual_chunks",
    ):
        self._client = MongoClient(uri)
        self._col: Collection = self._client[db_name][collection_name]
        self._ensure_indexes()
        logger.debug(f"VisualMongoStore hazır: {collection_name}")

    def _ensure_indexes(self) -> None:
        self._col.create_index("element_id", unique=True, background=True)
        self._col.create_index("doc_name", background=True)
        self._col.create_index("page", background=True)
        self._col.create_index("visual_type", background=True)
        self._col.create_index(
            [("vlm_description", "text"), ("figure_caption", "text")],
            name="visual_text_search",
            background=True,
        )

    def upsert_visuals(self, elements: list[VisualElement]) -> int:
        """Görsel elementleri toplu upsert eder."""
        if not elements:
            return 0

        ops = []
        now = datetime.now(tz=timezone.utc)

        for el in elements:
            doc = el.to_dict()
            doc["created_at"] = now
            ops.append(
                UpdateOne(
                    {"element_id": el.element_id},
                    {"$set": doc},
                    upsert=True,
                )
            )

        result = self._col.bulk_write(ops, ordered=False)
        total = result.upserted_count + result.modified_count
        logger.info(f"VisualMongo: {total} görsel upsert edildi")
        return total

    def get_by_ids(self, element_ids: list[str]) -> list[dict]:
        """element_id listesiyle görsel çekme."""
        docs = list(
            self._col.find(
                {"element_id": {"$in": element_ids}},
                {"_id": 0},
            )
        )
        order = {eid: i for i, eid in enumerate(element_ids)}
        docs.sort(key=lambda d: order.get(d["element_id"], 999))
        return docs

    def get_by_id(self, element_id: str) -> Optional[dict]:
        return self._col.find_one({"element_id": element_id}, {"_id": 0})

    def get_by_page(self, doc_name: str, page: int) -> list[dict]:
        """Belirli bir sayfanın tüm görsel chunk'larını getirir."""
        return list(self._col.find(
            {"doc_name": doc_name, "page": page}, {"_id": 0}
        ))

    def full_text_search(self, query: str, limit: int = 5) -> list[dict]:
        """VLM açıklaması ve figure caption üzerinde full-text search."""
        return list(
            self._col.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "_id": 0},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

    def get_element_ids_by_doc(self, doc_name: str) -> list[str]:
        return [
            d["element_id"]
            for d in self._col.find({"doc_name": doc_name}, {"element_id": 1, "_id": 0})
        ]

    def delete_by_doc(self, doc_name: str) -> int:
        result = self._col.delete_many({"doc_name": doc_name})
        logger.info(f"VisualMongo: '{doc_name}' için {result.deleted_count} görsel silindi")
        return result.deleted_count

    def count(self) -> int:
        return self._col.count_documents({})

    def close(self) -> None:
        self._client.close()


class VisualQdrantStore:
    """
    Görsel chunk CLIP embedding'lerini ayrı Qdrant collection'da saklar.
    """

    BATCH_SIZE = 128

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "spot_visual_chunks",
        vector_dim: int = 768,  # CLIP large = 768
    ):
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection_name
        self._dim = vector_dim
        self._ensure_collection()
        logger.debug(f"VisualQdrant hazır: {collection_name} (dim={vector_dim})")

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"VisualQdrant collection oluşturuldu: {self._collection}")

    def upsert_embeddings(
        self,
        elements: list[VisualElement],
        embeddings: np.ndarray,
    ) -> None:
        """Görsel CLIP embedding'lerini yazar."""
        assert len(elements) == len(embeddings)

        points = []
        for el, vec in zip(elements, embeddings):
            points.append(
                PointStruct(
                    id=el.element_id,  # element_id string UUID
                    vector=vec.tolist(),
                    payload={
                        "doc_name": el.doc_name,
                        "page": el.page,
                        "visual_type": el.visual_type,
                        "has_vlm": bool(el.vlm_description),
                        "figure_caption": el.figure_caption[:200],  # kısa özet
                    },
                )
            )

        for i in tqdm(
            range(0, len(points), self.BATCH_SIZE),
            desc="VisualQdrant upsert",
            unit="batch",
        ):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + self.BATCH_SIZE],
            )

        logger.success(f"VisualQdrant: {len(points)} görsel vektör upsert edildi")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.2,
        filter_docs: list[str] | None = None,
        filter_visual_type: str | None = None,
    ) -> list[dict]:
        """
        CLIP vektörü ile görsel arama.

        Hem metin sorgusu → CLIP text embed → görsel bul
        Hem görsel sorgu → CLIP image embed → benzer görsel bul
        şeklinde çalışır.
        """
        from qdrant_client.models import MatchAny

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
        if filter_visual_type:
            conditions.append(
                FieldCondition(key="visual_type", match=MatchValue(value=filter_visual_type))
            )

        qdrant_filter = Filter(must=conditions) if conditions else None

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
                "element_id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in result.points
        ]

    def delete_by_ids(self, element_ids: list[str]) -> None:
        if not element_ids:
            return
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=element_ids),
        )

    def count(self) -> int:
        return self._client.get_collection(self._collection).points_count

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)
        self._ensure_collection()
