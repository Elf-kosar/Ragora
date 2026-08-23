"""
MongoDB chunk store.

Rol: Ham chunk metinleri, metadata ve kaynak bilgilerini saklar.
Qdrant sadece vektörleri ve chunk_id'yi tutar.
Bir hit geldiğinde chunk_id ile MongoDB'den tam metni alırız.

Collection şeması
-----------------
{
  "_id": ObjectId,
  "chunk_id": str (UUID),       ← Qdrant ile join key
  "text": str,                  ← Prefix dahil zenginleştirilmiş metin
  "raw_text": str,              ← Ham metin
  "breadcrumb": [str],
  "breadcrumb_str": str,        ← "A > B > C" formatı (full-text search)
  "page_range": [int, int],
  "chunk_type": str,
  "doc_name": str,
  "created_at": datetime
}

Index'ler
---------
- chunk_id: unique (Qdrant lookup için)
- doc_name + page_range: sorgu filtresi
- breadcrumb_str: text search
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from chunkers.semantic_chunker import Chunk


class MongoChunkStore:
    """
    MongoDB'ye chunk yazar ve okur.

    Parameters
    ----------
    uri : str
        MongoDB bağlantı URI'si.
    db_name : str
        Veritabanı adı.
    collection_name : str
        Collection adı.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "spot_rag",
        collection_name: str = "chunks",
    ):
        self._client = MongoClient(uri)
        self._db = self._client[db_name]
        self._col: Collection = self._db[collection_name]
        self._ensure_indexes()
        logger.info(f"MongoDB bağlandı: {uri}/{db_name}.{collection_name}")

    def _ensure_indexes(self) -> None:
        """Gerekli index'leri oluşturur (idempotent)."""
        self._col.create_index("chunk_id", unique=True, background=True)
        self._col.create_index([("doc_name", 1), ("page_range", 1)], background=True)
        self._col.create_index("lang", background=True)
        self._col.create_index(
            [("breadcrumb_str", "text"), ("raw_text", "text")],
            name="full_text_search",
            background=True,
        )
        logger.debug("MongoDB index'leri hazır")

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Chunk'ları toplu olarak upsert eder.
        Aynı chunk_id varsa günceller, yoksa ekler.

        Returns
        -------
        int
            Eklenen/güncellenen kayıt sayısı.
        """
        if not chunks:
            return 0

        ops = []
        now = datetime.now(tz=timezone.utc)

        for chunk in chunks:
            doc = chunk.to_dict()
            doc["created_at"] = now
            ops.append(
                UpdateOne(
                    {"chunk_id": chunk.chunk_id},
                    {"$set": doc},
                    upsert=True,
                )
            )

        result = self._col.bulk_write(ops, ordered=False)
        total = result.upserted_count + result.modified_count
        logger.info(f"MongoDB: {total} chunk upsert edildi")
        return total

    def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        Qdrant'tan gelen chunk_id listesiyle chunk'ları getirir.
        RAG pipeline'ının retrieval adımında kullanılır.
        """
        docs = list(
            self._col.find(
                {"chunk_id": {"$in": chunk_ids}},
                {"_id": 0},
            )
        )
        # Girdi sırasını koru
        order = {cid: i for i, cid in enumerate(chunk_ids)}
        docs.sort(key=lambda d: order.get(d["chunk_id"], 999))
        return docs

    def get_by_chunk_id(self, chunk_id: str) -> Optional[dict]:
        """Tek bir chunk'ı getirir."""
        return self._col.find_one({"chunk_id": chunk_id}, {"_id": 0})

    def full_text_search(self, query: str, limit: int = 5, filter_langs: list[str] | None = None) -> list[dict]:
        """
        MongoDB full-text search (keyword fallback).
        Qdrant vektör aramasına alternatif/tamamlayıcı.
        """
        mongo_query: dict = {"$text": {"$search": query}}
        if filter_langs:
            mongo_query["lang"] = {"$in": filter_langs}
        results = list(
            self._col.find(
                mongo_query,
                {"score": {"$meta": "textScore"}, "_id": 0},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )
        return results

    def get_table_chunks_by_doc(self, doc_name: str) -> list[dict]:
        """Bir dokümandaki tüm tablo chunk'larını döndürür."""
        return list(self._col.find(
            {"doc_name": doc_name, "chunk_type": "table"},
            {"_id": 0},
        ))

    def get_chunks_by_page(self, doc_name: str, page: int) -> list[dict]:
        """Bir sayfadaki tüm chunk'ları döndürür."""
        return list(self._col.find(
            {"doc_name": doc_name, "page_range.0": {"$lte": page}, "page_range.1": {"$gte": page}},
            {"_id": 0},
        ))

    def get_chunk_ids_by_doc(self, doc_name: str) -> list[str]:
        """
        Belirli bir dokümana ait tüm chunk_id'leri döndürür.
        Qdrant'tan silme öncesi çağrılır.
        """
        docs = self._col.find({"doc_name": doc_name}, {"chunk_id": 1, "_id": 0})
        return [d["chunk_id"] for d in docs]

    def delete_by_doc(self, doc_name: str) -> int:
        """Bir dokümana ait tüm chunk'ları siler."""
        result = self._col.delete_many({"doc_name": doc_name})
        logger.info(f"MongoDB: '{doc_name}' için {result.deleted_count} chunk silindi")
        return result.deleted_count

    def list_docs(self) -> list[dict]:
        """
        Indexlenmiş doküman listesini özetler.
        Her doküman için: doc_name, chunk_count, page_range.
        """
        pipeline = [
            {"$group": {
                "_id": "$doc_name",
                "chunk_count": {"$sum": 1},
                "min_page": {"$min": {"$arrayElemAt": ["$page_range", 0]}},
                "max_page": {"$max": {"$arrayElemAt": ["$page_range", 1]}},
            }},
            {"$sort": {"_id": 1}},
        ]
        return [
            {
                "doc_name": r["_id"],
                "chunk_count": r["chunk_count"],
                "page_range": [r["min_page"], r["max_page"]],
            }
            for r in self._col.aggregate(pipeline)
        ]

    def count(self) -> int:
        return self._col.count_documents({})

    def drop(self) -> None:
        """Collection'ı siler (sıfırdan indexleme için)."""
        self._col.drop()
        logger.warning("MongoDB collection silindi")

    def close(self) -> None:
        self._client.close()
