"""
Doküman Registry — MongoDB'de hangi PDF'lerin indexlendiğini tutar.

Collection: doc_registry
------------------------
{
  "doc_id":      str,          ← dosya hash'i (SHA256 ilk 16 byte)
  "doc_name":    str,          ← "spot-user-manual-en" (uzantısız)
  "filename":    str,          ← "spot-user-manual-en.pdf"
  "file_path":   str,          ← indexleme anındaki tam yol
  "file_hash":   str,          ← SHA256 — değişiklik tespiti için
  "file_size":   int,          ← byte
  "page_count":  int,
  "chunk_count": int,
  "status":      str,          ← "indexed" | "failed" | "stale"
  "indexed_at":  datetime,
  "tags":        [str],        ← isteğe bağlı etiketler (örn. ["spot", "v5.1"])
  "meta":        dict,         ← serbest alan (yazar, versiyon vb.)
}

Bu kayıt sayesinde:
- Aynı PDF iki kez indexlenmez (hash kontrolü)
- Değişen PDF'ler tespit edilir (stale)
- Doküman bazlı arama filtrelenebilir
- Tüm indexlenmiş dokümanlar listelenebilir
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection


class DocRegistry:
    """
    PDF dokümanlarının indexleme durumunu yönetir.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "spot_rag",
    ):
        self._client = MongoClient(uri)
        self._col: Collection = self._client[db_name]["doc_registry"]
        self._ensure_indexes()
        logger.debug("DocRegistry hazır")

    def _ensure_indexes(self) -> None:
        self._col.create_index("doc_id", unique=True, background=True)
        self._col.create_index("doc_name", background=True)
        self._col.create_index("status", background=True)
        self._col.create_index("tags", background=True)

    # ─── Hash ────────────────────────────────────────────────────────

    @staticmethod
    def compute_hash(pdf_path: Path) -> str:
        """PDF içeriğinin SHA256 hash'ini hesaplar."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def make_doc_id(file_hash: str) -> str:
        """Hash'in ilk 16 karakteri = doc_id."""
        return file_hash[:16]

    @staticmethod
    def make_doc_name(pdf_path: Path) -> str:
        """'spot-user-manual-en.pdf' → 'spot-user-manual-en'"""
        return pdf_path.stem

    # ─── Kontrol ─────────────────────────────────────────────────────

    def check(self, pdf_path: Path) -> dict:
        """
        Bir PDF'in durumunu döndürür.

        Returns
        -------
        dict with keys:
          status : "new" | "indexed" | "stale" | "failed"
          record : mevcut kayıt (yoksa None)
          file_hash : hesaplanan hash

        "stale" → PDF dosyası değişmiş ama eski hash ile kayıt var
        """
        file_hash = self.compute_hash(pdf_path)
        doc_id = self.make_doc_id(file_hash)
        doc_name = self.make_doc_name(pdf_path)

        # Aynı hash → zaten indexlenmiş
        exact = self._col.find_one({"doc_id": doc_id}, {"_id": 0})
        if exact and exact["status"] == "indexed":
            return {"status": "indexed", "record": exact, "file_hash": file_hash}

        # Aynı isimde ama farklı hash → stale
        stale = self._col.find_one(
            {"doc_name": doc_name, "doc_id": {"$ne": doc_id}}, {"_id": 0}
        )
        if stale:
            return {"status": "stale", "record": stale, "file_hash": file_hash}

        # Hiç kayıt yok
        return {"status": "new", "record": None, "file_hash": file_hash}

    # ─── Yazma ───────────────────────────────────────────────────────

    def register(
        self,
        pdf_path: Path,
        file_hash: str,
        page_count: int,
        chunk_count: int,
        tags: list[str] | None = None,
        meta: dict | None = None,
    ) -> str:
        """
        Başarıyla indexlenen PDF'i kaydeder.

        Returns
        -------
        str
            doc_id
        """
        doc_id = self.make_doc_id(file_hash)
        doc_name = self.make_doc_name(pdf_path)

        record = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "filename": pdf_path.name,
            "file_path": str(pdf_path.resolve()),
            "file_hash": file_hash,
            "file_size": pdf_path.stat().st_size,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "status": "indexed",
            "indexed_at": datetime.now(tz=timezone.utc),
            "tags": tags or [],
            "meta": meta or {},
        }

        self._col.update_one(
            {"doc_id": doc_id},
            {"$set": record},
            upsert=True,
        )
        logger.info(f"Registry: '{doc_name}' kaydedildi (doc_id={doc_id})")
        return doc_id

    def mark_failed(self, pdf_path: Path, file_hash: str, error: str) -> None:
        """Başarısız indexlemeyi kaydeder."""
        doc_id = self.make_doc_id(file_hash)
        self._col.update_one(
            {"doc_id": doc_id},
            {"$set": {
                "doc_id": doc_id,
                "doc_name": self.make_doc_name(pdf_path),
                "filename": pdf_path.name,
                "file_hash": file_hash,
                "status": "failed",
                "error": error,
                "indexed_at": datetime.now(tz=timezone.utc),
            }},
            upsert=True,
        )

    def mark_stale(self, doc_name: str) -> None:
        """Aynı isimdeki eski kaydı stale olarak işaretler."""
        self._col.update_many(
            {"doc_name": doc_name, "status": "indexed"},
            {"$set": {"status": "stale"}},
        )

    def delete(self, doc_name: str) -> int:
        """Registry'den siler (chunk'ları silmez)."""
        result = self._col.delete_many({"doc_name": doc_name})
        return result.deleted_count

    # ─── Okuma ───────────────────────────────────────────────────────

    def list_all(self, status: str | None = None) -> list[dict]:
        """Tüm kayıtları döndürür."""
        query = {}
        if status:
            query["status"] = status
        return list(self._col.find(query, {"_id": 0}).sort("indexed_at", -1))

    def get(self, doc_name: str) -> Optional[dict]:
        """doc_name ile kayıt getirir."""
        return self._col.find_one({"doc_name": doc_name, "status": "indexed"}, {"_id": 0})

    def get_all_doc_names(self) -> list[str]:
        """Indexlenmiş tüm doküman adlarını döndürür."""
        return [
            r["doc_name"]
            for r in self._col.find({"status": "indexed"}, {"doc_name": 1, "_id": 0})
        ]

    def close(self) -> None:
        self._client.close()
