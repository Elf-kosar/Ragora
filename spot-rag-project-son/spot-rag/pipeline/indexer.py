"""
Indexer Pipeline

Parse → Chunk → Embed → Store

Kullanım
--------
    python -m pipeline.indexer --pdf data/spot-user-manual-en.pdf

veya Python'dan:
    from pipeline.indexer import Indexer
    indexer = Indexer()
    indexer.run("data/spot-user-manual-en.pdf")
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

from chunkers.semantic_chunker import SemanticChunker
from config.settings import settings
from embedders.local_embedder import LocalEmbedder
from parsers.docling_parser import DoclingPDFParser
from stores.mongo_store import MongoChunkStore
from stores.qdrant_store import QdrantVectorStore


class Indexer:
    """
    PDF'den Qdrant+MongoDB'ye tam pipeline.

    Parameters
    ----------
    reset : bool
        True ise mevcut veriler silinir, sıfırdan indexlenir.
    """

    def __init__(self, reset: bool = False):
        logger.info("Indexer başlatılıyor...")

        self.parser = DoclingPDFParser(
            enable_ocr=False,
            table_mode="accurate",
        )
        self.chunker = SemanticChunker(
            max_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
            min_tokens=settings.min_chunk_size,
        )
        self.embedder = LocalEmbedder(
            model_name=settings.embed_model,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
        self.mongo = MongoChunkStore(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
            collection_name=settings.mongo_collection,
        )
        self.qdrant = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection,
            vector_dim=self.embedder.dimension,
        )

        if reset:
            logger.warning("RESET: Mevcut veriler siliniyor...")
            self.mongo.drop()
            self.qdrant.delete_collection()

    def run(self, pdf_path: str | Path) -> dict:
        """
        Tam indexleme pipeline'ını çalıştırır.

        Returns
        -------
        dict
            İstatistikler: element_count, chunk_count, duration_seconds
        """
        pdf_path = Path(pdf_path)
        start = time.time()

        # ── 1. Parse ──────────────────────────────────────────────────
        logger.info("─" * 50)
        logger.info("ADIM 1/4: PDF Parse")
        elements = list(self.parser.parse(pdf_path))
        logger.success(f"  {len(elements)} yapısal eleman çıkarıldı")

        if not elements:
            logger.error("PDF'den hiç eleman çıkarılamadı!")
            return {}

        # ── 2. Chunk ──────────────────────────────────────────────────
        logger.info("─" * 50)
        logger.info("ADIM 2/4: Semantic Chunking")
        self.chunker.doc_name = pdf_path.stem
        chunks = self.chunker.chunk(elements)
        logger.success(f"  {len(chunks)} chunk oluşturuldu")

        # Chunk istatistikleri
        type_counts = {}
        for c in chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
        for t, cnt in type_counts.items():
            logger.info(f"    {t}: {cnt} chunk")

        # ── 3. Embed ──────────────────────────────────────────────────
        logger.info("─" * 50)
        logger.info("ADIM 3/4: Embedding")
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_chunks(texts)
        logger.success(f"  {len(embeddings)} embedding üretildi (dim={embeddings.shape[1]})")

        # ── 4. Store ──────────────────────────────────────────────────
        logger.info("─" * 50)
        logger.info("ADIM 4/4: Store (MongoDB + Qdrant)")

        self.mongo.upsert_chunks(chunks)
        self.qdrant.upsert_embeddings(chunks, embeddings)

        duration = time.time() - start
        stats = {
            "element_count": len(elements),
            "chunk_count": len(chunks),
            "chunk_types": type_counts,
            "embedding_dim": embeddings.shape[1],
            "duration_seconds": round(duration, 2),
            "mongo_total": self.mongo.count(),
            "qdrant_total": self.qdrant.count(),
        }

        logger.info("─" * 50)
        logger.success(f"İNDEKSLEME TAMAMLANDI — {duration:.1f}s")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        return stats


def main():
    parser = argparse.ArgumentParser(description="PDF → Qdrant+MongoDB indexer")
    parser.add_argument("--pdf", default=str(settings.pdf_path), help="PDF dosyası")
    parser.add_argument("--reset", action="store_true", help="Mevcut veriyi sil")
    args = parser.parse_args()

    indexer = Indexer(reset=args.reset)
    indexer.run(args.pdf)


if __name__ == "__main__":
    main()
