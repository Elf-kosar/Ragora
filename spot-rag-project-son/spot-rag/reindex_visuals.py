"""
Görsel chunk'ları tekil figür bazında yeniden indexler.

Eski: 165 tam sayfa görüntüsü
Yeni: ~206 tekil figür crop'u (daha hassas CLIP eşleşmesi)

Kullanım:
    python reindex_visuals.py              # tüm PDF'leri işle
    python reindex_visuals.py --no-vlm     # VLM olmadan (hızlı test)
    python reindex_visuals.py --pdf spot-user-manual-en.pdf  # tek PDF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "spot_rag"
VISUAL_MONGO_COL = "visual_chunks"
VISUAL_QDRANT_COL = "spot_visual_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
PDF_DIR = Path("data/pdfs")


def clear_visual_stores(doc_name: str | None = None) -> None:
    """Görsel chunk'ları siler. doc_name verilirse sadece o dokümanı siler."""
    mongo_col = MongoClient(MONGO_URI)[DB_NAME][VISUAL_MONGO_COL]
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if doc_name:
        # Tek doküman
        element_ids = [
            d["element_id"]
            for d in mongo_col.find({"doc_name": doc_name}, {"element_id": 1, "_id": 0})
        ]
        deleted_mongo = mongo_col.delete_many({"doc_name": doc_name}).deleted_count
        if element_ids:
            qdrant.delete(
                collection_name=VISUAL_QDRANT_COL,
                points_selector=PointIdsList(points=element_ids),
            )
        logger.info(f"'{doc_name}': {deleted_mongo} MongoDB + {len(element_ids)} Qdrant görsel silindi")
    else:
        # Tümünü sil
        deleted_mongo = mongo_col.delete_many({}).deleted_count
        try:
            qdrant.delete_collection(VISUAL_QDRANT_COL)
            logger.info("Qdrant spot_visual_chunks collection silindi ve yeniden oluşturulacak")
        except Exception:
            pass
        logger.info(f"Tüm görseller silindi: {deleted_mongo} MongoDB chunk")


def main():
    parser = argparse.ArgumentParser(description="Görsel chunk'ları tekil figür bazında yeniden indexle")
    parser.add_argument("--no-vlm", action="store_true", help="VLM açıklama üretme (sadece CLIP)")
    parser.add_argument("--pdf", type=str, default=None, help="Sadece bu PDF'i işle (dosya adı)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Hangi PDF'ler işlenecek?
    if args.pdf:
        pdf_name = args.pdf if args.pdf.endswith(".pdf") else args.pdf + ".pdf"
        pdfs = [PDF_DIR / pdf_name]
        if not pdfs[0].exists():
            logger.error(f"PDF bulunamadı: {pdfs[0]}")
            sys.exit(1)
    else:
        pdfs = sorted(PDF_DIR.glob("*.pdf"))

    logger.info(f"İşlenecek PDF sayısı: {len(pdfs)}")
    logger.info(f"VLM: {'KAPALI (sadece CLIP)' if args.no_vlm else 'AÇIK (Ollama)'}")
    logger.info("")

    # Import burada — model yüklemesi yavaş
    from pipeline.hybrid_indexer import HybridIndexer

    indexer = HybridIndexer(
        use_vlm=not args.no_vlm,
        use_ollama_vlm=not args.no_vlm,
        force=True,
    )

    total_old = sum(
        1 for _ in MongoClient(MONGO_URI)[DB_NAME][VISUAL_MONGO_COL].find({}, {"_id": 1})
    )
    logger.info(f"Mevcut görsel chunk sayısı: {total_old}")

    import time

    results = []
    for pdf_idx, pdf_path in enumerate(pdfs, 1):
        doc_name_stem = pdf_path.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"[{pdf_idx}/{len(pdfs)}] PDF: {pdf_path.name}")
        logger.info(f"{'='*60}")

        clear_visual_stores(doc_name=doc_name_stem)

        t0 = time.time()
        stats = indexer.run(pdf_path, text_only=False)
        elapsed = time.time() - t0

        vis_count = stats.get("visual_elements", 0)
        logger.info(f"  → {vis_count} figür indexlendi ({elapsed:.0f}s)")
        results.append(stats)

    # Özet
    logger.info(f"\n{'='*60}")
    logger.info("TÜM PDF'LER TAMAMLANDI")
    total_new = MongoClient(MONGO_URI)[DB_NAME][VISUAL_MONGO_COL].count_documents({})
    logger.info(f"Eski görsel chunk: {total_old}")
    logger.info(f"Yeni görsel chunk: {total_new}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
