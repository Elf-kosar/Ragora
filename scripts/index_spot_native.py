"""Run the native Spot project indexer against the shared services."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SPOT_ROOT = ROOT_DIR / "spot-rag-project-son" / "spot-rag"
DEFAULT_PDF_DIR = SPOT_ROOT / "data" / "pdfs"


def configure_environment() -> None:
    os.environ.setdefault("MONGO_URI", os.getenv("SPOT_MONGODB_URI", "mongodb://localhost:27017/"))
    os.environ.setdefault("MONGO_DB", os.getenv("SPOT_MONGODB_DB_NAME", "Ragora"))
    os.environ.setdefault("MONGO_COLLECTION", os.getenv("SPOT_MONGODB_COLLECTION", "spot_collection"))
    os.environ.setdefault("QDRANT_HOST", os.getenv("QDRANT_HOST", "localhost"))
    os.environ.setdefault("QDRANT_PORT", os.getenv("QDRANT_PORT", "6333"))
    os.environ.setdefault("QDRANT_COLLECTION", os.getenv("SPOT_QDRANT_COLLECTION", "spot_chunks"))
    os.environ.setdefault("VISUAL_MONGO_COLLECTION", os.getenv("SPOT_VISUAL_MONGODB_COLLECTION", "spot_visual_collection"))
    os.environ.setdefault("VISUAL_QDRANT_COLLECTION", os.getenv("SPOT_VISUAL_QDRANT_COLLECTION", "spot_visual_chunks"))
    os.environ.setdefault("EMBED_MODEL", os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large"))
    os.environ.setdefault("EMBED_DEVICE", os.getenv("EMBED_DEVICE", "cpu"))


def drop_legacy_conflicting_indexes() -> None:
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGO_URI"])
    db = client[os.environ["MONGO_DB"]]
    for collection_name, index_name in (
        (os.environ["MONGO_COLLECTION"], "full_text_search"),
        (os.environ["VISUAL_MONGO_COLLECTION"], "visual_text_search"),
    ):
        try:
            db[collection_name].drop_index(index_name)
            print(f"Legacy Mongo index dropped: {collection_name}.{index_name}")
        except Exception:
            pass
    client.close()


def import_spot_modules() -> None:
    if not (SPOT_ROOT / "pipeline").exists():
        raise SystemExit(f"Spot project not found: {SPOT_ROOT}")
    sys.path.insert(0, str(SPOT_ROOT))


def list_pdfs(pdf_dir: Path, limit: int = 0) -> list[Path]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit > 0:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise SystemExit(f"PDF bulunamadi: {pdf_dir}")
    return pdfs


def run_text_indexer(pdfs: list[Path], reset: bool) -> None:
    from pipeline.indexer import Indexer

    for index, pdf in enumerate(pdfs):
        indexer = Indexer(reset=reset and index == 0)
        indexer.run(pdf)


def run_hybrid_indexer(pdfs: list[Path], use_vlm: bool, force: bool) -> None:
    from pipeline.hybrid_indexer import HybridIndexer

    indexer = HybridIndexer(use_vlm=use_vlm, force=force)
    for pdf in pdfs:
        indexer.run(pdf, text_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Spot indexer for shared Ragora services.")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Directory containing Spot PDF manuals.")
    parser.add_argument("--mode", choices=["text", "hybrid"], default="hybrid", help="Native Spot indexer mode.")
    parser.add_argument("--reset", action="store_true", help="Reset text collections before text indexing.")
    parser.add_argument("--force", action="store_true", help="Force re-indexing in hybrid mode.")
    parser.add_argument("--with-vlm", action="store_true", help="Enable Spot VLM visual descriptions in hybrid mode.")
    parser.add_argument("--limit", type=int, default=0, help="Index only first N PDFs.")
    args = parser.parse_args()

    configure_environment()
    drop_legacy_conflicting_indexes()
    import_spot_modules()

    pdfs = list_pdfs(Path(args.pdf_dir), limit=args.limit)
    if args.mode == "text":
        run_text_indexer(pdfs, reset=args.reset)
    else:
        run_hybrid_indexer(pdfs, use_vlm=args.with_vlm, force=args.force)

    print(f"Tamamlandi. mode={args.mode}, pdf={len(pdfs)}")


if __name__ == "__main__":
    main()
