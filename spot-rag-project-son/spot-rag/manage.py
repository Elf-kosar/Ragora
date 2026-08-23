"""
Doküman Yönetim CLI

Indexlenmiş PDF'leri listele, sil, güncelle.

Kullanım
--------
    python manage.py list                          # Tüm dokümanları listele
    python manage.py list --status stale           # Sadece stale olanlar
    python manage.py show spot-user-manual-en      # Doküman detayı
    python manage.py delete spot-user-manual-en    # Dokümanı sil
    python manage.py stats                         # Genel istatistikler
"""
from __future__ import annotations

import argparse
import sys
from datetime import timezone

from loguru import logger

from config.settings import settings
from stores.doc_registry import DocRegistry
from stores.mongo_store import MongoChunkStore
from stores.qdrant_store import QdrantVectorStore


def get_stores():
    registry = DocRegistry(uri=settings.mongo_uri, db_name=settings.mongo_db)
    mongo = MongoChunkStore(
        uri=settings.mongo_uri,
        db_name=settings.mongo_db,
        collection_name=settings.mongo_collection,
    )
    qdrant = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
        vector_dim=1024,  # boyut sadece bağlantı için önemli değil
    )
    return registry, mongo, qdrant


# ─────────────────────────────────────────────────────────────────────────────
# Komutlar
# ─────────────────────────────────────────────────────────────────────────────

def cmd_list(args):
    """Tüm indexlenmiş dokümanları listeler."""
    registry, mongo, _ = get_stores()
    docs = registry.list_all(status=args.status or None)

    if not docs:
        print("Henüz indexlenmiş doküman yok.")
        return

    print(f"\n{'DOC NAME':<40} {'STATUS':<10} {'CHUNKS':<8} {'PAGES':<8} {'INDEXED AT'}")
    print("─" * 90)

    for d in docs:
        indexed_at = d.get("indexed_at", "—")
        if hasattr(indexed_at, "strftime"):
            indexed_at = indexed_at.strftime("%Y-%m-%d %H:%M")
        tags = ", ".join(d.get("tags", [])) or "—"
        print(
            f"{d['doc_name']:<40} "
            f"{d['status']:<10} "
            f"{d.get('chunk_count', '?'):<8} "
            f"{d.get('page_count', '?'):<8} "
            f"{indexed_at}"
        )
        if d.get("tags"):
            print(f"  {'':40} tags: {tags}")

    print(f"\nToplam: {len(docs)} doküman")


def cmd_show(args):
    """Belirli bir dokümanın detaylarını gösterir."""
    registry, mongo, _ = get_stores()
    doc = registry.get(args.doc_name)

    if not doc:
        print(f"'{args.doc_name}' bulunamadı.")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"Doküman  : {doc['doc_name']}")
    print(f"Dosya    : {doc['filename']}")
    print(f"Durum    : {doc['status']}")
    print(f"Sayfalar : {doc.get('page_count', '?')}")
    print(f"Chunk    : {doc.get('chunk_count', '?')}")
    print(f"Boyut    : {doc.get('file_size', 0) / 1024 / 1024:.1f} MB")
    print(f"Hash     : {doc.get('file_hash', '?')[:16]}...")
    print(f"Etiketler: {', '.join(doc.get('tags', [])) or '—'}")
    indexed_at = doc.get("indexed_at", "—")
    if hasattr(indexed_at, "strftime"):
        indexed_at = indexed_at.strftime("%Y-%m-%d %H:%M UTC")
    print(f"İndexleme: {indexed_at}")

    if doc.get("meta"):
        print(f"Meta     : {doc['meta']}")
    print(f"{'─'*50}\n")


def cmd_delete(args):
    """Dokümanı tamamen siler (chunk'lar + registry)."""
    registry, mongo, qdrant = get_stores()

    doc = registry.get(args.doc_name)
    if not doc:
        print(f"'{args.doc_name}' bulunamadı.")
        sys.exit(1)

    if not args.yes:
        confirm = input(
            f"'{args.doc_name}' ({doc.get('chunk_count', '?')} chunk) "
            f"silinecek. Devam? [y/N] "
        ).strip().lower()
        if confirm not in {"y", "yes", "e", "evet"}:
            print("İptal.")
            return

    # Chunk'ları sil
    chunk_ids = mongo.get_chunk_ids_by_doc(args.doc_name)
    mongo.delete_by_doc(args.doc_name)
    qdrant.delete_by_ids(chunk_ids)
    registry.delete(args.doc_name)

    print(f"✓ '{args.doc_name}' silindi ({len(chunk_ids)} chunk).")


def cmd_stats(args):
    """Genel istatistikleri gösterir."""
    registry, mongo, qdrant = get_stores()

    all_docs = registry.list_all()
    indexed = [d for d in all_docs if d["status"] == "indexed"]
    stale = [d for d in all_docs if d["status"] == "stale"]
    failed = [d for d in all_docs if d["status"] == "failed"]

    mongo_count = mongo.count()
    qdrant_count = qdrant.count()

    print(f"\n{'─'*40}")
    print("  RAG Sistem İstatistikleri")
    print(f"{'─'*40}")
    print(f"  Dokümanlar    : {len(indexed)} indexlenmiş")
    if stale:
        print(f"  Stale         : {len(stale)} güncelleme bekliyor")
    if failed:
        print(f"  Hatalı        : {len(failed)}")
    print(f"  MongoDB chunk : {mongo_count:,}")
    print(f"  Qdrant vektör : {qdrant_count:,}")

    if indexed:
        total_chunks = sum(d.get("chunk_count", 0) for d in indexed)
        total_pages = sum(d.get("page_count", 0) for d in indexed)
        print(f"  Toplam sayfa  : {total_pages:,}")
        print(f"  Toplam chunk  : {total_chunks:,}")

    print(f"{'─'*40}\n")

    # Doc bazlı özet
    if indexed:
        print(f"{'DOC NAME':<40} {'CHUNKS':>7} {'PAGES':>6}")
        print("─" * 56)
        for d in indexed:
            print(
                f"{d['doc_name']:<40} "
                f"{d.get('chunk_count', '?'):>7} "
                f"{d.get('page_count', '?'):>6}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Doküman Yönetimi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="Tüm dokümanları listele")
    p_list.add_argument("--status", choices=["indexed", "stale", "failed"], help="Durum filtresi")

    # show
    p_show = sub.add_parser("show", help="Doküman detayı")
    p_show.add_argument("doc_name", help="Doküman adı (örn: spot-user-manual-en)")

    # delete
    p_del = sub.add_parser("delete", help="Dokümanı sil")
    p_del.add_argument("doc_name", help="Doküman adı")
    p_del.add_argument("-y", "--yes", action="store_true", help="Onay sormadan sil")

    # stats
    sub.add_parser("stats", help="Genel istatistikler")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "delete": cmd_delete,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
