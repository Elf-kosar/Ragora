"""Inspect MongoDB and Qdrant state for the RAG database."""
import json
import sys
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient
from database.mongodb_manager import MongoDBManager
from config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_IMAGE_COLLECTION,
    SPOT_MONGODB_URI,
    SPOT_MONGODB_DB_NAME,
    SPOT_MONGODB_COLLECTION,
    SPOT_QDRANT_COLLECTION,
)
from pymongo import MongoClient


def qdrant_collection_info(client, name):
    try:
        info = client.get_collection(name)
    except Exception as exc:
        return {
            "name": name,
            "exists": False,
            "error": str(exc),
        }
    vectors = info.config.params.vectors
    return {
        "name": name,
        "exists": True,
        "points": int(info.points_count or 0),
        "vector_size": getattr(vectors, "size", None),
        "distance": str(getattr(vectors, "distance", "")),
    }


def main():
    mongo = MongoDBManager()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("=== MongoDB ===")
    total_docs = mongo.collection.count_documents({})
    print(f"docs: {total_docs}")
    print(f"docs_with_images: {mongo.collection.count_documents({'image_count': {'$gt': 0}})}")

    mongo_categories = Counter(
        doc.get("category", "unknown")
        for doc in mongo.collection.find({}, {"category": 1})
    )
    for category, count in sorted(mongo_categories.items()):
        print(f"  {category}: {count}")

    print("\n=== Qdrant ===")
    for collection in [QDRANT_COLLECTION, QDRANT_IMAGE_COLLECTION]:
        print(json.dumps(qdrant_collection_info(qdrant, collection), ensure_ascii=False))

    print("\n=== Spot Team Project ===")
    spot_client = MongoClient(SPOT_MONGODB_URI)
    spot_collection = spot_client[SPOT_MONGODB_DB_NAME][SPOT_MONGODB_COLLECTION]
    print(f"mongo: {SPOT_MONGODB_DB_NAME}.{SPOT_MONGODB_COLLECTION}")
    print(f"chunks: {spot_collection.count_documents({})}")
    sample_spot = spot_collection.find_one({}, {"_id": 0, "embedding": 0})
    if sample_spot:
        print(json.dumps({
            "doc_name": sample_spot.get("doc_name"),
            "page_range": sample_spot.get("page_range"),
            "breadcrumb_str": sample_spot.get("breadcrumb_str"),
            "raw_text_preview": (sample_spot.get("raw_text") or sample_spot.get("text") or "")[:300],
        }, ensure_ascii=False, indent=2))
    print(json.dumps(qdrant_collection_info(qdrant, SPOT_QDRANT_COLLECTION), ensure_ascii=False))
    spot_client.close()

    print("\n=== Sample Mongo Chunk ===")
    sample_doc = mongo.collection.find_one({"image_count": {"$gt": 0}})
    if sample_doc:
        sample_doc["_id"] = str(sample_doc["_id"])
        print(json.dumps({
            "_id": sample_doc.get("_id"),
            "category": sample_doc.get("category"),
            "source": sample_doc.get("source"),
            "image_count": sample_doc.get("image_count"),
            "image_hashes": sample_doc.get("image_hashes", [])[:5],
            "content_preview": sample_doc.get("content", "")[:500],
        }, ensure_ascii=False, indent=2))

    print("\n=== Sample Image Payload ===")
    points, _ = qdrant.scroll(
        collection_name=QDRANT_IMAGE_COLLECTION,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if points:
        print(json.dumps(points[0].payload, ensure_ascii=False, indent=2)[:3000])

    mongo.close()


if __name__ == "__main__":
    main()
