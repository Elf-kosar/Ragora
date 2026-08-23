"""
Kötü VLM açıklamalarını Ollama llava:34b ile yeniden üretir.

Kullanım:
    python redescribe_visuals.py              # sadece kötü açıklamaları güncelle
    python redescribe_visuals.py --all        # tümünü yeniden işle
    python redescribe_visuals.py --doc spot-dock-user-manual-en  # tek doküman
"""
from __future__ import annotations

import argparse
import sys
import time

import requests
from loguru import logger
from pymongo import MongoClient
from tqdm import tqdm

# ─── Ayarlar ──────────────────────────────────────────────────────────────────

MONGO_URI   = "mongodb://localhost:27017"
DB_NAME     = "spot_rag"
COLLECTION  = "visual_chunks"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llava:34b"

BAD_KEYWORDS = ["not clear", "too small", "blurry", "cannot", "unable",
                "not enough", "insufficient", "low resolution", "poor quality"]

DETAILED_PROMPT = (
    "You are analyzing a page from a Boston Dynamics Spot robot technical manual. "
    "RULES: (1) NEVER mention image quality, resolution, or clarity. "
    "(2) NEVER say you cannot read something. "
    "(3) ALWAYS describe the visible content regardless of difficulty.\n\n"
    "Describe exactly what you see:\n"
    "- Labeled diagrams: name every labeled part and its location on the page\n"
    "- Robot photos: describe the pose, visible components, any arrows or highlights\n"
    "- WARNING/CAUTION/DANGER/NOTE boxes: copy the exact text\n"
    "- Tables: list every row and column value\n"
    "- Flowcharts: describe each box, diamond, arrow, and outcome\n"
    "- Step-by-step illustrations: describe what each step shows\n\n"
    "Begin with the content type (diagram / photo / warning / table / flowchart). "
    "Then describe every visible element in detail."
)


# ─── Ollama çağrısı ───────────────────────────────────────────────────────────

def describe_image(image_b64: str, retries: int = 2) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": DETAILED_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"num_predict": 1024, "num_ctx": 4096},
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
            desc = r.json().get("response", "").strip()
            if desc:
                return desc
        except Exception as e:
            if attempt < retries:
                logger.warning(f"Deneme {attempt+1} başarısız: {e} — yeniden deniyor...")
                time.sleep(3)
            else:
                logger.error(f"Tüm denemeler başarısız: {e}")
    return ""


# ─── Filtreleme ───────────────────────────────────────────────────────────────

def is_bad(description: str) -> bool:
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in BAD_KEYWORDS)


# ─── Ana işlem ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Tüm görselleri yeniden işle")
    parser.add_argument("--doc", type=str, default=None,
                        help="Belirli bir dokümanı işle")
    args = parser.parse_args()

    col = MongoClient(MONGO_URI)[DB_NAME][COLLECTION]

    # Hangi chunk'ları işleyeceğiz?
    query: dict = {}
    if args.doc:
        query["doc_name"] = args.doc
        logger.info(f"Doküman filtresi: {args.doc}")
    elif not args.all:
        query["$or"] = [
            {"vlm_description": ""},
            {"vlm_description": {"$regex": "|".join(BAD_KEYWORDS), "$options": "i"}},
            {"vlm_description": {"$regex": "^Visual content from page", "$options": "i"}},
        ]
        logger.info("Sadece kötü/metadata açıklamalar işlenecek")
    else:
        logger.info("Tüm görsel chunk'lar yeniden işlenecek")

    targets = list(col.find(query, {"element_id": 1, "image_b64": 1,
                                     "doc_name": 1, "page": 1,
                                     "vlm_description": 1, "_id": 0}))
    logger.info(f"İşlenecek chunk sayısı: {len(targets)}")

    if not targets:
        logger.success("İşlenecek chunk yok.")
        return

    updated = 0
    failed  = 0

    for item in tqdm(targets, desc="VLM yeniden işleme"):
        image_b64 = item.get("image_b64", "")
        if not image_b64:
            logger.warning(f"image_b64 yok: {item['element_id']}")
            failed += 1
            continue

        new_desc = describe_image(image_b64)

        if not new_desc:
            logger.warning(f"Açıklama üretilemedi: doc={item['doc_name']} s.{item['page']}")
            failed += 1
            continue

        col.update_one(
            {"element_id": item["element_id"]},
            {"$set": {"vlm_description": new_desc}},
        )
        updated += 1
        logger.debug(f"  ✓ {item['doc_name']} s.{item['page']}: {new_desc[:80]}...")

    logger.success(f"Tamamlandı: {updated} güncellendi, {failed} başarısız")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
