"""
RAG Chatbot — Ollama + VisualRetriever

Kullanım
--------
    python chatbot.py

Sorgu formatları
----------------
    Metin sorgu   : How do I charge the battery?
    Doküman filtre: @spot-user-manual E-stop nerede?
    Görsel sorgu  : [/path/to/image.jpg] Bu parça ne?
    Görsel + filtre: @spot-user-manual [/path/to/img.jpg] Bu ne?

Ortam değişkenleri (.env)
-------------------------
    OLLAMA_HOST          http://localhost:11434
    OLLAMA_MODEL         llama3.2          (metin sorguları)
    OLLAMA_VISION_MODEL  llava             (görsel sorguları)
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import requests
from loguru import logger

from config.settings import settings
from pipeline.visual_retriever import HybridResult, VisualRetriever


# ─── Sistem prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant specialized in the Boston Dynamics Spot robot.
Answer questions based solely on the provided context from the Spot User Manual.
If the answer is not in the context, say so clearly.
Always cite the section and page number when answering.
Be precise and safety-conscious — this is a robotics safety manual."""


# ─── Ollama ──────────────────────────────────────────────────────────────────

def call_ollama(
    system: str,
    user: str,
    image_b64: str | None = None,
) -> str:
    """
    Ollama /api/chat çağrısı.

    image_b64 verilirse OLLAMA_VISION_MODEL kullanılır,
    aksi halde OLLAMA_MODEL kullanılır.
    """
    model = settings.ollama_vision_model if image_b64 else settings.ollama_model

    user_msg: dict = {"role": "user", "content": user}
    if image_b64:
        user_msg["images"] = [image_b64]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            user_msg,
        ],
        "stream": False,
        "options": {"num_ctx": 16384},
    }

    try:
        r = requests.post(
            f"{settings.ollama_host}/api/chat",
            json=payload,
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            f"[Bağlantı hatası] Ollama'ya ulaşılamıyor: {settings.ollama_host}\n"
            "Ollama çalışıyor mu? → ollama serve"
        )
    except requests.exceptions.HTTPError as e:
        return (
            f"[HTTP hatası] {e}\n"
            f"'{model}' modeli yüklü mü? → ollama pull {model}"
        )
    except Exception as e:
        return f"[Ollama hatası] {e}"


# ─── Prompt builder ──────────────────────────────────────────────────────────

def build_prompt(query: str, result: HybridResult) -> str:
    """
    Retrieval sonucundan LLM user mesajı oluşturur.

    Metin chunk'ları + görsel VLM açıklamaları tek metin bağlamına dönüştürülür.
    (Görsel görseller Ollama'ya ayrıca gönderilmez — VLM açıklamaları yeterli.)
    """
    parts: list[str] = ["Context from Spot User Manual:\n"]

    for i, chunk in enumerate(result.text_chunks, 1):
        section = " > ".join(chunk.breadcrumb) if chunk.breadcrumb else "General"
        pages = (
            f"p.{chunk.page_range[0]}-{chunk.page_range[1]}"
            if chunk.page_range[0] != chunk.page_range[1]
            else f"p.{chunk.page_range[0]}"
        )
        header = f"[{i}] {section} ({pages})"
        if chunk.chunk_type == "table":
            header += " [TABLE]"
        parts.append(f"{header}\n{chunk.raw_text}")

    if result.visual_chunks:
        parts.append("\nVisual Content from Manual:")
        for vis in result.visual_chunks:
            section = " > ".join(vis.breadcrumb) if vis.breadcrumb else "General"
            vis_header = f"[Visual, p.{vis.page}, {section}]"
            if vis.figure_caption:
                vis_header += f"\nCaption: {vis.figure_caption}"
            if vis.vlm_description:
                vis_header += f"\nDescription: {vis.vlm_description}"
            parts.append(vis_header)

    context = "\n\n---\n\n".join(parts)
    return f"{context}\n\n---\n\nQuestion: {query}\n\nAnswer based on the context above:"


# ─── Query parser ────────────────────────────────────────────────────────────

_IMAGE_RE = re.compile(r'\[([^\]]+\.(?:jpg|jpeg|png|bmp|webp))\]', re.IGNORECASE)


def parse_query(raw: str) -> tuple[str, str | None, list[str] | None]:
    """
    Ham sorgu satırını ayrıştırır.

    Returns
    -------
    (query_text, image_path_or_None, filter_docs_or_None)
    """
    text = raw.strip()

    # @docname filtresi
    filter_docs: list[str] | None = None
    if text.startswith("@"):
        parts = text.split(" ", 1)
        filter_docs = [parts[0][1:]]
        text = parts[1].strip() if len(parts) > 1 else ""

    # [/path/to/image.jpg] görseli
    image_path: str | None = None
    match = _IMAGE_RE.search(text)
    if match:
        candidate = match.group(1)
        if Path(candidate).exists():
            image_path = candidate
            text = _IMAGE_RE.sub("", text).strip()
        else:
            print(f"  ⚠  Görsel bulunamadı: {candidate}")

    return text, image_path, filter_docs


# ─── Chat loop ───────────────────────────────────────────────────────────────

def chat() -> None:
    logger.info("VisualRetriever yükleniyor...")
    retriever = VisualRetriever(top_k_text=5, top_k_visual=3)

    try:
        from stores.doc_registry import DocRegistry
        registry = DocRegistry(uri=settings.mongo_uri, db_name=settings.mongo_db)
        available_docs = registry.get_all_doc_names()
    except Exception:
        available_docs = []

    print("\n" + "═" * 62)
    print("  Spot Robot RAG Chatbot")
    print(f"  Metin modeli  : {settings.ollama_model}")
    print(f"  Vision modeli : {settings.ollama_vision_model}")
    if available_docs:
        print(f"  Dokümanlar    : {', '.join(available_docs)}")
    print()
    print("  Metin   : How do I charge the battery?")
    print("  Filtre  : @spot-user-manual E-stop nerede?")
    print("  Görsel  : [/path/to/image.jpg] Bu parça ne?")
    print("  Çıkış   : exit")
    print("═" * 62 + "\n")

    while True:
        try:
            raw = input("Soru: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGüle güle!")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "çıkış"}:
            print("Güle güle!")
            break

        query, image_path, filter_docs = parse_query(raw)

        if not query and not image_path:
            continue

        if filter_docs:
            print(f"  → '{filter_docs[0]}' içinde aranıyor...")

        # Retrieval
        if image_path:
            print(f"  → Görsel sorgu: {image_path}")
            result = retriever.retrieve_by_image(
                image_input=Path(image_path),
                query_text=query,
                filter_docs=filter_docs,
            )
        else:
            result = retriever.retrieve_by_text(
                query=query,
                filter_docs=filter_docs,
                auto_lang=True,
            )

        if not result.text_chunks and not result.visual_chunks:
            print("\n[Sonuç bulunamadı — sorguyu genişletmeyi dene]\n")
            continue

        # Sonuç özeti
        print(
            f"\n  {len(result.text_chunks)} metin chunk, "
            f"{len(result.visual_chunks)} görsel chunk bulundu:"
        )
        for c in result.text_chunks:
            section = " > ".join(c.breadcrumb[-2:]) if c.breadcrumb else "—"
            print(f"    [{c.chunk_type:5}] {section} | s.{c.page_range[0]} | {c.score:.3f}")
        for v in result.visual_chunks:
            section = v.breadcrumb[-1] if v.breadcrumb else "—"
            print(f"    [görsel] {section} | s.{v.page} | {v.score:.3f}")

        # Prompt
        effective_query = query or "Describe this image and find relevant information."
        user_message = build_prompt(effective_query, result)

        # Ollama çağrısı — görsel sorguysa kullanıcı görselini gönder
        query_image_b64: str | None = None
        if image_path:
            with open(image_path, "rb") as f:
                query_image_b64 = base64.b64encode(f.read()).decode()

        print("\n  Yanıt üretiliyor...\n")
        answer = call_ollama(SYSTEM_PROMPT, user_message, image_b64=query_image_b64)

        print(f"{answer}\n")
        print("─" * 62)


if __name__ == "__main__":
    chat()
