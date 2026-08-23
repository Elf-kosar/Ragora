"""
Visual RAG Chatbot — görsel + metin sorgu destekli CLI demo.

Kullanım
--------
    python chatbot_visual.py

Sorgu modları:
    Soru: E-Stop nasıl kullanılır?          → normal metin sorgu
    Soru: @image /path/to/photo.jpg          → sadece görsel sorgu
    Soru: @image /path/to/photo.jpg Buradaki parçanın adı ne?  → görsel + metin
    Soru: @image base64string...             → base64 görsel sorgu

Çıktı:
    - İlgili metin chunk'lar (breadcrumb + sayfa)
    - İlgili görsel chunk'lar (figure caption + VLM açıklaması)
    - LLM cevabı (multimodal: hem metin hem görsel bağlamla)
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings
from pipeline.visual_retriever import HybridResult, VisualRetriever


# ─── LLM Entegrasyonu ────────────────────────────────────────────────────────
# Multimodal LLM için (görsel + metin context):
#
# Ollama (LLaVA):
#   import requests, base64
#   def call_llm(text_context, images, query):
#       content = [{"type": "text", "text": f"Context:\n{text_context}\n\nQ: {query}"}]
#       for img in images:
#           content.append({"type": "image_url",
#               "image_url": {"url": f"data:image/jpeg;base64,{img['b64']}"}})
#       r = requests.post("http://localhost:11434/api/chat",
#           json={"model": "llava", "messages": [{"role": "user", "content": content}], "stream": False})
#       return r.json()["message"]["content"]
#
# OpenAI-compatible multimodal (vLLM + LLaVA / InternVL):
#   from openai import OpenAI
#   client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
#   def call_llm(text_context, images, query):
#       content = [{"type": "text", "text": f"Context:\n{text_context}\n\nQ: {query}"}]
#       for img in images[:3]:  # max 3 görsel
#           content.append({"type": "image_url",
#               "image_url": {"url": f"data:image/jpeg;base64,{img['b64']}"}})
#       r = client.chat.completions.create(
#           model="your-model",
#           messages=[{"role": "user", "content": content}])
#       return r.choices[0].message.content

def call_llm(text_context: str, images: list[dict], query: str) -> str:
    """PLACEHOLDER — kendi multimodal LLM'inle değiştir."""
    img_info = ""
    if images:
        img_info = f"\n\n[{len(images)} görsel bağlam mevcut]"
        for img in images:
            img_info += f"\n  - Sayfa {img['page']}: {img['caption'] or img['type']}"
    return (
        "[Multimodal LLM entegrasyonu bekleniyor]\n\n"
        f"Metin bağlam:\n{text_context[:500]}...{img_info}"
    )


SYSTEM_PROMPT = """You are a helpful assistant specialized in the Boston Dynamics Spot robot.
Answer questions based solely on the provided context from the Spot User Manual.
When images are provided, analyze them carefully and reference specific visual details.
Always cite the section and page number when answering.
Be precise and safety-conscious — this is a robotics safety manual."""


def build_multimodal_prompt(result: HybridResult) -> tuple[str, list[dict]]:
    """
    HybridResult'dan LLM prompt'u oluşturur.

    Returns
    -------
    (text_context, images)
    """
    payload = result.format_for_llm(include_images=True)
    return payload["text_context"], payload["images"]


# ─── Chatbot ─────────────────────────────────────────────────────────────────

def parse_image_query(raw_query: str) -> tuple[str | None, str]:
    """
    "@image /path/to/img.jpg Sorum nedir?" şeklindeki sorguları parse eder.

    Returns
    -------
    (image_path_or_b64, text_query)
    """
    if not raw_query.startswith("@image"):
        return None, raw_query

    parts = raw_query[len("@image"):].strip().split(" ", 1)
    image_ref = parts[0]
    text_query = parts[1] if len(parts) > 1 else ""
    return image_ref, text_query


def chat():
    logger.info("VisualRetriever yükleniyor...")
    retriever = VisualRetriever(top_k_text=5, top_k_visual=3)

    print("\n" + "═" * 65)
    print("  Spot Robot Visual RAG Chatbot")
    print("  Metin: E-Stop nerede?")
    print("  Görsel: @image /path/to/img.jpg")
    print("  Görsel+metin: @image /path/to/img.jpg Bu ne?")
    print("  Çıkış: exit")
    print("═" * 65 + "\n")

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

        # Sorguyu parse et
        image_ref, text_query = parse_image_query(raw)

        # Retrieval
        try:
            if image_ref:
                # Görsel + (opsiyonel) metin sorgu
                image_path = Path(image_ref)
                if image_path.exists():
                    result = retriever.retrieve_by_image(
                        image_input=image_path,
                        query_text=text_query,
                    )
                    mode = "görsel"
                else:
                    # Base64 string olabilir
                    result = retriever.retrieve_by_image(
                        image_input=image_ref,
                        query_text=text_query,
                    )
                    mode = "görsel (b64)"
            else:
                # Sadece metin sorgu
                result = retriever.retrieve_by_text(query=text_query or raw)
                mode = "metin"
        except Exception as e:
            print(f"\n[Hata: {e}]\n")
            continue

        # Sonuçları göster
        print(f"\n📋 Mod: {mode}")
        if result.text_chunks:
            print(f"📚 {len(result.text_chunks)} metin chunk:")
            for c in result.text_chunks:
                section = " > ".join(c.breadcrumb[-2:]) if c.breadcrumb else "—"
                print(
                    f"  • [{c.chunk_type}] {section} | "
                    f"s.{c.page_range[0]} | skor={c.score:.3f}"
                )

        if result.visual_chunks:
            print(f"🖼  {len(result.visual_chunks)} görsel chunk:")
            for v in result.visual_chunks:
                section = " > ".join(v.breadcrumb[-2:]) if v.breadcrumb else "—"
                caption = v.figure_caption[:60] + "..." if len(v.figure_caption) > 60 else v.figure_caption
                print(
                    f"  • [{v.visual_type}] {section} | "
                    f"s.{v.page} | skor={v.score:.3f}"
                )
                if caption:
                    print(f"    Caption: {caption}")
                if v.vlm_description:
                    desc_preview = v.vlm_description[:100].replace("\n", " ")
                    print(f"    VLM: {desc_preview}...")

        if not result.text_chunks and not result.visual_chunks:
            print("\n[Sonuç bulunamadı]\n")
            continue

        # LLM çağrısı
        text_context, images = build_multimodal_prompt(result)
        answer = call_llm(text_context, images, text_query or raw)

        print(f"\n🤖 Yanıt:\n{answer}\n")
        print("─" * 65)


if __name__ == "__main__":
    chat()
