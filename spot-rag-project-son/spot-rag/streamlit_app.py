"""
Spot RAG — Streamlit Web Arayüzü

Başlatmak için:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import base64
import io
import sys
from collections import Counter as _Counter
from pathlib import Path

import requests
import streamlit as st
from PIL import Image
from loguru import logger

# Loglama sessiz
logger.remove()
logger.add(sys.stderr, level="WARNING")

from config.settings import settings
from pipeline.visual_retriever import HybridResult, VisualRetriever
from stores.doc_registry import DocRegistry

# ─── Sayfa ayarları ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAGORA",
    page_icon="🔴",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── RAGORA Global ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { min-width: 220px !important; transform: none !important; }

.main .block-container {
    max-width: 760px;
    padding: 0 1.5rem 7rem 1.5rem;
    margin: 0 auto;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #D42B2B22; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #D42B2B66; }

/* ── RAGORA Header ── */
.rg-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 18vh 0 4vh 0;
    margin-bottom: 0;
}
.rg-header-row {
    display: flex;
    align-items: center;
    gap: 14px;
}
.rg-logo {
    width: 42px; height: 42px;
    background: #D42B2B;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 19px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(212,43,43,0.25);
}
.rg-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 4px;
    color: #111;
}

/* ── User message ── */
.rg-user-row {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 6px 0;
}
.rg-user-bubble {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 18px 4px 18px 18px;
    padding: 10px 16px;
    max-width: 78%;
    font-size: 0.95rem;
    color: #1a1a1a;
    line-height: 1.6;
}

/* ── AI answer card ── */
[data-testid="stChatMessage"] {
    background: #fff !important;
    border: none !important;
    border-left: 3px solid #D42B2B !important;
    border-radius: 0 12px 12px 0 !important;
    padding: 2px 16px 12px 20px !important;
    margin: 6px 0 4px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

.rg-ai-header {
    color: #bbb;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 10px 0 10px 0;
    margin-bottom: 6px;
    border-bottom: 1px solid #f5f5f5;
}

/* ── Welcome / boş ekran ── */
.rg-welcome {
    text-align: center;
    padding: 0;
    color: #bbb;
    font-size: 0.9rem;
}

/* ── KAYNAKLAR card ── */
.rg-sources {
    background: #fff;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0 16px 0;
}

/* ── Image uploaded ── */
.rg-user-image {
    max-width: 200px;
    border-radius: 10px;
    margin-top: 6px;
}

/* ── Input bar ── */
[data-testid="stChatInput"] > div {
    border: 1.5px solid #e0e0e0 !important;
    border-radius: 50px !important;
    background: #fff !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #D42B2B !important;
    box-shadow: 0 2px 12px rgba(212,43,43,0.10) !important;
}

/* ── Temizle butonu ── */
.rg-clear-btn {
    text-align: right;
    margin-bottom: 8px;
}

/* ── Expanders & tables ── */
[data-testid="stExpander"] {
    border: 1px solid #f0f0f0 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #888 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-size: 0.85rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    with st.spinner("Modeller yükleniyor (mE5 + CLIP)..."):
        st.session_state.retriever = VisualRetriever(top_k_text=5, top_k_visual=3)

if "doc_names" not in st.session_state:
    try:
        reg = DocRegistry(uri=settings.mongo_uri, db_name=settings.mongo_db)
        st.session_state.doc_names = reg.get_all_doc_names()
    except Exception:
        st.session_state.doc_names = []

filter_docs = None
score_threshold = 0.10

# Sidebar'ı her zaman açık tut
st.markdown("""
<script>
(function() {
    var key = Object.keys(localStorage).find(k => k.includes('sidebar'));
    if (key) localStorage.removeItem(key);
    var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) sidebar.style.transform = 'none';
})();
</script>
""", unsafe_allow_html=True)

# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

_SPEC_KEYWORDS = {
    "weight", "mass", "kg", "pound", "lb", "speed", "dimension",
    "length", "width", "height", "degree", "freedom", "agirlik", "kilo",
}

def _inject_spec_chunks(query: str, prompt: str) -> str:
    """Temel spec sorgusu algılanırsa Physical Properties tablosunu prompt'a ekler."""
    import re as _re
    words = set(_re.sub(r"[^\w\s]", "", query.lower()).split())
    if not words & _SPEC_KEYWORDS:
        return prompt
    try:
        from pymongo import MongoClient
        col = MongoClient(settings.mongo_uri)[settings.mongo_db][settings.mongo_collection]
        row = col.find_one(
            {"doc_name": "spot-user-manual-en", "breadcrumb": ["Physical properties"]},
            {"raw_text": 1, "page_range": 1},
        )
        if row:
            table_text = row["raw_text"]
            pr = row["page_range"]
            header = f"[Spot User Manual, p.{pr[0]}] Physical properties [TABLE]"
            injection = f"\n\n{header}\n{table_text}"
            # Context bloğundan hemen sonra, Question'dan önce ekle
            insert_at = prompt.find("\n\nQuestion:")
            if insert_at != -1:
                prompt = prompt[:insert_at] + injection + prompt[insert_at:]
    except Exception:
        pass
    return prompt


def _ollama_unload(model: str, wait: float = 0.0) -> None:
    """Modeli VRAM'dan boşaltır (keep_alive=0). wait>0 ise boşalma için bekler."""
    import time as _time
    try:
        requests.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
    except Exception:
        pass
    if wait > 0:
        _time.sleep(wait)


def call_ollama_vision(image_b64: str, prompt: str) -> str:
    """minicpm-v ile görseli tanımlar, tam cevabı string olarak döner."""
    payload = {
        "model": settings.ollama_vision_model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 1500, "stop": ["<|endoftext|>", "<|im_end|>"]},
    }
    try:
        r = requests.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        return f"[Görsel analiz hatası: {e}]"


def call_ollama_stream(system: str, user: str, image_b64: str | None = None):
    """Ollama stream. RAM yetersizse 9b fallback, model gecikmesi için 1 retry."""
    import json as _json, time as _time
    primary_model = settings.ollama_vision_model if image_b64 else settings.ollama_model
    fallback_model = "qwen3.5:9b"
    models_to_try = [primary_model]
    if primary_model != fallback_model:
        models_to_try.append(fallback_model)

    user_msg: dict = {"role": "user", "content": user}
    if image_b64:
        user_msg["images"] = [image_b64]
    messages = [{"role": "system", "content": system}, user_msg]
    if not image_b64:
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n"})

    for model_idx, model in enumerate(models_to_try):
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": 16384, "num_predict": 2048, "stop": ["<|endoftext|>", "<|im_end|>"]},
        }
        for attempt in range(2):
            try:
                with requests.post(
                    f"{settings.ollama_host}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=300,
                ) as r:
                    if r.status_code == 500:
                        body = r.text
                        if "memory" in body.lower() and model_idx == 0 and model != fallback_model:
                            yield f"\n\n*({model} yüklenemedi, {fallback_model} ile devam ediliyor...)*\n\n"
                            break  # fallback modele geç
                        if attempt == 0:
                            _time.sleep(8)
                            continue
                        yield f"\n\n⚠️ Hata: {body[:200]}"
                        return
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            chunk = _json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if chunk.get("done"):
                                break
                return
            except requests.exceptions.ConnectionError:
                yield f"\n\n⚠️ Ollama bağlantı hatası: {settings.ollama_host}"
                return
            except Exception as e:
                if attempt == 0:
                    _time.sleep(8)
                    continue
                yield f"\n\n⚠️ Hata: {e}"
                return
            return


_NO_ANSWER_PHRASES = (
    # "not X" kalıpları
    "not specified", "not mentioned", "not found", "not provided",
    "not available", "not included", "not covered", "not discussed",
    "not addressed", "not listed", "not stated", "not detailed",
    # "does/do not" kalıpları
    "does not specify", "does not mention", "does not include",
    "does not contain", "does not provide", "does not cover",
    "does not discuss", "does not state", "does not address",
    "do not specify", "do not mention", "do not include",
    # diğer
    "cannot be found", "no information", "no mention",
    "no details", "no data", "no warranty", "no pricing",
    "unable to find", "context does not", "not contain",
    "manual does not", "document does not",
    # Türkçe
    "bulunamadı", "belirtilmemiş", "bilgi yok", "yer almıyor",
    "bahsedilmiyor", "içermemektedir",
)

def _llm_found_answer(text: str) -> bool:
    """LLM cevabı gerçek bir bilgi içeriyor mu, yoksa 'bulunamadı' mı diyor?"""
    # Uzun yanıtlar kısmi "not found" içerse de gerçek bilgi barındırır
    if len(text.split()) > 40:
        return True
    low = text.lower()
    return not any(p in low for p in _NO_ANSWER_PHRASES)


_DOC_LABELS = {
    "spot-user-manual-en": "Spot User Manual",
    "spot-arm-user-manual-en": "Spot Arm Manual",
    "spot-dock-user-manual-en": "Spot Dock Manual",
    "spot-station-user-manual-en": "Spot Station Manual",
    "spot-cam-2-user-manual-en": "Spot Cam 2 Manual",
    "spot-power-supply-user-manual-en": "Spot Power Supply Manual",
}

_BOILERPLATE_TITLES = {"REQUIRED READING"}
_SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>")


def _sanitize(text: str) -> str:
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text


def build_prompt(query: str, result: HybridResult, image_mode: bool = False) -> str:
    parts = ["Context from Spot manuals:\n"]
    # Görsel sorguda CLIP zaten dokümanı filtreledi — düşük eşik kullan
    score_min = 0.1 if image_mode else 0.5
    seen_ids: set = set()
    relevant_chunks = []
    for c in sorted(
        [
            c for c in result.text_chunks
            if c.score >= score_min
            and (c.breadcrumb[0].strip().upper() not in _BOILERPLATE_TITLES if c.breadcrumb else True)
        ],
        key=lambda c: c.score,
        reverse=True,
    ):
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            relevant_chunks.append(c)
        if len(relevant_chunks) == 8:
            break
    for i, chunk in enumerate(relevant_chunks, 1):
        section = " > ".join(chunk.breadcrumb) if chunk.breadcrumb else "General"
        pages = (
            f"p.{chunk.page_range[0]}-{chunk.page_range[1]}"
            if chunk.page_range[0] != chunk.page_range[1]
            else f"p.{chunk.page_range[0]}"
        )
        doc_label = _DOC_LABELS.get(chunk.doc_name, chunk.doc_name)
        header = f"[{doc_label}, {pages}] {section}"
        if chunk.chunk_type == "table":
            header += " [TABLE]"
        parts.append(f"{header}\n{_sanitize(chunk.raw_text)}")

    # Only include stored visual chunk descriptions for text-only queries.
    # For image queries the VLM description is appended separately; stored visuals add noise.
    if result.visual_chunks and not getattr(result, "query_image_b64", None):
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
    return f"{context}\n\n---\n\nQuestion: {query}\n\nAnswer based on the context above: /no_think"


SYSTEM_PROMPT = """You are a Boston Dynamics Spot manual assistant. Answer in the same language as the question. Answer using ONLY the provided context. Report exactly what the manual says — include all relevant details and descriptions, word for word where possible. Do NOT omit information, do NOT summarize. Do NOT add commentary or information not in the context.
Format your answer with clear paragraphs and line breaks. For step-by-step procedures, use numbered steps (1. 2. 3.). Use bullet points for non-ordered lists. Each distinct topic or component should be its own paragraph or section. Do not run everything into one block of text.
Place citations only at the END of each paragraph or section — NOT after every sentence. Use a single citation per paragraph at the end, e.g. (Spot User Manual, p.12). Do NOT place citations at the start of a line. Do NOT add a separate source list at the end.
Do NOT use opening phrases like "Based on the context" or "According to the manual". Do NOT use "Note:", "Notice:", or similar lead-ins.
Answer ONLY what was asked. Ignore context chunks unrelated to the question. If multiple variants exist (e.g. Spot alone vs Spot with Arm), list each clearly.
When the context mentions both operating temperature and storage temperature, treat them as distinct values. Operating temperature = the range during active use. Storage temperature = the range when powered off and stored. Never label storage temperature as operating temperature.
If not found in context, respond with ONLY this sentence: "This information is not available in the Spot User Manual." Do NOT add notes, caveats, related information, or partial answers.
Reproduce any relevant [TABLE] in full markdown — every row, no abbreviation."""

VISION_SYSTEM_PROMPT = """You are a Boston Dynamics Spot manual assistant. You will receive a visual analysis of an uploaded image and relevant manual context.
Answer directly. Include all manual details relevant to the question — full procedure steps, specifications, and warnings. Do NOT pad the answer with unrelated manual sections.
Format your answer with clear paragraphs. For step-by-step procedures, use numbered steps (1. 2. 3.). Do not run everything into one block of text.
Place citations only at the END of each paragraph or section — NOT after every sentence. Use a single citation per paragraph at the end, e.g. (Spot Station Manual, p.23). Do NOT place citations at the start of a line. Do NOT add a separate source list at the end.
Do NOT use opening phrases like "Based on the context" or "According to the manual". Do NOT use "Note:", "Notice:", or similar lead-ins.
If the note says CLIP matched this image to a specific Spot product, accept that identification and use it. Otherwise, identify the device from visible labels or distinctive visual features (e.g. quadruped robot shape, charger case, cables). List the components visible in the image. Then answer the question using the manual context.
Use the visual analysis to identify components. Ignore dimensions, part numbers, or descriptions in the visual analysis that are NOT confirmed by the manual context — the image analysis may contain errors.
If not found in context, respond with ONLY this sentence: "This information is not available in the Spot manuals." Do NOT add notes or partial answers."""


def render_inline_tables(result: HybridResult, min_score: float = 0.35) -> None:
    """Tablo chunk'larını LLM cevabının altında doğrudan render eder.
    Sadece yeterince alakalı tabloları gösterir (appendix/URL tabloları filtreler)."""
    import re as _re

    def _is_front_matter(chunk) -> bool:
        """Gerçek cevap içermeyen ön/arka matter tabloları filtreler."""
        first = chunk.breadcrumb[0].strip() if chunk.breadcrumb else ""
        return bool(
            _re.match(r'^1\.[1-5][\.\s]', first)   # 1.1–1.5: About/Terminology/Scope/Conventions/Specs
            or _re.match(r'^7\.', first)            # 7.x: EU Declaration, Compliance, Certifications
            or first.upper() in ("REQUIRED READING",)
        )

    query_stems = {
        (w[:5] if len(w) >= 5 else w).lower()
        for w in _re.sub(r"[^\w\s]", "", result.query_text or "").split()
        if len(w) >= 4
    }
    _STOP = {
        "what", "whic", "wher", "when", "does", "spot", "abou",
        "show", "look", "tell", "give", "find", "list", "like",
        "make", "life", "time", "also", "need", "want", "this",
        "that", "with", "from", "have", "will", "them", "then",
    }
    query_stems -= _STOP

    def _table_matches_query(chunk) -> bool:
        if not query_stems:
            return True
        # Parantez içlerini sil: "(with Battery)" gibi niteleyiciler tabloyu ilgili yapmamalı
        raw_stripped = _re.sub(r'\([^)]*\)', '', chunk.raw_text).lower()
        haystack = raw_stripped + " " + " ".join(chunk.breadcrumb).lower()
        if len(query_stems) == 1:
            # Tek-stem sorgular çok geniş eşleşiyor — table_fetch tabloları gösterme
            return next(iter(query_stems)) in haystack
        return sum(1 for s in query_stems if s in haystack) >= 2

    # Yanıtta öne çıkan dokümanları bul (non-table chunk'lara göre top-2)
    doc_counts = _Counter(
        c.doc_name for c in result.text_chunks
        if c.chunk_type != "table" and c.score >= 0.5 and c.doc_name
    )
    top2_docs = {doc for doc, _ in doc_counts.most_common(2)} if doc_counts else None

    table_chunks = [
        c for c in result.text_chunks
        if c.chunk_type == "table"
        and not _is_front_matter(c)
        and (top2_docs is None or c.doc_name in top2_docs)  # yanıtla aynı dokümanlar
        and (
            (getattr(c, "source", "") == "table_fetch" and c.score >= 0.6 and _table_matches_query(c))
            or (getattr(c, "source", "") == "hybrid" and c.score >= 0.95 and _table_matches_query(c))
            or (getattr(c, "source", "") == "hybrid" and c.score >= 0.78 and _table_matches_query(c))
        )
    ]
    if not table_chunks:
        return

    st.markdown("**📊 Kaynak Tablolar**")
    for chunk in table_chunks:
        section = " > ".join(chunk.breadcrumb) if chunk.breadcrumb else "Tablo"
        page = chunk.page_range[0]
        with st.expander(f"{section} — s.{page}", expanded=True):
            st.markdown(chunk.raw_text)


def render_inline_visuals(result: HybridResult, min_score: float = 0.35) -> None:
    """Görselleri LLM cevabının hemen altında gösterir."""
    import re as _re2

    _VIS_STEMS = {"look", "show", "pictu", "diagr", "appea", "photo", "visua", "illus", "depic", "displ", "figur"}

    if getattr(result, "query_image_b64", None):
        # Görsel sorgu: tüm görselleri göster (image_query + page_match)
        visuals = [v for v in result.visual_chunks if v.image_b64 and v.score >= min_score]
    else:
        # Sorguda görsel niyet var mı? ("look like", "show me", "diagram" vb.)
        query = result.query_text or ""
        q_stems = {w[:5] for w in _re2.sub(r"[^\w\s]", "", query.lower()).split() if len(w) >= 4}
        has_visual_intent = bool(q_stems & _VIS_STEMS)

        if has_visual_intent:
            # "Look like / show me" tarzı sorgular: page_match + CLIP text_query (geniş threshold)
            visuals = [
                v for v in result.visual_chunks
                if v.image_b64 and (
                    (v.source == "page_match" and v.score >= min_score)
                    or (v.source == "text_query" and v.score >= 0.62)
                )
            ]
        else:
            # Spesifikasyon/prosedür soruları: yalnızca yüksek CLIP benzerliği
            visuals = [
                v for v in result.visual_chunks
                if v.image_b64 and v.source == "text_query" and v.score >= 0.74
            ]
    if not visuals:
        return

    visuals = visuals[:4]  # maksimum 4 görsel göster
    st.markdown("**📸 İlgili Görseller**")
    cols = st.columns(min(len(visuals), 3))
    for i, vis in enumerate(visuals):
        with cols[i % 3]:
            try:
                img_data = base64.b64decode(vis.image_b64)
                img = Image.open(io.BytesIO(img_data))
                caption = f"s.{vis.page}"
                if vis.figure_caption:
                    caption += f" — {vis.figure_caption[:60]}"
                st.image(img, use_container_width=True, caption=caption)
            except Exception:
                st.info(f"Görsel yüklenemedi (s.{vis.page})")


def render_retrieved_chunks(result: HybridResult) -> None:
    """Retrieval sonuçlarını chunk skoru bilgi kutusu olarak gösterir."""
    if not result.text_chunks and not result.visual_chunks:
        return

    visible_chunks = [c for c in result.text_chunks if c.score >= 0.5 and getattr(c, "source", "") != "page_sibling"]
    total = len(visible_chunks) + len(result.visual_chunks)
    with st.expander(f"🔍 Retrieval Detayları — {total} chunk", expanded=False):
        rows = []
        for c in visible_chunks:
            section = " > ".join(c.breadcrumb[-2:]) if c.breadcrumb else "—"
            rows.append({
                "Tür": c.chunk_type.upper(),
                "Doküman": c.doc_name.replace("-user-manual-en", ""),
                "Bölüm": section,
                "Sayfa": f"s.{c.page_range[0]}",
                "Kaynak": c.source,
                "Skor": round(c.score, 3),
            })
        for v in result.visual_chunks:
            section = v.breadcrumb[-1] if v.breadcrumb else "—"
            rows.append({
                "Tür": "GÖRSEL",
                "Doküman": v.doc_name.replace("-user-manual-en", ""),
                "Bölüm": section,
                "Sayfa": f"s.{v.page}",
                "Kaynak": getattr(v, "source", "clip"),
                "Skor": round(v.score, 3),
            })

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Skor": st.column_config.ProgressColumn(
                        "Skor", min_value=0, max_value=1, format="%.3f"
                    )
                },
            )


# ─── KAYNAKLAR ────────────────────────────────────────────────────────────────

def render_kaynaklar(result: HybridResult) -> None:
    # Her doc'un en yüksek TEXT/MIXED chunk skorunu bul
    doc_best: dict[str, float] = {}
    for c in result.text_chunks:
        if c.chunk_type in ("TEXT", "MIXED") and c.score >= 0.5 and c.doc_name:
            if c.score > doc_best.get(c.doc_name, 0):
                doc_best[c.doc_name] = c.score

    # Sadece en iyi doc'un 0.18 puan içindeki doc'ları göster
    if doc_best:
        best = max(doc_best.values())
        top_docs = {doc for doc, s in doc_best.items() if s >= best - 0.25}
    else:
        top_docs = None

    chunks = sorted(
        [c for c in result.text_chunks
         if c.score >= 0.5 and (top_docs is None or c.doc_name in top_docs)],
        key=lambda c: c.score, reverse=True
    )
    seen = set()
    sources = []
    for c in chunks:
        key = (c.doc_name, c.page_range[0])
        if key not in seen:
            seen.add(key)
            sources.append(c)
        if len(sources) >= 5:
            break
    if not sources:
        return
    items_html = ""
    for c in sources:
        doc = _DOC_LABELS.get(c.doc_name, c.doc_name)
        page = c.page_range[0]
        page2 = c.page_range[1]
        page_str = f"p.{page}–{page2}" if page != page2 else f"p.{page}"
        pdf_url = f"/app/static/pdfs/{c.doc_name}.pdf#page={page}" if c.doc_name else ""
        page_badge = (
            f'<a href="{pdf_url}" target="_blank" style="color:#D42B2B;font-size:0.78rem;float:right;text-decoration:none;">{page_str} &#8599;</a>'
            if pdf_url else
            f'<span style="color:#D42B2B;font-size:0.78rem;float:right;">{page_str}</span>'
        )
        items_html += (
            f'<p style="margin:3px 0;font-size:0.83rem;border-bottom:1px solid #f5f5f5;padding-bottom:3px;">'
            f'<span style="color:#D42B2B;margin-right:6px;">&#9679;</span>'
            f'<strong style="color:#333;">{doc}</strong>'
            f'{page_badge}'
            f'</p>'
        )
    st.markdown(
        f'<div style="background:#fff;border:1px solid #eee;border-radius:10px;padding:12px 16px;margin:6px 0 16px 0;">'
        f'<p style="font-size:0.72rem;font-weight:700;color:#999;letter-spacing:1.5px;margin:0 0 8px 0;">KAYNAKLAR</p>'
        f'{items_html}</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar — Belgeler ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p style="font-size:0.7rem;font-weight:700;color:#aaa;letter-spacing:1.5px;margin:8px 0 12px 0;">BELGELER</p>',
        unsafe_allow_html=True,
    )
    doc_names = st.session_state.get("doc_names", [])
    if doc_names:
        for dn in sorted(doc_names):
            label = _DOC_LABELS.get(dn, dn)
            url = f"/app/static/pdfs/{dn}.pdf"
            st.markdown(
                f'<div style="padding:8px 10px;border-radius:8px;margin-bottom:4px;background:#fff;border:1px solid #eee;">'
                f'<span style="color:#D42B2B;margin-right:8px;font-size:0.75rem;">&#9679;</span>'
                f'<a href="{url}" target="_blank" style="font-size:0.83rem;color:#333;text-decoration:none;">{label}</a>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Belge bulunamadı.")

    _STATIC_DOCS = ["SSB 2026 Ürün Kataloğu", "Clearpath Robotics"]
    for label in _STATIC_DOCS:
        st.markdown(
            f'<div style="padding:8px 10px;border-radius:8px;margin-bottom:4px;background:#fff;border:1px solid #eee;">'
            f'<span style="color:#D42B2B;margin-right:8px;font-size:0.75rem;">&#9679;</span>'
            f'<span style="font-size:0.83rem;color:#333;">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─── Ana chat arayüzü ─────────────────────────────────────────────────────────

# RAGORA Header
if not st.session_state.messages:
    st.markdown("""
<div class="rg-header">
<div class="rg-header-row">
<div class="rg-logo">R</div>
<span class="rg-title">RAGORA</span>
</div>
</div>
""", unsafe_allow_html=True)

# Temizle butonu
if st.session_state.messages:
    col_spacer, col_btn = st.columns([6, 1])
    with col_btn:
        if st.button("Temizle", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# Geçmiş mesajları göster
for msg in st.session_state.messages:
    if msg["role"] == "user":
        img_html = ""
        if msg.get("has_image"):
            img_html = '<br><span style="color:#999;font-size:0.8rem;">📎 Görsel eklendi</span>'
        st.markdown(f"""
        <div class="rg-user-row">
            <div class="rg-user-bubble">{msg["content"]}{img_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar=None):
            st.markdown('<div class="rg-ai-header">&#9679;&nbsp; RAGORA</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if "result" in msg and msg["result"] and _llm_found_answer(msg.get("content", "")):
                if not getattr(msg["result"], "query_image_b64", None):
                    render_inline_tables(msg["result"])
                render_inline_visuals(msg["result"])
        if "result" in msg and msg["result"] and _llm_found_answer(msg.get("content", "")):
            render_kaynaklar(msg["result"])
        render_retrieved_chunks(msg["result"]) if "result" in msg and msg["result"] else None

chat_input = st.chat_input(
    "Belgeler hakkında bir soru sor...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "bmp", "webp"],
)

query = chat_input.text if chat_input else None
uploaded_image = chat_input.files[0] if (chat_input and chat_input.files) else None

if query:
    img_html = '<br><span style="color:#999;font-size:0.8rem;">📎 Görsel eklendi</span>' if uploaded_image else ""
    st.markdown(f"""
    <div class="rg-user-row">
        <div class="rg-user-bubble">{query}{img_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": query, "has_image": bool(uploaded_image)})

    # Retrieval
    retriever: VisualRetriever = st.session_state.retriever
    retriever._text_retriever.top_k = 12
    retriever._text_retriever.score_threshold = score_threshold
    retriever.top_k_text = 12
    retriever.top_k_visual = 3

    with st.spinner("Belgeler aranıyor..."):
        if uploaded_image:
            uploaded_image.seek(0)
            img_bytes = uploaded_image.read()
            query_image_b64 = base64.b64encode(img_bytes).decode()

            import tempfile, os
            suffix = Path(uploaded_image.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = Path(tmp.name)

            result = retriever.retrieve_by_image(
                image_input=tmp_path,
                query_text=query,
                filter_docs=filter_docs,
            )
            os.unlink(tmp_path)
        else:
            query_image_b64 = None
            _TR_CHARS = set('ıİşŞçÇğĞöÖüÜ')
            _TR_EN = {
                "çalışma sıcaklığı": "operating temperature",
                "depolama sıcaklığı": "storage temperature",
                "sıcaklık": "temperature",
                "pil": "battery",
                "şarj": "charge",
                "güç": "power",
                "ağırlık": "weight",
                "hız": "speed",
                "güvenlik": "safety",
                "kaldır": "lift",
                "boyut": "dimensions",
            }
            retrieval_query = query
            if any(c in _TR_CHARS for c in query):
                q_lower = query.lower()
                for tr, en in _TR_EN.items():
                    if tr in q_lower:
                        retrieval_query = f"{query} {en}"
                        break
            result = retriever.retrieve_by_text(
                query=retrieval_query,
                filter_docs=filter_docs,
                auto_lang=True,
            )

    # Asistan cevabı
    with st.chat_message("assistant", avatar=None):
        st.markdown('<div class="rg-ai-header">&#9679;&nbsp; RAGORA</div>', unsafe_allow_html=True)
        if not result.text_chunks and not result.visual_chunks:
            answer = "⚠️ İlgili içerik bulunamadı. Soruyu farklı kelimelerle deneyin."
            st.warning(answer)
        else:
            if query_image_b64:
                # Step 1: VLM extracts text labels AND visually describes the image
                _ollama_unload(settings.ollama_model)  # 27b'yi boşalt, 7b için yer aç
                with st.spinner(f"🔍 Görsel analiz ediliyor ({settings.ollama_vision_model})..."):
                    vision_desc = call_ollama_vision(
                        query_image_b64,
                        "Analyze this technical image in two parts:\n"
                        "1. TEXT: List every piece of text, label, number, or warning visible in the image. Write each item on its own line. If no text is visible, write 'No text visible'.\n"
                        "2. VISUAL: Describe what you see in mechanical/technical detail — component shape, color, material, connections, cable routing, position relative to other parts, quantity of items, and any notable features. Be specific."
                    )
                _ollama_unload(settings.ollama_vision_model, wait=3.0)  # 7b bitti, 27b için yer aç

                # Step 2: Combine user question + VLM description for richer retrieval
                vision_search_query = f"{query} {vision_desc[:250]}".strip()
                with st.spinner("📖 Görsel açıklamasıyla manuel aranıyor..."):
                    vision_text_result = retriever.retrieve_by_text(
                        query=vision_search_query,
                        filter_docs=filter_docs,
                        auto_lang=False,
                    )

                # Step 3: Identify the document.
                # Priority: (1) VLM accessory keywords, (2) CLIP image match, (3) VLM text search
                _VLM_DOC_HINTS: dict[str, set[str]] = {
                    "spot-power-supply-user-manual-en": {
                        "charger", "shore power", "power cable", "charging case",
                        "rugged case", "charging port", "power supply",
                    },
                    "spot-arm-user-manual-en": {
                        "arm", "gripper", "end effector", "manipulator", "wrist",
                    },
                    "spot-dock-user-manual-en": {
                        "dock", "docking station", "charging dock",
                    },
                    "spot-cam-2-user-manual-en": {
                        "spot cam", "ptz camera", "pan-tilt",
                    },
                    "spot-station-user-manual-en": {
                        "station", "enclosure", "shackle", "lifting eye", "forklift pocket",
                    },
                }
                vlm_lower = vision_desc.lower()
                vlm_identified_doc = None
                for _doc, _keywords in _VLM_DOC_HINTS.items():
                    if any(kw in vlm_lower for kw in _keywords):
                        vlm_identified_doc = _doc
                        break

                from collections import Counter
                image_query_chunks = [
                    v for v in result.visual_chunks
                    if getattr(v, "source", "") == "image_query" and v.doc_name
                ]
                if vlm_identified_doc:
                    # VLM saw accessory-specific keywords — trust this over CLIP
                    identified_doc = vlm_identified_doc
                elif image_query_chunks:
                    clip_doc_counts = Counter(v.doc_name for v in image_query_chunks)
                    identified_doc = clip_doc_counts.most_common(1)[0][0]
                elif vision_text_result.text_chunks:
                    vlm_doc_counts = Counter(
                        c.doc_name for c in vision_text_result.text_chunks if c.doc_name
                    )
                    identified_doc = vlm_doc_counts.most_common(1)[0][0] if vlm_doc_counts else None
                else:
                    identified_doc = None

                if identified_doc:
                    # Keep only chunks from the identified document
                    result.text_chunks = [
                        c for c in result.text_chunks if c.doc_name == identified_doc
                    ]
                    existing_ids = {c.chunk_id for c in result.text_chunks}
                    for c in vision_text_result.text_chunks:
                        if c.chunk_id not in existing_ids and c.doc_name == identified_doc:
                            result.text_chunks.append(c)

                # Step 4: Build prompt with correct-doc context + VLM description
                # image_mode=True yalnızca CLIP dokümanı tespit ettiğinde aktif:
                # chunks zaten identified_doc'a filtrelendiği için 0.1 eşiği güvenli
                user_message = build_prompt(query, result, image_mode=(identified_doc is not None))
                doc_hint = (
                    f"\n\nNote: Visual search (CLIP) matched this image to: "
                    f"{_DOC_LABELS.get(identified_doc, identified_doc)}. "
                    f"Treat the device in the image as the corresponding Spot product."
                    if identified_doc else ""
                )
                user_message += f"\n\n---\n\nVisual Analysis of uploaded image:{doc_hint}\n{vision_desc}"
                if any(c in set('ıİşŞçÇğĞöÖüÜ') for c in query):
                    user_message = "Lütfen Türkçe yanıt ver.\n\n" + user_message
                llm_image = None
            else:
                _ollama_unload(settings.ollama_vision_model)  # 7b'yi boşalt, 27b için yer aç
                # Text-only: filter to top-2 docs by chunk count to avoid cross-manual noise
                # Spot User Manual her zaman dahil edilir — temel spec tabloları burada
                from collections import Counter
                doc_counts = Counter(c.doc_name for c in result.text_chunks if c.doc_name)
                if doc_counts:
                    top2_docs = {doc for doc, _ in doc_counts.most_common(2)}
                    top2_docs.add("spot-user-manual-en")
                    result.text_chunks = [c for c in result.text_chunks if c.doc_name in top2_docs]
                user_message = build_prompt(query, result)
                user_message = _inject_spec_chunks(query, user_message)
                # Türkçe sorgu tespiti — LLM'e anlamsal eşleştirme yapmasını söyle
                _TR_CHARS = set('ıİşŞçÇğĞöÖüÜ')
                if any(c in _TR_CHARS for c in query):
                    user_message += (
                        "\n\nNote: The question is in Turkish. "
                        "Match by meaning, not by literal words. "
                        "Turkish terms: pil=battery, şarj=charge, güç=power, "
                        "kaldır=lift/remove, tak=insert/attach, hız=speed, "
                        "ağırlık=weight, güvenlik=safety, "
                        "çalışma sıcaklığı=operating temperature (NOT storage temperature), "
                        "depolama sıcaklığı=storage temperature, sıcaklık=temperature. "
                        "Important: distinguish clearly between operating temperature and storage temperature."
                    )
                llm_image = None
            active_system_prompt = VISION_SYSTEM_PROMPT if query_image_b64 else SYSTEM_PROMPT
            answer_placeholder = st.empty()
            full_answer = ""
            for token in call_ollama_stream(active_system_prompt, user_message, llm_image):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)
            if _llm_found_answer(full_answer):
                if not query_image_b64:
                    render_inline_tables(result)
                render_inline_visuals(result)

    if _llm_found_answer(full_answer):
        render_kaynaklar(result)
    render_retrieved_chunks(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer if result.text_chunks or result.visual_chunks else answer,
        "result": result,
    })
