"""
Streamlit Chatbot Arayüzü
RAG Destekli Türkçe Soru-Cevap Sistemi + Vision Desteği
"""
import sys
from pathlib import Path
import html
import inspect
import re
import base64
from io import BytesIO
from urllib.parse import urlparse
import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from PIL import Image

# Ensure project root is importable regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CHATBOT_TITLE, CHATBOT_WELCOME, CATEGORIES
from rag.unified_router import UnifiedRAGRouter
from rag.vision_rag import VisionRAGSystem

TURKEY_TZ = ZoneInfo("Europe/Istanbul")


# Sayfa yapılandırması
st.set_page_config(
    page_title=CHATBOT_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Stil
st.markdown("""
<style>
    :root {
        --ragora-red: #e21b1b;
        --ragora-red-soft: #fff1f1;
        --ragora-border: #e5e7eb;
        --ragora-text: #111827;
        --ragora-muted: #6b7280;
        --ragora-bg: #fffdfb;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(226, 27, 27, 0.06), transparent 28%),
            linear-gradient(180deg, #ffffff 0%, #fffdfb 100%);
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #ffffff !important;
    }

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden;
        height: 0;
    }

    .block-container {
        padding-top: 18px;
        padding-bottom: 110px;
        max-width: 1160px;
    }

    section[data-testid="stSidebar"] {
        background: #fffaf6;
        border-right: 1px solid var(--ragora-border);
        min-width: 292px !important;
        width: 292px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 28px 18px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 64px;
        justify-content: flex-start;
        border: 1px solid var(--ragora-border);
        background: #ffffff;
        color: var(--ragora-text);
        border-radius: 8px;
        font-weight: 650;
        text-align: left;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.04);
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #fecaca;
        background: var(--ragora-red-soft);
        color: #991b1b;
    }

    .ragora-shell {
        max-width: 920px;
        margin: 0 auto;
        padding: 8px 4px 24px;
    }

    .ragora-header {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 18px;
        margin: 2px 0 28px;
        letter-spacing: 9px;
        font-size: 30px;
        font-weight: 850;
        color: var(--ragora-text);
    }

    .ragora-logo {
        display: inline-flex;
        width: 48px;
        height: 48px;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: linear-gradient(180deg, #ef4444 0%, #d70909 100%);
        color: #ffffff;
        letter-spacing: 0;
        box-shadow: 0 12px 24px rgba(226, 27, 27, 0.25);
    }

    .ragora-sidebar-title {
        color: var(--ragora-red);
        font-size: 13px;
        font-weight: 850;
        letter-spacing: 4px;
        margin: 22px 0 14px;
    }

    .ragora-sidebar-note {
        color: var(--ragora-muted);
        font-size: 12px;
        line-height: 1.45;
        margin-top: 18px;
    }

    .ragora-user-wrap {
        display: flex;
        justify-content: flex-end;
        margin: 14px 0 22px;
    }

    .ragora-user-bubble {
        max-width: min(560px, 82%);
        background: var(--ragora-red-soft);
        border: 1px solid #fecaca;
        color: var(--ragora-text);
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0 10px 28px rgba(226, 27, 27, 0.06);
    }

    .ragora-time {
        margin-top: 7px;
        color: var(--ragora-muted);
        font-size: 12px;
        text-align: right;
    }

    .ragora-answer-card {
        border: 1px solid #d1d5db;
        border-left: 4px solid var(--ragora-red);
        background: rgba(255, 255, 255, 0.92);
        border-radius: 8px;
        padding: 22px 26px;
        margin: 12px 0 16px;
        box-shadow: 0 16px 42px rgba(17, 24, 39, 0.05);
    }

    .ragora-card-head {
        display: flex;
        align-items: center;
        gap: 12px;
        color: var(--ragora-red);
        font-size: 13px;
        font-weight: 850;
        letter-spacing: 2px;
        margin-bottom: 18px;
    }

    .ragora-mini-logo {
        display: inline-flex;
        width: 34px;
        height: 34px;
        align-items: center;
        justify-content: center;
        border-radius: 7px;
        background: var(--ragora-red);
        color: #ffffff;
        font-size: 20px;
        letter-spacing: 0;
    }

    .ragora-answer-body {
        color: var(--ragora-text);
        font-size: 16px;
        line-height: 1.75;
    }

    .ragora-answer-body p {
        margin: 0 0 14px;
    }

    .ragora-answer-body p:last-child {
        margin-bottom: 0;
    }

    .ragora-answer-body h1,
    .ragora-answer-body h2,
    .ragora-answer-body h3,
    .ragora-answer-body h4 {
        color: var(--ragora-text);
        letter-spacing: 0;
        line-height: 1.25;
        margin: 20px 0 10px;
    }

    .ragora-answer-body h1 { font-size: 22px; }
    .ragora-answer-body h2 { font-size: 20px; }
    .ragora-answer-body h3 { font-size: 18px; }
    .ragora-answer-body h4 { font-size: 16px; }

    .ragora-answer-body ul,
    .ragora-answer-body ol {
        margin: 8px 0 16px 22px;
        padding: 0;
    }

    .ragora-answer-body li {
        margin: 5px 0;
        padding-left: 2px;
    }

    .ragora-answer-body strong {
        color: #0f172a;
        font-weight: 800;
    }

    .ragora-answer-body table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0 18px;
        font-size: 14px;
    }

    .ragora-answer-body th,
    .ragora-answer-body td {
        border: 1px solid #e5e7eb;
        padding: 9px 10px;
        text-align: left;
        vertical-align: top;
    }

    .ragora-answer-body th {
        background: #fff5f5;
        color: #991b1b;
        font-weight: 800;
    }

    .ragora-answer-body code {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 2px 5px;
        font-size: 0.92em;
    }

    .ragora-answer-body pre {
        background: #0f172a;
        color: #f8fafc;
        border-radius: 8px;
        padding: 14px 16px;
        overflow-x: auto;
        margin: 14px 0 18px;
    }

    .ragora-answer-body pre code {
        background: transparent;
        border: 0;
        color: inherit;
        padding: 0;
    }

    .ragora-sources {
        border: 1px solid #d1d5db;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 8px;
        padding: 22px 26px 14px;
        margin: 8px 0 30px;
        box-shadow: 0 16px 42px rgba(17, 24, 39, 0.04);
    }

    .ragora-sources-title {
        color: var(--ragora-red);
        font-size: 13px;
        font-weight: 850;
        letter-spacing: 3px;
        margin-bottom: 12px;
    }

    .ragora-source-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 18px;
        align-items: center;
        padding: 12px 0;
        border-top: 1px solid #edf0f3;
    }

    .ragora-source-main {
        min-width: 0;
    }

    .ragora-source-name {
        color: var(--ragora-text);
        font-weight: 760;
        overflow-wrap: anywhere;
    }

    .ragora-source-detail {
        color: var(--ragora-muted);
        margin-top: 3px;
        font-size: 13px;
        line-height: 1.45;
    }

    .ragora-source-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--ragora-red);
        font-size: 13px;
        white-space: nowrap;
    }

    .ragora-score {
        border: 1px solid #fecaca;
        background: var(--ragora-red-soft);
        border-radius: 999px;
        padding: 4px 8px;
        font-weight: 750;
    }

    .ragora-source-meta a {
        color: var(--ragora-red) !important;
        text-decoration: none;
        font-weight: 800;
    }

    div[data-testid="stChatInput"] {
        max-width: 920px;
        margin: 0 auto;
    }

    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stChatFloatingInputContainer"] {
        background: #ffffff !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    .ragora-attachment-shell {
        max-width: 920px;
        margin: 0 auto 10px;
        padding: 0 4px;
    }

    .ragora-attachment-panel {
        border: 1px solid #fecaca;
        background: #fffafa;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 12px 30px rgba(226, 27, 27, 0.05);
    }

    .ragora-attachment-title {
        color: var(--ragora-red);
        font-size: 12px;
        font-weight: 850;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .ragora-attachment-help {
        color: var(--ragora-muted);
        font-size: 12px;
        margin-top: 6px;
        line-height: 1.45;
    }

    .ragora-attachment-status {
        color: #14532d;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 8px 10px;
        font-size: 13px;
        margin-top: 10px;
    }

    .ragora-user-image,
    .ragora-pending-image {
        width: 150px;
        max-height: 110px;
        object-fit: cover;
        border-radius: 8px;
        border: 1px solid #fecaca;
        display: block;
        margin-top: 10px;
    }

    .ragora-user-image {
        margin-left: auto;
    }

    .ragora-pending-row {
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
    }

    div[data-testid="stChatInput"] > div {
        background: #ffffff !important;
        border: 1px solid var(--ragora-red) !important;
        border-radius: 18px !important;
        box-shadow: 0 12px 30px rgba(226, 27, 27, 0.08);
        padding: 10px 12px !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: var(--ragora-text) !important;
        border: 0 !important;
        min-height: 62px !important;
        box-shadow: none !important;
        font-size: 16px !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
    }

    div[data-testid="stChatInput"] button {
        background: var(--ragora-red-soft) !important;
        border: 1px solid #fecaca !important;
        color: var(--ragora-red) !important;
        border-radius: 12px !important;
    }

    .ragora-footnote {
        color: var(--ragora-muted);
        text-align: center;
        font-size: 13px;
        margin-top: 10px;
    }

    @media (max-width: 760px) {
        .ragora-header {
            font-size: 22px;
            letter-spacing: 5px;
            margin-bottom: 18px;
        }

        .ragora-logo {
            width: 42px;
            height: 42px;
        }

        .ragora-answer-card,
        .ragora-sources {
            padding: 18px 18px;
        }

        .ragora-source-row {
            grid-template-columns: 1fr;
            gap: 8px;
        }

        .ragora-user-bubble {
            max-width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """RAG sistemini başlat (cache ile)"""
    return UnifiedRAGRouter()


@st.cache_resource
def initialize_vision_rag(_rag_system):
    """Vision RAG sistemini başlat (cache ile)
    
    Args:
        _rag_system: Mevcut RAG sistemi (underscore ile başlar = cache'de ignore edilir)
    """
    clearpath_rag = getattr(_rag_system, "clearpath_rag", _rag_system)
    return VisionRAGSystem(text_rag=clearpath_rag)


def _format_time(value) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(TURKEY_TZ).strftime("%H:%M")
    return _now_turkey().strftime("%H:%M")


def _now_turkey() -> datetime:
    return datetime.now(TURKEY_TZ)


def _category_label(category_key: str) -> str:
    if category_key in CATEGORIES:
        return CATEGORIES[category_key].get("name", category_key)
    return category_key or "Belge"


def _document_label(source: str, category: str = "") -> str:
    category_name = _category_label(category)
    source_text = source or ""
    path = urlparse(source_text).path.lower()

    if category == "spot" and source_text:
        return source_text if source_text.endswith(".pdf") else f"{source_text}.pdf"
    if "user_manual" in path:
        return f"{category_name} Kullanım Kılavuzu"
    if "accessories" in path:
        return f"{category_name} Dokümanı"
    if "docs.clearpathrobotics.com" in source_text:
        return f"{category_name} Dokümantasyonu"
    return category_name or "Kaynak Doküman"


def _source_detail_text(source: dict) -> str:
    section_title = str(source.get("section_title") or "").strip()
    page_range = source.get("page_range") or []
    document_name = _document_label(
        str(source.get("source", "")),
        str(source.get("category", "")),
    )

    page_text = ""
    if isinstance(page_range, list) and page_range:
        page_text = f" s.{page_range[0]}"
    if section_title:
        return f"{document_name}{page_text} içinde \"{section_title}\" bölümünde yer almaktadır."
    return f"{document_name} içinde yer almaktadır."


def _source_href(source: dict) -> str:
    href = str(source.get("source_url") or source.get("href") or source.get("url") or source.get("source", "") or "").strip()
    return href if href.startswith(("http://", "https://", "/app/static/")) else "#"


def _render_inline_markdown(text: str) -> str:
    """Render trusted Markdown syntax after escaping raw HTML from model output."""
    escaped_text = html.escape(text or "")
    try:
        import markdown

        return markdown.markdown(
            escaped_text,
            extensions=["extra", "sane_lists", "nl2br"],
            output_format="html5",
        )
    except Exception:
        return _fallback_markdown_to_html(escaped_text)


def _render_inline_styles(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def _fallback_markdown_to_html(escaped_text: str) -> str:
    """Small Markdown fallback for headings, lists and paragraphs."""
    html_parts = []
    lines = escaped_text.splitlines()
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()

        if not line:
            idx += 1
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            html_parts.append(f"<h{level}>{_render_inline_styles(heading_match.group(2))}</h{level}>")
            idx += 1
            continue

        if re.match(r"^[-*]\s+", line):
            items = []
            while idx < len(lines) and re.match(r"^[-*]\s+", lines[idx].strip()):
                item = re.sub(r"^[-*]\s+", "", lines[idx].strip())
                items.append(f"<li>{_render_inline_styles(item)}</li>")
                idx += 1
            html_parts.append(f"<ul>{''.join(items)}</ul>")
            continue

        if re.match(r"^\d+\.\s+", line):
            items = []
            while idx < len(lines) and re.match(r"^\d+\.\s+", lines[idx].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[idx].strip())
                items.append(f"<li>{_render_inline_styles(item)}</li>")
                idx += 1
            html_parts.append(f"<ol>{''.join(items)}</ol>")
            continue

        paragraph_lines = [line]
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            next_line = lines[idx].strip()
            if re.match(r"^(#{1,4})\s+|^[-*]\s+|^\d+\.\s+", next_line):
                break
            paragraph_lines.append(next_line)
            idx += 1
        html_parts.append(f"<p>{_render_inline_styles('<br>'.join(paragraph_lines))}</p>")

    return "".join(html_parts)


def _text_to_html(text: str) -> str:
    return _render_inline_markdown(text or "")


def _image_bytes_to_data_uri(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""

    try:
        image = Image.open(BytesIO(image_bytes))
        image_format = (image.format or "PNG").lower()
        if image_format == "jpg":
            image_format = "jpeg"
    except Exception:
        image_format = "png"

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{image_format};base64,{encoded}"


def _render_html(html_content: str) -> None:
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        st.markdown(html_content, unsafe_allow_html=True)


def _render_app_header():
    header_html = """
    <div class="ragora-shell">
        <div class="ragora-header">
            <span class="ragora-logo">R</span>
            <span>RAGORA</span>
        </div>
    </div>
    """
    _render_html(header_html)


def _image_ref_from_item(item):
    """Extract a renderable image reference from a URL/path string or image metadata dict."""
    if isinstance(item, dict):
        return (
            item.get("local_path")
            or item.get("image_path")
            or item.get("url")
            or item.get("image_url")
            or item.get("src")
        )
    return item


def _image_caption_from_item(item, fallback: str = "") -> str:
    if not isinstance(item, dict):
        return fallback

    caption = item.get("caption") or item.get("title") or fallback
    category = item.get("category")
    if category and category not in caption:
        caption = f"{caption} - {category}" if caption else category
    return caption


def _render_image(image_ref: str, caption: str = ""):
    """Render a local image path or remote image URL."""
    image_ref = _image_ref_from_item(image_ref)
    if not image_ref:
        return

    try:
        image_path = Path(str(image_ref))
        if image_path.exists():
            st.image(str(image_path), width='stretch', caption=caption)
        else:
            st.image(str(image_ref), width='stretch', caption=caption)
    except Exception:
        if caption:
            st.caption(caption)


def _collect_source_images(sources: list, limit: int = 3) -> list:
    """Pick a few distinct source images to show directly with the answer."""
    collected = []
    seen = set()

    for source in sources or []:
        for image_item in source.get("images", []) or []:
            image_ref = _image_ref_from_item(image_item)
            key = str(image_ref)
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append({
                "image_ref": key,
                "caption": _image_caption_from_item(
                    image_item,
                    source.get("category", "Kaynak gorsel"),
                ),
            })
            if len(collected) >= limit:
                return collected

    return collected


def _render_inline_images(sources: list = None, similar_images: list = None):
    """Show the strongest visual evidence directly below the assistant answer."""
    visual_items = []

    for img_data in (similar_images or [])[:3]:
        visual_items.append({
            "image_ref": img_data.get("image_path") or img_data.get("image_url"),
            "caption": (
                f"{img_data.get('category', 'Benzer gorsel')} "
                f"({float(img_data.get('similarity', 0.0)):.2f})"
            ),
        })

    if not visual_items:
        visual_items = _collect_source_images(sources or [], limit=3)

    visual_items = [item for item in visual_items if item.get("image_ref")]
    if not visual_items:
        return

    st.markdown("**İlgili Görseller**")
    cols = st.columns(min(3, len(visual_items)))
    for idx, item in enumerate(visual_items):
        with cols[idx % len(cols)]:
            _render_image(item["image_ref"], item.get("caption", ""))


def _render_sources_card(sources: list):
    if not sources:
        return

    rows = []
    for source in sources:
        document_name = html.escape(_document_label(
            str(source.get("source", "")),
            str(source.get("category", "")),
        ))
        detail = html.escape(_source_detail_text(source))
        href = html.escape(_source_href(source), quote=True)
        score = float(source.get("similarity", 0.0))
        score_display = str(source.get("score_display") or "").strip()
        score_label = str(source.get("score_label") or "").strip()
        if score_display:
            score_html = html.escape(score_display)
        elif score_label:
            score_html = html.escape(score_label)
        else:
            score_html = f"{score:.1%}"
        rows.append(
            '<div class="ragora-source-row">'
            '<div class="ragora-source-main">'
            f'<div class="ragora-source-name">{document_name}</div>'
            f'<div class="ragora-source-detail">{detail}</div>'
            '</div>'
            '<div class="ragora-source-meta">'
            f'<span class="ragora-score">{score_html}</span>'
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">Kaynağa git</a>'
            '</div>'
            '</div>'
        )

    sources_html = (
        '<div class="ragora-shell">'
        '<div class="ragora-sources">'
        '<div class="ragora-sources-title">KAYNAKLAR</div>'
        f'{"".join(rows)}'
        '</div>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(sources_html)
    else:
        st.markdown(sources_html, unsafe_allow_html=True)


def _chat_input_supports_files() -> bool:
    try:
        return "accept_file" in inspect.signature(st.chat_input).parameters
    except Exception:
        return False


def _extract_uploaded_image(file_obj) -> dict:
    if not file_obj:
        return {}

    file_type = (getattr(file_obj, "type", "") or "").lower()
    file_name = getattr(file_obj, "name", "gorsel.png") or "gorsel.png"
    if file_type and not file_type.startswith("image/"):
        return {}

    try:
        image_bytes = file_obj.getvalue()
        Image.open(BytesIO(image_bytes)).verify()
        return {"bytes": image_bytes, "name": file_name}
    except Exception:
        return {}


def _extract_chat_input_payload(submission) -> tuple[str, list]:
    if submission is None:
        return "", []

    if isinstance(submission, str):
        return submission.strip(), []

    text = getattr(submission, "text", None)
    files = getattr(submission, "files", None)

    if isinstance(submission, dict):
        text = submission.get("text", text)
        files = submission.get("files", files)

    return (text or "").strip(), list(files or [])


def _render_image_attachment_panel() -> None:
    """Legacy placeholder kept for old references; chat_input owns attachments now."""
    return


def format_chat_message(
    role: str,
    content: str,
    sources: list = None,
    similar_images: list = None,
    timestamp=None,
    image_bytes: bytes = None,
    image_name: str = "",
):
    """Chat mesajını formatla."""
    if role == "user":
        image_html = ""
        if image_bytes:
            data_uri = _image_bytes_to_data_uri(image_bytes)
            alt_text = html.escape(image_name or "Eklenen görsel")
            image_html = f'<img class="ragora-user-image" src="{data_uri}" alt="{alt_text}">'

        _render_html(f"""
        <div class="ragora-shell">
            <div class="ragora-user-wrap">
                <div>
                    <div class="ragora-user-bubble">{html.escape(content or "")}</div>
                    {image_html}
                    <div class="ragora-time">{html.escape(_format_time(timestamp))}</div>
                </div>
            </div>
        </div>
        """)
        return

    answer_html = _text_to_html(content or "")
    _render_html(f"""
    <div class="ragora-shell">
        <div class="ragora-answer-card">
            <div class="ragora-card-head">
                <span class="ragora-mini-logo">R</span>
                <span>RAGORA</span>
            </div>
            <div class="ragora-answer-body">
                {answer_html}
            </div>
            <div class="ragora-time">{html.escape(_format_time(timestamp))}</div>
        </div>
    </div>
    """)

    if role == "assistant":
        _render_inline_images(sources=sources, similar_images=similar_images)
        _render_sources_card(sources or [])


def main():
    """Ana uygulama"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "pending_image" not in st.session_state:
        st.session_state.pending_image = None

    with st.sidebar:
        st.markdown('<div class="ragora-sidebar-title">BELGELER</div>', unsafe_allow_html=True)

        document_items = [("Tüm Belgeler", None)] + [
            (CATEGORIES[key].get("name", key), key)
            for key in CATEGORIES.keys()
        ]

        for label, category_key in document_items:
            is_selected = st.session_state.selected_category == category_key
            button_label = f"{'▌ ' if is_selected else ''}📄  {label}"
            if st.button(button_label, key=f"doc_{category_key or 'all'}", width='stretch'):
                st.session_state.selected_category = category_key
                st.rerun()

        st.markdown('<div class="ragora-sidebar-note">Sorular, seçili belge kapsamındaki içerik ve kaynaklarla yanıtlanır.</div>', unsafe_allow_html=True)

        if st.button("Sohbeti Temizle", width='stretch'):
            st.session_state.messages = []
            st.session_state.pending_image = None
            st.rerun()

    selected_category = st.session_state.selected_category
    _render_app_header()
    
    # Chat geçmişini göster
    for message in st.session_state.messages:
        format_chat_message(
            message["role"],
            message["content"],
            message.get("sources"),
            message.get("similar_images"),
            message.get("timestamp"),
            message.get("image_bytes"),
            message.get("image_name", ""),
        )

    # Chat input. Streamlit 1.54+ shows its built-in "+" attachment control.
    chat_input_kwargs = {}
    if _chat_input_supports_files():
        chat_input_kwargs = {
            "accept_file": True,
            "file_type": ["jpg", "jpeg", "png", "webp"],
        }

    submission = st.chat_input(
        "Belgeler veya eklediğiniz görsel hakkında bir soru sor...",
        **chat_input_kwargs,
    )
    prompt, chat_files = _extract_chat_input_payload(submission)

    if submission:
        chat_image = {}
        for file_obj in chat_files:
            chat_image = _extract_uploaded_image(file_obj)
            if chat_image:
                break

        pending_image = st.session_state.get("pending_image")
        attached_image = chat_image or pending_image or {}
        has_image = bool(attached_image.get("bytes"))

        if has_image and not prompt:
            prompt = "Bu görsel hakkında bilgi verir misin?"

        if not prompt:
            st.warning("Lütfen bir soru yazın veya bir görsel ekleyin.")
            st.stop()

        try:
            rag = initialize_rag()
            vision_rag = initialize_vision_rag(rag)
        except Exception as e:
            st.error(f"RAG sistemi başlatılamadı: {e}")
            st.info("Lütfen MongoDB ve Ollama'nın çalıştığından emin olun.")
            st.stop()
        
        # Kullanıcı mesajını ekle
        message_data = {
            "role": "user",
            "content": prompt,
            "timestamp": _now_turkey()
        }
        
        if has_image:
            message_data.update({
                "has_image": True,
                "image_bytes": attached_image.get("bytes"),
                "image_name": attached_image.get("name", "Eklenen görsel"),
            })
        
        st.session_state.messages.append(message_data)
        
        # Kullanıcı mesajını göster
        format_chat_message(
            "user",
            prompt,
            timestamp=message_data["timestamp"],
            image_bytes=message_data.get("image_bytes"),
            image_name=message_data.get("image_name", ""),
        )

        if has_image and pending_image and attached_image == pending_image:
            st.session_state.pending_image = None
        
        # Cevap üret
        if has_image:
            # Vision RAG kullan
            with st.spinner("Görsel analiz ediliyor..."):
                try:
                    routed_image_result = rag.analyze_image_with_context(
                        image_bytes=attached_image.get("bytes"),
                        question=prompt,
                        category=selected_category,
                    )
                    if routed_image_result.get("delegate_to_clearpath_vision"):
                        result = vision_rag.analyze_image_with_context(
                            image_bytes=attached_image.get("bytes"),
                            question=prompt,
                            category=selected_category
                        )
                    else:
                        result = routed_image_result
                    
                    # Assistant mesajını ekle
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "similar_images": result.get("similar_images", []),
                        "timestamp": _now_turkey(),
                        "has_image": True
                    })
                    
                    # Assistant mesajını göster
                    format_chat_message(
                        "assistant",
                        result["answer"],
                        result.get("sources", []),
                        result.get("similar_images", []),
                        st.session_state.messages[-1]["timestamp"]
                    )
                    
                    # Hata varsa uyar
                    if result.get("error"):
                        st.warning("Görsel analizi sırasında bir sorun oluştu.")
                
                except Exception as e:
                    error_msg = f"Görsel analizi hatası: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": _now_turkey()
                    })
        else:
            # Normal text RAG kullan
            with st.spinner("🤖 Cevap üretiliyor..."):
                try:
                    result = rag.generate_answer(prompt, category=selected_category)
                    
                    # Text-to-image search de yap (CLIP ile)
                    similar_images = []
                    if result.get("sources") and result.get("route") != "spot":
                        try:
                            similar_images = vision_rag.search_similar_images(
                                query_text=prompt,
                                category=selected_category,
                                limit=3
                            )
                        except:
                            pass  # Görsel arama başarısız olsa bile text cevabı göster
                    
                    # Assistant mesajını ekle
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "similar_images": similar_images,
                        "timestamp": _now_turkey()
                    })
                    
                    # Assistant mesajını göster
                    format_chat_message(
                        "assistant",
                        result["answer"],
                        result.get("sources", []),
                        similar_images,
                        st.session_state.messages[-1]["timestamp"]
                    )
                    
                    # Hata varsa uyar
                    if result.get("error"):
                        st.warning("⚠️ Cevap üretilirken bir sorun oluştu.")
                
                except Exception as e:
                    error_msg = f"Bir hata oluştu: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": _now_turkey()
                    })

    st.markdown(
        '<div class="ragora-footnote">Yanıtlar, seçili belgelerden alınan bilgilere dayanmaktadır.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
