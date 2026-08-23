"""
Dil yardımcıları — dosya adından ve içerikten dil tespiti.

İki yaklaşım
------------
1. Dosya adından: "spot-de.pdf" → "de", "spot-arm-ko.pdf" → "ko"
2. İçerikten: langdetect ile chunk metninden tespit

Desteklenen dil kodları (ISO 639-1)
------------------------------------
en, de, ko, ar, es, fr, ja, pt, tr
"""
from __future__ import annotations

import re
from pathlib import Path

LANG_NAMES = {
    "en": "English",
    "de": "Deutsch",
    "ko": "한국어",
    "ar": "العربية",
    "es": "Español",
    "fr": "Français",
    "ja": "日本語",
    "pt": "Português",
    "tr": "Türkçe",
}

_SUFFIX_PATTERN = re.compile(
    r'[-_](' + '|'.join(LANG_NAMES.keys()) + r')(?:[-_.]|$)',
    re.IGNORECASE,
)


def detect_lang_from_filename(pdf_path: str | Path) -> str:
    """
    Dosya adından dil kodunu çıkarır.
    spot-de.pdf → "de", spot-arm-ko.pdf → "ko"
    """
    name = Path(pdf_path).stem.lower()
    m = _SUFFIX_PATTERN.search(name)
    if m:
        return m.group(1).lower()
    if "gebrauchsanweisung" in name or "anweisung" in name:
        return "de"
    if "instrucciones" in name or "usuario" in name:
        return "es"
    return "en"


def detect_lang_from_text(text: str) -> str:
    """Metin içeriğinden dil tespit eder."""
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "en"


def detect_lang(pdf_path: str | Path, sample_text: str = "") -> str:
    """Önce dosya adından, başarısız olursa metinden dil tespit eder."""
    lang = detect_lang_from_filename(pdf_path)
    if lang != "en" or not sample_text:
        return lang
    text_lang = detect_lang_from_text(sample_text)
    return text_lang if text_lang in LANG_NAMES else lang


def lang_display(lang_code: str) -> str:
    name = LANG_NAMES.get(lang_code, lang_code)
    return f"{name} ({lang_code})"
