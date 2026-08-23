"""
qwen3-vl:8b  vs  qwen2.5vl:7b  — Görsel açıklama kalitesi karşılaştırması

Spot PDF'lerinden tipik sayfalar rasterize edilir ve her iki VLM'e aynı
prompt gönderilerek açıklama kalitesi karşılaştırılır.

Kullanım: python compare_vl_models.py
"""
import sys
import base64
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF

from config.settings import settings

MODELS = ["qwen3-vl:8b", "qwen2.5vl:7b"]

PDF_BASE = Path("data/pdfs")

# (PDF dosyası, sayfa no, açıklama — ne içerdiği)
TEST_PAGES = [
    ("spot-user-manual-en.pdf",        18,  "Teknik özellik tablosu"),
    ("spot-user-manual-en.pdf",        43,  "Kaldırma / taşıma prosedürü (diyagram)"),
    ("spot-arm-user-manual-en.pdf",     9,  "Spot Arm teknik özellikler"),
    ("spot-user-manual-en.pdf",        19,  "E-Stop / güvenlik uyarısı sayfası"),
    ("spot-power-supply-user-manual-en.pdf", 5, "Şarj ünitesi bağlantı diyagramı"),
]

VLM_PROMPT = (
    "You are analyzing a page from a Boston Dynamics Spot technical manual. "
    "Describe everything visible on this page in detail:\n"
    "1. List ALL text visible (headings, body text, captions, labels, warnings)\n"
    "2. Describe any diagrams, figures, or photos (what is shown, labeled parts, arrows, positions)\n"
    "3. Describe any tables (column headers, row values)\n"
    "4. Note any safety warning boxes (DANGER/WARNING/CAUTION) with their full text\n"
    "Be exhaustive — include every piece of information visible."
)


def rasterize_page(pdf_path: Path, page_num: int, dpi: int = 200) -> str:
    """PDF sayfasını base64 JPEG'e dönüştürür."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    return base64.b64encode(img_bytes).decode()


def ask_vl_model(model: str, image_b64: str, prompt: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 1500,
            "think": False,
            "stop": ["<|endoftext|>", "<|im_end|>"],
        },
    }
    t0 = time.time()
    r = requests.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - t0
    answer = r.json().get("message", {}).get("content", "").strip()
    return answer, elapsed


def count_words(text: str) -> int:
    return len(text.split())


def sep(char="─", n=80):
    print(char * n)


def print_wrapped(text: str, width: int = 100, indent: str = "  "):
    for line in text.split("\n"):
        while len(line) > width:
            print(indent + line[:width])
            line = "    " + line[width:]
        print(indent + line)


def main():
    print("\n" + "="*80)
    print("  SPOT RAG — VL MODEL KARŞILAŞTIRMA: qwen3-vl:8b  vs  qwen2.5vl:7b")
    print("="*80 + "\n")

    summary = []

    for idx, (pdf_name, page_no, desc) in enumerate(TEST_PAGES, 1):
        pdf_path = PDF_BASE / pdf_name
        if not pdf_path.exists():
            print(f"[{idx}] PDF bulunamadı: {pdf_path}, atlanıyor.\n")
            continue

        sep("═")
        print(f"\n[{idx}/{len(TEST_PAGES)}] {pdf_name}  —  Sayfa {page_no}")
        print(f"       Sayfa türü: {desc}\n")

        print("  Sayfa rasterize ediliyor (200 DPI)...", end=" ", flush=True)
        image_b64 = rasterize_page(pdf_path, page_no)
        print(f"hazır ({len(image_b64)//1024} KB)")
        print()

        model_results = {}
        for model in MODELS:
            print(f"  [{model}] açıklama üretiliyor...", end=" ", flush=True)
            try:
                answer, elapsed = ask_vl_model(model, image_b64, VLM_PROMPT)
                model_results[model] = (answer, elapsed)
                print(f"{elapsed:.1f}s — {count_words(answer)} kelime")
            except Exception as e:
                model_results[model] = (f"HATA: {e}", 0)
                print(f"HATA: {e}")

        print()
        for model in MODELS:
            answer, elapsed = model_results[model]
            sep("─")
            print(f"  ▶ {model}  ({elapsed:.1f}s, {count_words(answer)} kelime)")
            sep("─")
            print_wrapped(answer)
            print()

        summary.append({
            "page": f"{pdf_name[:20]}..p{page_no}",
            "desc": desc,
            "times": {m: model_results[m][1] for m in MODELS},
            "words": {m: count_words(model_results[m][0]) for m in MODELS},
        })

    # Özet
    sep("═")
    print("\n  ÖZET")
    sep("═")
    print(f"{'Sayfa':<32} {'8b süre':>9} {'7b süre':>9} {'8b kelime':>10} {'7b kelime':>10}")
    sep()
    for r in summary:
        t8  = r["times"].get("qwen3-vl:8b",   0)
        t7  = r["times"].get("qwen2.5vl:7b",  0)
        w8  = r["words"].get("qwen3-vl:8b",   0)
        w7  = r["words"].get("qwen2.5vl:7b",  0)
        faster = "✓8b" if t8 < t7 else "✓7b"
        richer = "✓8b" if w8 > w7 else "✓7b"
        print(f"{r['desc'][:30]:<32} {t8:>8.1f}s {t7:>8.1f}s {w8:>10} {w7:>10}  hız:{faster} detay:{richer}")

    sep()
    avg8  = sum(r["times"].get("qwen3-vl:8b",  0) for r in summary) / max(len(summary), 1)
    avg7  = sum(r["times"].get("qwen2.5vl:7b", 0) for r in summary) / max(len(summary), 1)
    avgw8 = sum(r["words"].get("qwen3-vl:8b",  0) for r in summary) / max(len(summary), 1)
    avgw7 = sum(r["words"].get("qwen2.5vl:7b", 0) for r in summary) / max(len(summary), 1)
    print(f"{'ORTALAMA':<32} {avg8:>8.1f}s {avg7:>8.1f}s {avgw8:>10.0f} {avgw7:>10.0f}")
    print()


if __name__ == "__main__":
    main()
