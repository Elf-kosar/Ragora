"""
qwen3.5:27b — Yanıt süresi + Hallucination testi
Kullanım: python test_27b.py
"""
import sys, time, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from pipeline.visual_retriever import VisualRetriever

MODEL = "qwen3.5:27b"

SYSTEM_PROMPT = """You are a Boston Dynamics Spot manual assistant. Answer using ONLY the provided context.
Report exactly what the manual says. Place citations at the END of each sentence: e.g. (Spot User Manual, p.12).
If not found in context, respond with ONLY: "This information is not available in the Spot User Manual." """

_DOC_LABELS = {
    "spot-user-manual-en": "Spot User Manual",
    "spot-arm-user-manual-en": "Spot Arm Manual",
    "spot-dock-user-manual-en": "Spot Dock Manual",
    "spot-station-user-manual-en": "Spot Station Manual",
    "spot-cam-2-user-manual-en": "Spot Cam 2 Manual",
    "spot-power-supply-user-manual-en": "Spot Power Supply Manual",
}
_BOILERPLATE = {"REQUIRED READING", "NOTICE", "WARNING", "CAUTION", "DANGER"}
_SPECIAL = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>")

# ── Test sorguları ─────────────────────────────────────────────────────────────

# Gerçek yanıt süresi testi — kılavuzda olan sorular
TIMING_QUERIES = [
    ("Kısa",   "What is Spot's maximum payload?"),
    ("Orta",   "How do I power on and power off Spot?"),
    ("Uzun",   "What are all the safety precautions I need to follow before operating Spot?"),
    ("Türkçe", "Spot'un pil ömrü ne kadar ve şarj süresi kaçtır?"),
    ("Tablo",  "What are the technical specifications of the Spot Arm?"),
]

# Hallucination testi — kılavuzda OLMAYAN / yanıltıcı sorular
HALLUCINATION_QUERIES = [
    # Kılavuzda kesinlikle olmayan bilgiler
    ("Fiyat",       "How much does Spot cost to purchase?"),
    ("Rakip",       "How does Spot compare to Boston Dynamics Atlas robot?"),
    ("Yazılım",     "What programming languages can I use with the Spot SDK?"),
    ("Garanti",     "What is the warranty period for Spot?"),
    # Yanlış sayı içeren tuzak sorular
    ("Tuzak-1",    "Is it true that Spot can carry up to 25 kg of payload?"),
    ("Tuzak-2",    "Can Spot operate for 6 hours on a single battery charge?"),
    ("Tuzak-3",    "What is the maximum speed of 5 m/s for Spot?"),
    # Yarı-doğru / belirsiz sorular
    ("Belirsiz-1", "Does Spot have a built-in camera?"),
    ("Belirsiz-2", "Can Spot climb stairs?"),
    ("Belirsiz-3", "What happens if Spot falls over?"),
]


def build_context(result) -> str:
    chunks = sorted(
        [c for c in result.text_chunks
         if c.score >= 0.5
         and (c.breadcrumb[0].strip().upper() not in _BOILERPLATE if c.breadcrumb else True)],
        key=lambda c: c.score, reverse=True
    )[:8]
    parts = ["Context from Spot manuals:\n"]
    for chunk in chunks:
        section = " > ".join(chunk.breadcrumb) if chunk.breadcrumb else "General"
        pages   = (f"p.{chunk.page_range[0]}-{chunk.page_range[1]}"
                   if chunk.page_range[0] != chunk.page_range[1]
                   else f"p.{chunk.page_range[0]}")
        label  = _DOC_LABELS.get(chunk.doc_name, chunk.doc_name)
        header = f"[{label}, {pages}] {section}"
        if chunk.chunk_type == "table": header += " [TABLE]"
        text = chunk.raw_text
        for t in _SPECIAL: text = text.replace(t, "")
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


def ask(context: str, query: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"},
        {"role": "assistant", "content": "<think>\n\n</think>\n"},
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 1024,
                    "stop": ["<|endoftext|>", "<|im_end|>"]},
    }
    t0 = time.time()
    r  = requests.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip(), time.time() - t0


def sep(c="─", n=80): print(c * n)

def wrap(text, w=95):
    for line in text.split("\n"):
        while len(line) > w:
            print("  " + line[:w]); line = "    " + line[w:]
        print("  " + line)


def main():
    print("\n" + "="*80)
    print(f"  {MODEL} — YANIT SÜRESİ + HALLUCİNATION TESTİ")
    print("="*80)

    print("\nRetriever yükleniyor...", end=" ", flush=True)
    ret = VisualRetriever(top_k_text=12, top_k_visual=1)
    ret._text_retriever.score_threshold = 0.10
    print("hazır.\n")

    # ── 1. YANIT SÜRESİ ────────────────────────────────────────────────────────
    sep("═")
    print("  BÖLÜM 1 — YANIT SÜRESİ (kılavuzda olan sorular)\n")

    timing_rows = []
    for tag, query in TIMING_QUERIES:
        result  = ret.retrieve_by_text(query=query, auto_lang=False)
        context = build_context(result)
        n_chunks = len([c for c in result.text_chunks if c.score >= 0.5])

        print(f"  [{tag}] {query}")
        print(f"         {n_chunks} chunk getirildi — yanıt bekleniyor...", end=" ", flush=True)
        answer, elapsed = ask(context, query)
        words = len(answer.split())
        print(f"{elapsed:.1f}s  ({words} kelime)")
        timing_rows.append((tag, query, elapsed, words, answer))

    print()
    sep()
    print(f"  {'Kategori':<12} {'Süre':>8}  {'Kelime':>8}  Sorgu")
    sep()
    for tag, q, t, w, _ in timing_rows:
        print(f"  {tag:<12} {t:>7.1f}s  {w:>8}  {q[:55]}")
    avg_t = sum(r[2] for r in timing_rows) / len(timing_rows)
    avg_w = sum(r[3] for r in timing_rows) / len(timing_rows)
    sep()
    print(f"  {'ORTALAMA':<12} {avg_t:>7.1f}s  {avg_w:>8.0f}")

    # ── 2. HALLUCINATION TESTİ ─────────────────────────────────────────────────
    print()
    sep("═")
    print("  BÖLÜM 2 — HALLUCİNATION TESTİ (tuzak sorular)\n")
    print("  Beklenen: 'not available' veya doğru sayı düzeltmesi")
    print("  Kötü:     uydurulmuş fiyat/garanti/karşılaştırma bilgisi\n")

    hallu_rows = []
    for tag, query in HALLUCINATION_QUERIES:
        result  = ret.retrieve_by_text(query=query, auto_lang=False)
        context = build_context(result)

        print(f"  [{tag}] {query}")
        print(f"         yanıt bekleniyor...", end=" ", flush=True)
        answer, elapsed = ask(context, query)
        words = len(answer.split())

        not_avail = "not available" in answer.lower() or "bulunamadı" in answer.lower()
        corrects_number = any(x in answer.lower() for x in ["14 kg", "not 25", "incorrect", "not specified", "does not"])
        status = "✅ REDDETTİ" if not_avail else ("⚠️  KISMI" if corrects_number else "❌ UYDU")

        print(f"{elapsed:.1f}s  {status}")
        hallu_rows.append((tag, query, elapsed, status, answer))

    print()
    sep("═")
    print("  DETAYLI YANIT ÇIKTISI\n")
    for tag, query, elapsed, status, answer in hallu_rows:
        sep("─")
        print(f"  [{tag}] {status}  —  {query}")
        sep("─")
        wrap(answer)
        print()

    # ── Özet ──────────────────────────────────────────────────────────────────
    sep("═")
    print("\n  ÖZET — HALLUCİNATION SKORU\n")
    reddetti = sum(1 for r in hallu_rows if "✅" in r[3])
    kismi    = sum(1 for r in hallu_rows if "⚠️" in r[3])
    uydu     = sum(1 for r in hallu_rows if "❌" in r[3])
    total    = len(hallu_rows)
    print(f"  ✅ Doğru reddetti : {reddetti}/{total}")
    print(f"  ⚠️  Kısmi/belirsiz : {kismi}/{total}")
    print(f"  ❌ Hallucination  : {uydu}/{total}")
    print()


if __name__ == "__main__":
    main()
