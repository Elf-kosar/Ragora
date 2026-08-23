"""
qwen3.5:9b  vs  qwen3.5:27b  — RAG yanıt kalitesi karşılaştırması
Kullanım: python compare_models.py
"""
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from pipeline.visual_retriever import VisualRetriever

MODELS = ["qwen3.5:9b", "qwen3.5:27b"]

TEST_QUERIES = [
    "How do I safely lift and carry Spot?",
    "What is Spot's maximum payload capacity?",
    "How does the E-Stop work and when should I use it?",
    "What are the battery charging safety precautions?",
    "How do I attach the Spot Arm to the robot?",
]

SYSTEM_PROMPT = """You are a Boston Dynamics Spot manual assistant. Answer using ONLY the provided context.
Report exactly what the manual says — include all relevant details and descriptions.
Format your answer with clear paragraphs. For step-by-step procedures, use numbered steps.
Place citations at the END of each sentence: e.g. (Spot User Manual, p.12).
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


def sanitize(text: str) -> str:
    for tok in _SPECIAL:
        text = text.replace(tok, "")
    return text


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
        pages = (f"p.{chunk.page_range[0]}-{chunk.page_range[1]}"
                 if chunk.page_range[0] != chunk.page_range[1]
                 else f"p.{chunk.page_range[0]}")
        doc_label = _DOC_LABELS.get(chunk.doc_name, chunk.doc_name)
        header = f"[{doc_label}, {pages}] {section}"
        if chunk.chunk_type == "table":
            header += " [TABLE]"
        parts.append(f"{header}\n{sanitize(chunk.raw_text)}")

    return "\n\n---\n\n".join(parts)


def ask_model(model: str, context: str, query: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"},
        {"role": "assistant", "content": "<think>\n\n</think>\n"},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "num_predict": 1024,
            "stop": ["<|endoftext|>", "<|im_end|>"],
        },
    }
    t0 = time.time()
    r = requests.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - t0
    answer = r.json().get("message", {}).get("content", "").strip()
    return answer, elapsed


def sep(char="─", n=80):
    print(char * n)


def main():
    print("\n" + "="*80)
    print("  SPOT RAG — MODEL KARŞILAŞTIRMA: qwen3.5:9b  vs  qwen3.5:27b")
    print("="*80)

    print("\nRetriever yükleniyor (mE5 + CLIP)...")
    retriever = VisualRetriever(top_k_text=12, top_k_visual=1)
    retriever._text_retriever.score_threshold = 0.10
    print("Hazır.\n")

    results_summary = []

    for qi, query in enumerate(TEST_QUERIES, 1):
        sep("═")
        print(f"\n[{qi}/{len(TEST_QUERIES)}] SORGU: {query}\n")

        # Retrieval
        result = retriever.retrieve_by_text(query=query, auto_lang=False)
        context = build_context(result)

        chunk_info = [
            f"  - {_DOC_LABELS.get(c.doc_name, c.doc_name)}, s.{c.page_range[0]}, skor={c.score:.3f}"
            for c in sorted(result.text_chunks, key=lambda c: c.score, reverse=True)
            if c.score >= 0.5
        ][:5]
        print("Getirilen chunk'lar:")
        print("\n".join(chunk_info) if chunk_info else "  (yok)")
        print()

        model_answers = {}
        for model in MODELS:
            print(f"  [{model}] yanıt üretiliyor...", end=" ", flush=True)
            try:
                answer, elapsed = ask_model(model, context, query)
                model_answers[model] = (answer, elapsed)
                print(f"{elapsed:.1f}s")
            except Exception as e:
                model_answers[model] = (f"HATA: {e}", 0)
                print("HATA")

        # Yanıtları göster
        print()
        for model in MODELS:
            answer, elapsed = model_answers[model]
            sep("─")
            print(f"  ▶ {model}  ({elapsed:.1f}s)")
            sep("─")
            # Satır uzunluğunu sınırla
            for line in answer.split("\n"):
                while len(line) > 100:
                    print(line[:100])
                    line = "    " + line[100:]
                print(line)
            print()

        results_summary.append({
            "query": query,
            "chunks": len([c for c in result.text_chunks if c.score >= 0.5]),
            "times": {m: model_answers[m][1] for m in MODELS},
            "lens": {m: len(model_answers[m][0]) for m in MODELS},
        })

    # Özet tablo
    sep("═")
    print("\n  ÖZET")
    sep("═")
    print(f"{'Sorgu':<45} {'9b süresi':>10} {'27b süresi':>11} {'9b uzunluk':>12} {'27b uzunluk':>12}")
    sep()
    for r in results_summary:
        q = r["query"][:43] + ".." if len(r["query"]) > 43 else r["query"]
        t9  = r["times"].get("qwen3.5:9b", 0)
        t27 = r["times"].get("qwen3.5:27b", 0)
        l9  = r["lens"].get("qwen3.5:9b", 0)
        l27 = r["lens"].get("qwen3.5:27b", 0)
        print(f"{q:<45} {t9:>9.1f}s {t27:>10.1f}s {l9:>12} {l27:>12}")
    sep()
    avg9  = sum(r["times"].get("qwen3.5:9b",  0) for r in results_summary) / len(results_summary)
    avg27 = sum(r["times"].get("qwen3.5:27b", 0) for r in results_summary) / len(results_summary)
    print(f"{'ORTALAMA':<45} {avg9:>9.1f}s {avg27:>10.1f}s")
    print()


if __name__ == "__main__":
    main()
