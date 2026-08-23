"""
Retrieval Yöntemi Karşılaştırma Tablosu
Her yöntem ayrı ayrı test edilir → tek tablo görseli üretilir.

Yöntemler:
  1. Semantic Search  — sadece Qdrant vektör araması
  2. Keyword Search   — sadece MongoDB BM25 araması
  3. Hybrid RRF       — semantic + keyword birleşimi
  4. CLIP (text→img)  — metin sorusu → görsel benzerlik

Kullanım:
    python compare_methods_table.py
    python compare_methods_table.py --demo
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

K = 5   # sabit K değeri

# ─── Ground truth (evaluate_retrieval.py ile aynı) ───────────────────────────
from evaluate_retrieval import EVAL_DATASET, is_relevant


# ─── Sonuç veri yapısı ───────────────────────────────────────────────────────
@dataclass
class MethodResult:
    name: str
    count: int
    hit_at_k: float
    mrr_at_k: float
    precision_at_k: float = 0.0


# ─── Her yöntem için retrieval çalıştır ──────────────────────────────────────

def run_semantic(retriever, k=K):
    """Sadece Qdrant vektör araması."""
    retriever._text_retriever.use_hybrid = False
    hits, mrrs, precs = [], [], []
    for gt in EVAL_DATASET:
        result = retriever.retrieve_by_text(query=gt["query"], auto_lang=False)
        chunks = sorted(result.text_chunks, key=lambda c: c.score, reverse=True)
        top = chunks[:k]
        hit  = int(any(is_relevant(c, gt) for c in top))
        mrr  = next((1/(i+1) for i, c in enumerate(top) if is_relevant(c, gt)), 0.0)
        prec = sum(is_relevant(c, gt) for c in top) / max(len(top), 1)
        hits.append(hit); mrrs.append(mrr); precs.append(prec)
    retriever._text_retriever.use_hybrid = True
    n = len(EVAL_DATASET)
    return MethodResult("Semantic Search\n(Qdrant)", n, sum(hits)/n, sum(mrrs)/n, sum(precs)/n)


def run_keyword(retriever, k=K):
    """Sadece MongoDB BM25 full-text araması."""
    mongo = retriever._text_retriever.mongo
    hits, mrrs, precs = [], [], []
    for gt in EVAL_DATASET:
        docs = mongo.full_text_search(gt["query"], limit=k*2)

        class FakeChunk:
            def __init__(self, d):
                self.raw_text   = d.get("raw_text", "")
                self.doc_name   = d.get("doc_name", "")
                self.chunk_id   = d.get("chunk_id", "")
                self.score      = 1.0
                self.breadcrumb = d.get("breadcrumb", [])
                self.page_range = d.get("page_range", [0,0])
                self.chunk_type = d.get("chunk_type", "text")
                self.lang       = d.get("lang", "en")

        top  = [FakeChunk(d) for d in docs][:k]
        hit  = int(any(is_relevant(c, gt) for c in top))
        mrr  = next((1/(i+1) for i, c in enumerate(top) if is_relevant(c, gt)), 0.0)
        prec = sum(is_relevant(c, gt) for c in top) / max(len(top), 1)
        hits.append(hit); mrrs.append(mrr); precs.append(prec)
    n = len(EVAL_DATASET)
    return MethodResult("Keyword Search\n(MongoDB)", n, sum(hits)/n, sum(mrrs)/n, sum(precs)/n)


def run_hybrid(retriever, k=K):
    """Semantic + Keyword RRF birleşimi."""
    retriever._text_retriever.use_hybrid = True
    hits, mrrs, precs = [], [], []
    for gt in EVAL_DATASET:
        result = retriever.retrieve_by_text(query=gt["query"], auto_lang=False)
        chunks = sorted(result.text_chunks, key=lambda c: c.score, reverse=True)
        top  = chunks[:k]
        hit  = int(any(is_relevant(c, gt) for c in top))
        mrr  = next((1/(i+1) for i, c in enumerate(top) if is_relevant(c, gt)), 0.0)
        prec = sum(is_relevant(c, gt) for c in top) / max(len(top), 1)
        hits.append(hit); mrrs.append(mrr); precs.append(prec)
    n = len(EVAL_DATASET)
    return MethodResult("Hybrid RRF\n(Semantic + BM25)", n, sum(hits)/n, sum(mrrs)/n, sum(precs)/n)


def run_clip_image2image(retriever, k=K, sample_per_doc=5):
    """
    CLIP image→image: Her görsel kendi embedding'iyle aranır.

    Ground truth: aynı dok ümanın görseli = relevant.
    Her dokümanından sample_per_doc adet görsel örneklenir.
    Sorgu görseli sonuçlardan çıkarılır (self-exclusion).
    """
    clip       = retriever._clip
    vis_qdrant = retriever._vis_qdrant
    vis_mongo  = retriever._vis_mongo

    # Her dokümanından eşit sayıda görsel örnekle
    samples = []
    for doc_name in vis_mongo._col.distinct("doc_name"):
        docs = list(vis_mongo._col.find(
            {"doc_name": doc_name},
            {"element_id": 1, "image_b64": 1, "doc_name": 1},
        ).limit(sample_per_doc))
        samples.extend(docs)

    hits, mrrs, precs = [], [], []
    for doc in samples:
        # Görseli CLIP ile embed et
        img_vec  = clip.embed_image_b64(doc["image_b64"])

        # Benzer görselleri ara (k+1: kendisi de dönebilir)
        raw = vis_qdrant.search(query_vector=img_vec, top_k=k + 1, filter_docs=None)

        # Kendini çıkar
        raw = [r for r in raw if r["element_id"] != doc["element_id"]][:k]

        if not raw:
            hits.append(0); mrrs.append(0.0); precs.append(0.0); continue

        # Dönen görsellerin metadata'sını çek
        elem_ids  = [r["element_id"] for r in raw]
        res_docs  = vis_mongo.get_by_ids(elem_ids)
        res_map   = {d["element_id"]: d["doc_name"] for d in res_docs}

        query_doc = doc["doc_name"]

        # Skor sırasına göre sırala, aynı doküman = relevant
        ordered = sorted(raw, key=lambda r: r["score"], reverse=True)
        hit = int(any(res_map.get(r["element_id"]) == query_doc for r in ordered))
        mrr = next(
            (1 / (i + 1) for i, r in enumerate(ordered)
             if res_map.get(r["element_id"]) == query_doc),
            0.0,
        )
        prec = sum(
            1 for r in ordered if res_map.get(r["element_id"]) == query_doc
        ) / max(len(ordered), 1)
        hits.append(hit); mrrs.append(mrr); precs.append(prec)

    n = len(samples)
    return MethodResult("CLIP\n(image→image)", n, sum(hits)/n, sum(mrrs)/n, sum(precs)/n)


# ─── Demo verisi ─────────────────────────────────────────────────────────────
DEMO_RESULTS = [
    MethodResult("Semantic Search\n(Qdrant)",         15, 1.00, 0.938, 0.720),
    MethodResult("Keyword Search\n(MongoDB)",          15, 0.93, 0.849, 0.613),
    MethodResult("Hybrid RRF\n(Semantic + BM25)",      15, 1.00, 0.872, 0.680),
    MethodResult("CLIP\n(image→image)",                29, 0.69, 0.465, 0.380),
]


# ─── Tablo Görseli ───────────────────────────────────────────────────────────

def draw_table(results: list[MethodResult], demo: bool):
    n_rows = len(results)

    fig_h = 1.4 + n_rows * 1.0
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Sütun başlıkları
    col_labels  = ["Method", "Count", f"Hit Rate@{K}", f"Precision@{K}", f"MRR@{K}"]
    col_widths  = [0.32, 0.10, 0.19, 0.19, 0.20]
    col_aligns  = ["left", "center", "center", "center", "center"]

    # Renkler
    HEADER_BG   = "#1E3A5F"
    HEADER_FG   = "white"
    ROW_BG_A    = "#F0F4FA"
    ROW_BG_B    = "white"
    BORDER      = "#C5CDD8"
    TEXT_MAIN   = "#111827"
    GREEN       = "#15803D"
    ORANGE      = "#B45309"

    total_w = sum(col_widths)
    x_starts = [sum(col_widths[:i]) / total_w for i in range(len(col_widths))]
    widths_n = [w / total_w for w in col_widths]

    row_h    = 1.0 / (n_rows + 1.5)
    header_y = 1 - row_h * 0.9

    # ── Başlık satırı ────────────────────────────────────────────────────────
    for j, (label, xs, wd, align) in enumerate(zip(col_labels, x_starts, widths_n, col_aligns)):
        rect = mpatches.FancyBboxPatch(
            (xs, header_y), wd, row_h,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG, edgecolor=BORDER, linewidth=0.8,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ha = "left" if align == "left" else "center"
        ox = 0.012 if align == "left" else 0
        ax.text(
            xs + wd * 0.5 + ox, header_y + row_h * 0.5,
            label, ha=ha, va="center",
            fontsize=11, fontweight="bold", color=HEADER_FG,
            transform=ax.transAxes, clip_on=False,
        )

    # ── Veri satırları ───────────────────────────────────────────────────────
    for i, row in enumerate(results):
        y = header_y - (i + 1) * row_h
        bg = ROW_BG_A if i % 2 == 0 else ROW_BG_B

        values = [row.name, str(row.count),
                  f"{row.hit_at_k:.2f}", f"{row.precision_at_k:.3f}", f"{row.mrr_at_k:.3f}"]

        for j, (val, xs, wd, align) in enumerate(zip(values, x_starts, widths_n, col_aligns)):
            rect = mpatches.FancyBboxPatch(
                (xs, y), wd, row_h,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor=BORDER, linewidth=0.8,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(rect)

            # Metrik sütunlarında renk kodlaması (j=2,3,4 → Hit, Precision, MRR)
            if j == 0:
                color = TEXT_MAIN; fs = 10; fw = "bold"
            elif j in (2, 3, 4):
                v = float(val)
                color = GREEN if v >= 0.85 else (ORANGE if v >= 0.60 else "#B91C1C")
                fs = 11; fw = "bold"
            else:
                color = TEXT_MAIN; fs = 10.5; fw = "normal"

            ha = "left" if align == "left" else "center"
            ox = 0.012 if align == "left" else 0
            ax.text(
                xs + wd * 0.5 + ox, y + row_h * 0.5,
                val, ha=ha, va="center",
                fontsize=fs, fontweight=fw, color=color,
                transform=ax.transAxes, clip_on=False,
                multialignment="center",
            )

    # Dış çerçeve
    outer = mpatches.FancyBboxPatch(
        (0, header_y - n_rows * row_h), 1.0, row_h * (n_rows + 1),
        boxstyle="square,pad=0",
        facecolor="none", edgecolor="#8899AA", linewidth=1.5,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(outer)

    suffix = " (Demo)" if demo else ""
    fig.suptitle(
        f"Retrieval Yöntemi Karşılaştırması — Hit Rate, Precision ve MRR  (K={K}){suffix}",
        fontsize=12, fontweight="bold", color="#1E3A5F", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(__file__).parent / "method_comparison_table.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Kaydedildi: {out.name}")
    return out


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Retriever olmadan demo verisiyle çalış")
    args = parser.parse_args()

    if args.demo:
        print("  [DEMO] Örnek verilerle tablo üretiliyor...\n")
        results = DEMO_RESULTS
    else:
        print("  Retriever yükleniyor...", end=" ", flush=True)
        from pipeline.visual_retriever import VisualRetriever
        ret = VisualRetriever(top_k_text=20, top_k_visual=3)
        ret._text_retriever.score_threshold = 0.05
        print("hazır.\n")

        print(f"  [1/4] Semantic Search değerlendiriliyor ({len(EVAL_DATASET)} sorgu)...")
        r1 = run_semantic(ret)
        print(f"        Hit@{K}={r1.hit_at_k:.3f}  Prec@{K}={r1.precision_at_k:.3f}  MRR@{K}={r1.mrr_at_k:.3f}")

        print(f"  [2/4] Keyword Search değerlendiriliyor...")
        r2 = run_keyword(ret)
        print(f"        Hit@{K}={r2.hit_at_k:.3f}  Prec@{K}={r2.precision_at_k:.3f}  MRR@{K}={r2.mrr_at_k:.3f}")

        print(f"  [3/4] Hybrid RRF değerlendiriliyor...")
        r3 = run_hybrid(ret)
        print(f"        Hit@{K}={r3.hit_at_k:.3f}  Prec@{K}={r3.precision_at_k:.3f}  MRR@{K}={r3.mrr_at_k:.3f}")

        print(f"  [4/4] CLIP image→image değerlendiriliyor (6 dok × 5 görsel = 30 sorgu)...")
        r4 = run_clip_image2image(ret, sample_per_doc=5)
        print(f"        Hit@{K}={r4.hit_at_k:.3f}  Prec@{K}={r4.precision_at_k:.3f}  MRR@{K}={r4.mrr_at_k:.3f}")

        results = [r1, r2, r3, r4]

    print("\n  Tablo görseli oluşturuluyor...")
    draw_table(results, demo=args.demo)
    print("  Bitti!")
