"""
RAG Retrieval Metriklerini Görselleştir — 4 Ayrı Akademik Grafik

Kullanım:
    python plot_metrics.py          # gerçek retrieval
    python plot_metrics.py --demo   # demo verisiyle

Çıktı:
    fig1_metrics_by_k.png    — Metrikler × K çizgi grafiği
    fig2_bar_at_k5.png       — K=5 için metrik bar grafiği
    fig3_per_query_mrr.png   — Sorgu başına MRR@5
    fig4_heatmap.png         — Hit Rate ısı haritası
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

K_VALUES   = [1, 3, 5, 10]
OUT_DIR    = Path(__file__).parent

# ─── Renk paleti (IEEE/akademik uyumlu, renk körü dostu) ────────────────────
C = {
    "mrr":       "#1A56A6",
    "recall":    "#1B7F4D",
    "precision": "#B91C1C",
    "hit":       "#6D28D9",
}
LABELS = {
    "mrr":       "MRR@K",
    "recall":    "Recall@K",
    "precision": "Precision@K",
    "hit":       "Hit Rate@K",
}

# ─── Demo verisi ──────────────────────────────────────────────────────────────
DEMO_RESULTS = [
    {"note": "Teknik özellik",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.33,"recall@3":0.67,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.6,"precision@10":0.3,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Hız spesifikasyonu",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.5,"recall@3":1.0,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.4,"precision@10":0.2,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Batarya ömrü",
     "mrr@1":0.0,"mrr@3":0.5,"mrr@5":0.5,"mrr@10":0.5,
     "recall@1":0.0,"recall@3":0.5,"recall@5":0.5,"recall@10":1.0,
     "precision@1":0.0,"precision@3":0.33,"precision@5":0.2,"precision@10":0.2,
     "hit@1":0.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Acil durdurma (E-Stop)",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.25,"recall@3":0.75,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":1.0,"precision@5":0.8,"precision@10":0.4,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Güvenlik önlemleri",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.2,"recall@3":0.6,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":1.0,"precision@10":0.5,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Devrilme kurtarma",
     "mrr@1":0.0,"mrr@3":0.33,"mrr@5":0.33,"mrr@10":0.33,
     "recall@1":0.0,"recall@3":0.5,"recall@5":0.5,"recall@10":1.0,
     "precision@1":0.0,"precision@3":0.33,"precision@5":0.2,"precision@10":0.2,
     "hit@1":0.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Şarj güvenliği",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.5,"recall@3":1.0,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.4,"precision@10":0.2,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Batarya takma/çıkarma",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.33,"recall@3":0.67,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.6,"precision@10":0.3,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Güç açma/kapama",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.25,"recall@3":0.75,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.6,"precision@10":0.3,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Motor aktivasyonu",
     "mrr@1":0.0,"mrr@3":0.33,"mrr@5":0.33,"mrr@10":0.5,
     "recall@1":0.0,"recall@3":0.5,"recall@5":0.5,"recall@10":1.0,
     "precision@1":0.0,"precision@3":0.33,"precision@5":0.2,"precision@10":0.2,
     "hit@1":0.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Arm takma prosedürü",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.5,"recall@3":1.0,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.4,"precision@10":0.2,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Arm yük kapasitesi",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.33,"recall@3":0.67,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":1.0,"precision@5":0.8,"precision@10":0.4,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Otonom docking",
     "mrr@1":0.0,"mrr@3":0.33,"mrr@5":0.33,"mrr@10":0.33,
     "recall@1":0.0,"recall@3":0.33,"recall@5":0.33,"recall@10":0.67,
     "precision@1":0.0,"precision@3":0.33,"precision@5":0.2,"precision@10":0.2,
     "hit@1":0.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Merdiven çıkma",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.5,"recall@3":1.0,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.4,"precision@10":0.2,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
    {"note": "Su/toz koruması (IP54)",
     "mrr@1":1.0,"mrr@3":1.0,"mrr@5":1.0,"mrr@10":1.0,
     "recall@1":0.5,"recall@3":1.0,"recall@5":1.0,"recall@10":1.0,
     "precision@1":1.0,"precision@3":0.67,"precision@5":0.4,"precision@10":0.2,
     "hit@1":1.0,"hit@3":1.0,"hit@5":1.0,"hit@10":1.0},
]


def avg(results, metric, k):
    return sum(r[f"{metric}@{k}"] for r in results) / len(results)


def style_ax(ax, title, xlabel, ylabel):
    """Tüm grafikler için ortak akademik stil."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#111827")
    ax.set_xlabel(xlabel, fontsize=11, color="#374151")
    ax.set_ylabel(ylabel, fontsize=11, color="#374151")
    ax.tick_params(labelsize=9.5, colors="#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_facecolor("white")


# ─── Fig 1 : Metrik × K Çizgi Grafiği ───────────────────────────────────────
def fig1_line(results, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    for metric, color in C.items():
        y = [avg(results, metric, k) for k in K_VALUES]
        ax.plot(K_VALUES, y, marker="o", linewidth=2, markersize=6,
                color=color, label=LABELS[metric], zorder=3)
        # Sadece K=5 değerini etiketle — kalabalık olmadan
        v5 = avg(results, metric, 5)
        ax.annotate(f"{v5:.2f}", xy=(5, v5),
                    xytext=(6, v5), fontsize=8.5,
                    color=color, fontweight="bold", va="center")

    style_ax(ax, "Retrieval Metriklerinin K'ya Göre Değişimi",
             "K  (Top-K retrieval sayısı)", "Skor")
    ax.set_xticks(K_VALUES)
    ax.set_xlim(0.5, 12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9.5, frameon=True, framealpha=0.9,
              edgecolor="#D1D5DB", loc="center right")

    plt.tight_layout()
    path = out_dir / "fig1_metrics_by_k.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path.name}")
    return path


# ─── Fig 2 : K=5 Metrik Bar Grafiği ─────────────────────────────────────────
def fig2_bar_k5(results, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor("white")

    metrics = list(C.keys())
    vals    = [avg(results, m, 5) for m in metrics]
    colors  = [C[m] for m in metrics]
    xlabels = [LABELS[m] for m in metrics]

    bars = ax.bar(xlabels, vals, color=colors, width=0.52, zorder=3,
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#111827")

    style_ax(ax, "K = 5 için Retrieval Metrikleri", "", "Skor")
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    path = out_dir / "fig2_bar_at_k5.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path.name}")
    return path


# ─── Fig 3 : Sorgu Başına MRR@5 Yatay Bar ───────────────────────────────────
def fig3_per_query_mrr(results, out_dir):
    notes  = [r["note"] for r in results]
    mrr5   = [r["mrr@5"] for r in results]
    mean5  = avg(results, "mrr", 5)

    # Büyükten küçüğe sırala
    order  = sorted(range(len(mrr5)), key=lambda i: mrr5[i], reverse=True)
    notes  = [notes[i]  for i in order]
    mrr5   = [mrr5[i]   for i in order]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("white")

    bar_colors = [C["mrr"] if v > 0 else "#CBD5E1" for v in mrr5]
    bars = ax.barh(range(len(notes)), mrr5, color=bar_colors,
                   height=0.6, zorder=3, edgecolor="white")

    ax.set_yticks(range(len(notes)))
    ax.set_yticklabels(notes, fontsize=9.5)
    ax.set_xlim(0, 1.2)
    ax.axvline(mean5, color="#EF4444", linewidth=1.6, linestyle="--",
               label=f"Ortalama MRR@5 = {mean5:.3f}", zorder=4)

    for bar, v in zip(bars, mrr5):
        if v > 0:
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=8.5,
                    fontweight="bold", color="#111827")

    style_ax(ax, "Sorgu Başına MRR@5", "MRR@5 Skoru", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8, linestyle="--", zorder=0)
    ax.spines["bottom"].set_visible(True)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9,
              edgecolor="#D1D5DB", loc="lower right")

    plt.tight_layout()
    path = out_dir / "fig3_per_query_mrr.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path.name}")
    return path


# ─── Fig 4 : Hit Rate Isı Haritası ───────────────────────────────────────────
def fig4_heatmap(results, out_dir):
    notes  = [r["note"] for r in results]
    matrix = np.array([[r[f"hit@{k}"] for k in K_VALUES] for r in results])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("white")

    # Yeşil-kırmızı renk skalası
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#FCA5A5", "#DCFCE7"], N=2
    )

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f"K={k}" for k in K_VALUES], fontsize=10)
    ax.set_yticks(range(len(notes)))
    ax.set_yticklabels(notes, fontsize=9)

    # Hücre içi ✓ / ✗ sembolleri
    for i in range(len(results)):
        for j in range(len(K_VALUES)):
            val   = matrix[i, j]
            sym   = "✓" if val == 1.0 else "✗"
            color = "#15803D" if val == 1.0 else "#B91C1C"
            ax.text(j, i, sym, ha="center", va="center",
                    fontsize=13, color=color, fontweight="bold")

    ax.set_title("Hit Rate Isı Haritası  (Sorgu × K)", fontsize=13,
                 fontweight="bold", pad=12, color="#111827")
    ax.tick_params(labelsize=9.5, colors="#374151")
    ax.spines[:].set_visible(False)

    # Renk çubuğu
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Bulunamadı (0)", "Bulundu (1)"], fontsize=8.5)

    plt.tight_layout()
    path = out_dir / "fig4_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path.name}")
    return path


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Gerçek retrieval olmadan demo verisiyle çalış")
    args = parser.parse_args()

    if args.demo:
        print("  [DEMO] Örnek verilerle grafik üretiliyor...\n")
        results = DEMO_RESULTS
    else:
        print("  Retrieval başlatılıyor (Qdrant + MongoDB)...\n")
        from evaluate_retrieval import collect_results
        results = collect_results(K_VALUES, pool_size=30)

    print(f"\n  {len(results)} sorgu — 4 grafik üretiliyor...\n")
    fig1_line(results, OUT_DIR)
    fig2_bar_k5(results, OUT_DIR)
    fig3_per_query_mrr(results, OUT_DIR)
    fig4_heatmap(results, OUT_DIR)
    print("\n  Tamamlandı.")
