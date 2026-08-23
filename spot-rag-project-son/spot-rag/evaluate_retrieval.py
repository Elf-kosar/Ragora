"""
RAG Retrieval Evaluation — MRR, Recall@K, Precision@K, Hit Rate@K

Ground truth: her sorgu için hangi anahtar kelimeler ve hangi doküman(lar)
retrieved chunk'larda bulunmalı. Keyword eşleşmesi = relevant chunk.

Kullanım:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --k 5        # sadece K=5 için
    python evaluate_retrieval.py --pool 30    # pool büyüklüğü (recall payda)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.visual_retriever import VisualRetriever

# ─── Ground Truth Veri Seti ──────────────────────────────────────────────────
# Her entry:
#   query          : test sorusu
#   relevant_kws   : bu keyword'lardan EN AZ BİRİ retrieved chunk'ta olmalı
#   expected_docs  : hangi doküman(lar)da cevap var
#   note           : rapor için açıklama

EVAL_DATASET = [
    # ── Genel Spot bilgisi ───────────────────────────────────────────────────
    {
        "query": "What is Spot's maximum payload capacity?",
        "relevant_kws": ["payload", "End-Effector", "attachment", "mount"],
        "expected_docs": ["spot-user-manual-en", "spot-arm-user-manual-en"],
        "note": "Teknik özellik sorgusu",
    },
    {
        "query": "What is Spot's maximum walking speed?",
        "relevant_kws": ["1.6 m/s", "speed", "maximum speed", "1.6m/s"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Hız spesifikasyonu",
    },
    {
        "query": "How long does Spot's battery last?",
        "relevant_kws": ["battery capacity", "564 Wh", "typical", "operating time", "mission time", "battery"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Batarya ömrü",
    },
    # ── Güvenlik ve E-Stop ───────────────────────────────────────────────────
    {
        "query": "How does the E-Stop work and when should I use it?",
        "relevant_kws": ["E-Stop", "emergency stop", "e-stop", "estop"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Acil durdurma mekanizması",
    },
    {
        "query": "What are the safety precautions before operating Spot?",
        "relevant_kws": ["safety", "precaution", "warning", "hazard", "safe operation"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Güvenlik önlemleri",
    },
    {
        "query": "What should I do if Spot falls over?",
        "relevant_kws": ["self-righting", "self righting", "fallen", "fall", "recovery"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Devrilme kurtarma",
    },
    # ── Batarya ve Şarj ─────────────────────────────────────────────────────
    {
        "query": "What are the battery charging safety precautions?",
        "relevant_kws": ["charging", "battery", "charge", "thermal", "temperature"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Şarj güvenliği",
    },
    {
        "query": "How do I install and remove the battery?",
        "relevant_kws": ["battery", "install", "remove", "insert", "latch"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Batarya takma/çıkarma",
    },
    # ── Güç ve Başlatma ──────────────────────────────────────────────────────
    {
        "query": "How do I power on and power off Spot?",
        "relevant_kws": ["power on", "power off", "startup", "shutdown", "motor enable"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Güç açma/kapama prosedürü",
    },
    {
        "query": "What is the motor enable button?",
        "relevant_kws": ["motor enable", "motor power", "enable motors"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Motor aktivasyonu",
    },
    # ── Spot Arm ────────────────────────────────────────────────────────────
    {
        "query": "How do I attach the Spot Arm to the robot?",
        "relevant_kws": ["attach", "mount", "install", "arm attachment", "connector"],
        "expected_docs": ["spot-arm-user-manual-en"],
        "note": "Arm takma prosedürü",
    },
    {
        "query": "What is the payload capacity of the Spot Arm?",
        "relevant_kws": ["payload", "arm payload", "kg", "load capacity", "end effector"],
        "expected_docs": ["spot-arm-user-manual-en"],
        "note": "Arm yük kapasitesi",
    },
    # ── Spot Dock ───────────────────────────────────────────────────────────
    {
        "query": "How does Spot autonomously dock?",
        "relevant_kws": ["dock", "autonomous", "docking", "return to dock", "AutoReturn"],
        "expected_docs": ["spot-dock-user-manual-en"],
        "note": "Otonom docking",
    },
    # ── Navigasyon ve Operasyon ──────────────────────────────────────────────
    {
        "query": "Can Spot climb stairs and what are the limitations?",
        "relevant_kws": ["stair", "stairs", "steps", "climbing", "incline", "slope"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Merdiven çıkma",
    },
    {
        "query": "What is the IP rating and water resistance of Spot?",
        "relevant_kws": ["IP54", "IP rating", "water", "dust", "ingress protection"],
        "expected_docs": ["spot-user-manual-en"],
        "note": "Su ve toz koruması",
    },
]


# ─── Relevance Fonksiyonu ────────────────────────────────────────────────────

def is_relevant(chunk, gt: dict) -> bool:
    """
    Chunk'ın ground truth'a göre ilgili olup olmadığını belirler.

    Kriter (AND):
      1. Chunk doc_name, beklenen dokümanlardan biri olmalı
      2. Chunk raw_text içinde en az 1 keyword geçmeli
    """
    if gt.get("expected_docs"):
        if chunk.doc_name not in gt["expected_docs"]:
            return False

    text_lower = chunk.raw_text.lower()
    return any(kw.lower() in text_lower for kw in gt["relevant_kws"])


# ─── Metrik Hesaplama ────────────────────────────────────────────────────────

def compute_mrr_at_k(chunks: list, gt: dict, k: int) -> float:
    """MRR@K: top-k içinde ilk relevant chunk'ın 1/rank'ı."""
    for rank, chunk in enumerate(chunks[:k], 1):
        if is_relevant(chunk, gt):
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(chunks: list, gt: dict, k: int) -> float:
    """Precision@K: top-k chunk içinde relevant olanların oranı."""
    if not chunks:
        return 0.0
    relevant_in_k = sum(1 for c in chunks[:k] if is_relevant(c, gt))
    return relevant_in_k / min(k, len(chunks))


def compute_recall_at_k(chunks: list, gt: dict, k: int, pool: list) -> float:
    """
    Recall@K: top-k içinde bulunan relevant chunk sayısı /
              pool (büyük retrieval) içindeki toplam relevant chunk sayısı.

    pool: tüm corpus'u temsil eden geniş retrieval seti (top_k=N ile).
    """
    total_relevant = sum(1 for c in pool if is_relevant(c, gt))
    if total_relevant == 0:
        return 0.0
    relevant_in_k = sum(1 for c in chunks[:k] if is_relevant(c, gt))
    return relevant_in_k / total_relevant


def compute_hit_rate_at_k(chunks: list, gt: dict, k: int) -> float:
    """Hit Rate@K: top-k içinde en az 1 relevant chunk var mı?"""
    return 1.0 if any(is_relevant(c, gt) for c in chunks[:k]) else 0.0


# ─── Ana Değerlendirme ───────────────────────────────────────────────────────

def sep(char="─", n=80):
    print(char * n)


def collect_results(k_values: list[int], pool_size: int = 30) -> list[dict]:
    """
    Değerlendirmeyi çalıştırır ve ham sonuç listesini döndürür.
    Görselleştirme scriptleri tarafından import edilebilir.
    """
    from pipeline.visual_retriever import VisualRetriever
    import time

    retriever = VisualRetriever(top_k_text=pool_size, top_k_visual=1)
    retriever._text_retriever.score_threshold = 0.05

    per_query_results = []
    for qi, gt in enumerate(EVAL_DATASET, 1):
        query = gt["query"]
        print(f"  [{qi:02d}/{len(EVAL_DATASET)}] {query[:65]}", end=" ", flush=True)
        t0 = time.time()
        result = retriever.retrieve_by_text(query=query, auto_lang=False)
        elapsed = time.time() - t0
        all_chunks = sorted(result.text_chunks, key=lambda c: c.score, reverse=True)

        row = {
            "query": query,
            "note": gt["note"],
            "chunks": all_chunks,
            "pool_size": len(all_chunks),
            "elapsed": elapsed,
        }
        for k in k_values:
            row[f"mrr@{k}"]       = compute_mrr_at_k(all_chunks, gt, k)
            row[f"recall@{k}"]    = compute_recall_at_k(all_chunks, gt, k, pool=all_chunks)
            row[f"precision@{k}"] = compute_precision_at_k(all_chunks, gt, k)
            row[f"hit@{k}"]       = compute_hit_rate_at_k(all_chunks, gt, k)

        relevant_count = sum(1 for c in all_chunks if is_relevant(c, gt))
        hit_k = k_values[0]
        status = "✓" if row[f"hit@{hit_k}"] == 1.0 else "✗"
        print(f"→ {relevant_count} relevant / {len(all_chunks)} pool  [{elapsed:.1f}s]  {status}")
        per_query_results.append(row)

    return per_query_results


def run_evaluation(k_values: list[int], pool_size: int = 30):
    print("\n" + "=" * 80)
    print("  SPOT RAG — RETRIEVAL EVALUATION  (MRR / Recall / Precision / Hit Rate)")
    print("=" * 80)
    print(f"  Test sorgu sayısı : {len(EVAL_DATASET)}")
    print(f"  K değerleri       : {k_values}")
    print(f"  Pool büyüklüğü    : {pool_size}")
    print()

    print("  Retriever yükleniyor (mE5 + CLIP)...", end=" ", flush=True)
    per_query_results = collect_results(k_values, pool_size)
    print()

    # ── Detaylı Sorgu Tablosu ────────────────────────────────────────────────
    print()
    sep("═")
    primary_k = k_values[0]
    header = (
        f"{'#':>2}  {'Sorgu':<50}  "
        f"{'MRR@' + str(primary_k):>6}  "
        f"{'Rec@' + str(primary_k):>6}  "
        f"{'Prec@' + str(primary_k):>7}  "
        f"{'Hit@' + str(primary_k):>6}"
    )
    print("  " + header)
    sep()
    for qi, row in enumerate(per_query_results, 1):
        q = row["query"][:48] + ".." if len(row["query"]) > 48 else row["query"]
        print(
            f"  {qi:>2}  {q:<50}  "
            f"{row[f'mrr@{primary_k}']:>6.3f}  "
            f"{row[f'recall@{primary_k}']:>6.3f}  "
            f"{row[f'precision@{primary_k}']:>7.3f}  "
            f"{row[f'hit@{primary_k}']:>6.1f}"
        )
    sep()

    # ── K Değerleri Karşılaştırma Tablosu ───────────────────────────────────
    print("\n  METRİK ÖZET (tüm K değerleri)\n")
    col_w = 10
    header_row = f"  {'Metrik':<15}" + "".join(f"{'K=' + str(k):>{col_w}}" for k in k_values)
    print(header_row)
    sep()

    metric_labels = {
        "mrr": "MRR@K",
        "recall": "Recall@K",
        "precision": "Precision@K",
        "hit": "Hit Rate@K",
    }
    for metric_key, label in metric_labels.items():
        row_str = f"  {label:<15}"
        for k in k_values:
            avg = sum(r[f"{metric_key}@{k}"] for r in per_query_results) / len(per_query_results)
            row_str += f"{avg:>{col_w}.4f}"
        print(row_str)
    sep()

    # ── Birincil Sonuçlar ────────────────────────────────────────────────────
    pk = primary_k
    avg_mrr       = sum(r[f"mrr@{pk}"]       for r in per_query_results) / len(per_query_results)
    avg_recall    = sum(r[f"recall@{pk}"]    for r in per_query_results) / len(per_query_results)
    avg_precision = sum(r[f"precision@{pk}"] for r in per_query_results) / len(per_query_results)
    avg_hit       = sum(r[f"hit@{pk}"]       for r in per_query_results) / len(per_query_results)

    print(f"""
  BİRİNCİL SONUÇLAR (K={pk}, N={len(EVAL_DATASET)} sorgu, pool={pool_size})
  ─────────────────────────────────────────────────────────────
  MRR@{pk}        : {avg_mrr:.4f}   (1.0 = ilk sonuç her zaman doğru)
  Recall@{pk}     : {avg_recall:.4f}   (1.0 = tüm ilgili chunk'lar getirildi)
  Precision@{pk}  : {avg_precision:.4f}   (1.0 = getirilen her chunk ilgili)
  Hit Rate@{pk}   : {avg_hit:.4f}   (1.0 = her sorguda en az 1 doğru sonuç)
""")

    # ── Başarısız Sorgular ───────────────────────────────────────────────────
    failed = [r for r in per_query_results if r[f"hit@{pk}"] == 0.0]
    if failed:
        print(f"  BULUNAMAYAN SORGULAR ({len(failed)} adet):")
        for r in failed:
            print(f"    ✗ {r['query']}")
    else:
        print(f"  Tüm {len(EVAL_DATASET)} sorgu için en az 1 relevant chunk bulundu.")
    print()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 3, 5, 10],
        help="K değerleri (örn: --k 1 3 5 10)"
    )
    parser.add_argument(
        "--pool", type=int, default=30,
        help="Recall paydasını hesaplamak için büyük retrieval havuzu (varsayılan: 30)"
    )
    args = parser.parse_args()

    run_evaluation(k_values=args.k, pool_size=args.pool)
