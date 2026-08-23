"""
Multi-PDF Hybrid Indexer

6 (veya daha fazla) PDF'i sırayla indexler.
Her PDF için HybridIndexer çalışır:
  Docling (metin + tablo) + VLM (görsel sayfalar) + CLIP embed

Özellikler
----------
- Hash kontrolü   : değişmeyen PDF'ler atlanır
- Stale detection : içerik değişen PDF'ler otomatik yeniden indexlenir
- Hata izolasyonu : bir PDF patlarsa diğerleri devam eder
- Tek model yükleme: BGE, CLIP, VLM tüm PDF'ler için bir kez yüklenir
- --no-vlm flag'i : VLM olmadan hızlı indexleme (sadece metin/tablo)

Kullanım
--------
    # Dizindeki tüm PDF'ler (VLM dahil)
    python -m pipeline.multi_indexer --dir data/pdfs/

    # VLM olmadan (hızlı, sadece metin+tablo)
    python -m pipeline.multi_indexer --dir data/pdfs/ --no-vlm

    # Belirli dosyalar
    python -m pipeline.multi_indexer --files a.pdf b.pdf c.pdf

    # Zorla yeniden indexle
    python -m pipeline.multi_indexer --dir data/pdfs/ --force

    # Etiket ekle
    python -m pipeline.multi_indexer --dir data/pdfs/ --tags robotics spot
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

from pipeline.hybrid_indexer import HybridIndexer
from stores.doc_registry import DocRegistry


class MultiIndexer:
    """
    Birden fazla PDF'i hibrit pipeline ile indexler.

    HybridIndexer'ı (Docling + VLM + CLIP) bir kez başlatır,
    tüm PDF'lerde aynı model instance'ı paylaşır.
    6 PDF için model 1 kez yüklenir, 6 kez değil.

    Parameters
    ----------
    use_vlm : bool
        False → VLM atlanır, sadece metin+tablo indexlenir.
    vlm_model : str
        HuggingFace VLM model adı.
    force : bool
        True → hash kontrolünü atla, her PDF'i yeniden indexle.
    tags : list[str]
        Her indexlenen dokümana eklenecek etiketler.
    """

    def __init__(
        self,
        use_vlm: bool = True,
        vlm_model: str = "llava-hf/llava-v1.6-34b-hf",
        force: bool = False,
        tags: list[str] | None = None,
        use_ollama_vlm: bool = True,
    ):
        logger.info("MultiIndexer başlatılıyor...")
        logger.info(f"  VLM: {'açık (' + vlm_model + ')' if use_vlm else 'kapalı'}")

        # Tüm modeller bir kez yüklenir
        self._hybrid = HybridIndexer(
            use_vlm=use_vlm,
            vlm_model=vlm_model,
            force=False,  # skip/force kararı burada verilir
            use_ollama_vlm=use_ollama_vlm,
        )
        self._registry = self._hybrid.registry
        self.force = force
        self.tags = tags or []
        self.text_only = False  # --text-only flag'i ile set edilir

        logger.success("MultiIndexer hazır")

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def index_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
        glob: str = "*.pdf",
    ) -> dict:
        """Bir dizindeki tüm PDF'leri indexler."""
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Dizin bulunamadı: {directory}")

        pattern = f"**/{glob}" if recursive else glob
        pdf_files = sorted(directory.glob(pattern))

        if not pdf_files:
            logger.warning(f"'{directory}' dizininde PDF bulunamadı")
            return {}

        logger.info(f"{len(pdf_files)} PDF bulundu → {directory}")
        return self.index_files(pdf_files)

    def index_files(self, pdf_paths: list[str | Path]) -> dict:
        """
        Verilen PDF listesini sırayla indexler.

        Returns
        -------
        dict  {total, indexed, skipped, failed, total_seconds, results}
        """
        pdf_paths = [Path(p) for p in pdf_paths]
        total_start = time.time()

        summary = {
            "total": len(pdf_paths),
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "results": [],
        }

        for i, pdf_path in enumerate(pdf_paths, 1):
            logger.info("=" * 60)
            logger.info(f"[{i}/{len(pdf_paths)}] {pdf_path.name}")

            result = self._process_one(pdf_path)
            summary["results"].append(result)

            if result["action"] == "indexed":
                summary["indexed"] += 1
            elif result["action"] == "skipped":
                summary["skipped"] += 1
            else:
                summary["failed"] += 1

        summary["total_seconds"] = round(time.time() - total_start, 2)
        self._print_summary(summary)
        return summary

    # ─────────────────────────────────────────────────────────────────
    # Tek PDF işleme
    # ─────────────────────────────────────────────────────────────────

    def _process_one(self, pdf_path: Path) -> dict:
        """Hash kontrolü yapar, gerekirse HybridIndexer ile indexler."""
        if not pdf_path.exists():
            return {
                "action": "failed",
                "doc_name": pdf_path.stem,
                "reason": f"Dosya bulunamadı: {pdf_path}",
                "stats": {},
            }

        doc_name = DocRegistry.make_doc_name(pdf_path)
        check = self._registry.check(pdf_path)

        # ── Skip / Stale kararı ──────────────────────────────────────
        if not self.force:
            if check["status"] == "indexed":
                logger.info(f"  ⏭  ATLANDI: '{doc_name}' zaten güncel")
                return {"action": "skipped", "doc_name": doc_name,
                        "reason": "already_indexed", "stats": {}}

            if check["status"] == "stale":
                logger.warning(f"  ♻  STALE: '{doc_name}' değişmiş, yeniden indexleniyor...")
        else:
            if check["status"] == "indexed":
                logger.warning(f"  🔄 FORCE: '{doc_name}' zorla yeniden indexleniyor...")

        # ── Hibrit indexleme ─────────────────────────────────────────
        try:
            # Skip kararını biz verdik; HybridIndexer'a force=True
            # geçerek kendi skip kontrolünü bypass ediyoruz.
            self._hybrid.force = True
            self._hybrid.tags = self.tags
            stats = self._hybrid.run(pdf_path, text_only=self.text_only)
            self._hybrid.force = False

            logger.success(
                f"  ✓ '{doc_name}' tamamlandı — "
                f"{stats.get('duration_seconds', '?')}s"
            )
            return {"action": "indexed", "doc_name": doc_name,
                    "reason": "", "stats": stats}

        except Exception as e:
            self._hybrid.force = False
            logger.error(f"  ✗ HATA: '{doc_name}' — {e}")
            try:
                self._registry.mark_failed(pdf_path, check.get("file_hash", ""), str(e))
            except Exception:
                pass
            return {"action": "failed", "doc_name": doc_name,
                    "reason": str(e), "stats": {}}

    # ─────────────────────────────────────────────────────────────────
    # Silme
    # ─────────────────────────────────────────────────────────────────

    def delete_document(self, doc_name: str) -> dict:
        """Dokümanı tamamen siler (metin + görsel chunk'lar + registry)."""
        self._hybrid._delete_existing(doc_name)
        self._registry.delete(doc_name)
        logger.success(f"'{doc_name}' silindi")
        return {"doc_name": doc_name}

    # ─────────────────────────────────────────────────────────────────
    # Özet
    # ─────────────────────────────────────────────────────────────────

    def _print_summary(self, summary: dict) -> None:
        logger.info("=" * 60)
        logger.success("ÇOKLU İNDEKSLEME TAMAMLANDI")
        logger.info(f"  Toplam PDF   : {summary['total']}")
        logger.info(f"  İndexlendi   : {summary['indexed']}")
        logger.info(f"  Atlandı      : {summary['skipped']}")
        logger.info(f"  Hatalı       : {summary['failed']}")
        logger.info(f"  Toplam süre  : {summary['total_seconds']}s")

        indexed = [r for r in summary["results"] if r["action"] == "indexed"]
        if indexed:
            logger.info("  " + "─" * 56)
            for r in indexed:
                s = r.get("stats", {})
                logger.info(
                    f"  ✓ {r['doc_name']:<35} "
                    f"metin={s.get('text_chunks', '?'):>4}  "
                    f"görsel={s.get('visual_elements', '?'):>3}  "
                    f"{s.get('duration_seconds', '?')}s"
                )

        if summary["failed"] > 0:
            logger.warning("  " + "─" * 56)
            for r in summary["results"]:
                if r["action"] == "failed":
                    logger.warning(f"  ✗ {r['doc_name']}: {r['reason'][:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Çoklu PDF → Hibrit RAG indexer (Docling + VLM + CLIP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # 6 PDF dizinden indexle (VLM dahil)
  python -m pipeline.multi_indexer --dir data/pdfs/

  # VLM olmadan hızlı indexle
  python -m pipeline.multi_indexer --dir data/pdfs/ --no-vlm

  # Alt dizinler dahil
  python -m pipeline.multi_indexer --dir data/pdfs/ --recursive

  # Belirli dosyalar
  python -m pipeline.multi_indexer --files a.pdf b.pdf c.pdf

  # Zorla yeniden indexle
  python -m pipeline.multi_indexer --dir data/pdfs/ --force

  # Farklı VLM modeli
  python -m pipeline.multi_indexer --dir data/pdfs/ \\
      --vlm-model Qwen/Qwen2-VL-7B-Instruct --tags spot robotics
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="PDF dizini")
    group.add_argument("--files", nargs="+", help="PDF dosyaları listesi")

    parser.add_argument("--recursive", action="store_true", help="Alt dizinleri de tara")
    parser.add_argument("--force", action="store_true", help="Hash kontrolünü atla")
    parser.add_argument("--no-vlm", action="store_true", help="VLM'i atla (sadece metin+tablo)")
    parser.add_argument("--text-only", action="store_true",
                        help="Sadece metin chunk'larını yeniden indexle, görsel chunk'lara dokunma")
    parser.add_argument("--hf-vlm", action="store_true",
                        help="HuggingFace VLM kullan (varsayılan: Ollama llava:34b)")
    parser.add_argument(
        "--vlm-model", default="llava-hf/llava-v1.6-34b-hf",
        help="HuggingFace VLM model adı (--hf-vlm ile kullanılır)",
    )
    parser.add_argument("--tags", nargs="*", default=[], help="Doküman etiketleri")

    args = parser.parse_args()

    indexer = MultiIndexer(
        use_vlm=not args.no_vlm,
        vlm_model=args.vlm_model,
        force=args.force,
        tags=args.tags,
        use_ollama_vlm=not args.hf_vlm,
    )
    indexer.text_only = args.text_only

    if args.dir:
        indexer.index_directory(args.dir, recursive=args.recursive)
    else:
        indexer.index_files(args.files)


if __name__ == "__main__":
    main()
