"""
Hybrid Indexer — Docling metin + VLM görsel birleşik pipeline.

Akış
----
PDF
 │
 ├─► Docling ──────────────────────────► metin/tablo chunk'lar
 │    │                                       │
 │    └─ figure_caption / picture tespit       │
 │         │                                  │
 │         ▼                                  │
 │    VisualParser (VLM)                       │
 │         │                                  │
 │         ▼                                  ▼
 │    VisualElement             SemanticChunker (Chunk)
 │         │                                  │
 │    CLIP embed             BGE embed         │
 │         │                        │          │
 │         ▼                        ▼          │
 │    VisualQdrant           Qdrant            │
 │    VisualMongo            MongoDB           │
 │                                             │
 └─────────────────────────────────────────────┘

Kullanım
--------
    python -m pipeline.hybrid_indexer --pdf data/spot-user-manual-en.pdf
    python -m pipeline.hybrid_indexer --pdf data/spot-user-manual-en.pdf --no-vlm
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

from chunkers.semantic_chunker import SemanticChunker
from config.settings import settings
from embedders.clip_embedder import CLIPEmbedder
from embedders.local_embedder import LocalEmbedder
from parsers.docling_parser import DoclingPDFParser, ParsedElement
from parsers.visual_parser import VisualParser
from stores.doc_registry import DocRegistry
from utils.lang_utils import detect_lang, lang_display
from stores.mongo_store import MongoChunkStore
from stores.qdrant_store import QdrantVectorStore
from stores.visual_store import VisualMongoStore, VisualQdrantStore


class HybridIndexer:
    """
    Docling + VLM birleşik indexleme pipeline'ı.

    Parameters
    ----------
    use_vlm : bool
        False → VLM atlanır, sadece figure_caption kaydedilir.
        Hız testi veya VRAM yetersizliğinde kullanılır.
    vlm_model : str
        HuggingFace VLM model adı.
    force : bool
        True → Hash kontrolü atlanır, yeniden indexlenir.
    """

    def __init__(
        self,
        use_vlm: bool = True,
        vlm_model: str = "llava-hf/llava-v1.6-34b-hf",
        force: bool = False,
        use_ollama_vlm: bool = True,
    ):
        logger.info("HybridIndexer başlatılıyor...")

        # Metin tarafı
        self.parser = DoclingPDFParser(enable_ocr=False, table_mode="accurate")
        self.chunker = SemanticChunker(
            max_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
            min_tokens=settings.min_chunk_size,
        )
        self.text_embedder = LocalEmbedder(
            model_name=settings.embed_model,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
        self.mongo = MongoChunkStore(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
            collection_name=settings.mongo_collection,
        )
        self.qdrant = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection,
            vector_dim=self.text_embedder.dimension,
        )

        # Görsel tarafı
        self.clip_embedder = CLIPEmbedder(
            model_name=settings.clip_model,
            device=settings.embed_device,
        )
        self.visual_mongo = VisualMongoStore(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
            collection_name=settings.visual_mongo_collection,
        )
        self.visual_qdrant = VisualQdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.visual_qdrant_collection,
            vector_dim=self.clip_embedder.dimension,
        )

        self.registry = DocRegistry(
            uri=settings.mongo_uri,
            db_name=settings.mongo_db,
        )

        self.use_vlm = use_vlm
        self.vlm_model = vlm_model if use_vlm else None
        self.force = force
        # Ollama VLM: HuggingFace yerine Ollama kullan (35GB MIG'a sığar)
        self.use_ollama_vlm = use_ollama_vlm and use_vlm

        # VLM başlangıçta yüklenmez — sadece görsel sayfalar varsa yüklenir
        self._visual_parser: VisualParser | None = None

    def run(self, pdf_path: str | Path, text_only: bool = False) -> dict:
        """
        Tam hibrit indexleme pipeline'ını çalıştırır.

        Parameters
        ----------
        text_only : bool
            True → sadece metin/tablo chunk'larını yeniden indexler,
            görsel chunk'lara dokunmaz. Chunker değişikliklerinde kullanılır.
        """
        pdf_path = Path(pdf_path)
        doc_name = DocRegistry.make_doc_name(pdf_path)
        start = time.time()

        # Hash kontrolü
        check = self.registry.check(pdf_path)
        if not self.force and check["status"] == "indexed":
            logger.info(f"'{doc_name}' zaten güncel, atlanıyor.")
            return {"action": "skipped", "doc_name": doc_name}

        file_hash = check["file_hash"]
        lang = detect_lang(pdf_path)
        logger.info(f"  Dil: {lang_display(lang)}")

        # Stale veya force ise eski verileri sil
        if check["status"] == "stale" or self.force:
            if text_only:
                self._delete_text_only(doc_name)
            else:
                self._delete_existing(doc_name)

        logger.info("=" * 60)
        logger.info(f"HYBRİD İNDEKSLEME: {pdf_path.name}")
        logger.info("=" * 60)

        # ── ADIM 1: Docling Parse ─────────────────────────────────────
        logger.info("ADIM 1/5: Docling parse (metin + tablo + görsel tespit)")
        elements = list(self.parser.parse(pdf_path))

        # Tekil figürleri PyMuPDF ile çıkar (bbox bazlı)
        visual_pages = self._extract_visual_pages(elements, doc_name, pdf_path)

        text_elements = [e for e in elements if e.label not in {"picture", "figure_caption"}]

        logger.info(f"  {len(text_elements)} metin/tablo elementi")
        logger.info(f"  {len(visual_pages)} görsel sayfa tespit edildi")

        # ── ADIM 2: Metin Chunking ────────────────────────────────────
        logger.info("ADIM 2/5: Semantic chunking")
        self.chunker.doc_name = doc_name
        self.chunker.lang = lang
        chunks = self.chunker.chunk(text_elements)
        logger.info(f"  {len(chunks)} chunk")

        # ── ADIM 3: Metin Embed + Store ───────────────────────────────
        logger.info("ADIM 3/5: Metin embed + store")
        texts = [c.text for c in chunks]
        text_embeddings = self.text_embedder.embed_chunks(texts)
        self.mongo.upsert_chunks(chunks)
        self.qdrant.upsert_embeddings(chunks, text_embeddings)

        # ── ADIM 4: VLM Görsel Açıklama ──────────────────────────────
        visual_elements = []

        if text_only:
            logger.info("ADIM 4/5: text_only modu — görsel atlanıyor")
            vis_count = self.visual_mongo.count()  # mevcut görsel sayısını koru
        else:
            logger.info("ADIM 4/5: VLM görsel işleme")
            if visual_pages:
                visual_parser = self._get_visual_parser(pdf_path)
                visual_elements = visual_parser.process_visual_pages(
                    visual_pages, doc_name
                )
                logger.info(f"  {len(visual_elements)} görsel element üretildi")
            else:
                logger.info("  Görsel sayfa yok, atlanıyor")

        # ── ADIM 5: Görsel Embed + Store ──────────────────────────────
        if text_only:
            logger.info("ADIM 5/5: text_only modu — görsel store atlanıyor")
        elif visual_elements:
            logger.info("ADIM 5/5: Görsel CLIP embed + store")
            vis_texts = [el.searchable_text for el in visual_elements]
            vis_embeddings = self.clip_embedder.embed_texts(vis_texts)
            self.visual_mongo.upsert_visuals(visual_elements)
            self.visual_qdrant.upsert_embeddings(visual_elements, vis_embeddings)
        else:
            logger.info("ADIM 5/5: Görsel yok, atlanıyor")

        # Registry'e kaydet
        page_count = max((e.page for e in elements), default=0)
        vis_count_final = len(visual_elements) if not text_only else self.visual_mongo.count()
        self.registry.register(
            pdf_path=pdf_path,
            file_hash=file_hash,
            page_count=page_count,
            chunk_count=len(chunks),
            tags=["hybrid", "vlm" if self.use_vlm else "no-vlm", f"lang:{lang}"],
            meta={
                "visual_count": vis_count_final,
                "vlm_used": self.use_vlm,
                "lang": lang,
            },
        )

        duration = round(time.time() - start, 2)
        stats = {
            "action": "indexed",
            "doc_name": doc_name,
            "text_chunks": len(chunks),
            "visual_elements": len(visual_elements),
            "duration_seconds": duration,
        }

        logger.info("=" * 60)
        logger.success(f"HYBRİD İNDEKSLEME TAMAMLANDI — {duration}s")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        return stats

    # ─────────────────────────────────────────────────────────────────
    # Görsel sayfa tespiti
    # ─────────────────────────────────────────────────────────────────

    def _extract_visual_pages(
        self,
        elements: list[ParsedElement],
        doc_name: str,
        pdf_path: Path | None = None,
    ) -> list[dict]:
        """
        PDF'deki tekil figürleri PyMuPDF ile tespit eder.

        Her figür için ayrı dict döner — sayfa başına birden fazla
        figür olabilir. Docling'in figure_caption'ları da eklenir.

        Returns
        -------
        list[dict]
          [{page, figure_idx, figure_caption, breadcrumb,
            visual_type, bbox, page_width, page_height}, ...]
        """
        import fitz

        # Docling elementlerinden sayfa → breadcrumb ve caption haritaları
        page_breadcrumb: dict[int, list[str]] = {}
        page_captions: dict[int, str] = {}

        for el in elements:
            if el.heading_level is not None and el.page not in page_breadcrumb:
                page_breadcrumb[el.page] = el.breadcrumb or []
            if el.label == "figure_caption" and el.text:
                existing = page_captions.get(el.page, "")
                page_captions[el.page] = (existing + " " + el.text).strip() if existing else el.text

        if pdf_path is None:
            logger.warning("pdf_path verilmedi — görsel çıkarma atlanıyor")
            return []

        figures: list[dict] = []

        try:
            doc = fitz.open(str(pdf_path))
            for page_idx in range(len(doc)):
                page_no = page_idx + 1
                page_obj = doc[page_idx]
                page_rect = page_obj.rect
                page_area = page_rect.width * page_rect.height
                if page_area == 0:
                    continue

                # Breadcrumb
                bc: list[str] = []
                for p in range(page_no, 0, -1):
                    if p in page_breadcrumb:
                        bc = page_breadcrumb[p]
                        break

                caption = page_captions.get(page_no, "")

                # PyMuPDF ile sayfadaki raster görüntü bbox'ları
                page_bboxes: list[dict] = []
                for img in page_obj.get_images(full=True):
                    xref = img[0]
                    try:
                        for rect in page_obj.get_image_rects(xref):
                            area_ratio = (rect.width * rect.height) / page_area
                            if area_ratio >= 0.05:  # en az sayfanın %5'i
                                page_bboxes.append({
                                    "x0": float(rect.x0), "y0": float(rect.y0),
                                    "x1": float(rect.x1), "y1": float(rect.y1),
                                })
                    except Exception:
                        continue

                # Üst üste gelen bbox'ları temizle (10 point tolerans)
                merged: list[dict] = []
                for bbox in sorted(page_bboxes, key=lambda b: b["y0"]):
                    duplicate = any(
                        abs(b["x0"] - bbox["x0"]) < 10 and abs(b["y0"] - bbox["y0"]) < 10
                        for b in merged
                    )
                    if not duplicate:
                        merged.append(bbox)

                if merged:
                    for fig_idx, bbox in enumerate(merged):
                        figures.append({
                            "page": page_no,
                            "figure_idx": fig_idx,
                            "figure_caption": caption if fig_idx == 0 else "",
                            "breadcrumb": bc,
                            "visual_type": "diagram",
                            "bbox": bbox,
                            "page_width": float(page_rect.width),
                            "page_height": float(page_rect.height),
                        })
                elif caption:
                    # Caption var ama raster görüntü bulunamadı → tam sayfa render
                    figures.append({
                        "page": page_no,
                        "figure_idx": 0,
                        "figure_caption": caption,
                        "breadcrumb": bc,
                        "visual_type": "diagram",
                        "bbox": None,
                        "page_width": float(page_rect.width),
                        "page_height": float(page_rect.height),
                    })

            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF figür tespiti başarısız: {e}")
            return []

        logger.info(f"  {len(figures)} tekil figür tespit edildi (sayfa bazlı değil, figür bazlı)")
        return figures

    # ─────────────────────────────────────────────────────────────────
    # Lazy VLM yükleme
    # ─────────────────────────────────────────────────────────────────

    def _get_visual_parser(self, pdf_path: Path) -> VisualParser:
        """
        VLM parser'ı ilk ihtiyaç anında yükler (lazy loading).
        Görsel sayfa yoksa VLM hiç GPU'ya yüklenmez.
        """
        if self._visual_parser is None:
            if self.use_ollama_vlm:
                # Ollama tabanlı VLM — 35GB MIG'a sığar
                self._visual_parser = VisualParser(
                    pdf_path=pdf_path,
                    vlm_model_name=None,
                    device=settings.embed_device,
                    ollama_host=settings.ollama_host,
                    ollama_vision_model=settings.ollama_vision_model,
                )
            else:
                # HuggingFace VLM — daha fazla VRAM gerektirir
                self._visual_parser = VisualParser(
                    pdf_path=pdf_path,
                    vlm_model_name=self.vlm_model,
                    device=settings.embed_device,
                )
        else:
            # PDF yolu değişmiş olabilir (çoklu indexleme)
            self._visual_parser.pdf_path = pdf_path
        return self._visual_parser

    # ─────────────────────────────────────────────────────────────────
    # Temizlik
    # ─────────────────────────────────────────────────────────────────

    def _delete_text_only(self, doc_name: str) -> None:
        """Sadece metin chunk'larını siler, görsel chunk'lara dokunmaz."""
        logger.info(f"'{doc_name}' metin chunk'ları siliniyor (görsel korunuyor)...")
        chunk_ids = self.mongo.get_chunk_ids_by_doc(doc_name)
        self.mongo.delete_by_doc(doc_name)
        self.qdrant.delete_by_ids(chunk_ids)
        self.registry.mark_stale(doc_name)

    def _delete_existing(self, doc_name: str) -> None:
        """Tüm eski verileri siler (metin + görsel)."""
        logger.info(f"'{doc_name}' eski veriler siliniyor...")

        # Metin chunk'lar
        chunk_ids = self.mongo.get_chunk_ids_by_doc(doc_name)
        self.mongo.delete_by_doc(doc_name)
        self.qdrant.delete_by_ids(chunk_ids)

        # Görsel chunk'lar
        vis_ids = self.visual_mongo.get_element_ids_by_doc(doc_name)
        self.visual_mongo.delete_by_doc(doc_name)
        self.visual_qdrant.delete_by_ids(vis_ids)

        self.registry.mark_stale(doc_name)


def main():
    parser = argparse.ArgumentParser(description="Hibrit PDF indexer (Docling + VLM)")
    parser.add_argument("--pdf", required=True, help="PDF dosyası")
    parser.add_argument("--no-vlm", action="store_true", help="VLM'i atla")
    parser.add_argument(
        "--vlm-model",
        default="llava-hf/llava-v1.6-34b-hf",
        help="VLM model adı (default: InternVL2-8B)",
    )
    parser.add_argument("--force", action="store_true", help="Hash kontrolünü atla")
    args = parser.parse_args()

    indexer = HybridIndexer(
        use_vlm=not args.no_vlm,
        vlm_model=args.vlm_model,
        force=args.force,
    )
    indexer.run(args.pdf)


if __name__ == "__main__":
    main()
