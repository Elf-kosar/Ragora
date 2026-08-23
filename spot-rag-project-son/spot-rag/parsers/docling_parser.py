"""
Docling tabanlı PDF parser.

Neden Docling?
- Layout-aware: başlıkları, tabloları, figür başlıklarını, listeleri
  yapısal olarak tanır; düz pdftotext gibi sıra karışmaz.
- Table recovery: PDF'deki tabloları satır/sütun yapısıyla çıkarır,
  hücreleri birleştirip Markdown'a dönüştürür.
- Reading order: çok sütunlu sayfalarda doğru okuma sırası.
- Başlık hiyerarşisi: H1/H2/H3 düzeylerini metadata olarak korur.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel
from loguru import logger


@dataclass
class ParsedElement:
    """
    PDF'den çıkarılan tek bir yapısal eleman.

    Attributes
    ----------
    element_id : str
        Doküman içinde benzersiz ID (sayfa + sıra numarası).
    label : str
        Elemanın türü: "section_header", "text", "table",
        "list_item", "figure_caption", "page_header", "page_footer" vb.
    text : str
        Ham metin içeriği (tablo için Markdown formatında).
    page : int
        1-tabanlı sayfa numarası.
    heading_level : Optional[int]
        Sadece section_header için: 1 (H1), 2 (H2), 3 (H3).
    breadcrumb : list[str]
        Bu elemanın üstündeki başlık zinciri.
        Örn: ["Safety", "Emergency Stop Procedures"]
    metadata : dict
        Ek bilgiler (bbox koordinatları vb.).
    """
    element_id: str
    label: str
    text: str
    page: int
    heading_level: Optional[int] = None
    breadcrumb: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def is_structural_noise(self) -> bool:
        """Sayfa başlığı/altbilgisi gibi gürültülü elemanları filtreler."""
        noise_labels = {
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.FOOTNOTE,
        }
        return self.label in {str(l) for l in noise_labels}

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "label": self.label,
            "text": self.text,
            "page": self.page,
            "heading_level": self.heading_level,
            "breadcrumb": self.breadcrumb,
            "metadata": self.metadata,
        }


class DoclingPDFParser:
    """
    Docling ile PDF'i yapısal elemanlara ayrıştırır.

    Kullanım
    --------
    >>> parser = DoclingPDFParser()
    >>> elements = list(parser.parse("spot-user-manual-en.pdf"))
    >>> print(f"{len(elements)} eleman çıkarıldı")
    """

    # Atlanacak etiketler (gürültü)
    SKIP_LABELS = {
        str(DocItemLabel.PAGE_HEADER),
        str(DocItemLabel.PAGE_FOOTER),
        str(DocItemLabel.FOOTNOTE),
    }

    def __init__(self, enable_ocr: bool = False, table_mode: str = "accurate"):
        """
        Parameters
        ----------
        enable_ocr : bool
            Taranan PDF'ler için OCR'ı etkinleştir.
            Bu PDF text-based olduğu için False yeterli.
        table_mode : str
            "accurate" → TableFormer ML modeli (daha iyi, daha yavaş)
            "fast"     → kural tabanlı (hızlı, basit tablolar için)
        """
        self._converter = self._build_converter(enable_ocr, table_mode)
        logger.info(
            f"DoclingPDFParser hazır | OCR={enable_ocr} | table_mode={table_mode}"
        )

    def _build_converter(
        self, enable_ocr: bool, table_mode: str
    ) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()

        # Tablo çıkarma
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if table_mode == "accurate"
            else TableFormerMode.FAST
        )

        # OCR (isteğe bağlı)
        pipeline_options.do_ocr = enable_ocr

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse(self, pdf_path: str | Path) -> Iterator[ParsedElement]:
        """
        PDF'i okur ve sıralı ParsedElement'ler üretir.

        Başlık zinciri (breadcrumb) her eleman için otomatik hesaplanır:
        eğer bir metin "Emergency Stop" bölümü altındaysa,
        breadcrumb = ["Safety Information", "Emergency Stop"] olur.
        Bu bağlamı chunk'larda korumak için kritik öneme sahiptir.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"PDF parse ediliyor: {pdf_path}")

        result = self._converter.convert(str(pdf_path))
        doc: DoclingDocument = result.document

        # Başlık hiyerarşisini takip et
        heading_stack: list[tuple[int, str]] = []  # (level, text)

        for idx, (element, _level) in enumerate(doc.iterate_items()):
            label = str(element.label)
            page = self._get_page(element)

            # Gürültüyü atla
            if label in self.SKIP_LABELS:
                continue

            # Metin içeriğini çıkar
            text = self._extract_text(element, doc)
            if not text or len(text.strip()) < 5:
                continue

            # Başlık seviyesini belirle
            heading_level = self._get_heading_level(element)

            # Breadcrumb'ı güncelle
            if heading_level is not None:
                # Bu bir başlık — stack'i güncelle
                heading_stack = [
                    (lvl, txt)
                    for lvl, txt in heading_stack
                    if lvl < heading_level
                ]
                heading_stack.append((heading_level, text.strip()))

            breadcrumb = [txt for _, txt in heading_stack]
            if heading_level is not None and breadcrumb:
                # Başlığın kendisini breadcrumb'dan çıkar
                breadcrumb = breadcrumb[:-1]

            element_id = f"p{page:03d}_e{idx:04d}"

            yield ParsedElement(
                element_id=element_id,
                label=label,
                text=text.strip(),
                page=page,
                heading_level=heading_level,
                breadcrumb=breadcrumb,
                metadata=self._get_metadata(element),
            )

        logger.success(f"Parse tamamlandı: {pdf_path.name}")

    def _extract_text(self, element, doc: DoclingDocument) -> str:
        """
        Eleman türüne göre metin çıkarır.
        Tablolar için Markdown formatı kullanır — bu sayede
        tablo semantiği (hangi değer hangi sütunda) korunur.
        """
        label = str(element.label)

        if label == str(DocItemLabel.TABLE):
            # Tabloyu Markdown'a çevir
            try:
                return element.export_to_markdown(doc)
            except TypeError:
                try:
                    return element.export_to_markdown()
                except Exception:
                    pass
            except Exception:
                pass

        # Standart text çıkarma
        try:
            return element.text or ""
        except AttributeError:
            pass

        try:
            return str(element.orig) if element.orig else ""
        except AttributeError:
            return ""

    def _get_page(self, element) -> int:
        """Elemanın sayfa numarasını güvenli şekilde alır."""
        try:
            prov = element.prov[0] if element.prov else None
            if prov and hasattr(prov, "page_no"):
                return prov.page_no
        except (IndexError, AttributeError):
            pass
        return 0

    def _get_heading_level(self, element) -> Optional[int]:
        """Başlık etiketlerinden seviye (1/2/3) çıkarır."""
        label = str(element.label)
        if label == str(DocItemLabel.SECTION_HEADER):
            # Docling level bilgisini doğrudan expose eder
            try:
                return element.level
            except AttributeError:
                return 1
        return None

    def _get_metadata(self, element) -> dict:
        """BBox ve diğer konum bilgilerini toplar."""
        meta = {}
        try:
            if element.prov:
                prov = element.prov[0]
                if hasattr(prov, "bbox"):
                    bbox = prov.bbox
                    meta["bbox"] = {
                        "l": bbox.l, "t": bbox.t,
                        "r": bbox.r, "b": bbox.b,
                    }
        except (IndexError, AttributeError):
            pass
        return meta

    def get_visual_pages_from_pdf(
        self,
        pdf_path: str | Path,
        elements: list,
        min_area_ratio: float = 0.10,
    ) -> list[dict]:
        """
        PyMuPDF ile PDF'deki anlamlı görselleri tespit eder.

        Docling'in sıfır picture/figure_caption ürettiği durumlarda
        fallback olarak çağrılır. Logo, ikon ve header gibi küçük dekoratif
        görselleri elemek için görsel alanının sayfa alanına oranı
        min_area_ratio eşiğinin üzerinde olmalıdır (varsayılan %5).

        Returns
        -------
        list[dict]
            [{page, figure_caption, breadcrumb, visual_type}, ...]
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF yüklü değil; görsel tespit atlandı. pip install pymupdf")
            return []

        pdf_path = Path(pdf_path)

        # Docling elementlerinden sayfa → breadcrumb haritası
        page_breadcrumb: dict[int, list[str]] = {}
        for el in elements:
            if el.heading_level is not None and el.page not in page_breadcrumb:
                page_breadcrumb[el.page] = el.breadcrumb or []

        # figure_caption'ları sayfa bazında topla
        page_captions: dict[int, str] = {}
        for el in elements:
            if el.label == "figure_caption" and el.text:
                existing = page_captions.get(el.page, "")
                page_captions[el.page] = (existing + " " + el.text).strip() if existing else el.text

        page_visuals: dict[int, dict] = {}

        try:
            doc = fitz.open(str(pdf_path))
            for page_idx in range(len(doc)):
                page_no = page_idx + 1  # 1-tabanlı
                page = doc[page_idx]
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                if page_area == 0:
                    continue

                images = page.get_images(full=True)
                best_ratio = 0.0

                for img in images:
                    xref = img[0]
                    try:
                        # Sayfadaki gerçek bbox boyutunu kullan (PDF koordinatları)
                        rects = page.get_image_rects(xref)
                        for r in rects:
                            ratio = (r.width * r.height) / page_area
                            if ratio > best_ratio:
                                best_ratio = ratio
                    except Exception:
                        continue

                if best_ratio >= min_area_ratio:
                    # En yakın önceki breadcrumb'ı bul
                    bc: list[str] = []
                    for p in range(page_no, 0, -1):
                        if p in page_breadcrumb:
                            bc = page_breadcrumb[p]
                            break

                    page_visuals[page_no] = {
                        "page": page_no,
                        "figure_caption": page_captions.get(page_no, ""),
                        "breadcrumb": bc,
                        "visual_type": "diagram",
                    }

            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF görsel tespiti başarısız: {e}")

        logger.info(f"PyMuPDF: {len(page_visuals)} görsel sayfa tespit edildi ({pdf_path.name})")
        return list(page_visuals.values())

    def export_json(self, pdf_path: str | Path, output_path: str | Path) -> None:
        """Parse sonuçlarını JSON dosyasına yazar (debug için)."""
        elements = list(self.parse(pdf_path))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in elements], f, ensure_ascii=False, indent=2)
        logger.info(f"JSON export: {output_path} ({len(elements)} eleman)")
