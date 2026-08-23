"""
Semantic + Hierarchical Chunker

Strateji
--------
1. Her ParsedElement bir "atom" olarak kabul edilir.
2. Atomlar bağlamsal olarak birleştirilir:
   - Aynı başlık altındaki kısa paragraflar birleştirilir
   - Büyük elemanlar (tablolar, uzun paragraflar) kendi başına chunk olur
   - Başlıklar bir sonraki içerikle mutlaka aynı chunk'ta bulunur
     (orphan heading problemi yok)
3. Her chunk'a şu metadata eklenir:
   - breadcrumb: başlık zinciri → retrieval'da bağlamı kaybetmemek için
   - source_elements: hangi atomlardan oluştuğu
   - page_range: [ilk_sayfa, son_sayfa]
   - chunk_type: "text" | "table" | "mixed"

Neden token-overlap kullanmak yerine bu yaklaşım?
- Sabit overlap, tablonun ortasında ya da madde listesinin ortasında
  kesilebilir; bu veri kaybıdır.
- Yapısal birleştirme, semantik bütünlüğü korur.
"""
from __future__ import annotations

import re
import uuid

# "1.", "2.3.", "5.6.2." gibi numaralı section başlıklarını tanır
_NUMBERED_SECTION_RE = re.compile(r'^\d+(\.\d+)*\.\s')
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from parsers.docling_parser import ParsedElement


@dataclass
class Chunk:
    """
    RAG pipeline'ına giren nihai chunk.

    Attributes
    ----------
    chunk_id : str
        UUID4 — MongoDB ve Qdrant'ta primary key.
    text : str
        LLM'e ve embedding modeline verilecek metin.
        Breadcrumb prefix dahil (bağlam için).
    raw_text : str
        Prefix olmadan ham metin (highlight için).
    breadcrumb : list[str]
        Bu chunk'ın doküman hiyerarşisindeki konumu.
    page_range : list[int]
        [başlangıç_sayfa, bitiş_sayfa]
    chunk_type : str
        "text" | "table" | "list" | "mixed"
    source_elements : list[str]
        Kaynak element_id'leri (izlenebilirlik için).
    doc_name : str
        PDF dosyasının adı.
    """
    chunk_id: str
    text: str
    raw_text: str
    breadcrumb: list[str]
    page_range: list[int]
    chunk_type: str
    source_elements: list[str]
    doc_name: str = "spot-user-manual-en"
    lang: str = "en"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "raw_text": self.raw_text,
            "breadcrumb": self.breadcrumb,
            "breadcrumb_str": " > ".join(self.breadcrumb),
            "page_range": self.page_range,
            "chunk_type": self.chunk_type,
            "doc_name": self.doc_name,
            "lang": self.lang,
        }


class SemanticChunker:
    """
    ParsedElement listesini Chunk listesine dönüştürür.

    Parameters
    ----------
    max_tokens : int
        Bir chunk'ın maksimum yaklaşık token sayısı.
        (Kelime sayısı × 1.3 heuristiği kullanılır — tam tokenizer
        gerektirmez, yeterince doğru.)
    overlap_tokens : int
        Sonraki chunk'a taşınan overlap miktarı.
        Sadece uzun paragraflar bölündüğünde uygulanır.
    min_tokens : int
        Bu değerin altındaki tek başına chunk'lar bir sonraki ile birleştirilir.
    doc_name : str
        Kaynak doküman adı.
    """

    WORDS_PER_TOKEN = 0.75  # yaklaşık: 1 token ≈ 0.75 kelime

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 64,
        doc_name: str = "spot-user-manual-en",
        lang: str = "en",
    ):
        self.max_words = int(max_tokens * self.WORDS_PER_TOKEN)
        self.overlap_words = int(overlap_tokens * self.WORDS_PER_TOKEN)
        self.min_words = int(min_tokens * self.WORDS_PER_TOKEN)
        self.doc_name = doc_name
        self.lang = lang
        logger.info(
            f"SemanticChunker | max_words≈{self.max_words} "
            f"overlap≈{self.overlap_words} min≈{self.min_words}"
        )

    def chunk(self, elements: list[ParsedElement]) -> list[Chunk]:
        """
        Ana giriş noktası.
        """
        # Adım 1: Tabloları ve uzun elemanları ayır
        groups = self._group_by_section(elements)

        # Adım 2: Her grubu chunk'lara böl
        chunks: list[Chunk] = []
        for group in groups:
            chunks.extend(self._group_to_chunks(group))

        # Adım 3: Çok küçük chunk'ları birleştir
        chunks = self._merge_small_chunks(chunks)

        logger.success(f"{len(elements)} element → {len(chunks)} chunk")
        return chunks

    # ─────────────────────────────────────────────────────────────────
    # Adım 1: Elemanları bölüm gruplarına ayır
    # ─────────────────────────────────────────────────────────────────

    def _group_by_section(
        self, elements: list[ParsedElement]
    ) -> list[list[ParsedElement]]:
        """
        Her heading seviyesinde yeni grup başlatır (H1, H2, H3, H4...).
        Her subsection kendi başına işlenir — içerik parçalanmaz.
        """
        groups: list[list[ParsedElement]] = []
        current: list[ParsedElement] = []

        for el in elements:
            is_structural = (
                el.heading_level is not None
                and _NUMBERED_SECTION_RE.match(el.text.strip())
            )
            if is_structural and current:
                groups.append(current)
                current = [el]
            else:
                current.append(el)

        if current:
            groups.append(current)

        return groups

    # ─────────────────────────────────────────────────────────────────
    # Adım 2: Grubu chunk'lara dönüştür
    # ─────────────────────────────────────────────────────────────────

    def _group_to_chunks(self, group: list[ParsedElement]) -> list[Chunk]:
        """
        Bir bölüm grubunu chunk'lara böler.

        Tablolar kendi başına chunk olur (tablo bütünlüğü).
        Metin ve liste öğeleri aynı chunk'ta tutulur — section içeriği
        parçalanmaz (ör. başlık + açıklama + madde listesi + spec tablosu
        aynı bölüme aitse birlikte kalır).
        """
        chunks: list[Chunk] = []
        buffer: list[ParsedElement] = []  # metin + liste birlikte

        def flush():
            if buffer:
                has_list = any(e.label == "list_item" for e in buffer)
                has_text = any(
                    e.label != "list_item" and e.heading_level is None
                    for e in buffer
                )
                if has_list and has_text:
                    ctype = "mixed"
                elif has_list:
                    ctype = "list"
                else:
                    ctype = "text"
                c = self._elements_to_chunk(buffer, ctype)
                if c:
                    chunks.extend(self._split_if_large(c))
                buffer.clear()

        for el in group:
            if el.label == "table":
                flush()
                c = self._elements_to_chunk([el], "table")
                if c:
                    chunks.append(c)
            else:
                buffer.append(el)

        flush()
        return chunks

    def _elements_to_chunk(
        self, elements: list[ParsedElement], chunk_type: str
    ) -> Optional[Chunk]:
        """ParsedElement listesinden tek bir Chunk oluşturur."""
        if not elements:
            return None

        # Breadcrumb: en son elemanın breadcrumb'ını kullan
        breadcrumb = elements[-1].breadcrumb or elements[0].breadcrumb

        # Sayfa aralığı
        pages = [e.page for e in elements if e.page > 0]
        page_range = [min(pages), max(pages)] if pages else [0, 0]

        # Ham metin
        raw_text = self._join_texts(elements)
        if not raw_text.strip():
            return None

        # Prefix ile zenginleştirilmiş metin (embedding kalitesi için)
        text = self._enrich_text(raw_text, breadcrumb, chunk_type)

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            raw_text=raw_text,
            breadcrumb=breadcrumb,
            page_range=page_range,
            chunk_type=chunk_type,
            source_elements=[e.element_id for e in elements],
            doc_name=self.doc_name,
            lang=self.lang,
        )

    def _join_texts(self, elements: list[ParsedElement]) -> str:
        """Elemanların metinlerini mantıklı şekilde birleştirir."""
        parts = []
        for el in elements:
            text = el.text.strip()
            if not text:
                continue
            if el.heading_level is not None:
                # Başlıkları belirgin yap
                prefix = "#" * el.heading_level
                parts.append(f"{prefix} {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)

    def _enrich_text(
        self, raw_text: str, breadcrumb: list[str], chunk_type: str
    ) -> str:
        """
        Chunk metnine bağlamsal prefix ekler.

        Örnek:
        "Section: Safety > Emergency Procedures\n\nPress the E-Stop button..."

        Bu prefix sayesinde embedding modeli, metnin hangi bölüme
        ait olduğunu bilir → retrieval doğruluğu artar.
        """
        if not breadcrumb:
            return raw_text

        prefix = "Section: " + " > ".join(breadcrumb)
        if chunk_type == "table":
            prefix = "Table in: " + " > ".join(breadcrumb)

        return f"{prefix}\n\n{raw_text}"

    # ─────────────────────────────────────────────────────────────────
    # Adım 2b: Büyük chunk'ları böl (overlap ile)
    # ─────────────────────────────────────────────────────────────────

    def _split_if_large(self, chunk: Chunk) -> list[Chunk]:
        """
        Eğer chunk max_words'ü aşıyorsa, cümle sınırlarından böler.
        Overlap uygulanır — bağlam kaybı olmaz.
        """
        words = chunk.raw_text.split()
        if len(words) <= self.max_words:
            return [chunk]

        logger.debug(
            f"Büyük chunk bölünüyor: {len(words)} kelime "
            f"(max={self.max_words})"
        )

        # Cümlelere böl
        sentences = re.split(r'(?<=[.!?])\s+', chunk.raw_text)

        sub_chunks: list[Chunk] = []
        buffer_sentences: list[str] = []
        buffer_words = 0
        overlap_sentences: list[str] = []

        for sent in sentences:
            sent_words = len(sent.split())

            if buffer_words + sent_words > self.max_words and buffer_sentences:
                # Mevcut buffer'dan chunk oluştur
                raw = " ".join(buffer_sentences)
                sub = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=self._enrich_text(raw, chunk.breadcrumb, chunk.chunk_type),
                    raw_text=raw,
                    breadcrumb=chunk.breadcrumb,
                    page_range=chunk.page_range,
                    chunk_type=chunk.chunk_type,
                    source_elements=chunk.source_elements,
                    doc_name=chunk.doc_name,
                    lang=chunk.lang,
                )
                sub_chunks.append(sub)

                # Overlap: son N kelimeyi yeni buffer'a taşı
                overlap_sentences = self._compute_overlap(
                    buffer_sentences, self.overlap_words
                )
                buffer_sentences = overlap_sentences + [sent]
                buffer_words = sum(len(s.split()) for s in buffer_sentences)
            else:
                buffer_sentences.append(sent)
                buffer_words += sent_words

        # Kalan buffer
        if buffer_sentences:
            raw = " ".join(buffer_sentences)
            sub = Chunk(
                chunk_id=str(uuid.uuid4()),
                text=self._enrich_text(raw, chunk.breadcrumb, chunk.chunk_type),
                raw_text=raw,
                breadcrumb=chunk.breadcrumb,
                page_range=chunk.page_range,
                chunk_type=chunk.chunk_type,
                source_elements=chunk.source_elements,
                doc_name=chunk.doc_name,
                lang=chunk.lang,
            )
            sub_chunks.append(sub)

        return sub_chunks

    def _compute_overlap(
        self, sentences: list[str], overlap_words: int
    ) -> list[str]:
        """Son N kelimeye karşılık gelen cümleleri döndürür."""
        result: list[str] = []
        count = 0
        for sent in reversed(sentences):
            w = len(sent.split())
            if count + w > overlap_words:
                break
            result.insert(0, sent)
            count += w
        return result

    # ─────────────────────────────────────────────────────────────────
    # Adım 3: Küçük chunk'ları birleştir
    # ─────────────────────────────────────────────────────────────────

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        min_words'ün altındaki chunk'ları birleştirir.
        - Tablo ve list chunk'larına dokunulmaz.
        - Çok küçük chunk'lar (<10 kelime) breadcrumb farkı olsa da birleşir.
        - Birleşemeyen çok kısa chunk'lar (<8 kelime) sonunda atılır.
        """
        if not chunks:
            return chunks

        TINY = 8  # Bu kadar kelimeden az olan chunk retrieval için işe yaramaz

        merged: list[Chunk] = [chunks[0]]

        for curr in chunks[1:]:
            prev = merged[-1]
            prev_words = len(prev.raw_text.split())
            curr_words = len(curr.raw_text.split())

            # Çok küçük chunk'larda bölüm sınırını yoksay
            breadcrumb_ok = (
                prev.breadcrumb == curr.breadcrumb
                or curr_words < TINY
            )

            can_merge = (
                curr_words < self.min_words
                and curr.chunk_type not in ("table", "list")
                and prev.chunk_type not in ("table", "list")
                and prev_words + curr_words <= self.max_words * 1.1
                and breadcrumb_ok
            )

            if can_merge:
                combined_raw = prev.raw_text + "\n\n" + curr.raw_text
                merged[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    text=self._enrich_text(
                        combined_raw, prev.breadcrumb, prev.chunk_type
                    ),
                    raw_text=combined_raw,
                    breadcrumb=prev.breadcrumb,
                    page_range=[prev.page_range[0], curr.page_range[1]],
                    chunk_type=prev.chunk_type,
                    source_elements=prev.source_elements + curr.source_elements,
                    doc_name=prev.doc_name,
                    lang=prev.lang,
                )
            else:
                merged.append(curr)

        before = len(merged)
        # Merge sonrası hâlâ çok kısa kalan chunk'ları at
        merged = [c for c in merged if
                  len(c.raw_text.split()) >= TINY
                  or c.chunk_type in ("table", "list")]
        dropped = before - len(merged)
        if dropped:
            logger.debug(f"  {dropped} kısa chunk atıldı (<{TINY} kelime)")

        logger.debug(f"Küçük chunk birleştirme: {len(chunks)} → {len(merged)}")
        return merged
