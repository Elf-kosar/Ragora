"""
Local Embedder — çok dilli mE5-large ile GPU embedding.

Model seçimi
------------
intfloat/multilingual-e5-large  (varsayılan)
  - 1024 dim, 560M parametre
  - 100+ dil: TR, DE, KO, AR, JA, ES, FR, PT, EN, ...
  - "Türkçe soru" → aynı vektör uzayında "German answer" bulunur
  - MTEB multilingual benchmark'ta SOTA (2024)

intfloat/multilingual-e5-base   (daha hızlı, düşük VRAM)
  - 768 dim, 278M parametre

BAAI/bge-large-en-v1.5          (sadece İngilizce, en yüksek EN skoru)
  - 1024 dim

mE5 prefix kuralı
-----------------
- Passage (indexleme): "passage: {text}"
- Query   (sorgulama): "query: {text}"
Bu prefix OLMADAN mE5 doğru çalışmaz. Kod otomatik ekler.
BGE için farklı prefix var; model adı "bge" içeriyorsa otomatik geçiş yapılır.
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class LocalEmbedder:
    """
    SentenceTransformers ile batch embedding — mE5 ve BGE prefix yönetimi dahil.

    Parameters
    ----------
    model_name : str
    device : str  "cuda" | "cpu" | "mps"
    batch_size : int
    """

    # mE5 prefix'leri
    ME5_QUERY_PREFIX   = "query: "
    ME5_PASSAGE_PREFIX = "passage: "

    # BGE prefix'i (sadece query için)
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        name_lower = model_name.lower()
        self._is_me5 = "multilingual-e5" in name_lower or ("e5" in name_lower and "bge" not in name_lower)
        self._is_bge = "bge" in name_lower

        logger.info(f"Embedding modeli yükleniyor: {model_name} @ {device}")
        logger.info(f"  Tip: {'mE5 (çok dilli)' if self._is_me5 else 'BGE (EN)' if self._is_bge else 'genel'}")
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.success(f"Model hazır | dim={self.dimension}")

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """
        Chunk metinlerini (passage) embed eder.
        mE5 için "passage: " prefix'i otomatik eklenir.

        Returns
        -------
        np.ndarray  shape: (len(texts), dimension), float32
        """
        logger.info(f"{len(texts)} chunk embed ediliyor...")
        if self._is_me5:
            texts = [self.ME5_PASSAGE_PREFIX + t for t in texts]
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Tek sorgu embed eder.
        mE5 için "query: " prefix, BGE için kendi prefix otomatik eklenir.
        Dil fark etmez — mE5 tüm dilleri aynı uzayda işler.

        Returns
        -------
        np.ndarray  shape: (dimension,), float32
        """
        if self._is_me5:
            query = self.ME5_QUERY_PREFIX + query
        elif self._is_bge:
            query = self.BGE_QUERY_PREFIX + query

        vec = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def _encode(self, texts: list[str]) -> np.ndarray:
        all_vecs = []
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Embedding",
            unit="batch",
        ):
            batch = texts[i : i + self.batch_size]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            all_vecs.append(vecs)
        return np.vstack(all_vecs).astype(np.float32)
