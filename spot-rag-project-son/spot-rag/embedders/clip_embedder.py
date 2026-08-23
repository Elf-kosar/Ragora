"""
CLIP Embedder — görsel sorgu ve görsel chunk embedding.

Neden CLIP?
-----------
Kullanıcı bir görsel gönderdiğinde iki şey yapmamız gerekir:
1. Görseli embed et → Qdrant'ta görsel chunk'larla karşılaştır
2. Görsel chunk'ların VLM açıklamalarını da embed et → metin sorgularla karşılaştır

CLIP bunu unified bir vektör uzayında yapar:
- image_embed(kedi fotoğrafı) ≈ text_embed("kedi")
- image_embed(E-Stop diyagramı) ≈ text_embed("E-Stop button location")

Bu sayede:
- Kullanıcı metin yazarsa → text embed → görsel chunk'ları da bulur
- Kullanıcı görsel gönderirse → image embed → ilgili görsel chunk'ları bulur
- Çapraz modal retrieval: "Bu görselde ne var?" + görsel → hem metin hem görsel chunk

Model
-----
openai/clip-vit-large-patch14  → 768 dim, en iyi kalite
openai/clip-vit-base-patch32   → 512 dim, daha hızlı
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image


class CLIPEmbedder:
    """
    CLIP ile hem metin hem görsel embed eder.

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model adı.
    device : str
        "cuda" | "cpu"
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

        logger.info(f"CLIP yükleniyor: {model_name} @ {device}")
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(model_name).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

        # Embedding boyutunu al
        self.dimension = self._model.config.projection_dim
        logger.success(f"CLIP hazır | dim={self.dimension}")

    # ─────────────────────────────────────────────────────────────────
    # Metin embedding (görsel chunk'ların searchable_text'i için)
    # ─────────────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Metinleri CLIP metin encoder ile embed eder.
        Görsel chunk'ların VLM açıklamalarını indexlerken kullanılır.

        Returns
        -------
        np.ndarray  shape: (len(texts), dimension), dtype=float32
        """
        import torch

        all_vecs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # CLIP'in max token limiti 77 — uzun metinleri kırp
            batch = [t[:500] for t in batch]

            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            with torch.no_grad():
                vecs = self._model.get_text_features(**inputs)
                if not isinstance(vecs, torch.Tensor):
                    vecs = vecs.pooler_output if hasattr(vecs, "pooler_output") else vecs[0]
                vecs = vecs / vecs.norm(dim=-1, keepdim=True)  # normalize

            all_vecs.append(vecs.cpu().numpy())

        return np.vstack(all_vecs).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    # Görsel embedding (sorgu görseli veya indexleme için)
    # ─────────────────────────────────────────────────────────────────

    def embed_image_b64(self, image_b64: str) -> np.ndarray:
        """
        Base64 encoded görüntüyü embed eder.
        Kullanıcı sorgu görseli gönderdiğinde çağrılır.

        Returns
        -------
        np.ndarray  shape: (dimension,), dtype=float32
        """
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self._embed_pil(image)

    def embed_image_path(self, image_path: str | Path) -> np.ndarray:
        """
        Dosya yolundan görüntüyü embed eder.

        Returns
        -------
        np.ndarray  shape: (dimension,), dtype=float32
        """
        image = Image.open(image_path).convert("RGB")
        return self._embed_pil(image)

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Ham bytes'tan görüntüyü embed eder.

        Returns
        -------
        np.ndarray  shape: (dimension,), dtype=float32
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self._embed_pil(image)

    def _embed_pil(self, image: Image.Image) -> np.ndarray:
        """PIL Image → normalized CLIP embedding."""
        import torch

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vec = self._model.get_image_features(**inputs)
            if not isinstance(vec, torch.Tensor):
                vec = vec.pooler_output if hasattr(vec, "pooler_output") else vec[0]
            vec = vec / vec.norm(dim=-1, keepdim=True)

        return vec.squeeze(0).cpu().numpy().astype(np.float32)

    # ─────────────────────────────────────────────────────────────────
    # Metin ile görsel karşılaştırma (similarity score)
    # ─────────────────────────────────────────────────────────────────

    def text_image_similarity(
        self, text: str, image_b64: str
    ) -> float:
        """
        Metin ile görüntü arasındaki CLIP cosine similarity'i döndürür.
        0.0 - 1.0 arası. ~0.25+ = ilgili.

        Kullanım: retrieve sonuçlarını re-rank etmek için.
        """
        text_vec = self.embed_texts([text])[0]
        image_vec = self.embed_image_b64(image_b64)
        return float(np.dot(text_vec, image_vec))
