"""
Visual Parser — LLaVA-1.6-34B ile görsel sayfa işleme.

Model
-----
llava-hf/llava-v1.6-34b-hf  → 34B parametre, bfloat16 ~68GB VRAM
H200 141GB VRAM'de sorunsuz çalışır.

LLaVA-1.6 (llava-next), önceki versiyonlara göre:
- Yüksek çözünürlük desteği (anyres): sayfayı 4 tile'a böler,
  her tile'ı ayrı encode eder → küçük metin ve etiketleri daha iyi okur
- Teknik diyagram anlama benchmark'larında belirgin iyileşme
- Tablo hücre değerlerini daha doğru çıkarır

Prompt stratejisi
-----------------
LLaVA-1.6 için <image> token'ı processor tarafından otomatik eklenir,
chat template uygulanır. max_new_tokens=1024 ile daha detaylı açıklama
alınır (8B versiyonunda 512 yeterliydi, 34B daha uzun çıktı üretir).

DPI
---
200 DPI kullanılır (150 yerine). LLaVA-1.6'nın anyres özelliği
yüksek çözünürlükten fayda sağlar — küçük etiketler (E-STOP¹, vb.)
daha iyi okunur.
"""
from __future__ import annotations

import base64
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


@dataclass
class VisualElement:
    """
    Bir PDF sayfasındaki görsel içerik.

    Attributes
    ----------
    element_id : str
    page : int
    figure_caption : str
        Docling'in tespit ettiği caption metni.
    vlm_description : str
        LLaVA-1.6-34B'nin ürettiği detaylı açıklama.
    image_b64 : str
        Sayfanın base64 JPEG'i — multimodal LLM'e doğrudan verilebilir.
    breadcrumb : list[str]
    doc_name : str
    visual_type : str
        "diagram" | "photo" | "table_image" | "warning_box" | "unknown"
    """
    element_id: str
    page: int
    figure_caption: str
    vlm_description: str
    image_b64: str
    breadcrumb: list[str]
    doc_name: str
    visual_type: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        """Embedding ve full-text search için birleşik metin."""
        parts = []
        if self.breadcrumb:
            parts.append("Section: " + " > ".join(self.breadcrumb))
        if self.figure_caption:
            parts.append(f"Figure: {self.figure_caption}")
        if self.vlm_description:
            parts.append(self.vlm_description)
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "page": self.page,
            "figure_caption": self.figure_caption,
            "vlm_description": self.vlm_description,
            "image_b64": self.image_b64,
            "breadcrumb": self.breadcrumb,
            "breadcrumb_str": " > ".join(self.breadcrumb),
            "doc_name": self.doc_name,
            "visual_type": self.visual_type,
            "searchable_text": self.searchable_text,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model yükleyiciler — her model için ayrı fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def _load_llava(model_name: str, device: str):
    """
    LLaVA-1.5 / LLaVA-1.6 (llava-next) yükler.
    llava-hf/llava-v1.6-34b-hf için kullanılır.
    """
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    import torch

    logger.info(f"LLaVA yükleniyor: {model_name}")
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # H200 tek GPU'ya otomatik map
        low_cpu_mem_usage=True,
    ).eval()
    logger.success(f"LLaVA hazır: {model_name}")
    return model, processor, "llava"


def _load_internvl(model_name: str, device: str):
    """InternVL2 yükler."""
    from transformers import AutoModel, AutoProcessor
    import torch

    logger.info(f"InternVL yükleniyor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    logger.success(f"InternVL hazır: {model_name}")
    return model, processor, "internvl"


def _load_qwen(model_name: str, device: str):
    """Qwen2-VL yükler."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch

    logger.info(f"Qwen2-VL yükleniyor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    logger.success(f"Qwen2-VL hazır: {model_name}")
    return model, processor, "qwen"


def _detect_model_family(model_name: str) -> str:
    """Model adından hangi aileye ait olduğunu belirler."""
    name = model_name.lower()
    if "llava" in name:
        return "llava"
    if "internvl" in name or "intern_vl" in name:
        return "internvl"
    if "qwen" in name:
        return "qwen"
    # Fallback: llava-style AutoProcessor dene
    return "llava"


# ─────────────────────────────────────────────────────────────────────────────
# Ana parser sınıfı
# ─────────────────────────────────────────────────────────────────────────────

class VisualParser:
    """
    Görsel içeren PDF sayfalarını LLaVA-1.6-34B ile işler.

    Parameters
    ----------
    pdf_path : Path
        İşlenecek PDF.
    vlm_model_name : str | None
        HuggingFace model adı.
        None → VLM atlanır, sadece figure_caption kaydedilir.
        Default: "llava-hf/llava-v1.6-34b-hf"
    device : str
        "cuda" | "cpu"
    dpi : int
        Rasterizasyon kalitesi.
        200 DPI — LLaVA-1.6 anyres için önerilir.
    max_new_tokens : int
        VLM çıktı uzunluğu. 34B model için 1024 önerilir.
    """

    # LLaVA-1.6 için chat template formatı
    # <image> token'ı processor tarafından otomatik embed edilir
    LLAVA_PROMPT_TEMPLATE = (
        "[INST] <image>\n"
        "This is a page from a Boston Dynamics Spot robot technical manual. "
        "Describe this image in detail:\n"
        "- If it's a labeled diagram: list ALL part names, arrows, callouts, "
        "and their spatial relationships (e.g. 'E-STOP button is located at "
        "the top-rear of the body')\n"
        "- If it's a photo of the robot: describe the robot's pose, visible "
        "components, and any highlighted areas\n"
        "- If it's a WARNING/CAUTION/NOTE box: transcribe the EXACT text "
        "including the severity label\n"
        "- If it's a table rendered as an image: extract ALL cell values "
        "preserving row/column structure\n"
        "- If it contains multiple elements: describe each separately\n"
        "Be exhaustive. Include every visible text label, number, and symbol. "
        "[/INST]"
    )

    # InternVL2 / Qwen için daha kısa prompt (chat template farklı)
    GENERIC_PROMPT = (
        "This is a page from a Boston Dynamics Spot robot technical manual. "
        "Describe this image in detail. "
        "If it's a diagram: list ALL labeled parts and their relationships. "
        "If it's a warning box: transcribe the exact text. "
        "If it's a table: extract all cell values. "
        "Include every visible text label."
    )

    def __init__(
        self,
        pdf_path: str | Path,
        vlm_model_name: str | None = "llava-hf/llava-v1.6-34b-hf",
        device: str = "cuda",
        dpi: int = 200,
        max_new_tokens: int = 1024,
        ollama_host: str | None = None,
        ollama_vision_model: str | None = None,
    ):
        self.pdf_path = Path(pdf_path)
        self.device = device
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._model_family = None

        # Ollama tabanlı VLM (HuggingFace yerine tercih edilir — daha az VRAM)
        self._ollama_host = ollama_host
        self._ollama_vision_model = ollama_vision_model

        if ollama_host and ollama_vision_model:
            logger.info(f"Ollama VLM kullanılıyor: {ollama_vision_model} @ {ollama_host}")
        elif vlm_model_name:
            self._load_model(vlm_model_name)

    def _load_model(self, model_name: str) -> None:
        """Model ailesini tespit edip uygun yükleyiciyi çağırır."""
        family = _detect_model_family(model_name)
        try:
            if family == "llava":
                self._model, self._processor, self._model_family = _load_llava(model_name, self.device)
            elif family == "internvl":
                self._model, self._processor, self._model_family = _load_internvl(model_name, self.device)
            elif family == "qwen":
                self._model, self._processor, self._model_family = _load_qwen(model_name, self.device)
        except Exception as e:
            logger.warning(f"VLM yüklenemedi: {e} — görsel açıklama atlanacak")
            self._model = None

    # ─────────────────────────────────────────────────────────────────
    # Ana işleme metodu
    # ─────────────────────────────────────────────────────────────────

    def process_visual_pages(
        self,
        visual_pages: list[dict],
        doc_name: str,
    ) -> list[VisualElement]:
        """
        Docling'den gelen görsel sayfa bilgilerini işler.

        Parameters
        ----------
        visual_pages : list[dict]
            [{page, figure_caption, breadcrumb, visual_type}, ...]
        doc_name : str
        """
        if not visual_pages:
            return []

        logger.info(f"{len(visual_pages)} figür işleniyor (DPI={self.dpi})...")
        results = []

        for i, info in enumerate(visual_pages, 1):
            page_num = info["page"]
            fig_idx = info.get("figure_idx", 0)
            logger.info(f"  [{i}/{len(visual_pages)}] p.{page_num} fig#{fig_idx} render + VLM...")
            try:
                el = self._process_page(
                    page_num=page_num,
                    figure_caption=info.get("figure_caption", ""),
                    breadcrumb=info.get("breadcrumb", []),
                    doc_name=doc_name,
                    visual_type=info.get("visual_type", "unknown"),
                    bbox=info.get("bbox"),
                    figure_idx=fig_idx,
                )
                results.append(el)
                desc_preview = el.vlm_description[:60].replace("\n", " ") if el.vlm_description else "(açıklama yok)"
                logger.info(f"  ✓ p.{page_num} fig#{fig_idx} → {desc_preview}...")
            except Exception as e:
                logger.error(f"  ✗ p.{page_num} fig#{fig_idx}: {e}")

        logger.success(f"{len(results)}/{len(visual_pages)} figür işlendi")
        return results

    def _process_page(
        self,
        page_num: int,
        figure_caption: str,
        breadcrumb: list[str],
        doc_name: str,
        visual_type: str,
        bbox: dict | None = None,
        figure_idx: int = 0,
    ) -> VisualElement:
        # 1. Render — crop bölgesi varsa sadece o bölge, yoksa tam sayfa
        image_b64 = self._extract_figure_image(page_num, bbox)

        # 2. VLM için geçici dosya (HuggingFace modeli path bekliyorsa)
        tmp_img: Path | None = None
        if self._model is not None:
            tmp_img = Path(tempfile.mktemp(suffix=".jpg"))
            with open(tmp_img, "wb") as f:
                f.write(base64.b64decode(image_b64))

        # 3. VLM açıklama
        if self._ollama_host and self._ollama_vision_model:
            vlm_description = self._describe_ollama(image_b64)
        elif self._model is not None and tmp_img:
            vlm_description = self._describe(tmp_img)
        else:
            vlm_description = f"Caption: {figure_caption}" if figure_caption else ""

        # 4. Tür tespiti
        if visual_type == "unknown":
            visual_type = self._infer_type(figure_caption, vlm_description)

        return VisualElement(
            element_id=str(uuid.uuid4()),
            page=page_num,
            figure_caption=figure_caption,
            vlm_description=vlm_description,
            image_b64=image_b64,
            breadcrumb=breadcrumb,
            doc_name=doc_name,
            visual_type=visual_type,
            metadata={
                "dpi": self.dpi,
                "vlm_family": self._model_family or "ollama",
                "max_new_tokens": self.max_new_tokens,
                "bbox": bbox,
                "figure_idx": figure_idx,
            },
        )

    # ─────────────────────────────────────────────────────────────────
    # Rasterizasyon
    # ─────────────────────────────────────────────────────────────────

    def _extract_figure_image(self, page_num: int, bbox: dict | None) -> str:
        """
        PyMuPDF ile sayfanın belirli bir bölgesini (veya tamamını) JPEG'e dönüştürür.

        Parameters
        ----------
        page_num : int
            1-tabanlı sayfa numarası.
        bbox : dict | None
            {"x0", "y0", "x1", "y1"} PDF koordinatlarında crop bölgesi.
            None → tam sayfa render edilir.

        Returns
        -------
        str
            Base64 JPEG string.
        """
        import fitz

        doc = fitz.open(str(self.pdf_path))
        try:
            page = doc[page_num - 1]
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            if bbox:
                clip = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
                padding = 8  # PDF points
                clip = fitz.Rect(
                    max(page.rect.x0, clip.x0 - padding),
                    max(page.rect.y0, clip.y0 - padding),
                    min(page.rect.x1, clip.x1 + padding),
                    min(page.rect.y1, clip.y1 + padding),
                )
                pix = page.get_pixmap(matrix=mat, clip=clip, colorspace=fitz.csRGB)
            else:
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

            img_bytes = pix.tobytes("jpeg", jpg_quality=88)
            return base64.b64encode(img_bytes).decode("utf-8")
        finally:
            doc.close()

    # ─────────────────────────────────────────────────────────────────
    # VLM dispatch
    # ─────────────────────────────────────────────────────────────────

    def _describe(self, image_path: Path) -> str:
        """Model ailesine göre doğru inference metodunu çağırır."""
        try:
            if self._model_family == "llava":
                return self._describe_llava(image_path)
            elif self._model_family == "internvl":
                return self._describe_internvl(image_path)
            elif self._model_family == "qwen":
                return self._describe_qwen(image_path)
            return ""
        except Exception as e:
            logger.warning(f"VLM inference hatası: {e}")
            return ""
        finally:
            # Geçici dosyayı temizle
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _describe_llava(self, image_path: Path) -> str:
        """
        LLaVA-1.6 (llava-next) inference.

        LlavaNextProcessor görüntüyü anyres modunda işler:
        - Orijinal görüntü 1 kez encode edilir
        - Yüksek çözünürlük için 2x2 tile grid oluşturulur (4 tile)
        - Toplam 5 görüntü patch encode edilir
        Bu sayede 200 DPI'da A4 sayfasındaki küçük etiketler okunabilir.
        """
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        # LlavaNextProcessor chat template uygular
        inputs = self._processor(
            text=self.LLAVA_PROMPT_TEMPLATE,
            images=image,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,       # do_sample=False ile etkisiz ama açık
                repetition_penalty=1.1, # tekrar eden listeleri önler
            )

        # Sadece yeni üretilen token'ları decode et
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _describe_internvl(self, image_path: Path) -> str:
        """InternVL2 inference."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=self.GENERIC_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        return self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _describe_qwen(self, image_path: Path) -> str:
        """Qwen2-VL inference."""
        from PIL import Image
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self.GENERIC_PROMPT},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image = Image.open(image_path)
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        return self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    # ─────────────────────────────────────────────────────────────────
    # Yardımcı
    # ─────────────────────────────────────────────────────────────────

    def _describe_ollama(self, image_b64: str) -> str:
        """Ollama /api/generate ile görsel açıklama (VRAM dostu)."""
        try:
            payload = {
                "model": self._ollama_vision_model,
                "prompt": self.GENERIC_PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {"num_predict": self.max_new_tokens},
            }
            r = requests.post(
                f"{self._ollama_host}/api/generate",
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama VLM hatası: {e}")
            return ""

    def _infer_type(self, caption: str, description: str) -> str:
        combined = (caption + " " + description).lower()
        if any(w in combined for w in ["warning", "caution", "danger", "note"]):
            return "warning_box"
        if any(w in combined for w in ["table", "row", "column", "specification", "value"]):
            return "table_image"
        if any(w in combined for w in ["pose", "stand", "sit", "crouching", "the robot is"]):
            return "photo"
        return "diagram"
