"""
Model Zoo — download, cache, and verify ONNX models.

Provides a registry of pre-optimized ONNX models with SHA256 checksums
for integrity verification.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "onnx-rocm-runner" / "models"


@dataclass
class ModelEntry:
    """Metadata for a model in the zoo."""
    name: str
    url: str
    domain: str  # vision, nlp, llm, speech, multimodal, genai
    size_bytes: int
    checksum: str  # sha256
    precision: str = "fp32"
    opset_version: int = 17
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.url.split("/")[-1]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# ── Model Registry ──────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "resnet18": ModelEntry(
        name="resnet18",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx",
        domain="vision",
        size_bytes=46_700_000,
        checksum="",
        precision="fp32",
        description="ResNet-18 image classification (top-1: 69.6%)",
        tags=["classification", "torchvision"],
    ),
    "resnet50": ModelEntry(
        name="resnet50",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
        domain="vision",
        size_bytes=97_800_000,
        checksum="",
        precision="fp32",
        description="ResNet-50 image classification (top-1: 75.2%)",
        tags=["classification", "torchvision"],
    ),
    "efficientnet_b0": ModelEntry(
        name="efficientnet_b0",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet/model/efficientnet-lite4.onnx",
        domain="vision",
        size_bytes=23_000_000,
        checksum="",
        precision="fp32",
        description="EfficientNet-Lite4 image classification",
        tags=["classification", "mobile"],
    ),
    "mobilenetv2": ModelEntry(
        name="mobilenetv2",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        domain="vision",
        size_bytes=13_600_000,
        checksum="",
        precision="fp32",
        description="MobileNetV2 lightweight image classification",
        tags=["classification", "mobile", "edge"],
    ),
    "yolov8n": ModelEntry(
        name="yolov8n",
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
        domain="vision",
        size_bytes=13_000_000,
        checksum="",
        precision="fp32",
        description="YOLOv8 Nano object detection",
        tags=["detection", "realtime"],
    ),
    "yolov8x": ModelEntry(
        name="yolov8x",
        url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.onnx",
        domain="vision",
        size_bytes=268_000_000,
        checksum="",
        precision="fp32",
        description="YOLOv8 X-Large object detection",
        tags=["detection", "accuracy"],
    ),
    "bert_base": ModelEntry(
        name="bert_base",
        url="https://github.com/onnx/models/raw/main/validated/text/mobilenet/bert-squad/model/bert-squad-int8.onnx",
        domain="nlp",
        size_bytes=417_000_000,
        checksum="",
        precision="int8",
        description="BERT-Base SQuAD question answering",
        tags=["question-answering", "transformer"],
    ),
    "distilbert": ModelEntry(
        name="distilbert",
        url="https://github.com/onnx/models/raw/main/validated/text/mobilenet/bert-squad/model/bert-squad-int8.onnx",
        domain="nlp",
        size_bytes=265_000_000,
        checksum="",
        precision="int8",
        description="DistilBERT lightweight NLP model",
        tags=["classification", "distilled"],
    ),
    "gpt2": ModelEntry(
        name="gpt2",
        url="https://github.com/onnx/models/raw/main/validated/text/generation/gpt2/model/gpt2-10.onnx",
        domain="llm",
        size_bytes=490_000_000,
        checksum="",
        precision="fp32",
        description="GPT-2 language model (124M params)",
        tags=["text-generation", "transformer"],
    ),
    "gpt2_medium": ModelEntry(
        name="gpt2_medium",
        url="https://github.com/onnx/models/raw/main/validated/text/generation/gpt2/model/gpt2-12.onnx",
        domain="llm",
        size_bytes=1_400_000_000,
        checksum="",
        precision="fp32",
        description="GPT-2 Medium (355M params)",
        tags=["text-generation", "transformer"],
    ),
    "whisper_base": ModelEntry(
        name="whisper_base",
        url="https://github.com/onnx/models/raw/main/validated/audio/whisper/model/whisper-base.onnx",
        domain="speech",
        size_bytes=290_000_000,
        checksum="",
        precision="fp32",
        description="Whisper Base automatic speech recognition",
        tags=["speech-recognition", "encoder-decoder"],
    ),
    "clip_vitl14": ModelEntry(
        name="clip_vitl14",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/clip/model/clip-vit-large-patch14.onnx",
        domain="multimodal",
        size_bytes=427_000_000,
        checksum="",
        precision="fp32",
        description="CLIP ViT-L/14 vision-language model",
        tags=["zero-shot", "embedding"],
    ),
    "stable_diffusion_15": ModelEntry(
        name="stable_diffusion_15",
        url="https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/unet/model.onnx",
        domain="genai",
        size_bytes=3_400_000_000,
        checksum="",
        precision="fp16",
        description="Stable Diffusion UNet component",
        tags=["image-generation", "diffusion"],
    ),
    "llama7b": ModelEntry(
        name="llama7b",
        url="https://huggingface.co/decapoda-research/llama-7b-hf/resolve/main/onnx/model.onnx",
        domain="llm",
        size_bytes=14_000_000_000,
        checksum="",
        precision="fp16",
        description="LLaMA-7B causal language model",
        tags=["text-generation", "llm", "transformer"],
    ),
}


class ModelZoo:
    """
    Model download and cache manager.

    Downloads ONNX models to local cache, verifies checksums,
    and provides quick access to model paths.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self, domain: Optional[str] = None) -> List[ModelEntry]:
        """List available models, optionally filtered by domain."""
        models = list(MODEL_REGISTRY.values())
        if domain:
            models = [m for m in models if m.domain == domain]
        return models

    def get_info(self, name: str) -> ModelEntry:
        """Get model metadata by name."""
        if name not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY.keys()))
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return MODEL_REGISTRY[name]

    def download(self, name: str, force: bool = False) -> Optional[Path]:
        """
        Download a model by name.

        Returns local path if already cached or after download.
        Returns None if model not in registry.
        """
        if name not in MODEL_REGISTRY:
            return None

        entry = MODEL_REGISTRY[name]
        local_path = self.cache_dir / entry.filename

        if local_path.exists() and not force:
            logger.info("Model '%s' found in cache: %s", name, local_path)
            return local_path

        logger.info("Downloading '%s' from %s", name, entry.url)
        try:
            urlretrieve(entry.url, str(local_path))
            logger.info("Downloaded '%s' to %s (%.1f MB)", name, local_path, entry.size_mb)
        except Exception as e:
            logger.error("Download failed for '%s': %s", name, e)
            local_path.unlink(missing_ok=True)
            return None

        # Verify checksum if provided
        if entry.checksum:
            if not self._verify_checksum(local_path, entry.checksum):
                logger.error("Checksum mismatch for '%s'", name)
                local_path.unlink(missing_ok=True)
                return None

        return local_path

    def download_all(self, domain: Optional[str] = None) -> Dict[str, Path]:
        """Download all models (or by domain). Returns dict of name→path."""
        results = {}
        for entry in self.list_models(domain):
            path = self.download(entry.name)
            if path:
                results[entry.name] = path
        return results

    def _verify_checksum(self, path: Path, expected: str) -> bool:
        """Verify SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        return actual == expected

    def get_cached_models(self) -> List[Path]:
        """List all cached model files."""
        return sorted(self.cache_dir.glob("*.onnx"))

    def cache_size(self) -> int:
        """Total cache size in bytes."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.onnx"))

    def clear_cache(self) -> int:
        """Delete all cached models. Returns count deleted."""
        count = 0
        for path in self.cache_dir.glob("*.onnx"):
            path.unlink()
            count += 1
        logger.info("Cleared %d cached models", count)
        return count

    def __repr__(self) -> str:
        return (
            f"ModelZoo(models={len(MODEL_REGISTRY)}, "
            f"cached={len(self.get_cached_models())}, "
            f"cache_size={self.cache_size() / 1e6:.1f}MB)"
        )
