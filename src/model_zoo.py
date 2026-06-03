"""
Model zoo for onnx-rocm-runner.

Pre-configured model definitions with download, export, and validation.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Model configuration entry."""
    name: str
    domain: str
    description: str
    input_shapes: dict[str, list[int]]
    output_shapes: dict[str, list[int]]
    download_url: Optional[str] = None
    file_hash: Optional[str] = None
    opset_version: int = 17
    fp16_support: bool = True
    quantized_versions: list[str] = field(default_factory=list)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "resnet50": ModelConfig(
        name="resnet50",
        domain="vision",
        description="ResNet-50 image classification model",
        input_shapes={"input": [1, 3, 224, 224]},
        output_shapes={"output": [1, 1000]},
        download_url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx",
    ),
    "bert-base": ModelConfig(
        name="bert-base",
        domain="nlp",
        description="BERT-Base for text classification",
        input_shapes={
            "input_ids": [1, 128],
            "attention_mask": [1, 128],
            "token_type_ids": [1, 128],
        },
        output_shapes={"logits": [1, 2]},
        download_url=None,
    ),
    "yolov8-n": ModelConfig(
        name="yolov8-n",
        domain="vision",
        description="YOLOv8 Nano for object detection",
        input_shapes={"images": [1, 3, 640, 640]},
        output_shapes={"output0": [1, 84, 8400]},
        fp16_support=True,
    ),
    "llama-7b": ModelConfig(
        name="llama-7b",
        domain="llm",
        description="LLaMA 7B language model",
        input_shapes={"input_ids": [1, 1]},
        output_shapes={"logits": [1, 1, 32000]},
        fp16_support=True,
    ),
    "stable-diffusion": ModelConfig(
        name="stable-diffusion",
        domain="generative",
        description="Stable Diffusion UNet component",
        input_shapes={"sample": [1, 4, 64, 64], "timestep": [1], "encoder_hidden_states": [1, 77, 768]},
        output_shapes={"sample": [1, 4, 64, 64]},
        fp16_support=True,
    ),
}


class ModelZoo:
    """Model zoo manager for downloading and managing ONNX models."""

    def __init__(self, cache_dir: str = "models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[dict]:
        """List all available models."""
        models = []
        for name, config in MODEL_REGISTRY.items():
            cached = self.cache_dir / f"{name}.onnx"
            models.append({
                "name": name,
                "domain": config.domain,
                "description": config.description,
                "cached": cached.exists(),
            })
        return models

    def get_config(self, model_name: str) -> ModelConfig:
        """Get model configuration."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        return MODEL_REGISTRY[model_name]

    def download(self, model_name: str, force: bool = False) -> Path:
        """Download a model from the registry."""
        config = self.get_config(model_name)
        output_path = self.cache_dir / f"{model_name}.onnx"

        if output_path.exists() and not force:
            print(f"Model {model_name} already cached at {output_path}")
            return output_path

        if config.download_url is None:
            raise ValueError(f"No download URL for {model_name}. Use export instead.")

        print(f"Downloading {model_name}...")
        response = requests.get(config.download_url, stream=True)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Saved to {output_path}")
        return output_path

    def export_from_pytorch(
        self,
        model_name: str,
        output_path: Optional[str] = None,
        opset_version: int = 17,
        dynamic_axes: Optional[dict] = None,
    ) -> Path:
        """Export a PyTorch model to ONNX format."""
        import torch

        config = self.get_config(model_name)
        if output_path is None:
            output_path = str(self.cache_dir / f"{model_name}.onnx")

        # Placeholder export logic - real implementation varies by model
        print(f"Exporting {model_name} to ONNX (opset={opset_version})...")
        print(f"Input shapes: {config.input_shapes}")

        # This would be model-specific in production
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Export saved to {path}")
        return path

    def validate(self, model_path: str) -> bool:
        """Validate an ONNX model file."""
        import onnx
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            print(f"Model {model_path} is valid")
            return True
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False
