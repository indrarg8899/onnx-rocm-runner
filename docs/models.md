# Model Zoo Documentation

## Overview

The model zoo provides pre-validated ONNX models optimized for ROCm inference.
All models are downloaded to `~/.cache/onnx-rocm-runner/models/` with SHA256
integrity verification.

## Usage

```python
from src.models import ModelZoo

zoo = ModelZoo()

# List available models
models = zoo.list_models(domain="vision")
for m in models:
    print(f"{m.name}: {m.description} ({m.size_mb:.1f} MB)")

# Download a model
path = zoo.download("resnet50")
print(f"Downloaded to: {path}")

# Get metadata
info = zoo.get_info("resnet50")
print(f"Domain: {info.domain}, Precision: {info.precision}")
```

## Available Models

### Vision

#### ResNet-18 (`resnet18`)
- **Params**: 11.7M | **Size**: 45 MB
- **Top-1 Accuracy**: 69.6% (ImageNet)
- **Input**: `[batch, 3, 224, 224]` (NCHW, float32)
- **Output**: `[batch, 1000]` (class logits)
- **Opset**: 17
- **Tags**: classification, torchvision, lightweight

#### ResNet-50 (`resnet50`)
- **Params**: 25.6M | **Size**: 98 MB
- **Top-1 Accuracy**: 75.2% (ImageNet)
- **Input**: `[batch, 3, 224, 224]`
- **Output**: `[batch, 1000]`
- **Tags**: classification, torchvision, baseline

#### EfficientNet-Lite4 (`efficientnet_b0`)
- **Params**: 13M | **Size**: 23 MB
- **Top-1 Accuracy**: 80.4% (ImageNet)
- **Tags**: classification, mobile, efficient

#### MobileNetV2 (`mobilenetv2`)
- **Params**: 3.4M | **Size**: 14 MB
- **Top-1 Accuracy**: 71.8% (ImageNet)
- **Tags**: classification, mobile, edge, tiny

#### YOLOv8 Nano (`yolov8n`)
- **Params**: 3.2M | **Size**: 13 MB
- **mAP**: 37.3 (COCO val2017)
- **Tags**: detection, realtime, tiny

#### YOLOv8 X-Large (`yolov8x`)
- **Params**: 68M | **Size**: 268 MB
- **mAP**: 53.9 (COCO val2017)
- **Tags**: detection, accuracy, large

### NLP

#### BERT-Base SQuAD (`bert_base`)
- **Params**: 110M | **Size**: 417 MB (INT8)
- **F1 Score**: 88.5 (SQuAD v1.1)
- **Input**: `[batch, seq_len]` (token IDs)
- **Tags**: question-answering, transformer, quantized

#### DistILBERT (`distilbert`)
- **Params**: 66M | **Size**: 265 MB
- **Tags**: classification, distilled, fast

### LLM

#### GPT-2 (`gpt2`)
- **Params**: 124M | **Size**: 490 MB
- **Vocabulary**: 50,257 tokens
- **Context**: 1024 tokens
- **Tags**: text-generation, transformer, baseline

#### GPT-2 Medium (`gpt2_medium`)
- **Params**: 355M | **Size**: 1.4 GB
- **Tags**: text-generation, transformer, medium

#### LLaMA-7B (`llama7b`)
- **Params**: 7B | **Size**: 14 GB (FP16)
- **Context**: 4096 tokens
- **Tags**: text-generation, llm, transformer, large

### Speech

#### Whisper Base (`whisper_base`)
- **Params**: 74M | **Size**: 290 MB
- **WER**: ~10% (LibriSpeech)
- **Tags**: speech-recognition, encoder-decoder

### Multimodal

#### CLIP ViT-L/14 (`clip_vitl14`)
- **Params**: 427M | **Size**: 427 MB
- **Zero-shot accuracy**: 75.3% (ImageNet)
- **Tags**: zero-shot, embedding, vision-language

### GenAI

#### Stable Diffusion 1.5 UNet (`stable_diffusion_15`)
- **Params**: 860M | **Size**: 3.4 GB
- **Resolution**: 512×512
- **Tags**: image-generation, diffusion

## CLI Usage

```bash
# Download a model
python scripts/run_model.py --model resnet50 --download-only

# List all models
python -c "from src.models import ModelZoo; [print(m.name) for m in ModelZoo().list_models()]"

# Clear cache
python -c "from src.models import ModelZoo; ModelZoo().clear_cache()"
```

## Adding Custom Models

To add a model to the zoo, register it in `src/models.py`:

```python
MODEL_REGISTRY["my_model"] = ModelEntry(
    name="my_model",
    url="https://example.com/my_model.onnx",
    domain="vision",
    size_bytes=10_000_000,
    checksum="abc123...",  # SHA256
    precision="fp32",
    description="My custom model",
    tags=["custom"],
)
```
