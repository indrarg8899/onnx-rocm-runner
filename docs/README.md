# onnx-rocm-runner Documentation

## Setup

### Requirements
- Python 3.10+
- ONNX Runtime with ROCm EP
- AMD GPU with ROCm 6.0+

### Installation

```bash
pip install -r requirements.txt
```

## Model Zoo

### Available Models

| Model | Domain | Download |
|-------|--------|----------|
| ResNet-50 | Vision | Auto |
| BERT-Base | NLP | Auto |
| YOLOv8-n | Object Detection | Manual export |
| LLaMA-7B | LLM | Manual export |
| Stable Diffusion | Image Generation | Manual export |

### Adding Custom Models

```python
from src.model_zoo import ModelConfig, MODEL_REGISTRY

MODEL_REGISTRY["my_model"] = ModelConfig(
    name="my_model",
    domain="custom",
    description="My custom model",
    input_shapes={"input": [1, 3, 224, 224]},
    output_shapes={"output": [1, 1000]},
    download_url="https://example.com/model.onnx",
)
```

## ROCm Execution Provider

The ROCm EP enables ONNX Runtime to execute operators on AMD GPUs.

### Configuration

```python
providers = [
    ("ROCMExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB
    }),
    "CPUExecutionProvider",
]
```

### Supported Operators

Most standard ONNX operators are supported. Check [ONNX Runtime ROCm docs](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) for full list.
