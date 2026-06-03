# onnx-rocm-runner

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16+-005CED.svg)](https://onnxruntime.ai/)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)

ONNX Runtime benchmark and inference runner optimized for AMD ROCm GPUs with model zoo and performance analysis.

## Architecture

```
┌──────────────────────────────────────────┐
│           onnx-rocm-runner               │
├────────────┬────────────┬────────────────┤
│  Model     │ Benchmark  │    Profiler    │
│  Zoo       │  Suite     │                │
├────────────┴────────────┴────────────────┤
│        ONNX Runtime Execution            │
├──────────────────────────────────────────┤
│    ROCm Execution Provider (ROCm EP)     │
├──────────────────────────────────────────┤
│         AMD GPU Hardware                 │
└──────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run a model
python -m src.runner --model resnet50 --batch-size 32

# Run benchmarks
python -m src.benchmark --model zoo --iterations 100

# List available models
python -m src.model_zoo list
```

## Features

- **Model Zoo** - Pre-configured models (ResNet, BERT, YOLO, LLM variants)
- **Benchmark Suite** - Latency, throughput, and memory profiling
- **ROCm EP** - Native AMD GPU execution provider
- **Batch Inference** - Configurable batch sizes with padding
- **Profiling** - Detailed execution trace and timing analysis
- **Export Tools** - Convert PyTorch models to ONNX format

## Usage

### Run Inference

```python
from src.runner import ONNXRunner

runner = ONNXRunner(
    model_path="models/resnet50.onnx",
    execution_provider="ROCMExecutionProvider",
    device_id=0,
)

output = runner.run(input_data)
```

### Benchmark

```python
from src.benchmark import BenchmarkSuite

suite = BenchmarkSuite(model_path="models/resnet50.onnx")
results = suite.run(
    iterations=100,
    batch_sizes=[1, 8, 32, 64],
    input_shapes={"input": [3, 224, 224]},
)
suite.print_results(results)
```

### Model Zoo

```bash
# List models
python -m src.model_zoo list

# Download a model
python -m src.model_zoo download resnet50

# Export PyTorch model to ONNX
python -m src.model_zoo export --pytorch resnet50 --output models/resnet50.onnx
```

## Supported Models

| Model | Domain | Input Shape | FP16 Support |
|-------|--------|------------|-------------|
| ResNet-50 | Vision | [3, 224, 224] | ✅ |
| BERT-Base | NLP | [1, 128] | ✅ |
| YOLOv8-n | Vision | [3, 640, 640] | ✅ |
| LLaMA-7B | LLM | Variable | ✅ |
| Stable Diffusion | GenAI | [4, 64, 64] | ✅ |

## License

MIT License. See [LICENSE](LICENSE) for details.
