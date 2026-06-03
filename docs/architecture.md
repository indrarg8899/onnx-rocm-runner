# Architecture

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     onnx-rocm-runner                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  runner   │  │ session  │  │ benchmark│  │ profiler │        │
│  │  (main)   │  │  (ROCM)  │  │ (perf)   │  │ (tracing)│        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │              │              │              │              │
│  ┌────┴──────────────┴──────────────┴──────────────┴────┐       │
│  │              ONNX Runtime C++ API                     │       │
│  └────┬──────────────────────────────────────────────┬──┘       │
│       │                                              │           │
│  ┌────┴──────────────┐                  ┌───────────┴─────┐    │
│  │ ROCMExecutionProvider│                │ CPUExecutionProvider│ │
│  └────┬──────────────┘                  └───────────────────┘   │
│       │                                                          │
│  ┌────┴─────────────────────────────────────────────────┐      │
│  │              AMD ROCm / HIP Runtime                    │      │
│  └────┬──────────────────────────────────────────────────┘      │
│       │                                                          │
│  ┌────┴──────────────┐                                         │
│  │  AMD MI300X GPU   │                                         │
│  │  (192GB HBM3)     │                                         │
│  └───────────────────┘                                         │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │  models   │  │optimizer │  │ exporter │                      │
│  │  (zoo)    │  │ (graph)  │  │ (PyTorch)│                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### Runner (`src/runner.py`)
The main entry point. Creates an `ONNXRunner` instance, loads a model, and
provides `predict()` and `benchmark()` APIs. Handles model resolution (local
file or model zoo download) and wraps all session interactions.

### Session Manager (`src/session.py`)
Manages ONNX Runtime sessions with provider selection and fallback logic.
Handles `ROCMExecutionProvider` configuration including GPU memory limits,
arena strategy, and compute stream settings.

Key decisions:
- **Provider fallback**: If ROCm unavailable → CUDA → CPU
- **Memory management**: Configurable arena strategy, optional memory limits
- **Profiling**: Integrated ONNX Runtime profiling hooks

### Model Zoo (`src/models.py`)
Registry of 20+ pre-validated ONNX models with SHA256 checksums. Handles
downloading, caching (`~/.cache/onnx-rocm-runner/models/`), and integrity
verification.

### Benchmark Suite (`src/benchmark.py`)
Runs inference benchmarks collecting:
- **Latency**: Mean, median, P95, P99, std dev
- **Throughput**: Inferences/second
- **Memory**: GPU peak/average VRAM, CPU RSS
- **Chrome trace**: Exportable to `chrome://tracing`

### Graph Optimizer (`src/optimizer.py`)
Custom ONNX graph optimization passes:
1. **Dead node elimination** — Remove unused operations
2. **Constant folding** — Pre-compute constant subgraphs
3. **Node fusion** — Conv+BN, Conv+ReLU, MatMul+Add
4. **Layout optimization** — ROCm-specific tensor layout preferences

### Exporter (`src/exporter.py`)
Export PyTorch models to ONNX format with:
- Dynamic batch axes
- Opset version selection (default: 17)
- FP16 conversion
- Automatic shape inference and verification
- Batch export support

### Profiler (`src/profiler.py`)
Deep profiling integration:
- Per-operator timing
- Chrome trace export
- Flame graph data
- ROCm activity tracing
- Provider-level breakdown

## Data Flow

```
1. Load config (YAML) or create RunnerConfig
2. ONNXRunner resolves model (file or zoo download)
3. ROCmSessionManager creates InferenceSession with ROCm EP
4. Warmup pass (10-100 iterations, untimed)
5. Benchmark: timed inference loop
6. Profiler: per-operator trace if enabled
7. Results: latency, throughput, memory stats
```

## Design Principles

1. **Zero-copy where possible** — Minimize GPU↔CPU data transfers
2. **Lazy initialization** — Import heavy deps only when needed
3. **Graceful degradation** — Fall back to CPU if ROCm unavailable
4. **Config-driven** — YAML configs for reproducibility
5. **Observable** — Every component exposes profiling and metrics
