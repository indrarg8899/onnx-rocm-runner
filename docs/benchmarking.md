# Benchmarking

## Running Benchmarks

```bash
# Full model zoo benchmark
python -m src.benchmark --model zoo --iterations 100

# Specific model
python -m src.benchmark --model models/resnet50.onnx --batch-sizes 1,4,8,16

# Compare providers
python -m src.benchmark --model models/resnet50.onnx --providers ROCMExecutionProvider,CPUExecutionProvider
```

## Interpreting Results

- **Latency**: Time per inference in milliseconds
- **Throughput**: Inferences per second (batch_size / avg_latency)
- **P50/P95/P99**: Percentile latencies showing tail behavior
- **Memory**: Peak GPU memory usage

## ROCm Performance Tips

1. Use `ORT_ENABLE_ALL` graph optimization level
2. Enable memory arena with `kNextPowerOfTwo` strategy
3. Increase batch size to utilize GPU compute units
4. Use FP16 models where possible for 2x throughput
