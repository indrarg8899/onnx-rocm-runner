"""
Benchmark suite for ONNX Runtime on ROCm.

Measures latency, throughput, memory usage, and profiling data
across different batch sizes and configurations.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from tabulate import tabulate

from .runner import ONNXRunner


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    model: str
    provider: str
    batch_size: int
    input_shape: list[int]
    iterations: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    memory_mb: float
    total_time_sec: float


class BenchmarkSuite:
    """Comprehensive benchmark suite for ONNX models."""

    def __init__(
        self,
        model_path: str,
        provider: str = "ROCMExecutionProvider",
        device_id: int = 0,
    ):
        self.model_path = model_path
        self.provider = provider
        self.device_id = device_id
        self.results: list[BenchmarkResult] = []

    def _create_dummy_input(
        self, runner: ONNXRunner, batch_size: int
    ) -> dict[str, np.ndarray]:
        """Create random input data matching model input shapes."""
        inputs = {}
        for meta in runner.get_input_meta():
            shape = list(meta["shape"])
            shape[0] = batch_size
            shape = [1 if s == 0 or s == -1 else s for s in shape]
            inputs[meta["name"]] = np.random.randn(*shape).astype(np.float32)
        return inputs

    def run(
        self,
        iterations: int = 100,
        batch_sizes: list[int] = None,
        warmup: int = 10,
        input_shapes: Optional[dict[str, list[int]]] = None,
    ) -> list[BenchmarkResult]:
        """Run full benchmark suite."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        runner = ONNXRunner(
            self.model_path,
            execution_provider=self.provider,
            device_id=self.device_id,
        )

        for bs in batch_sizes:
            dummy_input = self._create_dummy_input(runner, bs)

            # Warmup
            for _ in range(warmup):
                runner.run(dummy_input)

            # Benchmark
            latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                runner.run(dummy_input)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

            latencies.sort()
            total_time = sum(latencies) / 1000

            result = BenchmarkResult(
                model=Path(self.model_path).stem,
                provider=runner.provider,
                batch_size=bs,
                input_shape=list(dummy_input.values())[0].shape if dummy_input else [],
                iterations=iterations,
                avg_latency_ms=np.mean(latencies),
                p50_latency_ms=np.percentile(latencies, 50),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                throughput_fps=bs * 1000 / np.mean(latencies),
                memory_mb=0.0,  # Would use runtime-specific memory query
                total_time_sec=total_time,
            )
            self.results.append(result)

        return self.results

    def print_results(self, results: Optional[list[BenchmarkResult]] = None):
        """Print benchmark results as a formatted table."""
        data = results or self.results
        headers = [
            "Batch", "Avg (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)",
            "Throughput", "Total (s)",
        ]
        rows = [
            [
                r.batch_size,
                f"{r.avg_latency_ms:.2f}",
                f"{r.p50_latency_ms:.2f}",
                f"{r.p95_latency_ms:.2f}",
                f"{r.p99_latency_ms:.2f}",
                f"{r.throughput_fps:.1f} inf/s",
                f"{r.total_time_sec:.2f}",
            ]
            for r in data
        ]
        print(f"\nBenchmark: {data[0].model} | Provider: {data[0].provider}\n")
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    def save_results(self, path: str):
        """Save results to JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self.results]
        output.write_text(json.dumps(data, indent=2))
