"""
ONNX Runtime inference benchmarking suite.

Measures latency, throughput, memory usage, and generates
comparable performance reports across backends.
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    model_name: str
    provider: str
    device: str
    precision: str
    batch_size: int
    input_shapes: Dict[str, List[int]]

    # Timing
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0

    # Throughput
    throughput_inf_per_sec: float = 0.0

    # Memory
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_avg_mb: float = 0.0
    cpu_memory_mb: float = 0.0

    # Metadata
    warmup_iterations: int = 0
    measured_iterations: int = 0
    total_time_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "input_shapes": self.input_shapes,
            "latency": {
                "mean_ms": round(self.latency_mean_ms, 3),
                "median_ms": round(self.latency_median_ms, 3),
                "p95_ms": round(self.latency_p95_ms, 3),
                "p99_ms": round(self.latency_p99_ms, 3),
                "std_ms": round(self.latency_std_ms, 3),
                "min_ms": round(self.latency_min_ms, 3),
                "max_ms": round(self.latency_max_ms, 3),
            },
            "throughput": {
                "inf_per_sec": round(self.throughput_inf_per_sec, 2),
            },
            "memory": {
                "gpu_peak_mb": round(self.gpu_memory_peak_mb, 1),
                "gpu_avg_mb": round(self.gpu_memory_avg_mb, 1),
                "cpu_mb": round(self.cpu_memory_mb, 1),
            },
            "iterations": {
                "warmup": self.warmup_iterations,
                "measured": self.measured_iterations,
            },
            "total_time_sec": round(self.total_time_sec, 2),
            "extra": self.extra,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Model: {self.model_name} | Provider: {self.provider}\n"
            f"  Latency: {self.latency_mean_ms:.2f}ms (mean), "
            f"{self.latency_p95_ms:.2f}ms (p95), "
            f"{self.latency_p99_ms:.2f}ms (p99)\n"
            f"  Throughput: {self.throughput_inf_per_sec:.1f} inf/s\n"
            f"  VRAM: {self.gpu_memory_peak_mb:.1f}MB peak\n"
            f"  Iterations: {self.warmup_iterations} warmup + "
            f"{self.measured_iterations} measured"
        )


def _percentile(data: List[float], pct: float) -> float:
    """Calculate percentile without scipy."""
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB (ROCm/HIP)."""
    try:
        import rocm
        # Fallback: try reading from sysfs
        with open("/sys/class/drm/card0/device/mem_info_vram_used", "r") as f:
            return int(f.read().strip()) / (1024 * 1024)
    except Exception:
        pass
    try:
        # Try hipMemGetInfo
        import ctypes
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return (total.value - free.value) / (1024 * 1024)
    except Exception:
        return 0.0


def _get_cpu_memory_usage() -> float:
    """Get current process CPU memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        with open(f"/proc/{__import__('os').getpid()}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


class BenchmarkSuite:
    """
    Run inference benchmarks on an ONNX Runtime session.

    Measures latency distribution, throughput, and GPU/CPU memory.
    """

    def __init__(
        self,
        session: Any,
        output_names: List[str],
        model_name: str = "unknown",
    ):
        self.session = session
        self.output_names = output_names
        self.model_name = model_name

    def run(
        self,
        input_feed: Dict[str, np.ndarray],
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Run benchmark and return results as dict.

        Args:
            input_feed: Input data dict {name: numpy_array}
            num_warmup: Warmup iterations (not timed)
            num_iterations: Timed iterations

        Returns:
            Dictionary with benchmark metrics
        """
        provider = self._get_provider()
        input_shapes = {k: list(v.shape) for k, v in input_feed.items()}

        # Warmup
        logger.info("Warming up: %d iterations", num_warmup)
        for _ in range(num_warmup):
            self.session.run(self.output_names, input_feed)
        gc.collect()

        # Baseline memory
        mem_before = _get_cpu_memory_usage()
        gpu_before = _get_gpu_memory_usage()

        # Benchmark
        latencies = []
        gpu_snapshots = []
        start_total = time.perf_counter()

        logger.info("Benchmarking: %d iterations", num_iterations)
        for i in range(num_iterations):
            # Force GPU sync before timing
            self.session.run(self.output_names, input_feed)

            start = time.perf_counter()
            self.session.run(self.output_names, input_feed)
            elapsed_ms = (time.perf_counter() - start) * 1000

            latencies.append(elapsed_ms)

            # Sample GPU memory every 10 iterations
            if i % 10 == 0:
                gpu_snapshots.append(_get_gpu_memory_usage())

        total_time = time.perf_counter() - start_total

        # Compute stats
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        batch_size = list(input_feed.values())[0].shape[0] if input_feed else 1

        result = BenchmarkResult(
            model_name=self.model_name,
            provider=provider,
            device="rocm",
            precision="fp32",
            batch_size=batch_size,
            input_shapes=input_shapes,
            latency_mean_ms=mean_lat,
            latency_median_ms=median_lat,
            latency_p95_ms=_percentile(latencies, 95),
            latency_p99_ms=_percentile(latencies, 99),
            latency_std_ms=std_lat,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            throughput_inf_per_sec=(1000.0 / mean_lat) * batch_size,
            gpu_memory_peak_mb=max(gpu_snapshots) if gpu_snapshots else 0.0,
            gpu_memory_avg_mb=statistics.mean(gpu_snapshots) if gpu_snapshots else 0.0,
            cpu_memory_mb=_get_cpu_memory_usage() - mem_before,
            warmup_iterations=num_warmup,
            measured_iterations=num_iterations,
            total_time_sec=total_time,
        )

        logger.info("Benchmark complete:\n%s", result.summary())
        return result.to_dict()

    def _get_provider(self) -> str:
        """Get active provider name."""
        providers = self.session.get_providers()
        return providers[0] if providers else "unknown"


class MultiModelBenchmark:
    """
    Benchmark multiple models and produce comparative results.
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a benchmark result."""
        # Convert dict back to dataclass for storage
        br = BenchmarkResult(
            model_name=result["model_name"],
            provider=result["provider"],
            device=result["device"],
            precision=result["precision"],
            batch_size=result["batch_size"],
            input_shapes=result["input_shapes"],
            latency_mean_ms=result["latency"]["mean_ms"],
            throughput_inf_per_sec=result["throughput"]["inf_per_sec"],
            gpu_memory_peak_mb=result["memory"]["gpu_peak_mb"],
        )
        self.results.append(br)

    def compare_table(self) -> str:
        """Generate a comparison table."""
        lines = [
            f"{'Model':<20} {'Provider':<15} {'Latency':>10} {'Throughput':>12} {'VRAM':>8}",
            "-" * 70,
        ]
        for r in sorted(self.results, key=lambda x: x.latency_mean_ms):
            lines.append(
                f"{r.model_name:<20} {r.provider:<15} "
                f"{r.latency_mean_ms:>8.2f}ms "
                f"{r.throughput_inf_per_sec:>10.1f}/s "
                f"{r.gpu_memory_peak_mb:>6.1f}MB"
            )
        return "\n".join(lines)

    def to_json(self) -> List[Dict[str, Any]]:
        """Export all results as JSON-serializable list."""
        return [r.to_dict() for r in self.results]
