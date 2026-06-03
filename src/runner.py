"""
ONNX Runtime inference runner for AMD ROCm GPUs.

Provides optimized inference with ROCm execution provider,
dynamic batching, and profiling support.
"""

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort


class ONNXRunner:
    """ONNX Runtime runner with ROCm execution provider support."""

    DEFAULT_PROVIDERS = [
        ("ROCMExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo"}),
        "CPUExecutionProvider",
    ]

    def __init__(
        self,
        model_path: str,
        execution_provider: Union[str, list[str]] = "ROCMExecutionProvider",
        device_id: int = 0,
        num_threads: int = 4,
        graph_optimization: str = "ORT_ENABLE_ALL",
    ):
        self.model_path = model_path
        self.device_id = device_id
        self._stats = {"total_runs": 0, "total_time_ms": 0.0}

        providers = self._build_providers(execution_provider, device_id)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, graph_optimization, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def _build_providers(
        self, provider: Union[str, list[str]], device_id: int
    ) -> list:
        """Build execution provider list."""
        if isinstance(provider, str):
            return [
                (provider, {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        return provider

    @property
    def provider(self) -> str:
        """Current active execution provider."""
        return self.session.get_providers()[0]

    def run(
        self,
        input_data: dict[str, np.ndarray],
        run_options: Optional[ort.RunOptions] = None,
    ) -> dict[str, np.ndarray]:
        """Run inference on a single input."""
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, input_data, run_options)
        elapsed = (time.perf_counter() - start) * 1000

        self._stats["total_runs"] += 1
        self._stats["total_time_ms"] += elapsed

        return dict(zip(self.output_names, outputs))

    def run_batch(
        self,
        batch_inputs: list[dict[str, np.ndarray]],
        batch_size: int = 1,
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on a batch of inputs."""
        results = []
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            # Stack into batch dimension
            batched = {}
            for name in self.input_names:
                batched[name] = np.stack([inp[name] for inp in batch])

            outputs = self.run(batched)
            # Split batch into individual results
            for j in range(len(batch)):
                result = {name: outputs[name][j] for name in self.output_names}
                results.append(result)

        return results

    def get_input_meta(self) -> list[dict]:
        """Get model input metadata."""
        return [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in self.session.get_inputs()
        ]

    def get_output_meta(self) -> list[dict]:
        """Get model output metadata."""
        return [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in self.session.get_outputs()
        ]

    def get_stats(self) -> dict:
        """Return runner statistics."""
        stats = self._stats.copy()
        if stats["total_runs"] > 0:
            stats["avg_latency_ms"] = stats["total_time_ms"] / stats["total_runs"]
        stats["provider"] = self.provider
        stats["model"] = str(self.model_path)
        return stats
