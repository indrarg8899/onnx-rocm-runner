"""
ONNX Runtime ROCm Inference Runner.

Main entry point for running ONNX model inference on AMD ROCm GPUs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from .session import ROCmSessionManager, SessionConfig
from .benchmark import BenchmarkSuite
from .models import ModelZoo

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for a single inference result."""
    output: Any
    latency_ms: float
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    provider: str = "ROCMExecutionProvider"


@dataclass
class RunnerConfig:
    """Configuration for the inference runner."""
    model_path: str
    providers: List[str] = field(default_factory=lambda: ["ROCMExecutionProvider"])
    device_id: int = 0
    graph_optimization: str = "ORT_ENABLE_ALL"
    thread_count: int = 1
    memory_limit: int = 0  # 0 = unlimited
    arena_extend_strategy: str = "kNextPowerOfTwo"
    enable_profiling: bool = False
    input_feed: Optional[Dict[str, Any]] = None
    num_warmup: int = 10
    num_iterations: int = 100

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> RunnerConfig:
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunnerConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ONNXRunner:
    """
    High-performance ONNX Runtime inference runner for AMD ROCm.

    Supports batch inference, streaming, dynamic shapes, and
    deep profiling integration.
    """

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._session: Optional[Any] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._session_mgr: Optional[ROCmSessionManager] = None
        self._model_zoo = ModelZoo()
        self._warmup_done = False

    def load(self) -> "ONNXRunner":
        """Load the ONNX model and create inference session."""
        model_path = self._resolve_model(self.config.model_path)

        session_config = SessionConfig(
            providers=self.config.providers,
            device_id=self.config.device_id,
            graph_optimization_level=self.config.graph_optimization,
            thread_count=self.config.thread_count,
            memory_limit=self.config.memory_limit,
            arena_extend_strategy=self.config.arena_extend_strategy,
            enable_profiling=self.config.enable_profiling,
        )

        self._session_mgr = ROCmSessionManager(session_config)
        self._session = self._session_mgr.create_session(str(model_path))

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        logger.info(
            "Model loaded: %s (inputs=%d, outputs=%d)",
            model_path.name,
            len(self._input_names),
            len(self._output_names),
        )
        return self

    def _resolve_model(self, path: str) -> Path:
        """Resolve model path — download from zoo if needed."""
        p = Path(path)
        if p.exists():
            return p
        # Try model zoo
        model_path = self._model_zoo.download(path)
        if model_path:
            return Path(model_path)
        raise FileNotFoundError(f"Model not found: {path}")

    def predict(
        self,
        input_feed: Optional[Dict[str, np.ndarray]] = None,
    ) -> InferenceResult:
        """Run single inference and return result with timing."""
        if self._session is None:
            raise RuntimeError("Call load() before predict()")

        feed = input_feed or self.config.input_feed
        if feed is None:
            raise ValueError("No input_feed provided")

        start = time.perf_counter()
        outputs = self._session.run(self._output_names, feed)
        latency_ms = (time.perf_counter() - start) * 1000

        result_output = outputs[0] if len(outputs) == 1 else outputs

        return InferenceResult(
            output=result_output,
            latency_ms=latency_ms,
            input_shapes={k: list(v.shape) for k, v in feed.items()},
            output_shapes={},
            provider=self.config.providers[0],
        )

    def predict_batch(
        self,
        input_feed: List[Dict[str, np.ndarray]],
    ) -> List[InferenceResult]:
        """Run inference on a batch of inputs."""
        return [self.predict(feed) for feed in input_feed]

    def warmup(self, input_feed: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Warm up the execution engine."""
        feed = input_feed or self.config.input_feed
        if feed is None:
            logger.warning("No input_feed for warmup, skipping")
            return
        logger.info("Warming up with %d iterations", self.config.num_warmup)
        for _ in range(self.config.num_warmup):
            self._session.run(self._output_names, feed)
        self._warmup_done = True
        logger.info("Warmup complete")

    def benchmark(
        self,
        input_feed: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Run benchmarking suite on current session."""
        if self._session is None:
            raise RuntimeError("Call load() before benchmark()")

        feed = input_feed or self.config.input_feed
        if feed is None:
            raise ValueError("No input_feed for benchmarking")

        suite = BenchmarkSuite(self._session, self._output_names)
        return suite.run(
            input_feed=feed,
            num_warmup=self.config.num_warmup,
            num_iterations=self.config.num_iterations,
        )

    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get model input metadata."""
        return [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in self._session.get_inputs()
        ]

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get model output metadata."""
        return [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in self._session.get_outputs()
        ]

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def close(self) -> None:
        """Release session resources."""
        if self._session_mgr:
            self._session_mgr.close()
            self._session = None
            logger.info("Session closed")

    def __enter__(self) -> "ONNXRunner":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"ONNXRunner(model={self.config.model_path}, status={status})"


def create_runner(
    model_path: str,
    providers: Optional[List[str]] = None,
    **kwargs: Any,
) -> ONNXRunner:
    """Factory function to create a configured runner."""
    config = RunnerConfig(
        model_path=model_path,
        providers=providers or ["ROCMExecutionProvider"],
        **kwargs,
    )
    return ONNXRunner(config)
