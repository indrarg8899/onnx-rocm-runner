"""
ONNX Runtime Profiler.

Profiling support for ONNX Runtime inference on ROCm,
including ROCm activity tracing and flame graph generation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class OperatorProfile:
    """Profile data for a single operator."""
    name: str
    op_type: str
    provider: str
    duration_ms: float
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "provider": self.provider,
            "duration_ms": round(self.duration_ms, 4),
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
        }


@dataclass
class ProfileResult:
    """Complete profiling results for an inference run."""
    model_name: str
    total_duration_ms: float
    operators: List[OperatorProfile]
    kernel_times: Dict[str, float]  # op_type → total_ms
    provider_times: Dict[str, float]  # provider → total_ms
    memory_allocations: int = 0
    peak_memory_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_kernels(self) -> List[OperatorProfile]:
        """Top 10 most time-consuming kernels."""
        return sorted(self.operators, key=lambda x: x.duration_ms, reverse=True)[:10]

    def summary(self) -> str:
        """Human-readable profiling summary."""
        lines = [
            f"Profile: {self.model_name}",
            f"Total: {self.total_duration_ms:.2f}ms",
            f"Operators: {len(self.operators)}",
            "",
            "Top Kernels:",
        ]
        for op in self.top_kernels:
            lines.append(
                f"  {op.op_type:<25} {op.name:<40} "
                f"{op.duration_ms:>8.3f}ms  [{op.provider}]"
            )

        lines.append("")
        lines.append("Provider Breakdown:")
        for provider, ms in sorted(self.provider_times.items(), key=lambda x: -x[1]):
            pct = (ms / self.total_duration_ms * 100) if self.total_duration_ms > 0 else 0
            lines.append(f"  {provider:<30} {ms:>8.2f}ms ({pct:>5.1f}%)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_duration_ms": round(self.total_duration_ms, 3),
            "operator_count": len(self.operators),
            "operators": [op.to_dict() for op in self.operators],
            "kernel_times": {k: round(v, 4) for k, v in self.kernel_times.items()},
            "provider_times": {k: round(v, 4) for k, v in self.provider_times.items()},
        }

    def to_chrome_trace(self) -> Dict[str, Any]:
        """Export as Chrome trace format (chrome://tracing)."""
        events = []
        ts = 0
        for op in self.operators:
            duration_us = op.duration_ms * 1000
            events.append({
                "name": f"{op.op_type}:{op.name}",
                "cat": "inference",
                "ph": "X",
                "ts": ts,
                "dur": int(duration_us),
                "pid": 1,
                "tid": 0,
                "args": {"provider": op.provider},
            })
            ts += int(duration_us)

        return {"traceEvents": events, "displayTimeUnit": "ms"}


class ONNXProfiler:
    """
    Profile ONNX Runtime inference and extract kernel-level timings.

    Supports:
    - Per-operator profiling
    - Chrome trace export
    - ROCm activity trace integration
    """

    def __init__(self, session: Any, output_names: List[str]):
        self.session = session
        self.output_names = output_names

    def profile_single(
        self,
        input_feed: Dict[str, Any],
        model_name: str = "unknown",
    ) -> ProfileResult:
        """Run a single profiled inference and collect results."""
        # Enable profiling on the session
        options = self.session.get_session_options()
        options.enable_profiling = True

        # Run inference
        start = time.perf_counter()
        self.session.run(self.output_names, input_feed)
        total_ms = (time.perf_counter() - start) * 1000

        # Get profiling data
        profile_file = self.session.end_profiling()
        operators = self._parse_profile(profile_file)

        # Aggregate by kernel type
        kernel_times: Dict[str, float] = {}
        provider_times: Dict[str, float] = {}
        for op in operators:
            kernel_times[op.op_type] = kernel_times.get(op.op_type, 0) + op.duration_ms
            provider_times[op.provider] = provider_times.get(op.provider, 0) + op.duration_ms

        result = ProfileResult(
            model_name=model_name,
            total_duration_ms=total_ms,
            operators=operators,
            kernel_times=kernel_times,
            provider_times=provider_times,
        )

        logger.info("Profiling complete:\n%s", result.summary())
        return result

    def profile_multiple(
        self,
        input_feed: Dict[str, Any],
        num_iterations: int = 5,
        model_name: str = "unknown",
    ) -> ProfileResult:
        """Run multiple profiled iterations for stable results."""
        all_operators: List[OperatorProfile] = []
        all_kernel_times: Dict[str, List[float]] = {}
        all_provider_times: Dict[str, List[float]] = {}
        total_ms_sum = 0.0

        for i in range(num_iterations):
            result = self.profile_single(input_feed, model_name)
            total_ms_sum += result.total_duration_ms

            for op in result.operators:
                all_operators.append(op)

            for k, v in result.kernel_times.items():
                all_kernel_times.setdefault(k, []).append(v)

            for k, v in result.provider_times.items():
                all_provider_times.setdefault(k, []).append(v)

        # Average kernel/provider times
        avg_kernel = {k: sum(v) / len(v) for k, v in all_kernel_times.items()}
        avg_provider = {k: sum(v) / len(v) for k, v in all_provider_times.items()}

        return ProfileResult(
            model_name=model_name,
            total_duration_ms=total_ms_sum / num_iterations,
            operators=self._average_operators(all_operators, num_iterations),
            kernel_times=avg_kernel,
            provider_times=avg_provider,
        )

    def _parse_profile(self, profile_file: str) -> List[OperatorProfile]:
        """Parse ONNX Runtime profile JSON file."""
        try:
            with open(profile_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to parse profile file: %s", e)
            return []

        operators = []
        events = data if isinstance(data, list) else data.get("traceEvents", [])

        for event in events:
            if event.get("cat") == "Node" or event.get("cat") == "inference":
                operators.append(OperatorProfile(
                    name=event.get("name", "unknown"),
                    op_type=event.get("args", {}).get("op_name", "unknown"),
                    provider=event.get("args", {}).get("provider", "unknown"),
                    duration_ms=event.get("dur", 0) / 1000,  # μs → ms
                ))

        return operators

    def _average_operators(
        self,
        operators: List[OperatorProfile],
        n: int,
    ) -> List[OperatorProfile]:
        """Average operator times across iterations."""
        op_map: Dict[str, List[float]] = {}
        for op in operators:
            op_map.setdefault(op.name, []).append(op.duration_ms)

        return [
            OperatorProfile(
                name=name,
                op_type=ops[0].op_type if ops else "unknown",
                provider=ops[0].provider if ops else "unknown",
                duration_ms=sum(durations) / len(durations),
            )
            for name, durations in op_map.items()
            for ops in [[op for op in operators if op.name == name]]
        ]

    @staticmethod
    def export_chrome_trace(result: ProfileResult, output_path: Union[str, Path]) -> None:
        """Export profile result as Chrome trace JSON."""
        trace = result.to_chrome_trace()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(trace, f, indent=2)
        logger.info("Chrome trace saved: %s", output_path)

    @staticmethod
    def export_flame_graph_data(result: ProfileResult, output_path: Union[str, Path]) -> None:
        """Export flame graph compatible data."""
        stacks = []
        for op in result.operators:
            stacks.append(f"{op.provider};{op.op_type};{op.name} {op.duration_ms:.3f}ms")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(stacks))

        logger.info("Flame graph data saved: %s", output_path)


class ROCmActivityTracer:
    """
    ROCm activity trace integration.

    Collects ROCm kernel dispatches, memory transfers, and
    synchronization events during inference.
    """

    def __init__(self):
        self._tracing = False
        self._activities: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Start ROCm activity tracing."""
        self._tracing = True
        self._activities = []
        logger.info("ROCm activity tracing started")

    def stop(self) -> List[Dict[str, Any]]:
        """Stop tracing and return collected activities."""
        self._tracing = False
        logger.info("ROCm activity tracing stopped (%d events)", len(self._activities))
        return self._activities

    def record(self, activity_type: str, name: str, duration_us: float) -> None:
        """Record a ROCm activity event."""
        if self._tracing:
            self._activities.append({
                "type": activity_type,
                "name": name,
                "duration_us": duration_us,
                "timestamp": time.time(),
            })

    @property
    def is_tracing(self) -> bool:
        return self._tracing

    def export(self, output_path: Union[str, Path]) -> None:
        """Export trace data as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._activities, f, indent=2)
        logger.info("ROCm trace exported: %s", output_path)
