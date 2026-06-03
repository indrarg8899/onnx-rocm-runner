#!/usr/bin/env python3
"""
Backend Comparison Benchmark.

Compare inference performance across multiple execution backends:
CPU, CUDA, ROCm, TensorRT, DirectML.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent dir to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session import ROCmSessionManager, SessionConfig
from src.benchmark import BenchmarkSuite, BenchmarkResult
from src.models import ModelZoo

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


BACKEND_CONFIGS = {
    "cpu": {
        "providers": ["CPUExecutionProvider"],
        "label": "CPU",
    },
    "cuda": {
        "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "label": "CUDA (NVIDIA)",
    },
    "rocm": {
        "providers": ["ROCMExecutionProvider", "CPUExecutionProvider"],
        "label": "ROCm (AMD)",
    },
    "tensorrt": {
        "providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "label": "TensorRT",
    },
    "directml": {
        "providers": ["DirectMLExecutionProvider", "CPUExecutionProvider"],
        "label": "DirectML",
    },
    "openvino": {
        "providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        "label": "OpenVINO",
    },
    "coreml": {
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "label": "CoreML (Apple)",
    },
}


def run_comparison(
    model_path: str,
    backends: List[str],
    input_shapes: Dict[str, List[int]],
    num_warmup: int = 50,
    num_iterations: int = 500,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run benchmark across all specified backends."""
    results = []

    for backend_name in backends:
        if backend_name not in BACKEND_CONFIGS:
            logger.warning("Unknown backend: %s (available: %s)", backend_name, list(BACKEND_CONFIGS.keys()))
            continue

        config = BACKEND_CONFIGS[backend_name]
        logger.info("Testing backend: %s (%s)", backend_name, config["label"])

        session_config = SessionConfig(
            providers=config["providers"],
            graph_optimization_level="ORT_ENABLE_ALL",
        )

        try:
            mgr = ROCmSessionManager(session_config)
            session = mgr.create_session(model_path)
        except Exception as e:
            logger.warning("Backend %s failed to initialize: %s", backend_name, e)
            continue

        output_names = [out.name for out in session.get_outputs()]

        # Generate dummy input
        import numpy as np
        input_feed = {}
        for inp in session.get_inputs():
            shape = list(inp.shape)
            # Replace dynamic dims with 1
            shape = [max(s, 1) for s in shape]
            input_feed[inp.name] = np.random.randn(*shape).astype(np.float32)

        # Run benchmark
        try:
            suite = BenchmarkSuite(
                session=session,
                output_names=output_names,
                model_name=Path(model_path).stem,
            )
            result = suite.run(
                input_feed=input_feed,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            result["backend"] = backend_name
            result["backend_label"] = config["label"]
            results.append(result)
            logger.info(
                "  %s: %.2f ms, %.1f inf/s",
                config["label"],
                result["latency"]["mean_ms"],
                result["throughput"]["inf_per_sec"],
            )
        except Exception as e:
            logger.warning("Benchmark failed for %s: %s", backend_name, e)
        finally:
            mgr.close()

    # Print comparison table
    if results:
        print("\n" + "=" * 80)
        print("BACKEND COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Backend':<20} {'Latency (ms)':>12} {'Throughput':>12} {'VRAM (MB)':>10}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x["latency"]["mean_ms"]):
            print(
                f"{r['backend_label']:<20} "
                f"{r['latency']['mean_ms']:>10.2f}ms "
                f"{r['throughput']['inf_per_sec']:>10.1f}/s "
                f"{r['memory']['gpu_peak_mb']:>8.1f}MB"
            )
        print("=" * 80)

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_file)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare ONNX Runtime backends")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["cpu", "rocm"],
        choices=list(BACKEND_CONFIGS.keys()),
        help="Backends to compare",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=500, help="Measured iterations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    run_comparison(
        model_path=args.model,
        backends=args.backends,
        input_shapes={},  # Auto-detected from model
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
