#!/usr/bin/env python3
"""
CLI entry point for running ONNX model inference on ROCm.

Usage:
    python scripts/run_model.py --config configs/resnet50.yml
    python scripts/run_model.py --model resnet50 --iterations 1000
    python scripts/run_model.py --model models/bert.onnx --benchmark
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner import ONNXRunner, RunnerConfig
from src.models import ModelZoo
from src.optimizer import GraphOptimizer, OptimizationLevel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_random_input(config: RunnerConfig, shapes: list = None) -> dict:
    """Generate random input data for testing."""
    if config.input_feed:
        return config.input_feed

    # Default: generate based on model info (requires session)
    return {}


def main():
    parser = argparse.ArgumentParser(description="ONNX ROCm Runner CLI")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--input", type=str, help="Input image/file path")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--provider", type=str, default="ROCMExecutionProvider",
                        choices=["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--optimize", action="store_true", help="Optimize graph before inference")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--output", type=str, help="Output file (JSON results)")
    parser.add_argument("--download-only", action="store_true", help="Only download model, don't run")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    # List models
    if args.list_models:
        zoo = ModelZoo()
        for m in zoo.list_models():
            print(f"  {m.name:<25} {m.domain:<12} {m.size_mb:>8.1f}MB  {m.description}")
        return

    # Load config
    if args.config:
        config = RunnerConfig.from_yaml(args.config)
    elif args.model:
        config = RunnerConfig(
            model_path=args.model,
            providers=[args.provider],
            device_id=args.device_id,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            enable_profiling=args.profile,
        )
    else:
        parser.error("--config or --model required")
        return

    # Download only
    if args.download_only:
        zoo = ModelZoo()
        path = zoo.download(config.model_path)
        if path:
            print(f"Downloaded: {path}")
        else:
            print(f"Model not in zoo: {config.model_path}")
        return

    # Optional graph optimization
    if args.optimize:
        logger.info("Optimizing graph...")
        optimizer = GraphOptimizer(OptimizationLevel.AGGRESSIVE)
        result = optimizer.optimize(config.model_path, config.model_path + ".opt")
        print(result.summary)
        config.model_path = config.model_path + ".opt"

    # Create runner
    runner = ONNXRunner(config)
    runner.load()

    # Generate dummy input
    input_info = runner.get_input_info()
    input_feed = {}
    for info in input_info:
        shape = [max(s, 1) for s in info["shape"]]
        if len(shape) >= 1:
            shape[0] = args.batch_size
        input_feed[info["name"]] = np.random.randn(*shape).astype(np.float32)

    if args.benchmark:
        # Run benchmark
        runner.config.input_feed = input_feed
        results = runner.benchmark(input_feed=input_feed)
        print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved: {args.output}")
    else:
        # Single inference
        runner.warmup(input_feed=input_feed)
        result = runner.predict(input_feed=input_feed)
        print(f"Inference: {result.latency_ms:.2f}ms ({result.provider})")
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "latency_ms": result.latency_ms,
                    "provider": result.provider,
                    "input_shapes": result.input_shapes,
                }, f, indent=2)

    runner.close()


if __name__ == "__main__":
    main()
