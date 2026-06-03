#!/usr/bin/env python3
"""
Run benchmarks across all models in the model zoo.

Generates a comprehensive performance report for ROCm GPU inference.
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
from src.benchmark import BenchmarkSuite, MultiModelBenchmark

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SHAPES = {
    "resnet18": {"input": [1, 3, 224, 224]},
    "resnet50": {"input": [1, 3, 224, 224]},
    "efficientnet_b0": {"input": [1, 3, 224, 224]},
    "mobilenetv2": {"input": [1, 3, 224, 224]},
    "yolov8n": {"images": [1, 3, 640, 640]},
    "yolov8x": {"images": [1, 3, 640, 640]},
    "bert_base": {
        "input_ids": [1, 128],
        "attention_mask": [1, 128],
        "token_type_ids": [1, 128],
    },
    "distilbert": {"input_ids": [1, 128]},
    "gpt2": {"input_ids": [1, 64]},
    "gpt2_medium": {"input_ids": [1, 64]},
    "whisper_base": {"audio_pcm": [1, 16000]},
}


def benchmark_model(
    model_name: str,
    zoo: ModelZoo,
    provider: str,
    warmup: int,
    iterations: int,
    batch_size: int,
) -> dict:
    """Benchmark a single model."""
    # Download model
    model_path = zoo.download(model_name)
    if not model_path:
        logger.error("Failed to download model: %s", model_name)
        return {}

    config = RunnerConfig(
        model_path=str(model_path),
        providers=[provider, "CPUExecutionProvider"],
        num_warmup=warmup,
        num_iterations=iterations,
    )

    try:
        runner = ONNXRunner(config)
        runner.load()

        # Generate input data
        input_info = runner.get_input_info()
        input_feed = {}
        for info in input_info:
            shape = [max(s, 1) for s in info["shape"]]
            if len(shape) >= 1:
                shape[0] = batch_size
            input_feed[info["name"]] = np.random.randn(*shape).astype(np.float32)

        # Run benchmark
        result = runner.benchmark(input_feed=input_feed)
        result["model_name"] = model_name
        runner.close()
        return result

    except Exception as e:
        logger.error("Benchmark failed for %s: %s", model_name, e)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument("--device", default="rocm", choices=["rocm", "cuda", "cpu"],
                        help="Target device")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--domains", nargs="+", default=["vision", "nlp", "llm"],
                        choices=["vision", "nlp", "llm", "speech", "multimodal"],
                        help="Model domains to include")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32],
                        help="Batch sizes to test")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Measured iterations")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file")
    parser.add_argument("--fast", action="store_true",
                        help="Quick mode: minimal iterations")
    args = parser.parse_args()

    if args.fast:
        args.warmup = 5
        args.iterations = 20

    provider_map = {
        "rocm": "ROCMExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    provider = provider_map[args.device]

    zoo = ModelZoo()

    # Select models
    if args.models:
        model_names = args.models
    else:
        model_names = []
        for domain in args.domains:
            model_names.extend(m.name for m in zoo.list_models(domain))

    # Limit to models with known input shapes
    model_names = [m for m in model_names if m in DEFAULT_SHAPES]

    logger.info("Benchmarking %d models with %s", len(model_names), provider)

    bench = MultiModelBenchmark()
    all_results = []

    for model_name in model_names:
        for batch_size in args.batch_sizes:
            logger.info("Benchmarking: %s (batch=%d)", model_name, batch_size)
            result = benchmark_model(
                model_name=model_name,
                zoo=zoo,
                provider=provider,
                warmup=args.warmup,
                iterations=args.iterations,
                batch_size=batch_size,
            )
            if result:
                all_results.append(result)
                bench.add_result(result)

    # Print comparison table
    print("\n" + bench.compare_table())

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s (%d models)", args.output, len(all_results))


if __name__ == "__main__":
    main()
