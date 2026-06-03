"""Tests for ONNX ROCm Runner."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Set env to suppress ROCm warnings during testing
os.environ.setdefault("ONNXRUNTIME_DISABLE_ROCM", "1")


class TestRunnerConfig:
    """Test RunnerConfig creation and validation."""

    def test_from_dict(self):
        from src.runner import RunnerConfig
        config = RunnerConfig(
            model_path="test.onnx",
            providers=["CPUExecutionProvider"],
            device_id=0,
        )
        assert config.model_path == "test.onnx"
        assert config.providers == ["CPUExecutionProvider"]
        assert config.device_id == 0

    def test_defaults(self):
        from src.runner import RunnerConfig
        config = RunnerConfig(model_path="test.onnx")
        assert config.providers == ["ROCMExecutionProvider"]
        assert config.num_warmup == 10
        assert config.num_iterations == 100

    def test_from_yaml(self):
        from src.runner import RunnerConfig
        import yaml

        data = {
            "model_path": "resnet50.onnx",
            "providers": ["ROCMExecutionProvider"],
            "num_warmup": 50,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()

            config = RunnerConfig.from_yaml(f.name)
            assert config.model_path == "resnet50.onnx"
            assert config.num_warmup == 50

        os.unlink(f.name)


class TestModelZoo:
    """Test ModelZoo functionality."""

    def test_list_models(self):
        from src.models import ModelZoo, MODEL_REGISTRY
        zoo = ModelZoo(cache_dir=Path(tempfile.mkdtemp()))
        models = zoo.list_models()
        assert len(models) == len(MODEL_REGISTRY)
        assert all(hasattr(m, "name") for m in models)

    def test_list_by_domain(self):
        from src.models import ModelZoo
        zoo = ModelZoo(cache_dir=Path(tempfile.mkdtemp()))
        vision = zoo.list_models(domain="vision")
        assert len(vision) > 0
        assert all(m.domain == "vision" for m in vision)

    def test_get_info(self):
        from src.models import ModelZoo
        zoo = ModelZoo(cache_dir=Path(tempfile.mkdtemp()))
        info = zoo.get_info("resnet50")
        assert info.name == "resnet50"
        assert info.domain == "vision"

    def test_get_info_unknown(self):
        from src.models import ModelZoo
        zoo = ModelZoo(cache_dir=Path(tempfile.mkdtemp()))
        with pytest.raises(KeyError):
            zoo.get_info("nonexistent_model")

    def test_cache_dir(self):
        from src.models import ModelZoo
        tmpdir = Path(tempfile.mkdtemp())
        zoo = ModelZoo(cache_dir=tmpdir)
        assert zoo.cache_dir.exists()
        assert zoo.get_cached_models() == []

    def test_download_nonexistent(self):
        from src.models import ModelZoo
        zoo = ModelZoo(cache_dir=Path(tempfile.mkdtemp()))
        result = zoo.download("nonexistent_model")
        assert result is None


class TestSessionConfig:
    """Test SessionConfig."""

    def test_to_provider_options(self):
        from src.session import SessionConfig
        config = SessionConfig(
            providers=["ROCMExecutionProvider"],
            device_id=0,
        )
        opts = config.to_provider_options()
        assert "device_id" in opts
        assert opts["device_id"] == 0

    def test_custom_memory_limit(self):
        from src.session import SessionConfig
        config = SessionConfig(memory_limit=8 * 1024**3)
        assert config.memory_limit == 8 * 1024**3


class TestBenchmark:
    """Test benchmark suite."""

    def test_result_summary(self):
        from src.benchmark import BenchmarkResult
        result = BenchmarkResult(
            model_name="test",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="fp32",
            batch_size=1,
            input_shapes={"input": [1, 3, 224, 224]},
            latency_mean_ms=1.5,
            latency_median_ms=1.4,
            latency_p95_ms=2.0,
            latency_p99_ms=2.5,
            latency_std_ms=0.3,
            latency_min_ms=1.0,
            latency_max_ms=3.0,
            throughput_inf_per_sec=666.67,
            gpu_memory_peak_mb=50.0,
        )
        summary = result.summary()
        assert "1.50ms" in summary
        assert "666.7" in summary

    def test_result_to_dict(self):
        from src.benchmark import BenchmarkResult
        result = BenchmarkResult(
            model_name="test",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="fp32",
            batch_size=1,
            input_shapes={"input": [1, 3, 224, 224]},
            latency_mean_ms=1.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "test"
        assert "latency" in d
        assert "throughput" in d
        assert "memory" in d


class TestOptimizer:
    """Test graph optimizer."""

    def test_optimization_result(self):
        from src.optimizer import OptimizationResult
        result = OptimizationResult(
            graph_modified=True,
            nodes_removed=5,
            nodes_added=0,
            nodes_fused=3,
            constants_folded=2,
            bytes_saved=1024,
            passes_applied=["constant_folding", "node_fusion"],
        )
        assert "5 nodes removed" in result.summary
        assert "3 fused" in result.summary

    def test_optimization_level(self):
        from src.optimizer import OptimizationLevel
        assert OptimizationLevel.NONE.value < OptimizationLevel.BASIC.value
        assert OptimizationLevel.BASIC.value < OptimizationLevel.EXTENDED.value
        assert OptimizationLevel.EXTENDED.value < OptimizationLevel.AGGRESSIVE.value


class TestProfiler:
    """Test profiler."""

    def test_operator_profile(self):
        from src.profiler import OperatorProfile
        op = OperatorProfile(
            name="conv1",
            op_type="Conv",
            provider="ROCMExecutionProvider",
            duration_ms=0.5,
            input_shapes=[[1, 3, 224, 224]],
            output_shapes=[[1, 64, 112, 112]],
        )
        d = op.to_dict()
        assert d["op_type"] == "Conv"
        assert d["duration_ms"] == 0.5

    def test_profile_result_summary(self):
        from src.profiler import ProfileResult, OperatorProfile
        result = ProfileResult(
            model_name="test",
            total_duration_ms=10.0,
            operators=[],
            kernel_times={"Conv": 5.0, "Relu": 3.0},
            provider_times={"ROCMExecutionProvider": 8.0, "CPUExecutionProvider": 2.0},
        )
        summary = result.summary()
        assert "test" in summary
        assert "Conv" in summary

    def test_chrome_trace_export(self):
        from src.profiler import ProfileResult, OperatorProfile, ONNXProfiler
        result = ProfileResult(
            model_name="test",
            total_duration_ms=10.0,
            operators=[
                OperatorProfile("conv1", "Conv", "ROCMExecutionProvider", 5.0),
                OperatorProfile("relu1", "Relu", "ROCMExecutionProvider", 3.0),
            ],
            kernel_times={"Conv": 5.0},
            provider_times={"ROCMExecutionProvider": 8.0},
        )
        trace = result.to_chrome_trace()
        assert "traceEvents" in trace
        assert len(trace["traceEvents"]) == 2


class TestExporter:
    """Test exporter configuration."""

    def test_export_config(self):
        from src.exporter import ExportConfig
        config = ExportConfig(
            opset_version=17,
            dynamic_batch=True,
            enable_fp16=True,
        )
        assert config.opset_version == 17
        assert config.dynamic_batch is True
        assert config.enable_fp16 is True

    def test_export_config_defaults(self):
        from src.exporter import ExportConfig
        config = ExportConfig()
        assert config.opset_version == 17
        assert config.dynamic_batch is True
        assert config.verify is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
