"""
PyTorch → ONNX Model Exporter.

Handles exporting PyTorch models to ONNX format with proper
dynamic axes, opset versions, and ROCm-specific optimizations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    from onnx import shape_inference
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    dynamic_batch: bool = True
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])
    do_constant_folding: bool = True
    export_params: bool = True
    verbose: bool = False
    enable_fp16: bool = False
    external_data_format: bool = False
    training_mode: bool = False
    verify: bool = True


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.

    Supports dynamic axes, FP16 conversion, opset selection,
    and automatic verification.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def export(
        self,
        model: "nn.Module",
        sample_input: Any,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export PyTorch model to ONNX.

        Args:
            model: PyTorch model
            sample_input: Example input tensor(s)
            output_path: Destination .onnx file

        Returns:
            Path to exported ONNX model
        """
        if not HAS_TORCH:
            raise ImportError("torch required for export")
        if not HAS_ONNX:
            raise ImportError("onnx required for export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()

        # Build dynamic axes config
        dynamic_axes = self.config.dynamic_axes
        if dynamic_axes is None and self.config.dynamic_batch:
            dynamic_axes = {}
            if isinstance(sample_input, (tuple, list)):
                for i, name in enumerate(self.config.input_names):
                    if i < len(sample_input):
                        dynamic_axes[name] = {0: "batch_size"}
            elif isinstance(sample_input, torch.Tensor):
                dynamic_axes[self.config.input_names[0]] = {0: "batch_size"}

        # FP16 conversion
        if self.config.enable_fp16:
            model = model.half()
            if isinstance(sample_input, torch.Tensor):
                sample_input = sample_input.half()
            elif isinstance(sample_input, (tuple, list)):
                sample_input = tuple(
                    t.half() if isinstance(t, torch.Tensor) else t
                    for t in sample_input
                )

        # Export
        logger.info("Exporting model to %s (opset=%d)", output_path, self.config.opset_version)

        if isinstance(sample_input, (tuple, list)):
            export_input = sample_input
        else:
            export_input = (sample_input,)

        torch.onnx.export(
            model,
            export_input,
            str(output_path),
            opset_version=self.config.opset_version,
            dynamic_axes=dynamic_axes or {},
            input_names=self.config.input_names[:len(export_input)],
            output_names=self.config.output_names,
            do_constant_folding=self.config.do_constant_folding,
            export_params=self.config.export_params,
            training=torch.onnx.TrainingMode.EVAL if not self.config.training_mode else torch.onnx.TrainingMode.TRAINING,
            verbose=self.config.verbose,
        )

        # Post-processing
        self._post_process(str(output_path))

        # Verification
        if self.config.verify:
            self._verify(str(output_path))

        logger.info("Export complete: %s", output_path)
        return output_path

    def _post_process(self, model_path: str) -> None:
        """Apply post-export optimizations."""
        try:
            model = onnx.load(model_path)
            model = shape_inference.infer_shapes(model)
            model = onnx.compose.add_prefix(model, prefix="")

            # Add metadata
            model.producer_name = "onnx-rocm-runner"
            model.producer_version = "1.0.0"
            model.domain = "onnx-rocm-runner"

            onnx.save(model, model_path)
        except Exception as e:
            logger.warning("Post-processing failed: %s", e)

    def _verify(self, model_path: str) -> bool:
        """Verify exported model is valid."""
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model verification passed")

            # Print model info
            logger.info(
                "  Nodes: %d, Inputs: %d, Outputs: %d",
                len(model.graph.node),
                len(model.graph.input),
                len(model.graph.output),
            )
            return True
        except onnx.checker.ValidationError as e:
            logger.error("Model verification failed: %s", e)
            return False

    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Get metadata from an ONNX model file."""
        if not HAS_ONNX:
            raise ImportError("onnx required")

        model = onnx.load(model_path)
        graph = model.graph

        inputs = []
        for inp in graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else f"dynamic:{dim.dim_param}")
            inputs.append({"name": inp.name, "shape": shape})

        outputs = []
        for out in graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else f"dynamic:{dim.dim_param}")
            outputs.append({"name": out.name, "shape": shape})

        op_types = {}
        for node in graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

        return {
            "ir_version": model.ir_version,
            "opset_import": model.opset_import[0].version if model.opset_import else 0,
            "producer": model.producer_name,
            "nodes": len(graph.node),
            "inputs": inputs,
            "outputs": outputs,
            "op_distribution": dict(sorted(op_types.items(), key=lambda x: -x[1])),
            "file_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
        }

    @staticmethod
    def convert_fp32_to_fp16(model_path: str, output_path: str) -> None:
        """Convert an ONNX model from FP32 to FP16."""
        if not HAS_ONNX:
            raise ImportError("onnx required")

        import numpy as np

        model = onnx.load(model_path)

        # Convert initializers
        for init in model.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT:
                arr = onnx.numpy_helper.to_array(init)
                new_init = onnx.numpy_helper.from_array(
                    arr.astype(np.float16), name=init.name
                )
                init.CopyFrom(new_init)

        # Convert node inputs to float16 for relevant ops
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                    attr.i = onnx.TensorProto.FLOAT16

        onnx.save(model, output_path)
        logger.info("FP16 conversion complete: %s", output_path)

    @staticmethod
    def batch_export(
        models: Dict[str, "nn.Module"],
        sample_inputs: Dict[str, Any],
        output_dir: Union[str, Path],
        config: Optional[ExportConfig] = None,
    ) -> Dict[str, Path]:
        """Export multiple models to ONNX."""
        exporter = ONNXExporter(config)
        results = {}

        for name, model in models.items():
            if name not in sample_inputs:
                logger.warning("No sample input for %s, skipping", name)
                continue

            output_path = Path(output_dir) / f"{name}.onnx"
            try:
                path = exporter.export(model, sample_inputs[name], output_path)
                results[name] = path
            except Exception as e:
                logger.error("Export failed for %s: %s", name, e)

        return results
