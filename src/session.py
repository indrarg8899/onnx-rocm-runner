"""
ROCm Execution Provider session management.

Handles creation, configuration, and lifecycle of ONNX Runtime
sessions with ROCm GPU acceleration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors when ROCm isn't available
_ort = None


def _get_ort():
    """Lazy-load onnxruntime."""
    global _ort
    if _ort is None:
        try:
            import onnxruntime as ort
            _ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime-rocm"
            )
    return _ort


@dataclass
class SessionConfig:
    """Configuration for ONNX Runtime session."""
    providers: List[str] = field(default_factory=lambda: ["ROCMExecutionProvider"])
    device_id: int = 0
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    thread_count: int = 1
    memory_limit: int = 0  # bytes, 0 = unlimited
    arena_extend_strategy: str = "kNextPowerOfTwo"
    enable_profiling: bool = False
    profiling_file_prefix: str = "onnx_profile"
    do_copy_in_default_stream: bool = True
    has_user_compute_stream: bool = False
    external_stream: Optional[int] = None
    execution_mode: str = "ORT_SEQUENTIAL"
    cuda_mem_limit: int = 0  # Deprecated alias for memory_limit
    allow_mem_pattern: bool = True
    enable_cpu_mem_arena: bool = True
    log_severity_level: int = 3  # 0=VERBOSE, 3=WARNING
    log_verbosity_level: int = 0

    def to_session_options(self) -> Any:
        """Convert to onnxruntime.SessionOptions."""
        ort = _get_ort()
        opts = ort.SessionOptions()

        # Graph optimization
        opt_map = {
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        opts.graph_optimization_level = opt_map.get(
            self.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )

        opts.intra_op_num_threads = self.thread_count
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.do_copy_in_default_stream = self.do_copy_in_default_stream

        # Memory configuration
        if self.memory_limit > 0:
            opts.memory_limit = self.memory_limit

        # Memory arena
        arena_map = {
            "kNextPowerOfTwo": ort.ArenaExtendStrategy.kNextPowerOfTwo,
            "kSameAsRequested": ort.ArenaExtendStrategy.kSameAsRequested,
        }
        opts.arena_extend_strategy = arena_map.get(
            self.arena_extend_strategy,
            ort.ArenaExtendStrategy.kNextPowerOfTwo,
        )
        opts.allow_mem_pattern = self.allow_mem_pattern
        opts.enable_cpu_mem_arena = self.enable_cpu_mem_arena

        # Profiling
        opts.enable_profiling = self.enable_profiling
        opts.profile_file_prefix = self.profiling_file_prefix

        # Logging
        opts.log_severity_level = self.log_severity_level
        opts.log_verbosity_level = self.log_verbosity_level

        return opts

    def to_provider_options(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """Build ROCm provider options."""
        device = device_id if device_id is not None else self.device_id
        opts: Dict[str, Any] = {
            "device_id": device,
            "gpu_mem_limit": self.cuda_mem_limit or self.memory_limit or (1 << 31),
            "do_copy_in_default_stream": self.do_copy_in_default_stream,
        }
        if self.has_user_compute_stream and self.external_stream:
            opts["has_user_compute_stream"] = True
            opts["user_compute_stream"] = self.external_stream
        return opts


class ROCmSessionManager:
    """
    Manages ONNX Runtime sessions with ROCm execution provider.

    Handles session creation, provider fallback, and resource cleanup.
    """

    def __init__(self, config: SessionConfig):
        self.config = config
        self._session: Optional[Any] = None
        self._available_providers: List[str] = []
        self._active_provider: str = ""

    def _check_providers(self) -> List[str]:
        """Check available execution providers and resolve fallbacks."""
        ort = _get_ort()
        available = set(ort.get_available_providers())
        self._available_providers = list(available)

        resolved = []
        for provider in self.config.providers:
            if provider in available:
                resolved.append(provider)
                logger.info("Provider available: %s", provider)
            else:
                logger.warning(
                    "Provider '%s' not available. Options: %s",
                    provider,
                    ", ".join(sorted(available)),
                )
                # Try fallback
                fallbacks = {
                    "ROCMExecutionProvider": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    "CUDAExecutionProvider": ["CPUExecutionProvider"],
                }
                for fb in fallbacks.get(provider, []):
                    if fb in available:
                        resolved.append(fb)
                        logger.info("Fallback to: %s", fb)
                        break

        if not resolved:
            resolved = ["CPUExecutionProvider"]
            logger.warning("No GPU provider available, using CPU")

        self._active_provider = resolved[0]
        return resolved

    def create_session(self, model_path: str) -> Any:
        """Create and return an inference session."""
        ort = _get_ort()
        providers = self._check_providers()
        session_options = self.config.to_session_options()

        # Build provider options list
        provider_options = []
        for p in providers:
            if p == "ROCMExecutionProvider":
                provider_options.append(self.config.to_provider_options())
            elif p == "CUDAExecutionProvider":
                provider_options.append(self.config.to_provider_options())
            else:
                provider_options.append({})

        logger.info(
            "Creating session: model=%s, providers=%s",
            model_path,
            providers,
        )

        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )

        actual_providers = self._session.get_providers()
        logger.info("Active providers: %s", actual_providers)

        return self._session

    @property
    def session(self) -> Optional[Any]:
        return self._session

    @property
    def active_provider(self) -> str:
        return self._active_provider

    @property
    def available_providers(self) -> List[str]:
        return self._available_providers

    def switch_provider(self, provider: str) -> None:
        """Switch execution provider on existing session."""
        if self._session is None:
            raise RuntimeError("No active session")

        available = set(self._session.get_providers())
        if provider not in available:
            raise ValueError(
                f"Provider '{provider}' not in session. Available: {available}"
            )
        self._session.set_providers([provider])
        self._active_provider = provider
        logger.info("Switched to provider: %s", provider)

    def get_session_info(self) -> Dict[str, Any]:
        """Get session metadata."""
        if self._session is None:
            return {"status": "not_initialized"}

        return {
            "providers": self._session.get_providers(),
            "active_provider": self._active_provider,
            "input_count": len(self._session.get_inputs()),
            "output_count": len(self._session.get_outputs()),
            "input_names": [i.name for i in self._session.get_inputs()],
            "output_names": [o.name for o in self._session.get_outputs()],
        }

    def close(self) -> None:
        """Release session resources."""
        if self._session is not None:
            # ONNX Runtime sessions don't have explicit close,
            # but we clear the reference for GC
            self._session = None
            self._active_provider = ""
            logger.info("Session manager closed")

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        status = self._active_provider if self._session else "not initialized"
        return f"ROCmSessionManager(provider={status})"


def check_rocm_environment() -> Dict[str, Any]:
    """Check ROCm runtime environment and return status."""
    info = {
        "rocm_home": os.environ.get("ROCM_HOME", ""),
        "hip_path": os.environ.get("HIP_PATH", ""),
        "device_count": 0,
        "provider_available": False,
    }

    try:
        ort = _get_ort()
        available = ort.get_available_providers()
        info["provider_available"] = "ROCMExecutionProvider" in available
        info["all_providers"] = list(available)
    except Exception as e:
        info["error"] = str(e)

    try:
        import rocm
        info["rocm_version"] = getattr(rocm, "__version__", "unknown")
    except ImportError:
        pass

    return info
