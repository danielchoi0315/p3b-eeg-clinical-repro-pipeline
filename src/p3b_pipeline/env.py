"""Environment and performance controls.

This module keeps backward-compatible wrappers while delegating the runtime policy
(single source of truth) to `common.hardware`.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional

from common.hardware import (
    apply_cpu_thread_env,
    configure_cuda_arch_list,
    configure_torch_backends,
    conservative_thread_count,
    detect_hardware_info,
)


@dataclass(frozen=True)
class ThreadConfig:
    """Thread budget for BLAS/OMP env vars.

    If fields are None, a conservative GH200-friendly default is used.
    """

    omp_num_threads: Optional[int] = None
    mkl_num_threads: Optional[int] = None
    openblas_num_threads: Optional[int] = None
    numexpr_max_threads: Optional[int] = None


def apply_thread_config(cfg: ThreadConfig, *, allow_override: bool = True) -> None:
    if (
        cfg.omp_num_threads is None
        and cfg.mkl_num_threads is None
        and cfg.openblas_num_threads is None
        and cfg.numexpr_max_threads is None
    ):
        apply_cpu_thread_env(threads=conservative_thread_count(), allow_override=allow_override)
        return

    mapping = {
        "OMP_NUM_THREADS": cfg.omp_num_threads,
        "MKL_NUM_THREADS": cfg.mkl_num_threads,
        "OPENBLAS_NUM_THREADS": cfg.openblas_num_threads,
        "NUMEXPR_MAX_THREADS": cfg.numexpr_max_threads,
    }
    for key, value in mapping.items():
        if value is None:
            continue
        if allow_override and key in os.environ:
            continue
        os.environ[key] = str(int(value))


def env_snapshot() -> Dict[str, Any]:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "TORCH_CUDA_ARCH_LIST",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        **{k: os.environ.get(k) for k in keys},
    }


def configure_torch_runtime(*, enable_tf32: bool = True) -> None:
    # Runtime path: do not force TORCH_CUDA_ARCH_LIST.
    configure_cuda_arch_list(for_extension_build=False)
    configure_torch_backends(enable_tf32=enable_tf32)


def require_cuda_or_die() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing GPU module execution to avoid silent CPU fallback."
        )


def cuda_device_summary() -> Dict[str, Any]:
    import torch

    info = detect_hardware_info()
    if not info.cuda_available:
        return {"cuda_available": False}

    idx = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_index": idx,
        "name": info.gpu_name,
        "total_memory_gb": round(info.total_memory_bytes / (1024**3), 3),
        "compute_capability": f"{info.compute_capability[0]}.{info.compute_capability[1]}",
        "is_hopper": info.is_hopper,
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }
