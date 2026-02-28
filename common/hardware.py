"""Hardware/runtime utilities for GH200-targeted training runs.

This module is the single source of truth for:
- conservative CPU thread environment defaults
- GH200/H100 CUDA arch handling
- torch backend flags and mixed precision policy
- optional FP8 probing behind an explicit flag
- NVML GPU utilization logging
- batch-size auto-tuning based on VRAM headroom
- optional prefetch-to-GPU loader wrapper
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


_THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS")


@dataclass(frozen=True)
class HardwareInfo:
    cuda_available: bool
    gpu_name: str
    compute_capability: Tuple[int, int]
    total_memory_bytes: int

    @property
    def is_hopper(self) -> bool:
        name = (self.gpu_name or "").lower()
        return self.compute_capability == (9, 0) or "h100" in name or "hopper" in name or "gh200" in name


@dataclass(frozen=True)
class FP8ProbeResult:
    enabled: bool
    reason: str


@dataclass(frozen=True)
class BatchTuningResult:
    batch_size: int
    reason: str
    measurements: List[Dict[str, float]]


class GPUUtilLogger:
    """Background NVML logger at 1 Hz (or caller-defined interval)."""

    def __init__(
        self,
        csv_path: Path,
        *,
        gpu_index: int = 0,
        interval_s: float = 1.0,
        tag: str = "",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self.tag = str(tag)
        self.logger = logger

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._nvml = None
        self._handle = None

    def start(self) -> bool:
        try:
            import pynvml  # type: ignore
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("NVML logger disabled (pynvml unavailable): %s", exc)
            return False

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning("NVML logger disabled (init failed): %s", exc)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return False

        self._nvml = pynvml
        self._handle = handle
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started = True
        if self.logger is not None:
            self.logger.info("Started GPU util logging -> %s", self.csv_path)
        return True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._started = False
        if self.logger is not None:
            self.logger.info("Stopped GPU util logging")

    def _run(self) -> None:
        assert self._nvml is not None
        assert self._handle is not None
        pynvml = self._nvml

        header = [
            "utc_iso",
            "unix_time",
            "tag",
            "gpu_index",
            "util_gpu_pct",
            "util_mem_pct",
            "mem_used_mb",
            "mem_total_mb",
            "power_w",
            "temp_c",
        ]

        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
                f.flush()

            while not self._stop.is_set():
                now = time.time()
                utc_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

                util_gpu = float("nan")
                util_mem = float("nan")
                mem_used_mb = float("nan")
                mem_total_mb = float("nan")
                power_w = float("nan")
                temp_c = float("nan")

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    util_gpu = float(util.gpu)
                    util_mem = float(util.memory)
                    mem_used_mb = float(mem.used) / (1024.0**2)
                    mem_total_mb = float(mem.total) / (1024.0**2)
                    try:
                        power_w = float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
                    except Exception:
                        power_w = float("nan")
                    try:
                        temp_c = float(
                            pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                        )
                    except Exception:
                        temp_c = float("nan")
                except Exception:
                    pass

                writer.writerow(
                    [
                        utc_iso,
                        f"{now:.3f}",
                        self.tag,
                        self.gpu_index,
                        f"{util_gpu:.3f}",
                        f"{util_mem:.3f}",
                        f"{mem_used_mb:.3f}",
                        f"{mem_total_mb:.3f}",
                        f"{power_w:.3f}",
                        f"{temp_c:.3f}",
                    ]
                )
                f.flush()
                self._stop.wait(self.interval_s)


def conservative_thread_count() -> int:
    """Return a conservative per-process thread budget.

    GH200 Grace CPUs expose many cores; we intentionally under-commit BLAS/OMP
    threads to avoid hidden oversubscription once DataLoader workers are active.
    """
    n = os.cpu_count() or 8
    if n <= 8:
        return max(2, n // 2)
    return max(4, min(24, n // 4))


def apply_cpu_thread_env(*, threads: Optional[int] = None, allow_override: bool = True) -> Dict[str, str]:
    t = int(threads or conservative_thread_count())

    out: Dict[str, str] = {}
    for key in _THREAD_ENV_KEYS:
        if allow_override and key in os.environ:
            out[key] = str(os.environ[key])
            continue
        os.environ[key] = str(t)
        out[key] = str(t)
    return out


def detect_hardware_info() -> HardwareInfo:
    try:
        import torch
    except Exception:
        return HardwareInfo(cuda_available=False, gpu_name="", compute_capability=(0, 0), total_memory_bytes=0)

    if not torch.cuda.is_available():
        return HardwareInfo(cuda_available=False, gpu_name="", compute_capability=(0, 0), total_memory_bytes=0)

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return HardwareInfo(
        cuda_available=True,
        gpu_name=str(props.name),
        compute_capability=(int(props.major), int(props.minor)),
        total_memory_bytes=int(props.total_memory),
    )


def configure_cuda_arch_list(*, for_extension_build: bool) -> str:
    """Set/unset TORCH_CUDA_ARCH_LIST according to detected GPU family.

    On GH200/H100 (SM90), we never hard-code Blackwell 12.0.
    - for runtime: unset TORCH_CUDA_ARCH_LIST
    - for extension build only: set to "9.0a+PTX"
    """
    info = detect_hardware_info()
    if not info.cuda_available:
        return "cuda-unavailable"

    if info.is_hopper:
        if for_extension_build:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a+PTX"
            return "set:9.0a+PTX"
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        return "unset"

    return "unchanged"


def configure_torch_backends(*, enable_tf32: bool = True) -> None:
    import torch

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        torch.backends.cudnn.benchmark = True


def probe_fp8_support(*, enable_fp8: bool, rel_err_threshold: float = 0.08) -> FP8ProbeResult:
    """Probe whether FP8 can be enabled safely on this runtime.

    We test both capability and a tiny numerical sanity check.
    """
    if not enable_fp8:
        return FP8ProbeResult(enabled=False, reason="disabled-by-flag")

    try:
        import torch
    except Exception as exc:
        return FP8ProbeResult(enabled=False, reason=f"torch-import-failed:{exc}")

    info = detect_hardware_info()
    if not info.cuda_available:
        return FP8ProbeResult(enabled=False, reason="cuda-unavailable")
    if not info.is_hopper:
        return FP8ProbeResult(enabled=False, reason="gpu-not-hopper")
    if not hasattr(torch, "float8_e4m3fn"):
        return FP8ProbeResult(enabled=False, reason="float8-dtype-missing")

    try:
        # Quick numerical sanity check: quantize/dequantize with float8 and
        # ensure relative error remains bounded versus fp16 baseline.
        x = torch.randn((1024, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 512), device="cuda", dtype=torch.float16)
        y_ref = (x @ w).float()

        x_q = x.to(torch.float8_e4m3fn).to(torch.float16)
        w_q = w.to(torch.float8_e4m3fn).to(torch.float16)
        y_q = (x_q @ w_q).float()

        rel = torch.mean(torch.abs(y_ref - y_q) / (torch.abs(y_ref) + 1e-6)).item()
        if rel <= float(rel_err_threshold):
            return FP8ProbeResult(enabled=True, reason=f"probe-pass-relerr={rel:.4f}")
        return FP8ProbeResult(enabled=False, reason=f"probe-fail-relerr={rel:.4f}")
    except Exception as exc:
        return FP8ProbeResult(enabled=False, reason=f"probe-exception:{exc}")


def resolve_autocast_dtype(*, amp: str, enable_fp8: bool) -> Tuple["torch.dtype", FP8ProbeResult]:
    import torch

    amp_norm = (amp or "bf16").strip().lower()
    fp8_probe = probe_fp8_support(enable_fp8=enable_fp8)

    if amp_norm in {"", "bf16", "default"}:
        return torch.bfloat16, fp8_probe
    if amp_norm == "fp16":
        return torch.float16, fp8_probe
    if amp_norm in {"fp8", "fp8_if_available"}:
        if fp8_probe.enabled:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn):
                    _ = torch.ones((1,), device="cuda")
                return torch.float8_e4m3fn, fp8_probe
            except Exception:
                return torch.bfloat16, FP8ProbeResult(
                    enabled=False,
                    reason=f"{fp8_probe.reason};autocast-fallback-bf16",
                )
        return torch.bfloat16, fp8_probe

    raise ValueError(f"Unknown amp mode: {amp}")


def start_gpu_util_logger(
    *,
    csv_path: Path,
    tag: str,
    logger: Optional[logging.Logger] = None,
    interval_s: float = 1.0,
) -> GPUUtilLogger:
    glog = GPUUtilLogger(csv_path=csv_path, tag=tag, interval_s=interval_s, logger=logger)
    glog.start()
    return glog


def current_vram_fraction() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    used = float(total - free)
    return float(used / max(total, 1))


def auto_tune_batch_size(
    *,
    initial_batch_size: int,
    probe_fn: Callable[[int], float],
    min_batch_size: int = 32,
    max_batch_size: Optional[int] = None,
    target_low: float = 0.85,
    target_high: float = 0.92,
    growth: float = 1.35,
    backoff: float = 0.90,
    max_trials: int = 12,
    logger: Optional[logging.Logger] = None,
) -> BatchTuningResult:
    """Tune batch size by probing peak VRAM fraction from `probe_fn`.

    `probe_fn(batch_size)` should run a minimal forward/backward step and return
    peak VRAM utilization fraction in [0, 1].
    """
    import torch

    bs = max(int(initial_batch_size), int(min_batch_size))
    best_bs = bs
    reason = "target-not-reached"
    measurements: List[Dict[str, float]] = []

    for _ in range(max_trials):
        if max_batch_size is not None:
            bs = min(bs, int(max_batch_size))

        try:
            frac = float(probe_fn(bs))
            measurements.append({"batch_size": float(bs), "vram_fraction": frac})

            if logger is not None:
                logger.info("Batch tuner probe: batch_size=%d vram_fraction=%.4f", bs, frac)

            if target_low <= frac <= target_high:
                best_bs = bs
                reason = "target-window"
                break

            if frac < target_low:
                best_bs = bs
                next_bs = int(max(bs + 1, round(bs * growth)))
                if max_batch_size is not None and next_bs > max_batch_size:
                    reason = "max-batch-hit"
                    break
                if next_bs == bs:
                    reason = "growth-stalled"
                    break
                bs = next_bs
                continue

            # frac > target_high
            backed = int(max(min_batch_size, round(bs * backoff)))
            best_bs = backed
            reason = "above-target-backoff"
            break

        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg:
                torch.cuda.empty_cache()
                backed = int(max(min_batch_size, round(best_bs * backoff)))
                best_bs = backed
                reason = "oom-backoff"
                if logger is not None:
                    logger.warning("Batch tuner OOM at batch_size=%d; backing off to %d", bs, backed)
                break
            raise

    if logger is not None:
        logger.info("Batch tuner selected batch_size=%d reason=%s", best_bs, reason)

    return BatchTuningResult(batch_size=int(best_bs), reason=reason, measurements=measurements)


def _move_to_device(batch: Any, device: "torch.device") -> Any:
    import torch

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        out = [_move_to_device(x, device) for x in batch]
        return type(batch)(out)
    return batch


class PrefetchToDeviceLoader:
    """Optional CUDA stream prefetch wrapper for DataLoader batches."""

    def __init__(self, loader: Iterable[Any], *, device: "torch.device", enabled: bool) -> None:
        import torch

        self.loader = loader
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda")
        self.stream = torch.cuda.Stream() if self.enabled else None

    def __iter__(self) -> Iterator[Any]:
        if not self.enabled:
            for batch in self.loader:
                yield batch
            return

        import torch

        iterator = iter(self.loader)
        next_batch: Optional[Any] = None

        def preload() -> Optional[Any]:
            try:
                b = next(iterator)
            except StopIteration:
                return None
            assert self.stream is not None
            with torch.cuda.stream(self.stream):
                return _move_to_device(b, self.device)

        next_batch = preload()
        while next_batch is not None:
            assert self.stream is not None
            torch.cuda.current_stream().wait_stream(self.stream)
            current = next_batch
            next_batch = preload()
            yield current


def auto_tune_gpu_parallel_procs(
    *,
    min_procs: int = 8,
    max_procs: int = 12,
    headroom_frac: float = 0.15,
    probe_bytes_per_proc: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Estimate a safe GPU process fan-out from free VRAM.

    This is intentionally conservative and fail-closed:
    - if CUDA is unavailable, returns 1 process
    - never exceeds [min_procs, max_procs] bounds on CUDA systems
    """
    try:
        import torch
    except Exception as exc:
        return {
            "selected_procs": 1,
            "reason": f"torch-import-failed:{exc}",
            "cuda_available": False,
        }

    if not torch.cuda.is_available():
        return {
            "selected_procs": 1,
            "reason": "cuda-unavailable",
            "cuda_available": False,
        }

    free_b, total_b = torch.cuda.mem_get_info()
    free_b = int(max(0, free_b))
    total_b = int(max(1, total_b))

    # Conservative default: ~6 GiB per process unless caller provides probe.
    default_probe = int(6 * (1024**3))
    per_proc = int(probe_bytes_per_proc) if probe_bytes_per_proc is not None else default_probe
    per_proc = max(int(1 * (1024**3)), per_proc)

    usable = int(max(0, free_b - headroom_frac * float(total_b)))
    raw = int(usable // per_proc) if per_proc > 0 else 1

    lo = max(1, int(min_procs))
    hi = max(lo, int(max_procs))
    selected = int(max(lo, min(hi, max(1, raw))))

    payload = {
        "selected_procs": int(selected),
        "reason": "vram-headroom-estimate",
        "cuda_available": True,
        "free_bytes": int(free_b),
        "total_bytes": int(total_b),
        "headroom_frac": float(headroom_frac),
        "probe_bytes_per_proc": int(per_proc),
        "raw_estimate": int(raw),
        "min_procs": int(lo),
        "max_procs": int(hi),
    }
    if logger is not None:
        logger.info(
            "GPU proc tuner: free=%.2fGiB total=%.2fGiB per_proc=%.2fGiB raw=%d selected=%d",
            float(free_b) / (1024.0**3),
            float(total_b) / (1024.0**3),
            float(per_proc) / (1024.0**3),
            int(raw),
            int(selected),
        )
    return payload


def summarize_gpu_util_csv(csv_path: Path) -> Dict[str, float]:
    """Summarize util/memory/power columns from GPUUtilLogger CSV."""
    out = {
        "rows": 0.0,
        "util_gpu_mean": float("nan"),
        "util_gpu_median": float("nan"),
        "util_mem_mean": float("nan"),
        "util_mem_median": float("nan"),
        "mem_used_mb_mean": float("nan"),
        "mem_used_mb_median": float("nan"),
        "power_w_mean": float("nan"),
        "power_w_median": float("nan"),
    }
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return out
    try:
        import pandas as pd
        import numpy as np
    except Exception:
        return out
    try:
        df = pd.read_csv(p)
    except Exception:
        return out
    if df.empty:
        return out

    out["rows"] = float(len(df))
    col_map = {
        "util_gpu_pct": "util_gpu",
        "util_mem_pct": "util_mem",
        "mem_used_mb": "mem_used_mb",
        "power_w": "power_w",
    }
    for col, key in col_map.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_median"] = float(np.median(vals))
    return out
