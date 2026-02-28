"""Normative modeling (Torch GPU) for expected P3b amplitude distribution.

We implement a practical, scalable alternative to GP regression: **deep ensembles**
of heteroscedastic regressors.

Input covariates (as requested):
  x = [memory_load, age, trial_order]  (+ optional extras if you add them)

Output:
  Predictive distribution p(y | x) ~ Normal(mu(x), sigma^2(x))

Ensemble uncertainty is computed with the law of total variance:
  Var[y|x] = E_k[ sigma_k^2(x) + mu_k(x)^2 ] - (E_k[mu_k(x)])^2
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.hardware import PrefetchToDeviceLoader
from .torch_utils import safe_compile


class HeteroscedasticMLP(nn.Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, 2)  # mu, log_var

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        out = self.head(h)
        mu = out[:, 0]
        log_var = out[:, 1].clamp(min=-12.0, max=12.0)
        return mu, log_var


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Mean negative log likelihood of a Normal with predicted log-variance."""
    # Stabilize in float32
    y = y.float()
    mu = mu.float()
    log_var = log_var.float()
    var = torch.exp(log_var).clamp_min(1e-6)
    return 0.5 * (log_var + (y - mu) ** 2 / var + math.log(2.0 * math.pi)).mean()


@torch.no_grad()
def predict_ensemble(models: List[HeteroscedasticMLP], x: torch.Tensor, *, amp_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_mean, pred_var) for an ensemble."""
    device = x.device
    mus = []
    second_moments = []
    for m in models:
        m.eval()
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
            mu, log_var = m(x)
        var = torch.exp(log_var.float()).clamp_min(1e-6)
        mus.append(mu.float())
        second_moments.append(var + mu.float() ** 2)

    mu_ens = torch.stack(mus, dim=0).mean(dim=0)
    second_moment = torch.stack(second_moments, dim=0).mean(dim=0)
    var_ens = (second_moment - mu_ens**2).clamp_min(1e-6)
    return mu_ens, var_ens


def save_model_atomic(path: Path, model: nn.Module, optim: torch.optim.Optimizer, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "meta": meta,
    }
    torch.save(payload, tmp)
    tmp.replace(path)


def load_model_checkpoint(path: Path, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optim is not None and "optim_state" in payload:
        optim.load_state_dict(payload["optim_state"])
    return payload.get("meta", {})


def train_one_member(
    *,
    member_idx: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg: Dict[str, object],
    out_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    compile_model: bool,
    resume: bool,
    data_seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> HeteroscedasticMLP:
    """Train a single ensemble member (with checkpoint/resume)."""
    hidden_sizes = list(cfg.get("hidden_sizes", [128, 128, 64]))
    dropout = float(cfg.get("dropout", 0.05))
    lr = float(cfg.get("lr", 2e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    batch_size = int(cfg.get("batch_size", 16384))
    num_workers = int(cfg.get("num_workers", 20))
    max_epochs = int(cfg.get("max_epochs", 80))
    seed = int(cfg.get("seed", 0)) + 1000 * member_idx
    if data_seed is None:
        data_seed = seed
    n_train = int(x_train.shape[0])
    if n_train <= 0:
        raise RuntimeError("Normative training received zero rows.")
    effective_batch_size = max(1, min(batch_size, n_train))
    if effective_batch_size != batch_size and logger is not None:
        logger.warning(
            "Clamped batch_size from %d to %d due to n_train=%d",
            batch_size,
            effective_batch_size,
            n_train,
        )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = HeteroscedasticMLP(in_features=x_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(device)
    model = safe_compile(
        model,
        enabled=compile_model,
        backend="inductor",
        logger=logger,
        context=f"normative.member_{member_idx:02d}",
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_path = out_dir / f"member_{member_idx:02d}" / "checkpoint.pt"
    start_epoch = 0
    if resume and ckpt_path.exists():
        meta = load_model_checkpoint(ckpt_path, model, optim)
        start_epoch = int(meta.get("epoch", 0)) + 1

    # DataLoader (CPU -> GPU streaming)
    ds = torch.utils.data.TensorDataset(x_train, y_train)

    gen = torch.Generator()
    gen.manual_seed(int(data_seed) + 7919 + member_idx)

    def _worker_init_fn(worker_id: int) -> None:
        import random

        worker_seed = int(data_seed) + 100000 * member_idx + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,  # avoid dynamic shape recompiles
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=gen,
    )
    if len(dl) <= 0:
        raise RuntimeError(
            f"Empty DataLoader for normative member={member_idx} (n_train={n_train}, batch_size={effective_batch_size}, drop_last=True)."
        )
    prefetch_to_gpu = bool(cfg.get("prefetch_to_gpu", True))
    stream_loader = PrefetchToDeviceLoader(dl, device=device, enabled=prefetch_to_gpu)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        losses = []
        for xb, yb in stream_loader:
            if xb.device.type != device.type:
                xb = xb.to(device, non_blocking=True)
            if yb.device.type != device.type:
                yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                mu, log_var = model(xb)
                loss = gaussian_nll(yb, mu, log_var)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            losses.append(float(loss.detach().cpu()))

        # checkpoint every epoch
        save_model_atomic(
            ckpt_path,
            model,
            optim,
            meta={
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)) if losses else float("nan"),
                "batch_size": int(effective_batch_size),
                "prefetch_to_gpu": bool(prefetch_to_gpu),
            },
        )

    return model
