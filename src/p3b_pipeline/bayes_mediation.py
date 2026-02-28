"""Hierarchical Bayesian mediation model (Torch GPU, variational inference).

Goal:
  Test whether pupil dilation response (PDR; LC–NE proxy) mediates the effect of
  memory load on single-trial P3b amplitude.

Model (standardized variables; per trial t, subject i):

  PDR_it  ~ Normal( a0_i + a1 * Load_it + a2 * Age_i + a3 * TrialOrder_it,  sigma_pdr )
  P3b_it  ~ Normal( c0_i + c' * Load_it + b  * PDR_it + g2 * Age_i + g3 * TrialOrder_it, sigma_p3b )

Hierarchical priors:
  a0_i ~ Normal(mu_a0, sigma_a0)
  c0_i ~ Normal(mu_c0, sigma_c0)
  slopes ~ Normal(0, 1)
  sigmas ~ HalfNormal(1)

We fit with mean-field variational inference (ELBO) on GPU.
This is designed for massive trial counts (mini-batching) and torch.compile.

Notes on correctness:
- Because we parameterize positive scales via softplus(u), we include the
  Jacobian term log|d softplus(u)/du| in the log-prior to respect the prior
  defined on sigma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_normal(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Elementwise log N(x|mu,sigma)."""
    # Work in float32 for numerical stability even under autocast.
    x = x.float()
    mu = mu.float()
    sigma = sigma.float().clamp_min(1e-6)
    return -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _log_half_normal_sigma(sigma: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Log density of HalfNormal(sigma; scale) for sigma>0 (elementwise)."""
    sigma = sigma.float().clamp_min(1e-12)
    # half-normal: p(s) = sqrt(2)/(scale*sqrt(pi)) * exp(-s^2/(2*scale^2)) for s>=0
    return (
        0.5 * math.log(2.0)
        - math.log(scale)
        - 0.5 * math.log(math.pi)
        - (sigma**2) / (2.0 * (scale**2))
    )


def _softplus_with_jacobian(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sigma=softplus(u), log|d sigma/du|)."""
    sigma = F.softplus(u)
    # derivative of softplus is sigmoid
    log_j = torch.log(torch.sigmoid(u).clamp_min(1e-12))
    return sigma, log_j


@dataclass
class MediationBatch:
    subj_idx: torch.Tensor  # (B,) int64
    load_z: torch.Tensor    # (B,) float
    age_z: torch.Tensor     # (B,) float
    order_z: torch.Tensor   # (B,) float
    pdr_z: torch.Tensor     # (B,) float
    p3b_z: torch.Tensor     # (B,) float


class HierarchicalMediationVI(nn.Module):
    """Mean-field VI for the hierarchical mediation model."""

    def __init__(self, n_subjects: int):
        super().__init__()
        self.n_subjects = int(n_subjects)

        # --- Variational parameters: q(theta) = Normal(mu, sigma) ---
        # Global parameters
        self.q_mu = nn.ParameterDict(
            {
                # intercept hypermeans
                "mu_a0": nn.Parameter(torch.zeros(())),
                "mu_c0": nn.Parameter(torch.zeros(())),
                # slopes
                "a_load": nn.Parameter(torch.zeros(())),
                "a_age": nn.Parameter(torch.zeros(())),
                "a_order": nn.Parameter(torch.zeros(())),
                "b_pdr": nn.Parameter(torch.zeros(())),
                "c_load": nn.Parameter(torch.zeros(())),  # c'
                "g_age": nn.Parameter(torch.zeros(())),
                "g_order": nn.Parameter(torch.zeros(())),
                # raw scales (unconstrained); transformed via softplus
                "u_sigma_a0": nn.Parameter(torch.zeros(())),
                "u_sigma_c0": nn.Parameter(torch.zeros(())),
                "u_sigma_pdr": nn.Parameter(torch.zeros(())),
                "u_sigma_p3b": nn.Parameter(torch.zeros(())),
                # subject intercepts
                "a0_subj": nn.Parameter(torch.zeros((self.n_subjects,))),
                "c0_subj": nn.Parameter(torch.zeros((self.n_subjects,))),
            }
        )
        self.q_log_std = nn.ParameterDict({k: nn.Parameter(torch.full_like(v, -3.0)) for k, v in self.q_mu.items()})

    def sample_params(self) -> Dict[str, torch.Tensor]:
        params: Dict[str, torch.Tensor] = {}
        for k in self.q_mu.keys():
            mu = self.q_mu[k]
            std = torch.exp(self.q_log_std[k]).clamp_min(1e-6)
            eps = torch.randn_like(mu)
            params[k] = mu + std * eps
        return params

    def log_q(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for k, v in params.items():
            mu = self.q_mu[k]
            std = torch.exp(self.q_log_std[k]).clamp_min(1e-6)
            total = total + _log_normal(v, mu, std).sum()
        return total

    def log_prior(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Priors are defined on transformed sigma parameters where relevant.
        total = torch.zeros((), device=next(self.parameters()).device)

        # hypermeans ~ N(0,1)
        total = total + _log_normal(params["mu_a0"], torch.zeros(()), torch.ones(())).sum()
        total = total + _log_normal(params["mu_c0"], torch.zeros(()), torch.ones(())).sum()

        # slopes ~ N(0,1)
        for k in ["a_load", "a_age", "a_order", "b_pdr", "c_load", "g_age", "g_order"]:
            total = total + _log_normal(params[k], torch.zeros(()), torch.ones(())).sum()

        # hierarchical stds ~ HalfNormal(1), with change-of-variables correction
        for u_name in ["u_sigma_a0", "u_sigma_c0", "u_sigma_pdr", "u_sigma_p3b"]:
            sigma, log_j = _softplus_with_jacobian(params[u_name])
            total = total + _log_half_normal_sigma(sigma, scale=1.0).sum() + log_j.sum()

        sigma_a0, _ = _softplus_with_jacobian(params["u_sigma_a0"])
        sigma_c0, _ = _softplus_with_jacobian(params["u_sigma_c0"])

        # subject intercepts: a0_i ~ N(mu_a0, sigma_a0), c0_i ~ N(mu_c0, sigma_c0)
        total = total + _log_normal(params["a0_subj"], params["mu_a0"], sigma_a0).sum()
        total = total + _log_normal(params["c0_subj"], params["mu_c0"], sigma_c0).sum()

        return total

    def log_likelihood(self, params: Dict[str, torch.Tensor], batch: MediationBatch) -> torch.Tensor:
        sigma_pdr, _ = _softplus_with_jacobian(params["u_sigma_pdr"])
        sigma_p3b, _ = _softplus_with_jacobian(params["u_sigma_p3b"])

        a0 = params["a0_subj"][batch.subj_idx]
        c0 = params["c0_subj"][batch.subj_idx]

        mu_pdr = a0 + params["a_load"] * batch.load_z + params["a_age"] * batch.age_z + params["a_order"] * batch.order_z
        mu_p3b = (
            c0
            + params["c_load"] * batch.load_z
            + params["b_pdr"] * batch.pdr_z
            + params["g_age"] * batch.age_z
            + params["g_order"] * batch.order_z
        )

        ll_pdr = _log_normal(batch.pdr_z, mu_pdr, sigma_pdr).sum()
        ll_p3b = _log_normal(batch.p3b_z, mu_p3b, sigma_p3b).sum()
        return ll_pdr + ll_p3b

    def forward(self, batch: MediationBatch, *, scale_factor: float) -> torch.Tensor:
        """Return the **negative ELBO** (loss) for a batch."""
        params = self.sample_params()
        logp = self.log_prior(params) + scale_factor * self.log_likelihood(params, batch)
        logq = self.log_q(params)
        elbo = logp - logq
        return -elbo

    @torch.no_grad()
    def sample_mediation_effect(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw mediation effect samples: a_load * b_pdr."""
        out = torch.empty((n,), device=device, dtype=torch.float32)
        for i in range(n):
            p = self.sample_params()
            out[i] = (p["a_load"] * p["b_pdr"]).float()
        return out
