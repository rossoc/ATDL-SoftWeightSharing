import torch
from torch import Tensor


@torch.jit.script
def log_mog_prob(w: Tensor, mu: Tensor, sigma: Tensor, pi: Tensor) -> Tensor:
    """log p(w) under a factorised MoG prior (one scalar per weight)"""
    w = w.unsqueeze(-1)  # [W,1]
    log_norm = -0.5 * (((w - mu) / sigma).square() + (2 * sigma.log()).mul(2))  # [W,K]
    log_norm += pi.log()  # [W,K]
    return torch.logsumexp(log_norm, dim=-1)  # [W]


@torch.jit.script
def responsibilities(w: Tensor, mu: Tensor, sigma: Tensor, pi: Tensor) -> Tensor:
    """γ_ik = p(z_i=k | w_i)  (used for quantisation)"""
    w = w.unsqueeze(-1)
    log_y = -0.5 * (((w - mu) / sigma).square() + (2 * sigma.log()).mul(2)) + pi.log()
    y = (log_y - torch.logsumexp(log_y, dim=-1, keepdim=True)).exp()
    return y  # [W,K]


def log_beta_prob(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """
    Log-probability of Beta(alpha, beta) at x ∈ (0,1).
    Works for batched x; constants are broadcast.
    """
    x = x.clamp(1e-8, 1 - 1e-8)  # numeric guard
    log_norm = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    return (alpha - 1) * x.log() + (beta - 1) * (1 - x).log() - log_norm


def log_inverse_gamma_prob(x2: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """
    Log-probability of Inverse-Gamma(alpha, beta) at x² (scale parameter).
    x2 can be σ² or any positive tensor.
    """
    log_norm = alpha * beta.log() - torch.lgamma(alpha)
    return -(alpha + 1) * x2.log() - beta / x2 - log_norm
