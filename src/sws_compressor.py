from __future__ import annotations
import torch
import copy
import typing as _T
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from util import log_mog_prob, responsibilities, log_beta_prob, log_inverse_gamma_prob


class SoftWeightSharingCompressor:
    """
    Compress a pretrained nn.Module with the soft weight-sharing algorithm.
    All tensor ops stay on `device`; no numpy is used so it is CUDA-friendly.
    """

    def __init__(
        self,
        tau: float = 5e-3,
        n_components: int = 17,
        zero_component_prior_alpha: _T.Optional[float] = None,
        zero_component_prior_beta: _T.Optional[float] = None,
        other_var_prior_alpha: _T.Optional[float] = None,  # Inv-Γ(α,β) on σ²ⱼ (j≠0)
        other_var_prior_beta: _T.Optional[float] = None,  # Inv-Γ(α,β) on σ²ⱼ (j≠0)
        lr_weights: float = 1e-4,
        lr_mixture: float = 5e-4,
        wd: float = 0.0,
        prune_threshold: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.tau = tau
        self.K = n_components
        self.device = device
        self.thr = prune_threshold
        self.wd = wd

        # hyper-priors
        self.zero_beta_alpha = torch.as_tensor(
            zero_component_prior_alpha, dtype=torch.float, device=device
        )
        self.zero_beta_beta = torch.as_tensor(
            zero_component_prior_beta, dtype=torch.float, device=device
        )
        self.inv_gamma_alpha = torch.as_tensor(
            other_var_prior_alpha, dtype=torch.float, device=device
        )
        self.inv_gamma_beta = torch.as_tensor(
            other_var_prior_beta, dtype=torch.float, device=device
        )

        # optimiser lr groups
        self.lr_w = lr_weights
        self.lr_mog = lr_mixture

    # --------------------------------------------------------------
    def __call__(
        self,
        model: nn.Module,
        loader: DataLoader,
        epochs: int = 40,
    ) -> tuple[dict, dict]:
        """
        Returns
        -------
        compressed_state : dict
            state_dict with *quantised* weights (float32 centres)
        code_book : dict
            {"centres": μ, "responsible": int-tensor with cluster indices}
        """
        model = model.to(self.device)
        flat_w, shapes, slices = self._flatten(model)
        mu, sigma, pi = self._init_mog(flat_w)
        opt = torch.optim.Adam(
            [
                {"params": [flat_w], "lr": self.lr_w, "weight_decay": self.wd},
                {"params": [mu, sigma, pi], "lr": self.lr_mog},
            ]
        )

        # ----------------- training loop -----------------
        le = torch.tensor(float("nan"), device=self.device)
        lc = torch.tensor(float("nan"), device=self.device)

        pbar = trange(1, epochs + 1, desc="Retraining")
        for epoch in pbar:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                le, lc = self._loss(model, x, y, flat_w, mu, sigma, pi)
                loss = le + self.tau * lc
                loss.backward()
                opt.step()
                # project simplex
                with torch.no_grad():
                    mu.clamp_min_(1e-6)
                    pi.div_(pi.sum())

            pbar.set_postfix(
                {
                    "Epoch": epoch,
                    r"$\mathcal{L}_E$": le.item(),
                    r"$\mathcal{L}_C$": lc.item(),
                }
            )

        # ----------------- post-process -----------------
        with torch.no_grad():
            flat_w_q, code = self._quantize(flat_w, mu, sigma, pi)
            mu, sigma, pi = self._merge_components(mu, sigma, pi)
            flat_w_q, code = self._quantize(flat_w, mu, sigma, pi)  # again after merge
        self._restore_shapes(model, flat_w_q, shapes, slices)
        code_book = {"centres": mu.cpu(), "responsible": code.cpu()}

        # Return mixture parameters for compression analysis
        mixture_params = {
            "mu": mu.cpu(),
            "sigma": sigma.cpu(),
            "pi": pi.cpu(),
            "J": len(mu),
        }
        return copy.deepcopy(model.state_dict()), code_book, mixture_params

    # --------------------------------------------------------------
    # internals
    # --------------------------------------------------------------
    def _flatten(self, model: nn.Module):
        """return 1-D tensor and meta-data to reconstruct state_dict"""
        w_list, shapes, slices = [], [], []
        for p in model.parameters():
            if p.requires_grad:
                w_list.append(p.detach().view(-1))
                shapes.append(p.shape)
                slices.append(len(w_list[-1]))
        flat = torch.cat(w_list).to(self.device)
        slices = torch.tensor([0] + slices).cumsum(0)
        return flat, shapes, slices

    def _restore_shapes(self, model, flat_w, shapes, slices):
        idx = 0
        for p, sh in zip(model.parameters(), shapes):
            p.data = flat_w[slices[idx] : slices[idx + 1]].reshape(sh)
            idx += 1

    def _init_mog(self, flat_w):
        K = self.K
        mu = torch.linspace(flat_w.min(), flat_w.max(), K).to(self.device)
        mu[0] = 0.0  # fixed zero component
        sigma = torch.ones(K).to(self.device) * 0.5
        pi = torch.ones(K).to(self.device) / K
        # make them learnable
        mu = nn.Parameter(mu)
        sigma = nn.Parameter(sigma)
        pi = nn.Parameter(pi)
        return mu, sigma, pi

    def _loss(self, model, x, y, flat_w, mu, sigma, pi):
        logits = model(x)
        le = torch.nn.functional.cross_entropy(logits, y)
        log_prior = log_mog_prob(flat_w, mu, sigma, pi).sum()
        lc = -log_prior
        lc -= log_beta_prob(pi[0], self.zero_beta_alpha, self.zero_beta_beta)
        lc -= log_inverse_gamma_prob(
            sigma[1:] ** 2, self.inv_gamma_alpha, self.inv_gamma_beta
        ).sum()

        return le, lc

    def _quantize(self, flat_w, μ, σ, π):
        γ = responsibilities(flat_w, μ, σ, π)
        best = γ.argmax(1)
        return μ[best], best

    def _merge_components(self, mu, sigma, pi):
        """KL-merge as in paper §4.3"""
        K = len(mu)
        done = False
        while not done:
            done = True
            for i in range(K):
                for j in range(i + 1, K):
                    if pi[i] < 1e-8 or pi[j] < 1e-8:
                        continue
                    kl = 0.5 * (
                        (sigma[i] ** 2 + (mu[i] - mu[j]) ** 2) / (sigma[j] ** 2 + 1e-8)
                        + (sigma[j] ** 2 + (mu[j] - mu[i]) ** 2)
                        / (sigma[i] ** 2 + 1e-8)
                        - 2
                        + 2 * torch.log(sigma[j] / sigma[i] + 1e-8).abs()
                    )
                    if kl < self.thr:
                        pi_new = pi[i] + pi[j]
                        mu_new = (pi[i] * mu[i] + pi[j] * mu[j]) / pi_new
                        var_new = (
                            pi[i] * sigma[i] ** 2 + pi[j] * sigma[j] ** 2
                        ) / pi_new
                        sigma_new = var_new.sqrt()
                        # merge into i
                        pi[i], mu[i], sigma[i] = pi_new, mu_new, sigma_new
                        pi[j] = 0.0
                        done = False
        keep = pi > 1e-8
        return mu[keep], sigma[keep], pi[keep]
