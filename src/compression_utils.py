import math, collections
from typing import Dict, Tuple
import torch
import torch.nn as nn


def _shannon_bits(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    bits = 0.0
    for c in counts.values():
        p = c / total
        bits += -c * math.log2(max(p, 1e-12))
    return bits


def flatten_conv_to_2d(W: torch.Tensor) -> torch.Tensor:
    """
    Flatten convolutional weights to 2D matrix.
    
    For a conv layer with shape (out_channels, in_channels, k_h, k_w),
    reshape to (out_channels, in_channels * k_h * k_w).
    """
    if W.dim() == 4:  # Conv layer weights (out_ch, in_ch, k_h, k_w)
        out_ch, in_ch, k_h, k_w = W.shape
        return W.view(out_ch, in_ch * k_h * k_w)
    elif W.dim() == 2:  # Linear layer weights
        return W
    else:
        # For any other shape, flatten all but the first dimension
        return W.view(W.size(0), -1)


def _csr_bits_for_layer(
    W2d: torch.Tensor,
    comp_ids: torch.Tensor,
    J: int,
    is_conv: bool,
    pbits_fc: int = 5,
    pbits_conv: int = 8,
    use_huffman: bool = True,
) -> Tuple[int, int, int, int, int]:
    """
    Returns (bits_IR, bits_IC, bits_A, bits_codebook, nnz)
    """
    W = W2d.detach().cpu().numpy()
    comp = comp_ids.detach().cpu().numpy()

    rows, cols = W.shape
    nonzeros = []
    rowptr = [0]
    colidx = []
    for r in range(rows):
        nz_cols = (comp[r] > 0).nonzero()[0]
        nz_ids = comp[r, nz_cols]
        nonzeros.extend(list(nz_ids))
        colidx.extend(list(nz_cols))
        rowptr.append(len(nonzeros))

    nnz = len(nonzeros)

    # IR: row pointer counts in fixed bit budget
    p_ir = max(1, int(math.ceil(math.log2(max(nnz, 1) + 1))))
    bits_IR = (rows + 1) * p_ir

    # IC: relative column indices with p-bit diffs (padded for long gaps)
    pbits = pbits_conv if is_conv else pbits_fc
    span = (1 << pbits) - 1
    ic_diffs = []
    padded_nonzeros = []
    for r in range(rows):
        s, e = rowptr[r], rowptr[r + 1]
        prev_c = 0
        for k in range(s, e):
            c = colidx[k]
            diff = c - prev_c
            while diff > span:
                ic_diffs.append(span)
                padded_nonzeros.append(0)
                diff -= span
                prev_c += span
            ic_diffs.append(diff)
            padded_nonzeros.append(nonzeros[k])
            prev_c = c

    if use_huffman:
        counts = collections.Counter(ic_diffs)
        bits_IC = int(round(_shannon_bits(counts)))
    else:
        bits_IC = len(ic_diffs) * pbits

    # A: Huffman on non-zero codebook indices (or fixed log2 J)
    nz_vals = [z for z in padded_nonzeros if z != 0]
    if use_huffman:
        countsA = collections.Counter(nz_vals)
        bits_A = int(round(_shannon_bits(countsA)))
    else:
        bits_A = len(nz_vals) * int(math.ceil(math.log2(max(2, J))))

    # codebook: (J-1) stored 32-bit means (zero is implicit)
    bits_codebook = (J - 1) * 32
    return bits_IR, bits_IC, bits_A, bits_codebook, nnz


class Prior:
    """
    A wrapper class to hold mixture model parameters that can be used with the compression_report function.
    """
    def __init__(self, mu: torch.Tensor, sigma2: torch.Tensor, pi: torch.Tensor):
        self.mu = mu
        self.sigma2 = sigma2  # variance (not standard deviation)
        self.pi = pi
        self.J = len(mu)  # number of components

    def mixture_params(self):
        """
        Return the mixture model parameters.
        Returns mu, sigma^2 (variance), and pi.
        """
        return self.mu, self.sigma2, self.pi


@torch.no_grad()
def compression_report(
    model: torch.nn.Module,
    prior,
    dataset: str,
    use_huffman: bool = True,
    pbits_fc: int = 5,
    pbits_conv: int = 8,
    skip_last_matrix: bool = False,
    assign_mode: str = "ml",  # IMPORTANT: match quantization (ml or map)
) -> Dict:
    """
    Compute Han-style CSR bit cost using mixture assignments for all layers,
    except (optionally) the last 2D weight, which can be treated as 32-bit
    passthrough (uncompressed) to preserve accuracy on small datasets.
    """
    mu, sigma2, pi = prior.mixture_params()
    device = mu.device
    log_pi = torch.log(pi + 1e-8)
    const = -0.5 * torch.log(2 * math.pi * sigma2)
    inv_s2 = 1.0 / sigma2

    total_orig_bits = 0
    total_bits_IR = total_bits_IC = total_bits_A = total_codebook = 0
    passthrough_bits = 0
    total_nnz = 0
    layers = []

    # collect compressible modules
    layers_mod = [
        m for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
    ]
    last2d = len(layers_mod) - 1 if layers_mod else -1

    for li, m in enumerate(layers_mod):
        W = m.weight.data
        total_orig_bits += W.numel() * 32

        if skip_last_matrix and li == last2d:
            nnz = int((W != 0).sum().item())
            passthrough_bits += W.numel() * 32
            total_nnz += nnz
            layers.append(
                {
                    "layer": m.__class__.__name__,
                    "shape": list(W.shape),
                    "orig_bits": W.numel() * 32,
                    "bits_IR": 0,
                    "bits_IC": 0,
                    "bits_A": 0,
                    "bits_codebook": 0,
                    "nnz": nnz,
                    "passthrough": True,
                }
            )
            continue

        # mixture assignment
        w = W.view(-1, 1)
        if assign_mode == "map":
            scores = (
                log_pi.unsqueeze(0)
                + const.unsqueeze(0)
                - 0.5 * ((w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0))
            )
        elif assign_mode == "ml":
            scores = const.unsqueeze(0) - 0.5 * (
                (w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0)
            )
        else:
            raise ValueError(f"Unknown assign_mode: {assign_mode}")

        idx = torch.argmax(scores, dim=1)
        comp_ids = idx.view_as(W)
        comp_ids = torch.where(comp_ids > 0, comp_ids, torch.tensor(0, device=device))

        # CSR on flattened 2D view
        W2d = flatten_conv_to_2d(W)
        comp2d = flatten_conv_to_2d(comp_ids)
        bits_IR, bits_IC, bits_A, bits_codebook, nnz = _csr_bits_for_layer(
            W2d,
            comp2d,
            prior.J,
            is_conv=isinstance(m, torch.nn.Conv2d),
            pbits_fc=pbits_fc,
            pbits_conv=pbits_conv,
            use_huffman=use_huffman,
        )
        total_bits_IR += bits_IR
        total_bits_IC += bits_IC
        total_bits_A += bits_A
        total_codebook += bits_codebook
        total_nnz += nnz
        layers.append(
            {
                "layer": m.__class__.__name__,
                "shape": list(W.shape),
                "orig_bits": W.numel() * 32,
                "bits_IR": bits_IR,
                "bits_IC": bits_IC,
                "bits_A": bits_A,
                "bits_codebook": bits_codebook,
                "nnz": nnz,
                "passthrough": False,
            }
        )

    total_compressed_bits = (
        total_bits_IR + total_bits_IC + total_bits_A + total_codebook + passthrough_bits
    )
    CR = total_orig_bits / max(total_compressed_bits, 1)

    return {
        "orig_bits": int(total_orig_bits),
        "compressed_bits": int(total_compressed_bits),
        "CR": float(CR),
        "nnz": int(total_nnz),
        "layers": layers,
    }


def logsumexp(x, dim=-1):
    m = torch.max(x, dim=dim, keepdim=True).values
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(x - m), dim=dim))


class MixturePrior(nn.Module):
    r"""
    Factorized Gaussian mixture prior over weights:
        p(w) = Π_i Σ_{j=0}^{J-1} π_j N(w_i | μ_j, σ_j^2)

    j=0 is the pruning component with μ₀=0; π₀ is fixed close to 1.

    Defaults match the tutorial layer from the original authors:
      - init σ ≈ 0.25
      - Gamma priors on precisions (different for zero vs non-zero)
    """

    def __init__(
        self,
        J: int,
        pi0: float = 0.999,
        learn_pi0: bool = False,
        init_means: torch.Tensor = None,
        init_log_sigma2: float = math.log(0.25**2),
        # --- Hyper-priors ON by default (tutorial values) ---
        gamma_alpha: float = 250.0,  # non-zero comps
        gamma_beta: float = 0.1,
        gamma_alpha0: float = 5000.0,  # zero comp
        gamma_beta0: float = 2.0,
        # Beta prior on π0 (unused unless learn_pi0=True):
        beta_alpha: float = None,
        beta_beta: float = None,
    ):
        super().__init__()
        assert J >= 2
        self.J = J
        self.learn_pi0 = learn_pi0
        self.pi0_init = float(pi0)

        if init_means is None:
            init_means = torch.linspace(-0.6, 0.6, steps=J - 1)
        self.mu = nn.Parameter(init_means.clone())  # (J-1,)
        self.log_sigma2 = nn.Parameter(torch.full((J - 1,), init_log_sigma2))
        self.pi_logits = nn.Parameter(torch.zeros(J - 1))  # softmax -> π_{1:}

        self.mu0 = torch.tensor(0.0)
        self.log_sigma2_0 = nn.Parameter(torch.tensor(init_log_sigma2))

        # Hyper-priors
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.gamma_alpha0 = gamma_alpha0
        self.gamma_beta0 = gamma_beta0
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.eps = 1e-8

    def mixture_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_nonzero = torch.softmax(self.pi_logits, dim=0)
        if self.learn_pi0:
            pi0 = torch.clamp(
                torch.tensor(self.pi0_init, device=self.mu.device), 0.5, 0.9999
            )
            pi = torch.cat([pi0.unsqueeze(0), (1.0 - pi0) * pi_nonzero])
        else:
            pi = torch.cat(
                [
                    torch.tensor([self.pi0_init], device=self.mu.device),
                    (1.0 - self.pi0_init) * pi_nonzero,
                ]
            )
        mu = torch.cat([self.mu0.to(self.mu.device).unsqueeze(0), self.mu])
        sigma2 = torch.cat(
            [self.log_sigma2_0.exp().unsqueeze(0), self.log_sigma2.exp()]
        )
        sigma2 = torch.clamp(sigma2, min=1e-8)
        return mu, sigma2, pi

    def log_prob_w(self, w_flat: torch.Tensor) -> torch.Tensor:
        mu, sigma2, pi = self.mixture_params()
        w = w_flat.unsqueeze(1)
        log_pi = torch.log(pi + self.eps).unsqueeze(0)
        log_norm = -0.5 * (
            torch.log(2 * math.pi * sigma2).unsqueeze(0)
            + (w - mu.unsqueeze(0)) ** 2 / sigma2.unsqueeze(0)
        )
        return logsumexp(log_pi + log_norm, dim=1)

    def complexity_loss(
        self, weights: list[torch.Tensor], chunk: int = 1_000_000
    ) -> torch.Tensor:
        # −∑ log p(w)  (+ hyper-priors)
        total = 0.0
        for W in weights:
            w = W.view(-1)
            for start in range(0, w.numel(), chunk):
                part = w[start : start + chunk]
                total = total + (-self.log_prob_w(part).sum())

        # --- Gamma priors on precisions λ=1/σ² ---
        mu, sigma2, pi = self.mixture_params()
        lam_all = 1.0 / sigma2
        # zero component
        if (self.gamma_alpha0 is not None) and (self.gamma_beta0 is not None):
            a0, b0 = self.gamma_alpha0, self.gamma_beta0
            lam0 = lam_all[0]
            total = total - (
                a0 * math.log(b0)
                - math.lgamma(a0)
                + (a0 - 1.0) * torch.log(lam0)
                - b0 * lam0
            )
        # non-zero components
        if (self.gamma_alpha is not None) and (self.gamma_beta is not None):
            a, b = self.gamma_alpha, self.gamma_beta
            lam = lam_all[1:]
            total = (
                total
                - (
                    (a * math.log(b) - math.lgamma(a))
                    + ((a - 1.0) * torch.log(lam) - b * lam)
                ).sum()
            )

        # --- Beta prior on π0 (optional) ---
        if (
            self.learn_pi0
            and (self.beta_alpha is not None)
            and (self.beta_beta is not None)
        ):
            a, b = self.beta_alpha, self.beta_beta
            pi0 = pi[0]
            total = total - (
                math.lgamma(a + b)
                - math.lgamma(a)
                - math.lgamma(b)
                + (a - 1.0) * torch.log(pi0 + self.eps)
                + (b - 1.0) * torch.log(1.0 - pi0 + self.eps)
            )
        return total

    @torch.no_grad()
    def _kl_gauss(self, mu0, s20, mu1, s21) -> float:
        return (
            0.5 * (torch.log(s21 / s20) + (s20 + (mu0 - mu1) ** 2) / s21 - 1.0).item()
        )

    @torch.no_grad()
    def merge_components(self, kl_threshold: float = 1e-10, max_iter: int = 200):
        mu, sigma2, pi = [t.clone() for t in self.mixture_params()]
        it = 0
        while it < max_iter:
            it += 1
            best = None
            for i in range(1, len(mu)):
                for j in range(i + 1, len(mu)):
                    d = 0.5 * (
                        self._kl_gauss(mu[i], sigma2[i], mu[j], sigma2[j])
                        + self._kl_gauss(mu[j], sigma2[j], mu[i], sigma2[i])
                    )
                    if d < kl_threshold:
                        best = (i, j, d)
                        break
                if best:
                    break
            if not best:
                break
            i, j, _ = best
            pnew = pi[i] + pi[j]
            if pnew <= 0:
                break
            mu_new = (pi[i] * mu[i] + pi[j] * mu[j]) / pnew
            s2_new = (pi[i] * sigma2[i] + pi[j] * sigma2[j]) / pnew
            mu[i], sigma2[i], pi[i] = mu_new, s2_new, pnew
            mu = torch.cat([mu[:j], mu[j + 1 :]])
            sigma2 = torch.cat([sigma2[:j], sigma2[j + 1 :]])
            pi = torch.cat([pi[:j], pi[j + 1 :]])
        with torch.no_grad():
            self.mu.data = mu[1:]
            self.log_sigma2.data = torch.log(sigma2[1:])
            pi1 = pi[1:]
            pi1 = pi1 / (pi1.sum() + 1e-12)
            self.pi_logits.data = torch.log(pi1 + 1e-12)

    @torch.no_grad()
    def quantize_model(
        self, model, *, skip_last_matrix: bool = True, assign: str = "ml"
    ):
        """
        Hard-quantize weights to mixture means.

        assign:
        - "map": MAP under the learned mixture (uses mixing π and σ^2)
        - "ml" : maximum likelihood per component (equal mixing; ignores π)
                This avoids a strong bias towards the zero spike during snapping.
        """
        mu, sigma2, pi = self.mixture_params()
        log_pi = torch.log(pi + self.eps)
        const = -0.5 * torch.log(2 * math.pi * sigma2)
        inv_s2 = 1.0 / sigma2

        # collect weights in order
        Ws = []
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                Ws.append(m.weight)

        # last 2D weight index
        last2d = max((i for i, W in enumerate(Ws) if W.ndim >= 2), default=-1)

        for i, W in enumerate(Ws):
            if skip_last_matrix and i == last2d:
                continue

            w = W.data.view(-1, 1)

            if assign == "map":
                # MAP: includes mixing proportions (π)
                scores = (
                    log_pi.unsqueeze(0)
                    + const.unsqueeze(0)
                    - 0.5 * ((w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0))
                )
            elif assign == "ml":
                # ML: ignore π, keep per-component likelihood (equal mixing)
                scores = const.unsqueeze(0) - 0.5 * (
                    (w - mu.unsqueeze(0)) ** 2 * inv_s2.unsqueeze(0)
                )
            else:
                raise ValueError(f"Unknown assign mode: {assign}")

            idx = torch.argmax(scores, dim=1)
            snapped = mu[idx].view_as(W)
            # exact zeros for component 0 (numerical hygiene)
            snapped[idx.view_as(W) == 0] = 0.0
            W.data.copy_(snapped)

    def snapshot(self) -> Dict:
        mu, sigma2, pi = self.mixture_params()
        return {
            "mu": mu.detach().cpu().tolist(),
            "sigma2": sigma2.detach().cpu().tolist(),
            "pi": pi.detach().cpu().tolist(),
            "J": int(self.J),
        }


def init_mixture(
    model: nn.Module,
    J: int,
    pi0: float,
    init_means_mode: str = "from_weights",
    init_range_min: float = -0.6,
    init_range_max: float = 0.6,
    init_sigma: float = 0.25,
    device=None,
) -> MixturePrior:
    # means: from pretrained weights' range, or fixed range [-0.6,0.6]
    if init_means_mode == "from_weights":
        weights = torch.cat(
            [p.detach().flatten().cpu() for p in collect_weight_params(model)]
        )
        wmin, wmax = weights.min().item(), weights.max().item()
        if wmin == wmax:
            wmin, wmax = -0.6, 0.6
        means = torch.linspace(wmin, wmax, steps=J - 1)
    else:
        means = torch.linspace(init_range_min, init_range_max, steps=J - 1)
    prior = MixturePrior(
        J=J,
        pi0=pi0,
        learn_pi0=False,
        init_means=means.to(device) if device else means,
        init_log_sigma2=math.log(init_sigma**2),
    )
    return prior


def collect_weight_params(model) -> list:
    params = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            params.append(m.weight)
    return params