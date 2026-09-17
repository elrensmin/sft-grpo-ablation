import torch
from typing import Dict


@torch.no_grad()
def spectral_drift(base_state: Dict[str, torch.Tensor],
                   tuned_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Per-module Frobenius norm ratio: ||ΔW||_F / ||W_base||_F."""
    out = {}
    for name, w_base in base_state.items():
        if name not in tuned_state:
            continue
        delta = tuned_state[name].float() - w_base.float()
        denom = w_base.float().norm().item() + 1e-12
        out[name] = delta.norm().item() / denom
    return out


@torch.no_grad()
def task_vector_cosine(tv_a: Dict[str, torch.Tensor],
                       tv_b: Dict[str, torch.Tensor]) -> float:
    flat_a, flat_b = [], []
    for k in tv_a:
        if k in tv_b:
            flat_a.append(tv_a[k].float().flatten())
            flat_b.append(tv_b[k].float().flatten())
    a = torch.cat(flat_a)
    b = torch.cat(flat_b)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@torch.no_grad()
def top_principal_direction_overlap(delta: torch.Tensor,
                                    base: torch.Tensor,
                                    k: int = 10) -> float:
    # SVD on 2D projections only
    if delta.dim() != 2 or base.dim() != 2:
        return 0.0
    U_b, S_b, _ = torch.svd_lowrank(base.float(), q=k)
    U_d, S_d, _ = torch.svd_lowrank(delta.float(), q=k)
    # Principal angles: sum of squared cosines
    M = U_b.T @ U_d
    return (M ** 2).sum().item() / k
