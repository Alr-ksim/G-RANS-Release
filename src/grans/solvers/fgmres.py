"""
Torch-based FGMRES solver with optional Jacobi preconditioning.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List
import time

import torch


def _matvec(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)
    return A @ v


def jacobi_preconditioner(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if A.is_sparse:
        A_coo = A.coalesce()
        indices = A_coo.indices()
        values = A_coo.values()
        mask = indices[0] == indices[1]
        diag_idx = indices[0][mask]
        diag_vals = values[mask]
        n = A.shape[0]
        diag = torch.full((n,), eps, dtype=A.dtype, device=A.device)
        diag = diag.scatter_add(0, diag_idx, diag_vals)
    else:
        diag = torch.diagonal(A)
        diag = diag + eps
    return 1.0 / diag


def _finalize_history(residual_history: List[float], time_history: List[float], device, dtype):
    residuals = torch.tensor(residual_history, device=device, dtype=dtype)
    times = torch.tensor(time_history, device=device, dtype=dtype)
    return residuals, times


def fgmres_torch_solve_with_history(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 0.0,
    restart: int = 40,
    maxiter: int = 200,
    preconditioner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    D_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, float, int, torch.Tensor, torch.Tensor]:
    """FGMRES with per-iteration residual/time history."""
    device = b.device
    dtype = b.dtype

    def apply_precond(v: torch.Tensor) -> torch.Tensor:
        if preconditioner is not None:
            return preconditioner(v)
        if D_inv is not None:
            return D_inv * v
        return v

    x = torch.zeros_like(b)
    r = b - _matvec(A, x)
    b_norm = torch.norm(b)
    tol = max(atol, rtol * b_norm.item())

    if b_norm.item() == 0.0:
        residuals, times = _finalize_history([], [], device, dtype)
        return x, 0, 0.0, 0, residuals, times

    residual_history = []
    time_history = []
    use_cuda_timer = A.is_cuda
    total_iters = 0

    max_outer = (maxiter + restart - 1) // restart

    for _ in range(max_outer):
        r = b - _matvec(A, x)
        beta = torch.norm(r)
        if beta.item() <= tol:
            residuals, times = _finalize_history(residual_history, time_history, device, dtype)
            return x, 0, beta.item(), total_iters, residuals, times

        m = min(restart, maxiter - total_iters)
        V = torch.zeros((b.numel(), m + 1), device=device, dtype=dtype)
        Z = torch.zeros((b.numel(), m), device=device, dtype=dtype)
        H = torch.zeros((m + 1, m), device=device, dtype=dtype)
        g = torch.zeros(m + 1, device=device, dtype=dtype)
        cs = torch.zeros(m, device=device, dtype=dtype)
        sn = torch.zeros(m, device=device, dtype=dtype)

        V[:, 0] = r / beta
        g[0] = beta

        for j in range(m):
            if use_cuda_timer:
                torch.cuda.synchronize()
            iter_start = time.perf_counter()

            z = apply_precond(V[:, j])
            Z[:, j] = z
            w = _matvec(A, z)

            H[:j + 1, j] = V[:, :j + 1].T @ w
            w = w - V[:, :j + 1] @ H[:j + 1, j]
            H[j + 1, j] = torch.norm(w)
            if H[j + 1, j].item() != 0.0:
                V[:, j + 1] = w / H[j + 1, j]

            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            denom = torch.sqrt(H[j, j] * H[j, j] + H[j + 1, j] * H[j + 1, j])
            if denom.item() == 0.0:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom

            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            temp = cs[j] * g[j] + sn[j] * g[j + 1]
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
            g[j] = temp

            total_iters += 1
            resid = torch.abs(g[j + 1])

            if use_cuda_timer:
                torch.cuda.synchronize()
            iter_elapsed = time.perf_counter() - iter_start

            residual_history.append(resid.item())
            time_history.append(iter_elapsed)

            if resid.item() <= tol:
                y = torch.linalg.solve_triangular(
                    H[:j + 1, :j + 1],
                    g[:j + 1].unsqueeze(1),
                    upper=True
                ).squeeze(1)
                x = x + Z[:, :j + 1] @ y
                residuals, times = _finalize_history(residual_history, time_history, device, dtype)
                return x, 0, resid.item(), total_iters, residuals, times

        y = torch.linalg.solve_triangular(
            H[:m, :m],
            g[:m].unsqueeze(1),
            upper=True
        ).squeeze(1)
        x = x + Z[:, :m] @ y

        if total_iters >= maxiter:
            break

    final_resid = torch.norm(b - _matvec(A, x)).item()
    residuals, times = _finalize_history(residual_history, time_history, device, dtype)
    return x, maxiter, final_resid, total_iters, residuals, times
