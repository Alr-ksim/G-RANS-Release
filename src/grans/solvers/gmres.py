"""
Torch-based GMRES solver for GPU/CPU benchmarking.

This implementation keeps all linear algebra on the selected device and avoids
SciPy dependencies, making it suitable for GPU timing comparisons.
"""

from __future__ import annotations

from typing import Tuple
import time

import torch


def _matvec(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)
    return A @ v


def gmres_torch_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 0.0,
    restart: int = 40,
    maxiter: int = 200,
) -> Tuple[torch.Tensor, int, float, int]:
    """
    Minimal GMRES solver (with restart) for Ax=b using torch ops.

    Returns:
        x: solution vector
        info: 0 if converged, maxiter otherwise
        final_residual: residual norm ||b - Ax||
        iterations: number of inner iterations performed
    """
    n = b.shape[0]
    device = b.device
    dtype = b.dtype

    x = torch.zeros_like(b)
    b_norm = torch.norm(b)
    tol = max(atol, rtol * b_norm.item())

    if b_norm.item() == 0.0:
        return x, 0, 0.0, 0

    total_iters = 0
    max_outer = (maxiter + restart - 1) // restart

    for _ in range(max_outer):
        r = b - _matvec(A, x)
        beta = torch.norm(r)
        if beta.item() <= tol:
            return x, 0, beta.item(), total_iters

        m = min(restart, maxiter - total_iters)
        V = torch.zeros((n, m + 1), device=device, dtype=dtype)
        H = torch.zeros((m + 1, m), device=device, dtype=dtype)
        V[:, 0] = r / beta

        cs = torch.zeros(m, device=device, dtype=dtype)
        sn = torch.zeros(m, device=device, dtype=dtype)
        g = torch.zeros(m + 1, device=device, dtype=dtype)
        g[0] = beta

        for j in range(m):
            w = _matvec(A, V[:, j])
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
            if resid.item() <= tol:
                y = torch.linalg.solve_triangular(
                    H[:j + 1, :j + 1],
                    g[:j + 1].unsqueeze(1),
                    upper=True
                ).squeeze(1)
                x = x + V[:, :j + 1] @ y
                return x, 0, resid.item(), total_iters

        y = torch.linalg.solve_triangular(
            H[:m, :m],
            g[:m].unsqueeze(1),
            upper=True
        ).squeeze(1)
        x = x + V[:, :m] @ y

        if total_iters >= maxiter:
            break

    final_resid = torch.norm(b - _matvec(A, x)).item()
    return x, maxiter, final_resid, total_iters


def gmres_torch_solve_with_history(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 0.0,
    restart: int = 40,
    maxiter: int = 200,
) -> Tuple[torch.Tensor, int, float, int, torch.Tensor, torch.Tensor]:
    """GMRES solver with per-iteration residual and timing history."""
    n = b.shape[0]
    device = b.device
    dtype = b.dtype

    x = torch.zeros_like(b)
    b_norm = torch.norm(b)
    tol = max(atol, rtol * b_norm.item())

    if b_norm.item() == 0.0:
        residuals = torch.zeros(0, device=device, dtype=dtype)
        times = torch.zeros(0, device=device, dtype=dtype)
        return x, 0, 0.0, 0, residuals, times

    total_iters = 0
    max_outer = (maxiter + restart - 1) // restart
    residual_history = []
    time_history = []
    use_cuda_timer = A.is_cuda

    for _ in range(max_outer):
        r = b - _matvec(A, x)
        beta = torch.norm(r)
        if beta.item() <= tol:
            residuals = torch.tensor(residual_history, device=device, dtype=dtype)
            times = torch.tensor(time_history, device=device, dtype=dtype)
            return x, 0, beta.item(), total_iters, residuals, times

        m = min(restart, maxiter - total_iters)
        V = torch.zeros((n, m + 1), device=device, dtype=dtype)
        H = torch.zeros((m + 1, m), device=device, dtype=dtype)
        V[:, 0] = r / beta

        cs = torch.zeros(m, device=device, dtype=dtype)
        sn = torch.zeros(m, device=device, dtype=dtype)
        g = torch.zeros(m + 1, device=device, dtype=dtype)
        g[0] = beta

        for j in range(m):
            if use_cuda_timer:
                torch.cuda.synchronize()
            iter_start = time.perf_counter()

            w = _matvec(A, V[:, j])
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
                x = x + V[:, :j + 1] @ y
                residuals = torch.tensor(residual_history, device=device, dtype=dtype)
                times = torch.tensor(time_history, device=device, dtype=dtype)
                return x, 0, resid.item(), total_iters, residuals, times

        y = torch.linalg.solve_triangular(
            H[:m, :m],
            g[:m].unsqueeze(1),
            upper=True
        ).squeeze(1)
        x = x + V[:, :m] @ y

        if total_iters >= maxiter:
            break

    final_resid = torch.norm(b - _matvec(A, x)).item()
    residuals = torch.tensor(residual_history, device=device, dtype=dtype)
    times = torch.tensor(time_history, device=device, dtype=dtype)
    return x, maxiter, final_resid, total_iters, residuals, times
