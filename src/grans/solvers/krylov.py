"""
Torch-based Krylov solvers for GPU/CPU benchmarking.

Includes CG, BiCGSTAB, and MINRES with per-iteration residual/time history.
"""

from __future__ import annotations

from typing import Tuple, List
import time

import torch


def _matvec(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if A.is_sparse:
        return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)
    return A @ v


def _finalize_history(residual_history: List[float], time_history: List[float], device, dtype):
    residuals = torch.tensor(residual_history, device=device, dtype=dtype)
    times = torch.tensor(time_history, device=device, dtype=dtype)
    return residuals, times


def cg_torch_solve_with_history(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 0.0,
    maxiter: int = 200,
) -> Tuple[torch.Tensor, int, float, int, torch.Tensor, torch.Tensor]:
    """Conjugate Gradient (SPD) with per-iteration history."""
    device = b.device
    dtype = b.dtype
    x = torch.zeros_like(b)
    r = b - _matvec(A, x)
    p = r.clone()

    b_norm = torch.norm(b)
    tol = max(atol, rtol * b_norm.item())

    if b_norm.item() == 0.0:
        residuals, times = _finalize_history([], [], device, dtype)
        return x, 0, 0.0, 0, residuals, times

    rsold = torch.dot(r, r)
    residual_history = []
    time_history = []
    use_cuda_timer = A.is_cuda

    for k in range(maxiter):
        if use_cuda_timer:
            torch.cuda.synchronize()
        iter_start = time.perf_counter()

        Ap = _matvec(A, p)
        denom = torch.dot(p, Ap)
        if denom.item() == 0.0:
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        r_norm = torch.sqrt(rsnew).item()

        if use_cuda_timer:
            torch.cuda.synchronize()
        iter_elapsed = time.perf_counter() - iter_start

        residual_history.append(r_norm)
        time_history.append(iter_elapsed)

        if r_norm <= tol:
            residuals, times = _finalize_history(residual_history, time_history, device, dtype)
            return x, 0, r_norm, k + 1, residuals, times

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    final_resid = torch.norm(b - _matvec(A, x)).item()
    residuals, times = _finalize_history(residual_history, time_history, device, dtype)
    return x, maxiter, final_resid, len(residual_history), residuals, times


def bicgstab_torch_solve_with_history(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 0.0,
    maxiter: int = 200,
) -> Tuple[torch.Tensor, int, float, int, torch.Tensor, torch.Tensor]:
    """BiCGSTAB with per-iteration history."""
    device = b.device
    dtype = b.dtype
    x = torch.zeros_like(b)
    r = b - _matvec(A, x)
    r_hat = r.clone()

    b_norm = torch.norm(b)
    tol = max(atol, rtol * b_norm.item())

    if b_norm.item() == 0.0:
        residuals, times = _finalize_history([], [], device, dtype)
        return x, 0, 0.0, 0, residuals, times

    rho_old = torch.tensor(1.0, device=device, dtype=dtype)
    alpha = torch.tensor(1.0, device=device, dtype=dtype)
    omega = torch.tensor(1.0, device=device, dtype=dtype)
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    residual_history = []
    time_history = []
    use_cuda_timer = A.is_cuda

    for k in range(maxiter):
        if use_cuda_timer:
            torch.cuda.synchronize()
        iter_start = time.perf_counter()

        rho = torch.dot(r_hat, r)
        if rho.item() == 0.0:
            break
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = _matvec(A, p)
        denom = torch.dot(r_hat, v)
        if denom.item() == 0.0:
            break
        alpha = rho / denom
        s = r - alpha * v

        s_norm = torch.norm(s).item()
        if s_norm <= tol:
            x = x + alpha * p
            if use_cuda_timer:
                torch.cuda.synchronize()
            iter_elapsed = time.perf_counter() - iter_start
            residual_history.append(s_norm)
            time_history.append(iter_elapsed)
            residuals, times = _finalize_history(residual_history, time_history, device, dtype)
            return x, 0, s_norm, k + 1, residuals, times

        t = _matvec(A, s)
        t_dot_t = torch.dot(t, t)
        if t_dot_t.item() == 0.0:
            break
        omega = torch.dot(t, s) / t_dot_t
        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = torch.norm(r).item()

        if use_cuda_timer:
            torch.cuda.synchronize()
        iter_elapsed = time.perf_counter() - iter_start

        residual_history.append(r_norm)
        time_history.append(iter_elapsed)

        if r_norm <= tol:
            residuals, times = _finalize_history(residual_history, time_history, device, dtype)
            return x, 0, r_norm, k + 1, residuals, times

        if omega.item() == 0.0:
            break
        rho_old = rho

    final_resid = torch.norm(b - _matvec(A, x)).item()
    residuals, times = _finalize_history(residual_history, time_history, device, dtype)
    return x, maxiter, final_resid, len(residual_history), residuals, times


def minres_torch_solve_with_history(
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 0.0,
    maxiter: int = 200,
) -> Tuple[torch.Tensor, int, float, int, torch.Tensor, torch.Tensor]:
    """MINRES (symmetric) with per-iteration history via Lanczos + LS."""
    device = b.device
    dtype = b.dtype
    x0 = torch.zeros_like(b)

    b_norm = torch.norm(b)
    tol = max(atol, rtol * b_norm.item())

    if b_norm.item() == 0.0:
        residuals, times = _finalize_history([], [], device, dtype)
        return x0, 0, 0.0, 0, residuals, times

    r0 = b - _matvec(A, x0)
    beta1 = torch.norm(r0)
    if beta1.item() == 0.0:
        residuals, times = _finalize_history([], [], device, dtype)
        return x0, 0, 0.0, 0, residuals, times

    V = []
    v_prev = torch.zeros_like(b)
    v = r0 / beta1
    V.append(v)

    alphas = []
    betas = []
    residual_history = []
    time_history = []
    use_cuda_timer = A.is_cuda

    for k in range(maxiter):
        if use_cuda_timer:
            torch.cuda.synchronize()
        iter_start = time.perf_counter()

        w = _matvec(A, v)
        if k > 0:
            w = w - betas[-1] * v_prev
        alpha = torch.dot(v, w)
        w = w - alpha * v
        beta = torch.norm(w)

        alphas.append(alpha)
        betas.append(beta)

        if beta.item() != 0.0:
            v_prev = v
            v = w / beta
            V.append(v)

        # Build tridiagonal T_k
        T = torch.zeros((k + 1, k + 1), device=device, dtype=dtype)
        for i in range(k + 1):
            T[i, i] = alphas[i]
            if i > 0:
                T[i, i - 1] = betas[i - 1]
                T[i - 1, i] = betas[i - 1]

        e1 = torch.zeros(k + 1, device=device, dtype=dtype)
        e1[0] = beta1
        y = torch.linalg.lstsq(T, e1.unsqueeze(1), driver="gels").solution.squeeze(1)
        V_mat = torch.stack(V[:k + 1], dim=1)
        x = x0 + V_mat @ y
        r = b - _matvec(A, x)
        r_norm = torch.norm(r).item()

        if use_cuda_timer:
            torch.cuda.synchronize()
        iter_elapsed = time.perf_counter() - iter_start

        residual_history.append(r_norm)
        time_history.append(iter_elapsed)

        if r_norm <= tol:
            residuals, times = _finalize_history(residual_history, time_history, device, dtype)
            return x, 0, r_norm, k + 1, residuals, times

        if beta.item() == 0.0:
            break

    final_resid = torch.norm(b - _matvec(A, x)).item()
    residuals, times = _finalize_history(residual_history, time_history, device, dtype)
    return x, maxiter, final_resid, len(residual_history), residuals, times
