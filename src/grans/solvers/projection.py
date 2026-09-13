"""Shared sparse projection primitives used by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProjectionResult:
    correction: torch.Tensor
    residual: torch.Tensor
    coefficients: torch.Tensor
    image_basis: torch.Tensor


def coo_triplets_to_sparse(A_i_v: torch.Tensor, size: int) -> torch.Tensor:
    indices = A_i_v[:, :2].long().T.contiguous()
    values = A_i_v[:, 2].float()
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(size, size),
        dtype=torch.float32,
        device=A_i_v.device,
    ).coalesce()


def sparse_mm(A: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    original_dtype = dense.dtype
    device_type = dense.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        result = torch.sparse.mm(A.float(), dense.float())
    return result.to(original_dtype)


def sparse_mv(A: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return sparse_mm(A, vector.unsqueeze(1)).squeeze(1)


def orthonormalize(basis: torch.Tensor, max_columns: int | None = None) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError(f"basis must be rank 2, got shape {tuple(basis.shape)}")
    if basis.shape[0] < basis.shape[1]:
        raise ValueError(
            f"basis has more columns than rows: {basis.shape[1]} > {basis.shape[0]}"
        )
    original_dtype = basis.dtype
    Q, _ = torch.linalg.qr(basis.float(), mode="reduced")
    if max_columns is not None and Q.shape[1] > max_columns:
        Q = Q[:, -max_columns:]
    return Q.to(original_dtype)


def extend_orthonormal_basis(
    accumulated: torch.Tensor | None,
    candidates: torch.Tensor,
    max_columns: int,
    method: str = "incremental",
    reorthogonalize: bool = True,
) -> torch.Tensor:
    """Extend a basis while avoiding a full QR of all accumulated columns.

    ``incremental`` retains the newest old columns, projects candidates against
    them (twice by default for numerical stability), and applies QR only to the
    candidate residual block. ``full`` reproduces the previous full-QR path.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be rank 2, got shape {tuple(candidates.shape)}")
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")
    if method not in {"incremental", "full"}:
        raise ValueError("method must be 'incremental' or 'full'")
    if accumulated is not None:
        if accumulated.ndim != 2 or accumulated.shape[0] != candidates.shape[0]:
            raise ValueError("accumulated basis and candidates must have matching row counts")

    if method == "full":
        combined = candidates if accumulated is None else torch.cat([accumulated, candidates], 1)
        return orthonormalize(combined, max_columns)

    candidate_block = candidates[:, -max_columns:]
    old_capacity = max_columns - candidate_block.shape[1]
    old_basis = None
    if accumulated is not None and old_capacity > 0:
        old_basis = accumulated[:, -old_capacity:]

    residual_block = candidate_block.float()
    candidate_scale = torch.linalg.vector_norm(residual_block)
    old_fp32 = None if old_basis is None else old_basis.float()
    if old_fp32 is not None and old_fp32.shape[1] > 0:
        residual_block = residual_block - old_fp32 @ (old_fp32.T @ residual_block)
        if reorthogonalize:
            residual_block = residual_block - old_fp32 @ (old_fp32.T @ residual_block)

    new_basis, triangular = torch.linalg.qr(residual_block, mode="reduced")
    diagonal = torch.abs(torch.diagonal(triangular))
    tolerance = (
        torch.finfo(residual_block.dtype).eps
        * max(residual_block.shape)
        * candidate_scale
    )
    new_basis = new_basis[:, diagonal > tolerance].to(candidates.dtype)

    if old_basis is None or old_basis.shape[1] == 0:
        return new_basis
    if new_basis.shape[1] == 0:
        return old_basis
    return torch.cat([old_basis, new_basis], dim=1)


def projected_correction(
    A: torch.Tensor,
    residual: torch.Tensor,
    basis: torch.Tensor,
) -> ProjectionResult:
    """Minimize ``||A Q y - r||`` and return ``Q y`` and the next residual."""
    image_basis = sparse_mm(A, basis)
    rhs = residual.float().unsqueeze(1)
    image_basis_fp32 = image_basis.float()
    try:
        coefficients = torch.linalg.lstsq(
            image_basis_fp32,
            rhs,
            driver="gels",
        ).solution
    except RuntimeError:
        coefficients = torch.linalg.pinv(image_basis_fp32) @ rhs
    coefficients = coefficients.to(basis.dtype)
    correction = (basis @ coefficients).squeeze(1)
    projected = (image_basis @ coefficients).squeeze(1)
    return ProjectionResult(
        correction=correction,
        residual=residual - projected,
        coefficients=coefficients,
        image_basis=image_basis,
    )


def relative_residual(residual: torch.Tensor, initial_norm: float) -> float:
    if initial_norm == 0.0:
        return 0.0
    return torch.linalg.vector_norm(residual).item() / initial_norm
