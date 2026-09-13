import torch

from grans.solvers.projection import (
    coo_triplets_to_sparse,
    extend_orthonormal_basis,
    orthonormalize,
    projected_correction,
)


def test_projection_does_not_increase_residual() -> None:
    dense = torch.tensor(
        [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]]
    )
    indices = dense.nonzero()
    triplets = torch.cat([indices.float(), dense[indices[:, 0], indices[:, 1], None]], dim=1)
    A = coo_triplets_to_sparse(triplets, 3)
    residual = torch.tensor([1.0, 2.0, -1.0])
    basis = orthonormalize(torch.eye(3))
    result = projected_correction(A, residual, basis)
    assert torch.linalg.vector_norm(result.residual) <= torch.linalg.vector_norm(residual)
    assert torch.linalg.vector_norm(result.residual) < 1.0e-5


def test_incremental_extension_is_orthonormal() -> None:
    torch.manual_seed(4)
    old = orthonormalize(torch.randn(20, 4))
    candidates = torch.randn(20, 5)
    basis = extend_orthonormal_basis(old, candidates, max_columns=9)
    identity = torch.eye(basis.shape[1])
    assert torch.allclose(basis.T @ basis, identity, atol=2.0e-5)


def test_incremental_and_full_qr_span_the_same_subspace() -> None:
    torch.manual_seed(8)
    old = orthonormalize(torch.randn(20, 4))
    candidates = torch.randn(20, 5)
    incremental = extend_orthonormal_basis(old, candidates, 9, method="incremental")
    full = extend_orthonormal_basis(old, candidates, 9, method="full")
    assert torch.allclose(
        incremental @ incremental.T,
        full @ full.T,
        atol=3.0e-5,
    )


def test_incremental_extension_discards_redundant_candidates() -> None:
    torch.manual_seed(12)
    old = orthonormalize(torch.randn(20, 4))
    basis = extend_orthonormal_basis(old, old[:, :2], max_columns=9)
    assert basis.shape == old.shape
    assert torch.allclose(basis @ basis.T, old @ old.T, atol=2.0e-5)
