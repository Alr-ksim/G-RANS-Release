import torch

from grans.solvers.grans import GRANSSolver


class ResidualDirection(torch.nn.Module):
    def forward(self, residual, coords, edge_index, edge_attr, batch=None):
        output = torch.zeros((residual.numel(), 1), device=residual.device)
        if batch is None:
            return residual.reshape(-1, 1) / torch.linalg.vector_norm(residual)
        for index in range(int(batch.max().item()) + 1):
            mask = batch == index
            values = residual[mask]
            output[mask, 0] = values / torch.linalg.vector_norm(values)
        return output


def _identity_sample(scale: float):
    indices = torch.arange(4, dtype=torch.float32)
    return {
        "A_i_v": torch.stack([indices, indices, torch.ones(4)], dim=1),
        "b": scale * torch.tensor([1.0, -2.0, 3.0, -4.0]),
        "x": scale * torch.tensor([1.0, -2.0, 3.0, -4.0]),
        "coords": torch.zeros((4, 2)),
    }


def test_batched_solver_solves_identity_in_one_step() -> None:
    solver = GRANSSolver(
        ResidualDirection(),
        "cpu",
        max_iterations=2,
        max_basis_size=2,
        absolute_tolerance=1.0e-5,
        min_reduction=0.0,
    )
    results = solver.solve([_identity_sample(1.0), _identity_sample(2.0)], batch_size=2)
    assert all(result.iterations == 1 for result in results)
    assert all(result.relative_residual < 1.0e-6 for result in results)
