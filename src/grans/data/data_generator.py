"""Generate the four configurable 2D FEM problem families used by G-RANS."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch

from .fem_solver import FEMSolver


class DataGenerator:
    """Generate sparse PDE systems from a validated problem configuration."""

    def __init__(
        self,
        problem: Mapping[str, Any],
        mesh_size: int = 250,
        device: str = "cpu",
        randomize_params: bool = False,
        param_variation: float = 0.1,
        use_preconditioner: bool = False,
        seed: int = 42,
    ) -> None:
        self.problem = deepcopy(dict(problem))
        self.pde_type = str(self.problem["pde_type"])
        self.mesh_size = mesh_size
        self.device = device
        self.randomize_params = randomize_params
        self.param_variation = param_variation
        self.use_preconditioner = use_preconditioner
        self.rng = np.random.default_rng(seed)
        self.solver = FEMSolver(target_nodes=mesh_size)

        supported = {
            "poisson",
            "advection_diffusion",
            "helmholtz",
            "reaction_diffusion",
        }
        if self.pde_type not in supported:
            raise ValueError(f"unsupported PDE type in problem config: {self.pde_type}")

    def generate_sample(self, randomize: bool = False) -> dict[str, Any]:
        """Generate one sample, optionally perturbing the configured domain."""
        parameters = self._sample_parameters()
        center, radius = self._sample_inclusion()
        domain, boundary_points = self._sample_domain(randomize)

        def inside(xvec: tuple[float, float]) -> bool:
            x, y = xvec
            return float(np.hypot(x - center[0], y - center[1])) <= radius

        def kappa(xvec: tuple[float, float]) -> float:
            return parameters["kappa_inner"] if inside(xvec) else parameters["kappa_outer"]

        def source(xvec: tuple[float, float]) -> float:
            return parameters["source_strength"] if inside(xvec) else 0.0

        def boundary(xvec: tuple[float, float]) -> float:
            x, _ = xvec
            return parameters["bc_scale"] * (1.0 - x**2)

        self.solver.set_domain(domain, boundary_points)
        if self.pde_type == "poisson":
            shift = parameters["poisson_shift"]
            if shift > 0.0:
                solution, matrix, rhs = self.solver.solve_helmholtz(
                    kappa, lambda _: shift, source, boundary
                )
            else:
                solution, matrix, rhs = self.solver.solve(kappa, source, boundary)
        elif self.pde_type == "advection_diffusion":
            velocity_scale = parameters["velocity_scale"]

            def velocity(xvec: tuple[float, float]) -> tuple[float, float]:
                x, y = xvec
                return -y * velocity_scale, x * velocity_scale

            solution, matrix, rhs = self.solver.solve_advection_diffusion(
                kappa, velocity, source, boundary
            )
        else:
            reaction_inner = parameters["reaction_inner"]
            reaction_outer = parameters["reaction_outer"]

            def reaction(xvec: tuple[float, float]) -> float:
                return reaction_inner if inside(xvec) else reaction_outer

            solution, matrix, rhs = self.solver.solve_helmholtz(
                kappa, reaction, source, boundary
            )

        param_info: dict[str, Any] = dict(parameters)
        param_info.update(circle_center=center, circle_radius=radius)
        return self._to_sample(matrix, solution, rhs, param_info)

    def _sample_parameters(self) -> dict[str, float]:
        parameters = {
            name: float(value) for name, value in self.problem["parameters"].items()
        }
        if not self.randomize_params:
            return parameters
        for name, value in parameters.items():
            if value != 0.0:
                parameters[name] = self._vary_relative(value, 1.0)
        return parameters

    def _sample_inclusion(self) -> tuple[tuple[float, float], float]:
        inclusion = self.problem["inclusion"]
        center = np.asarray(inclusion["center"], dtype=float)
        radius = float(inclusion["radius"])
        if self.randomize_params:
            center_jitter = float(inclusion["center_jitter"]) * self.param_variation
            if center_jitter > 0.0:
                center = center + self.rng.uniform(-center_jitter, center_jitter, size=2)
            radius = self._vary_relative(
                radius, float(inclusion["radius_relative_variation"])
            )
        return (float(center[0]), float(center[1])), radius

    def _sample_domain(
        self,
        randomize: bool,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        domain_config = self.problem["domain"]
        vertices = np.asarray(domain_config["vertices"], dtype=float)
        if randomize:
            jitter = float(domain_config["top_corner_jitter"])
            for index in domain_config["jitter_vertices"]:
                vertices[int(index)] += self.rng.uniform(-jitter, jitter, size=2)
        domain = [tuple(map(float, point)) for point in vertices]
        boundary = [
            tuple(map(float, point)) for point in domain_config["dirichlet_points"]
        ]
        return domain, boundary

    def _vary_relative(self, value: float, scale: float) -> float:
        width = self.param_variation * scale
        return value * (1.0 + float(self.rng.uniform(-width, width)))

    def _to_sample(
        self,
        matrix: Any,
        solution: np.ndarray,
        rhs: np.ndarray,
        param_info: dict[str, Any],
    ) -> dict[str, Any]:
        triplets = np.stack([matrix.row, matrix.col, matrix.data], axis=1)
        A_i_v = torch.tensor(triplets, dtype=torch.float32, device=self.device)
        x = torch.tensor(solution, dtype=torch.float32, device=self.device)
        b = torch.tensor(rhs, dtype=torch.float32, device=self.device)
        coords = torch.tensor(self.solver.nodes, dtype=torch.float32, device=self.device)
        sample: dict[str, Any] = {
            "A_i_v": A_i_v,
            "x": x,
            "b": b,
            "coords": coords,
            "param_info": param_info,
        }
        if self.use_preconditioner:
            sample["A_i_v"], sample["b"], sample["D_inv"] = (
                self._apply_diagonal_preconditioner(A_i_v, b)
            )
        return sample

    def _apply_diagonal_preconditioner(
        self,
        A_i_v: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Jacobi left preconditioning to ``A`` and ``b``."""
        rows = A_i_v[:, 0].long()
        cols = A_i_v[:, 1].long()
        values = A_i_v[:, 2]
        diagonal_mask = rows == cols
        diagonal = torch.full(
            (b.shape[0],), 1.0e-10, dtype=torch.float32, device=self.device
        )
        diagonal[rows[diagonal_mask]] = values[diagonal_mask]
        inverse = 1.0 / diagonal
        preconditioned_values = values * inverse[rows]
        matrix = torch.stack([rows.float(), cols.float(), preconditioned_values], dim=1)
        return matrix, inverse * b, inverse
