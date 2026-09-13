"""Canonical batched G-RANS inference implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable

import torch
from torch_geometric.data import Batch, Data

from ..utils import synchronize
from .projection import (
    coo_triplets_to_sparse,
    extend_orthonormal_basis,
    projected_correction,
    sparse_mv,
)


@dataclass
class SolveResult:
    solution: torch.Tensor
    iterations: int
    initial_residual: float
    final_residual: float
    relative_residual: float
    reductions: list[float]
    stop_reason: str
    model_time: float
    total_time: float
    relative_l2_error: float | None = None

    def metrics(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("solution")
        return values


class GRANSSolver:
    """Residual-aware subspace solver with batched model inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        max_iterations: int = 50,
        max_basis_size: int = 400,
        absolute_tolerance: float = 1.0e-5,
        min_reduction: float = 0.05,
        min_reduction_patience: int = 2,
        orthogonalization: str = "incremental",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.max_iterations = max_iterations
        self.max_basis_size = max_basis_size
        self.absolute_tolerance = absolute_tolerance
        self.min_reduction = min_reduction
        self.min_reduction_patience = min_reduction_patience
        self.orthogonalization = orthogonalization

    def solve(
        self,
        samples: Iterable[dict[str, Any]],
        batch_size: int = 64,
        warmup_iterations: int = 1,
    ) -> list[SolveResult]:
        sample_list = list(samples)
        output: list[SolveResult] = []
        for start in range(0, len(sample_list), batch_size):
            output.extend(
                self._solve_batch(
                    sample_list[start : start + batch_size],
                    warmup_iterations=warmup_iterations if start == 0 else 0,
                )
            )
        return output

    def _prepare(self, sample: dict[str, Any]) -> dict[str, Any]:
        A_i_v = sample["A_i_v"].to(self.device)
        b = sample["b"].to(self.device)
        coords = sample["coords"].to(self.device)
        A = coo_triplets_to_sparse(A_i_v, b.numel())
        x = torch.zeros_like(b)
        residual = b - sparse_mv(A, x)
        initial_norm = torch.linalg.vector_norm(residual).item()
        x_true = sample.get("x")
        if x_true is not None:
            x_true = x_true.to(self.device)
        return {
            "A": A,
            "b": b,
            "coords": coords,
            "edge_index": A.indices(),
            "edge_attr": A.values().unsqueeze(1),
            "x": x,
            "x_true": x_true,
            "residual": residual,
            "initial_norm": initial_norm,
            "basis": None,
            "reductions": [],
            "low_reduction_streak": 0,
            "stop_reason": None,
            "model_time": 0.0,
            "total_time": 0.0,
            "iterations": 0,
            "active": initial_norm > 0.0,
        }

    @staticmethod
    def _model_batch(states: list[dict[str, Any]], active: list[int]) -> Batch:
        graphs = []
        for index in active:
            state = states[index]
            graphs.append(
                Data(
                    residual=state["residual"],
                    coords=state["coords"],
                    edge_index=state["edge_index"],
                    edge_attr=state["edge_attr"],
                    num_nodes=state["b"].numel(),
                )
            )
        return Batch.from_data_list(graphs)

    def _warmup(self, states: list[dict[str, Any]], iterations: int) -> None:
        if iterations <= 0 or not states:
            return
        active = list(range(len(states)))
        batch = self._model_batch(states, active)
        with torch.inference_mode():
            for _ in range(iterations):
                self.model(
                    batch.residual,
                    batch.coords,
                    batch.edge_index,
                    batch.edge_attr,
                    batch=batch.batch,
                )
        synchronize(self.device)

    def _solve_batch(
        self,
        samples: list[dict[str, Any]],
        warmup_iterations: int,
    ) -> list[SolveResult]:
        if not samples:
            return []
        states = [self._prepare(sample) for sample in samples]
        self._warmup(states, warmup_iterations)

        for _ in range(self.max_iterations):
            active = [i for i, state in enumerate(states) if state["active"]]
            if not active:
                break

            synchronize(self.device)
            iteration_start = perf_counter()
            batch = self._model_batch(states, active)

            synchronize(self.device)
            model_start = perf_counter()
            with torch.inference_mode():
                new_basis_batch = self.model(
                    batch.residual,
                    batch.coords,
                    batch.edge_index,
                    batch.edge_attr,
                    batch=batch.batch,
                )
            synchronize(self.device)
            model_elapsed = perf_counter() - model_start

            for local_index, state_index in enumerate(active):
                state = states[state_index]
                new_basis = new_basis_batch[batch.batch == local_index]
                basis = extend_orthonormal_basis(
                    state["basis"],
                    new_basis,
                    self.max_basis_size,
                    method=self.orthogonalization,
                )

                before = torch.linalg.vector_norm(state["residual"]).item()
                projection = projected_correction(state["A"], state["residual"], basis)
                state["x"] = state["x"] + projection.correction
                state["residual"] = state["b"] - sparse_mv(state["A"], state["x"])
                after = torch.linalg.vector_norm(state["residual"]).item()
                reduction = 0.0 if before == 0.0 else (before - after) / before

                state["basis"] = basis
                state["reductions"].append(reduction)
                state["iterations"] += 1
                state["model_time"] += model_elapsed / len(active)

                if after <= self.absolute_tolerance:
                    state["active"] = False
                    state["stop_reason"] = "tolerance"
                elif reduction < self.min_reduction:
                    state["low_reduction_streak"] += 1
                    if state["low_reduction_streak"] >= self.min_reduction_patience:
                        state["active"] = False
                        state["stop_reason"] = "stagnation"
                else:
                    state["low_reduction_streak"] = 0

            synchronize(self.device)
            iteration_elapsed = perf_counter() - iteration_start
            for state_index in active:
                states[state_index]["total_time"] += iteration_elapsed / len(active)

        results = []
        for state in states:
            final_norm = torch.linalg.vector_norm(state["residual"]).item()
            stop_reason = state["stop_reason"]
            if stop_reason is None:
                stop_reason = "zero_rhs" if state["initial_norm"] == 0.0 else "max_iterations"
            relative_l2 = None
            if state["x_true"] is not None:
                denominator = torch.linalg.vector_norm(state["x_true"]).item()
                if denominator > 0.0:
                    relative_l2 = (
                        torch.linalg.vector_norm(state["x"] - state["x_true"]).item() / denominator
                    )
            relative = 0.0 if state["initial_norm"] == 0.0 else final_norm / state["initial_norm"]
            results.append(
                SolveResult(
                    solution=state["x"],
                    iterations=state["iterations"],
                    initial_residual=state["initial_norm"],
                    final_residual=final_norm,
                    relative_residual=relative,
                    reductions=state["reductions"],
                    stop_reason=stop_reason,
                    model_time=state["model_time"],
                    total_time=state["total_time"],
                    relative_l2_error=relative_l2,
                )
            )
        return results
