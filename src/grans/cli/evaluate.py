"""Evaluate G-RANS and matched-accuracy Krylov baselines."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy
import torch

from ..config import ExperimentConfig
from ..data.data_generator import DataGenerator
from ..data.io import load_dataset
from ..models.factory import load_model_checkpoint
from ..solvers.fgmres import fgmres_torch_solve_with_history, jacobi_preconditioner
from ..solvers.gmres import gmres_torch_solve
from ..solvers.grans import GRANSSolver, SolveResult
from ..solvers.krylov import minres_torch_solve_with_history
from ..solvers.projection import coo_triplets_to_sparse, sparse_mv
from ..utils import seed_everything, synchronize
from .common import add_device_argument, resolve_device


def _statistics(values: list[float], include_std: bool = True) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    statistics = {"mean": float(np.mean(array))}
    if include_std:
        statistics["std"] = float(np.std(array))
    return statistics


def _summary(results: list[SolveResult]) -> dict[str, Any]:
    numeric = {
        "iterations": True,
        "initial_residual": False,
        "final_residual": False,
        "relative_residual": True,
        "model_time": True,
        "total_time": True,
        "relative_l2_error": True,
    }
    summary: dict[str, Any] = {"samples": len(results)}
    for key, include_std in numeric.items():
        values = [getattr(result, key) for result in results]
        values = [value for value in values if value is not None]
        if values:
            summary[key] = _statistics(values, include_std=include_std)
    reasons = sorted({result.stop_reason for result in results})
    summary["stop_reasons"] = {
        reason: sum(result.stop_reason == reason for result in results) for reason in reasons
    }
    return summary


def _baseline_call(
    name: str,
    A: torch.Tensor,
    b: torch.Tensor,
    rtol: float,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, int, float, int]:
    if name == "gmres":
        return gmres_torch_solve(
            A,
            b,
            rtol=rtol,
            atol=0.0,
            restart=config.evaluation.krylov_restart,
            maxiter=config.evaluation.krylov_max_iterations,
        )
    if name == "fgmres":
        x, info, residual, iterations, _, _ = fgmres_torch_solve_with_history(
            A,
            b,
            rtol=rtol,
            atol=0.0,
            restart=config.evaluation.krylov_restart,
            maxiter=config.evaluation.krylov_max_iterations,
            D_inv=jacobi_preconditioner(A),
        )
        return x, info, residual, iterations
    if name == "minres":
        x, info, residual, iterations, _, _ = minres_torch_solve_with_history(
            A,
            b,
            rtol=rtol,
            atol=0.0,
            maxiter=config.evaluation.krylov_max_iterations,
        )
        return x, info, residual, iterations
    raise ValueError(f"unknown baseline {name}")


def _run_baseline(
    name: str,
    samples: list[dict[str, Any]],
    neural_results: list[SolveResult],
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    records = []
    if samples and config.evaluation.warmup_iterations > 0:
        sample = samples[0]
        b = sample["b"].to(device)
        A = coo_triplets_to_sparse(sample["A_i_v"].to(device), b.numel())
        target_rtol = neural_results[0].relative_residual
        for _ in range(config.evaluation.warmup_iterations):
            _baseline_call(name, A, b, target_rtol, config)
        synchronize(device)
    for sample, neural in zip(samples, neural_results):
        b = sample["b"].to(device)
        A = coo_triplets_to_sparse(sample["A_i_v"].to(device), b.numel())
        target_rtol = neural.relative_residual if neural.initial_residual > 0.0 else 0.0
        synchronize(device)
        started = perf_counter()
        x, info, _, iterations = _baseline_call(name, A, b, target_rtol, config)
        synchronize(device)
        elapsed = perf_counter() - started
        final_residual = torch.linalg.vector_norm(b - sparse_mv(A, x)).item()
        initial_residual = torch.linalg.vector_norm(b).item()
        records.append(
            {
                "time": elapsed,
                "iterations": iterations,
                "info": info,
                "final_residual": final_residual,
                "relative_residual": 0.0
                if initial_residual == 0.0
                else final_residual / initial_residual,
            }
        )
    times = [record["time"] for record in records]
    neural_times = [result.total_time for result in neural_results]
    return {
        "summary": {
            "time": _statistics(times),
            "iterations": _statistics([record["iterations"] for record in records]),
            "final_residual": _statistics(
                [record["final_residual"] for record in records], include_std=False
            ),
            "relative_residual": _statistics(
                [record["relative_residual"] for record in records]
            ),
            "speedup_over_grans": float(np.mean(times) / np.mean(neural_times)),
        },
        "samples": records,
    }


def compact_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    """Build an aggregate-only view from a full evaluation payload."""
    experiment = payload["experiment_config"]
    result_records = payload["grans"]["samples"]
    numeric = {
        "iterations": True,
        "initial_residual": False,
        "final_residual": False,
        "relative_residual": True,
        "model_time": True,
        "total_time": True,
        "relative_l2_error": True,
    }
    grans: dict[str, Any] = {"samples": len(result_records)}
    for key, include_std in numeric.items():
        values = [result.get(key) for result in result_records]
        values = [float(value) for value in values if value is not None]
        if values:
            output_key = f"{key}_seconds" if key in {"model_time", "total_time"} else key
            grans[output_key] = _statistics(values, include_std=include_std)
    reasons = sorted({result["stop_reason"] for result in result_records})
    grans["stop_reasons"] = {
        reason: sum(result["stop_reason"] == reason for result in result_records)
        for reason in reasons
    }

    baseline_summaries: dict[str, Any] = {}
    neural_times = [float(result["total_time"]) for result in result_records]
    for name, baseline in payload["baselines"].items():
        records = baseline["samples"]
        metrics: dict[str, Any] = {"samples": len(records)}
        fields = {
            "time_seconds": ("time", True),
            "iterations": ("iterations", True),
            "final_residual": ("final_residual", False),
            "relative_residual": ("relative_residual", True),
        }
        for output_key, (record_key, include_std) in fields.items():
            metrics[output_key] = _statistics(
                [float(record[record_key]) for record in records],
                include_std=include_std,
            )
        per_sample_speedups = [
            float(record["time"]) / neural_time
            for record, neural_time in zip(records, neural_times)
            if neural_time > 0.0
        ]
        speedup = _statistics(per_sample_speedups)
        speedup["ratio_of_mean_times"] = float(
            baseline["summary"]["speedup_over_grans"]
        )
        metrics["speedup_over_grans"] = speedup
        baseline_summaries[name] = metrics

    return {
        "format_version": 1,
        "problem": {
            "pde_type": experiment["data"]["pde_type"],
            "mesh_size": experiment["data"]["mesh_size"],
            "test_samples": len(result_records),
            "seed": experiment["seed"],
        },
        "checkpoint": payload["checkpoint"],
        "environment": payload["environment"],
        "grans": grans,
        "baselines": baseline_summaries,
    }


def _default_summary_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}_summary{output.suffix}")
    return output.with_name(f"{output.name}_summary.json")


def _generate_test_samples(config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = config.seed + 10_000
    generator = DataGenerator(
        problem=config.data.problem,
        mesh_size=config.data.mesh_size,
        device="cpu",
        randomize_params=config.data.randomize_params,
        param_variation=config.data.param_variation,
        use_preconditioner=config.data.use_preconditioner,
        seed=seed,
    )
    samples = [
        generator.generate_sample(randomize=config.data.randomize_geometry)
        for _ in range(config.data.test_samples)
    ]
    return samples, {"generated_for_evaluation": True, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="Optional deterministic test dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Compact aggregate JSON path (default: OUTPUT with '_summary' suffix)",
    )
    add_device_argument(parser)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    device = resolve_device(args.device)
    seed_everything(config.seed)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    checkpoint_model = checkpoint["experiment_config"]["model"]
    if checkpoint_model != config.to_dict()["model"]:
        raise ValueError("checkpoint model configuration does not match --config")

    if args.data:
        samples, data_metadata = load_dataset(args.data)
        test_data_config = data_metadata.get("experiment_config", {}).get("data")
        if test_data_config is not None and test_data_config != config.to_dict()["data"]:
            raise ValueError("test dataset data configuration does not match --config")
        if len(samples) < config.data.test_samples:
            raise ValueError(
                f"test dataset has {len(samples)} samples, need {config.data.test_samples}"
            )
        samples = samples[: config.data.test_samples]
    else:
        samples, data_metadata = _generate_test_samples(config)

    solver = GRANSSolver(
        model=model,
        device=device,
        max_iterations=config.solver.max_iterations,
        max_basis_size=config.solver.max_basis_size,
        absolute_tolerance=config.solver.absolute_tolerance,
        min_reduction=config.solver.min_reduction,
        min_reduction_patience=config.solver.min_reduction_patience,
        orthogonalization=config.solver.orthogonalization,
    )
    results = solver.solve(
        samples,
        batch_size=config.evaluation.batch_size,
        warmup_iterations=config.evaluation.warmup_iterations,
    )

    baselines = {}
    enabled = {
        "gmres": config.evaluation.compare_gmres,
        "fgmres": config.evaluation.compare_fgmres,
        "minres": config.evaluation.compare_minres
        and config.data.pde_type in {"poisson", "reaction_diffusion"},
    }
    for name, should_run in enabled.items():
        if should_run:
            print(f"running matched-accuracy {name}", flush=True)
            baselines[name] = _run_baseline(name, samples, results, config, device)

    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    payload = {
        "format_version": 1,
        "experiment_config": config.to_dict(),
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "epoch": checkpoint.get("epoch"),
            "run_id": checkpoint.get("run_id"),
        },
        "data": data_metadata,
        "environment": environment,
        "grans": {
            "summary": _summary(results),
            "samples": [result.metrics() for result in results],
        },
        "baselines": baselines,
    }
    output = Path(args.output)
    summary_output = (
        Path(args.summary_output) if args.summary_output else _default_summary_path(output)
    )
    if summary_output.resolve() == output.resolve():
        raise ValueError("--summary-output must be different from --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_payload = compact_evaluation(payload)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["grans"]["summary"], indent=2))
    print(f"saved evaluation to {output}")
    print(f"saved compact summary to {summary_output}")


if __name__ == "__main__":
    main()
