from pathlib import Path
from types import SimpleNamespace

import pytest

from grans.cli.evaluate import _default_summary_path, _summary, compact_evaluation


def _result(total_time: float, final_residual: float) -> dict:
    return {
        "iterations": 2,
        "initial_residual": 4.0,
        "final_residual": final_residual,
        "relative_residual": final_residual / 4.0,
        "reductions": [0.25, 0.25],
        "stop_reason": "max_iterations",
        "model_time": total_time / 2.0,
        "total_time": total_time,
        "relative_l2_error": 0.1,
    }


def test_default_summary_path() -> None:
    assert _default_summary_path(Path("results/evaluation.json")) == Path(
        "results/evaluation_summary.json"
    )
    assert _default_summary_path(Path("results/evaluation")) == Path(
        "results/evaluation_summary.json"
    )


def test_full_summary_uses_standard_deviation_selectively() -> None:
    results = [
        SimpleNamespace(
            iterations=2,
            initial_residual=4.0,
            final_residual=2.0,
            relative_residual=0.5,
            model_time=0.5,
            total_time=1.0,
            relative_l2_error=0.1,
            stop_reason="max_iterations",
        ),
        SimpleNamespace(
            iterations=4,
            initial_residual=6.0,
            final_residual=1.0,
            relative_residual=0.25,
            model_time=1.0,
            total_time=2.0,
            relative_l2_error=0.2,
            stop_reason="tolerance",
        ),
    ]

    summary = _summary(results)

    assert summary["initial_residual"] == {"mean": 5.0}
    assert summary["final_residual"] == {"mean": 1.5}
    assert summary["total_time"] == {"mean": 1.5, "std": 0.5}
    assert "variance" not in str(summary)


def test_compact_summary_contains_only_aggregate_metrics() -> None:
    results = [_result(1.0, 2.0), _result(2.0, 1.0)]
    payload = {
        "experiment_config": {
            "seed": 42,
            "data": {"pde_type": "poisson", "mesh_size": 64},
        },
        "checkpoint": {"path": "checkpoint.pt", "epoch": 1, "run_id": "smoke"},
        "environment": {"device": "cpu", "device_name": "CPU"},
        "grans": {"summary": {}, "samples": results},
        "baselines": {
            "gmres": {
                "summary": {"speedup_over_grans": 2.0},
                "samples": [
                    {
                        "time": 2.0,
                        "iterations": 4,
                        "info": 0,
                        "final_residual": 2.0,
                        "relative_residual": 0.5,
                    },
                    {
                        "time": 4.0,
                        "iterations": 6,
                        "info": 0,
                        "final_residual": 1.0,
                        "relative_residual": 0.25,
                    },
                ],
            }
        },
    }

    summary = compact_evaluation(payload)

    assert summary["grans"]["total_time_seconds"] == {
        "mean": 1.5,
        "std": 0.5,
    }
    assert summary["grans"]["final_residual"] == {"mean": 1.5}
    assert summary["baselines"]["gmres"]["samples"] == 2
    speedup = summary["baselines"]["gmres"]["speedup_over_grans"]
    assert speedup["mean"] == pytest.approx(2.0)
    assert speedup["std"] == pytest.approx(0.0)
    assert speedup["ratio_of_mean_times"] == pytest.approx(2.0)
    assert "variance" not in str(summary)
