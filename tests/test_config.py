from pathlib import Path

from grans.config import ExperimentConfig


def test_smoke_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    config = ExperimentConfig.load(path)
    assert config.data.pde_type == "poisson"
    assert config.data.problem["parameters"]["kappa_inner"] == 1.1
    assert config.data.problem_config == "problems/poisson.yaml"
    assert config.training.stages[0].steps == 1
    assert config.solver.orthogonalization == "incremental"


def test_problem_config_is_embedded_in_serialized_config() -> None:
    path = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    serialized = ExperimentConfig.load(path).to_dict()
    assert serialized["data"]["problem"]["pde_type"] == "poisson"
    assert serialized["data"]["problem"]["domain"]["vertices"][0] == [-1.0, -1.0]
