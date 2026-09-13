from copy import deepcopy
from pathlib import Path

import torch
import yaml

from grans.data import DataGenerator


PROBLEM_DIR = Path(__file__).parents[1] / "configs" / "problems"


def _problem(name: str) -> dict:
    with (PROBLEM_DIR / f"{name}.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_all_paper_problem_families_follow_sample_schema() -> None:
    problem_types = [
        "poisson",
        "advection_diffusion",
        "helmholtz",
        "reaction_diffusion",
    ]
    for index, problem_type in enumerate(problem_types):
        generator = DataGenerator(
            problem=_problem(problem_type),
            mesh_size=32,
            randomize_params=True,
            seed=100 + index,
        )
        sample = generator.generate_sample(randomize=True)
        assert sample["A_i_v"].ndim == 2
        assert sample["A_i_v"].shape[1] == 3
        assert sample["b"].numel() == sample["x"].numel()
        assert sample["coords"].shape == (sample["b"].numel(), 2)
        assert all(torch.isfinite(value).all() for value in sample.values() if torch.is_tensor(value))


def test_generation_is_reproducible_for_a_fixed_seed() -> None:
    kwargs = dict(
        problem=_problem("poisson"),
        mesh_size=32,
        randomize_params=True,
        seed=123,
    )
    first = DataGenerator(**kwargs).generate_sample(randomize=True)
    second = DataGenerator(**kwargs).generate_sample(randomize=True)
    for key in ["A_i_v", "b", "x", "coords"]:
        assert torch.equal(first[key], second[key])


def test_problem_parameters_are_taken_from_config() -> None:
    problem = deepcopy(_problem("poisson"))
    problem["parameters"]["source_strength"] = 0.35
    generator = DataGenerator(problem=problem, mesh_size=32, randomize_params=False)
    sample = generator.generate_sample()
    assert sample["param_info"]["source_strength"] == 0.35
