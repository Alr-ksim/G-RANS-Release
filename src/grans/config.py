"""Validated experiment configuration for G-RANS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class ModelConfig:
    coord_dim: int = 2
    pos_enc_dim: int = 16
    hidden_channels: int = 48
    num_layers: int = 4
    basis_per_layer: int = 5
    heads: int = 3
    dropout: float = 0.08


@dataclass
class DataConfig:
    pde_type: str = "poisson"
    mesh_size: int = 1000
    train_samples: int = 1000
    test_samples: int = 64
    randomize_geometry: bool = True
    randomize_params: bool = True
    param_variation: float = 0.1
    use_preconditioner: bool = False
    problem_config: str | None = None
    problem: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageConfig:
    steps: int
    epochs: int


@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip: float = 1.0
    save_interval: int = 50
    max_basis_size: int = 400
    orthogonalization: str = "incremental"
    stages: list[StageConfig] = field(
        default_factory=lambda: [
            StageConfig(steps=1, epochs=150),
            StageConfig(steps=2, epochs=150),
            StageConfig(steps=3, epochs=300),
        ]
    )


@dataclass
class SolverConfig:
    max_iterations: int = 50
    max_basis_size: int = 400
    absolute_tolerance: float = 1.0e-5
    min_reduction: float = 0.05
    min_reduction_patience: int = 2
    orthogonalization: str = "incremental"


@dataclass
class EvaluationConfig:
    batch_size: int = 64
    warmup_iterations: int = 1
    compare_gmres: bool = True
    compare_fgmres: bool = True
    compare_minres: bool = True
    krylov_restart: int = 40
    krylov_max_iterations: int = 500


@dataclass
class ExperimentConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        _reject_unknown(raw, cls, "root")
        stages = [StageConfig(**stage) for stage in raw.get("training", {}).get("stages", [])]
        training_raw = dict(raw.get("training", {}))
        if stages:
            training_raw["stages"] = stages
        data = _build(DataConfig, raw.get("data", {}), "data")
        if data.problem_config:
            problem_path = Path(data.problem_config)
            if not problem_path.is_absolute():
                problem_path = config_path.parent / problem_path
            with problem_path.open("r", encoding="utf-8") as handle:
                problem = yaml.safe_load(handle) or {}
            if not isinstance(problem, dict):
                raise ValueError(f"problem config must contain a mapping: {problem_path}")
            data.problem = problem

        config = cls(
            seed=raw.get("seed", 42),
            model=_build(ModelConfig, raw.get("model", {}), "model"),
            data=data,
            training=_build(TrainingConfig, training_raw, "training"),
            solver=_build(SolverConfig, raw.get("solver", {}), "solver"),
            evaluation=_build(EvaluationConfig, raw.get("evaluation", {}), "evaluation"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        supported = {"poisson", "advection_diffusion", "helmholtz", "reaction_diffusion"}
        if self.data.pde_type not in supported:
            raise ValueError(f"data.pde_type must be one of {sorted(supported)}")
        if self.data.mesh_size <= 0 or self.data.train_samples <= 0 or self.data.test_samples <= 0:
            raise ValueError("mesh size and sample counts must be positive")
        if not 0.0 <= self.data.param_variation <= 1.0:
            raise ValueError("data.param_variation must be in [0, 1]")
        _validate_problem(self.data.problem, self.data.pde_type)
        if self.model.num_layers <= 0 or self.model.basis_per_layer <= 0:
            raise ValueError("model basis dimensions must be positive")
        if not self.training.stages:
            raise ValueError("training.stages cannot be empty")
        if self.training.save_interval <= 0:
            raise ValueError("training.save_interval must be positive")
        if any(stage.steps <= 0 or stage.epochs <= 0 for stage in self.training.stages):
            raise ValueError("each training stage needs positive steps and epochs")
        if self.solver.min_reduction_patience <= 0:
            raise ValueError("solver.min_reduction_patience must be positive")
        valid_orthogonalization = {"incremental", "full"}
        if self.training.orthogonalization not in valid_orthogonalization:
            raise ValueError("training.orthogonalization must be 'incremental' or 'full'")
        if self.solver.orthogonalization not in valid_orthogonalization:
            raise ValueError("solver.orthogonalization must be 'incremental' or 'full'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _reject_unknown(raw: dict[str, Any], cls: type[Any], section: str) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {unknown}")


def _build(cls: type[T], raw: dict[str, Any], section: str) -> T:
    _reject_unknown(raw, cls, section)
    return cls(**raw)


def _validate_problem(problem: dict[str, Any], expected_type: str) -> None:
    if not problem:
        raise ValueError("data.problem_config must resolve to a non-empty problem config")
    allowed = {"pde_type", "domain", "inclusion", "parameters"}
    unknown = sorted(set(problem) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in problem config: {unknown}")
    if problem.get("pde_type") != expected_type:
        raise ValueError(
            "problem config pde_type does not match data.pde_type: "
            f"{problem.get('pde_type')!r} != {expected_type!r}"
        )

    domain = problem.get("domain")
    inclusion = problem.get("inclusion")
    parameters = problem.get("parameters")
    if not isinstance(domain, dict) or not isinstance(inclusion, dict):
        raise ValueError("problem config requires mapping sections: domain and inclusion")
    if not isinstance(parameters, dict):
        raise ValueError("problem config requires a parameters mapping")

    domain_required = {
        "vertices",
        "dirichlet_points",
        "jitter_vertices",
        "top_corner_jitter",
    }
    inclusion_required = {
        "center",
        "radius",
        "center_jitter",
        "radius_relative_variation",
    }
    if missing := sorted(domain_required - set(domain)):
        raise ValueError(f"missing problem domain keys: {missing}")
    if missing := sorted(inclusion_required - set(inclusion)):
        raise ValueError(f"missing problem inclusion keys: {missing}")
    if unknown := sorted(set(domain) - domain_required):
        raise ValueError(f"unknown problem domain keys: {unknown}")
    if unknown := sorted(set(inclusion) - inclusion_required):
        raise ValueError(f"unknown problem inclusion keys: {unknown}")
    common = {"kappa_inner", "kappa_outer", "source_strength", "bc_scale"}
    specific = {
        "poisson": {"poisson_shift"},
        "advection_diffusion": {"velocity_scale"},
        "helmholtz": {"reaction_inner", "reaction_outer"},
        "reaction_diffusion": {"reaction_inner", "reaction_outer"},
    }
    required_parameters = common | specific[expected_type]
    missing = sorted(required_parameters - set(parameters))
    if missing:
        raise ValueError(f"missing problem parameters: {missing}")
    if unknown := sorted(set(parameters) - required_parameters):
        raise ValueError(f"unknown problem parameters: {unknown}")
    for name in parameters:
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"problem parameter {name!r} must be numeric")

    vertices = domain.get("vertices")
    boundary_points = domain.get("dirichlet_points")
    if not isinstance(vertices, list) or len(vertices) != 4:
        raise ValueError("problem domain.vertices must contain four 2D points")
    if not isinstance(boundary_points, list) or not boundary_points:
        raise ValueError("problem domain.dirichlet_points cannot be empty")
    _validate_point_list(vertices, "problem domain.vertices")
    _validate_point_list(boundary_points, "problem domain.dirichlet_points")
    _validate_point_list([inclusion.get("center")], "problem inclusion.center")
    jitter_vertices = domain.get("jitter_vertices")
    if not isinstance(jitter_vertices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 4
        for index in jitter_vertices
    ):
        raise ValueError("problem domain.jitter_vertices must contain vertex indices 0..3")
    for section, names in (
        (domain, ("top_corner_jitter",)),
        (inclusion, ("radius", "radius_relative_variation", "center_jitter")),
    ):
        for name in names:
            value = section.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"problem {name!r} must be a non-negative number")


def _validate_point_list(points: Any, section: str) -> None:
    if not isinstance(points, list):
        raise ValueError(f"{section} must be a list")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{section} must contain 2D points")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
            raise ValueError(f"{section} coordinates must be numeric")
