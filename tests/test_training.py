import math
from pathlib import Path

import pytest
import torch

from grans.config import ExperimentConfig, StageConfig
from grans.models.factory import build_model
from grans.training import ProgressiveBootstrapTrainer, latest_valid_checkpoint
from grans.training import checkpoint as checkpoint_io


def _identity_sample(scale: float):
    indices = torch.arange(12, dtype=torch.float32)
    values = torch.ones(12)
    return {
        "A_i_v": torch.stack([indices, indices, values], dim=1),
        "b": scale * torch.linspace(-1.0, 1.0, 12),
        "x": scale * torch.linspace(-1.0, 1.0, 12),
        "coords": torch.stack([torch.linspace(0.0, 1.0, 12), torch.zeros(12)], dim=1),
    }


def test_two_step_bootstrap_training(tmp_path) -> None:
    config = ExperimentConfig.load(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    trainer = ProgressiveBootstrapTrainer(
        build_model(config.model),
        config,
        "cpu",
        tmp_path,
    )
    loss = trainer.train_batch([_identity_sample(1.0), _identity_sample(2.0)], steps=2)
    assert math.isfinite(loss)
    assert loss >= 0.0


def test_release_model_returns_raw_candidates() -> None:
    config = ExperimentConfig.load(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    assert build_model(config.model).use_qr is False


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch) -> None:
    target = tmp_path / "latest.pt"
    target.write_bytes(b"previous-checkpoint")

    def interrupted_save(payload, handle) -> None:
        handle.write(b"partial-checkpoint")
        raise RuntimeError("simulated filesystem failure")

    monkeypatch.setattr(checkpoint_io.torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="failed to save checkpoint atomically"):
        checkpoint_io.atomic_torch_save({"epoch": 1}, target)

    assert target.read_bytes() == b"previous-checkpoint"
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_restores_training_state_and_skips_completed_epochs(tmp_path) -> None:
    config = ExperimentConfig.load(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config.training.stages = [StageConfig(steps=1, epochs=1), StageConfig(steps=1, epochs=1)]
    samples = [_identity_sample(1.0), _identity_sample(2.0)]
    first = ProgressiveBootstrapTrainer(
        build_model(config.model), config, "cpu", tmp_path, run_id="resume-test"
    )
    original_train_batch = first.train_batch

    def interrupt_after_first_epoch(batch, steps):
        if first.epoch > 1:
            raise KeyboardInterrupt
        return original_train_batch(batch, steps)

    first.train_batch = interrupt_after_first_epoch
    with pytest.raises(KeyboardInterrupt):
        first.fit(samples)
    assert first.epoch == 1

    resumed = ProgressiveBootstrapTrainer(
        build_model(config.model), config, "cpu", tmp_path, run_id="resume-test"
    )
    resumed.load_checkpoint(tmp_path / "latest.pt")
    assert resumed.epoch == 1
    assert resumed.best_loss == first.best_loss
    final_path = resumed.fit(samples)
    assert resumed.epoch == 2
    assert final_path.is_file()


def test_latest_valid_checkpoint_skips_corrupt_legacy_archive(tmp_path) -> None:
    valid = tmp_path / "checkpoint_epoch_50.pt"
    corrupt = tmp_path / "checkpoint_epoch_100.pt"
    torch.save({"model_state_dict": {}, "epoch": 50}, valid)
    corrupt.write_bytes(b"incomplete")

    assert latest_valid_checkpoint(tmp_path) == valid
