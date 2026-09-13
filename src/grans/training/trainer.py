"""Batched progressive-bootstrap training."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from ..config import ExperimentConfig, StageConfig
from ..models.factory import CHECKPOINT_FORMAT_VERSION
from ..solvers.projection import (
    coo_triplets_to_sparse,
    extend_orthonormal_basis,
    projected_correction,
)
from .checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    exclusive_run_lock,
    restore_rng_state,
)


class ProgressiveBootstrapTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        device: torch.device | str,
        output_dir: str | Path,
        run_id: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        total_epochs = sum(stage.epochs for stage in config.training.stages)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs)
        self.epoch = 0
        self.best_loss = float("inf")
        self.last_loss = float("inf")

    def fit(self, samples: Sequence[dict[str, Any]]) -> Path:
        loader = DataLoader(
            samples,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=list,
        )
        total_epochs = sum(stage.epochs for stage in self.config.training.stages)
        if self.epoch > total_epochs:
            raise ValueError(
                f"checkpoint epoch {self.epoch} exceeds configured total {total_epochs}"
            )
        with exclusive_run_lock(self.output_dir):
            completed_before_stage = 0
            for stage_index, stage in enumerate(self.config.training.stages, start=1):
                completed_in_stage = max(0, self.epoch - completed_before_stage)
                if completed_in_stage < stage.epochs:
                    self._fit_stage(loader, stage, stage_index, completed_in_stage)
                completed_before_stage += stage.epochs
            final_path = self.output_dir / "final_model.pt"
            self.save_checkpoint(final_path, self.last_loss)
            return final_path

    def _fit_stage(
        self,
        loader: DataLoader,
        stage: StageConfig,
        stage_index: int,
        completed_epochs: int = 0,
    ) -> None:
        for stage_epoch in range(completed_epochs + 1, stage.epochs + 1):
            self.epoch += 1
            started = perf_counter()
            losses = [self.train_batch(batch, steps=stage.steps) for batch in loader]
            epoch_loss = float(np.mean(losses))
            self.scheduler.step()
            self.best_loss = min(self.best_loss, epoch_loss)
            self.last_loss = epoch_loss
            print(
                f"epoch={self.epoch} stage={stage_index} steps={stage.steps} "
                f"loss={epoch_loss:.6e} lr={self.scheduler.get_last_lr()[0]:.3e} "
                f"seconds={perf_counter() - started:.2f}",
                flush=True,
            )
            self.save_checkpoint(self.output_dir / "latest.pt", epoch_loss)
            if (
                self.epoch % self.config.training.save_interval == 0
                or stage_epoch == stage.epochs
            ):
                self.save_checkpoint(
                    self.output_dir / f"checkpoint_epoch_{self.epoch}.pt",
                    epoch_loss,
                )

    def train_batch(self, samples: Sequence[dict[str, Any]], steps: int) -> float:
        self.model.train()
        graphs = [self._prepare(sample) for sample in samples]
        residuals = [graph["b"].clone() for graph in graphs]
        accumulated: list[torch.Tensor | None] = [None] * len(graphs)
        total_loss = torch.zeros((), device=self.device)

        for step in range(steps):
            batch = self._build_model_batch(graphs, residuals)
            new_basis_batch = self.model(
                batch.residual,
                batch.coords,
                batch.edge_index,
                batch.edge_attr,
                batch=batch.batch,
            )
            next_residuals = []
            for graph_index, graph in enumerate(graphs):
                new_basis = new_basis_batch[batch.batch == graph_index]
                basis = extend_orthonormal_basis(
                    accumulated[graph_index],
                    new_basis,
                    self.config.training.max_basis_size,
                    method=self.config.training.orthogonalization,
                )
                projection = projected_correction(graph["A"], residuals[graph_index], basis)
                total_loss = total_loss + torch.mean(projection.residual.square())
                next_residuals.append(projection.residual.detach())
                accumulated[graph_index] = basis.detach()
            residuals = next_residuals

        total_loss = total_loss / (len(graphs) * steps)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clip,
        )
        self.optimizer.step()
        return total_loss.detach().item()

    def _prepare(self, sample: dict[str, Any]) -> dict[str, Any]:
        A_i_v = sample["A_i_v"].to(self.device)
        b = sample["b"].to(self.device)
        A = coo_triplets_to_sparse(A_i_v, b.numel())
        return {
            "A": A,
            "b": b,
            "coords": sample["coords"].to(self.device),
            "edge_index": A.indices(),
            "edge_attr": A.values().unsqueeze(1),
        }

    @staticmethod
    def _build_model_batch(
        graphs: Sequence[dict[str, Any]],
        residuals: Sequence[torch.Tensor],
    ) -> Batch:
        data = []
        for graph, residual in zip(graphs, residuals):
            data.append(
                Data(
                    residual=residual,
                    coords=graph["coords"],
                    edge_index=graph["edge_index"],
                    edge_attr=graph["edge_attr"],
                    num_nodes=graph["b"].numel(),
                )
            )
        return Batch.from_data_list(data)

    def save_checkpoint(self, path: str | Path, loss: float) -> None:
        atomic_torch_save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": loss,
                "best_loss": self.best_loss,
                "run_id": self.run_id,
                "rng_state": capture_rng_state(),
                "experiment_config": self.config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format {checkpoint.get('format_version')!r}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}"
            )
        if checkpoint.get("experiment_config") != self.config.to_dict():
            raise ValueError("checkpoint experiment configuration does not match --config")
        checkpoint_run_id = checkpoint.get("run_id")
        if checkpoint_run_id is not None and checkpoint_run_id != self.run_id:
            raise ValueError(
                f"checkpoint run ID {checkpoint_run_id!r} does not match {self.run_id!r}"
            )
        required = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
        if missing := sorted(required - set(checkpoint)):
            raise ValueError(f"checkpoint is not resumable; missing fields: {missing}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = int(checkpoint["epoch"])
        self.best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", float("inf"))))
        self.last_loss = float(checkpoint.get("loss", self.best_loss))
        restore_rng_state(checkpoint.get("rng_state"))
