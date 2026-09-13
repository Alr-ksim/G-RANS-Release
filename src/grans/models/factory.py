"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import ModelConfig
from .poly_gat import PolyGATWithPE


CHECKPOINT_FORMAT_VERSION = 1


def build_model(config: ModelConfig) -> PolyGATWithPE:
    return PolyGATWithPE(
        coord_dim=config.coord_dim,
        pos_enc_dim=config.pos_enc_dim,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        basis_per_layer=config.basis_per_layer,
        heads=config.heads,
        dropout=config.dropout,
        use_residual_norm=True,
        use_qr=False,
    )


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | str,
) -> tuple[PolyGATWithPE, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    experiment = checkpoint.get("experiment_config")
    if not isinstance(experiment, dict) or "model" not in experiment:
        raise ValueError("checkpoint does not contain experiment_config.model")
    model = build_model(ModelConfig(**experiment["model"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
