"""Versioned dataset serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DATASET_FORMAT_VERSION = 1


def save_dataset(
    path: str | Path,
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_samples = []
    for sample in samples:
        cpu_samples.append(
            {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in sample.items()
            }
        )
    torch.save(
        {
            "format_version": DATASET_FORMAT_VERSION,
            "metadata": metadata,
            "samples": cpu_samples,
        },
        path,
    )


def load_dataset(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(
            "unsupported dataset format; regenerate it with the release data command"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("dataset must contain a non-empty samples list")
    required = {"A_i_v", "b", "coords"}
    for index, sample in enumerate(samples):
        missing = required - set(sample)
        if missing:
            raise ValueError(f"sample {index} is missing fields: {sorted(missing)}")
        if sample["A_i_v"].ndim != 2 or sample["A_i_v"].shape[1] != 3:
            raise ValueError(f"sample {index} has invalid A_i_v shape")
        if sample["b"].numel() != sample["coords"].shape[0]:
            raise ValueError(f"sample {index} has inconsistent b and coords")
    return samples, payload.get("metadata", {})
