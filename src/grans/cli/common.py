from __future__ import annotations

import argparse

import torch


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cpu, cuda, or cuda:1 (default: auto)",
    )
