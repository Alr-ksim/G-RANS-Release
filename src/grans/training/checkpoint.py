"""Durable checkpoint I/O and run-level coordination."""

from __future__ import annotations

import fcntl
import os
import random
import shutil
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write a checkpoint beside its target and publish it with ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    except Exception as error:
        try:
            free_bytes = shutil.disk_usage(target.parent).free
            free_text = f"{free_bytes / (1024**3):.2f} GiB"
        except OSError:
            free_text = "unknown"
        raise RuntimeError(
            f"failed to save checkpoint atomically to {target} "
            f"(filesystem free space: {free_text}); check disk space, quota, "
            "filesystem health, and that each training process uses a separate run ID"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def latest_valid_checkpoint(run_dir: str | Path) -> Path:
    """Return the newest readable checkpoint, falling back past partial legacy files."""
    directory = Path(run_dir)
    candidates: list[Path] = []
    for preferred in (directory / "latest.pt", directory / "final_model.pt"):
        if preferred.is_file():
            candidates.append(preferred)

    def epoch_number(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    archives = sorted(directory.glob("checkpoint_epoch_*.pt"), key=epoch_number, reverse=True)
    candidates.extend(path for path in archives if path not in candidates)
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found in run directory: {directory}")

    errors = []
    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "model_state_dict" in payload:
                return candidate
            errors.append(f"{candidate.name}: invalid payload")
        except Exception as error:
            errors.append(f"{candidate.name}: {error}")
    details = "; ".join(errors)
    raise RuntimeError(f"no readable checkpoints found in {directory}: {details}")


@contextmanager
def exclusive_run_lock(run_dir: str | Path) -> Iterator[None]:
    """Prevent concurrent trainers from writing the same run directory."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".training.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"run directory is already being trained: {directory} ({owner})"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
