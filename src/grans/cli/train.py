"""Train G-RANS with batched progressive bootstrap."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..config import ExperimentConfig
from ..data.io import load_dataset
from ..models.factory import build_model
from ..training.trainer import ProgressiveBootstrapTrainer
from ..training.checkpoint import latest_valid_checkpoint
from ..utils import seed_everything
from .common import add_device_argument, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID; checkpoints are written under OUTPUT_DIR/RUN_ID",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from CHECKPOINT, or from the newest valid run checkpoint if omitted",
    )
    add_device_argument(parser)
    args = parser.parse_args()

    valid_run_id = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    if args.run_id is not None and not re.fullmatch(valid_run_id, args.run_id):
        parser.error(
            "--run-id must contain only letters, digits, '.', '_' or '-' "
            "and start alphanumeric"
        )

    config = ExperimentConfig.load(args.config)
    device = resolve_device(args.device)
    seed_everything(config.seed)
    samples, metadata = load_dataset(args.data)
    data_config = metadata.get("experiment_config", {}).get("data")
    if data_config is not None and data_config != config.to_dict()["data"]:
        raise ValueError("dataset data configuration does not match the training config")

    output_dir = Path(args.output_dir)
    if args.run_id is not None:
        output_dir = output_dir / args.run_id
    existing = (
        (output_dir / "latest.pt").exists()
        or (output_dir / "final_model.pt").exists()
        or any(output_dir.glob("checkpoint_epoch_*.pt"))
    )
    if existing and args.resume is None:
        parser.error(
            f"run directory already contains checkpoints: {output_dir}; "
            "use --resume or choose a different --run-id"
        )

    model = build_model(config.model)
    trainer = ProgressiveBootstrapTrainer(model, config, device, output_dir, args.run_id)
    if args.resume is not None:
        resume_path = (
            latest_valid_checkpoint(output_dir)
            if args.resume == "latest"
            else Path(args.resume)
        )
        trainer.load_checkpoint(resume_path)
        print(f"resumed run from {resume_path} at epoch {trainer.epoch}", flush=True)
    final_path = trainer.fit(samples)
    print(f"saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
