"""Generate a deterministic G-RANS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..data.data_generator import DataGenerator
from ..data.io import save_dataset
from ..utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    seed = config.seed if args.split == "train" else config.seed + 10_000
    seed_everything(seed)
    count = args.samples
    if count is None:
        count = config.data.train_samples if args.split == "train" else config.data.test_samples
    if count <= 0:
        raise ValueError("--samples must be positive")

    generator = DataGenerator(
        problem=config.data.problem,
        mesh_size=config.data.mesh_size,
        device="cpu",
        randomize_params=config.data.randomize_params,
        param_variation=config.data.param_variation,
        use_preconditioner=config.data.use_preconditioner,
        seed=seed,
    )
    samples = []
    for index in range(count):
        sample = generator.generate_sample(randomize=config.data.randomize_geometry)
        samples.append(sample)
        print(
            f"generated {index + 1}/{count}: nodes={sample['b'].numel()} "
            f"nnz={sample['A_i_v'].shape[0]}",
            flush=True,
        )

    save_dataset(
        Path(args.output),
        samples,
        {
            "split": args.split,
            "seed": seed,
            "experiment_config": config.to_dict(),
        },
    )
    print(f"saved {count} samples to {args.output}")


if __name__ == "__main__":
    main()
