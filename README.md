# G-RANS

G-RANS is a research implementation of **Generalizable Residual-Aware Neural
Solvers for Sparse Systems** (ICML 2026). It solves sparse linear systems from
 PDE discretizations with residual-aware neural
subspace corrections.

Openreview: https://openreview.net/forum?id=uizi6lvkSW
ICML Poster: https://icml.cc/virtual/2026/poster/60991

## Installation

Python 3.10 or newer is required. Create an environment, install a PyTorch build
matching the target CUDA driver if needed, then install this repository in
editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CUDA users: install the appropriate PyTorch wheel first.
python -m pip install -e '.[dev]'
```

The package provides `grans-generate`, `grans-train`, `grans-evaluate`, and
`grans-summarize`. Check the environment with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest
ruff check .
```

## Configuration

An experiment YAML controls the complete workflow. The `data.problem_config`
path is resolved relative to that YAML and the parsed problem is embedded in
every dataset, checkpoint, and evaluation report.

- `configs/problems/*.yaml` defines PDE coefficients, domain geometry, boundary
  markers, inclusion geometry, and perturbation scales.
- `configs/paper/*.yaml` defines model, mesh size, sample counts, training,
  solver, and baseline evaluation settings.
- `configs/smoke.yaml` is a small CPU/GPU pipeline check.

Unknown keys and invalid values fail during config loading. `mesh_size` is a
target FEM node count, so the generated matrix dimension can vary slightly.
Parameter and geometry randomization are controlled independently. Test samples
use `seed + 10000` so they are deterministic and separate from training data.

## Step-by-step workflow

The reusable demonstration script runs the three stages independently while
keeping their artifacts in one directory:

```bash
scripts/run_pipeline_step_by_step.sh \
  configs/smoke.yaml artifacts/smoke cpu
```

The equivalent commands, which are convenient when extending the workflow, are:

```bash
grans-generate --config configs/smoke.yaml --split train \
  --output artifacts/smoke/data/train.pt
grans-generate --config configs/smoke.yaml --split test \
  --output artifacts/smoke/data/test.pt

grans-train --config configs/smoke.yaml \
  --data artifacts/smoke/data/train.pt \
  --output-dir artifacts/smoke/checkpoints --device cpu

grans-evaluate --config configs/smoke.yaml \
  --checkpoint artifacts/smoke/checkpoints/final_model.pt \
  --data artifacts/smoke/data/test.pt \
  --output artifacts/smoke/results/evaluation.json --device cpu
```

Use `--device cuda:0` for a visible GPU. `grans-evaluate` writes both the full
per-sample `evaluation.json` and the aggregate-only
`evaluation_summary.json`. The summary reports means for all core metrics and
standard deviations only for iteration counts, relative errors, timings, and
per-sample speedups.

If a full report already exists, regenerate only its compact view:

```bash
grans-summarize artifacts/smoke/results/evaluation.json
```

## Paper-scale reproduction

`scripts/reproduce.sh` performs dataset generation, training, and evaluation in
one command. It accepts a configuration and an artifact directory:

```bash
DEVICE=cuda:0 scripts/reproduce.sh \
  configs/paper/poisson_n1000.yaml artifacts/poisson_n1000
```

The output layout is:

```text
artifacts/poisson_n1000/
  data/train.pt
  data/test.pt
  checkpoints/final_model.pt
  checkpoints/checkpoint_epoch_*.pt
  results/evaluation.json
  results/evaluation_summary.json
```

The eight paper configurations cover the four PDE families at target sizes
1000 and 2000. Compare regenerated configuration hashes and artifact hashes
with the exact data/checkpoints used for publication before making numerical
reproduction claims.

## Checkpoints and resume

Training atomically updates `latest.pt` after every completed epoch and keeps
numbered checkpoints at stage boundaries and `training.save_interval`. Use a
unique run ID when several experiments share an output root:

```bash
grans-train --config configs/paper/poisson_n1000.yaml \
  --data artifacts/poisson_n1000/data/train.pt \
  --output-dir artifacts/runs --run-id poisson-n1000-seed42 \
  --device cuda:0
```

Resume from the newest readable checkpoint:

```bash
grans-train --config configs/paper/poisson_n1000.yaml \
  --data artifacts/poisson_n1000/data/train.pt \
  --output-dir artifacts/runs --run-id poisson-n1000-seed42 \
  --resume --device cuda:0
```

The saved state includes model, optimizer, scheduler, completed epoch, best
loss, and random-number generators. A run-directory lock prevents concurrent
writers. Atomic replacement protects the previous checkpoint from interrupted
writes; it cannot fix an exhausted disk, quota, inode limit, or unhealthy
network filesystem.

## Source layout

```text
src/grans/data/       FEM assembly, problem generation, dataset I/O
src/grans/models/     positional encoding and residual-aware Poly-GAT
src/grans/solvers/    projection, G-RANS, GMRES/FGMRES/MINRES baselines
src/grans/training/   progressive-bootstrap trainer and checkpoint handling
src/grans/cli/        generate, train, evaluate, and summarize commands
configs/              smoke, paper, and PDE problem configurations
scripts/              one-command and step-by-step workflows
tests/                unit and integration-oriented tests
```

Training and inference share `solvers/projection.py`. The default
`orthogonalization: incremental` projects each new candidate block against the
retained basis and QR-factorizes only that block; `full` remains available as a
reference path. Public dataset and checkpoint formats are versioned.

For the Chinese server guide, see
[`docs/SERVER_GUIDE_ZH.md`](docs/SERVER_GUIDE_ZH.md).
