# G-RANS 服务器运行指南

以下命令均在包含 `pyproject.toml`、`src/` 和 `configs/` 的仓库根目录执行。

## 1. 环境准备

建议使用 Python 3.10 或 3.11，并为项目建立独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

GPU 服务器先按 NVIDIA 驱动和 CUDA 版本从 PyTorch 官网安装匹配的 PyTorch，随后安装本项目：

```bash
python -m pip install -e '.[dev]'
```

检查环境：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); \
print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import grans, torch_geometric, meshpy, scipy; print('imports OK')"
pytest
ruff check .
```

## 2. 配置和数据

配置分为两层：

- `configs/problems/*.yaml`：PDE 系数、区域几何、边界和扰动参数。
- `configs/paper/*.yaml`：模型、网格规模、样本数、训练、求解和基线设置。

`data.problem_config` 相对于实验 YAML 解析。修改问题配置后必须重新生成数据；解析后的完整
问题、实验配置和随机种子会嵌入数据集与 checkpoint。`mesh_size` 是目标节点数，实际矩阵大小
会因网格剖分略有变化。测试样本使用 `seed + 10000`，与训练样本保持独立。

## 3. 分步工作流

推荐先运行小规模 smoke 流程：

```bash
scripts/run_pipeline_step_by_step.sh configs/smoke.yaml artifacts/smoke cpu
```

GPU 测试：

```bash
scripts/run_pipeline_step_by_step.sh configs/smoke.yaml artifacts/smoke cuda:0
```

脚本对应三个可独立扩展的阶段：

```bash
# 生成训练集和测试集
grans-generate --config configs/smoke.yaml --split train \
  --output artifacts/smoke/data/train.pt
grans-generate --config configs/smoke.yaml --split test \
  --output artifacts/smoke/data/test.pt

# 训练
grans-train --config configs/smoke.yaml \
  --data artifacts/smoke/data/train.pt \
  --output-dir artifacts/smoke/checkpoints --device cpu

# 评估
grans-evaluate --config configs/smoke.yaml \
  --checkpoint artifacts/smoke/checkpoints/final_model.pt \
  --data artifacts/smoke/data/test.pt \
  --output artifacts/smoke/results/evaluation.json --device cpu
```

评估输出两份文件：

- `evaluation.json`：完整的逐样例 G-RANS 和基线记录。
- `evaluation_summary.json`：聚合均值，以及迭代数、相对误差、耗时和逐样例加速比的标准差。

已有完整结果可以直接转换：

```bash
grans-summarize artifacts/smoke/results/evaluation.json
```

## 4. 论文规模复现

使用通用脚本运行任意论文配置：

```bash
DEVICE=cuda:0 scripts/reproduce.sh \
  configs/paper/poisson_n1000.yaml artifacts/poisson_n1000
```

输出目录结构：

```text
artifacts/poisson_n1000/
  data/train.pt
  data/test.pt
  checkpoints/final_model.pt
  checkpoints/checkpoint_epoch_*.pt
  results/evaluation.json
  results/evaluation_summary.json
```

## 5. 后台运行与断点续训

长任务可以后台运行：

```bash
mkdir -p logs
nohup env DEVICE=cuda:0 scripts/reproduce.sh \
  configs/paper/poisson_n1000.yaml artifacts/poisson_n1000 \
  > logs/poisson_n1000.log 2>&1 &
```

多个任务必须使用不同 GPU、不同输出目录或不同 `RUN_ID`。训练器每个完整 epoch 原子更新
`latest.pt`，并在阶段结束及 `training.save_interval` 到达时保存带 epoch 编号的 checkpoint。

```bash
grans-train --config configs/paper/poisson_n1000.yaml \
  --data artifacts/poisson_n1000/data/train.pt \
  --output-dir artifacts/runs --run-id poisson-n1000-seed42 \
  --resume --device cuda:0
```

`--resume` 默认读取最新可读 checkpoint，也可以显式传入 checkpoint 路径。恢复内容包括模型、
优化器、学习率调度器、已完成 epoch、最优 loss 和随机数状态；

```bash
df -h /path/to/output
df -i /path/to/output
du -sh /path/to/output
quota -s 2>/dev/null || true
```
