# Expert Parallelism Load Balancer (EPLB)

This benchmark optimizes an Expert Parallelism Load Balancer (EPLB) for Mixture-of-Experts (MoE) models. The goal is to rearrange and replicate experts across GPUs to balance load while keeping the rebalancing algorithm itself fast.

## Setup

1. **Install PyTorch** (required by the evaluator):

   ```bash
   uv pip install torch
   ```

2. **Download the workload file**:

   ```bash
   bash benchmarks/ADRS/eplb/evaluator/download_dataset.sh
   ```

3. **Set your API key:**

   ```bash
   export OPENAI_API_KEY=...
   ```

## Run

From the repo root:

```bash
uv run coding-agent \
  benchmarks/ADRS/eplb/initial_program.py \
  benchmarks/ADRS/eplb/evaluator \
  -c benchmarks/ADRS/eplb/config.yaml \
  -s island_mcts \
  -i 100 \
  -o eplb_output
```

Or from this directory:

```bash
uv run coding-agent initial_program.py evaluator \
  -c config.yaml \
  -s island_mcts \
  -i 100
```

## Evaluate a saved program

```bash
python evaluate_best_program.py
```

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Baseline `rebalance_experts` function to evolve |
| `evaluator/` | Containerized scorer and workload setup |
| `config.yaml` | Task-specific config (LLM, evaluator timeout, system prompt) |
| `evaluate_best_program.py` | Standalone script to evaluate a saved best program |
| `evaluator/expert-load.json` | Workload data (must be downloaded — see Setup) |
