# Benchmarks

The maintained benchmark suite contains mathematical discovery tasks and ADRS systems-optimization workloads. Every task provides a seed program, an evaluator, and a problem-specific YAML configuration.

| Suite | Variants | Scope |
|---|---:|---|
| [`math/`](math/) | 17 | Packing, extremal geometry, inequalities, finite sets, matrix multiplication, and signal processing |
| [`ADRS/`](ADRS/) | 5 | Cloud transfer, MoE balancing, model placement, prefix caching, and transaction scheduling |

## Run a task

```bash
uv sync --extra math --extra adrs
export OPENAI_API_KEY="<your-key>"

uv run coding-agent \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search island_mcts \
  --iterations 100
```

Use the evaluator directory instead of `evaluator.py` to run the containerized version of a task. Task-specific READMEs document any additional data setup.

## Task structure

Plain Python evaluators use:

```text
task/
├── initial_program.py
├── evaluator.py
└── config.yaml
```

Containerized evaluators use:

```text
task/
├── initial_program.py
├── config.yaml
└── evaluator/
    ├── Dockerfile
    ├── evaluate.sh
    ├── evaluator.py
    └── requirements.txt
```

`evaluate.sh` receives the candidate path and evaluation mode, then prints one JSON object containing `status`, `combined_score`, optional numeric `metrics`, and optional text `artifacts`.

## Add a benchmark

1. Create a seed with an `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` region.
2. Implement `evaluate(program_path) -> dict`, returning at least `combined_score`.
3. Add a YAML file containing the domain prompt and evaluator limits.
4. Smoke-test the task with two iterations before starting a full run.

For dependencies that should not run on the host, copy the container layout from [`math/heilbronn_triangle`](math/heilbronn_triangle/).
