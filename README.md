# Coding Agent

<p align="center">
  Hierarchical MCTS for LLM-guided scientific and systems discovery
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="Apache-2.0 license"></a>
  <img src="https://img.shields.io/badge/python-3.10–3.13-blue" alt="Python 3.10–3.13">
</p>

Hierarchical MCTS Coding Agent searches over executable programs. It asks an LLM to propose candidate algorithms, executes them in an isolated evaluation loop, and uses hierarchical Monte Carlo Tree Search (MCTS) to decide what to improve next.

The repository includes hierarchical MCTS, adaptive island management, program-lineage search, prompting, evaluation, checkpointing, monitoring, and benchmark harnesses. Setup scripts fetch the large input datasets used by a few ADRS workloads.

## Why this framework

Many program-discovery loops greedily refine the current best solution. That is effective early, but it can collapse onto one family of ideas. This framework maintains several evolving islands and searches at two levels:

```text
outer MCTS: choose or expand an island
    └── inner MCTS: choose a program lineage
            └── mode bandit: refine, seek novelty, or change paradigm
                    └── LLM proposal → evaluation → reward backup
```

Key capabilities:

- **Hierarchical MCTS** over both island lineages and program lineages.
- **Progressive widening** to grow the search only when evidence supports it.
- **Per-island mode selection** with UCT or discounted UCT for non-stationary search.
- **Quality-diversity archives** and configurable novelty rewards.
- **Dynamic islands and migration** to preserve competing solution families.
- **Reflection and paradigm shifts** when local improvement stalls.
- **Scalar or Pareto objectives**, including mixed maximize/minimize metrics.
- **Checkpoint/resume**, structured run logs, and a live browser dashboard.
- **Python, containerized, and Harbor-style evaluators**.

## Repository layout

```text
coding_agent/           framework package and CLI
  search/              MCTS engine and comparison search strategies
  evaluation/          Python, Docker, and Harbor evaluators
  context_builder/     prompt/context construction and reflection memory
  llm/                 model-provider interface and model pools
benchmarks/
  math/                mathematical discovery suite
  ADRS/                real-world systems optimization suite
configs/               reusable run configurations
examples/              minimal custom-task example
scripts/reproduce/     benchmark reproduction entry points
tests/                  unit and integration tests
```

## Installation

Requirements: Python 3.10–3.13, [uv](https://docs.astral.sh/uv/), and an API key for the selected model provider. Docker is optional and is only required by containerized evaluators.

```bash
git clone https://github.com/<your-github-username>/Coding-Agent.git
cd Coding-Agent

uv sync --extra math --extra adrs
export OPENAI_API_KEY="<your-key>"
```

Other OpenAI-compatible endpoints can be selected with `--api-base`; provider-prefixed model names such as `gemini/...` and `anthropic/...` are also supported.

## Quick start

Run hierarchical MCTS on the circle-packing benchmark:

```bash
uv run coding-agent \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search island_mcts \
  --model gpt-5 \
  --iterations 100 \
  --output outputs/circle_packing
```

Run an ADRS systems benchmark:

```bash
uv run coding-agent \
  benchmarks/ADRS/cloudcast/initial_program.py \
  benchmarks/ADRS/cloudcast/evaluator \
  --config benchmarks/ADRS/cloudcast/config.yaml \
  --search island_mcts \
  --model gpt-5 \
  --iterations 100 \
  --output outputs/cloudcast
```

Resume a run from a saved checkpoint:

```bash
uv run coding-agent \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search island_mcts \
  --checkpoint outputs/circle_packing/checkpoints/checkpoint_50 \
  --iterations 100
```

## Defining a discovery task

A task needs an evaluator and may optionally provide a seed program.

### 1. Seed program

Mark the mutable region with `EVOLVE-BLOCK` comments. If the markers are omitted, the complete file can be edited.

```python
# EVOLVE-BLOCK-START
def solve(instance):
    return baseline_solution(instance)
# EVOLVE-BLOCK-END
```

### 2. Evaluator

The evaluator receives the generated program path and returns numeric metrics. `combined_score` is maximized by default; artifacts are fed back into later proposals.

```python
def evaluate(program_path):
    score, diagnostics = run_experiment(program_path)
    return {
        "combined_score": score,
        "artifacts": {"diagnostics": diagnostics},
    }
```

Then run:

```bash
uv run coding-agent initial_program.py evaluator.py \
  --config configs/island_mcts.yaml \
  --iterations 100
```

The seed is optional. Supplying only an evaluator asks the model to construct the first solution from scratch:

```bash
uv run coding-agent evaluator.py \
  --config configs/island_mcts.yaml \
  --iterations 100
```

## Python API

```python
from coding_agent import run_discovery

result = run_discovery(
    evaluator="evaluator.py",
    initial_program="initial_program.py",
    config="configs/island_mcts.yaml",
    model="gpt-5",
    iterations=100,
    output_dir="outputs/my_experiment",
)

print(result.best_score)
print(result.best_solution)
```

For small experiments, `discover_solution()` also accepts inline source and a callable evaluator.

## Included benchmarks

### Mathematics

The suite covers 17 task variants across geometric packing, extremal geometry, autocorrelation inequalities, finite-set constructions, matrix multiplication, and signal processing. See [`benchmarks/math`](benchmarks/math/README.md) for the complete list.

### ADRS systems optimization

Five workloads exercise multi-cloud transfer planning, mixture-of-experts load balancing, GPU model placement, prefix-cache-aware SQL column ordering, and transaction scheduling. See [`benchmarks/ADRS`](benchmarks/ADRS/README.md) for task details and dataset setup.

Reproduce either suite from the repository root:

```bash
bash scripts/reproduce/math.sh
bash scripts/reproduce/adrs.sh

# Run both suites sequentially
bash scripts/reproduce/run_all.sh
```

Set `MODEL`, `ITERATIONS`, and `OUTPUT_ROOT` in the environment to override the defaults, for example:

```bash
MODEL=gpt-5 ITERATIONS=10 bash scripts/reproduce/math.sh
```


## License

Released under the [Apache License 2.0](LICENSE).
