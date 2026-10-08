#!/usr/bin/env bash
# Reproduce the five ADRS systems benchmarks with hierarchical island MCTS.
set -euo pipefail

MODEL="${MODEL:-gpt-5}"
ITERATIONS="${ITERATIONS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reproduce}"

cd "$(dirname "$0")/../.."
uv sync --extra adrs

cloudcast_data="benchmarks/ADRS/cloudcast/evaluator/profiles/cost.csv"
if [[ ! -f "$cloudcast_data" ]]; then
  bash benchmarks/ADRS/cloudcast/evaluator/download_dataset.sh
fi

sql_data="benchmarks/ADRS/llm_sql/evaluator/datasets"
if ! compgen -G "$sql_data/*.csv" >/dev/null; then
  bash benchmarks/ADRS/llm_sql/evaluator/download_dataset.sh
fi

eplb_data="benchmarks/ADRS/eplb/evaluator/expert-load.json"
if [[ ! -f "$eplb_data" ]] || grep -q "git-lfs.github.com/spec" "$eplb_data"; then
  bash benchmarks/ADRS/eplb/evaluator/download_dataset.sh
fi

benchmarks=(cloudcast eplb llm_sql prism txn_scheduling)

for name in "${benchmarks[@]}"; do
  dir="benchmarks/ADRS/$name"
  echo "== island_mcts: ADRS/$name =="
  uv run coding-agent \
    "$dir/initial_program.py" \
    "$dir/evaluator" \
    --config "$dir/config.yaml" \
    --search island_mcts \
    --model "$MODEL" \
    --iterations "$ITERATIONS" \
    --output "$OUTPUT_ROOT/island_mcts/ADRS/$name"
done

echo "adrs.sh: all ${#benchmarks[@]} runs finished."
