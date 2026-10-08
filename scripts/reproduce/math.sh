#!/usr/bin/env bash
# Reproduce the 17 mathematical benchmark variants with hierarchical island MCTS.
set -euo pipefail

MODEL="${MODEL:-gpt-5}"
ITERATIONS="${ITERATIONS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reproduce}"

cd "$(dirname "$0")/../.."
uv sync --extra math

benchmarks=(
  benchmarks/math/circle_packing
  benchmarks/math/circle_packing_rect
  benchmarks/math/erdos_min_overlap
  benchmarks/math/first_autocorr_ineq
  benchmarks/math/second_autocorr_ineq
  benchmarks/math/third_autocorr_ineq
  benchmarks/math/uncertainty_ineq
  benchmarks/math/hexagon_packing/11
  benchmarks/math/hexagon_packing/12
  benchmarks/math/heilbronn_convex/13
  benchmarks/math/heilbronn_convex/14
  benchmarks/math/heilbronn_triangle
  benchmarks/math/minimizing_max_min_dist/2
  benchmarks/math/minimizing_max_min_dist/3
  benchmarks/math/matmul
  benchmarks/math/signal_processing
  benchmarks/math/sums_diffs_finite_sets
)

for dir in "${benchmarks[@]}"; do
  evaluator="$dir/evaluator.py"
  [[ -f "$evaluator" ]] || evaluator="$dir/evaluator"
  echo "== island_mcts: ${dir#benchmarks/} =="
  uv run coding-agent \
    "$dir/initial_program.py" \
    "$evaluator" \
    --config "$dir/config.yaml" \
    --search island_mcts \
    --model "$MODEL" \
    --iterations "$ITERATIONS" \
    --output "$OUTPUT_ROOT/island_mcts/${dir#benchmarks/}"
done

echo "math.sh: all ${#benchmarks[@]} runs finished."
