#!/usr/bin/env bash
# Run the maintained benchmark suites sequentially to avoid oversubscribing resources.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
bash "$script_dir/math.sh"
bash "$script_dir/adrs.sh"

echo "All maintained benchmark runs finished."
