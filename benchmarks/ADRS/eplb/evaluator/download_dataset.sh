#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
url="https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json"

echo "Downloading EPLB workload (~234 MB)..."
curl --fail --location --progress-bar "$url" --output "$script_dir/expert-load.json"
echo "Saved $script_dir/expert-load.json"
