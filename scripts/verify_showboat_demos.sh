#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Keep all tool caches inside the repo by default for reproducible verification.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$PROJECT_DIR/.cache/uv-tools}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.cache}"
export HOME="${HN_RERANK_HOME:-$PROJECT_DIR}"
mkdir -p "$UV_CACHE_DIR" "$UV_TOOL_DIR" "$XDG_CACHE_HOME"

if [[ "$#" -gt 0 ]]; then
    demos=("$@")
else
    mapfile -t demos < <(rg --files -g 'demo*.md' | sort)
fi

if [[ "${#demos[@]}" -eq 0 ]]; then
    echo "error: no demo markdown files found"
    exit 1
fi

status=0
for demo in "${demos[@]}"; do
    if [[ ! -f "$demo" ]]; then
        echo "[-] Missing demo file: $demo"
        status=1
        continue
    fi

    echo "[*] showboat verify $demo"
    if uvx showboat verify "$demo"; then
        echo "[+] PASS $demo"
    else
        echo "[-] FAIL $demo"
        status=1
    fi
done

exit "$status"
