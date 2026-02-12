#!/bin/bash
# HN Rerank Auto-Update Script
set -e

# Configuration
PROJECT_DIR="/home/dev/hn_rerank"
LOG_FILE="$PROJECT_DIR/update.log"

cd "$PROJECT_DIR"

# Ensure environment variables are loaded.
# Priority:
# 1) Existing process environment (e.g., systemd EnvironmentFile)
# 2) User secret file (~/.config/hn_rerank/secrets.env)
# 3) Repo-local .env (legacy fallback)
if [ -z "${GROQ_API_KEY:-}" ] && [ -f "$HOME/.config/hn_rerank/secrets.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$HOME/.config/hn_rerank/secrets.env"
    set +a
fi

if [ -z "${GROQ_API_KEY:-}" ] && [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# API Keys check
if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "Error: GROQ_API_KEY not set in environment or .env file" >&2
    exit 1
fi

echo "[$(date)] Starting update..." >> "$LOG_FILE"

# Run the generation script using uv
# Arguments are now primarily handled via hn_rerank.toml
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
    echo "Error: uv not found in PATH and \$HOME/.local/bin/uv is missing" >&2
    exit 1
fi

if "$UV_BIN" run generate_html.py --no-tldr >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] Update successful." >> "$LOG_FILE"
else
    echo "[$(date)] Update FAILED. Check $LOG_FILE for details." >> "$LOG_FILE"
    exit 1
fi
