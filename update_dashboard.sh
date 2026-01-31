#!/bin/bash
# HN Rerank Auto-Update Script
set -e

# Configuration
PROJECT_DIR="/home/dev/hn_rerank"
USERNAME="pure_coder"
LOG_FILE="$PROJECT_DIR/update.log"

# API Keys - set via environment or systemd EnvironmentFile
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY not set" >&2
    exit 1
fi

cd "$PROJECT_DIR"

echo "[$(date)] Starting update for @$USERNAME..." >> "$LOG_FILE"

# Run the generation script using uv
if /home/dev/.local/bin/uv run generate_html.py "$USERNAME" --days 7 >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] Update successful." >> "$LOG_FILE"
else
    echo "[$(date)] Update FAILED. Check $LOG_FILE for details." >> "$LOG_FILE"
    exit 1
fi
