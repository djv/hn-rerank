#!/bin/bash
# HN Rerank Auto-Update Script
set -e

# Configuration
PROJECT_DIR="/home/dev/hn_rerank"
LOG_FILE="$PROJECT_DIR/update.log"

cd "$PROJECT_DIR"

# Ensure environment variables are loaded if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# API Keys check
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY not set in environment or .env file" >&2
    exit 1
fi

echo "[$(date)] Starting update..." >> "$LOG_FILE"

# Run the generation script using uv
# Arguments are now primarily handled via hn_rerank.toml
if uv run generate_html.py >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] Update successful." >> "$LOG_FILE"
else
    echo "[$(date)] Update FAILED. Check $LOG_FILE for details." >> "$LOG_FILE"
    exit 1
fi