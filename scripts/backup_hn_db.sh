#!/usr/bin/env bash
set -euo pipefail

# Configurable via env or first CLI arg
DB_PATH="${1:-${HN_DB_PATH:-hn_rewrite.db}}"
RCLONE_REMOTE="${HN_BACKUP_REMOTE:-drive:hn-rewrite/backups}"
KEEP_N="${HN_KEEP_N:-30}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

WORK="$(mktemp -d)"
trap "rm -rf '$WORK'" EXIT

# 1. Safe SQLite snapshot (avoids copy of mid-write file)
sqlite3 "$DB_PATH" ".backup '$WORK/hn_rewrite.db'"

# 2. Checksum
( cd "$WORK" && sha256sum hn_rewrite.db > hn_rewrite.db.sha256 )

# 3. Upload to dated subfolder
rclone copyto "$WORK/hn_rewrite.db"        "$RCLONE_REMOTE/$TIMESTAMP/hn_rewrite.db"
rclone copyto "$WORK/hn_rewrite.db.sha256" "$RCLONE_REMOTE/$TIMESTAMP/hn_rewrite.db.sha256"

# 4. Verify upload checksum matches local
LOCAL_SHA="$(awk '{print $1}' "$WORK/hn_rewrite.db.sha256")"
REMOTE_SHA="$(rclone cat "$RCLONE_REMOTE/$TIMESTAMP/hn_rewrite.db.sha256" | awk '{print $1}')"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    echo "ERROR: checksum mismatch after upload" >&2
    exit 1
fi

echo "Backed up $DB_PATH to $RCLONE_REMOTE/$TIMESTAMP/"

# 5. Retention: keep newest N dated subfolders
if [[ "$KEEP_N" =~ ^[0-9]+$ ]] && (( KEEP_N > 0 )); then
    rclone lsf --dirs-only "$RCLONE_REMOTE" \
        | sort -r \
        | tail -n +"$((KEEP_N + 1))" \
        | while read -r d; do
            [[ -n "$d" ]] && rclone purge "$RCLONE_REMOTE/$d"
        done
fi
