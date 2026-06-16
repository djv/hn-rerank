# Lock File Safety - DO NOT DELETE LOCK FILES BLINDLY

## Critical Rule
Never delete `.cache/generate_html.lock` without first checking for a running process.

## Procedure (mandatory)
1. Check if lock file exists: `ls .cache/generate_html.lock 2>/dev/null`
2. If it exists, check for a running generate process:
   - `ps aux | grep generate_html | grep -v grep`
   - `lsof .cache/generate_html.lock`
3. If NO process holds the lock, the file is stale and safe to delete
4. If a process IS running, wait for it to complete — NEVER delete the lock

## Why
Deleting the lock while a process is running causes duplicate generation, race conditions, and corrupt dashboard output.
