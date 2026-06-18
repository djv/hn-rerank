#!/usr/bin/env python3
"""Import feedback from hn_rerank's JSON format into the rewrite's SQLite DB."""

import json
import sys
from pathlib import Path
from database import Database, Story

DEFAULT_SOURCE_PATH = Path.home() / "hn_rerank/.cache/user_feedback/dashboard_feedback.json"


def migrate(source_path: Path, db_path: str) -> None:
    if not source_path.exists():
        print(f"Source feedback file not found at {source_path}", file=sys.stderr)
        return

    db = Database(db_path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read source file: {e}", file=sys.stderr)
        return

    records = raw.get("records", {})
    imported = 0

    for rec in records.values():
        story_id = rec.get("id")
        action = rec.get("action")

        if story_id is None or action not in ("up", "neutral", "down"):
            continue

        title = rec.get("title", "")
        url = rec.get("url")
        text_content = rec.get("text_content", "")
        source = rec.get("source", "hn")
        story_time = rec.get("time", 0) or 0
        score = rec.get("score", 0) or 0
        comment_count = rec.get("comment_count")
        discussion_url = rec.get("discussion_url")

        # Insert into stories table if details are present so embeddings can be pre-computed / cached
        if title and text_content:
            db.upsert_story(
                Story(
                    id=story_id,
                    title=title,
                    url=url,
                    score=score,
                    time=story_time,
                    text_content=text_content,
                    source=source,
                    comment_count=comment_count,
                    discussion_url=discussion_url,
                )
            )

        # Insert into feedback
        db.upsert_feedback(
            story_id=story_id,
            action=action,
        )
        imported += 1

    db.close()
    print(f"Imported {imported} feedback records into {db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--db", default="hn_rewrite.db")
    args = parser.parse_args()
    migrate(Path(args.source), args.db)
