from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Story:
    id: int
    title: str
    url: str | None
    score: int
    time: int
    text_content: str
    source: str = "hn"
    comment_count: int | None = None
    discussion_url: str | None = None
    comment_count_at_fetch: int = 0
    self_text: str = ""
    top_comments: str = ""
    article_body: str = ""


@dataclass(frozen=True)
class FeedbackRecord:
    story_id: int
    action: Literal["up", "neutral", "down"]
    title: str
    url: str | None
    text_content: str
    source: str
    updated_at: float


class Database:
    def __init__(self, path: str = "hn_rewrite.db") -> None:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    id             INTEGER PRIMARY KEY,
                    title          TEXT NOT NULL,
                    url            TEXT,
                    score          INTEGER NOT NULL DEFAULT 0,
                    time           INTEGER NOT NULL DEFAULT 0,
                    text_content   TEXT NOT NULL DEFAULT '',
                    source         TEXT NOT NULL DEFAULT 'hn',
                    comment_count  INTEGER,
                    discussion_url TEXT,
                    fetched_at     REAL NOT NULL,
                    comment_count_at_fetch INTEGER NOT NULL DEFAULT 0,
                    self_text      TEXT NOT NULL DEFAULT '',
                    top_comments   TEXT NOT NULL DEFAULT '',
                    article_body   TEXT NOT NULL DEFAULT ''
                )
            """)
            cursor = self.conn.execute("PRAGMA table_info(stories)")
            columns = {row[1] for row in cursor.fetchall()}
            if "comment_count_at_fetch" not in columns:
                self.conn.execute(
                    "ALTER TABLE stories ADD COLUMN comment_count_at_fetch INTEGER NOT NULL DEFAULT 0"
                )
            for col in ("self_text", "top_comments", "article_body"):
                if col not in columns:
                    self.conn.execute(
                        f"ALTER TABLE stories ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )

            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stories_time ON stories(time)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stories_source ON stories(source)"
            )

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    story_id      INTEGER PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    embedding     BLOB NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                )
            """)

            # Run migration of article_cache to stories table if article_cache exists
            tbl_cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='article_cache'")
            if tbl_cursor.fetchone():
                logging.info("Migrating article_cache records to stories table...")
                self.conn.execute("""
                    UPDATE stories SET article_body = (
                      SELECT article_text FROM article_cache 
                      WHERE article_cache.story_id = stories.id
                    ) WHERE EXISTS (
                      SELECT 1 FROM article_cache WHERE article_cache.story_id = stories.id
                    )
                """)
                self.conn.execute("DROP TABLE article_cache")
                logging.info("Migration completed, article_cache table dropped.")

            cursor = self.conn.execute("PRAGMA table_info(feedback)")
            columns = [row[1] for row in cursor.fetchall()]

            if columns:
                if "title" in columns:
                    self.conn.execute("""
                        CREATE TABLE feedback_new (
                            story_id     INTEGER PRIMARY KEY,
                            action       TEXT NOT NULL CHECK(action IN ('up', 'neutral', 'down')),
                            updated_at   REAL NOT NULL,
                            FOREIGN KEY (story_id) REFERENCES stories(id)
                        )
                    """)
                    self.conn.execute("""
                        INSERT INTO feedback_new (story_id, action, updated_at)
                        SELECT story_id, action, updated_at FROM feedback
                    """)
                    self.conn.execute("DROP TABLE feedback")
                    self.conn.execute("ALTER TABLE feedback_new RENAME TO feedback")
            else:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        story_id     INTEGER PRIMARY KEY,
                        action       TEXT NOT NULL CHECK(action IN ('up', 'neutral', 'down')),
                        updated_at   REAL NOT NULL,
                        FOREIGN KEY (story_id) REFERENCES stories(id)
                    )
                """)

    def close(self) -> None:
        self.conn.close()

    # Stories
    def upsert_story(self, story: Story) -> None:
        # Check if the story already exists and has longer cached content
        cursor = self.conn.execute(
            "SELECT self_text, top_comments, article_body FROM stories WHERE id = ?",
            (story.id,)
        )
        row = cursor.fetchone()
        if row:
            db_self, db_comments, db_body = row[0] or "", row[1] or "", row[2] or ""
            
            # Keep the longest available version of each field
            final_self = story.self_text if len(story.self_text) >= len(db_self) else db_self
            final_comments = story.top_comments if len(story.top_comments) >= len(db_comments) else db_comments
            final_body = story.article_body if len(story.article_body) >= len(db_body) else db_body
            
            # Recompose if database fields were merged
            if final_self != story.self_text or final_comments != story.top_comments or final_body != story.article_body:
                from pipeline import compose_story_text
                new_text = compose_story_text(story.title, final_self, final_comments, final_body)
                story = replace(
                    story,
                    self_text=final_self,
                    top_comments=final_comments,
                    article_body=final_body,
                    text_content=new_text,
                )

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO stories (
                    id, title, url, score, time, text_content, source,
                    comment_count, discussion_url, fetched_at,
                    comment_count_at_fetch, self_text, top_comments,
                    article_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    url=excluded.url,
                    score=excluded.score,
                    time=excluded.time,
                    text_content=excluded.text_content,
                    source=excluded.source,
                    comment_count=excluded.comment_count,
                    discussion_url=excluded.discussion_url,
                    fetched_at=excluded.fetched_at,
                    comment_count_at_fetch=excluded.comment_count_at_fetch,
                    self_text=excluded.self_text,
                    top_comments=excluded.top_comments,
                    article_body=excluded.article_body
                """,
                (
                    story.id,
                    story.title,
                    story.url,
                    story.score,
                    story.time,
                    story.text_content,
                    story.source,
                    story.comment_count,
                    story.discussion_url,
                    time.time(),
                    story.comment_count_at_fetch,
                    story.self_text,
                    story.top_comments,
                    story.article_body,
                ),
            )

    @staticmethod
    def _row_to_story(row: tuple) -> Story:
        return Story(
            id=row[0],
            title=row[1],
            url=row[2],
            score=row[3],
            time=row[4],
            text_content=row[5] or "",
            source=row[6],
            comment_count=row[7],
            discussion_url=row[8],
            comment_count_at_fetch=row[9],
            self_text=row[10] or "",
            top_comments=row[11] or "",
            article_body=row[12] or "",
        )

    def get_story(self, story_id: int) -> Story | None:
        cursor = self.conn.execute(
            """
            SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url,
                   comment_count_at_fetch, self_text, top_comments, article_body
            FROM stories WHERE id = ?
            """,
            (story_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_story(row)

    def get_stories(self, ids: list[int]) -> list[Story]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url,
                   comment_count_at_fetch, self_text, top_comments, article_body
            FROM stories WHERE id IN ({placeholders})
        """
        cursor = self.conn.execute(query, ids)
        return [self._row_to_story(row) for row in cursor.fetchall()]

    def prune_stories(self, max_age_days: int = 60) -> int:
        cutoff = time.time() - (max_age_days * 86400)
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM stories WHERE fetched_at < ? AND id NOT IN (SELECT story_id FROM feedback)",
                (cutoff,),
            )
            return cursor.rowcount

    # Embeddings
    def upsert_embedding(
        self, story_id: int, model_version: str, vec: NDArray[np.float32]
    ) -> None:
        blob = vec.astype(np.float32).tobytes()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO embeddings (story_id, model_version, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT(story_id) DO UPDATE SET
                    model_version=excluded.model_version,
                    embedding=excluded.embedding
                """,
                (story_id, model_version, blob),
            )

    def get_embedding(
        self, story_id: int, model_version: str
    ) -> NDArray[np.float32] | None:
        cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE story_id = ? AND model_version = ?",
            (story_id, model_version),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def get_embeddings_batch(
        self, ids: list[int], model_version: str
    ) -> dict[int, NDArray[np.float32]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT story_id, embedding FROM embeddings
            WHERE model_version = ? AND story_id IN ({placeholders})
        """
        params = [model_version] + ids
        cursor = self.conn.execute(query, params)
        return {
            row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor.fetchall()
        }



    # Feedback
    def upsert_feedback(
        self,
        story_id: int,
        action: Literal["up", "neutral", "down"],
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO feedback (story_id, action, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(story_id) DO UPDATE SET
                    action=excluded.action,
                    updated_at=excluded.updated_at
                """,
                (story_id, action, time.time()),
            )

    def get_all_feedback(self) -> list[FeedbackRecord]:
        cursor = self.conn.execute(
            """
            SELECT f.story_id, f.action, s.title, s.url, s.text_content, s.source, f.updated_at
            FROM feedback f
            LEFT JOIN stories s ON s.id = f.story_id
            ORDER BY f.updated_at DESC
            """
        )
        return [
            FeedbackRecord(
                story_id=row[0],
                action=row[1],
                title=row[2] or "",
                url=row[3],
                text_content=row[4] or "",
                source=row[5] or "hn",
                updated_at=row[6],
            )
            for row in cursor.fetchall()
        ]

    def delete_feedback(self, story_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM feedback WHERE story_id = ?", (story_id,))

    def get_feedback_for_training(self) -> tuple[list[Story], list[int], list[float]]:
        cursor = self.conn.execute(
            """
            SELECT f.story_id, f.action, s.title, s.url, s.text_content, s.source,
                   f.updated_at, COALESCE(s.score, 0), COALESCE(s.time, 0),
                   COALESCE(s.comment_count, 0), COALESCE(s.self_text, ''),
                   COALESCE(s.top_comments, ''), COALESCE(s.article_body, '')
            FROM feedback f
            LEFT JOIN stories s ON s.id = f.story_id
            """
        )
        stories: list[Story] = []
        labels: list[int] = []
        vote_times: list[float] = []
        action_to_label = {"down": 0, "neutral": 1, "up": 2}
        for (
            story_id,
            action,
            title,
            url,
            text_content,
            source,
            updated_at,
            score,
            story_time,
            comment_count,
            self_text,
            top_comments,
            article_body,
        ) in cursor.fetchall():
            if action not in action_to_label:
                continue

            stories.append(
                Story(
                    id=story_id,
                    title=title or "",
                    url=url,
                    score=score,
                    time=story_time,
                    text_content=text_content or "",
                    source=source or "hn",
                    comment_count=comment_count,
                    self_text=self_text,
                    top_comments=top_comments,
                    article_body=article_body,
                )
            )
            labels.append(action_to_label[action])
            vote_times.append(updated_at)
        return stories, labels, vote_times
