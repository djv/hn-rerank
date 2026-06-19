from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
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
                    fetched_at     REAL NOT NULL
                )
            """)
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

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_signals (
                    story_id    INTEGER NOT NULL,
                    signal_type TEXT NOT NULL,
                    scraped_at  REAL NOT NULL,
                    PRIMARY KEY (story_id, signal_type)
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS article_cache (
                    story_id    INTEGER PRIMARY KEY,
                    url         TEXT NOT NULL,
                    article_text TEXT NOT NULL,
                    fetched_at  REAL NOT NULL,
                    FOREIGN KEY (story_id) REFERENCES stories(id) ON DELETE CASCADE
                )
            """)

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
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO stories (
                    id, title, url, score, time, text_content, source,
                    comment_count, discussion_url, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    url=excluded.url,
                    score=excluded.score,
                    time=excluded.time,
                    text_content=excluded.text_content,
                    source=excluded.source,
                    comment_count=excluded.comment_count,
                    discussion_url=excluded.discussion_url,
                    fetched_at=excluded.fetched_at
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
            text_content=row[5],
            source=row[6],
            comment_count=row[7],
            discussion_url=row[8],
        )

    def get_story(self, story_id: int) -> Story | None:
        cursor = self.conn.execute(
            """
            SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url
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
            SELECT id, title, url, score, time, text_content, source, comment_count, discussion_url
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

    # User signals
    def set_user_signals(self, signal_type: str, ids: set[int]) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM user_signals WHERE signal_type = ?", (signal_type,)
            )
            now = time.time()
            self.conn.executemany(
                "INSERT INTO user_signals (story_id, signal_type, scraped_at) VALUES (?, ?, ?)",
                [(story_id, signal_type, now) for story_id in ids],
            )

    def get_user_signals(self) -> dict[str, set[int]]:
        cursor = self.conn.execute("SELECT story_id, signal_type FROM user_signals")
        res: dict[str, set[int]] = {"favorite": set(), "upvote": set(), "hidden": set()}
        for story_id, signal_type in cursor.fetchall():
            if signal_type in res:
                res[signal_type].add(story_id)
        return res

    # Article body cache
    def get_cached_article(self, story_id: int, url: str) -> str | None:
        cursor = self.conn.execute(
            "SELECT url, article_text FROM article_cache WHERE story_id = ?",
            (story_id,),
        )
        row = cursor.fetchone()
        if row and row[0] == url:
            return row[1]
        return None

    def set_cached_article(self, story_id: int, url: str, article_text: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO article_cache (story_id, url, article_text, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(story_id) DO UPDATE SET
                    url=excluded.url,
                    article_text=excluded.article_text,
                    fetched_at=excluded.fetched_at
                """,
                (story_id, url, article_text, time.time()),
            )

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
                   f.updated_at, COALESCE(s.score, 0), COALESCE(s.time, 0)
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
                )
            )
            labels.append(action_to_label[action])
            vote_times.append(updated_at)
        return stories, labels, vote_times
