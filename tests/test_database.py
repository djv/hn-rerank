import time
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from database import Database, Story


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test.db"
    db_instance = Database(str(db_file))
    yield db_instance
    db_instance.close()


def test_upsert_and_get_story(db):
    story = Story(
        id=123,
        title="Test Story",
        url="https://example.com/test",
        score=100,
        time=1600000000,
        text_content="Some body text",
        source="hn",
        comment_count=42,
        discussion_url="https://news.ycombinator.com/item?id=123",
    )
    db.upsert_story(story)

    fetched = db.get_story(123)
    assert fetched is not None
    assert fetched.id == story.id
    assert fetched.title == story.title
    assert fetched.url == story.url
    assert fetched.score == story.score
    assert fetched.time == story.time
    assert fetched.text_content == story.text_content
    assert fetched.source == story.source
    assert fetched.comment_count == story.comment_count
    assert fetched.discussion_url == story.discussion_url

    # Check update
    updated_story = Story(
        id=123,
        title="Updated Title",
        url="https://example.com/test",
        score=105,
        time=1600000000,
        text_content="Some updated text",
        source="hn",
        comment_count=43,
        discussion_url="https://news.ycombinator.com/item?id=123",
    )
    db.upsert_story(updated_story)
    fetched2 = db.get_story(123)
    assert fetched2.title == "Updated Title"
    assert fetched2.score == 105


def test_get_stories(db):
    s1 = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    s2 = Story(id=2, title="S2", url=None, score=20, time=200, text_content="T2")
    db.upsert_story(s1)
    db.upsert_story(s2)

    stories = db.get_stories([1, 2, 3])
    assert len(stories) == 2
    ids = {s.id for s in stories}
    assert ids == {1, 2}


def test_upsert_embedding_roundtrip(db):
    story = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    db.upsert_story(story)

    vec = np.random.randn(384).astype(np.float32)
    db.upsert_embedding(1, "v1", vec)

    fetched = db.get_embedding(1, "v1")
    assert fetched is not None
    assert np.allclose(fetched, vec)

    # Cache miss test
    assert db.get_embedding(2, "v1") is None
    assert db.get_embedding(1, "v2") is None


def test_get_embeddings_batch(db):
    s1 = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    s2 = Story(id=2, title="S2", url=None, score=20, time=200, text_content="T2")
    db.upsert_story(s1)
    db.upsert_story(s2)

    vec1 = np.ones(384, dtype=np.float32)
    vec2 = np.zeros(384, dtype=np.float32)
    db.upsert_embedding(1, "v1", vec1)
    db.upsert_embedding(2, "v1", vec2)

    batch = db.get_embeddings_batch([1, 2, 3], "v1")
    assert len(batch) == 2
    assert np.allclose(batch[1], vec1)
    assert np.allclose(batch[2], vec2)


def test_feedback_crud(db):
    story = Story(
        id=456,
        title="Feedback Title",
        url="https://example.com/feedback",
        score=0,
        time=0,
        text_content="Feedback text content",
        source="rss_lobsters",
    )
    db.upsert_story(story)
    db.upsert_feedback(story_id=456, action="up")

    feedbacks = db.get_all_feedback()
    assert len(feedbacks) == 1
    assert feedbacks[0].story_id == 456
    assert feedbacks[0].action == "up"
    assert feedbacks[0].title == "Feedback Title"
    assert feedbacks[0].url == "https://example.com/feedback"
    assert feedbacks[0].text_content == "Feedback text content"
    assert feedbacks[0].source == "rss_lobsters"

    # Update
    db.upsert_feedback(story_id=456, action="down")
    feedbacks2 = db.get_all_feedback()
    assert feedbacks2[0].action == "down"

    # Delete
    db.delete_feedback(456)
    assert len(db.get_all_feedback()) == 0


def test_prune_stories(db):
    story = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    db.upsert_story(story)

    # Pruning with age = 0 should delete everything
    deleted = db.prune_stories(max_age_days=0)
    assert deleted == 1
    assert db.get_story(1) is None


def test_prune_stories_preserves_feedback_stories(db):
    story1 = Story(id=1, title="S1", url=None, score=10, time=100, text_content="T1")
    story2 = Story(id=2, title="S2", url=None, score=20, time=100, text_content="T2")
    db.upsert_story(story1)
    db.upsert_story(story2)
    db.upsert_feedback(1, "up")

    # Pruning with age = 0 should delete story2 but not story1
    deleted = db.prune_stories(max_age_days=0)
    assert deleted == 1
    assert db.get_story(1) is not None
    assert db.get_story(2) is None


def test_feedback_training_data(db):
    db.upsert_story(
        Story(
            id=1,
            title="T1",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text1",
            source="hn",
        )
    )
    db.upsert_story(
        Story(
            id=2,
            title="T2",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text2",
            source="hn",
        )
    )
    db.upsert_story(
        Story(
            id=3,
            title="T3",
            url=None,
            score=100,
            time=1600000000,
            text_content="Text3",
            source="hn",
        )
    )

    db.upsert_feedback(1, "up")
    db.upsert_feedback(2, "down")
    db.upsert_feedback(3, "neutral")

    stories, labels, vote_times = db.get_feedback_for_training()
    assert len(stories) == 3
    assert len(labels) == 3

    # Map mapping IDs to actions labels
    id_to_label = {s.id: lbl for s, lbl in zip(stories, labels)}
    assert id_to_label[1] == 2  # up
    assert id_to_label[2] == 0  # down
    assert id_to_label[3] == 1  # neutral


@given(
    # Timestamps relative to now (in days offset around the threshold)
    fetched_offsets=st.lists(
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
        min_size=5,
        max_size=50,
    ),
    # Indices of stories to attach feedback to
    feedback_indices=st.sets(st.integers(min_value=0, max_value=49)),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_story_pruning_integrity_invariants(
    tmp_path, fetched_offsets, feedback_indices
):
    import uuid

    db_file = tmp_path / f"test_prune_{uuid.uuid4().hex}.db"
    db = Database(str(db_file))
    try:
        now = time.time()
        max_age_days = 30
        cutoff = now - (max_age_days * 86400)

        stories_meta = []
        # Insert stories with variable ages
        for i, offset_days in enumerate(fetched_offsets):
            # fetched_at is offset around the threshold
            fetched_at = cutoff + (offset_days * 86400)
            story = Story(
                id=i,
                title=f"Story {i}",
                url=None,
                score=100,
                time=int(now),
                text_content="Content",
            )
            db.upsert_story(story)

            # Override fetched_at directly in DB to simulate temporal aging
            db.conn.execute(
                "UPDATE stories SET fetched_at = ? WHERE id = ?", (fetched_at, i)
            )

            # Apply feedback if indexed
            has_feedback = i in feedback_indices and i < len(fetched_offsets)
            if has_feedback:
                db.upsert_feedback(i, "up")

            stories_meta.append((i, fetched_at, has_feedback))

        db.prune_stories(max_age_days=max_age_days)

        # Assert temporal and referential invariants
        for sid, fetched_at, has_feedback in stories_meta:
            story = db.get_story(sid)
            is_older = fetched_at < cutoff

            if is_older:
                if has_feedback:
                    # Invariant: Must survive despite age due to referential link
                    assert story is not None, (
                        f"Story {sid} with feedback was pruned despite relative age."
                    )
                else:
                    # Invariant: Must be deleted
                    assert story is None, (
                        f"Old story {sid} without feedback survived pruning."
                    )
            else:
                # Invariant: Young stories always survive
                assert story is not None, f"Young story {sid} was incorrectly pruned."
    finally:
        db.close()
        try:
            db_file.unlink()
        except OSError:
            pass
