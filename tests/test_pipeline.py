import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Literal
from database import Database, Story
from pipeline import (
    Embedder,
    RankedStory,
    clean_text,
    compose_story_text,
    get_or_compute_embeddings,
    mmr_filter,
    rank_stories,
    Config,
)


@pytest.fixture
def db():
    db_instance = Database(":memory:")
    yield db_instance
    db_instance.close()


@pytest.fixture(scope="module")
def embedder():
    # Uses the real downloaded ONNX model
    return Embedder("onnx_model")


def test_embedder_output_shape(embedder):
    texts = ["Hello world", "Hacker News rewrite plan"]
    embs = embedder.encode(texts)
    assert embs.shape == (2, 384)
    # Check normalization: norms should be close to 1.0
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedder_cache_hit(db, embedder):
    story = Story(
        id=999,
        title="Cache Test",
        url=None,
        score=0,
        time=0,
        text_content="Unique test content for caching.",
    )
    db.upsert_story(story)

    # First call: computes and caches
    embs1 = get_or_compute_embeddings([story], embedder, db)
    assert embs1.shape == (1, 384)

    # Second call: check if cached
    model_version = "all-MiniLM-L6-v2|mean|norm|256"
    cached = db.get_embedding(999, model_version)
    assert cached is not None
    assert np.allclose(cached, embs1[0])

    # Re-run get_or_compute_embeddings, should load from cache
    embs2 = get_or_compute_embeddings([story], embedder, db)
    assert np.allclose(embs1, embs2)


def test_compose_text():
    title = "<strong>Some title</strong>"
    self_text = "Some self text content."
    comments = ["First comment text here.", "Second comment is long."]
    composed = compose_story_text(title, self_text, comments)
    assert "Some title" in composed
    assert "Some self text" in composed
    assert "First comment" in composed


def test_strip_html():
    raw_html = "<p>Hello &amp; welcome to <a href='#'>Hacker News</a>!</p>"
    cleaned = clean_text(raw_html)
    assert cleaned == "Hello & welcome to Hacker News!"


def test_clean_text():
    assert clean_text("Valid text content here that has enough length.") != ""
    assert clean_text("Short", min_len=20) == ""  # too short
    assert clean_text("!!!???---***+++///\\\\|") == ""  # excessive punctuation
    # Braille block characters
    assert clean_text("⠠⠓⠑⠫⠕ ⠺⠕⠗⠇⠙") == ""


def test_rank_no_feedback_fallback(db, embedder):
    candidates = [
        Story(
            id=1,
            title="Machine Learning",
            url=None,
            score=0,
            time=0,
            text_content="AI and machine learning tutorial.",
        ),
        Story(
            id=2,
            title="Cooking recipes",
            url=None,
            score=0,
            time=0,
            text_content="How to bake bread and cook pasta.",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    config = Config()
    ranked = rank_stories(
        candidates=candidates,
        candidate_embeddings=cand_embs,
        db=db,
        config=config,
        embedder=embedder,
    )

    assert len(ranked) == 2
    # Without feedback, all stories default to score 0.5
    assert ranked[0].score == 0.5
    assert ranked[1].score == 0.5


def test_rank_svm_path(db, embedder):
    config = Config()
    # Populate DB with enough feedback to activate Feedback SVM (min_feedback_labels = 10)
    # We need >= 10 up (label=2) and >= 10 down (label=0)
    for i in range(10):
        db.upsert_story(
            Story(
                id=100 + i,
                title=f"Up story {i}",
                url=None,
                score=0,
                time=0,
                text_content="Deep learning AI research artificial intelligence",
            )
        )
        db.upsert_story(
            Story(
                id=200 + i,
                title=f"Down story {i}",
                url=None,
                score=0,
                time=0,
                text_content="Baking sourdough bread cake cookie recipe kitchen",
            )
        )
        db.upsert_feedback(100 + i, "up")
        db.upsert_feedback(200 + i, "down")

    candidates = [
        Story(
            id=1,
            title="AI systems",
            url=None,
            score=0,
            time=0,
            text_content="Training large language models and neural networks.",
        ),
        Story(
            id=2,
            title="Cake recipe",
            url=None,
            score=0,
            time=0,
            text_content="Delicious chocolate chip cake baking guide.",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = rank_stories(
        candidates=candidates,
        candidate_embeddings=cand_embs,
        db=db,
        config=config,
        embedder=embedder,
    )

    assert len(ranked) == 2
    # AI story should be ranked 1st
    assert ranked[0].story.id == 1
    # Check that score is probability-based (usually between 0.0 and 1.0)
    assert 0.0 <= ranked[0].score <= 1.0


def test_rss_synthetic_id():
    # Stable negative ID check
    link1 = "https://example.com/story-1"
    link2 = "https://example.com/story-2"

    import hashlib

    def get_id(link):
        h = hashlib.md5(link.encode("utf-8")).digest()
        val = int.from_bytes(h[:4], "big")
        return -(val % (2**31))

    id1_a = get_id(link1)
    id1_b = get_id(link1)
    id2 = get_id(link2)

    assert id1_a == id1_b
    assert id1_a != id2
    assert id1_a < 0
    assert id2 < 0


@given(st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=2, max_size=50))
def test_mmr_output_is_subset(scores):
    ranked = []
    embeddings_map = {}
    for i, s in enumerate(scores):
        story = Story(id=i, title=f"S{i}", url=None, score=0, time=0, text_content="")
        ranked.append(RankedStory(story=story, score=s, best_match_title=""))
        v = np.zeros(384, dtype=np.float32)
        v[i % 384] = 1.0
        embeddings_map[i] = v

    filtered = mmr_filter(ranked, embeddings_map, threshold=0.85, limit=10)

    filtered_ids = [item.story.id for item in filtered]
    input_ids = [item.story.id for item in ranked]
    for fid in filtered_ids:
        assert fid in input_ids
    assert filtered_ids == sorted(filtered_ids, key=lambda x: input_ids.index(x))


def test_rank_no_feedback_frontpage_sort(db, embedder):
    config = Config()
    # Absolutely no feedback in DB
    candidates = [
        Story(
            id=1, title="Low points story", url=None, score=10, time=0, text_content=""
        ),
        Story(
            id=2,
            title="High points story",
            url=None,
            score=500,
            time=0,
            text_content="",
        ),
    ]
    cand_embs = embedder.encode([s.text_content for s in candidates])

    ranked = rank_stories(
        candidates=candidates,
        candidate_embeddings=cand_embs,
        db=db,
        config=config,
        embedder=embedder,
    )

    assert len(ranked) == 2
    # Should sort by points (score) descending (high points story first)
    assert ranked[0].story.id == 2
    assert ranked[1].story.id == 1
    # Scores should be neutral (0.5)
    assert ranked[0].score == 0.5
    assert ranked[1].score == 0.5


@given(text=st.text(), min_len=st.integers(min_value=0, max_value=100))
@settings(max_examples=25)
def test_clean_text_properties(text, min_len):
    import re

    cleaned = clean_text(text, min_len=min_len)

    if cleaned != "":
        # Length constraint
        assert len(cleaned) > min_len
        # Alphanumeric density
        alnum = sum(c.isalnum() for c in cleaned)
        assert alnum / len(cleaned) >= 0.5
        # No Braille
        assert not re.search(r"[\u2800-\u28FF]", cleaned)
        # No unescaped tags
        assert not re.search(r"<[^>]+>", cleaned)


@given(
    meta=st.lists(
        st.tuples(
            st.integers(min_value=-1000, max_value=1_000_000),
            st.floats(
                min_value=-86400.0,
                max_value=10_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=25)
def test_augment_features_properties(meta):
    from pipeline import _augment_features

    n = len(meta)
    embeddings = np.random.randn(n, 384).astype(np.float32)
    scores = [item[0] for item in meta]
    ages = [item[1] for item in meta]

    features = _augment_features(embeddings, scores, ages)

    assert features.shape == (n, 385)
    assert np.allclose(features[:, :384], embeddings)
    assert np.all(features[:, 384] >= 0.0) and np.all(features[:, 384] <= 1.0)

    # When all 7 derived features are provided, shape expands to (n, 392)
    comment_counts = np.array([max(s, 0) for s in scores])
    text_lengths = np.array([abs(a) % 10000 for a in ages])
    hn_quality = comment_counts.astype(np.float32) / (np.abs(ages) + 1)
    sim_up = np.random.uniform(-1, 1, n).astype(np.float32)
    sim_down = np.random.uniform(-1, 1, n).astype(np.float32)
    closest_up = np.random.uniform(-1, 1, n).astype(np.float32)
    closest_down = np.random.uniform(-1, 1, n).astype(np.float32)

    features7 = _augment_features(
        embeddings,
        scores,
        ages,
        comment_counts=comment_counts,
        text_lengths=text_lengths,
        hn_quality=hn_quality,
        sim_to_upvoted=sim_up,
        sim_to_downvoted=sim_down,
        closest_upvoted=closest_up,
        closest_downvoted=closest_down,
    )
    assert features7.shape == (n, 392)
    assert np.allclose(features7[:, :384], embeddings)
    # All normalized cols in [0, 1]
    assert np.all(features7[:, 384:] >= 0.0) and np.all(features7[:, 384:] <= 1.0)


@given(
    score_a=st.floats(0.5, 1.0, allow_nan=False),
    score_b=st.floats(0.0, 0.49, allow_nan=False),
    eng_a=st.integers(min_value=0, max_value=500),
    eng_b=st.integers(min_value=0, max_value=2000),
)
@settings(max_examples=25)
def test_mmr_engagement_promotion_boundaries(score_a, score_b, eng_a, eng_b):
    emb = np.zeros(384, dtype=np.float32)
    emb[0] = 1.0

    story_a = Story(
        id=1, title="A", url=None, score=eng_a, comment_count=0, time=0, text_content=""
    )
    story_b = Story(
        id=2, title="B", url=None, score=eng_b, comment_count=0, time=0, text_content=""
    )

    ranked = [
        RankedStory(story=story_a, score=score_a, best_match_title=""),
        RankedStory(story=story_b, score=score_b, best_match_title=""),
    ]
    embs_map = {1: emb, 2: emb}

    filtered = mmr_filter(ranked, embs_map, threshold=0.55, limit=10)

    assert len(filtered) == 1
    selected_id = filtered[0].story.id

    if eng_b > eng_a * 2.0 + 30:
        assert selected_id == 2
    else:
        assert selected_id == 1


@given(
    feedback_actions=st.lists(
        st.sampled_from(["up", "neutral", "down"]), min_size=0, max_size=20
    ),
    cand_count=st.integers(min_value=1, max_value=10),
)
@settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=1000,
)
def test_svm_fitting_robustness(embedder, feedback_actions, cand_count):
    db = Database(":memory:")
    try:
        model_version = "all-MiniLM-L6-v2|mean|norm|256"
        for i, action in enumerate(feedback_actions):
            story = Story(
                id=1000 + i,
                title=f"Feedback Story {i}",
                url=None,
                score=np.random.randint(0, 1000),
                time=int(1600000000 + i * 100),
                text_content=f"Sample semantic content for history {i}",
            )
            db.upsert_story(story)
            db.upsert_embedding(
                story.id, model_version, np.random.randn(384).astype(np.float32)
            )
            db.upsert_feedback(story.id, action)

        candidates = []
        for i in range(cand_count):
            candidates.append(
                Story(
                    id=i,
                    title=f"Candidate Story {i}",
                    url=None,
                    score=np.random.randint(0, 500),
                    time=int(1600000000),
                    text_content=f"Sample candidate content {i}",
                )
            )

        cand_embs = np.random.randn(cand_count, 384).astype(np.float32)
        config = Config()
        ranked = rank_stories(candidates, cand_embs, db, config, embedder)

        assert len(ranked) == cand_count
        for item in ranked:
            assert 0.0 <= item.score <= 1.0
            if not feedback_actions:
                assert item.score == 0.5
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fetch_candidates_returns_tuple(tmp_path, monkeypatch):
    """fetch_candidates returns (list[Story], int), not just list."""
    from pipeline import Config, fetch_candidates
    from database import Database

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    config = Config(
        db_path=str(db_file),
        output=str(tmp_path / "index.html"),
        server_port=0,
    )

    class MockResp:
        status_code = 200

        def json(self):
            return {"hits": []}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get(self, *a, **kw):
            return MockResp()

    monkeypatch.setattr("pipeline.httpx.AsyncClient", lambda **kw: MockClient())
    result = await fetch_candidates(config, set(), set(), db)
    assert isinstance(result, tuple), f"expected tuple, got {type(result)}"
    assert len(result) == 2
    candidates, count = result
    assert isinstance(candidates, list)
    assert isinstance(count, int)
    assert count == len(candidates)


# Tests for auto-refetch of comment text on growth (fetch_candidates refetch block)
def _seed_story(
    db: Database,
    sid: int,
    *,
    comment_count: int,
    comment_count_at_fetch: int,
    age_hours: int,
    feedback_action: Literal["up", "neutral", "down"] | None = None,
) -> Story:
    """Insert a story in the live window with controlled engagement metrics."""
    from time import time as _now

    story_time = int(_now()) - (age_hours * 3600)
    story = Story(
        id=sid,
        title=f"Test story {sid}",
        url=f"https://example.com/{sid}",
        score=100,
        time=story_time,
        text_content=f"initial text content for story {sid}",
        source="hn",
        comment_count=comment_count,
        discussion_url=f"https://news.ycombinator.com/item?id={sid}",
        comment_count_at_fetch=comment_count_at_fetch,
    )
    db.upsert_story(story)
    if feedback_action is not None:
        db.upsert_feedback(sid, feedback_action)
    return story


def test_refetch_eligibility_triggers_at_30pct_growth():
    """comment_count grown from 10 to 14 (+40%) at age 1h is eligible."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    sid = 1001
    now = int(_now())
    eligible, new_count = _is_refetch_eligible(
        sid=sid,
        comment_count_at_fetch=10,
        new_comment_count=14,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is True
    assert new_count == 14


def test_refetch_eligibility_skipped_below_30pct_growth():
    """10 to 12 (+20%) is below threshold; not eligible."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1002,
        comment_count_at_fetch=10,
        new_comment_count=12,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_after_24h_age():
    """Eligible growth but age > 24h is skipped."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1003,
        comment_count_at_fetch=10,
        new_comment_count=14,
        story_time=now - (25 * 3600),
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_for_feedback_story():
    """Stories in feedback_ids are protected from refetch (training contract)."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1004,
        comment_count_at_fetch=10,
        new_comment_count=14,
        story_time=now - 3600,
        feedback_ids={1004},
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_when_baseline_zero():
    """comment_count_at_fetch == 0 means no baseline; cannot compute growth."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1005,
        comment_count_at_fetch=0,
        new_comment_count=14,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_eligibility_skipped_when_count_did_not_grow():
    """If new_comment_count <= comment_count_at_fetch, no refetch."""
    from pipeline import _is_refetch_eligible
    from time import time as _now

    now = int(_now())
    eligible, _ = _is_refetch_eligible(
        sid=1006,
        comment_count_at_fetch=10,
        new_comment_count=10,
        story_time=now - 3600,
        feedback_ids=set(),
        now_ts=now,
    )
    assert eligible is False


def test_refetch_selects_top_n_by_growth(db, embedder):
    """Only MAX_REFETCH_PER_REGEN stories are selected, prioritized by growth."""
    from pipeline import _select_refetch_ids, MAX_REFETCH_PER_REGEN
    from time import time as _now

    now = int(_now())
    candidates = []
    fresh_metadata = {}
    for i in range(MAX_REFETCH_PER_REGEN + 5):
        sid = 2000 + i
        candidates.append(
            _seed_story(
                db,
                sid,
                comment_count=100 + i,
                comment_count_at_fetch=10,
                age_hours=1,
            )
        )
        fresh_metadata[sid] = {"score": 200, "comment_count": 100 + i}
    selected = _select_refetch_ids(
        candidates=candidates,
        fresh_metadata=fresh_metadata,
        feedback_ids=set(),
        now_ts=now,
    )
    assert len(selected) == MAX_REFETCH_PER_REGEN
    # Should be the highest-growth stories (those with highest new_count, since
    # baseline is the same for all in this test). Growth ratio is monotonic in
    # new_count when baseline is fixed.
    selected_set = set(selected)
    assert max(selected_set) > min(selected_set)  # at least some spread


async def test_refetch_failure_keeps_stale_data(db, embedder):
    """If refetch_story_text fails, the original story is preserved (no DB change)."""
    import httpx
    from pipeline import refetch_story_text

    _seed_story(
        db,
        3001,
        comment_count=14,
        comment_count_at_fetch=10,
        age_hours=1,
    )
    original = db.get_story(3001)
    original_text = original.text_content

    def _handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("simulated algolia outage")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await refetch_story_text(client, db, embedder, 3001, current_count=14)

    assert result is None
    after = db.get_story(3001)
    assert after.text_content == original_text
    assert after.comment_count == original.comment_count
    assert after.comment_count_at_fetch == original.comment_count_at_fetch


async def test_refetch_updates_comment_count_at_fetch_on_success(db, embedder):
    """Successful refetch sets comment_count_at_fetch == new comment_count."""
    import httpx
    from pipeline import refetch_story_text

    sid = 3002
    _seed_story(
        db,
        sid,
        comment_count=14,
        comment_count_at_fetch=10,
        age_hours=1,
    )

    fake_item = {
        "type": "story",
        "title": "Updated Title",
        "url": "https://example.com/3002",
        "points": 200,
        "num_comments": 16,
        "created_at_i": int(__import__("time").time()),
        "story_text": "Updated self text",
        "children": [
            {"text": f"comment {i}", "score": 10 - i, "children": []} for i in range(5)
        ],
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fake_item)

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await refetch_story_text(client, db, embedder, sid, current_count=16)

    assert result is not None
    assert result.comment_count == 16
    assert result.comment_count_at_fetch == 16
    persisted = db.get_story(sid)
    assert persisted.comment_count_at_fetch == 16
    assert persisted.text_content != ""  # recomposed


async def test_run_pipeline_badge_assignment(tmp_path, monkeypatch):
    """Verify that badges are only assigned to extra stories and excluded from default ones."""
    from pipeline import Config, run_pipeline, RankedStory
    from database import Database, Story
    import numpy as np
    import time

    db_file = tmp_path / "test.db"
    db = Database(str(db_file))
    
    # 1. Create candidates list
    now = time.time()
    candidates = []
    # Default path candidates: IDs 1-7
    for i in range(1, 8):
        candidates.append(
            Story(
                id=i,
                title=f"Default Story {i}",
                url=None,
                score=100 - i,
                time=now,
                text_content=f"content {i}",
                comment_count=0,
            )
        )
    # Extra/discovery candidates:
    # ID 8: high comments -> discussion rich
    candidates.append(
        Story(
            id=8,
            title="Discussion Rich Story",
            url=None,
            score=10,
            time=now,
            text_content="content 8",
            comment_count=100,
        )
    )
    # ID 9: high score -> high engagement
    candidates.append(
        Story(
            id=9,
            title="High Engagement Story",
            url=None,
            score=500,
            time=now,
            text_content="content 9",
            comment_count=0,
        )
    )
    # ID 10: high similarity -> similar
    candidates.append(
        Story(
            id=10,
            title="Similar Story",
            url=None,
            score=10,
            time=now,
            text_content="content 10",
            comment_count=0,
        )
    )
    # ID 11: seed for novel story (we will set its max sim low)
    candidates.append(
        Story(
            id=11,
            title="Novel Story",
            url=None,
            score=6,
            time=now,
            text_content="content 11",
            comment_count=0,
        )
    )

    # Persist feedback in DB to train / simulate closest up
    db.upsert_story(Story(id=999, title="Upvoted Story", url=None, score=100, time=now, text_content="upvoted text"))
    db.upsert_feedback(999, "up")

    # 2. Config setup
    config = Config(
        db_path=str(db_file),
        output=str(tmp_path / "index.html"),
        count=10,  # 7 default path, 3 uncertain slots
    )

    # 3. Mock dependencies
    async def mock_fetch_candidates(*args, **kwargs):
        return candidates, len(candidates)
    monkeypatch.setattr("pipeline.fetch_candidates", mock_fetch_candidates)

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            pass
    monkeypatch.setattr("pipeline.Embedder", DummyEmbedder)

    def mock_get_or_compute_embeddings(stories, embedder, db_inst):
        # We return unit embeddings.
        # ID 999 has embedding [1, 0, 0...]
        # ID 10 (sleeper) has embedding [0.6, 0...] to ensure closest_up > 0.55
        # ID 11 (novel) has embedding [0.0, 0...] to ensure closest_up/closest_down <= sim_threshold
        # others have [0.3, 0...]
        embs = []
        for s in stories:
            vec = np.zeros(384, dtype=np.float32)
            if s.id == 999:
                vec[0] = 1.0
            elif s.id == 10:
                vec[0] = 0.6
            elif s.id == 11:
                vec[0] = 0.0
            else:
                vec[0] = 0.3
            embs.append(vec)
        return np.array(embs)
    monkeypatch.setattr("pipeline.get_or_compute_embeddings", mock_get_or_compute_embeddings)

    # Mock rank_stories to return a pre-sorted list (fallback score/probability behavior)
    def mock_rank_stories(candidates_list, *args, **kwargs):
        res = []
        for s in candidates_list:
            if s.id <= 7:
                score = 0.9 - s.id * 0.05
            else:
                score = 0.2
            res.append(
                RankedStory(
                    story=s,
                    score=score,
                    best_match_title="",
                    prob_down=0.1,
                    prob_neutral=0.1,
                    prob_up=0.8,
                )
            )
        return sorted(res, key=lambda x: x.score, reverse=True)
    monkeypatch.setattr("pipeline.rank_stories", mock_rank_stories)

    # Capture final stories passed to generate_dashboard
    captured_final = []
    def mock_generate_dashboard(final_list, *args, **kwargs):
        nonlocal captured_final
        captured_final = list(final_list)
    monkeypatch.setattr("pipeline.generate_dashboard", mock_generate_dashboard)

    await run_pipeline(config)

    # Assertions
    # We expect default stories (1-7) and at least some extra decorated stories
    assert len(captured_final) > 7
    
    for r in captured_final:
        if r.story.id <= 7:
            # Default path stories MUST NOT have any badges assigned
            assert not r.is_uncertain, f"Story {r.story.id} has uncertain badge"
            assert not r.is_novel, f"Story {r.story.id} has novel badge"
            assert not r.is_similar, f"Story {r.story.id} has similar badge"
            assert not r.is_discussion_rich, f"Story {r.story.id} has discussion badge"
            assert not r.is_high_engagement, f"Story {r.story.id} has engagement badge"
        else:
            # Extra discovery stories can have badges
            if r.story.id == 8:
                assert r.is_discussion_rich
            if r.story.id == 9:
                assert r.is_high_engagement
            if r.story.id == 10:
                assert r.is_similar
            if r.story.id == 11:
                assert r.is_novel
