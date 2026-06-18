import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
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
def db(tmp_path):
    db_file = tmp_path / "test.db"
    db_instance = Database(str(db_file))
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
        Story(id=1, title="Machine Learning", url=None, score=0, time=0, text_content="AI and machine learning tutorial."),
        Story(id=2, title="Cooking recipes", url=None, score=0, time=0, text_content="How to bake bread and cook pasta."),
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
        db.upsert_story(Story(id=100 + i, title=f"Up story {i}", url=None, score=0, time=0, text_content="Deep learning AI research artificial intelligence"))
        db.upsert_story(Story(id=200 + i, title=f"Down story {i}", url=None, score=0, time=0, text_content="Baking sourdough bread cake cookie recipe kitchen"))
        db.upsert_feedback(100 + i, "up")
        db.upsert_feedback(200 + i, "down")

    candidates = [
        Story(id=1, title="AI systems", url=None, score=0, time=0, text_content="Training large language models and neural networks."),
        Story(id=2, title="Cake recipe", url=None, score=0, time=0, text_content="Delicious chocolate chip cake baking guide."),
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


def test_rank_single_feedback_vote(db, embedder):
    config = Config()
    # Add exactly one feedback record (upvote) to DB
    db.upsert_story(Story(id=100, title="Only Up", url=None, score=0, time=0, text_content="Only upvote text content"))
    db.upsert_feedback(100, "up")
    
    candidates = [
        Story(id=1, title="Match Story", url=None, score=0, time=0, text_content="Only upvote text content similar"),
        Story(id=2, title="Other Story", url=None, score=0, time=0, text_content="Completely unrelated recipes for cake cooking"),
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
    # Should rank the matching story higher
    assert ranked[0].story.id == 1
    assert ranked[0].score > ranked[1].score


def test_rank_no_feedback_frontpage_sort(db, embedder):
    config = Config()
    # Absolutely no feedback in DB
    candidates = [
        Story(id=1, title="Low points story", url=None, score=10, time=0, text_content=""),
        Story(id=2, title="High points story", url=None, score=500, time=0, text_content=""),
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


@given(
    text=st.text(),
    min_len=st.integers(min_value=0, max_value=100)
)
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
            st.floats(min_value=-86400.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False)
        ),
        min_size=1,
        max_size=50
    )
)
def test_augment_features_properties(meta):
    from pipeline import _augment_features
    n = len(meta)
    embeddings = np.random.randn(n, 384).astype(np.float32)
    scores = [item[0] for item in meta]
    ages = [item[1] for item in meta]
    
    features = _augment_features(embeddings, scores, ages)
    
    assert features.shape == (n, 386)
    assert np.allclose(features[:, :384], embeddings)
    assert np.all(features[:, 384] >= 0.0) and np.all(features[:, 384] <= 1.0)
    assert np.all(features[:, 385] >= 0.0) and np.all(features[:, 385] <= 1.0)


@given(
    score_a=st.floats(0.5, 1.0, allow_nan=False),
    score_b=st.floats(0.0, 0.49, allow_nan=False),
    eng_a=st.integers(min_value=0, max_value=500),
    eng_b=st.integers(min_value=0, max_value=2000),
)
def test_mmr_engagement_promotion_boundaries(score_a, score_b, eng_a, eng_b):
    emb = np.zeros(384, dtype=np.float32)
    emb[0] = 1.0
    
    story_a = Story(id=1, title="A", url=None, score=eng_a, comment_count=0, time=0, text_content="")
    story_b = Story(id=2, title="B", url=None, score=eng_b, comment_count=0, time=0, text_content="")
    
    ranked = [
        RankedStory(story=story_a, score=score_a, best_match_title=""),
        RankedStory(story=story_b, score=score_b, best_match_title="")
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
    feedback_actions=st.lists(st.sampled_from(["up", "neutral", "down"]), min_size=0, max_size=20),
    cand_count=st.integers(min_value=1, max_value=10)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_svm_fitting_robustness(tmp_path, embedder, feedback_actions, cand_count):
    import uuid
    db_file = tmp_path / f"test_svm_prop_{uuid.uuid4().hex}.db"
    db = Database(str(db_file))
    try:
        for i, action in enumerate(feedback_actions):
            story = Story(
                id=1000 + i,
                title=f"Feedback Story {i}",
                url=None,
                score=np.random.randint(0, 1000),
                time=int(1600000000 + i * 100),
                text_content=f"Sample semantic content for history {i}"
            )
            db.upsert_story(story)
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
                    text_content=f"Sample candidate content {i}"
                )
            )
            
        cand_embs = embedder.encode([s.text_content for s in candidates])
        config = Config()
        ranked = rank_stories(candidates, cand_embs, db, config, embedder)
        
        assert len(ranked) == cand_count
        for item in ranked:
            assert 0.0 <= item.score <= 1.0
            if not feedback_actions:
                assert item.score == 0.5
    finally:
        db.close()
        try:
            db_file.unlink()
        except OSError:
            pass

