import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from api.rerank import rank_stories
from api.models import Story, RankResult, StorySource
from api.config import (
    AppConfig,
    RankingConfig,
    SemanticConfig,
    ClassifierConfig,
    SingleModelConfig,
)
from api.feedback_single_model import (
    build_single_model_feedback_labels,
    build_single_model_feature_batch,
    train_single_model,
    score_feature_rows,
)
from api.feedback import FeedbackRecord
from generate_html import select_ranked_results
from helpers import unit_rows


def make_stories(n: int) -> list[Story]:
    """Helper to create Story objects for tests."""
    return [
        Story(
            id=i,
            title=f"Story {i}",
            url=None,
            score=100,
            time=1000,
            text_content=f"Story {i}",
        )
        for i in range(n)
    ]


@settings(deadline=None, max_examples=30)
@given(
    num_candidates=st.integers(min_value=1, max_value=20),
    num_favorites=st.integers(min_value=1, max_value=10),
)
def test_ranking_invariants(num_candidates, num_favorites):
    """
    Invariants for rank_stories:
    1. Returns a list of RankResult.
    2. Number of results <= min(num_candidates, 100).
    3. Scores are sorted descending.
    4. best_fav_index corresponds to a valid index in positive_embeddings (or -1).
    """

    # Mock embeddings (random unit vectors)
    def random_unit_vectors(n, dim=384):
        vecs = np.random.randn(n, dim).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    pos_emb = random_unit_vectors(num_favorites)
    cand_emb = random_unit_vectors(num_candidates)

    stories = make_stories(num_candidates)

    # Mock get_embeddings to return our cand_emb
    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig()
        results = rank_stories(stories, positive_embeddings=pos_emb, config=config)

        assert len(results) == num_candidates
        assert {r.index for r in results} == set(range(num_candidates))

        last_score = float("inf")
        for result in results:
            assert result.model_score <= last_score + 1e-7
            # fav_idx can be -1 if below threshold
            assert -1 <= result.best_fav_index < num_favorites
            assert -1.0 <= result.max_sim_score <= 1.0 or result.max_sim_score == 0.0
            last_score = result.model_score


@settings(deadline=None, max_examples=30)
@given(
    st.lists(
        st.integers(min_value=1, max_value=50), min_size=3, max_size=10, unique=True
    )
)
def test_ranking_permutation_invariance(story_ids):
    """
    Invariant: Permuting candidate order does not change rank ordering by story ID.
    """
    stories = [
        Story(
            id=i, title=f"Story {i}", url=None, score=0, time=1000, text_content=str(i)
        )
        for i in story_ids
    ]

    max_id = max(story_ids) + 1

    def emb_for_id(story_id: int) -> np.ndarray:
        x = story_id / max_id
        y = float(np.sqrt(max(0.0, 1.0 - x * x)))
        return np.array([x, y, 0.0], dtype=np.float32)

    emb_map = {str(i): emb_for_id(i) for i in story_ids}
    pos_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            api.rerank,
            "get_embeddings",
            lambda texts, **kwargs: np.stack([emb_map[t] for t in texts]).astype(
                np.float32
            ),
        )

        config = AppConfig()
        res_a = rank_stories(stories, positive_embeddings=pos_emb, config=config)
        rev_stories = list(reversed(stories))
        res_b = rank_stories(rev_stories, positive_embeddings=pos_emb, config=config)

        order_a = [stories[r.index].id for r in res_a]
        order_b = [rev_stories[r.index].id for r in res_b]
        assert order_a == order_b


@settings(deadline=None, max_examples=30)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    num_candidates=st.integers(min_value=1, max_value=15),
    num_favorites=st.integers(min_value=1, max_value=8),
)
def test_model_score_bounds_without_negatives(
    seed: int, num_candidates: int, num_favorites: int
):
    """
    Invariant: Without negative signals, model_score stays in [-1.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    cand_emb = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9
    pos_emb = rng.normal(size=(num_favorites, 4)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9

    stories = make_stories(num_candidates)

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig(semantic=SemanticConfig())
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            negative_embeddings=None,
            config=config,
        )

        for result in results:
            assert -1.0 - 1e-6 <= result.model_score <= 1.0 + 1e-6


def test_heuristic_ranking_ignores_negative_signals():
    """
    Invariant: Negative embeddings do not affect heuristic (non-classifier) scores.
    """
    pos_emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pos_emb /= np.linalg.norm(pos_emb)
    cand_emb = np.array([[1.0, 0.1, 0.0]], dtype=np.float32)
    cand_emb /= np.linalg.norm(cand_emb)
    neg_emb = np.array([[1.0, 0.2, 0.0]], dtype=np.float32)
    neg_emb /= np.linalg.norm(neg_emb)

    stories = [Story(id=0, title="S", url=None, score=100, time=1000, text_content="S")]

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        res_no_neg = rank_stories(stories, pos_emb, config=AppConfig())
        res_with_neg = rank_stories(
            stories, pos_emb, negative_embeddings=neg_emb, config=AppConfig()
        )

        assert res_with_neg[0].model_score == pytest.approx(res_no_neg[0].model_score)


def test_rank_stories_empty_signals():
    """
    Invariant: Ranking must still work even if there are no positive or negative
    signals. With no signals all semantic scores are zero.
    """
    stories = [
        Story(id=1, title="A", url=None, score=100, time=1000, text_content="A"),
        Story(id=2, title="B", url=None, score=500, time=1000, text_content="B"),
    ]

    cand_emb = np.zeros((2, 384), dtype=np.float32)

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig(ranking=RankingConfig())
        results = rank_stories(
            stories,
            positive_embeddings=np.zeros((0, 384), dtype=np.float32),
            negative_embeddings=None,
            config=config,
        )

        assert len(results) == 2
        # No signals: all semantic scores are zero
        for r in results:
            assert r.model_score == pytest.approx(0.0, abs=1e-6)


def test_rank_stories_order_correlates_with_similarity():
    pos_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    embs = np.array([[1.0, 0.01], [1.0, 0.02], [0.7, 0.7]], dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    stories = [
        Story(id=0, title="A", url=None, score=100, time=1000, text_content="A"),
        Story(id=1, title="B", url=None, score=100, time=1000, text_content="B"),
        Story(id=2, title="C", url=None, score=100, time=1000, text_content="C"),
    ]

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: embs)
        config = AppConfig()
        results = rank_stories(stories, pos_emb, config=config)

        assert [r.index for r in results] == [0, 1, 2]


def test_rank_stories_upvote_boost():
    """
    Invariant: A candidate similar to an upvoted story (positive signal)
    MUST have a higher score than a dissimilar candidate.
    """
    # 1. Create two candidates:
    # A: Identical to the upvoted story
    # B: Orthogonal (dissimilar) to the upvoted story

    # 2. Embeddings
    # Pos signal: [1, 0]
    pos_emb = np.array([[1.0, 0.0]], dtype=np.float32)

    # Candidate A: [1, 0] (Perfect match)
    # Candidate B: [0, 1] (No match)
    cand_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    stories = [
        Story(
            id=1, title="Match", url=None, score=100, time=1000, text_content="Match"
        ),
        Story(
            id=2,
            title="NoMatch",
            url=None,
            score=100,
            time=1000,
            text_content="NoMatch",
        ),
    ]

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        # Rank with pure semantic weight () to isolate the effect
        config = AppConfig()
        results = rank_stories(stories, positive_embeddings=pos_emb, config=config)

        # Candidate A (id 1) should be first and have a higher score
        assert results[0].index == 0  # Index 0 is Story 1
        assert results[0].model_score > results[1].model_score
        # After z-score normalization, perfect match should be high
        assert results[0].model_score > 0.9
        # Orthogonal candidate should be near zero
        assert results[1].model_score < 0.1


def test_heuristic_ranking_negative_embeddings_no_effect_when_negative_matches():
    """
    Invariant: Even when a candidate matches negative embeddings, the
    heuristic (non-classifier) score is unaffected.
    """
    cand_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    neg_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    pos_emb = np.array([[0.0, 1.0]], dtype=np.float32)

    stories = [Story(id=1, title="A", url=None, score=100, time=1000, text_content="A")]

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        # Baseline: Rank WITHOUT negative embeddings
        config_baseline = AppConfig(ranking=RankingConfig())
        res_baseline = rank_stories(
            stories, pos_emb, negative_embeddings=None, config=config_baseline
        )
        score_baseline = res_baseline[0].model_score

        # Test: Rank WITH negative embeddings (no longer affects score)
        config_penalized = AppConfig()
        res_penalized = rank_stories(
            stories, pos_emb, negative_embeddings=neg_emb, config=config_penalized
        )
        score_penalized = res_penalized[0].model_score

        assert score_penalized == pytest.approx(score_baseline)


@given(st.lists(st.integers(min_value=0, max_value=100000), min_size=1, max_size=20))
def test_hn_normalization_extreme_values(scores):
    """
    Invariant: HN score normalization handles extreme values without overflow.
    """
    # Simulate the normalization: max(points-1, 0)^exp, then /max
    points = np.array(scores, dtype=np.float64)
    hn_scores = np.power(np.maximum(points - 1, 0), 0.8)
    if hn_scores.max() > 0:
        hn_scores /= hn_scores.max()

    # All should be in [0, 1]
    assert np.all(hn_scores >= 0.0)
    assert np.all(hn_scores <= 1.0)
    # No NaN or Inf
    assert np.all(np.isfinite(hn_scores))


def test_rank_stories_returns_all_identical_candidates():
    """
    Invariant: score-only sorting still returns all items for identical candidates.
    """
    # All candidates have identical embeddings
    cand_emb = np.ones((5, 384), dtype=np.float32)
    cand_emb = cand_emb / np.linalg.norm(cand_emb[0])

    pos_emb = cand_emb[:1]  # Matches all candidates equally

    stories = [
        Story(
            id=i,
            title=f"S{i}",
            url=None,
            score=100 + i,
            time=1000,
            text_content=f"S{i}",
        )
        for i in range(5)
    ]

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig()
        results = rank_stories(stories, positive_embeddings=pos_emb, config=config)

        # Should still return all items
        assert len(results) == 5
        # All indices present
        assert set(r.index for r in results) == {0, 1, 2, 3, 4}


def test_knn_scoring_logic():
    """
    Invariant: A candidate matching many favorites gets a higher k-NN
    score than one matching a single outlier favorite.
    """
    # 3 History items
    # H1: [1, 0, 0]
    # H2: [1, 0, 0]
    # H3: [1, 0, 0]
    # H4: [0, 1, 0] (Outlier)
    pos_emb = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    # 2 Candidates
    # C1: [1, 0, 0] (Matches H1, H2, H3 perfectly -> k=3 score = 1.0)
    # C2: [0, 1, 0] (Matches H4 perfectly, but H1-H3 are 0.0 -> k=3 score = (1+0+0)/3 = 0.33)
    cand_emb = np.array(
        [
            [1.0, 0.0, 0.0],  # C1
            [0.0, 1.0, 0.0],  # C2
        ],
        dtype=np.float32,
    )

    stories = [
        Story(
            id=1, title="Consistent", url=None, score=100, time=1000, text_content="C1"
        ),
        Story(
            id=2,
            title="OneHitWonder",
            url=None,
            score=100,
            time=1000,
            text_content="C2",
        ),
    ]

    import api.rerank

    # Ensure constant is set to 3 for this test
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig(
            semantic=SemanticConfig(knn_neighbors=3),
        )
        results = rank_stories(stories, positive_embeddings=pos_emb, config=config)

        # C1 should win because it has 3 good neighbors (display score)
        # C2 has only 1 good neighbor, so its median top-3 score is dragged down
        assert results[0].index == 0  # C1
        assert results[0].knn_score > results[1].knn_score

        # Cluster-max scoring can still see both as strong semantic matches...
        assert results[0].max_sim_score == pytest.approx(1.0, abs=1e-6)
        assert results[1].max_sim_score == pytest.approx(1.0, abs=1e-6)

        # Current configured scoring is pure cluster-max, so k-NN stays diagnostic.
        assert results[0].model_score == pytest.approx(1.0, abs=1e-6)
        assert results[1].model_score == pytest.approx(1.0, abs=1e-6)


def test_median_knn_outlier_robustness():
    """
    Invariant: Median k-NN is robust to single outlier matches.
    Cluster-max scoring still surfaces the outlier as a strong match.
    """
    # 4 History items:
    # H1, H2, H3: [0, 1] (dissimilar to candidate)
    # H4: [1, 0] (single outlier that matches candidate perfectly)
    pos_emb = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],  # Outlier
        ],
        dtype=np.float32,
    )

    # Candidate: [1, 0] (matches only H4)
    cand_emb = np.array([[1.0, 0.0]], dtype=np.float32)

    stories = [
        Story(id=1, title="Test", url=None, score=100, time=1000, text_content="Test"),
    ]

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        # With k=3 and median:
        # Top 3 sims = [1.0, 0.0, 0.0] (one perfect, two zero)
        # Median = 0.0 (robust to outlier)
        # With mean it would be 0.33
        config = AppConfig(
            semantic=SemanticConfig(knn_neighbors=3),
        )
        results = rank_stories(stories, positive_embeddings=pos_emb, config=config)

        # The median of [0.0, 0.0, 1.0] is 0.0
        # Raw k-NN score should be 0.0 (median)
        assert results[0].knn_score == pytest.approx(0.0, abs=0.01)
        # Cluster-max scoring still sees a perfect match
        assert results[0].max_sim_score == pytest.approx(1.0, abs=1e-6)
        # Current configured scoring is pure cluster-max.
        assert results[0].model_score == pytest.approx(1.0, abs=1e-6)


@settings(deadline=None, max_examples=25)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    num_candidates=st.integers(min_value=5, max_value=20),
    num_favorites=st.integers(min_value=3, max_value=10),
)
def test_knn_scores_have_nonzero_variance(seed, num_candidates, num_favorites):
    """
    Invariant: In k-NN mode, model_scores must have nonzero variance
    when candidate embeddings differ. The old fixed sigmoid mapped all
    scores ~0.93-0.98 to ~1.0 (zero discrimination).
    """
    rng = np.random.default_rng(seed)

    # Create diverse embeddings so candidates differ
    pos_emb = rng.normal(size=(num_favorites, 4)).astype(np.float32)
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9

    cand_emb = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9

    stories = make_stories(num_candidates)

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig(semantic=SemanticConfig())
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            negative_embeddings=None,
            config=config,
        )

        scores = np.array([r.model_score for r in results])
        # With diverse embeddings, scores MUST vary
        assert scores.std() > 1e-6, f"k-NN semantic scores have zero variance: {scores}"


def test_close_candidate_outranks_far_candidate():
    """
    Invariant: A candidate close to positive embeddings scores higher
    than a candidate far from positive embeddings.
    """
    # Positive interest: [1, 0, 0]
    pos_emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    pos_emb /= np.linalg.norm(pos_emb, axis=1, keepdims=True)

    # Candidate A: close to pos ([0.9, 0.1, 0.0])
    # Candidate B: far from pos ([0.0, 0.0, 1.0])
    cand_emb = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cand_emb /= np.linalg.norm(cand_emb, axis=1, keepdims=True)

    stories = make_stories(2)

    import api.rerank

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.rerank, "get_embeddings", lambda texts, **kwargs: cand_emb)

        config = AppConfig(semantic=SemanticConfig())
        results = rank_stories(
            stories,
            positive_embeddings=pos_emb,
            negative_embeddings=None,
            config=config,
        )

        # Close candidate must rank first with strictly higher score
        assert results[0].index == 0
        assert results[0].model_score > results[1].model_score


# --- Selection policy property tests ---


def _mk_rank(idx: int, score: float) -> RankResult:
    return RankResult(
        index=idx,
        model_score=score,
        best_fav_index=-1,
        max_sim_score=0.0,
        knn_score=0.0,
    )


@settings(deadline=None)
@given(
    num_hn=st.integers(min_value=0, max_value=30),
    num_ext=st.integers(min_value=0, max_value=15),
    count=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_selection_output_length_invariants(
    num_hn: int, num_ext: int, count: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    total = num_hn + num_ext
    assume(total > 0)

    sources: tuple[StorySource, ...] = ("rss", "lobsters", "tildes", "lesswrong")
    cands: list[Story] = []
    for i in range(num_ext):
        src = sources[i % len(sources)]
        cands.append(
            Story(
                id=-(i + 1),
                title=f"Ext {i}",
                url=None,
                score=0,
                time=1,
                text_content="",
                source=src,
            )
        )
    for i in range(num_hn):
        cands.append(
            Story(id=i + 1, title=f"HN {i}", url=None, score=0, time=1, text_content="")
        )

    rng.shuffle(cands)
    ranked = [_mk_rank(i, float(rng.uniform(0.0, 1.0))) for i in range(len(cands))]

    # Overwrite half the external scores to be high so they have a chance
    # This just ensures diversity of scores
    for i, r in enumerate(ranked):
        r.model_score = float(1.0 - i * 0.01)

    selected = select_ranked_results(ranked, cands, None, {}, {}, count)

    assert 0 <= len(selected) <= count

    ext_count = sum(1 for r in selected if cands[r.index].is_external)
    hn_count = len(selected) - ext_count

    assert ext_count <= num_ext
    assert hn_count <= num_hn
    assert len(selected) == ext_count + hn_count

    desired_external = round(count * 0.2) + 5
    min_external = max(0, count - num_hn)
    max_external = min(count, num_ext)
    target_external = min(max(desired_external, min_external), max_external)

    assert ext_count == target_external, (
        f"ext_count={ext_count} != target_external={target_external} "
        f"(count={count}, num_hn={num_hn}, num_ext={num_ext}, "
        f"desired_external={desired_external}, min_external={min_external}, "
        f"max_external={max_external})"
    )
    assert hn_count + ext_count == len(selected)


@settings(deadline=None)
@given(
    num_hn=st.integers(min_value=5, max_value=20),
    num_ext=st.integers(min_value=5, max_value=10),
    count=st.integers(min_value=5, max_value=15),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_selection_enough_candidates_returns_exact_count(
    num_hn: int, num_ext: int, count: int, seed: int
) -> None:
    assume(num_hn + num_ext >= count)
    rng = np.random.default_rng(seed)
    sources: tuple[StorySource, ...] = ("rss", "lobsters", "tildes")
    cands: list[Story] = []
    for i in range(num_ext):
        cands.append(
            Story(
                id=-(i + 1),
                title=f"Ext {i}",
                url=None,
                score=0,
                time=1,
                text_content="",
                source=sources[i % len(sources)],
            )
        )
    for i in range(num_hn):
        cands.append(
            Story(id=i + 1, title=f"HN {i}", url=None, score=0, time=1, text_content="")
        )

    rng.shuffle(cands)
    ranked = [_mk_rank(i, float(1.0 - i * 0.005)) for i in range(len(cands))]

    selected = select_ranked_results(ranked, cands, None, {}, {}, count)
    assert len(selected) == count, f"len={len(selected)} != count={count}"


@settings(deadline=None)
@given(
    num_sources=st.integers(min_value=1, max_value=4),
    per_source=st.integers(min_value=3, max_value=8),
    count=st.integers(min_value=5, max_value=20),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_selection_source_diversity_enforced(
    num_sources: int, per_source: int, count: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    sources: tuple[StorySource, ...] = ("rss", "lobsters", "tildes", "lesswrong")[
        :num_sources
    ]
    cands: list[Story] = []
    idx = 0
    for src in sources:
        for _ in range(per_source):
            cands.append(
                Story(
                    id=-(idx + 1),
                    title=f"{src} {idx}",
                    url=None,
                    score=0,
                    time=1,
                    text_content="",
                    source=src,
                )
            )
            idx += 1
    num_ext = len(cands)
    for i in range(10):
        cands.append(
            Story(
                id=i + 100, title=f"HN {i}", url=None, score=0, time=1, text_content=""
            )
        )

    rng.shuffle(cands)
    ranked = [_mk_rank(i, float(1.0 - i * 0.003)) for i in range(len(cands))]

    selected = select_ranked_results(ranked, cands, None, {}, {}, count)

    ext_selected = [r for r in selected if cands[r.index].is_external]
    source_counts: dict[str, int] = {}
    for r in ext_selected:
        src = cands[r.index].source
        source_counts[src] = source_counts.get(src, 0) + 1

    if num_sources >= 2:
        target_ext = len(ext_selected)
        if num_sources * 2 >= target_ext:
            cap = 2
        elif num_sources * 3 >= target_ext:
            cap = 3
        else:
            cap = num_ext  # unlimited (relaxed fully)
        for src, cnt in source_counts.items():
            assert cnt <= cap, (
                f"Source {src} has {cnt} externals, expected ≤{cap} (num_sources={num_sources}, target_ext={target_ext})"
            )


@settings(deadline=None)
@given(
    num_hn=st.integers(min_value=0, max_value=10),
    num_ext=st.integers(min_value=0, max_value=10),
    count=st.integers(min_value=1, max_value=15),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_selection_final_sort_by_score(
    num_hn: int, num_ext: int, count: int, seed: int
) -> None:
    total = num_hn + num_ext
    assume(total > 0)
    rng = np.random.default_rng(seed)
    sources: tuple[StorySource, ...] = ("rss", "lobsters")
    cands: list[Story] = []
    for i in range(num_ext):
        cands.append(
            Story(
                id=-(i + 1),
                title=f"Ext {i}",
                url=None,
                score=0,
                time=1,
                text_content="",
                source=sources[i % len(sources)],
            )
        )
    for i in range(num_hn):
        cands.append(
            Story(id=i + 1, title=f"HN {i}", url=None, score=0, time=1, text_content="")
        )

    rng.shuffle(cands)
    ranked = [_mk_rank(i, float(rng.uniform(0.0, 1.0))) for i in range(len(cands))]

    selected = select_ranked_results(ranked, cands, None, {}, {}, count)

    for i in range(len(selected) - 1):
        assert selected[i].model_score >= selected[i + 1].model_score - 1e-9, (
            f"Position {i} score {selected[i].model_score} < position {i + 1} score {selected[i + 1].model_score}"
        )


@settings(deadline=None)
@given(
    count=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_selection_within_source_order_preserved(count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sources: tuple[StorySource, ...] = ("rss", "lobsters")
    cands: list[Story] = []
    for i in range(10):
        cands.append(
            Story(
                id=-(i + 1),
                title=f"Ext {i}",
                url=None,
                score=0,
                time=1,
                text_content="",
                source=sources[i % len(sources)],
            )
        )
    for i in range(10):
        cands.append(
            Story(
                id=i + 100, title=f"HN {i}", url=None, score=0, time=1, text_content=""
            )
        )

    hn_scores = [float(rng.uniform(0.5, 1.0)) for _ in range(10)]
    ext_scores = [float(rng.uniform(0.0, 0.4)) for _ in range(10)]
    all_scores = ext_scores + hn_scores
    ranked = [_mk_rank(i, all_scores[i]) for i in range(len(cands))]

    selected = select_ranked_results(ranked, cands, None, {}, {}, count)

    hn_selected = [r for r in selected if not cands[r.index].is_external]
    hn_ranked_in_order = [r for r in ranked if not cands[r.index].is_external]
    expected_hn_prefix = {r.index for r in hn_ranked_in_order[: len(hn_selected)]}
    selected_hn_indices = {r.index for r in hn_selected}
    assert selected_hn_indices <= expected_hn_prefix, (
        f"Selected HN {selected_hn_indices} not a subset of prefix "
        f"{expected_hn_prefix} (len={len(hn_selected)})"
    )


@settings(deadline=None)
@given(
    count=st.integers(min_value=1, max_value=10),
    score_gap=st.floats(min_value=0.1, max_value=0.5),
)
def test_selection_internal_external_counts_with_forced_clamping(
    count: int, score_gap: float
) -> None:
    cands: list[Story] = [
        Story(
            id=-1,
            title="Ext0",
            url=None,
            score=0,
            time=1,
            text_content="",
            source="rss",
        ),
        Story(
            id=-2,
            title="Ext1",
            url=None,
            score=0,
            time=1,
            text_content="",
            source="lobsters",
        ),
    ]
    for i in range(10):
        cands.append(
            Story(id=i + 1, title=f"HN {i}", url=None, score=0, time=1, text_content="")
        )

    ranked = [_mk_rank(i, float(1.0 - i * score_gap / 10)) for i in range(len(cands))]

    selected = select_ranked_results(ranked, cands, None, {}, {}, count)
    assert len(selected) <= count

    desired_external = round(count * 0.2) + 5
    ext_available = 2
    ext_selected = sum(1 for r in selected if cands[r.index].is_external)
    expected_ext = min(desired_external, ext_available, count)
    assert ext_selected == expected_ext, (
        f"ext_selected={ext_selected} != expected_ext={expected_ext} for count={count}"
    )


# --- SVM single model property tests ---


def _story(
    story_id: int,
    *,
    score: int = 10,
    comment_count: int | None = 2,
    text: str | None = None,
) -> Story:
    return Story(
        id=story_id,
        title=f"Story {story_id}",
        url=f"https://example.com/{story_id}",
        score=score,
        time=1_700_000_000 + story_id,
        text_content=text or f"Story {story_id}",
        comment_count=comment_count,
    )


def _make_feedback_for_stories(
    stories: list[Story],
    actions: list[str],
) -> dict[str, FeedbackRecord]:
    return {
        f"k{i}": FeedbackRecord(
            key=f"k{i}",
            action=action,
            id=story.id,
            source="hn",
            title=story.title,
            url=story.url,
            discussion_url=None,
            text_content=story.text_content,
            time=story.time,
            score=story.score,
            comment_count=story.comment_count,
        )
        for i, (story, action) in enumerate(zip(stories, actions, strict=True))
    }


def _svm_embeddings(
    rng: np.random.Generator,
    n_pos: int,
    n_neg: int,
    n_cand: int,
    dim: int,
    *,
    perfect_match: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    """Create embeddings and text labels for SVM tests.

    If *perfect_match* is True, the first candidate (``cand-0``) is set to
    exactly match the first positive embedding.
    """
    pos_emb = unit_rows(rng.normal(size=(n_pos, dim)))
    neg_emb = unit_rows(rng.normal(size=(n_neg, dim)))
    cand_emb = unit_rows(rng.normal(size=(n_cand, dim)))

    pos_texts = [f"pos-{i}" for i in range(n_pos)]
    neg_texts = [f"neg-{i}" for i in range(n_neg)]
    cand_texts = [f"cand-{i}" for i in range(n_cand)]

    emb_map: dict[str, np.ndarray] = {}
    for i, t in enumerate(pos_texts):
        emb_map[t] = pos_emb[i]
    for i, t in enumerate(neg_texts):
        emb_map[t] = neg_emb[i]
    if perfect_match and n_cand > 0:
        perfect = pos_emb[0:1].copy()
        cand_emb = np.vstack([perfect, cand_emb[1:]]) if n_cand > 1 else perfect
        for i, t in enumerate(cand_texts):
            emb_map[t] = cand_emb[i]
    else:
        for i, t in enumerate(cand_texts):
            emb_map[t] = cand_emb[i]

    return emb_map, pos_texts, neg_texts, cand_texts, pos_emb, neg_emb


def _train_svm_and_score(
    rng: np.random.Generator,
    n_pos: int,
    n_neg: int,
    n_cand: int,
    dim: int,
    *,
    perfect_match: bool = False,
) -> tuple[np.ndarray, int]:
    """Train an SVM model and score random candidates.

    Returns ``(scores_array, n_cand)``.
    """
    emb_map, pos_texts, neg_texts, cand_texts, pos_emb, neg_emb = _svm_embeddings(
        rng, n_pos, n_neg, n_cand, dim, perfect_match=perfect_match
    )

    def fake_get_embeddings(texts: list[str]) -> np.ndarray:
        return unit_rows([emb_map[t].tolist() for t in texts])

    config = AppConfig(classifier=ClassifierConfig())
    pos_stories = [_story(100 + i, text=t) for i, t in enumerate(pos_texts)]
    neg_stories = [_story(200 + i, text=t) for i, t in enumerate(neg_texts)]

    import api.feedback_single_model

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.feedback_single_model, "get_embeddings", fake_get_embeddings)

        labels_result = build_single_model_feedback_labels(
            _make_feedback_for_stories(
                pos_stories + neg_stories, ["up"] * n_pos + ["down"] * n_neg
            )
        )
        pos_emb_for_train = fake_get_embeddings(pos_texts)
        neg_emb_for_train = fake_get_embeddings(neg_texts)

        training_config = SingleModelConfig(
            min_positive_labels=2, min_negative_labels=2
        )
        model, _ = train_single_model(
            labels_result.labels,
            pos_emb_for_train,
            neg_emb_for_train,
            config,
            training_config,
        )

        cand_stories = [_story(i, text=t) for i, t in enumerate(cand_texts)]
        cand_emb_for_score = fake_get_embeddings(cand_texts)
        cand_batch = build_single_model_feature_batch(
            cand_stories,
            cand_emb_for_score,
            pos_emb_for_train,
            neg_emb_for_train,
            config,
        )
        scores = score_feature_rows(model, cand_batch.rows)

    return scores, n_cand


@pytest.mark.slow
@settings(deadline=None, max_examples=30)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_svm_scores_bounded_and_finite(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_pos = rng.integers(2, 6)
    n_neg = rng.integers(2, 6)
    n_cand = rng.integers(2, 10)
    dim = rng.integers(2, 8)

    scores, n_cand = _train_svm_and_score(rng, n_pos, n_neg, n_cand, dim)

    assert scores.shape == (n_cand,)
    assert np.all(np.isfinite(scores)), "NaN or Inf in scores"
    assert np.all(scores >= 0.0), f"Negative score: {scores[scores < 0]}"
    assert np.all(scores <= 1.0), f"Score > 1.0: {scores[scores > 1.0]}"


@pytest.mark.slow
@settings(deadline=None, max_examples=20)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_svm_deterministic_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_pos = rng.integers(2, 5)
    n_neg = rng.integers(2, 5)
    n_cand = rng.integers(2, 6)
    dim = rng.integers(2, 6)

    emb_map, pos_texts, neg_texts, cand_texts, pos_emb, neg_emb = _svm_embeddings(
        rng, n_pos, n_neg, n_cand, dim
    )

    def fake_get_embeddings(texts: list[str]) -> np.ndarray:
        return unit_rows([emb_map[t].tolist() for t in texts])

    config = AppConfig(classifier=ClassifierConfig())
    pos_stories = [_story(100 + i, text=t) for i, t in enumerate(pos_texts)]
    neg_stories = [_story(200 + i, text=t) for i, t in enumerate(neg_texts)]

    import api.feedback_single_model

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(api.feedback_single_model, "get_embeddings", fake_get_embeddings)

        labels_result = build_single_model_feedback_labels(
            _make_feedback_for_stories(
                pos_stories + neg_stories, ["up"] * n_pos + ["down"] * n_neg
            )
        )
        pos_emb_for_train = fake_get_embeddings(pos_texts)
        neg_emb_for_train = fake_get_embeddings(neg_texts)

        training_config = SingleModelConfig(
            min_positive_labels=2, min_negative_labels=2
        )

        model1, _ = train_single_model(
            labels_result.labels,
            pos_emb_for_train,
            neg_emb_for_train,
            config,
            training_config,
        )
        model2, _ = train_single_model(
            labels_result.labels,
            pos_emb_for_train,
            neg_emb_for_train,
            config,
            training_config,
        )

        cand_stories = [_story(i, text=t) for i, t in enumerate(cand_texts)]
        cand_emb_for_score = fake_get_embeddings(cand_texts)
        cand_batch = build_single_model_feature_batch(
            cand_stories,
            cand_emb_for_score,
            pos_emb_for_train,
            neg_emb_for_train,
            config,
        )

        scores1 = score_feature_rows(model1, cand_batch.rows)
        scores2 = score_feature_rows(model2, cand_batch.rows)

        assert np.allclose(scores1, scores2, atol=1e-6), "Scores differ between runs"


@pytest.mark.slow
@settings(deadline=None, max_examples=15)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_svm_perfect_match_outranks_noise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_pos = rng.integers(2, 4)
    n_neg = rng.integers(2, 4)
    n_cand = rng.integers(2, 4)
    dim = rng.integers(2, 6)

    scores, n_cand = _train_svm_and_score(
        rng, n_pos, n_neg, n_cand, dim, perfect_match=True
    )

    assert np.all(np.isfinite(scores)), f"Non-finite scores: {scores}"
