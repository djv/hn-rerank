from __future__ import annotations

from collections.abc import Callable

import pytest

from api.models import RankResult, Story, StoryDict, StorySource


@pytest.fixture
def make_story() -> Callable[..., Story]:
    def _make_story(
        story_id: int = 1,
        *,
        title: str | None = None,
        url: str | None = None,
        score: int = 100,
        time: int = 1000,
        text_content: str | None = None,
        comments: list[str] | None = None,
        source: StorySource = "hn",
        discussion_url: str | None = None,
    ) -> Story:
        resolved_title = title if title is not None else f"Story {story_id}"
        return Story(
            id=story_id,
            title=resolved_title,
            url=url,
            score=score,
            time=time,
            discussion_url=discussion_url,
            comments=list(comments or []),
            text_content=text_content if text_content is not None else resolved_title,
            source=source,
        )

    return _make_story


@pytest.fixture
def make_stories(make_story: Callable[..., Story]) -> Callable[[int], list[Story]]:
    def _make_stories(n: int) -> list[Story]:
        return [make_story(i, title=f"S{i}", text_content=f"S{i}") for i in range(n)]

    return _make_stories


@pytest.fixture
def make_story_dict() -> Callable[..., StoryDict]:
    def _make_story_dict(
        story_id: int = 1,
        *,
        title: str | None = None,
        url: str | None = None,
        score: int = 0,
        time: int = 0,
        text_content: str | None = None,
        comments: list[str] | None = None,
        source: StorySource = "hn",
    ) -> StoryDict:
        resolved_title = title if title is not None else f"Story {story_id}"
        return StoryDict(
            id=story_id,
            title=resolved_title,
            url=url,
            score=score,
            time=time,
            discussion_url=None,
            comments=list(comments or []),
            text_content=text_content if text_content is not None else resolved_title,
            source=source,
        )

    return _make_story_dict


@pytest.fixture
def make_rank_result() -> Callable[..., RankResult]:
    def _make_rank_result(
        index: int = 0,
        *,
        hybrid_score: float = 1.0,
        best_fav_index: int = -1,
        max_sim_score: float = 0.0,
        knn_score: float = 0.0,
        max_cluster_score: float = 0.0,
        semantic_score: float = 0.0,
        hn_score: float = 0.0,
        freshness_boost: float = 0.0,
    ) -> RankResult:
        return RankResult(
            index=index,
            hybrid_score=hybrid_score,
            best_fav_index=best_fav_index,
            max_sim_score=max_sim_score,
            knn_score=knn_score,
            max_cluster_score=max_cluster_score,
            semantic_score=semantic_score,
            hn_score=hn_score,
            freshness_boost=freshness_boost,
        )

    return _make_rank_result


@pytest.fixture
def isolate_fetch_caches(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr("api.fetching.CACHE_PATH", tmp_path / "stories")
    monkeypatch.setattr("api.fetching.CANDIDATE_CACHE_PATH", tmp_path / "candidates")
    monkeypatch.setattr("api.fetching.atomic_write_json", lambda _path, _data: None)
    monkeypatch.setattr(
        "api.fetching.evict_old_cache_files", lambda *args, **kwargs: None
    )
    return tmp_path
