from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.query_hn_vector_archive import (
    ArchiveHit,
    archive_parquet_paths,
    load_cached_user_ids,
    load_profile_cache,
    metadata_comment_count,
    metadata_url,
    save_profile_cache,
    top_archive_hits,
    vector_dot,
)


def test_load_cached_user_ids_reads_nested_cache_shape(tmp_path) -> None:
    cache_dir = tmp_path / "user"
    cache_dir.mkdir()
    (cache_dir / "alice.json").write_text(
        json.dumps(
            {
                "ts": 123,
                "ids": {
                    "pos": [1, "2", "bad"],
                    "upvoted": [3],
                    "favorites": [4],
                    "hidden": [5],
                },
            }
        )
    )

    ids = load_cached_user_ids("alice", cache_dir)

    assert ids["pos"] == {1, 2}
    assert ids["upvoted"] == {3}
    assert ids["favorites"] == {4}
    assert ids["hidden"] == {5}


def test_vector_dot_normalizes_archive_vector() -> None:
    profile = np.array([1.0, 0.0], dtype=np.float32)

    assert vector_dot(profile, [2.0, 0.0]) == 1.0
    assert vector_dot(profile, [0.0, 2.0]) == 0.0


def test_metadata_url_extracts_optional_url() -> None:
    assert metadata_url('{"url":"https://example.com/post"}') == "https://example.com/post"
    assert metadata_url('{"url":""}') is None
    assert metadata_url("not-json") is None


def test_metadata_comment_count_extracts_descendants() -> None:
    assert metadata_comment_count('{"descendants":12}') == 12
    assert metadata_comment_count('{"descendants":"7"}') == 7
    assert metadata_comment_count('{"descendants":""}') == 0
    assert metadata_comment_count("not-json") == 0


def test_top_archive_hits_keeps_highest_similarity() -> None:
    profile = np.array([1.0, 0.0], dtype=np.float32)
    rows = [
        (1, "Low", "2024-01-01", 10, "a", '{"url":"https://low.test","descendants":4}', "low text", [0.0, 1.0]),
        (2, "High", "2024-01-02", 20, "b", '{"url":"https://high.test","descendants":8}', "high text", [1.0, 0.0]),
        (3, "Mid", "2024-01-03", 30, "c", '{"descendants":2}', "mid text", [0.5, 0.5]),
    ]

    hits = top_archive_hits(rows, profile=profile, top_k=2, min_comments=1)

    assert [hit.id for hit in hits] == [2, 3]
    assert all(isinstance(hit, ArchiveHit) for hit in hits)
    assert hits[0].similarity > hits[1].similarity
    assert hits[0].comments == 8


def test_top_archive_hits_filters_min_comments() -> None:
    profile = np.array([1.0, 0.0], dtype=np.float32)
    rows = [
        (1, "No comments", "2024-01-01", 100, "a", '{"descendants":0}', "text", [1.0, 0.0]),
        (2, "Discussed", "2024-01-02", 10, "b", '{"descendants":3}', "text", [0.5, 0.5]),
    ]

    hits = top_archive_hits(rows, profile=profile, top_k=2, min_comments=1)

    assert [hit.id for hit in hits] == [2]


def test_profile_cache_round_trip_and_invalidates_on_ids(tmp_path) -> None:
    path = tmp_path / "profile.json"
    profile = np.ones(384, dtype=np.float32)

    save_profile_cache(path, "alice", [3, 2, 1], profile)

    assert np.allclose(load_profile_cache(path, "alice", [3, 2, 1]), profile)
    assert load_profile_cache(path, "alice", [3, 2]) is None
    assert load_profile_cache(path, "bob", [3, 2, 1]) is None


def test_archive_parquet_paths_can_limit_newest(monkeypatch) -> None:
    def fake_list_repo_files(repo: str, repo_type: str):
        assert repo
        assert repo_type == "dataset"
        return ["README.md", "train-00000.parquet", "train-00001.parquet", "train-00002.parquet"]

    def fake_hf_hub_url(repo: str, filename: str, repo_type: str):
        assert repo
        assert repo_type == "dataset"
        return f"https://example.test/{filename}"

    pytest.importorskip("huggingface_hub")
    monkeypatch.setattr("huggingface_hub.list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", fake_hf_hub_url)

    assert archive_parquet_paths(2) == [
        "https://example.test/train-00000.parquet",
        "https://example.test/train-00001.parquet",
    ]
    assert archive_parquet_paths(2, newest=True) == [
        "https://example.test/train-00001.parquet",
        "https://example.test/train-00002.parquet",
    ]
