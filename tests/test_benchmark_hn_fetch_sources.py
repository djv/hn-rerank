from __future__ import annotations

from datetime import UTC, datetime

from scripts.benchmark_hn_fetch_sources import (
    SourceRun,
    build_bigquery_sql,
    compare_last_runs,
    story_from_bigquery_row,
    summarize_runs,
)


def test_story_from_bigquery_row_converts_to_story():
    row = {
        "id": 123,
        "title": "A &amp; B",
        "url": "https://example.com/post",
        "text": "<p>Self text</p>",
        "score": 42,
        "timestamp": datetime(2026, 4, 29, 12, 0, tzinfo=UTC),
        "comment_count": 17,
        "comments": [
            {"text": "<p>First &amp; useful comment</p>", "score": 5},
            {"text": "", "score": 4},
            {"text": "<i>Second comment</i>", "score": 3},
        ],
    }

    story = story_from_bigquery_row(row)

    assert story is not None
    assert story.id == 123
    assert story.title == "A & B"
    assert story.url == "https://example.com/post"
    assert story.score == 42
    assert story.time == 1777464000
    assert story.discussion_url == "https://news.ycombinator.com/item?id=123"
    assert story.comment_count == 17
    assert story.comments == ["First & useful comment", "Second comment"]
    assert "A & B" in story.text_content
    assert "Self text" in story.text_content
    assert "First & useful comment" in story.text_content


def test_story_from_bigquery_row_rejects_empty_payload():
    assert story_from_bigquery_row({"id": 0}) is None
    assert story_from_bigquery_row({"id": 123, "title": "", "comments": []}) is None


def test_summarize_runs_reports_mean_median_and_latest_count():
    runs = [
        SourceRun("algolia", 2.0, 10, {1, 2}, 100, {}),
        SourceRun("algolia", 4.0, 12, {2, 3}, 120, {}),
        SourceRun("algolia", 3.0, 11, {3, 4}, 110, {}),
    ]

    summary = summarize_runs("algolia", runs)

    assert summary["source"] == "algolia"
    assert summary["runs"] == 3
    assert summary["median_seconds"] == 3.0
    assert summary["mean_seconds"] == 3.0
    assert summary["story_count"] == 11
    assert summary["newest_time"] == 120


def test_compare_last_runs_reports_overlap_and_freshness_lag():
    algolia = SourceRun("algolia", 2.0, 4, {1, 2, 3, 4}, 200, {})
    bigquery = SourceRun("bigquery", 1.0, 4, {3, 4, 5, 6}, 150, {})

    comparison = compare_last_runs(algolia, bigquery)

    assert comparison["overlap_count"] == 2
    assert comparison["algolia_only_count"] == 2
    assert comparison["bigquery_only_count"] == 2
    assert comparison["overlap_ratio_vs_algolia"] == 0.5
    assert comparison["freshness_lag_seconds"] == 50
    assert comparison["newest_time"] == 200


def test_compare_last_runs_reports_fair_overlap():
    # Algolia has 2 new stories (100, 101) and 2 older stories (1, 2)
    algolia = SourceRun(
        "algolia", 2.0, 4, {100, 101, 1, 2}, 200, {100: 200, 101: 190, 1: 100, 2: 90}
    )
    # BigQuery only has older stories, newest is 150
    bigquery = SourceRun(
        "bigquery", 1.0, 4, {1, 2, 3, 4}, 150, {1: 100, 2: 90, 3: 80, 4: 70}
    )

    comparison = compare_last_runs(algolia, bigquery)

    # Raw overlap is 2 (ids 1, 2)
    assert comparison["overlap_count"] == 2

    # Fair overlap excludes 100 and 101 from Algolia because they are > 150.
    # Fair Algolia set: {1, 2}. Fair BigQuery set: {1, 2, 3, 4}.
    # Fair overlap is 2, but out of 2 Algolia stories instead of 4.
    assert comparison["fair_overlap_count"] == 2
    assert comparison["fair_overlap_ratio_vs_algolia"] == 1.0


def test_build_bigquery_sql_is_parameterized_and_uses_public_table():
    sql = build_bigquery_sql()

    assert "`bigquery-public-data.hacker_news.full`" in sql
    assert "WITH RECURSIVE" in sql
    assert "@start_ts" in sql
    assert "@end_ts" in sql
    assert "@candidate_limit" in sql
    assert "@max_comment_depth" in sql
