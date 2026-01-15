import pytest
from datetime import datetime, timedelta, UTC
import math


def generate_windows(now, days):
    # Logic copied from api/fetching.py to verify independently
    days_since_monday = now.weekday()
    anchor = (now - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    anchor_ts = int(anchor.timestamp())
    ts_now = int(now.timestamp() // 900 * 900)  # Round to 15m

    windows = []

    # 1. Live Window: Now -> Anchor (Last Monday)
    if ts_now > anchor_ts:
        windows.append((anchor_ts, ts_now, True))

    # 2. Archive Windows: 7-day chunks back from Anchor
    cutoff_ts = int((now - timedelta(days=days)).timestamp())
    current_end = anchor_ts

    max_archive_weeks = math.ceil(days / 7) + 1

    for _ in range(max_archive_weeks):
        if current_end <= cutoff_ts:
            break
        current_start = current_end - (7 * 86400)
        windows.append((current_start, current_end, False))
        current_end = current_start

    return windows


@pytest.mark.parametrize(
    "date_str, days",
    [
        ("2024-02-29 12:00:00", 30),  # Leap day
        ("2024-03-01 12:00:00", 30),  # Day after leap
        ("2023-12-31 23:59:59", 30),  # End of year
        ("2024-01-01 00:00:01", 30),  # Start of year
        ("2025-01-12 10:00:00", 1),  # Short duration
        ("2025-01-12 10:00:00", 365),  # Long duration
        ("2025-01-13 10:00:00", 7),  # Monday
    ],
)
def test_window_continuity_and_coverage(date_str, days):
    now = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    windows = generate_windows(now, days)

    assert len(windows) > 0

    # 1. Check Continuity
    # Sort by start time ascending
    windows.sort(key=lambda x: x[0])

    for i in range(len(windows) - 1):
        # End of current should equal Start of next
        assert windows[i][1] == windows[i + 1][0], f"Gap or overlap at index {i}"

    # 2. Check Coverage
    oldest_start = windows[0][0]
    newest_end = windows[-1][1]

    cutoff_ts = int((now - timedelta(days=days)).timestamp())
    ts_now = int(now.timestamp() // 900 * 900)

    # We expect oldest_start to be <= cutoff_ts (it goes back far enough)
    # UNLESS days is very small and we are at start of week?
    # No, if days=1. Cutoff = Now - 1 day.
    # Anchor = Mon.
    # If Now = Mon + 0.1. Cutoff = Mon - 0.9.
    # Live: Mon -> Mon + 0.1.
    # Arch 1: Mon-7 -> Mon.
    # Mon-7 < Mon-0.9. Covered.

    # If Now = Mon + 6. Cutoff = Mon + 5.
    # Live: Mon -> Mon + 6.
    # Mon (Anchor) < Mon + 5 (Cutoff).
    # Wait.
    # If cutoff > anchor, do we generate archive windows?
    # loop: `if current_end (Anchor) <= cutoff_ts: break`.
    # If Anchor (Mon) <= Cutoff (Mon+5). True. Break.
    # So we generate NO archive windows.
    # Live window covers Anchor -> Now (Mon -> Mon+6).
    # Range [Cutoff, Now] is [Mon+5, Mon+6].
    # Live window covers [Mon, Mon+6].
    # So [Mon+5, Mon+6] is inside Live window.
    # Correct.

    # So strictly: oldest_start <= cutoff_ts
    assert oldest_start <= cutoff_ts, (
        f"Oldest start {oldest_start} not <= cutoff {cutoff_ts}"
    )

    # And newest_end should be ts_now
    # If ts_now > anchor.
    # If ts_now <= anchor (e.g. exactly midnight? or negative skew?), windows logic handles it.
    if ts_now > int(get_anchor_ts(now)):
        assert newest_end == ts_now


def get_anchor_ts(now):
    days_since_monday = now.weekday()
    anchor = (now - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int(anchor.timestamp())


def test_specific_boundary_monday_midnight():
    # Monday 00:00:00
    now = datetime(2026, 1, 12, 0, 0, 0, tzinfo=UTC)  # A Monday
    days = 7

    # Anchor should be Now
    windows = generate_windows(now, days)
    # Live window?
    # ts_now = timestamp(now).
    # anchor_ts = timestamp(now).
    # ts_now > anchor_ts is False.
    # No live window.

    # Archive windows
    # current_end = Now.
    # cutoff = Now - 7 days.
    # 1. end=Now. > cutoff. Gen Now-7 -> Now.
    # 2. end=Now-7. <= cutoff. Break.

    # Result: 1 window: [Now-7, Now].
    assert len(windows) == 1
    assert windows[0][0] == int((now - timedelta(days=7)).timestamp())
    assert windows[0][1] == int(now.timestamp())
    assert windows[0][2] is False


if __name__ == "__main__":
    # Manually run if executed as script
    pass
