"""Tests for the reel-parity features: overlap presence, cleaning, unattended
alerts, queue estimates and peak hours."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import supervision as sv

from app.analytics import Session, peak_hours
from app.geometry import box_polygon_overlap_ratio
from app.occupancy import OccupancyTracker, TableStatus
from app.schemas import OccupancySettings, TableDef

TABLE = TableDef(id="T1", name="Table 1", polygon=[(100, 100), (200, 100), (200, 200), (100, 200)])


def settings(**overrides: object) -> OccupancySettings:
    values: dict[str, object] = {
        "confidence_threshold": 0.3,
        "enter_seconds": 5,
        "leave_seconds": 10,
        "reference_point": "bottom_center",
        "presence_hold_seconds": 1.0,
    }
    values.update(overrides)
    return OccupancySettings(**values)


def people(*boxes: tuple[float, float, float, float], ids: list[int] | None = None) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.full(len(boxes), 0.9),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.array(ids or list(range(1, len(boxes) + 1))),
    )


def run(tracker: OccupancyTracker, detections: sv.Detections, start: float, end: float, step: float = 0.5) -> None:
    for now in np.arange(start, end, step):
        tracker.update(detections, now=float(now))


def status(tracker: OccupancyTracker) -> TableStatus:
    return tracker.states[0].status


# A seated guest: feet (bottom centre 150, 240) outside the table, but most of
# the box overlaps the table outline.
SEATED = (130.0, 150.0, 170.0, 240.0)
AT_TABLE = (140.0, 110.0, 160.0, 150.0)


def test_overlap_ratio_math() -> None:
    polygon = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
    assert box_polygon_overlap_ratio(SEATED, polygon) > 0.5
    assert box_polygon_overlap_ratio((300.0, 300.0, 340.0, 380.0), polygon) == 0.0


def test_overlap_presence_counts_seated_guest() -> None:
    anchor_only = OccupancyTracker([TABLE], settings(enter_seconds=0))
    anchor_only.update(people(SEATED), now=0)
    assert status(anchor_only) is TableStatus.AVAILABLE

    overlap = OccupancyTracker([TABLE], settings(enter_seconds=0, presence_overlap_ratio=0.2))
    overlap.update(people(SEATED), now=0)
    assert status(overlap) is TableStatus.OCCUPIED
    assert overlap.states[0].people_count == 1


def test_cleaning_flow() -> None:
    tracker = OccupancyTracker([TABLE], settings(cleaning_enabled=True, cleaning_clear_seconds=3))
    run(tracker, people(AT_TABLE), 0, 20)
    run(tracker, people(), 20, 31)
    assert status(tracker) is TableStatus.NEEDS_CLEANING
    assert [a.kind for a in tracker.pending_alerts] == ["needs_cleaning"]

    run(tracker, people(AT_TABLE), 31, 33)  # staff present, not long enough yet
    assert status(tracker) is TableStatus.NEEDS_CLEANING
    run(tracker, people(AT_TABLE), 33, 35)
    assert status(tracker) is TableStatus.AVAILABLE


def test_mark_cleaned() -> None:
    tracker = OccupancyTracker([TABLE], settings(cleaning_enabled=True))
    run(tracker, people(AT_TABLE), 0, 20)
    run(tracker, people(), 20, 31)
    assert status(tracker) is TableStatus.NEEDS_CLEANING
    events = tracker.mark_cleaned("T1", now=31)
    assert status(tracker) is TableStatus.AVAILABLE
    assert events[0].new_status is TableStatus.AVAILABLE


def test_unattended_alert_and_staff_reset() -> None:
    tracker = OccupancyTracker([TABLE], settings(unattended_seconds=60))
    run(tracker, people(AT_TABLE, ids=[1]), 0, 70)
    assert status(tracker) is TableStatus.OCCUPIED
    assert [a.kind for a in tracker.pending_alerts] == ["unattended"]
    assert tracker.is_unattended(tracker.states[0], now=70)

    # A new track (staff) arrives: the timer resets.
    tracker.update(people(AT_TABLE, (120.0, 110.0, 140.0, 150.0), ids=[1, 2]), now=70)
    assert not tracker.is_unattended(tracker.states[0], now=70)
    assert tracker.states[0].last_staff_visit == 70


def test_queue_ranking_prefers_longest_seated_without_history() -> None:
    t2 = TableDef(id="T2", name="Table 2", polygon=[(300, 100), (400, 100), (400, 200), (300, 200)])
    tracker = OccupancyTracker([TABLE, t2], settings())
    run(tracker, people(AT_TABLE, ids=[1]), 0, 10)
    run(tracker, people(AT_TABLE, (340.0, 110.0, 360.0, 150.0), ids=[1, 2]), 10, 20)
    assert tracker.queue_ranking(now=20) == ["T1", "T2"]


def test_peak_hours_splits_across_hours() -> None:
    start = datetime(2026, 11, 27, 18, 30).timestamp()  # a Friday
    sessions = [Session("T1", "Table 1", start, start + 3600)]
    hours = peak_hours(sessions)
    by_hour = {h["hour"]: h for h in hours}
    assert by_hour[18]["occupied_seconds"] == 1800
    assert by_hour[19]["occupied_seconds"] == 1800
    assert by_hour[18]["session_count"] == 1
    assert by_hour[18]["weekday"] == 4  # Friday
