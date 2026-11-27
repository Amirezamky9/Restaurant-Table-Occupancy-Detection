"""Table occupancy: which tables have people at them, with enter/leave delays.

Pure logic with no drawing and no web code, so it can be tested with fake
detections and explicit timestamps.

Each table runs this state machine:

    AVAILABLE -> PENDING_OCCUPIED -> OCCUPIED -> PENDING_AVAILABLE -> AVAILABLE

* It becomes OCCUPIED only after someone has been at it continuously for
  ``enter_seconds``. If they leave before that (a passer-by), it goes straight
  back to AVAILABLE.
* It becomes AVAILABLE only after nobody has been at it continuously for
  ``leave_seconds``. If someone comes back before that, it stays OCCUPIED.

Times are wall-clock seconds, so the behaviour does not depend on the frame
rate. A person is "at" a table when the chosen anchor point of their box
(bottom centre or centre) is inside the table's polygon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import supervision as sv

from app.geometry import box_polygon_overlap_ratio
from app.schemas import OccupancySettings, TableDef

_ANCHORS = {"bottom_center": sv.Position.BOTTOM_CENTER, "center": sv.Position.CENTER}


class TableStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    PENDING_OCCUPIED = "PENDING_OCCUPIED"
    OCCUPIED = "OCCUPIED"
    PENDING_AVAILABLE = "PENDING_AVAILABLE"
    NEEDS_CLEANING = "NEEDS_CLEANING"

    @property
    def counts_as_occupied(self) -> bool:
        """A pending table keeps its previous status until the delay has passed."""
        return self in (TableStatus.OCCUPIED, TableStatus.PENDING_AVAILABLE)


def anchor_position(reference_point: str) -> sv.Position:
    """The supervision anchor for a config's reference_point."""
    return _ANCHORS[reference_point]


@dataclass(frozen=True)
class OccupancyEvent:
    """A table changed status."""

    table_id: str
    old_status: TableStatus
    new_status: TableStatus
    timestamp: float  # Unix time in seconds


@dataclass(frozen=True)
class AlertEvent:
    """A table needs attention (kind: "unattended" or "needs_cleaning")."""

    table_id: str
    kind: str
    timestamp: float


@dataclass
class TableState:
    """Current status and statistics of one table."""

    table_id: str
    name: str
    status: TableStatus = TableStatus.AVAILABLE
    people_count: int = 0
    track_ids: list[int] = field(default_factory=list)
    occupied_since: float | None = None  # when the people of the current session arrived
    total_occupied_seconds: float = 0.0  # finished sessions; see occupied_seconds()
    session_count: int = 0
    # Timers of the state machine.
    last_seen: float | None = None  # last time anyone was detected at the table
    presence_started: float | None = None  # start of the current run of presence
    absence_started: float | None = None  # start of the current run of absence
    # Cleaning and staff-attention tracking.
    cleaning_since: float | None = None  # when the table entered NEEDS_CLEANING
    last_staff_visit: float | None = None  # last time a new person arrived at the occupied table
    unattended_alerted: bool = False  # the unattended alert already fired for this stretch
    session_track_ids: set[int] = field(default_factory=set)  # tracks seen at the table this session
    awaiting_absence: bool = False  # staff just cleaned; their presence must not start a session

    def occupied_seconds(self, now: float) -> float:
        """Total occupied time, including the session in progress."""
        if self.occupied_since is None:
            return self.total_occupied_seconds
        # While waiting to become AVAILABLE, the session ended when people left.
        leaving = self.status is TableStatus.PENDING_AVAILABLE and self.absence_started is not None
        session_end = self.absence_started if leaving else now
        return self.total_occupied_seconds + max(0.0, session_end - self.occupied_since)  # type: ignore[operator]


class OccupancyTracker:
    """Runs the occupancy state machine for a set of tables."""

    def __init__(self, tables: list[TableDef], settings: OccupancySettings) -> None:
        self.tables: list[TableDef] = []
        self.settings = settings
        self._zones: dict[str, sv.PolygonZone] = {}
        self._states: dict[str, TableState] = {}
        self.pending_alerts: list[AlertEvent] = []
        self.set_tables(tables, settings)

    def set_tables(self, tables: list[TableDef], settings: OccupancySettings | None = None) -> None:
        """Switch to new table outlines (and settings) without losing state.

        Tables whose ID is kept keep their status, timers and statistics, so
        editing an outline does not reset an occupied table. New tables start
        AVAILABLE; removed tables are forgotten.
        """
        if settings is not None:
            self.settings = settings
        anchor = anchor_position(self.settings.reference_point)
        self._zones = {
            table.id: sv.PolygonZone(
                polygon=np.rint(np.array(table.polygon)).astype(int),
                triggering_anchors=(anchor,),
            )
            for table in tables
        }
        old_states = self._states
        self._states = {table.id: old_states.get(table.id) or TableState(table.id, table.name) for table in tables}
        for table in tables:
            self._states[table.id].name = table.name
        self.tables = list(tables)

    @property
    def states(self) -> list[TableState]:
        """State of every table, in config order."""
        return [self._states[table.id] for table in self.tables]

    def update(self, detections: sv.Detections, now: float | None = None) -> list[OccupancyEvent]:
        """Feed the people found in one frame; returns the status changes it caused.

        ``now`` is the frame's wall-clock time (defaults to the current time).
        """
        now = time.time() if now is None else now
        if detections.confidence is not None and len(detections):
            detections = detections[detections.confidence >= self.settings.confidence_threshold]

        events: list[OccupancyEvent] = []
        for table in self.tables:
            state = self._states[table.id]
            inside = self._zones[table.id].trigger(detections)
            if self.settings.presence_overlap_ratio > 0 and len(detections):
                polygon = [(float(x), float(y)) for x, y in table.polygon]
                overlap = np.array([
                    box_polygon_overlap_ratio(tuple(float(v) for v in box), polygon)
                    >= self.settings.presence_overlap_ratio
                    for box in detections.xyxy
                ])
                inside = inside | overlap
            state.people_count = int(inside.sum())
            track_ids = detections.tracker_id[inside] if detections.tracker_id is not None else []
            state.track_ids = sorted(int(track_id) for track_id in track_ids if track_id >= 0)
            self._note_staff_visit(state, now)
            events.extend(self._advance(state, bool(inside.any()), now))
            self._check_unattended(state, now)
        return events

    def _note_staff_visit(self, state: TableState, now: float) -> None:
        """A track arriving at an occupied table counts as a staff visit.

        Guests already seated are in ``session_track_ids``; anyone new showing
        up at the table is treated as staff attending to it (a late-joining
        guest also resets the timer, which is the safe direction).
        """
        if state.status is not TableStatus.OCCUPIED:
            return
        new_tracks = [t for t in state.track_ids if t not in state.session_track_ids]
        if new_tracks:
            state.last_staff_visit = now
            state.unattended_alerted = False
            state.session_track_ids.update(new_tracks)

    def _check_unattended(self, state: TableState, now: float) -> None:
        limit = self.settings.unattended_seconds
        if limit <= 0 or state.status is not TableStatus.OCCUPIED or state.occupied_since is None:
            return
        since = state.last_staff_visit if state.last_staff_visit is not None else state.occupied_since
        if now - since >= limit and not state.unattended_alerted:
            state.unattended_alerted = True
            self.pending_alerts.append(AlertEvent(state.table_id, "unattended", now))

    def is_unattended(self, state: TableState, now: float) -> bool:
        limit = self.settings.unattended_seconds
        if limit <= 0 or state.status is not TableStatus.OCCUPIED or state.occupied_since is None:
            return False
        since = state.last_staff_visit if state.last_staff_visit is not None else state.occupied_since
        return now - since >= limit

    def mark_cleaned(self, table_id: str, now: float | None = None) -> list[OccupancyEvent]:
        """Staff confirmed the table is cleaned (dashboard action)."""
        state = self._states.get(table_id)
        if state is None or state.status is not TableStatus.NEEDS_CLEANING:
            return []
        now = time.time() if now is None else now
        state.status = TableStatus.AVAILABLE
        state.cleaning_since = None
        state.presence_started = None
        return [OccupancyEvent(table_id, TableStatus.NEEDS_CLEANING, TableStatus.AVAILABLE, now)]

    def expected_free_in(self, state: TableState, now: float) -> float | None:
        """Queue estimate: seconds until this occupied table is expected to free up.

        Uses the table's own average finished session; with no history, tables
        occupied longest are expected first (remaining time unknown -> elapsed).
        """
        if state.status is not TableStatus.OCCUPIED or state.occupied_since is None:
            return None
        elapsed = now - state.occupied_since
        if state.session_count > 1 and state.total_occupied_seconds > 0:
            average = state.total_occupied_seconds / (state.session_count - 1)
            return max(0.0, average - elapsed)
        return None

    def queue_ranking(self, now: float) -> list[str]:
        """Occupied table IDs, soonest expected to free up first."""
        occupied = [s for s in self.states if s.status is TableStatus.OCCUPIED]
        def key(state: TableState) -> tuple[float, float]:
            expected = self.expected_free_in(state, now)
            elapsed = now - state.occupied_since if state.occupied_since is not None else 0.0
            return (expected if expected is not None else 0.0, -elapsed)
        return [s.table_id for s in sorted(occupied, key=key)]

    def _advance(self, state: TableState, detected: bool, now: float) -> list[OccupancyEvent]:
        """Update one table's timers and status."""
        if detected:
            state.last_seen = now
        present = state.last_seen is not None and now - state.last_seen <= self.settings.presence_hold_seconds
        if not present:
            state.awaiting_absence = False

        if present:
            if state.presence_started is None:
                state.presence_started = now
            state.absence_started = None
        else:
            if state.absence_started is None:
                # Absence began when the table was last seen occupied.
                state.absence_started = state.last_seen if state.last_seen is not None else now
            state.presence_started = None

        events: list[OccupancyEvent] = []

        def move_to(new_status: TableStatus) -> None:
            events.append(OccupancyEvent(state.table_id, state.status, new_status, now))
            state.status = new_status

        # Loop so a zero delay can pass through a pending state in one update.
        for _ in range(4):
            status = state.status
            if status is TableStatus.AVAILABLE and present and not state.awaiting_absence:
                move_to(TableStatus.PENDING_OCCUPIED)
            elif status is TableStatus.PENDING_OCCUPIED:
                if not present:
                    move_to(TableStatus.AVAILABLE)  # a passer-by
                elif now - state.presence_started >= self.settings.enter_seconds:  # type: ignore[operator]
                    state.occupied_since = state.presence_started
                    state.session_count += 1
                    state.session_track_ids = set(state.track_ids)
                    state.last_staff_visit = None
                    state.unattended_alerted = False
                    move_to(TableStatus.OCCUPIED)
            elif status is TableStatus.OCCUPIED and not present:
                move_to(TableStatus.PENDING_AVAILABLE)
            elif status is TableStatus.PENDING_AVAILABLE:
                if present:
                    move_to(TableStatus.OCCUPIED)  # came back: same session
                elif now - state.absence_started >= self.settings.leave_seconds:  # type: ignore[operator]
                    state.total_occupied_seconds += max(0.0, state.absence_started - state.occupied_since)  # type: ignore[operator]
                    state.occupied_since = None
                    state.session_track_ids = set()
                    if self.settings.cleaning_enabled:
                        state.cleaning_since = now
                        move_to(TableStatus.NEEDS_CLEANING)
                        self.pending_alerts.append(AlertEvent(state.table_id, "needs_cleaning", now))
                    else:
                        move_to(TableStatus.AVAILABLE)
            elif status is TableStatus.NEEDS_CLEANING:
                if present and state.presence_started is not None and (
                    now - state.presence_started >= self.settings.cleaning_clear_seconds
                ):
                    state.cleaning_since = None
                    state.awaiting_absence = True  # the cleaner is still standing there
                    move_to(TableStatus.AVAILABLE)
            if state.status is status:
                break
        return events
