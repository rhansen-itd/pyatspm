"""
ATSPM Cycle Processing Engine (Imperative Shell)

Implements two mathematically distinct processing paths driven by the nature
of newly ingested data.  Both paths derive their working boundaries from real
data anchors — existing ``cycle_start`` timestamps and gap markers — rather
than from fixed time buffers.

Architecture
============

Path A – Fast Append (``fill_gaps=False``, default)
----------------------------------------------------
Assumption: new data has been **appended to the end** of the timeline.

Cycle detection is a causal, forward-looking algorithm: once a cycle boundary
is locked in, nothing that arrives *before* it can change it.  Therefore only
the last committed cycle boundary that precedes (or equals) the new data needs
to be re-opened.

Anchor derivation::

    CS_prev = MAX(cycle_start) WHERE cycle_start <= T_start
              Falls back to T_start itself when the cycles table is empty.
    CS_next = None   (no future data to bound against)

Working window::

    Fetch events  : [CS_prev, end-of-events-table)
    Delete cycles : (CS_prev, +∞)    ← CS_prev kept; everything after re-derived
    Insert        : all freshly computed cycles

Path B – Gap Fill (``fill_gaps=True``)
---------------------------------------
Assumption: data has been inserted into a **historical gap** in the timeline.
This path surgically repairs only the affected region without disturbing
unrelated cycles on either side.

Anchor derivation (gap markers act as hard stops)::

    Gap_prev = MAX(timestamp) WHERE event_code = -1 AND timestamp <= T_start
    Gap_next = MIN(timestamp) WHERE event_code = -1 AND timestamp >= T_end
    CS_prev  = MAX(cycle_start) WHERE cycle_start <= T_start
                                  AND (Gap_prev IS NULL OR cycle_start >= Gap_prev)
    CS_next  = MIN(cycle_start) WHERE cycle_start >= T_end
                                  AND (Gap_next IS NULL OR cycle_start <= Gap_next)

Working window::

    Fetch events  : [CS_prev, CS_next or Gap_next or end-of-events)
    Delete cycles : (CS_prev, CS_next)   ← both anchors excluded
    Insert        : freshly computed cycles (anchors stripped before insert)

Gap Marker Scrubbing (Path B only)
------------------------------------
Before recalculating, any ``event_code = -1`` markers that fall **strictly
inside** ``(T_start, T_end)`` are removed from the events table.  Those
markers exist solely because of the gap that has now been filled.  Markers at
the exact edges (``ts == T_start`` or ``ts == T_end``) are preserved — the
IngestionEngine already decided whether a hard discontinuity exists at those
seams.

Config Boundary Splitting (Both Paths)
-----------------------------------------
``events_df`` is split into sub-segments only where the Ring-Barrier (``RB_*``)
configuration actually changes.  Adjacent configs whose RB signature is
identical are merged; only the latest config dict is kept.  This means a
detector-map or IP update mid-span does not force an unnecessary segment break.

Package Location: src/atspm/data/processing.py
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from .manager import DatabaseManager
from ..analysis.cycles import (
    CycleDetectionError,
    assign_ring_phases,
    calculate_cycles,
)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _rb_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the Ring-Barrier keys from a config dict.

    Args:
        config: Full config dict as returned by
                :class:`~atspm.data.manager.DatabaseManager`.

    Returns:
        Sub-dict whose keys all start with ``RB_``.
    """
    return {k: v for k, v in config.items() if k.startswith("RB_")}


# ---------------------------------------------------------------------------
# CycleProcessor
# ---------------------------------------------------------------------------

class CycleProcessor:
    """Orchestrates cycle detection across both the Fast-Append and Gap-Fill paths.

    Responsibilities:
        - Create and migrate the ``cycles`` table on first use.
        - Accept ``(T_start, T_end, fill_gaps)`` work orders from
          :class:`~atspm.data.ingestion.IngestionEngine`.
        - Derive mathematically sound processing anchors from live DB state.
        - Split event windows at RB-config boundaries.
        - Delegate calculation to the Functional Core (``calculate_cycles``).
        - Persist results atomically via delete-then-insert.
    """

    def __init__(self, db_path: Path, timezone: str = None):
        """Initialise the processor.

        Args:
            db_path:  Path to the SQLite database file.
            timezone: IANA timezone string (e.g. ``'US/Mountain'``).
                      When ``None`` the value is read from the ``metadata``
                      table, falling back to ``'US/Mountain'``.
        """
        self.db_path = Path(db_path)
        self.tz = pytz.timezone(self._resolve_timezone(timezone))
        self._init_cycles_table()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _resolve_timezone(self, timezone: Optional[str]) -> str:
        """Return the best available timezone string.

        Args:
            timezone: Caller-supplied value or ``None``.

        Returns:
            IANA timezone string.
        """
        if timezone is not None:
            return timezone
        try:
            with DatabaseManager(self.db_path) as m:
                meta = m.get_metadata()
                if meta and meta.get("timezone"):
                    return meta["timezone"]
        except Exception:
            pass
        return "US/Mountain"

    def _init_cycles_table(self) -> None:
        """Create the ``cycles`` table and its index; migrate older schemas."""
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_start      REAL PRIMARY KEY,
                    coord_plan       REAL NOT NULL DEFAULT 0,
                    detection_method TEXT NOT NULL DEFAULT '',
                    r1_phases        TEXT NOT NULL DEFAULT 'None',
                    r2_phases        TEXT NOT NULL DEFAULT 'None'
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycles_start
                ON cycles (cycle_start)
            """)
            cur.execute("PRAGMA table_info(cycles)")
            existing = {row[1] for row in cur.fetchall()}
            for col in ("r1_phases", "r2_phases"):
                if col not in existing:
                    cur.execute(
                        f"ALTER TABLE cycles "
                        f"ADD COLUMN {col} TEXT NOT NULL DEFAULT 'None'"
                    )
                    print(
                        f"CycleProcessor: migrated cycles table – added '{col}'. "
                        "Run backfill_ring_phases() to populate existing rows."
                    )
            m.conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_span(
        self,
        t_start: float,
        t_end: float,
        fill_gaps: bool = False,
    ) -> None:
        """Process cycles for one ingested time span.

        Primary entry-point called by :class:`IngestionEngine` after each
        batch commit.

        Args:
            t_start:   UTC epoch of the first new event in the span.
            t_end:     UTC epoch of the last new event in the span.
            fill_gaps: When ``False`` (default) use Path A (Fast Append).
                       When ``True`` use Path B (Gap Fill).
        """
        if fill_gaps:
            self._process_gap_fill(t_start, t_end)
        else:
            self._process_fast_append(t_start, t_end)

    def run(self, fill_gaps: bool = False) -> None:
        """Full-pass (re)processing over all ingested spans.

        Iterates over every span in ``ingestion_log`` and calls
        :meth:`process_span`.  Use for initial backfill after ingesting a
        large archive, or to repair the cycles table after a schema migration.

        Args:
            fill_gaps: Forwarded to :meth:`process_span` for each span.
        """
        with DatabaseManager(self.db_path) as m:
            spans_df = m.get_ingestion_spans()

        if spans_df.empty:
            return

        for _, row in spans_df.iterrows():
            try:
                self.process_span(
                    float(row["span_start"]),
                    float(row["span_end"]),
                    fill_gaps=fill_gaps,
                )
            except Exception as exc:
                print(
                    f"CycleProcessor ERROR – span "
                    f"{self._fmt(row['span_start'])} → "
                    f"{self._fmt(row['span_end'])}: {exc}"
                )

    def backfill_ring_phases(self) -> int:
        """Populate ``r1_phases`` / ``r2_phases`` for rows carrying the default.

        Does not reingest events or rerun cycle detection.  Reads existing
        ``cycle_start`` values, fetches Code-1 events and the matching config,
        then writes the computed ring strings back in-place.

        Returns:
            Number of cycle rows updated.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute("""
                SELECT DISTINCT cycle_start FROM cycles
                WHERE r1_phases = 'None' AND r2_phases = 'None'
                ORDER BY cycle_start
            """)
            rows = cur.fetchall()

        if not rows:
            print("backfill_ring_phases: nothing to do – all rows already populated")
            return 0

        epochs = [r[0] for r in rows]
        start_epoch, end_epoch = min(epochs), max(epochs)

        with DatabaseManager(self.db_path) as m:
            events_df = m.query_events(
                start_time=start_epoch,
                end_time=end_epoch + 1,
                event_codes=[1],
            )
            config = m.get_config_at_date(
                datetime.fromtimestamp(start_epoch, self.tz)
            )

        if events_df.empty or config is None:
            print("backfill_ring_phases: no events or config found – aborting")
            return 0

        cycles_df = pd.DataFrame({"cycle_start": epochs})
        updated = assign_ring_phases(cycles_df, events_df, config)

        records = list(
            updated[["r1_phases", "r2_phases", "cycle_start"]]
            .itertuples(index=False, name=None)
        )
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                cur.executemany(
                    "UPDATE cycles SET r1_phases = ?, r2_phases = ? "
                    "WHERE cycle_start = ?",
                    records,
                )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Database error during backfill: {exc}")

        print(f"backfill_ring_phases: updated {len(records)} rows")
        return len(records)

    # ------------------------------------------------------------------
    # Path A – Fast Append
    # ------------------------------------------------------------------

    def _process_fast_append(self, t_start: float, t_end: float) -> None:
        """Path A: recalculate from the last known cycle boundary forward.

        ``CS_prev`` is the highest ``cycle_start`` that is ``<= T_start``.
        This is the last fully-committed cycle; its start timestamp is a
        mathematically proven safe re-entry point because everything before
        it is already correct and immutable.

        Fetch window  : [CS_prev, MAX(events.timestamp)]
        Delete window : (CS_prev, +∞)   — CS_prev itself is preserved

        Args:
            t_start: UTC epoch of the first newly ingested event.
            t_end:   UTC epoch of the last newly ingested event (used only in
                     the log message; the fetch extends to the DB end).
        """
        with DatabaseManager(self.db_path) as m:
            cs_prev = m.get_cs_prev(t_start)
            fetch_start = cs_prev if cs_prev is not None else t_start

            # Extend to the absolute end of the events table so
            # calculate_cycles sees all available context.
            events_df = m.query_events(start_time=fetch_start)
            configs = m.get_configs_for_range(
                datetime.fromtimestamp(fetch_start, self.tz),
                datetime.fromtimestamp(t_end, self.tz),
            )

        if events_df.empty or not configs:
            return

        fetch_end = events_df["timestamp"].max()
        segments = self._build_rb_segments(configs, fetch_start, fetch_end)
        all_cycles = self._run_segments(segments, events_df)

        if not all_cycles:
            return

        full_cycles = pd.concat(all_cycles, ignore_index=True)

        # Do not duplicate the anchor itself.
        if cs_prev is not None:
            full_cycles = full_cycles[full_cycles["cycle_start"] > cs_prev]

        if full_cycles.empty:
            return

        self._save_cycles_append(full_cycles, delete_gt=cs_prev)

        n = len(full_cycles)
        method = full_cycles["detection_method"].iloc[0]
        print(
            f"CycleProcessor [append] "
            f"{self._fmt(t_start)} → {self._fmt(t_end)} | "
            f"anchor {self._fmt(cs_prev)} | "
            f"{n} cycles [{method}]"
        )

    # ------------------------------------------------------------------
    # Path B – Gap Fill
    # ------------------------------------------------------------------

    def _process_gap_fill(self, t_start: float, t_end: float) -> None:
        """Path B: surgically repair the region affected by newly filled data.

        Steps:

        1. Scrub obsolete gap markers strictly inside ``(T_start, T_end)``.
        2. Derive gap-aware anchors from the current DB state.
        3. Fetch events, calculate cycles, delete between anchors, insert.

        The two anchors ``CS_prev`` and ``CS_next`` are like load-bearing walls
        in the cycle timeline: everything between them is torn down and rebuilt
        from the fresh event data, while everything outside is untouched.

        Args:
            t_start: UTC epoch of the first newly ingested event.
            t_end:   UTC epoch of the last newly ingested event.
        """
        # Step 1 ─ remove markers made obsolete by the newly filled data.
        self._scrub_gap_markers(t_start, t_end)

        with DatabaseManager(self.db_path) as m:
            gap_prev = m.get_gap_prev(t_start)
            gap_next = m.get_gap_next(t_end)
            cs_prev  = m.get_cs_prev_bounded(t_start, gap_prev)
            cs_next  = m.get_cs_next_bounded(t_end,   gap_next)

            # Determine the fetch end-point.
            # Priority: CS_next > Gap_next > open-ended (None = read to DB end).
            fetch_start = cs_prev if cs_prev is not None else t_start
            if cs_next is not None:
                fetch_end: Optional[float] = cs_next
            elif gap_next is not None:
                fetch_end = gap_next
            else:
                fetch_end = None

            events_df = m.query_events(
                start_time=fetch_start,
                end_time=fetch_end,
            )
            # Use T_end as a proxy upper bound for the config range when
            # fetch_end is open-ended, so we still pull the right configs.
            config_end_dt = datetime.fromtimestamp(
                fetch_end if fetch_end is not None else t_end, self.tz
            )
            configs = m.get_configs_for_range(
                datetime.fromtimestamp(fetch_start, self.tz),
                config_end_dt,
            )

        if events_df.empty or not configs:
            return

        seg_end = fetch_end if fetch_end is not None else events_df["timestamp"].max()
        segments = self._build_rb_segments(configs, fetch_start, seg_end)
        all_cycles = self._run_segments(segments, events_df)

        if not all_cycles:
            return

        full_cycles = pd.concat(all_cycles, ignore_index=True)

        # Strip anchors — they already exist in the DB and must not be
        # touched by the delete-then-insert operation.
        mask = pd.Series(True, index=full_cycles.index)
        if cs_prev is not None:
            mask &= full_cycles["cycle_start"] > cs_prev
        if cs_next is not None:
            mask &= full_cycles["cycle_start"] < cs_next
        full_cycles = full_cycles[mask]

        if full_cycles.empty:
            return

        self._save_cycles_gap_fill(full_cycles, cs_prev, cs_next)

        n = len(full_cycles)
        method = full_cycles["detection_method"].iloc[0]
        print(
            f"CycleProcessor [gap-fill] "
            f"{self._fmt(t_start)} → {self._fmt(t_end)} | "
            f"anchor_prev {self._fmt(cs_prev)} | "
            f"anchor_next {self._fmt(cs_next)} | "
            f"{n} cycles [{method}]"
        )

    # ------------------------------------------------------------------
    # Gap marker scrubbing (Path B only)
    # ------------------------------------------------------------------

    def _scrub_gap_markers(self, t_start: float, t_end: float) -> None:
        """Delete stale ``event_code = -1`` rows strictly inside (T_start, T_end).

        When a gap is filled by new ingestion, any ``-1`` marker that was
        originally inserted *because* those files were missing becomes
        incorrect.  Leaving it in place would cause ``calculate_cycles`` to
        treat the now-continuous region as two disconnected fragments.

        Only markers **strictly between** the edges are removed; markers
        at ``ts == T_start`` or ``ts == T_end`` are kept because the
        IngestionEngine already evaluated whether a real discontinuity exists
        at those seams and its decision must not be overridden here.

        Args:
            t_start: UTC epoch of the newly filled gap's start.
            t_end:   UTC epoch of the newly filled gap's end.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                cur.execute(
                    "DELETE FROM events "
                    "WHERE event_code = -1 "
                    "  AND timestamp > ? "
                    "  AND timestamp < ?",
                    (t_start, t_end),
                )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Error scrubbing gap markers: {exc}")

    # ------------------------------------------------------------------
    # Segment building and execution (shared by both paths)
    # ------------------------------------------------------------------

    def _build_rb_segments(
        self,
        configs: List[Dict[str, Any]],
        fetch_start: float,
        fetch_end: float,
    ) -> List[Tuple[float, float, Dict[str, Any]]]:
        """Produce processing segments split only at true RB config boundaries.

        Consecutive configs with identical RB signatures are merged into a
        single segment.  The *latest* config dict wins so non-RB fields
        (detector maps, IPs, etc.) are always current within each segment.

        Args:
            configs:     Config dicts sorted by ``start_date`` ascending, each
                         containing an ``_epoch_start`` key added by
                         :meth:`~atspm.data.manager.DatabaseManager.get_configs_for_range`.
            fetch_start: UTC epoch for the start of the processing window.
            fetch_end:   UTC epoch for the end of the processing window.

        Returns:
            List of ``(seg_start, seg_end, config)`` covering
            ``[fetch_start, fetch_end)``.
        """
        if not configs:
            return []

        segments: List[Tuple[float, float, Dict[str, Any]]] = []
        current_start  = fetch_start
        current_config = configs[0]
        current_rb     = _rb_signature(current_config)

        for next_cfg in configs[1:]:
            boundary: float = next_cfg["_epoch_start"]
            if boundary >= fetch_end:
                break
            next_rb = _rb_signature(next_cfg)
            if next_rb != current_rb:
                # RB changed: close the current segment and start a new one.
                segments.append((current_start, boundary, current_config))
                current_start  = boundary
                current_config = next_cfg
                current_rb     = next_rb
            else:
                # Non-RB change only: adopt the newer config but keep segment open.
                current_config = next_cfg

        segments.append((current_start, fetch_end, current_config))
        return segments

    def _run_segments(
        self,
        segments: List[Tuple[float, float, Dict[str, Any]]],
        events_df: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """Execute ``calculate_cycles`` for each segment and collect results.

        Each segment slices its own view from the shared ``events_df``;
        no per-segment DB round-trips are needed.

        Args:
            segments:  Output of :meth:`_build_rb_segments`.
            events_df: Full fetched events DataFrame.

        Returns:
            List of non-empty cycles DataFrames (one per segment that
            produced at least one cycle).
        """
        results: List[pd.DataFrame] = []
        for seg_start, seg_end, config in segments:
            seg_events = events_df[
                (events_df["timestamp"] >= seg_start) &
                (events_df["timestamp"] <  seg_end)
            ].copy()
            if seg_events.empty:
                continue
            try:
                cycles_df = calculate_cycles(seg_events, config)
            except CycleDetectionError as exc:
                print(
                    f"CycleProcessor WARNING: detection failed "
                    f"{self._fmt(seg_start)} → {self._fmt(seg_end)}: {exc}"
                )
                continue
            if not cycles_df.empty:
                results.append(cycles_df)
        return results

    # ------------------------------------------------------------------
    # DB persistence helpers
    # ------------------------------------------------------------------

    def _save_cycles_append(
        self,
        cycles_df: pd.DataFrame,
        delete_gt: Optional[float],
    ) -> None:
        """Persist cycles for Path A atomically.

        Deletes every ``cycle_start > delete_gt``, then bulk-inserts.

        Args:
            cycles_df: New cycles to insert.
            delete_gt: Exclusive lower bound for deletion; ``None`` clears all.
        """
        records = self._to_records(cycles_df)
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                if delete_gt is None:
                    cur.execute("DELETE FROM cycles")
                else:
                    cur.execute(
                        "DELETE FROM cycles WHERE cycle_start > ?",
                        (delete_gt,),
                    )
                cur.executemany(
                    "INSERT INTO cycles "
                    "(cycle_start, coord_plan, detection_method, r1_phases, r2_phases) "
                    "VALUES (?, ?, ?, ?, ?)",
                    records,
                )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Database error saving cycles (append): {exc}")

    def _save_cycles_gap_fill(
        self,
        cycles_df: pd.DataFrame,
        cs_prev: Optional[float],
        cs_next: Optional[float],
    ) -> None:
        """Persist cycles for Path B atomically.

        Deletes cycles **strictly between** the two anchors (both excluded),
        then inserts the new rows.  The anchors themselves are never touched.

        Args:
            cycles_df: New cycles to insert (anchors already stripped).
            cs_prev:   Lower anchor epoch, or ``None`` for no lower bound.
            cs_next:   Upper anchor epoch, or ``None`` for no upper bound.
        """
        records = self._to_records(cycles_df)
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            try:
                if cs_prev is not None and cs_next is not None:
                    cur.execute(
                        "DELETE FROM cycles "
                        "WHERE cycle_start > ? AND cycle_start < ?",
                        (cs_prev, cs_next),
                    )
                elif cs_prev is not None:
                    cur.execute(
                        "DELETE FROM cycles WHERE cycle_start > ?",
                        (cs_prev,),
                    )
                elif cs_next is not None:
                    cur.execute(
                        "DELETE FROM cycles WHERE cycle_start < ?",
                        (cs_next,),
                    )
                else:
                    cur.execute("DELETE FROM cycles")

                cur.executemany(
                    "INSERT INTO cycles "
                    "(cycle_start, coord_plan, detection_method, r1_phases, r2_phases) "
                    "VALUES (?, ?, ?, ?, ?)",
                    records,
                )
                m.conn.commit()
            except sqlite3.Error as exc:
                m.conn.rollback()
                raise RuntimeError(f"Database error saving cycles (gap-fill): {exc}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_records(cycles_df: pd.DataFrame) -> list:
        """Convert a cycles DataFrame to a list of DB-insertion tuples.

        Args:
            cycles_df: Standard cycles DataFrame.

        Returns:
            List of ``(cycle_start, coord_plan, detection_method,
            r1_phases, r2_phases)`` tuples.
        """
        return list(
            cycles_df[
                ["cycle_start", "coord_plan", "detection_method",
                 "r1_phases", "r2_phases"]
            ]
            .fillna(
                {"r1_phases": "None", "r2_phases": "None",
                 "detection_method": "", "coord_plan": 0}
            )
            .itertuples(index=False, name=None)
        )

    def _fmt(self, epoch: Optional[float]) -> str:
        """Format a UTC epoch as a local-timezone human-readable string.

        Args:
            epoch: UTC epoch float, or ``None``.

        Returns:
            Formatted local datetime string, or ``'None'``.
        """
        if epoch is None:
            return "None"
        return datetime.fromtimestamp(epoch, self.tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    # ------------------------------------------------------------------
    # Statistics and validation
    # ------------------------------------------------------------------

    def get_processing_stats(self) -> dict:
        """Return summary statistics about the cycles table.

        Returns:
            Dict with keys ``total_cycles``, ``detection_methods``,
            ``coord_plans``, ``date_range``.
        """
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cycles")
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT detection_method, COUNT(*) FROM cycles "
                "GROUP BY detection_method"
            )
            methods = dict(cur.fetchall())

            cur.execute(
                "SELECT coord_plan, COUNT(*) FROM cycles GROUP BY coord_plan"
            )
            plans = dict(cur.fetchall())

            cur.execute("SELECT MIN(cycle_start), MAX(cycle_start) FROM cycles")
            min_ts, max_ts = cur.fetchone()

        return {
            "total_cycles": total,
            "detection_methods": methods,
            "coord_plans": plans,
            "date_range": {
                "start": self._fmt(min_ts) if min_ts else None,
                "end":   self._fmt(max_ts) if max_ts else None,
            },
        }

    def validate_cycles_table(self) -> Tuple[bool, List[str]]:
        """Validate the cycles table for data-quality issues.

        Returns:
            ``(is_valid, list_of_issues)``
        """
        issues: List[str] = []
        with DatabaseManager(self.db_path) as m:
            cur = m.conn.cursor()
            cur.execute(
                "SELECT cycle_start, COUNT(*) FROM cycles "
                "GROUP BY cycle_start HAVING COUNT(*) > 1"
            )
            if cur.fetchall():
                issues.append("Found duplicate cycle_start timestamps")

            cur.execute("""
                SELECT MIN(next_start - cycle_start),
                       MAX(next_start - cycle_start)
                FROM (
                    SELECT cycle_start,
                           LEAD(cycle_start) OVER (ORDER BY cycle_start) AS next_start
                    FROM cycles
                )
                WHERE next_start IS NOT NULL
            """)
            res = cur.fetchone()
            if res and res[0] is not None:
                mn, mx = res
                if mn < 10.0:
                    issues.append(f"Cycle length < 10 s (min: {mn:.1f} s)")
                if mx > 300.0:
                    issues.append(f"Cycle length > 300 s (max: {mx:.1f} s)")

            cur.execute(
                "SELECT COUNT(*) FROM cycles "
                "WHERE coord_plan IS NULL OR detection_method IS NULL"
            )
            null_count = cur.fetchone()[0]
            if null_count:
                issues.append(f"{null_count} rows with NULL values")

        return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_cycle_processing(
    db_path: Path,
    fill_gaps: bool = False,
    timezone: str = None,
) -> None:
    """Convenience wrapper for a full-pass cycle-processing run.

    Args:
        db_path:   Path to the SQLite database.
        fill_gaps: When ``True`` use Path B (Gap Fill) for every span.
        timezone:  Local timezone string; ``None`` reads from metadata.
    """
    processor = CycleProcessor(db_path, timezone=timezone)
    processor.run(fill_gaps=fill_gaps)

    stats = processor.get_processing_stats()
    print("\nCycle Processing Statistics:")
    print(f"  Total cycles:      {stats['total_cycles']:,}")
    print(f"  Detection methods: {stats['detection_methods']}")
    print(f"  Coord plans:       {stats['coord_plans']}")
    if stats["date_range"]["start"]:
        print(
            f"  Date range:        "
            f"{stats['date_range']['start']} → {stats['date_range']['end']}"
        )
