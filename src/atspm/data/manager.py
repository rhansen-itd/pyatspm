"""
Database Manager for ATSPM System (Imperative Shell)

Handles all database operations including initialization, configuration
import, event queries, and the anchor-resolution queries that support the
two cycle-processing paths in ``processing.py``.

Package Location: src/atspm/data/manager.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class DatabaseManager:
    """Manages SQLite database operations for the ATSPM system.

    Responsibilities:
        - Database initialisation with WAL mode and full schema.
        - Configuration import and hybrid-schema transformation.
        - Transaction management via context-manager protocol.
        - Event and cycle query execution.
        - Anchor-resolution queries for cycle processing (Path A & B).
    """

    def __init__(self, db_path: Path):
        """Initialise with path to the SQLite database.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "DatabaseManager":
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn:
            self.conn.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Initialise database schema and indices.

        Creates:
            1. ``events``       – raw traffic event data
            2. ``config``       – hybrid schema for intersection configuration
            3. ``metadata``     – static intersection attributes
            4. ``ingestion_log`` – span-based ingestion state

        Enables WAL mode for concurrent access.
        """
        if not self.conn:
            raise RuntimeError("No connection. Use 'with DatabaseManager(...) as m:'.")

        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                timestamp  REAL    NOT NULL,
                event_code INTEGER NOT NULL,
                parameter  INTEGER NOT NULL,
                UNIQUE(timestamp, event_code, parameter) ON CONFLICT IGNORE
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events (timestamp)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_code_param
            ON events (event_code, parameter)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_ts_code
            ON events (timestamp, event_code)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date TEXT NOT NULL,
                end_date   TEXT,
                UNIQUE(start_date) ON CONFLICT REPLACE
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                lock_id           INTEGER PRIMARY KEY CHECK (lock_id = 1),
                intersection_id   TEXT,
                intersection_name TEXT,
                controller_ip     TEXT,
                detection_type    TEXT,
                detection_ip      TEXT,
                major_road_route  TEXT,
                major_road_name   TEXT,
                minor_road_route  TEXT,
                minor_road_name   TEXT,
                latitude          REAL,
                longitude         REAL,
                timezone          TEXT NOT NULL DEFAULT 'US/Mountain',
                agency_id         TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                span_start   REAL PRIMARY KEY,
                span_end     REAL NOT NULL,
                processed_at TEXT NOT NULL,
                row_count    INTEGER NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ingestion_span_end
            ON ingestion_log (span_end)
        """)

        self.conn.commit()
        print(f"Database initialised at {self.db_path}")

    # ------------------------------------------------------------------
    # Config table helpers
    # ------------------------------------------------------------------

    def get_config_columns(self) -> List[str]:
        """Return existing column names in the ``config`` table.

        Returns:
            List of column name strings.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(config)")
        return [row[1] for row in cur.fetchall()]

    def add_config_column(self, column_name: str, column_type: str = "TEXT") -> None:
        """Add a column to the ``config`` table if it does not already exist.

        Args:
            column_name: Name of the column to add.
            column_type: SQL type (default ``TEXT``).
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        if column_name not in self.get_config_columns():
            safe = column_name.replace('"', '""')
            self.conn.cursor().execute(
                f'ALTER TABLE config ADD COLUMN "{safe}" {column_type}'
            )
            self.conn.commit()

    def import_config(self, csv_path: Path) -> None:
        """Import intersection configuration from the legacy CSV format.

        Transforms ``int_cfg.csv`` into the hybrid schema:
        - Standard rows (TM, RB, Det, WD) become wide columns.
        - Exclusion rows (Exc) become JSON in ``TM_Exclusions``.

        Args:
            csv_path: Path to the ``int_cfg.csv`` file.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Config file not found: {csv_path}")

        df_cfg = pd.read_csv(csv_path, index_col=[0, 1])
        df_cfg = df_cfg.dropna(how="all", axis=1)
        df_cfg.columns = pd.to_datetime(df_cfg.columns, errors="coerce")
        df_cfg = df_cfg.loc[:, df_cfg.columns.notna()]
        if df_cfg.empty or len(df_cfg.columns) == 0:
            raise ValueError("No valid date columns found in config CSV")

        df_cfg = df_cfg.sort_index(axis=1)
        date_columns = sorted(df_cfg.columns)

        for i, start_date in enumerate(date_columns):
            end_date = date_columns[i + 1] if i < len(date_columns) - 1 else None
            config_data = self._transform_config_column(df_cfg[start_date])
            row_data = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat() if end_date else None,
            }
            row_data.update(config_data)
            for col in row_data:
                if col not in ("start_date", "end_date"):
                    self.add_config_column(col)
            self._insert_config_row(row_data)

        self.conn.commit()
        print(f"Imported {len(date_columns)} configuration periods from {csv_path}")

    def _transform_config_column(self, config_series: pd.Series) -> Dict[str, Any]:
        """Transform one date column from ``int_cfg.csv`` into a dict.

        Args:
            config_series: Pandas Series with multi-index (category, parameter).

        Returns:
            Dict of ``{column_name: value}``.
        """
        result: Dict[str, Any] = {}
        categories = config_series.index.get_level_values(0).unique()

        exclusions: list = []
        if "Exc:" in categories:
            exclusions = self._parse_exclusions(config_series.loc["Exc:"])

        for category in categories:
            cat_data = config_series.loc[category]
            if category == "Exc:":
                continue
            elif category == "TM:":
                for movement, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        result[f"TM_{movement}"] = str(value).strip()
                result["TM_Exclusions"] = json.dumps(exclusions)
            elif category in ("Det:", "Plt:"):
                for param, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        result[f"Det_{param.replace(' ', '_')}"] = str(value).strip()
            elif category == "RB:":
                for param, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        result[f"RB_{param}"] = str(value).strip()
            elif category == "WD:":
                for param, value in cat_data.items():
                    if pd.notna(value) and str(value).strip():
                        result[f"WD_{param.replace(' ', '_')}"] = str(value).strip()
        return result

    def _parse_exclusions(self, exc_series: pd.Series) -> list:
        """Parse exclusion rows into a list of dicts.

        Args:
            exc_series: Series with index labels Detector, Phase, Status.

        Returns:
            List of ``{"detector": int, "phase": int, "status": str}`` dicts.
        """
        def clean(val) -> str:
            return "" if pd.isna(val) else str(val).strip()

        det_str    = clean(exc_series.get("Detector"))
        phase_str  = clean(exc_series.get("Phase"))
        status_str = clean(exc_series.get("Status"))

        if not any([det_str, phase_str, status_str]):
            return []

        try:
            dets     = [int(d.strip()) for d in det_str.split(",")    if d.strip()]
            phases   = [int(p.strip()) for p in phase_str.split(",")  if p.strip()]
            statuses = [s.strip()      for s in status_str.split(",") if s.strip()]
            if not (len(dets) == len(phases) == len(statuses)):
                print(
                    f"Warning: exclusion row lengths differ – "
                    f"det:{len(dets)} ph:{len(phases)} status:{len(statuses)}"
                )
                return []
            return [
                {"detector": d, "phase": p, "status": s}
                for d, p, s in zip(dets, phases, statuses)
            ]
        except (ValueError, AttributeError) as exc:
            print(f"Warning: error parsing exclusions: {exc}")
            return []

    def _insert_config_row(self, row_data: Dict[str, Any]) -> None:
        """Insert one config row into the database.

        Args:
            row_data: Dict of column-name → value.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cols  = list(row_data.keys())
        names = ", ".join(f'"{c}"' for c in cols)
        ph    = ", ".join("?" for _ in cols)
        self.conn.cursor().execute(
            f"INSERT OR REPLACE INTO config ({names}) VALUES ({ph})",
            [row_data[c] for c in cols],
        )

    # ------------------------------------------------------------------
    # Config query methods
    # ------------------------------------------------------------------

    def get_config_at_date(self, query_date: datetime) -> Optional[Dict[str, Any]]:
        """Retrieve the configuration active at a specific datetime.

        Args:
            query_date: Datetime to query (naive or aware; date portion used).

        Returns:
            Config dict with a parsed ``exclusions`` key, or ``None``.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        q = query_date.isoformat()
        cur = self.conn.cursor()
        cur.execute("""
            SELECT * FROM config
            WHERE start_date <= ?
              AND (end_date IS NULL OR end_date > ?)
            ORDER BY start_date DESC
            LIMIT 1
        """, (q, q))
        row = cur.fetchone()
        if not row:
            return None
        cfg = dict(zip([d[0] for d in cur.description], row))
        cfg["exclusions"] = (
            json.loads(cfg["TM_Exclusions"]) if cfg.get("TM_Exclusions") else []
        )
        return cfg

    def get_configs_for_range(
        self,
        range_start: datetime,
        range_end: datetime,
    ) -> List[Dict[str, Any]]:
        """Return all configs whose validity period overlaps [range_start, range_end].

        Each returned dict is augmented with:
            ``_epoch_start``: UTC epoch at which *this* config becomes active
                              within the requested range (clamped to range_start).
            ``_epoch_end``:   UTC epoch at which it ceases (clamped to range_end).

        The list is sorted ascending by ``start_date``.

        Args:
            range_start: Aware datetime for the start of the window.
            range_end:   Aware datetime for the end of the window.

        Returns:
            List of augmented config dicts (may be empty).
        """
        if not self.conn:
            raise RuntimeError("No active connection.")

        cur = self.conn.cursor()
        cur.execute("""
            SELECT * FROM config
            WHERE start_date <= ?
              AND (end_date IS NULL OR end_date > ?)
            ORDER BY start_date ASC
        """, (range_end.isoformat(), range_start.isoformat()))
        rows = cur.fetchall()
        if not rows:
            return []

        columns = [d[0] for d in cur.description]
        tz = range_start.tzinfo
        results: List[Dict[str, Any]] = []

        for row in rows:
            cfg = dict(zip(columns, row))
            cfg["exclusions"] = (
                json.loads(cfg["TM_Exclusions"]) if cfg.get("TM_Exclusions") else []
            )
            # Compute epoch boundaries clamped to the requested range.
            cfg_start = datetime.fromisoformat(cfg["start_date"])
            eff_start = max(cfg_start, range_start.replace(tzinfo=None))
            cfg["_epoch_start"] = (
                tz.localize(eff_start).timestamp() if tz else eff_start.timestamp()
            )
            if cfg.get("end_date"):
                cfg_end   = datetime.fromisoformat(cfg["end_date"])
                eff_end   = min(cfg_end, range_end.replace(tzinfo=None))
            else:
                eff_end   = range_end.replace(tzinfo=None)
            cfg["_epoch_end"] = (
                tz.localize(eff_end).timestamp() if tz else eff_end.timestamp()
            )
            results.append(cfg)

        return results

    # ------------------------------------------------------------------
    # Anchor-resolution queries (used by processing.py)
    # ------------------------------------------------------------------

    def get_cs_prev(self, t_start: float) -> Optional[float]:
        """Return the highest ``cycle_start`` that is ``<= t_start``.

        Used by Path A (Fast Append) as the safe re-entry anchor.

        Args:
            t_start: UTC epoch of the first new event.

        Returns:
            UTC epoch float, or ``None`` if the cycles table is empty.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        cur.execute(
            "SELECT MAX(cycle_start) FROM cycles WHERE cycle_start <= ?",
            (t_start,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_gap_prev(self, t_start: float) -> Optional[float]:
        """Return the most recent gap-marker timestamp ``<= t_start``.

        Used by Path B (Gap Fill) to prevent anchor lookup from crossing a
        hard discontinuity boundary.

        Args:
            t_start: UTC epoch of the start of the newly filled range.

        Returns:
            UTC epoch float, or ``None`` if no preceding gap marker exists.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        cur.execute(
            "SELECT MAX(timestamp) FROM events "
            "WHERE event_code = -1 AND timestamp <= ?",
            (t_start,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_gap_next(self, t_end: float) -> Optional[float]:
        """Return the earliest gap-marker timestamp ``>= t_end``.

        Used by Path B to cap the fetch window so it does not cross into a
        separate discontinuous segment.

        Args:
            t_end: UTC epoch of the end of the newly filled range.

        Returns:
            UTC epoch float, or ``None`` if no following gap marker exists.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        cur.execute(
            "SELECT MIN(timestamp) FROM events "
            "WHERE event_code = -1 AND timestamp >= ?",
            (t_end,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_cs_prev_bounded(
        self,
        t_start: float,
        gap_prev: Optional[float],
    ) -> Optional[float]:
        """Return the gap-bounded lower cycle anchor for Path B.

        Finds ``MAX(cycle_start)`` that is ``<= t_start`` and also does not
        cross back past ``gap_prev`` (if one exists).  This ensures the anchor
        belongs to the same continuous segment as the newly filled data.

        Args:
            t_start:  UTC epoch of the start of the newly filled range.
            gap_prev: UTC epoch of the nearest preceding gap marker, or
                      ``None`` if no gap precedes the range.

        Returns:
            UTC epoch float, or ``None``.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        if gap_prev is not None:
            cur.execute(
                "SELECT MAX(cycle_start) FROM cycles "
                "WHERE cycle_start <= ? AND cycle_start >= ?",
                (t_start, gap_prev),
            )
        else:
            cur.execute(
                "SELECT MAX(cycle_start) FROM cycles WHERE cycle_start <= ?",
                (t_start,),
            )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_cs_next_bounded(
        self,
        t_end: float,
        gap_next: Optional[float],
    ) -> Optional[float]:
        """Return the gap-bounded upper cycle anchor for Path B.

        Finds ``MIN(cycle_start)`` that is ``>= t_end`` and also does not
        reach past ``gap_next`` (if one exists).

        Args:
            t_end:    UTC epoch of the end of the newly filled range.
            gap_next: UTC epoch of the nearest following gap marker, or
                      ``None`` if no gap follows the range.

        Returns:
            UTC epoch float, or ``None``.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        if gap_next is not None:
            cur.execute(
                "SELECT MIN(cycle_start) FROM cycles "
                "WHERE cycle_start >= ? AND cycle_start <= ?",
                (t_end, gap_next),
            )
        else:
            cur.execute(
                "SELECT MIN(cycle_start) FROM cycles WHERE cycle_start >= ?",
                (t_end,),
            )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    # ------------------------------------------------------------------
    # Event I/O
    # ------------------------------------------------------------------

    def insert_events(self, events: List[Tuple[float, int, int]]) -> int:
        """Bulk-insert events, silently ignoring duplicates.

        Args:
            events: List of ``(timestamp, event_code, parameter)`` tuples.

        Returns:
            Number of rows actually inserted.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        before = cur.fetchone()[0]
        cur.executemany(
            "INSERT OR IGNORE INTO events (timestamp, event_code, parameter) "
            "VALUES (?, ?, ?)",
            events,
        )
        cur.execute("SELECT COUNT(*) FROM events")
        after = cur.fetchone()[0]
        self.conn.commit()
        return after - before

    def query_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_codes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Query events from the database with optional filters.

        Args:
            start_time:  Minimum timestamp (inclusive).  ``None`` = no lower bound.
            end_time:    Maximum timestamp (exclusive).  ``None`` = no upper bound.
            event_codes: List of event codes to include.  ``None`` = all codes.

        Returns:
            DataFrame with columns ``[timestamp, event_code, parameter]``.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")

        clauses: List[str] = []
        params: list = []

        if start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("timestamp < ?")
            params.append(end_time)
        if event_codes:
            ph = ", ".join("?" for _ in event_codes)
            clauses.append(f"event_code IN ({ph})")
            params.extend(event_codes)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT timestamp, event_code, parameter FROM events "
            f"WHERE {where} ORDER BY timestamp, event_code, parameter"
        )
        return pd.read_sql_query(sql, self.conn, params=params)

    def get_event_date_range(self) -> Optional[Tuple[float, float]]:
        """Return ``(min_timestamp, max_timestamp)`` from the events table.

        Returns:
            Tuple of UTC epoch floats, or ``None`` if the table is empty.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        cur = self.conn.cursor()
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events")
        res = cur.fetchone()
        return None if res[0] is None else (res[0], res[1])

    # ------------------------------------------------------------------
    # Ingestion log
    # ------------------------------------------------------------------

    def get_ingestion_spans(self) -> pd.DataFrame:
        """Return all ingestion spans sorted by ``span_start``.

        Returns:
            DataFrame with columns
            ``[span_start, span_end, processed_at, row_count]``.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        return pd.read_sql_query(
            "SELECT span_start, span_end, processed_at, row_count "
            "FROM ingestion_log ORDER BY span_start",
            self.conn,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_metadata(
        self,
        intersection_id: Optional[str] = None,
        intersection_name: Optional[str] = None,
        timezone: str = "US/Mountain",
        controller_ip: Optional[str] = None,
        detection_type: Optional[str] = None,
        detection_ip: Optional[str] = None,
        major_road_route: Optional[str] = None,
        major_road_name: Optional[str] = None,
        minor_road_route: Optional[str] = None,
        minor_road_name: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        agency_id: Optional[str] = None,
    ) -> None:
        """Upsert intersection metadata (single-row table).

        Args:
            intersection_id:   Primary reference ID.
            intersection_name: Human-readable name.
            timezone:          IANA timezone string.
            controller_ip:     Controller IP for data retrieval.
            detection_type:    e.g. ``'Radar'``.
            detection_ip:      Detection system IP.
            major_road_route:  e.g. ``'US-95'``.
            major_road_name:   e.g. ``'Main St'``.
            minor_road_route:  e.g. ``'SR-44'``.
            minor_road_name:   e.g. ``'Oak Ave'``.
            latitude:          Decimal degrees.
            longitude:         Decimal degrees.
            agency_id:         Managing agency identifier.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        self.conn.cursor().execute("""
            INSERT OR REPLACE INTO metadata (
                lock_id, intersection_id, intersection_name, timezone,
                controller_ip, detection_type, detection_ip,
                major_road_route, major_road_name,
                minor_road_route, minor_road_name,
                latitude, longitude, agency_id
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            intersection_id, intersection_name, timezone,
            controller_ip, detection_type, detection_ip,
            major_road_route, major_road_name,
            minor_road_route, minor_road_name,
            latitude, longitude, agency_id,
        ))
        self.conn.commit()

    def get_metadata(self) -> Dict[str, Any]:
        """Return intersection metadata as a dict.

        Returns:
            Dict of metadata fields.  Returns ``{'timezone': 'US/Mountain'}``
            if the table does not yet exist or has no rows.
        """
        if not self.conn:
            raise RuntimeError("No active connection.")
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM metadata WHERE lock_id = 1")
        except sqlite3.OperationalError:
            return {"timezone": "US/Mountain"}
        row = cur.fetchone()
        if not row:
            return {"timezone": "US/Mountain"}
        return dict(zip([d[0] for d in cur.description], row))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        """Run ``VACUUM`` to reclaim space after large deletions."""
        if not self.conn:
            raise RuntimeError("No active connection.")
        self.conn.execute("VACUUM")
        print("Database vacuumed successfully")


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> None:
    """Initialise a database at ``db_path``.

    Args:
        db_path: Path to the SQLite database file.
    """
    with DatabaseManager(db_path) as m:
        m.init_db()


def import_config(csv_path: Path, db_path: Path) -> None:
    """Import configuration from a CSV file into the database.

    Args:
        csv_path: Path to ``int_cfg.csv``.
        db_path:  Path to the SQLite database file.
    """
    with DatabaseManager(db_path) as m:
        m.import_config(csv_path)
