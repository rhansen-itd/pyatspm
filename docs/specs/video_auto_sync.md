# Spec: automatic video/data sync from signal lamps

Written by Opus, 2026-10-01, for a Gemini run via `delegate`. Source: the ROADMAP item
*Automatic video/data sync from signal lamps* (`docs/ROADMAP.md`). This spec is the
contract. Where it and the roadmap disagree, this spec wins.

**What it does.** Today a user aligns a recorded clip to the controller data by eye with
`video-locate-phase-change`. This feature instead measures the signal lamps in the clip,
finds the clip start time that best matches the database's phase states over the whole
clip, and prints a corrected `--start`. If the evidence is weak, it refuses to print one.

## 0. Rules for this run

**Do not modify any of these files.** If a test in them looks wrong, stop and write
the reason in the report:
- `tests/analysis/test_video_sync.py` (golden tests for the math core, using the real
  201 clips)
- `tests/data/test_video_lamp_shapes.py` (tests for the shape schema)
- `tests/video/test_video_sync.py` (end-to-end shell tests on a synthetic clip, plus a
  CLI parser test)
- `tests/analysis/fixtures/video_sync_201/*` (the fixture data)
- every other existing test file

**Files you may create:**
- `src/atspm/analysis/video_sync.py`
- `src/atspm/video/sync.py`
- `tests/video/test_video_sync_cli.py`
- `tests/video/test_overlay_lamp.py`
- `docs/specs/video_auto_sync_REPORT.md`

**Files you may edit:**
- `src/atspm/data/video.py`
- `src/atspm/video/processor.py`
- `src/atspm/video/overlay.py`
- `src/atspm/video/calibrate.py`
- `src/atspm/video/__init__.py`
- `src/atspm/cli.py`
- `docs/PENDING_DOC_CHANGES.md` (append bullets only)

Touch nothing else. That includes `README.md`, every other file in `docs/`,
`ROADMAP.md` and `CLAUDE.md`.

**Acceptance check:**
```
.venv/bin/python -m pytest tests/ -q
```
The whole suite must pass. Before this work it was 408 tests once the new modules
existed.

**Stop and ask (write the question in the report, then end the run) if:**
- a frozen test can only pass by special-casing the test's data;
- a golden tolerance is missed by any margin;
- the algorithm in Part A would need a step that isn't in this spec, such as a new
  threshold, a different score, or extra smoothing, to pass;
- passing would mean changing `analysis/video.py`, `analysis/phases.py` or
  `data/reader.py`.

**Project rules (CLAUDE.md):**
- **Functional core:** `analysis/` takes DataFrames and arrays in and returns them.
  No I/O, no SQL, no cv2.
- **Gap marker:** `event_code = -1` must never be crossed.
- **Vectorize:** no `iterrows()`, and no Python loop over frames in the core. A loop
  over lamps (one to three items) is fine.
- **Docstrings:** Google style.
- **Banned words:** never write "legacy" or "orphan" in new code or comments.
- **Commits:** commit when done, with a message body that ends with the run's
  attribution lines.

**Report:** write `docs/specs/video_auto_sync_REPORT.md`, at most about 40 lines. It
must give:
- the files changed;
- the final test count;
- any stop-and-ask item;
- the measured `align_clip` wall time per 10-minute fixture clip;
- every decision you made that this spec didn't dictate.

---

## Part A. Functional core: `src/atspm/analysis/video_sync.py`

### A.1 Public API (the tests import exactly these)

```python
INDICATIONS = ("green", "yellow", "red")
DEBOUNCE_SECONDS = 0.25

@dataclass(frozen=True)
class LampSeries:
    kind: str          # "phase" | "overlap"  (from data.video.resolve_stopbar_target)
    number: int
    indication: str    # one of INDICATIONS
    bgr: np.ndarray    # (n_frames, 3) float: mean B, G, R inside the lamp ROI, per frame

@dataclass(frozen=True)
class SyncResult:
    accepted: bool
    reason: Optional[str]      # None iff accepted; short human-readable refusal reason
    start_epoch: float         # epoch of the frame whose frame time is 0 (see A.2)
    mid_start_epoch: float     # start that makes the clip-midpoint frame exact
    slip_s_per_10min: float
    score: float               # peak combined score
    runner_up_score: float
    agreement: float           # fraction of compared frame-lamp pairs where lamp == DB, at the peak
    n_frames_compared: int
    n_edges_used: int
    gap_clamped: bool

def lamp_contrast(bgr, indication) -> np.ndarray
def otsu_threshold(values) -> float
def debounce_frames(fps: float) -> int
def lamp_on_series(bgr, indication, fps) -> tuple[np.ndarray, float]   # (bool on-series, threshold)
def align_clip(frame_times_s, lamps, events_df, start_guess_epoch, *,
               search_s=30.0, fps=None, min_score=0.6, min_margin=0.2,
               min_edges=6, min_coverage=0.5) -> SyncResult
```

### A.2 Time convention

The epoch of frame `i` is `start_epoch + frame_times_s[i]`. This is the same
convention `render_overlay` uses (`start_epoch + POS_MSEC/1000`). Never assume uniform
frame spacing: the tests remove 50 frames and keep the surviving frame times.

### A.3 Lamp measurement → on/off series

1. **`lamp_contrast`**: input is BGR column order.

   | Indication | Contrast |
   |---|---|
   | green | `G − R` |
   | red | `R − G` |
   | yellow | `(R + G)/2 − B` |

   Any other indication raises `ValueError`.
2. **`otsu_threshold`**:
   - Drop non-finite values, then build a 256-bin `np.histogram`.
   - Choose the split that maximises between-class variance.
   - Return the right **edge** of the last bin in the lower class.
   - If the input is empty or constant, return `nan`.
   - Never use a hardcoded threshold.
3. **`debounce_frames(fps)`**: `max(1, ceil(DEBOUNCE_SECONDS * fps))`. That gives 3 at
   10 fps, 4 at 15, 8 at 30 and 1 at 2.
4. **`lamp_on_series`**:
   - Compute `on = contrast > otsu_threshold(contrast)`. If the threshold is `nan`,
     the series is all `False`.
   - Debounce: flip every **interior** run of identical values shorter than
     `debounce_frames(fps)` frames. A run that touches either end of the series is
     never flipped.
   - Repeat until nothing changes, at most 3 passes.
   - Vectorize with run-length encoding through `np.diff`/`np.flatnonzero`.

### A.4 DB state per frame (reuse; write no new interval logic)

- Use `phase_status_at_timestamps` for `kind == "phase"` and
  `overlap_status_at_timestamps` for `kind == "overlap"`, both from
  `atspm.analysis.video`.
- A lamp counts as **on** in the DB when the status letter matches its indication:
  `green` matches `'G'`, `yellow` matches `'Y'`, `red` matches `'R'`.
- Frames whose status is `'na'` are **excluded** from every count.

### A.5 Gap-marker clamp

1. Let the window be `W = [guess − search_s, guess + (t[-1] − t[0]) + search_s]`.
2. If no `event_code == -1` row falls strictly inside `W`, use the events unchanged.
   In that case `gap_clamped` is `False`.
3. Otherwise:
   - Split `W` at those markers and keep the longest piece.
   - Restrict `events_df` to the rows between the nearest marker at or before that
     piece and the nearest marker at or after it, markers included.
   - Set `gap_clamped = True`.
4. The status functions already return `'na'` for anything they can't resolve, so the
   frames outside the kept segment drop out on their own.

### A.6 Coarse search

1. **Frame rate:** `fps` defaults to `1 / median(diff(frame_times_s))`.
2. **Candidates:** `step = 0.5 / fps`. The candidate starts are
   `guess + arange(-search_s, search_s + step/2, step)`.
3. **Vectorize:** build one flattened query, `candidates[:, None] + t[None, :]`, and
   call the status function **once per lamp**. At 10 fps a 10-minute clip gives about
   1,200 candidates × 6,000 frames.
4. **Per-lamp score:** at each candidate, the score is the phi coefficient (Matthews
   correlation) between the lamp's on-series and the DB's on-series, over the valid
   frames only. If the denominator is 0, the score is 0, never `nan`.
5. **Combined score:** the mean of the per-lamp scores.
6. **Peak:** `k = argmax(score)`.
7. **Runner-up:** the highest **local maximum** that lies more than 2.0 s from the
   peak. A local maximum is a candidate equal to the maximum of the scores within
   ±2.0 s of it. If there is none, the runner-up is 0.0.
   - This is not simply the best score outside ±2 s. The main lobe falls away slowly,
     so that value would just be the lobe's shoulder.
8. **Agreement:** the fraction of valid frame-lamp pairs where the lamp matches the
   DB, at the peak candidate.
9. **`n_frames_compared`:** the number of valid frame-lamp pairs at the peak, divided
   by the number of lamps (integer).

### A.7 Sub-frame refinement and slip

Let `s0` be the coarse peak.

1. **Observed edges:** for each lamp, every index `k` where its on-series changes.
   - Edge time: the midpoint between the two frames, `(t[k] + t[k+1]) / 2`.
   - Polarity: on if the lamp switches on, off if it switches off.
2. **Matching DB edge:**
   - Sample the DB on-state for that lamp at `s0 + t_edge + g`, where `g` runs from
     −1.0 to +1.0 s in 0.01 s steps.
   - Use only adjacent sample pairs that are both valid.
   - Pick the change of the same polarity whose `g` is nearest to 0.
   - The residual `r` is the midpoint of that sample pair.
   - Skip the edge if there is no such change.
3. **Vectorize** across edges as a 2-D array with shape `(n_edges, n_grid)`.
4. **Pool** the residuals from all lamps.
5. **Inliers:** keep the residuals with `|r − median(r)| < 0.3`.
6. **Line fit:** least-squares `r = a + b·t_edge` over the inliers.
   - With fewer than 2 inliers, set `b = 0` and `a` to the single residual, or 0 if
     there are none.
7. **Outputs:**
   - `start_epoch = s0 + a`
   - `slip_s_per_10min = 600 · b`
   - `mid_start_epoch = start_epoch + b · (t[0] + (t[-1] − t[0]) / 2)`
   - `n_edges_used` = the number of inliers

### A.8 Accept or refuse

Check the conditions below in this order. The first one that fails sets `reason`, and
`accepted` is then `False`.

| # | Refuse when |
|---|---|
| 1 | `n_frames_compared / len(t) < min_coverage` |
| 2 | `score < min_score` |
| 3 | `score − runner_up < min_margin` |
| 4 | `n_edges_used < min_edges` |

- If `events_df` is empty, or the series are degenerate, the function must **return**
  a refused result. It must not raise.
- A refused result still fills in its numeric fields with the best estimate, so the
  CLI can print them.

**Performance:** `align_clip` on one 10-minute, 10 fps fixture clip with a single lamp
should take 3 s or less. Report the time you measure.

---

## Part B. Shape schema: `src/atspm/data/video.py`

1. **New constant and column:**
   - Add `LAMP_INDICATIONS = ("green", "yellow", "red")`.
   - Append `"indication"` to `_CSV_FIELDS`.
2. **New shape type `lamp`:**
   - `points` holds one point (sampled as a disc) or a polygon.
   - `phase` is resolved through `resolve_stopbar_target`, so overlaps such as `OLB`
     work.
   - `indication` is read with `row.get("indication")`, stripped and lower-cased.
3. **Only lamp rows get an `"indication"` key** in the loaded dict. The existing test
   `test_save_then_load_is_identity` requires loop and stopbar dicts to stay exactly
   as they are now.
4. **`load` validation** for lamp rows. Each failure raises `ValueError`, with a
   message that contains the given word:

   | Problem | Word in the message |
   |---|---|
   | indication missing or not in `LAMP_INDICATIONS` | `indication` |
   | phase missing | `phase` |
   | invalid phase | `range` (the message `resolve_stopbar_target` already raises) |

5. **`save`** writes the `indication` column: the value for lamps, an empty string
   for everything else.
6. **Old files still load.** Six-column files without the column load unchanged:
   `load` already takes the field names from the file's own header row.
7. **Accessors:**
   - `relevant_phases()` and `relevant_overlaps()` include lamp shapes as well as
     stopbars.
   - Add `lamp_shapes()`, which returns the lamp dicts in file order.

---

## Part C. Shell: `src/atspm/video/sync.py`

```python
LAMP_DISC_RADIUS = 3

@dataclass
class LampMeasurement:
    frame_times_s: np.ndarray   # starts at 0.0, strictly increasing
    lamps: List[LampSeries]     # one per shape_config.lamp_shapes(), same order
    fps: float
    timing_source: str          # "pts" | "fps"

def measure_lamps(video_path, shape_config) -> LampMeasurement
def sync_video(db_path, shape_config, video_path, start_guess: datetime,
               search_s: float = 30.0) -> SyncResult
```

### `measure_lamps`

1. If there are no lamp shapes, raise `ValueError` with a message that contains
   "lamp". Check this **before** opening heavy resources.
2. Open the video with `processor._open_capture` and call
   `shape_config.validate_resolution`. Its "resolution" message is what the test
   matches.
3. Build one boolean mask per lamp:
   - a single point: `cv2.circle` with radius `LAMP_DISC_RADIUS`, filled;
   - a polygon: `cv2.fillPoly`.
4. Decode every frame. After each `read()`, record `CAP_PROP_POS_MSEC`.
5. For each frame, take the mean BGR inside each mask. A per-frame loop is
   unavoidable here (this is decode). Keep the work inside it to a masked mean.
6. **Frame times:**
   - Use PTS when `processor._pts_usable` accepts the readings; otherwise use
     `index / fps`.
   - Subtract the first value, so the first frame is at 0.0.
7. Get each lamp's `(kind, number)` from `resolve_stopbar_target(shape["phase"])`.

### `sync_video`

1. Localize a naive `start_guess` with
   `localize_naive(start_guess, _resolve_timezone(db_path))`, the same way
   `render_overlay` does.
2. Fetch events with `get_events_with_cycles_df`:
   - **Window:**
     `[guess − search_s − 10 min, guess + clip_duration + search_s + 10 min]`.
   - **Codes:** `-1` plus every phase and overlap code from `processor._PHASE_CODES`
     and `_OVERLAP_CODES`.
3. Call `align_clip(..., search_s=search_s, fps=measurement.fps)` and return its
   result unchanged.

---

## Part D. CLI: `atspm video-sync` (`src/atspm/cli.py`)

### Parser

- Add a parser in `_build_parser`, modelled on `_add_video_locate_phase_change_parser`.
- **Target group:** a required, mutually exclusive `--target FOLDER` / `--targetid ID`.
  There is **no `--all`**: one video means one camera, the same exception the other
  video commands make. The frozen test expects `--all` to exit.
- **Arguments:**

  | Argument | Required | Notes |
  |---|---|---|
  | `--camera NAME` | yes | |
  | `--video PATH` | yes | resolved with `_resolve_video_path` |
  | `--start-guess ISO8601` | yes | stored as `args.start_guess`, a string; parsed in the handler |
  | `--search SECONDS` | no | float, default 30.0 |
  | `--timezone TZ` | no | |
  | `--verbose` | no | |

- Add a line for it to the module docstring's command list.

### Handler `handle_video_sync`

1. **Setup:** resolve the target, database, shape path and video path exactly as
   `handle_video_overlay` does. That includes its error messages.
2. **Missing lamp shapes:** if the shape config has none, `_die` with a hint to add
   them, for example
   `type=lamp, indication=green, a point on the lit lamp; see 'atspm video-calibrate-shapes'`.
3. **Parse the guess:** naive input is read as intersection-local time.
4. **Accepted result:** print the following, then exit 0.
   - the corrected start in local ISO-8601 with milliseconds, e.g.
     `2026-10-01T12:24:57.550-06:00`;
   - the delta from the guess, in seconds;
   - the mid-clip start;
   - the slip in s per 10 min;
   - the score, runner-up, agreement, edges used, and whether a gap clamp happened;
   - a ready-to-paste line:
     `atspm video-overlay --target <folder> --camera <cam> --video <video> --start <corrected>`.
5. **Refused result:** print the reason and the diagnostics, then the fallback
   command, then `sys.exit(2)`.
   - The fallback command is
     `atspm video-locate-phase-change --target <folder> --camera <cam> --video <video> --start <best estimate>`
     plus any flags that command requires. Read its parser and fill those in.
   - Do not print the best estimate as if it were a result.
6. **Tests:** add the handler tests to `tests/video/test_video_sync_cli.py`. Monkeypatch
   `sync_video` and the target or path helpers. Cover:
   - the output on the accepted path;
   - exit code 2 and the fallback text on the refused path.

---

## Part E. Overlay: draw each lamp's DB state

1. **`processor._apply_shapes`:** for a `lamp` shape, look up `(kind, num)` in
   `status_lookup` the same way the stopbar branch does, then call
   `draw_shape_overlay`.
   - `relevant_phases` now includes lamps (Part B), so the status arrays are already
     fetched.
2. **`overlay.draw_shape_overlay`:** add a `lamp` branch that draws a **filled dot
   with radius 4**, centred 10 px above the lamp:
   - centre `(x, y − 10)`, where `(x, y)` is the shape's first point, or the
     polygon's centroid;
   - colour from the stopbar colour map for the status (G/Y/R/na);
   - a 1 px black outline.

   The dot shows the **database's** state beside the real lamp, so a bad sync is
   visible at a glance.
3. **Tests:** add `tests/video/test_overlay_lamp.py`. Draw on a blank frame and assert
   the dot's pixel colour for `'G'`, `'R'` and `'na'`, and that the pixels at the
   lamp's own position are left untouched.

---

## Part F. Calibration GUI dot mode (`src/atspm/video/calibrate.py`)

1. **New key `'d'`:** switches to lamp mode. A **single click** completes a lamp
   shape with one point. `'d'` is unused today; the keys in use are
   `b c e g i l n p q r s u w`.
2. **Phase and indication:**
   - The phase comes from the existing `'p'` value, just as stopbars use it.
   - After the click, ask for the indication with a Tk dialog that accepts only
     green, yellow or red. Cancelling discards the shape.
3. **`_draw_shape_preview`:** draw a lamp as a radius-3 circle outline in its
   configured colour.
4. **Existing behaviour:**
   - In edit mode, lamps must cycle with `'n'`/`'b'` and drag like the other types.
     Use the existing point-drag path; a single point is enough.
   - Update the instruction text.
   - Don't restructure the state machine.
5. **Testing:** the GUI can't be tested headlessly. Keep the change small, and check
   it by reading it through. Say so in the report.

---

## Part G. Docs

- Append these bullets to `docs/PENDING_DOC_CHANGES.md`, following its format:
  - `[src/atspm/cli.py]` new `video-sync` subcommand
  - `[src/atspm/data/video.py]` `lamp` shape type and `indication` CSV column
- If you add exports to `src/atspm/video/__init__.py`, add one bullet for those too.
- Nothing else in `docs/`.

## Out of scope (do not do)

- **ntcip's overlay:** checking that it skips `lamp` rows is a different repo.
- **Batch sync from the collector's `manifest.json`:** that's a follow-up.
- **Dusk and night handling:** unproven, and needs new data first.
- **Excluding flash and preemption windows:** a follow-up. Note it in the report if
  you see a natural seam for it.
- **ROADMAP edits:** Opus will make them.
