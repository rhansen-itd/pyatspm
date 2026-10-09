# S-D2 design: detector health rules and threshold calibration

Scoped 2026-10-03, design only. Read this before building S-D2. Its input is
the S-D1 profile (`analysis/detector_activity.py`, `detector_activity_profile`).

## 1. Inputs

- **Profile**: S-D1 rows per `(date, window, detector)`, with windows `day`
  and `am` (`UDOT_AM_WINDOW`, 01:00–05:00). The PM peak window is the owner's
  to define; it's a `WD:` key, not a hardcoded default.
- **Roles**: `parse_detector_roles` on the config row in effect, used for
  `configured`, role, phase and `TM_*` movement (peer groups).
- **Bursts (new, needed for failsafe)**: the profile can't see same-timestamp
  onsets across channels. S-D2 adds a small pure helper over raw events,
  `onset_bursts(events, min_channels)`. It returns
  `ts, n_channels, channels, released_s` (the median time until those
  channels turn off). Censoring follows S-D1: a release that crosses a gap
  marker isn't measured.
- **Judgeable bins**: a rule judges a bin only when
  `observed_s / bin_s >= min_observed_share`. Calibrate it, starting at 0.9.
  Partial days are real: 315 on 2025-12-10 logged 58,492 s of 86,400.

## 2. Corpus and labels

| Site | logged days | gap markers | labels available |
|---|---|---|---|
| 201 | 11 (2026-03-18 → 10-01) | 4 | failsafe 2026-03-19 07:55 (LS 96–99 true); six configured-silent channels incl. 60/63/64 |
| 313 | 14 (2026-08-04 → 08-18) | 2 | none |
| 315 | 9 (2025-12-10 → 2026-01-12) | 3 | 42-channel bursts 2026-01-10/11; ~15 known-unconfigured active zones (S-D6) |
| 701 | 10 (2026-02-24 → 06-17) | 1 | round-hour file-boundary dumps (negative class for bursts) |

The labels are sparse. So for most rules, calibration produces a **review
list**: every flag at each candidate threshold, for the owner to mark real or
false. It doesn't produce an automatic FP rate. The rate is recorded once
it's reviewed, in the same style as the clock-step work ("0 of N
detector-days flagged").

## 3. Rules, statistic and calibration

The unit is the detector-day (or detector-AM window), on judgeable bins only.

| Rule | Statistic (profile columns) | Calibration grid / check |
|---|---|---|
| ConfiguredSilent | `configured & n_act == 0` | `min_observed_share` only. Must flag 201's six silent channels and nothing else on the corpus. |
| LowDetectorHits | `n_act < k` (day) | Plot the `n_act` distribution per role per site. Choose `k` below every healthy channel's minimum; review anything flagged. |
| UnconfiguredDetector | `~configured & n_act >= k` | Must flag 315's ~15 zones. Severity `info`; deliberate zones go on `WD_Ignore`. |
| StuckOn | `max(max_on_s, open_on_s) > X`; or AM bin with `occupancy >= 0.99 & n_act == 0` | X ∈ {5, 10, 15, 30, 60} min, **per role**. Presence zones hold through red, and 315 count loop 21 logged 437 s, so a role-free X will false-alarm. Report the flag count per X per role. |
| Chatter | `n_short / n_act`, `min_off_gap_s` | **Never an absolute share.** 315's count loops run 5–85 % short pulses depending on the loop (18: 3,599 of 5,253; 32: 163 of 3,502) over 2025-12-15 → 17. Compare the share to role/movement peers (ratio) or to the detector's own history. Use `min_off_gap_s` < 0.2 s as a second cue. Grid the peer ratio over {2, 3, 5}. |
| PeerRatio | **Superseded by §6** (group-level anomaly) | Kept only as the fallback note: outside lanes legitimately carry less (315's 52 runs at ~0.5× its peers 50/51), so a level threshold is the wrong tool. |
| BaselineDrift | **Superseded by §6** (entity-level anomaly, changepoint) | Needs history the corpus lacks (at most 2 same-weekday samples per site). |
| ControllerFault | codes 83–88, 91 | Absent everywhere. Pass-through, nothing to calibrate. |
| RecordCount | reuse `utils/quality.py` `compute_bin_quality` / `check_data_quality` | Parity only. |
| Failsafe | `onset_bursts` with `min_channels`, `released_s`, plus watchdog-zone calls | Grid `min_channels` ∈ {6, 8, 12, 16}. Positive: 201's 2026-03-19 07:55:12/15 pair (16-channel blocks). Negative: 701's round-hour dumps (7–10 channels at `HH:00:00/01`), excluded by a file-boundary test, not by count. Nightly 201/701 bursts → `info` inside a `WD:` reboot window when `released_s` is under the calibrated limit; measure that limit here (201 suggests 10–40 s). The 315 2026-01-10/11 bursts are outside any window and stay findings. |

## 4. Method

1. Profile every logged day at all four sites (`day` and `am`). The run is
   cheap: 3 days at 315 (921k events) take 0.8 s.
2. For each rule and each grid value, write the flagged rows to
   `docs/calibration/detector_health_<rule>.csv` (site, date, window,
   detector, role, value).
3. Put one summary table in the roadmap: flags per grid value per site.
   Choose the default as the loosest value that keeps every labelled
   positive and leaves a review list the owner can work through.
4. After the owner's review, record the FP rate per rule at the chosen
   default, and the table of calibrated defaults (the `WD:` override keys).

Steps 1–2 are execution-heavy and can be **delegated**: a sweep script plus a
verifier that checks every labelled positive appears. The rule functions,
the choice of default and the verdict stay with Opus.

## 5. Owner answers (2026-10-03)

1. **"Same unit" is an `int_cfg` grouping**, like other config-driven ideas
   in the roadmap: one row per detection unit, listing its channels (e.g.
   unit 1 → `[33, 34, …]`). The unit *type* is part of the config, keyed by
   BIU. For example, at 201 BIU 1 is a Currux video unit, BIU 2 a Currux Thunder
   and BIU 3 an Evo, with BIU 4 switched off (it was once in use, so older
   `int_cfg`s still assign detectors 49+); elsewhere BIUs 2, 3 and 4 may all be Evo.
   - **Settings can depend on the unit type** more often than on the
     intersection: reboot window, normal burst release time, and
     chatter/short-pulse norms (radar, video and loops behave differently).
     So calibrated defaults are a table keyed by unit type, with
     per-intersection `WD:` overrides on top.
   - **Populate it from Evo project files** where possible. ROADMAP.md's
     *EVO radar zones → controller detectors → lanes* item already parses
     `.iprj` zones to `OutputNumber` (= controller detector input), so each
     Evo sensor's channel list comes for free. Settle the key format with
     that item and the lane-geometry keys (ROADMAP.md's note on
     `int_cfg.csv` rows).
   - Until a site has unit rows, failsafe falls back to all channels at the
     intersection.
2. **Severity:** each finding carries a label so reports can filter
   (S-D4's `--min-severity`). Proposed default, pending owner confirmation:
   `info` = expected or known (scheduled reboot, ignored zone); `low` =
   suspicious, worth a look (suspected false watchdog call, a statistical
   anomaly); `high` = the detector is probably failed and affecting
   operation (stuck-on, configured-silent, confirmed failsafe).
3. **Windows: two different purposes, derived differently.**
   - **The PM window** is a *peak-traffic* period. Derive it per
     intersection from history, e.g. the weekday peak 60 minutes of volume,
     as a pure function over a 15-minute series (§6).
   - **The AM window (UDOT's 01:00–05:00)** is the opposite: a period with
     *essentially no traffic*. A detector held on, or a phase maxing out,
     there is suspicious because nothing should be calling it. If it's
     derived at all, it's the *quietest* sustained window from history,
     a separate function, never the peak derivation. The 01:00–05:00
     default stays until history says otherwise, and it must not overlap the
     nightly reboot window (§3 Failsafe).
   - A `WD:` key overrides either.
4. **`min_observed_share` = 0.9** (owner, 2026-10-03).
5. **Add configuration roles.** Every intersection has loops that aren't
   stop bar, occupancy or advance (315's 37/53 dilemma-zone loops, its
   P9 count-only loops), and they need handling, not suppressing. New roles
   are added to `detector_roles.py` as new `Det_P{N}_{Role}` families.
   Candidates: `dilemma`, and a generic monitor-only role for channels
   whose health matters but that feed no measure. Any role row makes a
   channel `configured` in S-D1, so new roles flow through with no profile
   change. Names are the owner's.

## 6. Statistical anomaly layer (owner direction, 2026-10-03)

The rules in §3 are adapted from UDOT's Watchdog. The owner wants broader,
statistics-based anomaly detection too, by importing a package rather than
hand-building it, the way `volume_explorer` and Inrix use **`traffic_anomaly`**
(2.5.4, MIT, PyPI; installed in `~/Inrix/.venv`, wrapped by
`~/Inrix/src/inrix_tools/changepoint.py`).

**What it gives, mapped to detector health:**

| `traffic_anomaly` | Use here | Replaces |
|---|---|---|
| `decompose(entity=detector, freq_minutes=15)`: rolling median + day/week seasonality + residual | A per-detector expected count for every 15 minutes | hand-built same-weekday medians |
| `anomaly(...)` entity level (`GEH=True` for counts) | A detector departs from its own history: sudden failure, partial detection | BaselineDrift |
| `anomaly(...)` group level (`group_grouping_columns`, `MAD=True`) | A detector departs from its peers. Groups: `TM_*` movement, phase + role, **unit** (§5.1), so a whole-unit fault shows as several group anomalies at once | PeerRatio |
| `changepoint(...)` | A persistent step: re-aimed zone, degraded loop, config change | — |

**How it splits with §3:** the deterministic rules stay for *physical
states* that need no statistics: stuck-on, configured-silent, chatter,
unconfigured, failsafe bursts and controller faults. The anomaly layer
covers *departures from normal*. Both write the same findings schema, with
`rule` = `EntityAnomaly` / `GroupAnomaly` / `Changepoint`.

**What it needs from us:**
- **A regular 15-minute series per detector** (count, on-time). S-D1's
  binning is general, but its named-window mode localizes edges, and on a
  fall-back day that drops the repeated hour. So add a grid mode instead:
  UTC-regular bins with a local label. It reuses the same censoring and
  observed-time logic, and bins with low `observed_s` are dropped rather
  than read as zero, because a log gap isn't a silent detector.
- **History.** `decompose` defaults to a 7-day rolling window, needs at
  least 480 samples (96 × 5) and drops the first 7 days. The corpus has
  9–14 *discontinuous* logged days per site, so only its static-median mode
  (`rolling_window_enable=False`) runs on it, and the weekly component is
  unidentifiable. **Calibrating the anomaly layer needs a longer continuous
  pull.** That's the blocker, the same one BaselineDrift had.
- **Dependency.** It requires `ibis-framework[duckdb]` (11.x). Neither is
  in the pyatspm venv. Add it as an optional extra (`anomaly =
  ["traffic_anomaly>=2.5,<3"]`), imported lazily inside a thin adapter
  (`analysis/detector_anomaly.py`), like Inrix's. The adapter only maps
  columns and defaults and copies no upstream code. It computes in-memory
  on DataFrames with no file I/O, so it can sit in the functional core.
  Without the extra installed, the anomaly rules are skipped with a clear
  message, and the §3 rules still run.

**Build order:** grid mode for S-D1 (Opus, small) → adapter + goldens on
synthetic series (Opus writes the tests; the wiring is Gemini-eligible) →
calibration once a long pull exists (Opus verdict).
