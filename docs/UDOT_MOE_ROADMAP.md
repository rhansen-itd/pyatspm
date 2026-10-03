# UDOT ATSPM Measure Parity — Scope

Deliberately separate from [ROADMAP.md](ROADMAP.md) so this track doesn't tangle with that file's active work. Scoped 2026-09-30 against the live codebase, the 4 intersection DBs in `intersections/`, and UDOT ATSPM v5 source (`OpenSourceTransportation/Atspm`; the v4.x `udotdevelopment/ATSPM` repo is deprecated). Each session is sized to be picked up cold. **Routing:** *Opus* = functional-core math, vectorization, gap-marker logic, threshold calibration, or a verdict on results. *Gemini-eligible* = executes against an Opus-written spec plus golden tests, via `delegate`. Plotting counts as Gemini-eligible only when the spec includes Opus-written tests, since otherwise it falls under the forbidden list in `AGENTS.md` (see the *Hybrid Claude + Gemini workflow* in `~/.claude/CLAUDE.md`).

## 1. Gap analysis

UDOT v5 measures (`Application/Business/*`, `ReportApi/ReportServices/*`), mapped to pyATSPM:

| UDOT measure | pyATSPM today | Status |
|---|---|---|
| Purdue Phase Termination | `plot-termination` | **Have** |
| Purdue Coordination Diagram | `plot-coordination` (detector arrivals on the split diagram) | **Have** — spot-check parity in S-M1 |
| Arrival on Green (in PCD / aggregation) | `aog` | **Have** |
| Turning Movement Counts | `counts` (`TM_*`) | **Have** |
| Split Monitor | `splits` (tables only) | **Partial**: no plot, no programmed-split overlay → S-M2 |
| **Watchdog** (detector/phase diagnostics) | `discrepancies` covers co-located pairs only | **Missing** → track D (priority) |
| Timing and Actuation | — | **Missing** → S-D5 |
| Purdue Split Failure | `split-failures` (`SplitFailureEngine`, union/mean lane aggregation) | **Have** (2026-10-02), on `Det_P{N}_Occupancy` presence zones |
| Arrivals on Red, Approach Delay | — | **Missing** → S-M3 |
| Yellow and Red Actuations | — | **Missing** → S-M4 |
| Pedestrian Delay, Wait Time | ped *counts* only | **Missing** → S-M5 |
| Preempt Detail / Service / Service Request | — | **Missing** → S-M6 |
| Green Time Utilization | — | **Missing** → S-M7 |
| Approach Volume (directional, D/K-factor) | partial via `counts` | **Partial** → S-M8 |
| Left Turn Gap Analysis | — | **Missing** → S-M9 |
| Left Turn Gap Report (protected-left feasibility) | — | **Missing** → S-M10 (last; depends on M1, M5, M8, M9) |
| Time-Space Diagram, Link Pivot | — | **Deferred**: corridor-level, needs a decision (§5) |
| Approach Speed | — | **Deferred**: needs a speed feed that `.datZ` doesn't carry (§5) |
| Transit Signal Priority, Priority Detail/Summary | — | **Deferred**: codes 112–119 are absent from every corpus DB |
| Ramp Metering | — | **Not applicable**: codes 1xxx, and there are no ramp meters |
| Aggregation tables (14 `AggregationType`s) | computed on the fly | **Not applicable**: with one DB per intersection, on-the-fly engines are fast enough |

### Is detector diagnostics a UDOT measure?

Yes, but it's a nightly scan rather than a chart: the **Watchdog** (`Infrastructure/Services/WatchDogServices/`). Its `WatchDogIssueTypes` for signalized locations are:

- `RecordCount`: total events for the day below `MinimumRecords`.
- `LowDetectorHits`: per-detector actuations in the PM peak (`PmPeakStartHour`–`PmPeakEndHour`) below `LowHitThreshold`.
- `StuckPed`: ped events (codes 21/23) per phase in the AM window above `MaximumPedestrianEvents`.
- `ForceOffThreshold` and `MaxOutThreshold`: the AM-window share of terminations above `PercentThreshold`, requiring at least `MinPhaseTerminations`. An overnight max-out is the classic symptom of a stuck-on detector.
- `UnconfiguredApproach`: a phase shows green but has no approach configured.
- `UnconfiguredDetector`: a channel actuates but isn't configured.

Types 8–11 are ramp-only. The Watchdog also keeps an ignore list (`WatchDogIgnoreEvent`) so known issues stop re-alerting. UDOT's visual detector check is the **Timing and Actuation** chart.

What UDOT does *not* do, and this scope adds: per-detector **stuck-on**, **chatter** and **baseline-drift** checks, and consuming the controller's own fault codes (83–88, 91–92). pyATSPM also already goes further than UDOT with co-located pair discrepancies (`analysis/detectors.py`).

## 2. Corpus facts that shape the scope

Measured 2026-09-30 on the 201, 313, 315 and 701 DBs:

- **Controller detector-fault codes 83–88 appear nowhere.** Code 92 (ped detector restored) appears only as a state dump at log restart: one per ped channel, at the same timestamps as the 131–149 plan dump. So diagnostics **must be inferred from actuation behavior**. Consume 83–88 and 91 if they ever appear, but never depend on them.
- **Available everywhere:** 81/82 (detector on/off), 43/44 (phase call registered/dropped, which wait time needs), 4/5/6 (termination), 131–149 (plan, cycle and splits, which the programmed-split overlay needs).
- **Available at some sites only:** 21–23 and 45 (ped walk and call; 315 and 701 only), 89/90 (ped detector on/off; 315 and 701), and 102–111 (preempt; 315 only, 24 events). 150–152 (coord state and yield point) appear at 315 only.
- **Absent everywhere:** 112–119 (TSP) and 1xxx (ramp).
- **Detector roles are per-channel, not shared.** At 201, P2 uses channel 49 for arrival, 60 for stop bar, 46 for occupancy and 63 for count (`TM_EBT`), so each radar zone is its own virtual channel. The existing `Det_P{N}_{Arrival,Stop_Bar,Occupancy,Pairs}` + `TM_*` families already give UDOT's "detection type" split: Arrival ≈ Advanced Count, Stop_Bar ≈ Stop Bar Presence, TM_* ≈ Lane-by-lane Count. **No new config category is needed** for the measures in track M, except lane/movement typing for the left-turn items.
- **Config-key drift, a live bug:** every corpus `int_cfg` uses `Det_P{N}_Stop_Bar`. `critical.py` accepts both spellings (`_STOPBAR_KEY_RE`), but `FlowRateEngine._resolve_stopbar_detectors` (`data/flow.py:355`) matches only `_Stopbar`, so `flow` finds no stop-bar detectors on real config. `Det_P{N}_Occupancy` is read only by the coordination plot's marker legend. S-D0 fixes the drift. *(Corrected 2026-10-02: the flow drift had already been fixed by `9317f4e`, which routed flow through `_parse_stopbar_sets`. S-D0 then replaced every per-module parser with the role table; see S-D0.)*
- **`WD_Sensor*` are watchdog zones (owner, 2026-09-30).** These are detection zones drawn where no vehicle can ever call them, so a call means the sensor unit is in failsafe. Logic statements (153/154) then switched the controller to an alternate detection plan. **They proved unreliable in the field.** Sometimes a watchdog zone stayed called while the unit was healthy and the other zones were calling normally; other times it worked as intended. So a watchdog-zone call is *evidence*, not a verdict (see the S-D2 rule). At 201 they are channels 56/57/58, configured only in the period starting 2020-01-01 and blank from 2026-06-01. They never actuate in 201's corpus window. **Logic statements 96–99 at 201 were the failsafe logic** (owner). They were disabled after the owner saw them misbehave, and their last event in the corpus is 2026-03-21. So controller-side failsafe corroboration exists only as history, for that 4-day window.
- **That window is labeled failsafe data, and it reveals the signature.** At 2026-03-19 07:55:12, detector channels 1–16 all switch on in the same decisecond, together with LS 96/97 true. At 07:55:15, channels 17–32 do the same, together with LS 98/99. Each statement pair appears to watch one 16-channel block. Simultaneous phase 2/6 max-outs start at 07:55:03. The watchdog channels 56–58 log nothing, even though the logic fired.
- **Corpus scan for that signature** (≥ 6 distinct channels turning on at one identical timestamp):
  - **201 and 701** both show 14–32-channel bursts at recurring times: daily 05:59–06:00 UTC (local midnight MDT) and 07:55 UTC (01:55 MDT), months apart. That looks like a *scheduled* sensor-side event, such as a nightly reboot or recalibration, rather than random faults. It may also explain the "watchdog zones stayed called on a healthy unit" observation.
  - **315** has 42-channel bursts on 2026-01-10/11.
  - **701** also has many 7–10-channel bursts at exact round hours (`HH:00:00/01`). Those sit on file boundaries and are more likely state dumps at the start of a log file than failsafe, so the rule must exclude them.
  - **313** shows only small 7–8-channel bursts.

  This is the starting point for S-D2's calibration, not a conclusion.

## 3. Track D — Detector diagnostics (priority)

Sessions D0 → D1 → D2 run in order. D3, D4 and D5 can run in parallel after D2 (D5 needs only D0).

- **S-D0: one detector-role parser, and the flow key fix.** *Gemini-eligible (Opus writes the tests).* Add a pure `parse_detector_roles(config) -> DataFrame[detector, phase, role, movement]` in `analysis/` covering `Det_P{N}_{Arrival, Stopbar|Stop_Bar, Occupancy, Pairs}` and `TM_*`. Point `data/aog.py`, `data/flow.py` and `analysis/critical.py`'s `_parse_stopbar_sets` at it, which fixes the `Stop_Bar` bug. Golden test: 201's real config row gives the expected role table, and `flow` on 201 is no longer empty. Small.

  **Built 2026-10-02 (branch `feat/detector-roles`).**
  - **Core (Opus):** `analysis/detector_roles.py`, with 34 golden tests on 201's and 315's real config rows.
    - `parse_detector_roles(config)` returns `detector, phase, role, movement, partner, key`, with roles `arrival`, `stop_bar` (both spellings, merged), `occupancy`, `pairs` (one row per member, the other in `partner`), `tm` and `watchdog` (`WD_Sensor*` only, so later `WD_` settings keys aren't mistaken for zones).
    - `movement` is the single `TM_*` label holding the detector. It is NaN when none or several hold it.
    - No inference.
  - **`detector_sets(roles, role)`:** returns `{phase: frozenset}`.
  - **Call-site migration (Gemini, `delegate`, 884 s, check green on the first run):**
    - `aog`, `flow`, `split_failures`, `optimizer` and `critical.movement_phase_map` now use the role table.
    - `reader.get_det_config` (the coordination plot) builds `"P{N} Arrival|Stop Bar|Occupancy"` from it, so the two spellings no longer give two traces and `Pairs` keys no longer leak in.
    - `critical`'s private parsers are deleted.
    - Pinned by `tests/data/test_detector_role_callsites.py`. That file includes a source scan that fails if `Det_P` key matching reappears outside `detector_roles.py`. `manager._parse_detector_pairs` is the exception: it keeps pair order.
  - **Verified on the 315 DB via the CLI:**
    - `flow` P2/P6 written.
    - `split-failures` lanes are exactly 34–36 and 50–52.
    - `plot-coordination` has one Ar/Oc/St trace per configured phase.
  - **The "flow is empty" premise was stale:** `flow` on 201 and 315 already found detectors on `main`. It's now a regression test.
  - **Not done:** 201's config is unchanged (the owner decides on channel 41 and 60/63/64).
- **S-D1: per-detector activity profile (functional core).** *Opus.* A pure function over events, returning per `(detector, bin)`: actuation count, total on-time and occupancy, max and p95 on-duration, count of on-pulses ≤ 0.1 s, min off-gap, whether it was on at the bin end, and whether it's configured (joined from S-D0). Reuse `_reconstruct_intervals` (`analysis/detectors.py:22`) rather than re-pairing, and vectorize it if it isn't already. Gap markers are strict: an on-interval with a `-1` between on and off is **censored** (reported as open, never measured), which also covers clock-step markers. Bin by local day plus a configurable sub-day window (UDOT's AM 1–5 and PM peak). Goldens: synthetic events for stuck-on, chatter, silence, an interval censored across a gap, and a DST-day bin.
- **S-D2: health rules plus threshold calibration.** *Opus: a design decision and a verdict.* Rules to evaluate against the D1 profile:
  - Parity with the UDOT Watchdog: `LowDetectorHits`, `UnconfiguredDetector`, `RecordCount` (reuse `check_data_quality` / `utils/quality.py`), and **configured-but-silent**.
  - Beyond UDOT:
    - **stuck-on**: an on-duration over *X* minutes, or on across an entire window.
    - **chatter**: the share of short pulses, or a count ratio against movement peers.
    - **baseline drift**: count against the same detector's trailing same-weekday median, which catches degraded or partial detection that absolute thresholds miss.
    - **peer ratio**: detectors in the same `TM_*` movement, or the same phase and role, diverging from each other.
    - **controller faults**: 83–88 and 91 when present.
    - **failsafe**: watchdog zones (`WD_Sensor*`) are corroborated, never trusted alone, because the owner saw them stay called on healthy units. A watchdog-zone call is classified by what the *same unit's* other zones do in the same window:
      - **failsafe confirmed** (high severity): the other zones also show the failsafe signature. The leading candidate (§2) is a **block-on burst**: many channels, likely a contiguous 16-channel block, turning on at one identical timestamp and staying on. The rule then reports how long the failsafe lasted (burst until the channels release) and which phases maxed out during it. Validate against the labeled 201 window (2026-03-18 → 03-21, LS 96–99 = ground truth). Exclude the round-hour file-boundary state dumps. The recurring nightly bursts at 201 and 701 are believed to be scheduled reboots (§6). They are downgraded to `info` inside a configured `WD:` reboot window, provided they release within the calibrated duration. A reboot burst that *doesn't release* still alarms.
      - **suspected false watchdog call** (low severity): the other zones are counting normally against their baseline. This is the known field failure, and it's worth reporting so it can be tracked.

      The reverse case is a failsafe signature with no watchdog call. It's still caught by the stuck-on and silence rules above. "Same unit" needs a detector-to-sensor grouping. With only `WD_Sensor1..N` to go on, either default to every channel at the intersection or add a `WD:` key mapping each sensor's channels; decide at spec time.

  Calibrate every threshold on the four corpus DBs and record the false-positive rate per rule. That uses the same method as the clock-step work in ROADMAP.md (e.g. "0 of 14,124 files flagged"). Output is a findings DataFrame: `date, detector, phase, role, rule, severity, value, threshold, message`. The deliverable includes a table of calibrated defaults, which the shell overrides through `WD:` config keys. The `WD_` category already exists and is unused, so it needs no schema change.
- **S-D3: phase-level Watchdog checks.** *Gemini-eligible against a D2-style spec.* Add `MaxOutThreshold` and `ForceOffThreshold` (the AM window, reusing termination events), `StuckPed` (from 89/90 when present, else 45), and `UnconfiguredApproach` (a phase served with no `Det_P{N}_*` / `TM_*` mapping). These use the same findings schema as D2, and the thresholds come from D2's calibration pass, not from the Gemini run.
- **S-D4: engine, CLI and report.** *Gemini-eligible, with the plot tests Opus-written.* Build a `DetectorHealthEngine` in `data/` that mirrors `AogEngine`/`DetectorEngine` exactly. Add `atspm detector-health` with `--target/--targetid/--all`, date or date-range, `--window am|pm|day`, and `--min-severity`. **Findings persist in the DB (owner, 2026-09-30).** Add a new `detector_findings` table: `date, window, detector, phase, role, rule, severity, value, threshold, message, computed_at`, with UNIQUE `(date, window, detector, phase, rule)`. Phase-level findings from D3 use `detector = -1`. A re-run over a date range deletes and replaces that range's rows in one transaction, so the run is idempotent and safe for a future scheduler to repeat. It's derived data, so `clear_ingested_data()` (`--rebuild`) should clear it alongside `events`/`cycles`/`ingestion_log`. Add the table in `DatabaseManager` schema setup. It's a schema change, so log it in `PENDING_DOC_CHANGES.md`. Keep the ignore list in `int_cfg` (config's source of truth), applied at read and report time rather than at write time, so ignored findings are still recorded. The command also writes a findings CSV and an HTML **detector × day heatmap** (count normalized to baseline, with findings overlaid) from a new `plotting/detector_health.py`. Also add an ignore list: `WD:` keys such as `WD_Ignore` = `det:rule[,…]`, the analogue of UDOT's `WatchDogIgnoreEvent`. Hook into `atspm report` so the daily report includes a findings summary. Log it in `PENDING_DOC_CHANGES.md` (new CLI subcommand, new public exports).
- **S-D5: Timing and Actuation plot.** *Opus for the trace layout, then Gemini for the CLI wiring.* This is UDOT's visual detector check: per-phase rows with green/yellow/red bars, plus detector on-intervals grouped under their phase by role (from D0), plus ped, call (43/44) and preempt rows. Reuse `_build_phase_intervals` and `_reconstruct_intervals`, with vectorized `[start, end, None]` segments and no row iteration. Add CLI `plot-timing-actuation`, and link to it from each D4 finding (the window around a stuck-on or chatter event).
- **S-D6: infer the detector configuration from events (owner, 2026-10-02).** *Opus: inference and a verdict.* Build a pure module that proposes the `Det_P{N}_*` / `TM_*` mapping from actuation behaviour alone, for the owner to review before anything is written to `int_cfg.csv`. Prompted by S-M1, where 315's per-lane presence zones were unconfigured and were recovered from the data (50/51/52 → P2, 34/35/36 → P6, owner-confirmed). Evidence to combine:
  - **Mode:** the on-duration distribution separates pulse/count loops (median ≈ 0.1–0.4 s) from presence zones (long holds through red).
  - **Phase:** presence zones release just after their phase's green onset and hold through its red. Count loops actuate only during, and just after, its green. Concurrent phases (2/6, 4/8) need a second cue, such as the side-of-barrier timing of a lagging/leading left, or the count totals below.
  - **Lane and sequence:** in light traffic, single vehicles give clean pulse chains (advance → presence → downstream count loop) with stable travel times. That orders channels along a lane and groups them into lanes.
  - **Movement:** per-lane totals match across a chain; channels of one approach split totals by lane.
  - **Silence and drift:** configured-but-silent channels and active-but-unconfigured ones (201's 60/63/64 vs its active zones) are flagged, as is a suspect role (201 channel 41).

  Output: a proposed role table with a confidence per row, diffed against the current config. Golden cases: 315's confirmed P2/P6 mapping, and its `TM_*` count loops. Depends on S-D0 (the role table it fills). Fills S-M1's `_Occupancy` (presence) keys and feeds S-D2's `UnconfiguredDetector` rule. **Scoped 2026-10-02:** design in `docs/design_detector_config_inference.md`. Read it before building S-D6. It covers 315's measured duration classes, the API, the steps (mode, then phase with concurrent-phase discrimination, then light-traffic lane chains), confidence, the diff statuses, the goldens and the calibration plan. Owner answers (2026-10-02): propose fixes (every row carries an inferred role and phase), and 37/53 are believed to be minor-approach advance zones that were deliberately left unconfigured. The data disagrees on 37/53; see below.

  **Core built 2026-10-02 (branch `feat/detector-inference`).** `analysis/detector_inference.py` holds `infer_detector_roles` and `diff_detector_roles`, with 19 goldens (a synthetic intersection with queueing, plus corpus goldens on 315 and 201). The design changed during prototyping:
  - **Green-share scores fail at concurrent phases, and so does a linear regression of pulse counts on green indicators.** The regression put P6 count loops on P1 because P1's tail overlaps P6's discharge.
  - **What works for count loops and presence is the *onset jump*:** events in `[g, g+8 s)` against `[g−8 s, g)` at each phase's green onsets. A concurrent phase that's already green doesn't jump.
  - **Phases whose onsets coincide on ≥ 90 % of cycles are reported as ties** (`P2|P6`), because no timing cue separates them. That covers 201's 2/6 (100 % coincident) and 313's 1/6.
  - **A ridge regression of presence releases on onset windows**, over groups of coincident phases, is a second cue. It helps on short left-turn phases.
  - **Lane chains:** the densest 1-s lag window in light-traffic hours (bin edges split 4.5 s lags) recovers every 315 lane, advance → presence (4.6 s) → count loop (1.1 s). Advance zones take their phase from their lane.
  - **The shell limits candidates to the `RB_*` ring phases**, so 201's virtual P15 stops winning.

  **Calibration (3 days each; configured, active channels):**

  | Site | match | consistent (tie includes config) | conflict | high-confidence conflicts |
  |---|---|---|---|---|
  | 315 (2025-12-15) | 19 | 2 | 3: left loops 22/30 (P3/P7 run 646/168 s in 3 days), 25 (no chain) | 0 |
  | 201 (2026-06-21) | 3 | 3 | 2: **41 → occupancy P3** (the suspected channel), 43 → tie P15/P16 | 1 (41) |
  | 313 (latest 3 days) | 4 | 1 | 2: 36 (arrival, tie), **43 configured P1 → P8 (medium)** | 0 |

  Only one high-confidence proposal disagrees with config, 201/41, which the owner already suspected. Six configured 201 channels are `silent`.

  **Findings for the owner:**
  1. 315's 37 and 53 behave as **approach-wide zones on the *major* approaches**. Every WB advance lane (38–40) reaches 37 about 1.7 s later, and every EB lane (54–56) reaches 53. They are proposed as `arrival` P6/P2 and flagged `wide`. That doesn't fit a minor-approach advance, so please check the sensor layout.
  2. 315 has about 15 active, unconfigured presence-like zones. They're proposed as new rows:
     - 33 → P1.
     - 41/42/43 → P3.
     - 44 → P8.
     - 47/60/63 → P4.
     - 49 → P5.
     - 58/59 → P7.
     - 57 → tie P2|P6.
  3. 313/43 (configured P1 occupancy) releases at P8 green onsets.

  **Shell/CLI:** `atspm infer-detectors` (Gemini, `docs/specs/detector_inference_shell.md`, tests `tests/data/test_detector_inference_engine.py`).

  **Not done:** a 701 holdout run. Thresholds are the prototype values above, chosen on 315/201/313 rather than formally frozen.

## 4. Track M — Remaining measures

Ordered by value multiplied by readiness. Each session follows the same shape: pure core function, then Engine, then CLI subcommand with target-group parity, then a plot where UDOT has one.

- **S-M1: Purdue Split Failure.** *Opus.* Per cycle, compute GOR (green occupancy ratio, over the stop-bar detectors of the phase) and ROR5 (occupancy in the first 5 s of red). The cycle is a failure when both are ≥ 0.79, UDOT's default. Inputs are `Stop_Bar` detectors from D0, with the phase windows from `_build_phase_intervals`. Occupancy is the union of detector on-time across lanes per UDOT, so decide union vs. mean deliberately. **Owner, 2026-10-02: offer both, as a lane-aggregation choice.** `union` (any lane occupied) matches a single multi-lane detector and so UDOT's method; `mean` averages the per-lane GOR and ROR5. The owner's concern is that the union method is suspect and imprecise: the same 0.79 threshold applies whether the approach has 1 or 4 lanes, yet the same occupancy means very different density on one lane than on four, and a union of independent lanes rises with lane count (1 − Π(1 − o_i)). Report per-lane GOR/ROR5 too, so both aggregates can be derived and compared. **Reference implementation:** `~/SPMs/Notebooks/spmfunctions/moes.py:81` (`split_failures`, `bin_SF` at :209), one stop-bar detector per phase. Use it as a test oracle (one lane, where union equals mean, and its window definitions) and for ideas, not as a port: it doesn't fit this architecture. Verified 2026-10-02 that it is single-detector: it filters `ID == det`, and every notebook call passes one `det`. It uses `DataFrame.append` (removed in pandas 2.0; the venv has 3.0.3), so running it as an oracle needs a small `pd.concat` shim. Its GOR window is 1→10 (includes yellow), its ROR window is the 5 s from code 10 (start of red clearance), and its threshold is 0.8; see `spms_notebook_ideas.md` §2.1 for its detector edge cases. A gap marker inside the green or red-5 window drops the cycle. Outputs: per-cycle GOR/ROR5/fail, and binned failure %. Plot: GOR/ROR5 scatter per cycle with fail markers. Also a parity spot-check that `plot-coordination` matches the UDOT PCD definition (arrival shift, cycle reference point). The results feed the throughput optimizer's saturation classification (`design_throughput_optimizer.md`).

  **Built 2026-10-02 (branch `feat/split-failures`).** Core `analysis/split_failures.py` (Opus, 33 golden tests, including parity with SPMs `split_failures` on one lane at thresholds 0.8 and 0.5; running SPMs needs two pandas-3 shims, `DataFrame.append` and its int `Period` sentinel in a datetime column). Shell `data/split_failures.py`, plot `plotting/split_failures.py` and `atspm split-failures` were done by Gemini against `tests/data/test_split_failures_engine.py`, then Opus-reviewed. Decisions:
  - **GOR = green only (1→8)**, the Purdue/UDOT definition, so UDOT's 0.79 means what it says; `--include-yellow` gives the SPMs 1→end-of-yellow window. **ROR5 = `[end of yellow, +5 s)`** (Code 9, else 10), clipped at the phase's next green. **Fail = GOR > thr and ROR5 > thr**, strict, as UDOT and SPMs do. `threshold` is a plain parameter, default 0.79.
  - Each stop-bar channel is one lane. `union` and `mean` are both always in the cycle CSV, with `n_lanes`, `n_lanes_failed` and a per-lane CSV, so any aggregate can be derived. `--aggregate` only picks which fills `gor`/`ror5`/`fail`.
  - Lane state is inferred within a segment only (leading OFF = on since the segment start; on at the end of the data = on to the last event). A lane silent for a whole segment is **unknown and excluded**, not 0. A gap marker anywhere in `[green, ror_end]` drops the cycle. Binned GOR/ROR5 are time-weighted, and `sf_pct = fails / cycles` per phase × plan × bin.

  **Findings, 2026-10-02.** 315 AM peak (06–09, 2025-12-15 → 19) and 201 Sunday midday (2026-06-21 11–14).
  - **The configured `Stop_Bar` channels are the wrong detectors for this measure.** Owner, 2026-10-02: at 315 they are short count loops downstream of the stop bar, used for counts and the discharge flow rate (median on-time 0.1 s; GOR 0.01–0.05). The per-lane presence zones are unconfigured. **Owner-confirmed: 50/51/52 = P2, 34/35/36 = P6** (inferred from long-ON releases in the first 6 s of the phase's green). At 201, P2/P3/P8's `Stop_Bar` channels (60/64/63) log nothing that day, every phase is single-lane, and P4's channel 41 (also `TM_WBT` and in `Det_P3_Pairs`) reports 31.7 % SF while P4's `_Occupancy` zone (39) reports 0 %. Treat 41 as mis-mapped.
  - **Union vs mean, 315 presence zones (3 lanes):**

    | Phase | cycles | GOR union / mean | ROR5 union / mean | SF union / mean / any-lane (0.79) | advisory pass (capped) |
    |---|---|---|---|---|---|
    | P2 | 403 | 0.33 / 0.16 | 0.10 / 0.03 | 0 / 0 / 0 % | 0.5 % (73 %) |
    | P6 | 396 | 0.34 / 0.19 | 0.19 / 0.08 | 0.3 / 0.3 / 0.3 % | 0.8 % (75 %) |

    One lane: union = mean exactly (every 201 phase). Three lanes: union reads about **2× mean on GOR and 2.5–3× on ROR5**. Measured union GOR sits *below* the independence prediction `1 − Π(1 − o_i)` (0.33 vs 0.39 on P2, 0.34 vs 0.44 on P6), because lanes discharge in correlated platoons. ROR5 matches it (0.10 vs 0.10, 0.19 vs 0.20). So lane count inflates union through the formula in red, and somewhat less in green. Tails: GOR p95 union 0.64 vs mean 0.42; ROR5 p95 0.90 vs 0.33. At threshold 0.5, union fails 2.1 % of cycles and mean 0.1 %.
  - **Against the optimizer advisory:** both say 315's P2/P6 are not saturated (SF ≤ 0.3 %, end-slack pass ≤ 0.8 % while 73–75 % of cycles force off), which fits `--validate`'s FAIL. On low-volume single-lane phases the advisory passes far more often than SF fails (315 P3 31 % vs 0 %, P7 23 % vs 0 %; 201 P7 49 % vs 13 % on its `_Occupancy` zone). A late arrival before a max-out passes end slack but leaves no residual queue. Split failures are the better advisory there, as step 2 expected. **Not wired into `atspm optimize` yet.**

  **Recommendation.**
  1. Keep **`union` + 0.79 as the default**, for parity with UDOT and the published thresholds, and label it as such.
  2. Give `mean` **no default failure threshold of its own yet**. At 0.79 it is close to unreachable on three lanes (p99 GOR 0.54), so read it as a diagnostic beside union. There are no saturated multi-lane cycles in the corpus to calibrate it against (the same blocker as optimizer step 6).
  3. **Add `any` (a cycle fails when any lane fails on its own GOR/ROR5) as a third aggregate.** It is the only rule under which 0.79 keeps its single-detector meaning at every lane count, and it catches one jammed lane beside empty ones, which `mean` averages away (golden test `test_one_jammed_lane_beside_empty_lanes`). `n_lanes_failed` already carries it, so the change is small. Report the worst lane's GOR/ROR5 with it. **Done 2026-10-02 (branch `feat/split-failures-any`):** `--aggregate any`. The worst lane is the one with the highest `min(GOR, ROR5)` (ties: higher sum, then lowest channel), so `fail` under `any` equals `n_lanes_failed > 0`. The cycle frame gains `g_occ_any, r_occ_any, gor_any, ror5_any, worst_det`, and `mean`/`union` stay beside it. 315 AM peaks (799 cycles): at 0.79, any = union = mean (P2 0 %, P6 0.3 %). At 0.5, any fails 0.75 %, union 2.1 % and mean 0.1 %. Worst-lane ROR5 p95 is 0.68 (P2) and 0.94 (P6).
  4. **Blocker for real-data use: read the presence zones, which already have a key.** `Plt: P{N} Occupancy` → `Det_P{N}_Occupancy` is the zone at the stop line (the coordination plot draws it at 0 s, between Arrival at −10 s and Stop Bar at +10 s). Filled at 201 (P2/3/4/6/7/8 = 46/53/39/33/38/51) and 313 (P1/2/6/8 = 43/38/34/42), blank at 315 and 701 (315's would be P2 = 50,51,52 and P6 = 34,35,36). **Done 2026-10-02:** `SplitFailureEngine` reads `_Occupancy` (shared `_parse_detector_sets` in `analysis/critical.py`) and ignores `Stop_Bar`. 315's `int_cfg.csv` now carries P2 = 50,51,52 and P6 = 34,35,36 (owner-confirmed), and `atspm split-failures` on the AM peaks reproduces the table above (P2 0/403, P6 1/396 under union and mean). S-D0's role parser should take over the key. (Corrected 2026-10-02: an earlier draft proposed a new `Det_P{N}_Presence` key.)
  - **PCD parity spot-check (2026-10-02).**
    - **`plot-coordination` is a ring split diagram with arrivals, not a UDOT PCD.**
      - Its y-axis is time from the *barrier cycle* start (`cycles.cycle_start`, all phases). UDOT's PCD is per phase, on red-to-red cycles (`9 → 1 → 8 → 9`; this is from memory of the UDOT ATSPM source, not re-verified here), with y = time since start of red.
      - Its arrival shift is a manual −30…+30 s slider per ring. UDOT's is per detector: distance ÷ speed + latency.
      - The ±10 s detector x-shift is cosmetic and doesn't touch y.
      - So a per-phase red-to-red PCD is a missing plot, not a parity tweak. Track it as a new M item if wanted.
    - **AoG undercounts when a phase is served twice inside one detected cycle (bug, `analysis/aog.py:arrival_on_green`).**
      - Mechanism: each arrival is tested only against the green window of the *last* `cycle_start ≤ t` (`searchsorted` over a `cycle_start` column with duplicates). Arrivals during the cycle's earlier greens therefore count as red.
      - Measured on 315, 2025-12-15, arrival detectors, against a red-to-red re-count:
        - P2: 0.675 vs 0.697.
        - P6: 0.573 vs 0.598.
        - Hourly differences: median about 3 points, up to 20 (P2) and 71 (P6) points.
        - Total arrivals agree (8,375 vs 8,366).
      - Exposure: 111 of 952 P6 greens share a `cycle_start` (cycles up to 1,066 s), mostly plan 0 at night, but also under plans 1 and 90.
      - Fix (Opus, small): match arrivals to green windows directly (`searchsorted` on `green_ts`, contained in `[green_ts, yellow_ts)`) and keep the cycle grouping only for totals. **Fixed 2026-10-02 (`fix/aog-multi-green`):** arrivals are tested against the green that contains them. A phase served twice in a cycle splits the cycle's other arrivals at the earlier green's yellow. `cycle_len` is now the true cycle length on every row (the first green of a shared cycle used to get 0, which made `green_pct` inf). Goldens are in `tests/analysis/test_aog.py`, including a corpus recount. On 315 on 2025-12-15, green arrivals now equal the recount exactly (P2 5,834, P6 4,361), day AoG agrees within 0.1 point, and the median hourly difference is ≤ 0.6 points (the rest is binning by `cycle_start` versus by arrival time).
- **S-M2: Split Monitor plot and programmed splits.** *Gemini-eligible.* Parse 131–149 into a plan-timeline DataFrame (`start, plan, cycle, offset, split_1..16`). The corpus has these codes everywhere. Plot the per-cycle split per phase, colored by termination type, with the programmed split as a step line, plus UDOT's per-plan percentile (50th/85th) stats. `splits` already has the durations.
- **S-M3: Arrivals on Red and Approach Delay.** *Opus.* Extend the AoG core to return green/yellow/red arrival shares (AoR = red arrivals ÷ total). Approach delay is UDOT's: for a vehicle arriving on red, delay is the time to the next green, from `Arrival` detectors shifted by distance/speed like AoG. It reuses the AoG shift and its gap-marker handling. Outputs: per-cycle and binned delay per vehicle, and total delay per hour. It could be one engine with AoG or a sibling; decide at spec time.
- **S-M4: Yellow and Red Actuations.** *Opus for the classification, then Gemini for the wiring.* Stop-bar detector on-events during yellow, red clearance and red, per phase. UDOT's severe-violation threshold is an actuation within *N* s after red start. It reuses the `TM_Exclusions` idea, where a detector already excluded for that state is skipped. Needs `Stop_Bar` roles (D0).
- **S-M5: Pedestrian Delay and Wait Time.** *Opus for one shared call-to-service pairing core, then Gemini.* Ped delay is the first ped call (90, else 45) to Begin Walk (21). Wait time is phase call registered (43) to green (1), split by whether the prior phase gapped out or maxed/forced off, per UDOT. Both are "first call since last service → next service" within a gap-free segment. `ped_counts` (`analysis/counts.py:128`) already has that pairing pattern, so generalize it rather than duplicate it. Ped delay can only be computed at 315 and 701, and the core must say so cleanly (empty result plus a reason) elsewhere.
- **S-M6: Preemption Detail / Service / Request.** *Gemini-eligible against a spec.* Pair 102 → 105 → 106 → 107 → 111 → 104 per preempt number into entry delay, track clear, dwell and exit durations. Include a request-vs-service count and the 116 force-offs. 315 has 24 events for goldens. The gap-marker rule applies to every pair. Small.
- **S-M7: Green Time Utilization.** *Gemini-eligible after M2.* For each phase, average the stop-bar actuations per 2 s (or configurable) bin of green, per plan, overlaid with the programmed split from M2. It shows wasted green at the end of the split, and pairs naturally with the flow-rate profiles.
- **S-M8: Approach Volume.** *Gemini-eligible.* Directional volumes from the `TM_*` sums grouped by approach (EB/WB/NB/SB prefix), with peak hour, K-factor and D-factor (the opposing-direction split). Mostly aggregation over `counts`.
- **S-M9: Left Turn Gap Analysis.** *Opus.* During the permissive left-turn green, histogram the gaps in the opposing-through detection (union of opposing `TM_*` through-lane detectors) into UDOT's bins (e.g. 1–3.3, 3.3–3.7, 3.7–7.4, > 7.4 s, all configurable), plus the sum of gaps ≥ the critical gap per cycle. **Needs one new config fact:** which phase is the permissive left and which movement opposes it. Derive it from `RB_*` plus the `TM_*` labels where unambiguous, and add an explicit `Det:` key otherwise. Coordinate with the *Intersection lane configuration* scoping, which touches the same config. That item is on branch `feat/video-ts-input` and isn't yet in ROADMAP.md on `main`.
- **S-M10: Left Turn Gap Report.** *Opus.* This is the protected/permissive feasibility screen, combining M9 gaps, M1 left-turn split failures, M5 ped actuations and M8 volumes (cross-product of left and opposing volume). Keep it last.

## 5. Deferred: needs a decision first

- **Time-Space Diagram and Link Pivot.** These are corridor-level, spanning multiple intersection DBs. They need a corridor config (ordered signals, link distances, speeds) that has no home in the one-DB-per-intersection design. Decide where corridor config lives before scoping.
- **Approach Speed.** UDOT uses a separate speed-event feed. Check whether the EVO radar "secondary" devices (`devices.json`) log per-vehicle speeds before scoping anything.
- **TSP / Priority.** Revisit when a site logs 112–119.

## 6. Open questions and settled decisions

**Settled (owner, 2026-09-30):**

- `WD_Sensor*` are failsafe watchdog zones, and they are unreliable. Handled in §2 and the S-D2 failsafe rule.
- Findings persist in a `detector_findings` table (S-D4).
- The Watchdog runs on demand only, since nothing in the project is scheduled. A scheduler may come later, so D4 must stay idempotent and need no interaction. That means no prompts and an exit code that reflects severity. Scheduling and notification themselves are out of scope until then.

- Logic statements 96–99 were 201's failsafe logic. They're disabled now and serve only as historical ground truth (§2).

- The nightly bursts at 00:00 and 01:55 MDT at 201 and 701 are believed to be **scheduled sensor reboots**. The owner thinks so but hasn't confirmed it against the sensor config. S-D2 treats them as expected events, and **does not hardcode the times**. Scheduled windows come from a `WD:` key (local time, so DST is handled by `resolve_pytz`). A burst inside a scheduled window that releases within a calibrated duration is recorded as `info`, not as a finding. One that doesn't release, or that runs unusually long, is a full failsafe finding. Measure the normal reboot duration while calibrating; the 201 window above shows about 10–40 s. A burst *outside* any window, like 315's on 2026-01-10/11, stays a finding.

**Open:** none at present.
