# Roadmap

Planned and deferred work only. The original backlog — a few months of Jules's accumulated review-only suggestions, triaged against the live code — was split into self-contained sessions (A–F, plus the Video Overlay build and audit) and worked through. That history lives in git; the decisions those sessions settled are reflected in [architecture.md](architecture.md) and the test suite. What remains below is the open fix candidates, one deferred decision, a future feature, and a standing "don't re-investigate" list.

## Open fix candidates

Small and actionable; each carries enough file/line detail to be picked up cold.

- ~~**Event timestamps are 0–1.0 s early: the decoder base ignores the header's sub-minute offset.**~~ **Done** (header offset added in `parse_datz_bytes`; `process --rebuild` added) — [archived](ROADMAP_ARCHIVE.md).
- ~~**Out-of-order timestamps from a backward clock set are ingested silently.**~~ **Done** (`_fence_clock_steps` inserts a hard-reset marker per backward step) — [archived](ROADMAP_ARCHIVE.md).
- ~~**`resolve_stopbar_target` has no phase-number range check.**~~ **Done** (rejects phases outside 1–16) — [archived](ROADMAP_ARCHIVE.md).
- **Operational: rebuild the production DBs onto the corrected timestamp basis.** The decision is settled and the tooling exists (see the archived header-offset item, [ROADMAP_ARCHIVE.md](ROADMAP_ARCHIVE.md)) — what remains is running it. `atspm process --rebuild --all` re-derives every `events` row from `raw_data/`, so any DB ingested before the header-offset fix still holds rows 0–1.0 s early until it is run. Only worth doing where `raw_data/` still holds the full history; anything already pruned rebuilds to whatever files survive, so check coverage per intersection first. Not urgent — the error is sub-second and uniform within a file — but until it happens each of those DBs is mixed-basis at the cutover point.
- **Fable edge-case pass on the SQLite fixture smoke tests.** `tests/data/test_manager.py` and `tests/data/test_reader.py` carry happy-path smoke tests plus `# TODO(fable):` stubs for the real edge cases: `get_metadata` falling back to `{"timezone": "US/Mountain"}` when the `metadata` table is *missing* vs. *present-but-empty* (two distinct code paths in `manager.py`), and `check_data_quality` completeness scoring around gap markers (marker on the window boundary, `gap_count > event_count` flooring at 0.0, the zero-event-count denominator guard). Adversarial edge-case reasoning → a Fable lane, carrying the standing `event_code == -1` gap-marker audit mandate (CLAUDE.md §5).

- **`ped_counts` misses some pedestrian services (found by `volume_explorer`, not yet investigated here).** Recorded in `~/volume_explorer/ROADMAP.md` ("Not this project's to fix", found 2026-07 during its Item 7). At 701, 4 of 175 pedestrian services across the five shared days (2026-02-24 to 02-28) are counted by the exported workbook but not by `ped_counts`. Example: a call (code 45, phase 3) at `2026-02-24 07:10:06` and its service (code 21, same phase) 30 s later. The events are in `701_data.db`, so the fault is in the functional core, not ingestion. volume_explorer's suspicion is a pandas-version difference in the segment/pairing step. Vehicle movements and every detector channel agree exactly. Start by reproducing the 4 misses on `701_data.db`, then check the code 45 → 21 pairing in `analysis/counts.py` and whether a gap marker sits between the call and the service.

## Session-scoped items

Promoted from *Future features*. Each is one session; the scope and the done-criteria are written so it can be picked up cold. Order matters: S2 consumes S1.

- **S1 — Decode `eos_set_time`'s clock marks: ped phases 14/15/16.** *(Promoted 2026-10-01. The measured record below was first written on branch `docs/eos-clock-marks` (707f200, never merged) and is carried over verbatim; that branch is now superseded.)* The upstream half shipped 2026-09-30 (`econ_itd_tools/EOS/eos_set_time.py`; field bundle `econ-field-tools`), and every head unit running it now writes these into the controller's own `.datZ`. Nothing here reads them yet. This is the decoder, and the input to the correction item below. Everything below was measured on the bench (10.70.10.51, firmware 03.02.60) by pairing 13 pulses sent against the 1-minute files they produced.

  **What is emitted.** Pedestrian calls on three ped phases the site does not use, logged as event **90 (PedDetector On) / 89 (Off)** with the ped phase as the parameter, plus a code **45** at each ON. The site config (`EOS_MARKER_PED_*`) names them; the shipped defaults are:

  | ped | role | held for (host time) |
  |---|---|---|
  | 15 | **BEHIND**: controller slow | `|drift|`, 0.1 s resolution, at least 0.1 s, capped at 30.0 s |
  | 16 | **AHEAD**: controller fast | same |
  | 14 | **SET**: brackets a clock correction | `L`, a whole multiple of 10 s, spanning every edit |

  `drift` is controller minus true (host) time. Positive means the controller is ahead, so its labels are *late* and true time = label − drift. A site may configure other peds, but only 9–16, all different, and the tool refuses any ped whose phase draws as in use on STATUS. The per-run JSONL (below) records which peds were used.

  **When.** An **hourly drift check** (cron `37 * * * *` UTC) logs in, fires one drift pulse, logs out, and changes nothing. The **daily set** (`17 9 * * *` UTC, retried up to 4× 30 min apart) fires, in this order: a drift pulse (the pre-set measurement) → the SET bracket, if the clock needs moving (|drift| ≥ 0.75 s) → a second drift pulse with the residual. A drift pulse never spans a clock step; the bracket always does. The bracket's ON comes after the preceding drift pulse's OFF.

  **Decoding a drift pulse.** `|drift| ≈ logged_width + 0.15 s`. Logged widths ran 0.0–0.3 s short of what was sent (mean −0.15, n = 13; the ON registers later than the OFF). With the 0.1 s minimum, a logged 0.0–0.1 means |drift| ≲ 0.2 s. A logged width near 29.8 s means the drift is saturated (≥ 30 s): take the magnitude from the JSONL. Precision is ±0.05 s on top of the bias. The tool times the controller's tick against the host to within the 100 ms front-panel frame interval.

  **Decoding the bracket.** The logged width is `w = L + shift`, where `shift` is the whole seconds the controller clock moved (positive = set forward). `L` is chosen so that `w ≥ 5 s` whatever the direction: a backward shift longer than the pulse would otherwise log the OFF before the ON, 113 s early in one emulator run. The shift is **always a whole number of seconds**, because a front-panel edit replaces the integer seconds and keeps the controller's sub-second phase. So:
  - `shift_est = −drift_pre`, from the drift pulse immediately before (sign from its ped, magnitude from its width);
  - `L = 10 · round((w − shift_est) / 10)`;
  - `shift = round(w − L)`, which is exact.

  Checked against four bench sets (−2, +3, +40, −40 s): all recovered exactly. Where the pre-set drift pulse is saturated, `L` is known only modulo 10 s from the log alone; use the JSONL.

  **Where the step is.** Inside the bracket: the first edit lands ≥ 1.4 s after ON, and the edits of one correction finish within ~4 s. A correction can be up to three edits (seconds, then minutes, then hours, where the shift crosses a minute or an hour boundary), each a separate step. Their total is `shift`. A backward shift shows as decreasing offsets, which `_fence_clock_steps` already fences. A forward one shows as a hole.

  **After a set the controller is not exactly right.** The panel cannot touch the sub-second phase, so the residual is up to ±0.5 s. It is reported by the post-set drift pulse and by every hourly check after it. That is the point of the hourly pulses: a drift *series* per controller, to model between sets and to apply sub-second corrections that no set can make.

  **Cross-check record.** Every run appends one JSON object to `eos-time.jsonl` beside `run.sh` on the head unit. It carries the mode, before/after drift with half-width, each pulse's ped, role, sent width and host ON/OFF epochs, and each field edit with its ENTER epoch and applied seconds. The MOE puller does not fetch it yet. It is the authority when a pulse is saturated or missing, and the pulses are the authority for *position*.

  **Two traps.** (1) Code 45 is in `_TERMINATION_CODES`. Exclude the marker peds explicitly rather than relying on no walk ever following. (2) Marks cover only this tool's adjustments. A keypad change or a power event is still unmarked.

  **Status check 2026-10-01.** The shipped code is unchanged since the record above was written (last `eos_set_time.py` commit 9cede43, 2026-09-30 12:21). **No local DB holds a single marker event yet**: 0 rows of code 89/90 on peds 9–16 in any of 201/313/315/701, and 201 is ingested through 2026-10-01 19:59 UTC. The owner confirmed the same day that **markers aren't running in the field yet**. So the bench files below are the only real marker data until rollout. The session can build and test against them, but the field-day done-criterion waits for rollout.

  **Bench fixtures, recovered 2026-10-01** from a `/tmp` scratchpad of the 2026-09-30 econ_itd_tools session (where they could have been lost on reboot) into `tests/fixtures/clock_marks_bench_2026_09_30/`, 80 KB, not yet committed:
  - `ECON_10.70.10.51_2026_09_30_1303`–`1310.datZ`: eight 1-minute files carrying the 13 drift pulses and the four SET brackets.
  - `bench.jsonl`: the head-unit send log, with two drift checks and sets of −2, +3 (two edits, crossing a minute), +40 and −40 s, each edit's ENTER epoch and applied seconds.
  - `bench{1,2,3}_log.json`: host ON/OFF epochs from the earlier CIB probes.
  - `emu.jsonl` and `py38.jsonl`: emulator runs only (`127.0.0.1`), no `.datZ`.
  - `decode.py` and `pairs.py`: the throwaway decoders used at the time, as a reference only.

  Visible in the files: the −40 s set leaves the 13:05 file's header at 13:05:13.0, and its bracket OFF lands in the next file. Both are cross-file cases the tests must cover.

  **Session scope.**
  - Pure core (`analysis/clock_marks.py`): pair 90→89 on the configured marker peds, stopping at `event_code = -1` per CLAUDE.md §5. Classify each pulse by role. Emit a **drift-sample** frame (`ts`, `drift`, `drift_lo`/`drift_hi` from the bias band, `saturated`) and a **set** frame (`bracket_on`, `bracket_off`, `shift`, `edits_window`), using the bracket arithmetic above. Unpaired or saturated pulses get flagged, never guessed.
  - Marker peds per intersection come from config: a new `int_cfg.csv` key, or the JSONL, but **no default**, mirroring upstream's no-default rule. The key name needs owner sign-off; it's the same class of change as `Min_P{N}_Split`.
  - Exclude the marker peds from the termination query (trap 1).
  - Shell engine plus an `atspm clock-drift` CLI with `--target/--targetid/--all`, writing the drift series as CSV and a drift-vs-time Plotly figure.
  - Golden tests built from the bench fixtures above, using `bench.jsonl` as the oracle, plus synthetic edge cases: saturated drift, a set crossing a file boundary, a bracket split by a comms gap, a missing post-set pulse.
  - **Done when** the four bench sets recover their exact `shift` and the hourly series decodes from at least one real field day.
  - Routing: the pairing and decoding math stays with Opus; the shell, CLI and plot are Gemini-eligible against the Opus tests.

- **S2 — Drift-corrected (true-time) axis.** *(Promoted 2026-10-01; feasibility assessed, decisions open.)* Use S1's drift series to map every controller label onto true time, `true = label − d(label)`, where `d` is the controller's drift modelled between samples and stepping by `shift` at each SET. The owner's framing was "if the drift over an hour was 0.5 s, stretch everything". The assessment below concludes the *offset* is what matters, while the *stretch* is negligible.

  **Magnitudes.** Upstream assumed "a few seconds per day", about 0.1 s per hour. The owner's 0.5 s/hour example is about 140 ppm. Even at that rate, a stretch rescales durations by about 1.4 × 10⁻⁴: a 10 s green moves by 1.4 ms and a 120 s cycle by 17 ms, both one to two orders below the logger's 0.1 s resolution. **No duration-based measure changes measurably from the stretch.** The absolute offset `d` is what matters: up to ±0.5 s right after a set (the panel can't touch the sub-second phase), and growing at the drift rate until the next set.

  **What it buys:** cross-source and cross-controller alignment, where hundreds of milliseconds are visible.
  - Comparing two controllers' timelines (corridor progression and offsets, travel time between intersections, the throughput optimizer's validation across sites).
  - Joining against sources that carry their own clock without a per-clip calibration: EVO radar logs, INRIX, a camera's burned-in clock (notebook idea P).
  - It does **not** improve `video-sync`, which measures the video-to-data offset per clip and so already absorbs controller drift.

  **What it costs at 0.1 s granularity.** Less than it seems.
  - Stored timestamps are *already* off a global decisecond grid: the header-offset fix adds each file's sub-second header fraction, so offsets are deciseconds from a per-file base, not from the epoch.
  - The decisecond assumptions that remain are all per-file and inside ingestion: `_CLOCK_STEP_MARKER_LEAD = 0.05` and the `prev + 0.1` comms-gap marker (`data/ingestion.py:50-54, 523-525`). They hold if the correction is applied *after* those markers are placed. `d` is monotone (|rate| ≪ 1), so a corrected marker still sorts immediately ahead of the event it fences, and `UNIQUE(timestamp, event_code, parameter)` gains no new collisions.
  - Bin edges (15-min counts) can move an event across a boundary by a few hundred ms, which is negligible.

  **The finding that changes the old (a)/(b)/(c) decision.** The *Future features* entry rejected (b), a real-time axis, mainly because "a single missed adjustment silently skews everything after it". Hourly drift samples remove that objection: each one is an **absolute** measurement against host time, so an error or a missed set heals at the next sample instead of accumulating. Drift-based correction is (b) with absolute anchoring. Because the correction is global rather than per-file, it also avoids (c)'s file-overlap problem and its synthetic per-file warp.

  **What still needs care.**
  - **Backward-set bands.** Inside a replayed band, the label→true map isn't a function: two real moments share labels. Sorted by timestamp, the pre-step tail and post-step head interleave. Telling them apart needs **file order**, which only ingestion has: `_fence_clock_steps` sees the break index. So the SET jump must be applied at ingest (re-timestamp the post-step band by the exact `shift`) or recorded as a segment id. A read-time-only correction can't do it.
  - **Noise versus rate.** One drift sample is good to about ±0.1 s (the ±0.15 s width bias band, ±0.05 s precision, and a ≲0.2 s floor where the sign is unresolved). That's comparable to an hour's drift, so point-to-point interpolation would follow noise. Fit `d` per inter-set segment (linear, or robust linear) instead. The JSONL's host-timed measurement (target half-width 0.06 s) is tighter than the pulse width, so prefer it for magnitude once the MOE puller fetches it; that fetch is an upstream dependency.
  - **Unmarked steps** (keypad sets, power events) still break the model. Where a backward one is fenced but unmarked, the correction must stop at that marker and restart from the next drift sample, never interpolate across (CLAUDE.md §5).
  - **"True" means the head unit's host clock.** Whether that's NTP-disciplined, and how well, bounds the whole scheme. Confirm it upstream.

  **Decisions — settled by the owner 2026-10-01, conditional on read-time lag, which was then measured as negligible.**
  1. **`events` keeps controller labels; the correction is applied on read.** Storage stays rebuildable and idempotent, and `--rebuild` stays exact. The backward-set band fix is the one exception and is applied at ingest, the only place file order exists. Rewriting everything at ingest was rejected: the next drift sample isn't known yet when a file is ingested.
  2. **`d` is derived on the fly** from the 89/90 marker rows already in `events`. No new table and no schema change.
  3. **`cycles.cycle_start` stays in label time** and is mapped on read alongside events, through the same function.

  **Lag measured** on 201's DB (1.6 M events): the marker query over a window ±2 h runs in under 1 ms, because it uses the existing `idx_events_ts_code`. The `np.interp` map costs about 10 ns per event, so about 0.3 ms per day of data and about 70 ms for the full 6.7 M-event corpus. Fitting a few hundred hourly samples is negligible. If a large batch run ever shows otherwise, revisit decision 2 (persist `d`) before anything else. The read window must pull marker rows back to the segment's previous SET or unmarked fence, not just ±2 h, so the fit has its anchors.

  **Session scope.** A pure `drift_model()` (per-segment fit, segment boundaries at SETs and unmarked gap markers) and `to_true_time(ts, model)` in `analysis/`. The ingest-time band fix in `_fence_clock_steps`'s path. The read-path map for `events` and `cycles`. An opt-in `--true-time` read flag threaded through the shell. Golden tests:
  - a synthetic controller at a known ppm with injected sets recovers true time to within the sample noise;
  - a backward-set band unscrambles to file order;
  - an unmarked fence stops the model.

  **Done when** two controllers sharing a time reference agree after correction to within the drift-sample noise. Routing: Opus throughout (timestamp semantics and gap-marker logic).

## Deferred — needs a project-wide decision first

- **`get_phase_splits()`'s 10-keyword-argument signature (`data/phases.py:423`).** Real, but *every* Engine's `get_X` convenience wrapper has a similarly long explicit kwarg list by convention (`get_vehicle_counts`, `get_arrival_on_green`, …). Fixing this one in isolation would make it inconsistent with its siblings. If worth doing, decide on a project-wide convention (e.g. a shared options dataclass) first, then apply everywhere at once — don't one-off it.

## Future features

- **Candidates from the archived SPMs notebooks** (split failures, timing-plan history from codes 131–149, preemption suite, code-13 unused green, sensor-fault detection, and others): see [spms_notebook_ideas.md](spms_notebook_ideas.md). It has the triage against pyatspm and the non-obvious details of each.

- **Throughput cycle-length optimizer (`atspm optimize`).** For oversaturated critical intersections that set corridor `C`: pick the cycle length and splits that maximize `Σ 3600·n_p(s_p)/C` over saturated critical phases, using the *measured* cumulative discharge curves from `analysis/flow.py` rather than a constant saturation flow. The hypothesis is that existing long cycles (up to 210 s) are too long, because measured rates decay after ~40 s of green, giving a finite interior optimum. Design is settled in two docs; don't re-litigate them: [design_throughput_optimizer.md](design_throughput_optimizer.md) (objective, regime, solver shape, scope) and [design_optimizer_solver.md](design_optimizer_solver.md) (implementation-ready addendum from 2026-07-19: curve contract, exact max-plus DP allocation, ring/barrier decomposition, interior/boundary result states and the "lengthen cycle and re-measure" directive, flatness metric, test battery). **Status 2026-10-01:** §6.1 critical movement analysis is done (`atspm critical`), and so is step 1 below (branch `feat/optimizer-flow-extensions`). Step 1's per-lane question was settled in step 2: by default all lanes must pass. Remaining sequence, each step independently testable (addendum D7):
  1. ~~Flow extensions~~ **done 2026-10-01**: `discharge_profiles()` with a shared `_select_cycles` helper and a `stratify` mode (plus `flow --stratify`), and `flow_rate(max_lost=None)` to keep non-qualifying cycles for the classifier.
  2. ~~`saturation_state()` classifier~~ **done 2026-10-01, as an advisory only.** **Owner decision, 2026-10-01:** the saturated phases are an **engineer's declaration** (step 5: `--saturated 2 6`). The classifier's pass rates print beside it and never decide. Curves come from percentile selection (top-`pct` busiest at a given split length, optionally stratified), never from per-cycle saturation flags. Why: end slack can't reliably separate saturated from unsaturated. Gap-outs score under `max_lost` because `lost` includes clearance, which is fixed by gating on max-out/force-off (codes 5/6, now `flow_rate`'s `termination` column). But **coordinated phases always force off** (owner), so for them the gate passes everything and only lane slack is left. A busy, unsaturated through movement often puts a departure within about 2 s of yellow, inside `max_lost = 10`. The advisory requires all lanes to pass by default (`all_lanes=False` judges each lane alone). **Eventual advisory:** split failures (GOR > 0.8 and ROR5 > 0.8; notebook idea A, [spms_notebook_ideas.md](spms_notebook_ideas.md) §2.1). Its red-occupancy term sees a residual queue directly. Port it as its own measure, then show it beside the declaration in place of end slack.
  3. ~~`analysis/optimizer.py` pure core with the D8 unit tests~~ **done 2026-10-01**: `optimize()`, with 37 tests in `tests/analysis/test_optimizer.py`. Choices the design left open: minimum splits round **up** to the grid. A C-edge `boundary` needs at least two feasible scan points. A `C*` held at the shortest *feasible* cycle by minimums or sufficiency is a warning (`c_star_at_feasibility_limit`), not a `boundary` state, because re-measuring wouldn't move it. In a ring-group with no optimised phase, surplus goes to its highest-numbered phase, flagged `surplus`. An unsaturated phase with a curve but no demand is pinned at minimum and flagged `sufficiency_unverified`. `optimum` also carries `binding_ring` and `group_min_at_c_max`. **D4 amended 2026-10-02:** `at_boundary` counts any split within the last 5 s of a still-rising domain, not just within Δ/2. Mean curves flatten in their last grid step, so the DP stopped just short of the edge, and a synthetic saturated site read `interior` when it was the textbook boundary case. A structure with no served phase raises, so the shell must pass real cycles to `ring_barrier_structure`; with none, every `observed_share` is 0.
  4. ~~`plotting/optimizer.py` (throughput-vs-C, allocation, marginal-rate figures)~~ **done 2026-10-02.**
  5. ~~`data/optimizer.py` `OptimizerEngine` plus the `optimize` CLI subcommand~~ **done 2026-10-02** (Gemini against `docs/specs/optimizer_shell.md`, Opus-reviewed). Runs end to end at 315 since `fix/fya-code12-in-green` was merged in (Ph2/4/6/8 log Code 12 at their start-delayed FYA overlap's permissive start; `_build_phase_intervals` used to treat it as the end of green).
  6. `--validate` against real data: existing TOD plans are natural experiments, and the model must predict observed throughput differences between plans before its recommendations are trusted. **Designed 2026-10-02** (D8 amended: pairwise anchor/target prediction in place of leave-one-plan-out, relative-change metric, ranking with a 2 % dead band, magnitude ≤ 3 pp). Pure core `analysis/optimizer_validation.py` `validate_plans()` is done, with 29 tests. The shell and CLI flag are Gemini-eligible against `docs/specs/optimizer_validate.md`. **Blocked on data:** no repo site is saturated, runs ≥ 2 TOD plans *and* has stop-bar config. 315 isn't saturated. 201 is unsaturated, with one plan. 313 (ramp terminal, a likely candidate) and 701 run one plan and have no `Det_P{N}_Stop_Bar` keys, and 313 has cycles for 2026-08-17 only. **Open, flagged by step 6:** D0's short-split bias, about saturation rate × clearance lost time per lane, favours shorter cycles (D8 amendment, last bullet).

  Open before step 5: ~~the `Min_P{N}_Split` sign-off~~ signed off 2026-10-02 (name kept; values are the full split, clearance included). The shell takes saturated phases from `--saturated` (no default inference; a phase not declared is sized by sufficiency from its stop-bar demand). The new `Min_P{N}_Split` `int_cfg.csv` key needs the owner's sign-off (name, and whether values include clearance). The saturation threshold (0.8), `boundary_rate_tol` (100 vph) and the validation tolerance are provisional until real distributions are seen. Routing: steps 1–3 are functional-core math → Opus; 4–5 follow established patterns and are Gemini-eligible against an Opus-written spec and tests.

- **Correcting for controller clock adjustments, informed by a marker pulse.** The near-term fix above only *fences off* a backward clock set. This is the ambition of actually repairing the affected window, using the SET bracket `eos_set_time.py` now fires across every adjustment (landed 2026-09-30, together with hourly drift pulses). **Superseded in part 2026-10-01 by session items S1 (decoder) and S2 (drift-corrected axis) above.** S2 reopens the (a)/(b)/(c) choice below: absolute hourly drift samples remove (b)'s main objection. The text below is kept as the design record.

  What the experiment established about the mechanism (2026-07-29, hardware-verified): recorded offsets track the *controller clock*, not elapsed real time. A file's clock window is always exactly nominal, but its real duration flexes — the −5 s set made a 1-minute file cover 65 real seconds, the +5 s set made one cover 55. So a backward set duplicates a band of labels and a forward set skips one. No events are ever lost or invented; only the labels are wrong.

  How the pulse recovers the correction: fire a pulse of known length `L` spanning the commit. Recorded length comes back `L + delta` for a forward set and `L − delta` for a backward one (negative if `delta > L`), so one interval yields magnitude, sign, and position at once. That works even when nothing else is happening, which matters because ~10% of empty forward-step holes still have a detector actuation spanning them whose duration silently gains `delta`.

  **Repair by re-timestamping, not reordering.** The events are in the correct sequence in the file — the logger appends chronologically. Only the labels are corrupt. So the repair is to add `delta` back to every event after the break, which restores both order and correct durations. Reordering by timestamp would be the wrong move; it treats the corrupted labels as authority.

  **The gating decision, and it is genuinely project-wide.** Correcting one file's tail by `+delta` makes it overlap the *next* file's head, because that file's labels start at the clock boundary. So re-timestamping alone does not close the problem, it relocates it. Three ways out:
  - (a) keep storing **controller-clock** timestamps and mark discontinuities (what the near-term fix does). Simple, honest, self-consistent per file; durations across a break stay wrong and the DB timeline is not real time.
  - (b) store a reconstructed **real-time** axis by accumulating every correction ever applied. Complete and makes durations correct everywhere, but it changes the meaning of every timestamp in the database, requires a gapless history of corrections (a single missed adjustment silently skews everything after it), and interacts with the `.datZ` base-time fix above — both change stored timestamps, so sequence them deliberately rather than shipping both blind.
  - (c) **re-timestamp, then normalise back into the file's own clock window** — add `delta` to post-break events to restore order and relative spacing, then scale the file's internal timeline by `nominal / (nominal + delta)` so it still ends where the next file begins. This keeps every file self-contained and local, needs no correction history and no downstream shifting, and fixes the ordering problem outright. The price is a known, bounded, uniform error on *every* duration in that one file:

    | file length | delta | scale | error on every duration | a 10.0 s green stores as |
    |---|---|---|---|---|
    | 1-min | 1 s | 0.98361 | 1.64% | 9.836 s |
    | 1-min | 2 s | 0.96774 | 3.23% | 9.677 s |
    | 1-min | 5 s | 0.92308 | 7.69% | 9.231 s |
    | 15-min | 1 s | 0.99889 | 0.11% | 9.989 s |
    | 15-min | 2 s | 0.99778 | 0.22% | 9.978 s |
    | 15-min | 5 s | 0.99448 | 0.55% | 9.945 s |

    A forward step is the same arithmetic with the sign flipped — the file holds `nominal − delta` real seconds and gets stretched rather than squeezed. Scope of the damage is one file per adjustment: with a daily sync that is 1.04% of 15-minute files or 0.069% of 1-minute files, and realistic drift is 1–2 s, so the practical case is the top and fourth rows.

    Two consequences to accept deliberately rather than discover later. First, stored timestamps become **synthetic** — they no longer correspond to any clock reading that ever existed, so cross-referencing that file against another source (video overlay, an EVO secondary device) will not align exactly. Second, the time base becomes piecewise-warped: two events genuinely 1 s apart store 1.000 s apart in most files and 0.984 s apart in a corrected 1-minute file. Both are almost certainly fine for ATSPM measures, and both should be documented rather than silent.

  (a) is the safe default and is already the near-term plan. (c) looks like the right eventual answer — it buys correct ordering and near-correct durations for a bounded, computable cost, without the all-or-nothing commitment of (b). Prefer 15-minute files at any intersection where this correction will run, since the same `delta` costs roughly 15× less there. Do not start (b) without deciding it explicitly. Also note the marker pulse only covers adjustments *this tooling* makes — a keypad change at the cabinet or a power event still has to be inferred, and forward sets are close to undetectable from the data alone (naturally-occurring 5 s quiet gaps run 31–47 per hour, so no hole threshold can separate them).
- **Automatic video/data sync from signal lamps (`lamp` shape + whole-clip alignment).** **Status 2026-10-01: implemented** as `atspm video-sync` (`analysis/video_sync.py`, `video/sync.py`; spec `docs/specs/video_auto_sync.md`). Real-data check: 201 clips 1 and 3 land 19 ms and 24 ms from the golden starts, and a guess one cycle out is refused (exit 2). Open follow-ups: the yellow lamp aligns 0.2-1.0 s early (cause unknown, so don't rely on yellow lamps); ntcip's overlay must skip `lamp` rows; batch sync from the collector's `manifest.json`; a dusk/night clip; excluding flash/preemption windows. The rest of this entry is the design record. Rewritten 2026-10-01 after a hand-built prototype on real data, which replaces the guesses this entry used to make. Today the sensor is the human eye. `video-locate-phase-change` predicts one green→yellow / yellow→red edge (`first_phase_transition_after`, `analysis/video.py`), renders a ±`--window` clip with a signed countdown (`extract_labeled_clip`, `video/processor.py`), and the user reads the value off the frame and passes it back as `--observed-delta`. The feature is to measure the lamps instead and print the corrected `--start` in one call.

  **Prototype evidence (201 fisheye, 720×720 @ 10 fps, 2026-10-01 collection).** Three 10-minute `.ts` clips from ntcip's `tools/collect_video_session.py` sit in `intersections/201_.../201_20261001_1209/`, alongside a `manifest.json` holding each clip's trigger-time guess. The NB head (phase 2) is a horizontal G-Y-R head. Each lamp is about **3×3 px**, and only the **green** lamp has strong contrast; red is faint. A 6×5 px ROI on the green lamp (x 640-646, y 415-420), measured as mean(G−R) per frame, is cleanly bimodal: about −5 off, +15 on. Using a 5 threshold, a 3-frame debounce and PTS frame times, every lamp edge was matched to the DB's phase-2 code 1/8 events. The implied first-frame time agreed **to within one frame across 13–20 edges per clip**. The few outliers were seconds off (occlusion during a long green). Results: 12:24:57.550, 12:39:55.952 and 12:54:54.950, which are 2.45 / 4.04 / 5.05 s before the trigger times. The trigger-time guess is therefore worth ±5 s, not ±1. Video-vs-controller slip within a clip was 0.1–0.3 s per 10 min. Cross-checks: the camera's burned-in clock read 12:24:58 on clip 1's first frame. `video-locate-phase-change` with the corrected start predicted 20.050 s, which is the lamp edge exactly. Overlay loop fills for detectors 46/17 matched a passing truck to within a frame. These three starts are **golden values** for tests.

  **Config: a `lamp` shape, one row per lamp, not a whole-head ROI.**
  ```
  type,points,color,input,phase,name,indication
  lamp,"643,417","0,255,0",,2,NB head,green
  ```
  - `points`: one point (sampled as a disc of about 3 px radius), or a small polygon, which the calibration GUI can already draw. A single-click "dot" mode is the GUI work.
  - `phase`: reuse `resolve_stopbar_target`, so overlaps (`OLB`) work for free.
  - `indication` (`green`/`yellow`/`red`): a **new column**. `input` can't carry it, because `ShapeConfig.load` casts it with `int()`. Old files stay loadable: `load` takes field names from the file's own header row and reads `name` via `row.get`, and `indication` must be read the same way. `save` writes `_CSV_FIELDS`, so append the column there.
  - Touch points: `ShapeConfig.load`/`.save`, `relevant_phases()`/`relevant_overlaps()` (both branch on `type`), and the overlay renderer. The renderer should draw each lamp as a dot coloured by the **DB's** state for that phase, beside the real lamp, so a bad sync is visible at a glance in every overlay.
  - ntcip's overlay (`ntcip_monitor/ui/overlay/`) reads its own flat-format shape CSVs. Check that it skips an unknown `lamp` type rather than raising.

  **Algorithm: whole-clip alignment, not nearest-edge matching.**
  1. For each lamp, compute a per-frame contrast series (lamp ROI vs. a local surround or the lamp's own off-level). Choose the on/off threshold **per clip** from the series' two modes (e.g. Otsu), never a hardcoded constant: 5 only suits this camera at midday.
  2. Build the DB's indication state for that phase (codes 1/8/9-10/11 → G/Y/R) sampled at the frame PTS times for a candidate offset. Search the offset that maximises agreement over the **whole clip**, using cross-correlation, the method the clock-skew work already settled on. Nearest-edge matching can lock onto the wrong cycle when the guess is off by about a phase length. Whole-clip agreement can't, and it absorbs occlusion outliers instead of being skewed by them.
  3. Refine the peak at sub-frame resolution from the edge midpoints. Fit the slip slope (offset vs. clip time) and report it. Optionally emit the mid-clip offset, which halves the worst-case error of a single `--start`.
  4. Report a confidence: agreement fraction, plus peak sharpness against the runner-up offset. **Refuse below a threshold and fall back to the labeled clip.** A wrong sync is worse than none, because every downstream overlay inherits it and looks fine.
  5. Never align across a gap marker (`event_code = -1`). Clamp the DB window at the first marker in or near the clip (201 had one at 12:05:06 on 2026-10-01, from a 2.6 s backward clock step).

  **Known hazards to test or bound, not discover:**
  - LED PWM aliasing against the shutter, giving single dark frames on a lit lamp. Size the debounce from fps, not a fixed 3 frames.
  - Night bloom, and glare or back-light. The relative-contrast metric should survive both, but that's unproven; collect a dusk/night clip.
  - Flashing operation and preemption windows: exclude them.
  - Heads seen edge-on.
  - A camera moved since calibration: low confidence should catch it.

  **Floor:** about half a frame (50 ms at 10 fps) plus the lamp's rise time. Fine for overlay alignment; don't chase more.

  **Lanes.** The shape schema, the alignment math (step 2-4), confidence and gap-marker handling, plus golden tests from the 2026-10-01 clips (with a small fixture cut from them, so the tests don't depend on the gitignored intersection folder): **Opus**. Calibration GUI dot mode, renderer drawing, CLI wiring (e.g. `video-sync --targetid --camera --video --start-guess`, or an `--auto` flag on `video-locate-phase-change`), and docs: **Gemini-eligible** against that spec, with the golden tests frozen.
- **Turning-movement counting via computer vision** (`EnhancedIOUTracker`, background-subtraction or YOLOv8 vehicle detection, approach-line crossing logic — carried over from the earlier `video_processing.py`). Deliberately split out of the Video Overlay work: model selection/tuning and tracking-accuracy validation is a substantially larger, different kind of problem. Revisit if there's appetite for it.
- **Intersection lane configuration (manual `int_cfg` schema & automated inference for Critical Movement Analysis).** Critical Movement Analysis (`src/atspm/analysis/critical.py`) currently computes `demand_per_lane = demand_vph / n_detectors`, using distinct stop-bar detector IDs (`Det_P{N}_Stopbar`) as a proxy for lane count. This proxy degrades when multiple detectors cover one lane (e.g. advance + stop-bar loops), when a single detector zone spans multiple lanes, or when shared lanes exist (e.g., shared Thru/Right or Thru/Left). Accurate Critical Movement Analysis and vphpl demand ratios require true lane counts and movement-to-lane mapping. Two complementary approaches to scope:

  - **Option 1: Explicit Manual Coding in `int_cfg.csv` (Deterministic / Configuration-based).**
    - Extend `int_cfg.csv` and the `config` table schema with explicit per-phase or per-movement lane configuration fields (e.g., `Lanes_P<N>` or `Lanes_TM_<Label>`).
    - Support shared-lane notation (e.g., `Shared_Lanes_P<N>_<M>`).
    - Update `phase_demand()` in `atspm.analysis.critical` to prefer configured lane counts over the `n_detectors` fallback when present.

  - **Option 2: Data-Driven Lane & Movement Inference (Detector & Flow Analysis).**
    - *Discharge Flow Rate / Saturation Capacity*: Compare observed maximum discharge rate during saturated green splits (`FlowRateEngine` / `atspm.analysis.flow`) against standard single-lane saturation flow (~1800–1900 vphpl) to infer effective number of discharge lanes per phase (e.g., ~3600 vph peak discharge → 2 lanes).
    - *Arrival / Departure Patterns & Shared Lanes*: Analyze vehicle-by-vehicle arrival patterns, co-actuations across movement detectors, and time-gap distributions during queue clearance to detect shared-lane behavior (e.g. left-turn blockages causing distinct headway distributions in shared thru/left lanes).
    - Surface inferred lane counts as an automated sanity check against `int_cfg.csv` configurations or as a fallback when configuration entries are omitted.

  **Status and direction (added 2026-10-01).** Neither option is built: `int_cfg.csv` has no lane rows (its row types today are `Det`, `Exc`, `Plt`, `RB`, `TM`, `WD`), and `phase_demand()` still divides by `n_detectors` (`analysis/critical.py:369-391`). Nor is either option decided: no key format, no shared-lane rule, no spec.
  - **Recommendation:** Option 1 first, as the source of truth, with Option 2 later as a cross-check. Have the config record **which lane each stop-bar detector covers**, not just a count per phase. A count fixes critical movement analysis's per-lane basis. Only the detector-to-lane map fixes the optimizer's exposure: its approach curve sums per-detector means (`design_optimizer_solver.md` D0), which double-counts two stop-bar detectors in one lane.
  - **Hand entry needn't be the only source.** Two cross-project feeds below could populate these rows instead: the *shared signal-network data store* (UTDF lane geometry) and the *EVO zone → detector → lane mapping* (radar sites). Settle the key format with both in mind, and with the optimizer's pending `Min_P{N}_Split` key, since all three add `int_cfg.csv` rows.
  - **Effect on the optimizer:** it needs lanes less than critical movement analysis does. Its solver uses approach totals and total vph, and the per-lane basis is "for criticality ranking only" (`design_optimizer_solver.md:183`). So lane configuration corrects its inputs but doesn't block building it.

- **Shared signal-network data store (central UTDF) with an `int_cfg` puller.** *(Added 2026-10-01, owner's direction; cross-project with `econ_itd_tools`.)* One store of each managed intersection's timing, phasing and, eventually, geometry and detection, built from the controller databases. `econ_itd_tools` (EOS) and pyatspm both read it, instead of each project hand-maintaining its own copy.
  - **What exists:** only a one-off script, `econ_itd_tools/working/build_utdf.py` (≈520 lines). It writes Synchro UTDF `Phasing.CSV` and `Timing.CSV` for the 6 Moscow intersections (`working/README.md`). Five sites are parsed from EOS DBPrint text exports. The sixth (6th & Main, an ASC/3) is read by hardcoded byte offsets from its binary database. Some values are hand-entered fallbacks ("per user verification" ped timing). It produces no lanes, detectors or volumes. Nothing in pyatspm or `volume_explorer` reads or writes UTDF. The one UTDF reference here is SPMs notebook idea O (a turning-movement volume export whose header looks invented; check it against Synchro's spec before reuse).
  - **What makes it buildable network-wide:** the EOS effort to reverse-engineer the controller database files so they can be read directly. That replaces DBPrint parsing and per-site offsets. It lives in `econ_itd_tools` and is that project's to sequence.
  - **pyatspm's side is a puller, not a second source of truth.** CLAUDE.md §3 makes `int_cfg.csv` the single source of truth for `config`. So the puller *writes `int_cfg.csv`* (or a proposed diff against it) from the store, and ingestion is unchanged. Hand edits stay possible, and the puller must report where they disagree with the store rather than silently overwrite them.
  - **Overlap with `int_cfg`:**

    | `int_cfg` today / planned | from the store |
    |---|---|
    | `RB` (ring-barrier) | phasing: phases in use and the ring/barrier structure |
    | `Det` (detector → phase) | the controller's detector assignment, which is the EOS 6 > 1 screen `econ_itd_tools` is already mapping; detector channel and position, if UTDF carries them |
    | `TM` (movement → detectors) | lane geometry plus detector placement |
    | planned lane rows (above) | UTDF lane geometry: lanes per lane group, shared lanes, storage |
    | planned `Min_P{N}_Split` (optimizer) | timing: min green plus clearance per phase |

    UTDF's lane data includes lane counts, shared designations and storage length. Whether it carries **detector** placement (number, position, size, channel) depends on the Synchro version. Verify against the UTDF spec of the version in use before designing around it. If it does, much of `Det`/`TM` could come from the store too.
  - **Open decisions (cross-project).**
    1. **Format.** UTDF files themselves, or a normalized store (SQLite) with UTDF as one export. Recommended: the latter. UTDF is Synchro's interchange format: one snapshot, no history, and no field for things like marker peds or controller channel numbering. A store can export UTDF on demand.
    2. **Ownership.** Recommended: `econ_itd_tools` owns the schema and the builder, since it reads the controllers. pyatspm owns only its puller.
    3. **Time.** `config` is temporal (`start_date`/`end_date`), so the store must be too. That ties it to the controller-settings history item below.
  - **Not yet:** spec, schema, or an entry in `econ_itd_tools/ROADMAP.md`. This entry is pyatspm's half; mirror it there when that project scopes the store.

- **Controller-settings history: programmed (EOS) and observed (event log).** *(Added 2026-10-01; cross-project.)* Two views of the same thing, settings over time, that should be built to meet.
  - **Observed (pyatspm):** SPMs notebook idea B in [spms_notebook_ideas.md](spms_notebook_ideas.md) §2.2. It rebuilds timing-plan history from controller events 131–149 (plan, cycle, offset, splits per phase each time a plan is selected). pyatspm reads only 131 today (`cycles.py:_merge_coordination_plan`). The notebook's filter bug is documented there: it drops legitimate 10 s splits and 0 s offsets, so don't copy it.
  - **Programmed (EOS):** a history tracker the owner plans in `econ_itd_tools`: periodic snapshots of the controller database, diffed over time. Not started, and not yet in that project's roadmap. Its nearest existing work is the detector-settings item there ("audit … against an intended baseline and diff two controllers").
  - **How they meet:** the event log shows what the controller *ran*, and the database shows what it was *programmed* to run. Cross-checking catches unlogged keypad changes, plans that never fire, and splits that differ from programming. The programmed history is also the natural feed for the time ranges of `int_cfg`'s temporal `config` and of the shared store above. Today those ranges are hand-set, as in the date header of each intersection's `int_cfg.csv`.
  - **pyatspm's piece, buildable now and independent of EOS:** decode 132–149 alongside 131 into a plan-history table or report, with a CLI subcommand (CLAUDE.md §7). Candidate for promotion once the history tracker's output format is known, so the two can share a shape.

- **EVO radar zones → controller detectors → lanes.** *(Added 2026-10-01; cross-project, about half of signals.)* EVO radar project files (`.iprj`, documented in `econ_itd_tools/EVO/iprj_designer/IPRJ_FORMAT.md`, with parse/write code in `iprj_io` and `dxf_iprj_excel_conv.py`) hold, per sensor, up to 64 event zones. Each zone has a polygon (≤10 vertices, in the background image's world coordinates), `ZoneType` (Motion / Presence / Sidewalk), `PhaseNumber`, `OutputNumber` and a free-text `ZoneName` (e.g. `"PH 4 SB inside"`). Zones were confirmed by overlay to "land precisely on the lanes". `OutputNumber` maps 1:1 to the controller detector input it drives.
  - **The chain:**
    1. EVO zone (polygon) → `OutputNumber`
    2. → controller detector, whose phase assignment comes from EOS's 6 > 1 screen
    3. → pyatspm's detector event parameter and `int_cfg` `Det` rows.

    Following it gives each pyatspm detector a physical footprint. That yields the detector-to-lane map recommended in the lane configuration entry, the stop-bar/advance distinction, setbacks, and which detectors really sit on a shared lane, without hand entry. It can feed the shared store, `int_cfg` directly, or both.
  - **Gaps:**
    - Lanes themselves aren't stored in `.iprj`: zones are drawn *over* them, so lane count and identity must come from UTDF geometry or hand entry, with zones assigned to lanes against it.
    - Coordinates are image pixels, so distances in feet need a per-site scale; check the EVO calibration plan (`CALIBRATION_ALIGNMENT_PLAN.md`) for one.
    - `OutputNumber` values 0 and 17–64 appear in the surveyed files, so verify the zone-output → controller-input numbering per site.
    - Only radar sites are covered, so loop sites need another source.
  - **Update 2026-10-01, from the cross-project review:**
    - Two pyatspm intersections have radar project files: `econ_itd_tools/sites/Banks` (201) and `Franklin_KCID` (315).
    - The pixel-scale gap is already solved upstream. iprj_designer's zone fit works in feet (Banks aligns at 5.6 ft mean residual; `model/zonefit.py`).
    - The controller-detector link is iprj_designer's Item 55, an EOS detection-settings editor and export built on the 6 > 1 / 6 > 2 maps.
  - **Next step:** a read-only prototype on one radar site that pyatspm already ingests (201 first). Join `.iprj` zones to `Det` rows through the controller's detector assignment, and hand-check the result against the background image.

- **Cross-project integration map.** *(Added 2026-10-01 from a review of every sibling project in `~/`.)* Where existing work elsewhere feeds, checks, or uses pyatspm. Each bullet is a lead to scope, not a decision; the matching half belongs in the other repo's roadmap.
  - **Intersection inventory.**
    - *volume_explorer* has a Future "intersection configuration builder" with lane counts per movement, already meant to seed from `int_cfg.csv` `RB:` rows. That's the same need as the lane configuration entry above.
    - Critical movement analysis now exists twice: pyatspm's `analysis/critical.py` and volume_explorer's presets. Once lanes land in `int_cfg`, volume_explorer should read pyatspm's per-lane demand through its existing DB intake (`atspm_source.py`) rather than recompute it.
    - *pst_conv* indexes an email archive that holds signal plan PDFs (e.g. `Traffic_Signal_3758_SH-55_SH_44_to_Beacon_Light.pdf`). Plan sheets carry lane geometry and loop layout, the source for sites without radar.
    - *Local Agreements* (about 11,800 cooperative agreements, searchable cache) gives the owning and maintaining agency per signal, which fills `metadata.agency_id` and an ownership field in the shared store.
  - **Detection health, built three times.** pyatspm (`Det_P{X}_Pairs`, after the fact), *ntcip* (live disagreement engine that saves video clips, with its own `paired_detector_id` pair format), and EVO radar fusion and comparisons.
    - Keep one pair definition, ideally in the shared store.
    - ntcip holds hand-labelled ground truth at Banks (201): `gt_anomalies_20260731_1830-2130.csv`, `…20260801_1300-1645`, `…20260802_0930-2229`. That's a labelled test set for `analyze_discrepancies`. But `201_data.db` has no events for 2026-07-19 to 08-02, so it needs those `.datZ` files retrieved, if the controller still retains them.
    - `econ_itd_tools/EVO/same_sensor_tmc_counter.py` counts turning movements from raw radar tracks. At radar sites that's an independent check on pyatspm's movement counts, and a cheaper route than the computer-vision counting item above.
    - econ_itd_tools' planned EOS web-app detector-status scrape gives live fault status alongside pyatspm's after-the-fact sensor-fault ideas (SPMs notebooks).
  - **One clock (builds on S2).** The EVO head unit runs both `eos_set_time` (the drift pulses) and the radar recorder, so radar tracks share the host clock S2 maps controller labels onto. After S2, controller events, radar tracks, ntcip's live log and camera footage line up without per-clip sync. Radar then becomes ground truth for arrival-on-green, queue length and discharge, including the optimizer's discharge curves. *traf_cams* (camera snapshot capture, Qognify client) is a video source for lamp sync and the camera-clock OCR route (SPMs idea P).
  - **Demand and results (closes the optimizer's loop).**
    - *Inrix*'s district screening ranks the worst corridors, which is where to aim the optimizer.
    - Inrix's before/after travel-time analysis (with changepoints) measures real corridor results after a retiming. That goes beyond the optimizer's own TOD-plan natural-experiment check.
    - *volume_explorer*'s TOD regime changepoints show where demand shifts in the day. Compare them with decoded plan history (events 131–149, the settings-history entry above) to see whether plan switch times match demand, and optimize C per regime.
    - *tcds-scraper* supplies long-term counts and seasonality profiles, to put a measured pyatspm period in annual context. Its MUTCD warrant module (`signalwarrants/`) could take pyatspm counts as input, for signal justification or removal studies from controller data.
  - **Collection.** *econ-field-tools*' TODO replaces the head-unit puller with a standalone tool built from pyatspm's retrieval code. That makes `data/retrieval.py` production-critical on head units. The same tool is the natural place to fetch `eos-time.jsonl`, which S2 wants for precise drift magnitudes.
  - **The larger shape.** All of this points to one per-intersection record that every tool reads and writes: configuration and geometry, detector health, clock drift, demand, corridor results, and the governing agreement. pyatspm's metadata plus the shared store above is the natural core. An intersection-status report could flag stale detectors, drifting clocks, plans that no longer fit demand, and corridors that got worse.

## Rejected / no action

Logged here so these don't get re-suggested or re-investigated later.

- **8 of 9 "Unused Import" suggestions were false positives** — checked every one against current usage; only `Literal` in `data/phases.py` was real (since removed). The rest (`argparse`/`sys` in `cli.py`, `List` in `reports/generators.py`, `Optional` in `utils/timezone.py`/`data/reader.py`, `Any`/`Dict` in `utils/logging.py`, `Dict` in `plotting/termination.py`, `pd` in `data/ingestion.py`/`data/detectors.py`) are all genuinely used. Likely stale from before recent edits.
- **`reports/__init__.py:22` "__all__ unused"** — references a class `ReportGenerator` that doesn't exist; the actual export is `PlotGenerator`. Stale/incorrect suggestion.
- **"Code Duplication in counts endpoints"** (`get_vehicle_counts`/`get_ped_counts`/`get_combined_counts`, `data/counts.py:595+`) — these are intentional thin wrappers matching the established Engine + `get_X` convenience-function pattern used everywhere else in `data/` (`AogEngine`/`get_arrival_on_green`, `PhaseEngine`/`get_phase_splits`, etc.). Collapsing them into one generic dispatcher would break consistency with their siblings for no real benefit — not recommended.
- **"Many Arguments" and "Overly Long" on `plot_coordination`** — both reference an old 11-argument signature that no longer exists. The function has already been refactored to 6 arguments and decomposed into ~15 helper functions (`_add_ring_bars`, `_add_coord_plan_markers`, `_add_detector_traces`, etc.). Stale — already addressed.
- **Notebook "Read stdout once Fix" / "List Files Fix"** (`notebooks/_Datz_SCP.ipynb`) — these already appear implemented, and are not part of the package itself (personal SCP automation). No action.
