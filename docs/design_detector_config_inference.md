# Design: infer the detector configuration from events (UDOT S-D6)

Status: **design only (2026-10-02)**, not built. Opus lane (inference, plus a verdict
on calibration). Depends on S-D0 (`analysis/detector_roles.py`, the role table this
fills and diffs against).

## Goal

Propose `Det_P{N}_{Arrival, Stop_Bar, Occupancy}` (and lane groups for `TM_*`) from
actuation behaviour alone, as a table **for the owner to review**. Nothing is written
to `int_cfg.csv`. The shell writes a CSV proposal and a diff against the current config.

## Ground truth available

- **315, owner-confirmed:** presence P2 = 50,51,52 and P6 = 34,35,36. Count loops
  (`Stop_Bar`) P2 = 26,27,28 and P6 = 18,19,20, plus single-lane P1/3/4/5/7/8 =
  17/22/31/25/30/23. Arrival P2 = 54,55,56 and P6 = 38,39,40. `TM_*` is the same channels
  as the count loops, plus the right turns 21/24/29/32.
- **201 (configured, believed correct):** the `_Occupancy` zones (46/53/39/33/38/51) and
  Arrival 49/36. **Known suspect:** channel 41 (`Stop_Bar` P4, `TM_WBT`, `Det_P3_Pairs`),
  and the `Stop_Bar` channels 60/63/64, which were silent on 2026-06-21.

## Evidence measured (315, 2025-12-15 06:00–09:00)

| class | channels | median on (s) | p90 on (s) | share on > 5 s |
|---|---|---|---|---|
| count loops (`Stop_Bar`, `TM_*`) | 17–32 | 0.1–0.3 | 0.2–1.3 | ≤ 0.02 |
| advance (`Arrival`) | 38–40, 54–56 | 0.3–0.4 | 0.4–0.6 | 0 |
| presence (`Occupancy`) | 34–36, 50–52 | 1.0–1.3 | 5.9–23 | 0.10–0.21 |
| **unconfigured, presence-like** | 33, 41–44, 47, 49, 57–60, 63 | 1.1–20 | 6.9–75 | 0.15–0.66 |
| **unconfigured, advance-like** | 37, 53 | 0.3–0.4 | 0.6–0.7 | 0 |

So mode is separable on duration alone at 315. The unconfigured rows are the real
payload: probably side-street and left-turn presence zones, plus two advance-like
channels with no phase. Their phases are unknown until the inference runs (open question 1).

## Core API (`analysis/detector_inference.py`, pure)

```python
infer_detector_roles(
    events_df,               # events with cycles: codes -1, 1, 8, 9, 10, 11, 12, 81, 82
    phases: list[int] | None = None,   # default: phases with any Code 1
    min_actuations: int = 50,
) -> pd.DataFrame
diff_detector_roles(proposed, configured_roles) -> pd.DataFrame
```

`infer_detector_roles` returns one row per active channel: `detector, role,
phase, lane_group, confidence, n_act, med_on, p90_on, frac_long, phase_score,
phase_margin, chain_lag_s, notes`. `role` is one of `arrival`, `stop_bar`,
`occupancy` or `unknown`. `phase` is Int64. `lane_group` is an int id, or NA.
`confidence` is `high`, `medium` or `low`.

`diff_detector_roles` outer-joins on `detector` against `parse_detector_roles(config)`
(phase roles only). Its `status` is one of:
- `match`: the same role and phase.
- `new`: active but unconfigured.
- `conflict`: a different role or phase. 201's channel 41 is expected here.
- `silent`: configured with fewer than `min_actuations` (201's 60/63/64).
- `low_volume`: active, but too few actuations to classify.

All interval work goes through `_reconstruct_intervals`. **Gap markers:** intervals that
span a `-1` are censored, so they are excluded from duration statistics, never
measured. No chain lag or green-relative timing pairs events across a gap.

## Inference steps

1. **Mode (duration).** Per channel, use the on-interval durations: median, p90, and the
   share > 5 s. Presence means a share > 5 s of at least `0.05`, or a median of at least
   `0.8 s`. Otherwise it's a pulse class. Pulse channels split into count loops versus
   advance by **when** they fire relative to green (step 2), not by duration: the
   0.1–0.3 s versus 0.3–0.4 s gap is too thin to rely on at other sites. Calibrate
   the presence cut on 201/313/315 (§ Calibration).
2. **Phase.** For each channel × candidate phase, take the phase's green onsets
   (Code 1) and yellow ends (Code 9).
   - *Presence:* the score is the share of long intervals (≥ 2 s) whose **off** falls in
     `[green_onset, green_onset + 6 s)`. That's the S-M1 recovery rule which found
     50/51/52 and 34/35/36.
   - *Count loop:* the share of ONs in `[green_onset, yellow_end + 3 s)`. Discharge
     happens only on that phase's green.
   - *Advance:* there's no green-locked signature (arrivals are spread over the cycle).
     Its phase comes from the chain (step 3).

   Pick the best phase. `phase_margin` = best − runner-up.
   **Concurrent phases (2/6, 4/8)** start green together in most cycles, so both
   score alike. Recompute the score on the **discriminating cycles** only: those where
   the two candidates' green onsets differ by ≥ 3 s (a lead/lag left, or one side's
   left turn gapping out early). If there are fewer than 20 such cycles, keep the tie
   and mark `low` confidence with a note. Don't guess.
3. **Lane chains (light traffic).** Restrict to hours with ≤ the 25th percentile of
   intersection actuations, so single vehicles are isolated. For each ordered pair
   (pulse/advance channel a, downstream channel b) in the same approach candidate set,
   build the lag histogram of `b_on − a_on` over `[0, 20] s`. A **sharp peak** (a
   modal-bin share of at least 0.3 of the matched pairs, with an IQR of 1.5 s or less)
   means a consistent travel time, so the two are the same lane. That orders the
   chain: advance → presence → count loop. Connected components give `lane_group`.
   An advance channel takes the phase of its chain's presence or count-loop member.
4. **Movement (`TM_*`).** Labels like `EBT` aren't inferable from events. Propose only
   the grouping: count-loop channels per phase, with right turns as a count-loop
   channel that has no presence or advance chain partner. Carry a configured `TM_*`
   label through when the channel already has one.
5. **Confidence.**
   - `high`: a clear mode, `phase_margin` of at least 0.3, and `n_act` of at least
     `min_actuations` (for advance, a chain peak meeting the step-3 bar).
   - `medium`: one of those is weak.
   - `low`: a concurrency tie, or no chain for an advance channel.

## Goldens (written before the build)

- **315 AM peaks (2025-12-15 → 17):**
  - 34/35/36 → `occupancy` P6 `high`, and 50/51/52 → `occupancy` P2 `high`.
  - 18/19/20 → `stop_bar` P6, and 26/27/28 → `stop_bar` P2.
  - 54/55/56 → `arrival` P2, and 38/39/40 → `arrival` P6 (via chains).
  - The diff shows these as `match`, and 33/41–44/47/49/57–60/63/37/53 as `new`.

  The test uses the corpus DB, so it skips when the DB is absent.
- **201, 2026-06-21:** channel 41 → `conflict`, and 60/63/64 → `silent`.
- **Synthetic (no DB needed):**
  - A presence zone that releases 1–3 s after its phase's green.
  - A count loop pulsing only in green.
  - An advance → presence → count chain with a 4 s and a 2 s lag.
  - A 2/6 pair with identical onsets: a tie, which gives `low`.
  - The same pair with 30 lead/lag cycles, which resolves.
  - An interval spanning a gap marker, which is excluded.

## Calibration (the verdict part)

Run on 201, 313 and 315 over their configured channels, and record per-rule
agreement with the config (excluding the known suspects). Report the confusion
matrix role × role and phase accuracy. Thresholds (presence cut, `phase_margin`,
the chain-peak bar) are frozen from that run, the same method as S-D2. 701 is the
holdout.

## Shell / CLI (Gemini-eligible once the core is in)

`atspm infer-detectors --target/--targetid/--all --start --end [--min-actuations]`.
It writes `Detector_Inference_{stamp}.csv` (proposal + diff) and prints the
`new`/`conflict`/`silent` rows. **It never edits `int_cfg.csv`.** The owner copies
accepted rows across by hand, as done for 315's presence zones.

## Feeds

- S-M1: fills `_Occupancy` at 701 and the remaining 315 phases.
- S-D2: the `UnconfiguredDetector` rule and configured-but-silent.
- S-M9: lane and movement typing.

## Owner decisions (2026-10-02)

1. **315 channels 37 and 53** are probably advance zones on the minor approaches, which were
   deliberately left unconfigured (the agency doesn't provide advance detection on minor
   approaches). The inference still proposes them, as `new`. The owner's choice not to
   configure them stands, and the proposal doesn't override it.
2. **Propose fixes:** every `conflict` or `new` row carries the inferred role and phase
   ("looks like P4 occupancy"), not just a flag. Nothing is written to `int_cfg.csv`.
