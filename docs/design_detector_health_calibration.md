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
| PeerRatio | `n_act / median(peer n_act)`, peers = same `TM_*` movement, else same phase + role | Outside lanes legitimately carry less: 315's 52 runs at ~0.5× its peers 50/51. Grid low ∈ {0.1, 0.2, 0.3} and high ∈ {3, 5}. Better: flag a change in a detector's ratio rather than the level. |
| BaselineDrift | `n_act` vs the trailing same-weekday median | **Can't be calibrated on this corpus** (at most 2 same-weekday samples per site). Build it with `min_history` (≥ 4 same-weekday days) so it stays silent here. Test it on synthetic data; calibrate when a longer pull exists. |
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

## 5. Decisions for the owner

1. **"Same unit" for failsafe:** all channels at the intersection by default,
   or a `WD:` key mapping each sensor to its channels? 201's 16-channel
   blocks (1–16, 17–32) suggest a block mapping.
2. **Severity scale:** `info` / `low` / `high`, as the roadmap sketches, or
   UDOT's single level?
3. **PM window** times for the `WD:` key.
4. **`min_observed_share`:** is a 90 % logged day judgeable?
5. **Deliberately unconfigured zones** (315's 37/53, now known dilemma-zone
   loops): suppress them through `WD_Ignore`, or configure them under a new
   role?
