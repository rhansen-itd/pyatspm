# Ideas carried over from the SPMs notebooks

Inventory of the half-built analyses in the old `~/SPMs` repo (GitHub `rhansen-itd/SPMs`,
private), written 2026-09-30 before SPMs was archived to the SSD. **Read this when picking
a new pyatspm feature**, or before you go digging in the archive. Source paths are
relative to `SPMs/Notebooks/`.

Everything in SPMs was triaged against pyatspm as of this date. §1 is what pyatspm
lacks. §2 holds the non-obvious details: the parts that a one-line description
wouldn't let you rebuild, quoted from the source. §3 lists what is already ported or
not worth keeping.

All SPMs code uses the old schema (`TS_start`, `Code`, `ID`, `Cycle_start`,
`Coord_plan`, plus `Duration`/`t_cs` from `comb_gyr_det`). It has no gap-marker
handling, and most of it iterates over rows. Treat it as a spec, not as code to port.

---

## 1. Features pyatspm doesn't have

| # | Idea | Source | Value | Notes |
|---|------|--------|-------|-------|
| A | **Split failures (GOR / ROR5)**: per cycle and binned | `spmfunctions/moes.py:81` `split_failures`, `bin_SF`; `Old Workflow/_Split_Failures.ipynb` | High | Standard ATSPM measure. Ran in production for the Eagle Rd study. See §2.1 |
| B | **Timing-plan history from controller events 131–149**: a table of coord plan, cycle, offset and splits each time a plan is selected | `Snippets/_Changes.ipynb` cells 10–11 | High | pyatspm reads only 131 (in `cycles.py:_merge_coordination_plan`). 132–149 are unused. See §2.2 |
| C | **Time-of-day schedule recovered from the data**: the plan in effect by weekday and time | `Snippets/_Changes.ipynb` cell 14 | Medium | Gives a check of the controller's TOD table against what actually ran. See §2.2 |
| D | **Preemption suite**: EVPE counts by bin and by weekday×hour; a per-cycle preempt table with cycles ±1–3 flagged; how much green the coordinated phases lose and how fast coordination recovers; green share 3 cycles before vs. after | `Snippets/_Pre-empts.ipynb` | High | pyatspm only *plots* code 105 on the termination chart. See §2.3 |
| E | **Preempt impact on corridor speed**: INRIX 5-min speeds before vs. during/after each preempt episode, with a paired t-test by preempt channel and direction | `_Pre-empts.ipynb` `pe_before_after_speed`, `read_inrix` | Medium | Needs the INRIX feed, so it may belong in `~/Inrix` instead. See §2.4 |
| F | **"Busiest preempt hours"**: the top-N *non-overlapping* 60-min windows | `_Pre-empts.ipynb` cells 33–35 | Low–Med | A greedy pick that never takes two overlapping windows. See §2.4 |
| G | **Before/after study regression**: `MOE ~ Vol + BA + BA:Vol` (OLS) for travel time, AoG% and SF%, by coord plan and day type | `moes.py:251` `combine_aog_sf_tt`, `moes.py:372` `tt_vol_anova`; `Old Workflow/_Travel_Time.ipynb` | Medium | Controls for volume when you evaluate a timing change. See §2.5 |
| H | **Unused green from extension-timer gap-out (code 13)**: time from a phase's code 13 to its actual termination or to the coordinated yield (151) | `Snippets/_Termination_Phase Call.ipynb` cells 9–12 | Medium | Code 13 logs the moment a phase *would have* gapped out, even when a force-off or max-out ends it later. See §2.6 |
| I | **Termination mix as percentages** per phase, per bin, per coord plan (GO/MO/FO ÷ total) | `_Termination_Phase Call.ipynb` cells 3–8 | Low | A table to go alongside the existing termination plot |
| J | **Sensor-fault / crosstalk detection**: timestamps where ≥N channels wired to the *same physical sensor* turn on in the same 0.1 s | `_more_data_exploration.ipynb` cell 9 `find_simultaneous_events_by_sensor` | Medium | Built for the Banks-Lowman radar faults (output file `..._crx_fail.html`). See §2.7 |
| K | **Inferring which detector calls which phase** from data: count how often a code 43 (phase call registered) shares a timestamp with a code 82 | `Snippets/_Detectors.ipynb` cells 10–11 | Medium | A first step toward the ROADMAP item on inferring lane configuration automatically. See §2.7 |
| L | **Speed between advance-detector stations**: per cycle, Δt between consecutive code 82s where `ID == prev ID − 1` (detectors numbered downstream-descending, e.g. 61→60→59) | `Snippets/_Detectors.ipynb` cells 4–8 | Low | Crude: it assumes consecutive actuations are the same vehicle. It worked for spot checks |
| M | **Arrival time into red** (`t_r`) for stop-bar detectors, counting only actuations while the phase is R/Rc | `_data_exploration.ipynb` cell 21 `process_dataframe` | Low | Unfinished. The header comment is about switching count detectors to presence |
| N | **Platoon ratio** (`AoG% ÷ G%`), binned, across a corridor and over before/after date pairs | `_data_exploration.ipynb` cells 13–18 | Low | Trivial on top of `AogEngine`. Worth adding as a column |
| O | **UTDF volume export** for Synchro from turning-movement count CSVs | `UTDF-Volumes.ipynb` | Low–Med | ⚠ The header it writes (`[UTDF VOLUME FILE]`, `VERSION=1.0` …) looks made up. Check it against Synchro's real UTDF spec before reusing |
| P | **Video capture helpers**: record RTSP clips in scheduled windows and name each by **OCR of the camera's burned-in clock** (EasyOCR, top-left ROI, `HH:MM:SS AM/PM`); a periodic-snapshot timelapse mode; download event clips over HTTP from the camera's `/vehicle_event/*.mp4` | `_record_mp4.ipynb`, `_more_data_exploration.ipynb` cell 10 | Low–Med | OCR on the camera clock is a **second route** for the ROADMAP item on automatic video/data sync: it gives the video's wall-clock time without detecting a phase change |
| R | **ACHD CSV ingestion**: event-log CSV exports from the ACHD portal into one SQLite DB per intersection, with containment filtering, a continuity report and gap markers | `spmfunctions/process_csv_db.py` (newest); older path `spmfunctions/process_achd.py` + `read_data.py:read_df_raw` / `fix_df_raw_ACHD` | High (if ACHD data is wanted) | **pyatspm ingests `.datZ` only** (`data/ingestion.py`), so there is no ACHD path today. `process_csv_db.py` already writes almost exactly pyatspm's `events` schema. See §2.8 |
| S | **ACHD portal scraper**: bulk export of event CSVs per intersection in half-month windows, with a present/missing matrix so a run can resume | `Download_ACHD.ipynb` | Medium | It works by replaying recorded **screen coordinates** with pyautogui, which is fragile. Rebuild before reuse (§2.8) |
| Q | Small one-offs: split percentiles (85/90/95/99) by plan and phase; phase-omit (46) counts by bin; checking whether 150 param 3 coincides with 175/200 alarms | `SPM_snippets.ipynb`, `_z_Snippets.ipynb`, `_Pre-empts.ipynb` cell 7 | Low | Each is a few lines. Listed so nobody goes looking for them |

---

## 2. The non-obvious details

### 2.1 Split failures (`moes.py:81`)

- **GOR window = Begin Green (1) → Begin Red Clearance (10)**, so it **includes yellow**.
  **ROR window = first `t_red`=5 s after code 10.** The ROR window matches Purdue/UDOT.
  The GOR window does not: the standard measures GOR over green only (1→8). Decide on
  purpose which one to use.
- A cycle is a split failure when `GOR > 0.8 and ROR > 0.8`.
- Detector edge cases it handles. Keep these in a port:
  1. Consecutive duplicate 81/82 events are collapsed first (`df_d[df_d.Code != df_d.Prev]`).
  2. A detector already **on at the start of the window with no transitions inside it**
     gets a synthetic 82 that covers the whole window: the occupancy state is
     forward-filled into the phase events, then rows with `Code==start & Next==end & Det==82` are kept.
  3. If the first detector event inside the window is an **81**, a synthetic 82 is added
     at the window start, running until that 81.
  4. An actuation that runs past the window end is clipped to the end
     (`TS_end = TS_next` when `Next == c_end`).
  5. Occupancy is capped at the window length (`min(occ, d)`).
- `bin_SF` sums **occupied seconds and window seconds**, then divides. Binned GOR/ROR is
  therefore time-weighted, not a mean of per-cycle ratios. `SF_pct = SF cycles / cycles`.
  It is computed per phase × coord plan.
- The stop-bar detector came from the `dets.csv` "Det 1" (or "Det 2") stop-bar row,
  one detector per phase. A port should OR together every stop-bar channel of a phase.

### 2.2 Timing-plan history (`_Changes.ipynb`)

```python
df_cp = df_raw[df_raw.Code.isin(range(131,150)) & ~df_raw.ID.isin([0,10,254,255])] \
          .drop_duplicates().set_index('TS_start')
df_cp = df_cp.pivot_table(index=df_cp.index, columns='Code', values='ID', aggfunc='first')
# 131 plan, 132 cycle, 133 offset, 134..149 = split for phase 1..16  (so split of phase p = 133+p)
```

- **The trick:** when a plan is selected, the controller logs 131–149 **spread over
  several tenths of a second**. The pivot therefore yields several partial rows, not one.
  `merge_complementary_rows_chain` folds consecutive rows together until two rows
  disagree on a column they both have. That rebuilds one complete parameter row per plan
  selection:

  ```python
  conflict = current.notna() & row.notna() & (current != row)
  if conflict.any(): emit(current); current = row
  else: current = current.combine_first(row)   # keep the first row's timestamp
  ```

- Rows without a 131 are dropped afterwards. Each remaining row gets a version label
  `plan + k/100` (1.01, 1.02, …).
  - **Improvement:** bump the version only when the parameter set *changes*. As written,
    every daily re-selection gets a new number.
- ⚠ **Bug, don't copy:** the `~ID.isin([0,10,254,255])` filter is meant for 131
  (254 = free, 255 = flash), but it runs on *all* of codes 131–149. A legitimate
  **10 s split or 0 s offset is silently dropped**. Filter 131 only.
- **TOD schedule (C):** resample the 131 events to 1 min and keep `first()` and `last()`
  of each minute. Group by weekday and keep rows where the plan differs from the
  previous one. The result is the (weekday, time → plan) transitions the controller
  actually ran.

### 2.3 Preemption (`_Pre-empts.ipynb`)

Event and parameter reference, from the notebook (cell 32). The 150 parameters are
**Econolite's 0–7 variant**; the Indiana PDF lists 0–6:

```
102 Preempt Call Input On   104 Preempt Call Input Off   105 Preempt Entry Started
107 Preempt Begin Dwell     111 Preempt Begin Exit Interval
102→105 delay (typ. 0 or 0.4 s) | 105→107 time to service | 107→111 (or 104) dwell
150 coord state: 0 Free, 1 In Step, 2 Transition-Add, 3 Transition-Subtract,
                 4 Transition-Dwell, 5 Local Zero, 6 Begin Pickup, 7 Master Cycle Zero
151 coordinated phase yield    152 coordinated phase begin (param = coord phase)
Eagle Rd channel→phase (site-specific): PE3=7&4 (SB), PE4=3&8 (NB), PE5=2&5 (WB), PE6=1&6 (EB)
```

- **`preempt_by_cycle`**: one row per cycle, holding counts of 105 per channel, a
  `Channels` string (e.g. `"3,5"`), and `Prev1..3` / `Aft1..3` (the `Channels` value
  shifted ±1–3 cycles). It also carries the codes-150 list for the cycle, and a
  `CP_change` flag on the plan-change cycle and its neighbours, so transition cycles
  can be kept out of the comparison.
- **Coordinated green-band recovery (cell 12).** This is the clever one. It measures
  how much of the coordinated phase's programmed green is actually delivered, cycle by
  cycle:
  1. coord phase per cycle = `min(ID)` of code 152 in that cycle
  2. programmed split = the most recent code `133 + coord_phase`, forward-filled
  3. local zero = code 150 with param 5; `C` = time to the next local zero
  4. green start/end for the coordinated phase (code 1 → code 10), in seconds from local zero
  5. drop false cycles: `|Start_s| < C`, `End_s > 0`, `|C| < 1000`
  6. clip to the band `[0, Split]` and set `G_band_pct = clipped green / Split`
  7. compare `G_band_pct` for cycles that **have** a preempt, for cycles with **Prev1 = that channel and no preempt now**,
     and for cycles with **Prev2 = channel**. This shows how many cycles coordination takes to recover.

  This **assumes the offset is referenced to the start of coordinated green**, i.e. local
  zero = coordinated green start. It also uses the full split as the band width, and
  yield comes before split end by Y+R. Revisit both if your offset reference differs.
- **Green share before/after (cells 44–46):** green+yellow per phase ÷ cycle length,
  over the preempt cycle plus 2 after vs. the 3 cycles before. Grouped by preempt channel and plan.

### 2.4 Preempt × INRIX speed (`pe_before_after_speed`) and busiest hours

- Episodes are built from 5-min bins. A bin counts as active if it has a 105 or the
  previous bin did. Each episode starts with the inactive bin just before the activity,
  which serves as the "before" sample. The speed before an episode is that bin; the
  speed after is the mean of the episode's active bins.
- Episode type codes: single bin + single channel = `n`; single bin + several channels = `9`;
  several bins + single channel = `n*11`; several bins + several channels = `99`.
- Per type and direction: `scipy.stats.ttest_rel(before, after, nan_policy='omit')`.
- **Busiest hours (F):** take the rolling 12-bin sum (12 × 5 min = 1 h) and rank windows
  by it. Walk down the ranking and zero the 11 bins on each side of each pick, so the
  top-N hours never overlap.

### 2.5 Before/after regression (`tt_vol_anova`)

- Join hourly AoG, SF and counts with the INRIX segment travel time for the phase's
  direction (`INRIX_key.csv` maps intersection+phase → segment).
- Volume for a direction is through + right turns (`{dir}T + {dir}R`), or an ATR count when one is available.
- Day groups by plan: plans 1 and 2 → Tue–Thu; 4 and 6 → Sat; "4,6" → Sun.
- For each of TT, AoG_pct and SF_pct: `ols('{v} ~ Vol + BA + BA * Vol')`. Report R², plus
  coef and p for `Vol`, `BA`, `BA:Vol`, and Δmedian TT and Δmedian Vol. A significant
  `BA` with an insignificant `BA:Vol` reads as a real shift that doesn't depend on volume.

### 2.6 Extension-timer gap-out (code 13)

- Indiana code 13 = *Extension Timer Gap Out*. It fires when the passage timer
  expires, **even when the phase isn't allowed to terminate**: the other ring is
  holding the barrier, the phase is coordinated, or a minimum or ped interval is
  still timing.
- Cell 9 finds cycles with code 13 on phases 2 and 6 *and* a force-off on 2. That
  catches coordinated phases that are gapping out yet held to their force-off.
- Cell 12 computes `151 (coordinated yield) − first 13 on phase 8` per cycle. That is how
  long phase 8 sat with no demand before it was released. The same pattern gives an
  **"unused green after gap-out"** measure for any phase: the time from 13 to that
  phase's 4/5/6/8.

### 2.7 Detector diagnostics

- **Simultaneous on-events (J):** a sensor maps to `(min_count, channel_ids)`. Keep the
  code-82 events on those channels, run `groupby('TS_start').ID.nunique()`, and flag
  timestamps where the count ≥ `min_count`. Real vehicles almost never trip several lanes
  of one radar or video sensor in the same 0.1 s, so a hit means the sensor reset or
  faulted. The Banks-Lowman settings were
  `{'Sensor1': (4, (33,38,39,40,42,43,45)), 'Sensor2': (5, (46,47,48,49,53,55,60,61,64)), 'Sensor3': (3, (34,35,36))}`.
- **Detector → phase inference (K):** group codes 43 and 82 by exact timestamp. Rows
  whose sorted code list is `[43 82]` pair a call with the actuation that placed it,
  and `value_counts()` of the (phase, detector) pairs gives the mapping. This only
  works while the phase is *not* already being called, so use a quiet period. Delay
  and extend detectors won't show up.

### 2.8 ACHD data: export format, ingestion, scraping

**Export format.** There are two vintages; the code handles both.
- *Current* (`process_csv_db.py`): files are named `{IntID}_Events_*.csv`. Line 2 holds
  the export start time and line 3 the end time, each **quoted**, in the form
  `"%A, %d %B %Y %H:%M:%S"` (e.g. `"Friday, 01 August 2025 00:00:00"`). Split on `"`,
  because the lines have trailing commas. After `skiprows=4` comes a header row:
  `Event Time, Event Code, Event Description, Event Parameter`. Strip whitespace from the
  column names. Event Time is `%m/%d/%y %H:%M:%S.%f` in **local time (US/Mountain)**.
- *Older* (`process_achd.py`, the `Download_ACHD` naming): `{IntID}_Events_{YYYYMMDD}T0000.csv`,
  one file per **half month** (1st–15th, 16th–end). Skip 5 lines. The columns are
  TS, Code, Description, Param, with no header. Miovision-sourced files start with
  `Signal,` and also skip 5 lines (`SPM_snippets.ipynb` cell 8).
- Intersection number → name: `_ACHD_int_key.csv`. For the Eagle Rd corridor:
  `{Chinden:250, Colchester:392, Fairview:213, Franklin:270, HobbleCreek:361, I84EB:339, I84WB:269,
  IslandWoods:331, McMillan:272, Pine:323, Riverside:325, RiverValley:423, StLuke:201,
  Ustick:271, Village:480, Wainwright:430}`. Downloads also used 341, which is not in this map.

**Ingestion (`process_csv_db.py`), the template for a pyatspm `ingest-csv` path:**
1. Group files by the `IntID` prefix. Read each file's start and end from its header
   lines only; the body isn't needed for this.
2. **Drop files whose [start, end] lies wholly inside another file.** Re-downloads and
   overlapping exports are common. *Partial* overlaps are left to the DB's
   `UNIQUE(timestamp, event, param) ON CONFLICT IGNORE`, the same constraint pyatspm uses.
3. Sort by start. If `start − prev_end > 1.5 s`, it is a discontinuity: insert a
   **`(-1, -1)` gap marker at the file's start**. That is the same gap-marker
   convention as pyatspm. The 1.5 s slack absorbs the boundary between daily files.
4. Write `{IntID}_data_ranges.json`, listing each file's span and whether it continues
   from the previous file.
5. Convert local timestamps to UTC epoch.
   - ⚠ It uses `pytz.localize` with the default `is_dst=False`. In the fall-back hour,
     both passes of 01:00–02:00 map to the same epochs, so events interleave and some
     de-duplicate wrongly.
   - A port should use `resolve_pytz` and decide what to do with ambiguous times. One
     option: detect the backward step in file order, the way `_fence_clock_steps` handles it.
- Differences from pyatspm's schema: table `logs` vs. `events`, and column
  `event_type` vs. `event_code`. Nothing fills `ingestion_log` or `metadata`.

**Cycles.** ACHD logs have no reliable code 31 barrier events. The old code inferred a
cycle start where a code-1 group that is all in {1,2,5,6} follows one that is all in
{3,4,7,8} (`read_df_raw`, the ACHD branch). pyatspm's
`cycles._detect_cycles_from_config` fallback already does the general form of this from
the ring-barrier config. ACHD sites therefore need `RB_*` filled in `int_cfg.csv`, and
otherwise need no new cycle code.

**Scraper (`Download_ACHD.ipynb`):**
- How it works now: `collect_coordinates` records the screen positions of
  "Intersection Selector" and "Export Excel Button", plus calendar cells (1st, 15th,
  16th, last day, Apply, next/previous month). The positions are recorded by hovering
  and pressing the spacebar, and saved to CSV.
- For each intersection it clicks the selector, types the number, presses Enter, waits
  10 s, then clicks Export. It polls the Downloads folder for the expected filename and
  gives up after 120 s.
- Month loop: 1st–15th, then 16th–last, then the next month.
- `find_missing_files` builds a date × intersection Present/Missing matrix. Passing it
  as `missing_df` skips exports that are already present, so a run can resume.
- **Why rebuild it:** the coordinates break whenever the screen resolution, zoom or
  layout changes. They also tie up the desktop while the run goes.
  - **Better option 1:** Playwright with DOM selectors. It can run headless, and it saves
    downloads through `page.expect_download()` instead of polling a folder.
  - **Better option 2:** first look in the browser's network tab for the request the
    Export button sends. If it is a plain authenticated GET/POST taking an intersection
    and a date range, call it directly with `requests` and skip the UI.
- Keep the Present/Missing resume logic and the half-month windowing. The windowing
  seems to be a portal limit on export size; confirm it before keeping it.

---

## 3. Already ported or not worth keeping

| SPMs piece | Status |
|---|---|
| `Snippets/_Headway.ipynb` (`flow_rate`, `rate_dfs`, `plot_flow_plotly`) | **Ported** → `analysis/flow.py` (`flow_rate`, `rate_profiles`), `plotting/flow.py` |
| `moes.arrival_on_green`, `bin_AOG`, `Old Workflow/_AOG.ipynb` | Ported → `AogEngine` |
| `counts.py` (`det_counts` with exclusions, `ped_counts` deduplicated per walk) | Ported → `CountEngine` |
| `plotting.plot_coord` / `ring_dfs` / `add_r1r2`, `plot_term` | Ported → `plotting/coordination.py`, `plotting/termination.py`, `analysis/cycles.py` |
| `plotting.create_detector_comparison_plot` | Ported → `plotting/detectors.py` / `DetectorEngine` |
| `Video_Overlay.ipynb`, `video_processing.VideoProcessor` | Ported → `atspm/video/` |
| `video_processing.EnhancedIOUTracker` (CV turning-movement counting) | Already a ROADMAP *Future features* item |
| `datz_decoder.ipynb`, `process_datz*.py`, `_Datz_SCP.ipynb` | Superseded by `analysis/decoders.py` and `data/retrieval.py` |
| `_convert_det_int_cfg.ipynb` (dets.csv → int_cfg.csv) | One-time migration, already done |
| `_archive.ipynb` (zips `.datZ` files in `Archive/` dirs and moves them to the G: drive) | Personal file housekeeping |
| `_process_achd.ipynb`, `_batch_process_CSV.ipynb`, `read_data.process_csv_files` (CSV → bz2 pickles), `processed_dbs/read.py`, `processed_dbs/de_duplicate.py` | The old pickle pipeline, plus one-off DB checks. The *ACHD logic* inside them is kept as items R/S and §2.8. The pickle plumbing is not needed |
| `SPM-GUI.ipynb` / `spm_gui.py` (Tk front-end) | Replaced by the CLI |
| `misc_tools.filter_*`, `phase_status`, `detector_status`, `overlap_status` | Replaced by `reader.py` and `analysis/video.py` status lookups |
| `Old Workflow/_Counts.ipynb` ped-count pivots (month × weekday × hour) | Report formatting only |

**Archiving note:** `_Datz_SCP.ipynb`, `_record_mp4.ipynb` and
`_more_data_exploration.ipynb` contain **plaintext device credentials**: controller SSH,
the EVO radar, and the Axis camera RTSP and HTTP logins. They are in git history too. The
GitHub repo is private, but the archive copy will carry them as well.
