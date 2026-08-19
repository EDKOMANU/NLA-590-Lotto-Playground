# Ghana NLA 5/90 Prediction System — Build Report

## Plan Discovery Engine (`--mode plans`, Jul 2026)

The engine that learns **plans** rather than voting by similarity. `trend`/`ncc` find weeks that look like the current one and credit whatever dropped after them — a vote with no mechanism, which can never answer *"how did you get this number?"*. A plan answers it.

**A plan = (MATCHER, LINK, LAG)** — combining *temporal* behaviour (how numbers travel week to week, up to ten weeks back) with *spatial* structure (where they sit on the chart grid, how the draw is spaced, which slot they came from).

- **Matcher** — what makes a past situation *like now*. **The initial step is pair-anchoring**, the way the reading actually starts: take a pair of the numbers just drawn and hunt the old papers for the weeks where that same pair came out together (`pair`), then narrow those anchored weeks further to the ones *built* like this one (`pair_structure`). `triple` demands 3+ shared numbers — usually too rare to learn from, and the report says so instead of silently dropping it. Each anchored match records **which pair anchored it**, and the report names those pairs. **Structural (identity-independent):** `structure` matches how a week is BUILT — the four gaps between sorted numbers, span, decade clustering, odd/even balance, the draw-order permutation (relative positioning), and interaction behaviour (internal pairs sharing a terminal, a digital root, or being literal neighbours) — so two weeks sharing *no numbers at all* can match, and two sharing four numbers may not. `profile` matches the joint condition profile (total band AND odd/even AND spread AND repeat). **Surface (retained deliberately as the baseline):** `image` (cross-correlation of the binary chart picture) and `overlap` (≥2 shared numbers) both score literal shared cells — the report shows best-structural vs best-surface side by side, so if structure isn't earning its keep, that's visible rather than hidden. `any` — **no filter, the control**.
- **Link** — the mechanism a dropped number arrived by, from a source week **1–10 weeks back**: literal carry, machine→win crossover, the calculation family (double, mirror, one-up, one-down, turning/digit-inversion, pair sum, pair difference, triplet sum), all nine traditional charts, the last-digit / digital-root families, **grid neighbours** (±9/±10/±11 — the cells physically adjacent on the 9×10 printed chart, i.e. genuinely spatial rather than merely arithmetic), and **positional dynamics** (`pos1`…`pos5`: does the number that sat in sorted-position *p* come back — the sharpest, one-number plans).

**Output, as specified:** **A. Primary plan** (step-by-step MATCH → LOOK → BACKTRACE → MEASURE → APPLY → RESULT, plus the backtrace table of real past weeks), **B. Secondary plans** (every qualifying runner-up with its own numbers, flagged structural or surface), **C. Plan comparison** (why the primary won and how each runner-up differs), **D. Every predicted number traced** — no pick is shown without a derivation naming the mechanism and the source draw it came from.

**How a plan is learned** — match situations → look at what dropped the week after → **backtrace each dropped number** to a source week via the link catalogue → the recurring origin→drop mechanism *is* the plan → replay it on the current window for this week's numbers. The backtrace and the scoring are the same computation seen from two sides, so the evidence trail is free: the UI shows, per plan, the literal past weeks it was learned from (*what dropped · which of those this plan explained · which earlier draw it traced back to*).

**The honest measurement.** A plan proposes a candidate set; its yield = candidates that dropped ÷ candidates proposed. Because E[hits] = |candidates| × 5/90 exactly, **the chance baseline is 5.56% for every plan regardless of how broad or narrow it is** — which is what makes a 2-number plan and a 40-number plan directly comparable. Yields are Wilson-shrunk by proposal count before scoring.

**Walk-forward backtest (7,966 test draws, all 7 games, plans re-learned from scratch at every test point):** 5 picks ≥1 hit **25.96% [25.0–26.9]** vs 25.37% random; 2 picks 11.37% [10.7–12.1] vs 10.86%; **AUC 0.500**. The most elaborate reasoning engine in the project — 1,600 plans, structural matching, ten-week backtracing, full derivation trails — lands exactly where every other strategy landed. That is the honest result, and the engine reports it rather than burying it.

**Bug found and fixed (Jul 2026) — the "turning" derivation went off the board.** Digit-inversion turns 19 into 91, which does not exist in a 1–90 game. The arithmetic `turning` link (and the spatial engine's `_op_turning` key operator) had been wrapping those modulo 90, so 19 silently produced **1** — a number with no reading relationship to 19 at all, manufactured purely by the arithmetic. That affected the eight numbers ending in 9 (19, 29 … 89). Both now produce **no derivation** for those numbers, which matches the traditional turning chart (`charts.TURNING` maps 19→19, a self-pointer, which this project excludes from scoring everywhere). Verified: `turning` now agrees with the traditional chart on every valid case (5→50, 23→32, 32→23, 50→5, 90→9), and a full sweep confirms **every link, on every draw in the 8,667-row archive, emits only numbers 1–90**. User-facing labels no longer display impossible numbers either ("mirror (91−n)" → "the number's mirror across the board — 23 becomes 68").

**Two guards, both load-bearing:**
- *Anti-echo*: `carry`/`mach_carry` at lag 1 are literally last week's numbers. Measured and displayed, **never allowed to pick** — the same quarantine `lap`/`positional` live under. Measured result: top-5 overlap with last week's draw = **0/5** on both MS and NW.
- *Look-elsewhere*: the engine evaluates **840 plans**, so the winner is guaranteed to look brilliant. `bootstrap_best_plan_pvalue()` re-runs the entire search on structure-destroyed synthetic charts. **Measured: NW p = 0.53 (pure noise); MS p = 0.05** — and even that is reported as *borderline*, because at 40 iterations the p-value resolves no finer than 0.025 and testing 7 games makes one borderline result per sweep the expected outcome. The engine says this in the UI next to its own best plan.

## Weekly refresh rebuilt (Jul 2026) — ghanayello went behind Cloudflare

`predictor.py update` / the app's **Fetch latest draws** button stopped working: ghanayello.com now sits behind Cloudflare bot protection and returns **HTTP 403** to plain requests (confirmed: `server: cloudflare`). Worse, the raw `HTTPError` escaped and crashed the whole Streamlit app — a third-party site going dark should never take the tool down.

`data.update()` is now **multi-source and fault-tolerant**:

- **Primary: theb2blotto.com's results endpoint** (`/ajax/get_latest_results.php?pn=N`) — no bot protection, carries machine numbers, covers all seven NLA games. Non-NLA games it also lists (B2B, Noon Rush, NLA VAG, Alpha) are deliberately unmapped, keeping the archive to the classic seven.
- **Fallback: ghanayello**, retained for the day its protection lifts.
- Each source is tried independently; one failing contributes nothing but never aborts the run. `UpdateError` (with a plain-language message) is raised only when **no** source can be read, and the app shows it in the sidebar instead of a traceback.
- Draws are deduped across sources and appended only when the `(date, game)` pair is absent, so repeat runs and overlapping sources cannot create duplicates. Every row must still pass the original acceptance rules: 5 distinct numbers in 1–90, and **the date's weekday must match the game's fixed draw day**.

**Verified before trusting it:** of the 10 draws where the new source overlapped the existing archive, **10 of 10 matched exactly** — independent confirmation of accuracy. The refresh then added 7 genuinely new draws (21–27 Jul), leaving **8,675 rows, 0 duplicates, 0 invalid sets**, current through 2026-07-27. A second run correctly added 0.

## Data v3 (Jul 2026): the 64-year archive

**`ghana_lotto_history.csv` now holds 8,666 clean draws from 29 Sep 1962 to 18 Jul 2026** — sourced from the digitized NLA chart archive (`ghana_hybrid_scraped.csv`, with event numbers from National Weekly's draw #1) merged with the verified GhanaYello era. Provenance: the Kaigee app's static `data.json` holds only Nigeria private-operator games; its Ghana NLA archive is loaded dynamically into a virtualized AG Grid, so it was extracted with an in-browser scroll-and-paginate scraper (harvesting the virtualized rows into a `Map` keyed by row-index, page by page, then sorting by index — the only way to capture a grid whose data never appears in a fetchable payload). Per game: NW 3,325 (1962→), MW 1,179 (2003→), MS 1,043 (2005→), LT 984 / FT 983 / FB 945 (2007→), SA 207 (2022→). Machine numbers from the 1970s onward (only the 1960s lack them). The raw Kaigee scrape (`kaigee_history.csv`) holds ~40 private-operator games (PG/PM/GC) — deliberately NOT merged into the NLA series.

**Repair pass** (backup: `ghana_lotto_history.pre_repair_backup.csv`): 13 conflicting duplicate groups and 93 wrong-weekday rows were resolved by (R1) arbitrating modern conflicts against the verified GhanaYello dumps — 4 divergent variants dropped; (R2) both-neighbor event-cadence re-dating — 55 mis-stamped dates moved to the slot both event-neighbors agree on; (R3) launch-segment chain reflow — 16 launch draws that had been stuffed onto one date re-dated backward from the first defect-free event (reconstructed, cadence-implied dates, so flagged in the log); no defect-free attested row was ever moved. Result: 0 duplicates, 0 invalid rows, 30 wrong-weekday rows kept as genuine historical anomalies.

**Consequences**: all rf/gbm/mlp/deep artifacts are stale twice over (the extended feature set grew from 37 to 53 columns with per-chart features, and the data tripled) — re-run `train.py` before using the ML modes. Historical-era heterogeneity is real (different decades, machines, procedures) — the recency-weighted components handle this naturally; the all-time-pooled rates (charts, transforms) now average over 64 years, which is a caveat to keep in mind when reading them.

## Kaigee/Lottobrains strategy port (Jul 19 2026)

Studied the Kaigee forecasting app's architecture (Angular UI over a static archive; three prediction web-workers). Its "Number Tree" plans engine is week-aligned window matching (already covered by our NCC + trend matchers). Two genuinely new strategy families were ported — **re-implemented with measured rates instead of Kaigee's heuristic point-scores** (theirs: `chainLength*15 + bonuses`, capped at 99%, never compared to chance):

- **`yearly`** (pattern engine, 12th component): anniversary recurrence — did the number appear within ±7 days of the upcoming draw's calendar date in previous years? Measured per number across up to 63 eligible years, Wilson-shrunk, with the honest chance baseline disclosed (a ~2-draw window gives every number ~11.5% per year by luck). Strongest case in 64 years of NW: number 58 at 13/63 years (21%) — notable, and exactly the size of fluctuation 90 numbers × 63 trials produces.
- **`periodic`** (spatial engine, 8th component): the "Counting Weeks" reading — a number that appeared at draws t, t+g, t+2g… is predicted at the next term. Every comparable historical progression's completion was measured first: **pair-chains 5.51% (n=132,520), established 3+-chains 5.71% (n=7,518) vs 5.56% chance**. The most decisive single measurement in the project: six-figure trial counts showing draw periodicity does not exist — delivered by the very component that implements the strategy, with each firing chain's dates and measured rate in the evidence panel.

### The 64-year walk-forward backtest (Jul 19 2026) — the definitive measurement

**7,966 walk-forward test draws per strategy** (4× the previous test set; margin of error now ~±1.0pp). Every live strategy, ≥1 hit with 5 picks vs **25.37% random chance**:

| Strategy | 5-pick ≥1 hit (95% CI) | AUC | Strategy | 5-pick ≥1 hit (95% CI) | AUC |
|----------|------------------------|-----|----------|------------------------|-----|
| hot | 25.73% [24.8–26.7] | 0.501 | charts | 25.18% [24.2–26.1] | 0.501 |
| recent | 25.76% [24.8–26.7] | 0.501 | charts2 | 25.40% [24.5–26.4] | 0.501 |
| overdue | 25.30% [24.4–26.3] | 0.501 | **pattern** | 25.62% [24.7–26.6] | 0.501 |
| blend | 25.90% [24.9–26.9] | 0.501 | **spatial** | 25.07% [24.1–26.0] | 0.500 |

Re-measured after adding the Kaigee-inspired components ('yearly' anniversary recurrence in pattern, 'periodic' counting-weeks progressions in spatial): pattern 25.77% [24.8–26.7] AUC 0.501, spatial 25.09% [24.2–26.1] AUC 0.499 — unchanged, at chance.

Every confidence interval straddles the chance rate; AUC is pinned at 0.500–0.501 for all eight strategies. Sixty-four years of draws, ~8,000 independent tests, eleven pattern components, seven spatial strategy families, and the upgraded chart engine all agree with the theory: **NLA draws are memoryless — no reading strategy, however sophisticated, moves the needle**. This is the strongest evidence the project has produced, and it is exactly what an honestly random 5/90 game must yield.

## Pattern Analysis v3 (Jul 2026): the full paper-reading strategy set

The `pattern` mode now scores **11 components**, all measured live from history (no training), all Wilson-shrunk by sample size, all auditable in the app's "Show your work" panel:

| Component | The paper-reading strategy it encodes |
|-----------|--------------------------------------|
| `pattern_trace` | Current pairs/triplets traced to their historical repeats and what won after ("how many weeks down"), with sorted-rank position tracking |
| `transform` | The "addition" technique generalized to a measured rule registry: double, mirror, **one-up, one-down**, pair sum/difference, triplet sum/mean |
| `charts` | The 9 traditional chart relationships (Bonanza, Counterpart, Turning, …), weighted by measured transfer rate |
| `mach_to_win` | "Machine numbers foreshadow winners" — per-number measured transfer rate |
| **`trend`** *(new)* | **Trend-similarity tracing**: the last 3 draws profiled as a trend (sums, spans, parity, decade/terminal distribution, internal repeats + the literal numbers); the 25 most similar *non-overlapping* older windows vote with what won immediately after each. An echo guard keeps windows/follow-ups strictly clear of the present |
| **`conditional`** *(new)* | **Patterns under different conditions**: every number's next-draw rate measured separately under the current draw's own conditions — sum band, parity profile, span band, repeat-from-previous (4 fixed, pre-declared dimensions; tercile bounds derived walk-forward from history only) |
| **`cross_game`** *(new)* | **"Today's results point at tomorrow's game"**: numbers drawn in other games since this game's last draw, weighted by each source game's measured transfer rate into this game |
| **`mach_trace`** *(new)* | Machine pairs/triplets traced against historical *machine* draws, crediting what won after each repeat |
| `terminal` / `group` | Last-digit terminal and digital-root groupings (crediting other members, never self) |
| `recent` | Recency-weighted frequency — the disclosed baseline the structural components are judged against |

Component weights are re-derived each call by `dynamic_weights()` (which components actually landed score-mass on real winners in the last 30 draws). `positional`/`lap` stay diagnostic-only (excluded from the score — they structurally echo last week's draw; see Methodology).

**Measured honestly, as always**: the new one-up/one-down rules transfer at 5.5%/5.7% (chance 5.56%), and cross-game transfer rates measure 4.8%–6.2% into MS across ~1,900 trials per source game (chance 5.56%) — the traditions are real as reading practices, but nothing here beats chance.

**v3 walk-forward backtest (2,002 test draws, all 7 games, full 11-component blend):**

| picks | ≥1 hit (95% CI) | ≥2 hits | random ≥1 | random ≥2 |
|-------|-----------------|---------|-----------|-----------|
| 2 | 10.74% [9.5–12.2] | 0.30% | 10.86% | 0.25% |
| 3 | 16.43% [14.9–18.1] | 0.75% | 15.93% | 0.73% |
| 5 | 26.07% [24.2–28.0] | 2.35% | 25.37% | 2.33% |

AUC 0.497 (chance = 0.5). **Anti-echo check** (312 sampled walk-forward points): top-5 picks overlap last week's own numbers at 0.292/5 and the actual next draw at 0.282/5 — both within sampling noise of the 0.278/5 chance level, so the new components forecast (at chance) rather than replay the previous draw. Conclusion unchanged: every reading strategy, now including trend-similarity, conditional, cross-game and machine-trace, sits exactly at random chance — as it must for an honestly random 5/90 draw.

## Spatial Pattern Matching Engine (`--mode spatial`, Jul 2026)

Implements the spatial-engine blueprint: the chart as a two-channel binary image (weeks × numbers, winning + machine channels in `lottery_core/spatial_engine.py`) plus the dense positional matrix, with five strategy families and Monte Carlo noise control:

- **NCC template matching ("the Plan")**: the current 3-week window slid backwards through history; the draws after the top-25 most-similar windows vote, weighted by score. *Spec correction, measured:* for binary 5/90 windows NCC = overlap/window-mass, so real scores top out ≈0.17 (MS history) — **a 0.85 threshold can mathematically never fire**; the engine uses top-K with every score disclosed. An echo guard keeps candidate windows and their "drop" rows strictly before the current window.
- **Diagonal trajectories**: completion rates measured per step vector Δc∈{±1,±2}: 5.2%–7.4% on MS (chance 5.56%) — noise-level; partial runs ending at the current draw are extrapolated at the Wilson-shrunk measured rate.
- **Box enclosures**: predictive form — historical pair-partner of a current number appears within 5 draws at a measured **24.9% vs 24.9% chance** (26,830 trials on MS): the tradition measured *exactly* at chance. Partners already in the current draw are disclosed but never scored (echo guard).
- **V-shapes**: detected and reported honestly — **zero full V-shapes exist** in MS's entire 402-draw history (expected at 5/90 density).
- **Key Identification Engine**: ~1,500 positional equations (turning/digit-inversion, mirror, ±K, double, pair sum/diff — identity excluded as the known lap-echo channel) measured and screened by exact binomial test vs the 1/90 slot chance. On MS: **16 keys passed the 0.01 screen — vs ~15 false positives expected from pure noise**, and the family-wise Monte Carlo bootstrap (best real key vs the best key each of N synthetic structure-destroyed charts produces) is the decisive test, run via `python predictor.py keys <GAME>`.
- **Machine→winning crossover**: lag-resolved transfer rates (τ=1..5): 5.1%–6.9% on MS vs 5.56% chance.
- **Strict joint-conditional cohorts** *(the "super" tier over the pattern system's conditional)*: every historical draw classified by its FULL condition profile — sum band AND parity AND span band AND repeat-from-previous — and each number's next-draw rate measured strictly within the cohort matching the current draw's profile (all dimensions at once, vs the pattern engine's independently-averaged marginals). On MS: 51 distinct profiles over 400 draws, median cohort 6, current profile matched 20× — strictness costs sample size, so rates are Wilson-shrunk by cohort trials (e.g. 4/20 = 20% raw → 8.1% scored) and the cohort size is disclosed everywhere.

Fixed disclosed blend weights (no auto-tuning): NCC 0.25, diagonal/box/conditional/keys 0.15 each, machine-crossover 0.10, V-shape 0.05. Backtestable as `python predictor.py backtest spatial`.

**Spatial walk-forward backtest (2,002 test draws, all 7 games, full 7-component blend):** 5 picks ≥1 hit 26.67% [24.8–28.7] vs 25.37% random; 2 picks 11.34% [10.0–12.8] vs 10.86%; AUC 0.502 (chance 0.5). The full blueprint engine — like every other strategy family — sits statistically at random chance.

## Upgraded chart strategy (`--mode charts2`, Jul 2026)

The legacy `charts` mode is kept untouched as the baseline; `lottery_core/chart_analysis.py` upgrades every methodological gap and measures what was previously assumed:

| Legacy 'charts' | Upgraded 'charts2' | What measurement showed |
|-----------------|--------------------|--------------------------|
| One pooled rate per chart (all ~90 entries identical) | Every entry's own record (~150 pooled trials each), EB-shrunk toward the chart mean (M=30 pseudo-trials) | Best entries (e.g. partner 47→44 at 19/140, p=0.0003) are **apophenia**: family-wise bootstrap P(noise beats it)=0.10 → per the decision rule, read entries as chart-average performers |
| Machine pointers at an **assumed 0.5×** | Machine-source rates **measured** per chart | 5.05%–6.04% — machine pointers transfer just like win pointers (~chance); the 0.5× assumption was **underweighting them by half** |
| Rates from one game's ~2,000 trials | Pooled across all 7 games: ~13,400 trials/chart | Chart rates 5.31%–5.88% vs 5.56% chance — precise, and precisely at chance |
| Lag-1 only | Transfer curve measured at lags 1–5 | Flat at ~5.3%–5.8% at every lag — no "comes within a few weeks" hump exists |
| Raw rates, no shrinkage | EB shrinkage by sample size, both raw and shrunk disclosed | — |

Scoring stays lag-1 and keeps the same 50/50 recency-baseline blend as legacy, so the walk-forward comparison isolates the chart-scoring upgrade.

**Head-to-head walk-forward backtest (2,002 test draws each):** legacy charts 26.67% / upgraded charts2 26.27% for ≥1 hit with 5 picks (random 25.37%); at 2 picks charts2 11.69% vs charts 10.74% (random 10.86%); AUC 0.503 both; Brier 0.1692 vs 0.1721. Every difference is inside the confidence intervals: **the upgrade is methodologically real (finer measurement, zero unmeasured assumptions) and predictively neutral — both sit at chance, as they must.** The measured lesson isn't a better hit rate; it's that the tradition's internal claims (special entries, machine discount, "within a few weeks") are now each individually tested and each individually at chance.

**Monte Carlo key validation, Monday Special (`predictor.py keys MS`)** — a textbook demonstration of why the bootstrap component exists: all 16 screened keys look individually impressive (rates ~2.7–3.0% vs 1.11% chance; per-key bootstrap p ≤ 0.012), yet the **family-wise max-statistic bootstrap gives P(noise produces a better best-key) = 0.88** — 88 of 100 structure-destroyed synthetic charts handed the search an even better "best key." Per the blueprint's own decision rule (p > 0.05 → discard): **every screened key is apophenia**, exactly the look-elsewhere effect the noise-control component was specified to catch. The keys remain visible in the UI as *candidates with their p-values*, never as findings.


## Data (expanded)
**`ghana_lotto_history.csv`** now holds **2,698 clean draws from 15 Dec 2017 to 13 Jul 2026** — the full archive available on GhanaYello, cross-verified against 590mobile's official API (7/7 recent draws matched). Per game: National Weekly 431, Fortune Thursday 423, Lucky Tuesday 419, MidWeek 415, Friday Bonanza 408, Monday Special 402, Sunday Aseda 200 (game launched Jul 2022). 20 duplicate/mislabeled rows were dropped by validation (5 unique numbers in 1–90, correct weekday per game, deduplication). Machine numbers are included from Aug 2018 onward.

## The system — `predictor.py` + `charts.py`
Six strategies, all selectable with `--mode`:

- **hot / recent / overdue / blend** — frequency, recency-weighted frequency, gap analysis, and a combination.
- **charts** — the traditional Ghana chart relationships from justlottoo.blogspot.com (Bonanza, Counterpart, Malta, String Key, Shadow, Partner, Equivalent, Code, Turning), encoded in `charts.py`. Each number from the last draw (winning + machine) "points" to its chart partners; pointers are weighted by each chart's measured historical transfer rate in that game.
- **ml** — a logistic-regression model (pure numpy, no extra installs) predicting each number's probability of appearing in the next draw from 10 features: all-time/30-draw/10-draw frequencies, recency-weighted frequency, gap ratio, co-occurrence with the last draw, chart pointers from winning and machine numbers, and draw-sum context. Trained walk-forward, refit every 100 draws.

### Commands
```
python3 predictor.py update                    # auto-update: pulls the latest months from ghanayello.com
python3 predictor.py predict                   # next draw of every game
python3 predictor.py predict MS                # one game (MS LT MW FT FB NW SA)
python3 predictor.py predict --date 2026-07-20 # the draw on a chosen date (uses only data before it)
python3 predictor.py predict --week 2026-07-20 # all 7 games of that week
python3 predictor.py predict MS --mode ml      # choose strategy: hot|recent|overdue|blend|charts|ml
python3 predictor.py backtest                  # re-run the honest evaluation
```
Historical dates work too — predictions always use only draws before the requested date, so you can check what the system would have said. The `update` command needs internet access on your machine; run it before predicting to stay current.

## Honest backtest — 1,998 test draws per strategy, walk-forward
| Strategy | 2 picks ≥1 hit | 5 picks ≥1 hit | 5 picks ≥2 hits |
|----------|---------------|----------------|-----------------|
| hot      | 11.4%         | 26.3%          | 2.5%            |
| recent   | 11.5%         | 25.4%          | 2.7%            |
| overdue  | 11.7%         | 25.1%          | 2.4%            |
| blend    | 10.5%         | 26.1%          | 2.3%            |
| **charts** | 10.5%       | 26.4%          | 2.1%            |
| **ml**   | 10.9%         | 24.8%          | 2.7%            |
| *pure chance* | *10.9%*  | *25.4%*        | *2.3%*          |

With ~2,000 tests the margin of error is roughly ±1.5 percentage points — **no strategy, including the ML model and the charts, separates from pure chance.**

### The charts, measured directly
Across ~13,400 number-transitions per chart (8.5 years of draws), the rate at which a chart partner of a drawn number appears in the next draw:

| Chart | Transfer rate | Chance |
|-------|--------------|--------|
| Bonanza 5.32% · Counterpart 5.50% · Malta 5.34% · String Key 5.65% · Shadow 5.66% · Partner 5.60% · Equivalent 5.62% · Code 5.35% · Turning 5.78% | | **5.56%** |

Every chart transfers at almost exactly the chance rate. The relationships are real as a numbering tradition, but 8.5 years of data shows they carry **no predictive signal** — and the ML model, given the chart pointers as features, learned to assign them near-zero weight.

## Bottom line
More data (8.5 years), a machine-learning model, and the traditional charts all confirm the same result: NLA draws behave as designed — randomly. Hitting 2+ of 5 numbers stays a ~1-in-40 event whatever the method. Use the system for informed, structured play; never stake money you can't afford to lose.

## Current picks (blend, as of 14 Jul 2026)
Run `python3 predictor.py predict` for live picks. Right now: Monday Special 87·1·74·6·55, and use `--mode charts` or `--mode ml` to compare strategies.
