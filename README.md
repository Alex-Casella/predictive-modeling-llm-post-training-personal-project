# Fantasy football projection + draft board

Predict the top 250 PPR fantasy football players for an upcoming season, turn
that into a draft board, and build a CLI assistant that recommends picks based
on the board plus current roster composition.

**Status:** model built and backtested; draft board and CLI running; draft agent
fine-tuned and scored (5.41 mean rank of 12 vs 5.80 un-tuned, 6.00 board).
Best result **69.1%** set overlap @250 against a **68.8%** baseline and a
**76.8%** structural ceiling. See `RESULTS.md` for every number and the
scripts that produce them.

`CLAUDE.md` holds the working rules and constraints. This file is the history —
what was done, what was found, and what is still open.

> **Data source:** All NFL statistics in this project come from
> [Pro-Football-Reference](https://www.pro-football-reference.com), a Sports
> Reference LLC site. Please credit them in any output, chart, or write-up
> derived from this data. See `ATTRIBUTION.md`.

---

## Getting started

Put all files in one folder, then:

```
cd path/to/fantasy-project
git init
claude
```

`CLAUDE.md` is read automatically at the start of every session. Paste this as
the first message:

```
Read CLAUDE.md first.

Session 1: establish the baseline and understand where the difficulty is.
No modeling yet.

Work through these in order, stopping after each so I can inspect the output:

1. Load fantasy_top250.csv. Print the season list, rows per season, and a count
   of blanks per column. Tell me anything that surprises you.

2. Implement the carry-forward baseline from scratch: treat last season's top N
   as the prediction for this season, measure set overlap, average across all 25
   consecutive year-pairs. Report it at N = 25, 50, 100, 150, 200, 250.
   My numbers are in CLAUDE.md. If yours disagree at any tier, stop and work out
   which of us is wrong before continuing.

3. The top-25 baseline is only 42.1%, meaning ~15 of each season's top 25 were
   not there the year before. Identify those players. For every season, list who
   entered the top 25 and split them into:
     (a) true rookies — no prior season anywhere in the data
     (b) returning players who were outside the top 25 the previous year
   For group (b), separate those who played fewer than 10 games the prior season
   from those who played a full season but scored poorly.
   Show the breakdown by position and by year.

4. Stop and tell me what that implies about which of the three groups is
   realistically predictable, and which features might reach them.

Ask me before making any judgment call about blanks, filtering, or scoring.
```

Step 3 is the point of the session. Whatever it shows about who those players
are should drive every modeling decision after it.

---

## Current contents

| File | What it is |
|---|---|
| `fantasy_top250.csv` | 6,500 rows. Top 250 players by PPR per season, 2000-2025, no gaps. |
| `draft_2026_offense.csv` | 81 skill-position players from the 2026 NFL draft. |
| `CLAUDE.md` | Project brief loaded automatically by Claude Code each session. |
| `vbd.py` | League-agnostic VBD module. Takes a LeagueConfig, works on actuals or predictions. |
| `ATTRIBUTION.md` | Data source credit and identifier documentation. |
| `README.md` | This file. |

---

## How the dataset was built

**Source:** 26 season-level fantasy ranking exports, one CSV per season, in
Pro-Football-Reference's fantasy table format (34 columns).

**Steps taken:**

1. Verified all 26 files share an identical 34-column schema.
2. Confirmed there are no repeated header rows mid-file (a common artifact in
   PFR exports — checked and absent here).
3. Deduplicated traded players, then ranked every player by PPR points and kept
   the top 250. Every season verified to yield exactly 250 rows.
4. Renamed the duplicated column names by stat group, added a `season` column,
   sorted by season then rank.

### Problems found and fixed

**Four files were byte-identical duplicates.** md5 comparison showed four pairs
of identical files. In each pair one label was wrong. Identified the true season
from the contents:

| Duplicate pair | Actual season | Identified by |
|---|---|---|
| 2002 / 2003 | 2003 | Priest Holmes age 30, LaDainian Tomlinson age 24, Ahman Green's 1,883-yard season |
| 2004 / 2005 | 2005 | Shaun Alexander's MVP / record-touchdown year |
| 2006 / 2007 | 2007 | Tom Brady's 50-TD season, Randy Moss's 23 receiving TDs |
| 2017 / 2018 | 2018 | Saquon Barkley as a rookie |

The four genuinely missing seasons (2002, 2004, 2006, 2017) were later
re-sourced and verified as distinct and correctly labeled. The dataset is now
complete.

**The 2019 file has no header row.** Every other file has one. Loading these in
a loop with `header=0` silently drops Christian McCaffrey's 2019 season and
shifts that file's column names by one row.

**Two column names were corrupted in the source.**

| Raw name | Actual meaning | Renamed to |
|---|---|---|
| `2:00 PM` | Two-point conversions made (`2PM`, mangled by a spreadsheet into a timestamp) | `two_pt_made` |
| `-9999` | Pro-Football-Reference player IDs (`FaulMa00`, `TaylJo02`) | `pfr_id` |

**Duplicate column names.** `Att`, `Yds`, and `TD` each appeared multiple times
because passing / rushing / receiving stats sat under merged group headers that
got flattened on export. Split by column position: 8-12 passing, 13-16 rushing,
17-21 receiving. Renamed with prefixes (`pass_`, `rush_`, `rec_`).

**Identifiers.** The source `player_id` column was renamed `pfr_id` and a
project-owned surrogate key `pid` (format `P0001`) was added as the primary key.
`pfr_id` is retained as a foreign key because it is the only thing separating
players who share a name — the dataset has 1,692 distinct players but 1,681
distinct names, and five of those collisions share a position. See
`ATTRIBUTION.md` for how `pid` is assigned.

**The source `Rk` column ranks by VBD, not PPR.** Sorting each season by every
candidate column showed `Rk` reproduces a VBD sort exactly (250/250 matches),
and VBD is computed by Pro-Football-Reference from standard scoring. Cutting the
top 250 on `Rk` excluded roughly 10 genuine PPR top-250 players per season —
pass-catching backs and slot receivers that standard scoring undervalues. The
dataset was re-cut and re-ranked on `ppr`. `rk` is now the PPR rank and `pos_rk`
the PPR rank within position.

**Standard-scoring columns dropped.** `FantPt`, `DKPt`, `FDPt`, `VBD`,
`PosRank` and `OvRank` were removed. `ppr` is the only scoring column.

**Traded players appeared multiple times.** A player traded mid-season gets an
aggregate row (team `2TM` / `3TM`) carrying his real stats, plus stub rows with
0 games and blank stats for teams he never played for. 12 such stubs existed
across the dataset. Rule applied: keep the row with the most games played per
player-season.

**Source data anomalies.** 7 rows out of 15,500 have a `FantPt` value that
disagrees with `PPR` minus receptions by more than a point — e.g. Ben Tate's
2014 row lists 67 standard points against 0 PPR. These appear to be source
errors. `ppr` was used directly rather than derived, so they do not propagate.

**Player name markers.** Names carried trailing `*` and `+`. Stripped into two
boolean columns, `pro_bowl` and `all_pro`. These are believed to mean Pro Bowl
and First-Team All-Pro respectively per PFR convention — worth confirming
against their glossary before using as features.

### Decisions left open on purpose

- **Blanks were not imputed.** Empty cells remain in `fl`, `two_pt_made`,
  `rush_ypc`, and `vbd`. They do not all mean the same thing — some are true
  zeros the source omits, and a blank `vbd` means something different from a
  blank `two_pt_made`. Decide per column.
- **Only `ppr` kept.** All other scoring columns were dropped, since the
  project is full-PPR only.
- **VBD is league-specific.** `vbd_10` covers a 10-team league on ESPN's
  standard lineup (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX). The baseline is the last
  player at each position to claim a starting slot, with flex slots going to
  the best remaining RB/WR/TE. Any other league size or lineup invalidates the
  column — recompute rather than reuse.

---

## Success metric and baseline

**League scope (MVP):** one 10-team ESPN standard PPR league — the actual
league this is being drafted for. Lineup is 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX,
1 D/ST, 1 K, 7 bench. D/ST and K are not modelled.

**Long-term goal: usable with anyone's league settings.** The MVP targets one
league, but league configuration is treated as an input throughout rather than
a constant, so generalising is a config change rather than a rewrite. See
Architecture.

**Scoring format: full PPR only.** One point per reception. Standard and
half-PPR are out of scope.

**Metric: set overlap at 250.** Of the 250 players predicted to finish in next
season's PPR top 250, how many actually do? All 250 ranks count equally — the
goal is a globally accurate board, not one tuned to a single draft position.

This was chosen over rank correlation or point error deliberately. Season-long
point projection is extremely hard — an analysis of eleven projection sources
across 2014-2025 found professional seasonal projections explain only 20-28% of
variance for RBs, 14-19% for WRs, 7-15% for QBs, and 16-26% for TEs, and are
systematically optimistic by an average of +21.6 points. Set membership is a
tractable target; exact ordering is not.

**Baseline: 68.8%.** Carrying last season's PPR top 250 forward unchanged,
measured across all 25 consecutive year-pairs. Range 63.6% to 73.6%.

**Target: 75%** — 6.2 points above the naive baseline.

### The baseline is not flat across ranks

Carry-forward overlap by rank tier, measured across all 25 consecutive
year-pairs:

| Tier | Mean | Worst year | Best year |
|---|---|---|---|
| Top 25 | 42.1% | 28.0% | 56.0% |
| Top 50 | 51.2% | 42.0% | 62.0% |
| Top 100 | 61.4% | 54.0% | 68.0% |
| Top 150 | 65.1% | 60.0% | 75.3% |
| Top 200 | 67.6% | 63.5% | 73.0% |
| Top 250 | 68.8% | 63.6% | 73.6% |

The headline 68.8% is carried by the stable tail. More than half the elite tier
turns over every year, and in the worst season 72% of the top 25 did. That is
where a model can actually add value, and it is also where draft picks are
most expensive — so global accuracy and draft usefulness point at the same
work rather than competing.

**Report all six tiers on every experiment.** Optimise the top-250 number, but
a model that improves the headline purely by getting better at ranks 200-250
has not done anything useful. Watch the top 25 and top 50 columns.

Per-pair carryover at 250:

```
00->01 66.0   01->02 66.8   02->03 73.2   03->04 70.4   04->05 70.0
05->06 68.0   06->07 68.0   07->08 67.6   08->09 70.4   09->10 73.6
10->11 72.4   11->12 66.4   12->13 70.4   13->14 63.6   14->15 66.4
15->16 66.4   16->17 66.8   17->18 65.2   18->19 68.0   19->20 70.8
20->21 68.0   21->22 69.2   22->23 68.8   23->24 71.2   24->25 71.2
```

The implication: roughly 68% of each season's top 250 is predictable by doing
nothing at all. All the difficulty lives in the ~31% annual turnover — rookies,
breakouts, injuries, and decline. That is where the 6.4 points must come from.

For reference, one published analysis found the year-over-year R² of fantasy
points per game is 0.59 — last season's number alone is a strong predictor.

---

## Architecture

Three layers, deliberately separated:

1. **Projection model** — tabular regression predicting next-season PPR points.
   scikit-learn / XGBoost / LightGBM. **Not an LLM.** LLMs are poor at numeric
   prediction over tables and cannot be calibrated or backtested the way this
   needs to be.
2. **Board builder** — VBD, positional tiers. Deterministic arithmetic, no
   model. A PPR VBD implementation already exists and is applied to historical
   actuals in `vbd_10`; the board applies the same baseline logic to
   *predicted* points.
3. **Draft assistant** — LLM-backed CLI. Reads the finished board plus current
   roster, recommends a pick, explains the reasoning. This is the only layer
   where an LLM belongs.

Build in that order. Layers 2 and 3 are worthless if layer 1 doesn't beat 68.8%.

### What varies by league and what doesn't

Player production does not depend on league settings — a receiver catches what
he catches. Everything downstream of that does. Keeping the boundary clean is
what makes the tool generalisable:

| Step | Depends on league? |
|---|---|
| Predict production | No |
| Apply scoring rules (PPR / half / standard) | Yes — scoring format |
| Compute VBD baselines | Yes — team count and lineup slots |
| Recommend a pick | Yes — plus current roster |

`vbd.py` implements the valuation step with `LeagueConfig` as an argument, so
it already works for any team count. Two consequences worth understanding:

**A model predicting PPR points can only serve PPR leagues.** Receptions cannot
be recovered from a PPR total, so standard and half-PPR are unreachable from
it. Full generality eventually means predicting component stats — receptions,
yards, touchdowns — and applying scoring afterward. The MVP does not need this,
but prediction, scoring, and valuation should stay separate steps so it isn't a
rewrite later.

**The top-250 cut is itself PPR-specific.** Each scoring format produces a
different top 250. Measured across the dataset, the union of all three formats'
top 250 is only about 260 players per season — roughly 4% larger. Supporting
other formats would mean re-cutting on that union.

---

## Known gaps

**Rookies.** This dataset contains nothing about a player before his first NFL
snap. Rookies are structurally unpredictable from it and require external data.

**No preseason-knowable inputs.** Every row is an end-of-season result. Real
preseason prediction needs depth charts, team changes, and injury history that
this file lacks.

**No weekly resolution.** Season totals only. Any in-season or week-to-week
modeling requires a different dataset.

### Where to fill them: Pro-Football-Reference

Use PFR, the same source the existing dataset came from. Decision made
deliberately over the alternative (nflverse / `nflreadpy`) for two reasons:

1. **IDs join natively.** `fantasy_top250.csv` keys on PFR player IDs
   (kept as `pfr_id`). PFR pages carry the same IDs, so joins are direct.
   nflverse uses GSIS IDs and would require a bridge table that loses rows
   wherever the mapping fails.
2. **Same workflow already proven.** The existing 26 files were exported using
   PFR's "Share & Export → Get table as CSV" button. No new tooling, no package
   install, nothing in R.

Relevant PFR pages:

| Page | What it provides | Fills which gap |
|---|---|---|
| `/years/<YEAR>/draft.htm` | Draft round and pick for that class | Rookie draft capital |
| `/teams/<TM>/<YEAR>_roster.htm` | Team roster for a season | Team context, roster churn |
| `/years/<YEAR>/fantasy.htm` | The fantasy table the existing data came from | — |

Combine data is deliberately excluded from this project. Do not add height,
weight, or other physical measurements as features.

Export a single year first and check the join rate against `pfr_id` before
committing to bulk exports. If the join is lossy, everything downstream
inherits that.

---

## Next steps

1. Reproduce the carry-forward baseline independently at all six tiers. If any
   number disagrees with the table above, resolve that before anything else.
2. Decompose the misses, weighting attention on the top 25 and top 50 where the
   baseline is weakest. Split them into true rookies (no prior season anywhere
   in the data) versus returning players who fell out of the tier. Break down by
   position and year. For the returning group, separate injured (low `g`) from
   healthy but unproductive — those are different prediction problems and likely
   need different features.
3. Only then decide what to model, based on what step 2 shows is reachable.

---

## Deferred decisions

Things consciously postponed rather than rejected. Written down so the reasoning
isn't lost.

**Other league sizes.** `vbd.py` handles any team count already; only the
cached `vbd_10` column is league-specific. A `vbd_12` column set was built,
verified, and then removed to keep the MVP pointed at one real league. Worth
knowing what the comparison showed before it was dropped: the two boards were nearly identical at
the top — in 2025 only Jaxon Smith-Njigba and Jahmyr Gibbs swapped in the top
ten — and diverged by at most about a dozen ranks in the tail. The mechanism is
that flex slots go overwhelmingly to receivers, so a 12-team league counts 35
startable WRs against a 10-team league's 27, while RBs move only from 23 to 25.
Deeper leagues stretch the receiver pool much further than the back pool. Restoring this is a config change, not new logic.

**Half-PPR and standard scoring.** Deferred. Blocked on two things: the model
would need to predict component stats rather than PPR points, and the top 250
would need re-cutting on the union of formats (about 260 players per season).

**User-supplied league settings.** The end goal — anyone enters their own
league and gets a board. Everything above is a step toward it.

**Weekly / in-season modelling.** Season totals only in the current data.

## Out of scope for v1

- ESPN integration. ESPN's fantasy API is undocumented and unofficial;
  private-league access requires `swid` and `espn_s2` cookies pulled from a
  browser session. A community Python wrapper exists (`cwendt94/espn-api`), but
  it is unofficial with no stability guarantee, and ESPN platform changes have
  broken community tooling before. Manual board entry first.
- Live draft tracking.
- Web app.
- Weekly / in-season projections.
