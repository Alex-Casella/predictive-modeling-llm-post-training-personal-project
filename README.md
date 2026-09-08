# Fantasy football projection + draft board + start/sit

Predict the top 250 PPR fantasy football players for an upcoming season, turn
that into a draft board, build a CLI assistant that recommends picks from the
board plus current roster — and then a second, separate assistant that answers
the in-season question: of the flex-eligible players on my roster this week,
which do I start?

The two assistants are two fine-tuned adapters over the same base model,
trained by one script and scored by one harness, which is what makes them
comparable.

**Status**

- **Draft agent beats the sort it was handed, and it replicates.** 5.41 mean
  rank of 12 against a 6.00 board — p = 0.0041 / 0.0040 / 0.0068 across three
  paired tests, and −0.587 with p = 0.0066 / 0.0070 / 0.0112 on a second
  independently trained adapter. The only adequately powered comparison here
  and the only one that replicates. Against the *un-tuned* base (5.80) the same
  model is suggestive but underpowered, p = 0.047 / 0.054 / 0.090.
- **Reinforcement learning reaches the same place from the other direction.**
  GRPO on 600 *unlabelled* prompts, trained from the base model with no right
  answer ever shown, scores **5.351 — −0.653 vs the board**, p = 0.0015 /
  0.0027 / 0.0152, adequately powered. Against the supervised adapter it is
  −0.067 (p = 0.74, n=33,212 needed): **it matches SFT, it does not beat it.**
  Four adapters, two training methods, every confidence interval below zero.
  Convergence from opposite directions is the result — the ceiling is upstream
  of the optimiser, in what the prompt can see.
- **Start/sit is a clean set of nulls.** 2.99 of 6 against a 2.92 rule and a
  3.02 un-tuned base. Five paired comparisons, five confidence intervals
  containing zero — and the board's own score swings 0.43 across the test
  seasons, so the bar is noisier than the effects being measured against it.
- **Projection: 72.2% set overlap @250 on held-out 2016–2025**, against a 68.6%
  carry-forward baseline on the same seasons and a 75.0% target. Reached by
  widening the candidate pool with NFL draft position (rookies) and stale
  returners. `@25` and `@50` have not moved on any experiment.
- **The stop-token defect is fixed.** The start/sit adapter used to run to the
  token cap on 96% of answers; `--pad-token auto` takes that to 0% and changes
  decisions by 0.014.

See `RESULTS.md` for every number and the script that produces it, including
the retractions.

`CLAUDE.md` holds the working rules and constraints. This file is the history —
what was done, what was found, and what is still open.

> **Data sources:** Season statistics come from
> [Pro-Football-Reference](https://www.pro-football-reference.com), a Sports
> Reference LLC site. Weekly statistics come from
> [nflverse](https://github.com/nflverse) via `nfl_data_py`. Please credit both
> in any output, chart, or write-up derived from this data. See
> `ATTRIBUTION.md`.

> **Copyright © 2026 Alex Casella. All rights reserved.** The code is published
> to be read, not reused; no license is granted. The data and the base model
> weights were never mine to license and carry their own terms. See
> `COPYRIGHT.txt` and `ATTRIBUTION.md`.

---

## What this is really about

The fantasy football is the vehicle. The subject is **post-training**: taking a
general model, making it better at one specific decision, and — the hard part —
knowing whether it actually worked.

Fluent output is free. An LLM will produce confident draft advice whether or not
the advice is any good, so the entire design exists to make "did it work?"
answerable:

- the model picks **one** of K named candidates, so the answer is checkable
  rather than judged
- each decision is labelled by hindsight **from the K shown**, so a right answer
  exists — labelling from everyone available returns "who won the season"
- the score is where the pick finished, with chance at (K+1)/2, so the scale has
  a known floor
- splits are by **season**, never random, so adjacent near-identical decisions
  cannot straddle them
- the **un-tuned base is scored first**, which separates "fine-tuning helped"
  from "the model could already do this"
- a **deterministic layer** is scored on the same items, which separates the
  language layer from the sort underneath it

Two tasks exist so the conclusion is not one anecdote. They already disagree,
which is the most useful thing either produced:

| | tabular layer | un-tuned Llama | SFT | GRPO | vs the sort |
|---|---|---|---|---|---|
| draft | 6.00 of 12 | 5.80 | **5.41** | **5.35** | **beats it, p ≤ 0.004, replicated on 4 adapters across 2 methods** |
| start/sit | **2.92** of 6 | 3.02 | 2.99 | — | no measurable difference |

"Can a language model beat the spreadsheet?" turns out to be task-dependent —
measured on the same base model through the same harness, rather than assumed.

One caveat that matters more than it looks. The board's own score swings 0.11
between the draft test seasons and **0.43** between the start/sit ones, while
the sit effects being chased were 0.014–0.126. On draft the effect is five
times its season noise; on sit it is a fraction of it. "Significant on one task,
five nulls on the other" is at least as much a fact about the two test sets as
about language models, and saying otherwise would be overclaiming.

---

## Setup

Everything runs locally except training, which rents a GPU for under an hour.

**1. Python.** Python 3.11+ and the project dependencies:

```bash
pip install -r requirements.txt
```

> pandas, scikit-learn, scipy, and `nfl_data_py` (the weekly data source).
> If `scipy` fails to import complaining about NumPy, pin it:
> `pip install "numpy>=2.0,<2.8"`. Installing llama.cpp's requirements into
> this environment will downgrade NumPy and break it — use a separate venv for
> that step, which `finish_adapter.sh` does.

**2. Ollama**, to serve the models locally. No API, no per-request cost:

```bash
brew install ollama          # macOS
ollama pull llama3.1:8b      # 4.9 GB, the base model
```

**3. Hugging Face**, only needed for training and GGUF conversion:

- accept the Llama 3.1 licence at
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> (the weights are
  gated)
- create a **read** token at <https://huggingface.co/settings/tokens>
- `export HF_TOKEN=hf_...` — never commit it, never paste it into a chat

**4. Modal**, for the GPU:

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=hf_...
```

**5. llama.cpp**, for the PEFT → GGUF conversion, cloned NEXT TO this repo:

```bash
git clone https://github.com/ggerganov/llama.cpp ../llama.cpp
```

> `finish_adapter.sh` looks for it at `../llama.cpp` and builds its own
> `gguf-convert-env` inside it, so llama.cpp's pinned NumPy never touches the
> project environment.

Only steps 1 and 2 are needed to run the analysis and talk to an existing
model. Steps 3–5 are needed only to train a new adapter.

### The loop, end to end

```bash
git pull                                   # always start here

python3 step11_weekly_data.py              # data, if rebuilding
python3 step12_start_sit_examples.py
python3 step13_render_startsit.py

# score the UN-TUNED base first -- this is the step that fixes the bar
python3 eval_agent.py --model llama3.1:8b \
    --test sit_test.jsonl --key start_sit_examples.jsonl

modal run train_adapter.py --dataset sit   # ~20 min on an A10G
export HF_TOKEN=hf_...
EXPECT_EPOCHS=2.0 ./finish_adapter.sh fantasy-sit sit   # download, convert,
                                                        # serve, score

# re-running the SAME dataset with a different hyperparameter needs a --tag,
# or both runs write to /adapter/sit and the second one silently scores the
# first. run_ablation.sh chains train -> serve -> score -> paired test:
./run_ablation.sh sit 1 e1

git add <the files you changed> <the eval csv>
git commit -m "sit/start: ... mean rank X.XX of 6 (rule 2.92, base 3.02)"
git push -u origin main
```

Never `git add .` — stage files explicitly, and keep the score in the commit
message so `git log --oneline` reads as a history of what worked.

---

## How this was built

Most of the code here was written by **Claude Opus 5 via Claude Code**, working
from my direction. 84 of the 90 commits carry a `Co-Authored-By` trailer, so
`git log` is the authoritative record rather than this paragraph.

What that division actually looked like, since "AI-assisted" covers a wide range:

| mine | the assistant's |
|---|---|
| what to measure, and against what bar | the implementation |
| which experiment to run next, and in what order | the statistics and plotting code |
| running every training job and eval | the data-pipeline code |
| reading results and deciding what they meant | the checks and guards |
| catching numbers that looked wrong | drafting the documentation |

Some of the decisions that shaped the results were mine and are worth naming,
because they are the reason the numbers mean anything: scoring the un-tuned
base model *before* training rather than after; running a seed replicate to
establish a noise floor before trusting a 0.126 gap; and checking whether the
matchup feature carried any signal at all before asking a model to use it. Each
of those changed a conclusion.

The assistant also produced several confidently wrong numbers during this work
— a stale adapter scored as a new one, a hyperparameter tuned on the test set,
a runaway rate misread from a scroll and published before it was checked. All
are recorded in `RESULTS.md` with their retractions. That is the honest picture
of building this way: fast, and requiring exactly the kind of verification this
project was already built to do.

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

**Data**

| File | What it is |
|---|---|
| `fantasy_top250.csv` | 6,500 rows. Top 250 players by PPR per season, 2000-2025, no gaps. Never modified. |
| `fantasy_top250_derived.csv` | The above plus derived columns (`step2_derive.py`). |
| `weekly_ppr.csv` | 84,909 player-weeks, nflverse, 2000-2024 (`step11_weekly_data.py`). |
| `board_2026.csv` | The projected draft board (`board.py`). |
| `draft_2026_offense.csv` | 81 skill-position players from the 2026 NFL draft. |
| `yoy.csv` | Year-over-year matched pairs (`step3_yoy.py`). |

**System 1 — the projection model**

| File | What it is |
|---|---|
| `step1_profile_g.py` … `step8_model.py` | Profiling, baseline, feature work, the model. |
| `features.py` | Feature table, no look-ahead by construction. |
| `evaluate.py` | Set-overlap harness with a deterministic tiebreak. |
| `vbd.py` | League-agnostic VBD. Takes a LeagueConfig; never hardcode settings elsewhere. |
| `board.py`, `draft.py` | The board, and the deterministic draft CLI (no LLM). |

**Systems 2 and 3 — the two agents**

| File | What it is |
|---|---|
| `step9` / `step10` | Draft decisions → SFT examples (2,850). |
| `step12` / `step13` | Start/sit decisions → SFT examples (1,774). |
| `train_adapter.py` | QLoRA on Modal. `--dataset draft\|sit` — one script, two adapters. |
| `eval_agent.py` | Scoring. `--test/--key` point it at either task. |
| `assistant.py` | The draft CLI backed by the fine-tuned model. |
| `paired_test.py` | Is the gap between two eval runs real? |
| `finish_adapter.sh` | Modal volume → GGUF → Ollama → scored, in one command. |
| `run_ablation.sh` | One hyperparameter varied, end to end, with the paired test. |

**Docs**

| File | What it is |
|---|---|
| `CLAUDE.md` | Working rules, loaded automatically each session. |
| `RESULTS.md` | The findings log — every number and the script that produces it. |
| `SERVING.md` | Adapter → GGUF → Ollama runbook. |
| `ATTRIBUTION.md` | Data source credit and identifier documentation. |
| `checks.py` | Assert harness: collect every failure, print the wall, then raise. |

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

**No weekly resolution — resolved for the second system.** `fantasy_top250.csv`
is season totals only, which blocked start/sit. `step11_weekly_data.py` now adds
`weekly_ppr.csv`: 84,909 player-weeks from nflverse, joined on identifiers and
validated by reconstruction against the season totals (median difference +0.00).
The season file is unchanged and still has no week column; the two files sit
side by side.

Two limits came with it: coverage collapses before 2010 (12% in 2000 against
99% from 2010, a `pfr_id` gap upstream) and 2025 is not published by
`nfl_data_py` 0.3.3. Weekly work runs 2010–2024.

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

**In-season updating.** Start/sit is built, but it replays completed seasons.
Predicting a week of a season currently in progress is a different system, and
it cannot be validated until that season finishes.

**Full-lineup optimisation.** Start/sit answers the flex slot only, because a
QB slot with one QB on the roster is not a decision. Naming all nine starters
is a different decision shape and none of the current scoring applies to it.

## Out of scope for v1

- ESPN integration. ESPN's fantasy API is undocumented and unofficial;
  private-league access requires `swid` and `espn_s2` cookies pulled from a
  browser session. A community Python wrapper exists (`cwendt94/espn-api`), but
  it is unofficial with no stability guarantee, and ESPN platform changes have
  broken community tooling before. Manual board entry first.
- Live draft tracking.
- Web app.
- Predicting a season already in progress (see Deferred).
