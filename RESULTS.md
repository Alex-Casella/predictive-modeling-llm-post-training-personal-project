# Results

What was measured, what it showed, and what is still open. `README.md` is the
project history; this file is the findings log. Every number here is reproduced
by a script in this repo — the filename is given next to each result.

> Data source: [Pro-Football-Reference](https://www.pro-football-reference.com).
> See `ATTRIBUTION.md`.

---

## Headline

| | @25 | @50 | @100 | @150 | @200 | **@250** |
|---|---|---|---|---|---|---|
| Carry-forward baseline | 42.1 | 51.2 | 61.4 | 65.1 | 67.6 | **68.8** |
| Best reordering method | 43.5 | 53.5 | 62.0 | 65.5 | 67.9 | **68.8** |
| Best method overall (widened pool) | 42.1 | 51.2 | 61.4 | 65.2 | 67.8 | **69.1** |
| Target | | | | | | 75.0 |
| **Structural ceiling without rookie data** | | | | | | **76.8** |

Set overlap %, mean over 25 walk-forward season pairs. The baseline was
reproduced independently to 0.0 at all six tiers (`step4_race.py`).

**The project beats its baseline by 0.3 points and is 5.9 short of target.**
The reason is structural and is the main finding below.

---

## Findings

### 1. The headline metric cannot be moved by reordering

`step4_race.py`, proven not assumed.

Carry-forward, PPG, and PPG×games all score **exactly 68.8%** at tier 250. All
three rank the same 250 candidates — last season's top 250 — and the top 250 of
a 250-player pool is the whole pool. Set overlap ignores order.

```
pool = last season's top 250  ─┬─ sort by ppr  ─┐
                               ├─ sort by ppg  ─┼─ take top 25  → three different sets
                               └─ sort by p×g  ─┘  take top 250 → the SAME set, always
```

Consequence: no feature engineering, no model, and no amount of tuning can move
@250. Only changing the candidate pool can.

### 2. The ceiling is 76.8%, not 100%

`step3_yoy.py`. Each season's actual top 250 decomposes as:

| origin | players | share | reachable by reordering? |
|---|---|---|---|
| in last season's top 250 | 171.9 | 68.8% | yes — this is the baseline |
| had a prior season, not last | 20.2 | 8.1% | yes, with a widened pool |
| no prior season at all | 57.9 | 23.2% | **no** |

The baseline already captures 89.6% of what is reachable. The 75% target sits
1.8 points under the ceiling and asks for 78% of the remaining headroom.

### 3. Widening the pool works, and is worth 0.3 points

`step6_widen_pool.py`. Candidates become every player with any prior season,
scored by `ppr × d^(gap-1)`. `d=0` reduces *exactly* to carry-forward (asserted
at every season and tier), so the baseline is a special case.

Best is `d=0.3` at **69.1%**. The mechanism, measured:

```
per season:  4.8 returners added,   2.4 hit  → 49% precision
             4.8 carried evicted,   1.5 hit  → 31% precision
             net +0.9 players of 250          = +0.36 points
```

The edge is real (49% > 31%) but the volume is tiny: only elite returners clear
the 250th slot at `d=0.3`, and raising `d` collapses precision.

### 4. A gradient-boosted model buys about what one sorted column buys

`step8_model.py`. HistGradientBoosting over 45 prior-season / career / lag
features, walk-forward, refit every season, 2006–2025:

| | @25 | @50 | @100 | @150 | @200 | @250 |
|---|---|---|---|---|---|---|
| carry-forward | 41.2 | 50.3 | 61.1 | 64.6 | 67.2 | 68.6 |
| regression (censored=0) | 41.6 | 52.5 | 61.8 | 65.5 | 67.9 | 68.6 |
| classifier P(top 250) | 32.0 | 46.3 | 58.8 | 64.6 | 67.5 | 68.6 |

Positive at every movable tier, but wins tier 50 in only 11 of 20 seasons.

**The classifier is the instructive failure.** It is worst exactly where draft
picks are most expensive. `P(makes the top 250)` saturates: across 2024's top
25 its spread is **sd 0.027** against the regression's **sd 54.5** on the same
players. Ordering the elite tier by a score with no variance is ordering by
noise. Its 2026 board ranks Josh Allen ninth — while projecting him 375.9
points, the highest in the pool. It answers *who is safest*, not *who is best*.

`PROJECT_CONTEXT.md` §3d proposes this exact framing for boom/bust. The same
trap is waiting there.

### 5. Availability is very nearly unpredictable

`step3_yoy.py`, `step7_truncation.py`. Year-over-year, on 4,297 matched pairs:

| feature | r | r² |
|---|---|---|
| `ppg` | 0.675 | 0.455 |
| `ppr` (total) | 0.556 | 0.309 |
| `g` (games played) | **0.154** | **0.024** |

`g` carries 2.4% of its own variance forward and stays at 0.005–0.024 under
every truncation, so this is not a measurement artifact. `PROJECT_CONTEXT.md`
§8d guessed "availability — close to noise"; it is now measured.

Consequence: the `season_total = ppg × games` decomposition half-works. The
first factor is the most predictable thing in the file; the second is close to
a constant plus noise.

---

## Data findings

**`max(g)` is not the season length** (`step1_profile_g.py`). It is wrong in
2004, 2019 and 2025. A player traded mid-season has two teams with different
bye weeks and can dodge both, playing 17 games in a 16-game season — Jerry Rice
2004, Emmanuel Sanders 2019, Rashid Shaheed 2025 (18). All are `2TM` rows.
Deriving from single-team rows only is correct in 26/26 seasons.
`PROJECT_CONTEXT.md` §8a proposed `max(g)` as both the derivation *and* its own
integrity check; a value cannot validate itself.

**Season totals are regular-season only** (§8g, resolved). No single-team player
exceeds 17 games; a Super Bowl run would reach 20 or 21.

**31.2% of targets are right-censored.** A player absent from the next top 250
did not score zero — he scored below that season's cutoff (41.4–65.1 PPR) and
the file does not record what. Both imputations were run; imputing 0 beat
imputing the cutoff at every tier. Censored rows were never dropped (§7).

**The published R² of 0.59 is not comparable** (`step7_truncation.py`). This
file measures 0.455, but R² falls monotonically to 0.170 as the population is
narrowed from top 250 to top 25 with the relationship unchanged. This dataset is
itself truncated, so 0.455 is a floor. The untruncated value is not recoverable
here.

---

## What exists and runs

```
python3 step1_profile_g.py     data profiling and integrity checks
python3 step2_derive.py        → fantasy_top250_derived.csv
python3 step3_yoy.py           → yoy.csv
python3 step4_race.py          baseline reproduction + three sorted rankings
python3 step5_shrinkage.py     shrunk PPG, k tuned on 2001-2013
python3 step6_widen_pool.py    widened candidate pool
python3 step7_truncation.py    truncation sensitivity
python3 step8_model.py         the model, walk-forward
python3 board.py               → board_2026.csv
python3 draft.py               the draft CLI
```

Shared modules: `checks.py` (assert harness), `common.py`, `evaluate.py`
(set-overlap harness), `features.py`, `vbd.py`.

---

---

## The agent (stages 3–5)

Draft decisions were reconstructed from 19 replayed seasons: a board built only
from seasons <= n-1, a simulated 10-team snake draft, and every pick labelled
with which of its 12 candidates turned out best. 2,850 examples, split by
season (train 2007–2019, val 2020–2022, test 2023–2025) because consecutive
picks share most of their candidate list and a random split would put
near-duplicates on both sides.

**The bar, fixed before any model was trained** (`eval_agent.py`, 450 held-out
picks). Each pick offers 12 candidates; score is where the chosen player lands
once the season is played out, lower is better:

| | validity | mean rank (of 12) | best of 12 |
|---|---|---|---|
| random | 100% | 6.36 | 8.4% |
| deterministic board (`draft.py`) | 100% | **6.00** | 9.6% |
| `llama3.1:8b` un-tuned base | — | *pending* | |
| fine-tuned adapter | — | *pending* | |

Random landing at 6.36 / 8.4% against a theoretical 6.50 / 8.3% is the check
that the metric is wired up correctly.

**The board beats random by 0.52 ranks out of 12.** Once VBD has sorted twelve
players into a narrow band, choosing between them is close to a coin flip. That
is both the opportunity for the agent and the reason to expect the fine-tune
may not beat 6.00. §11g: *"If the agent doesn't beat the number it was handed,
the language layer is decoration."*

A labelling error was caught and is documented in `step9_draft_examples.py`:
the first version labelled each pick with the best actual outcome over all ~200
available players, which returns whoever won the season rather than the right
pick — in 2021 the label was Cooper Kupp for ten consecutive picks, and a
150-pick draft carried only 4–11 distinct labels. Restricting the label to the
12 candidates actually shown fixed it (21–37 distinct labels), and an assert
guards the regression.

See `SERVING.md` for the conversion and serving runbook.

## What is NOT done

- **Rookies.** 23.2% of a real top 250 cannot appear on the board. Needs PFR
  draft results, 26 seasons, joined on `pfr_id`.
- **The LLM agent.** `PROJECT_CONTEXT.md` §11e stages 3–5. Stages 1 and 2 are
  complete, so this is unblocked.
- **Sit/start.** Blocked on weekly data. `fantasy_top250.csv` has no week
  column. `draft.py lineup` gives the static ordering, which never changes week
  to week.
- **The spec conflict.** `CLAUDE.md` and `README.md` scope v1 to the draft
  board; `PROJECT_CONTEXT.md` §1 adds a weekly model and sit/start. Unresolved.
