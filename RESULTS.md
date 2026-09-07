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
python3 draft.py               the draft CLI, deterministic (no LLM)
python3 assistant.py           the draft CLI, backed by the fine-tuned model
python3 paired_test.py         is the gap between two eval runs real?

python3 step11_weekly_data.py  → weekly_ppr.csv  (nflverse, joined on ids)
python3 step12_start_sit_examples.py  → start_sit_examples.jsonl + the bars
python3 step13_render_startsit.py     → sit_train/val/test.jsonl
```

The eval harness serves both tasks; the defaults are the draft files:

```
python3 eval_agent.py --model board
python3 eval_agent.py --model fantasy-sit \
    --test sit_test.jsonl --key start_sit_examples.jsonl
```

`train_adapter.py --dataset draft|sit` picks the data and the volume
subdirectory from one script, so a difference between two results cannot be a
drifted hyperparameter.

`assistant.py` imports its prompt from `step10_render_sft.render_user` and its
answer parsing from `eval_agent.extract_pick` rather than reimplementing either.
A LoRA adapter learns the shape of its training prompt, so a second "equivalent"
renderer would move the served input off the trained distribution and the damage
would read as a bad fine-tune rather than a formatting bug.

It shows the model's pick beside the board's and says when they disagree. It
does **not** score them: `board_2026.csv` projects a season whose outcome is
unknown. Scoring lives in `eval_agent.py`, on 2023-2025, where an answer key
exists.

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

All 450 held-out picks, seasons 2023-2025:

| | validity | mean rank (of 12) | best of 12 |
|---|---|---|---|
| random | 100% | 6.36 | 8.4% |
| deterministic board (`draft.py`) | 100% | 6.00 | 9.6% |
| `llama3.1:8b` **un-tuned** | 100% | **5.80** | 11.1% |
| `fantasy-draft` **fine-tuned** | 100% | **5.41** | 15.1% |

Random landing at 6.36 / 8.4% against a theoretical 6.50 / 8.3% is the check
that the metric is wired up correctly.

**The un-tuned base model beats the board by 0.20 ranks, with no training at
all.** Validity 100% matters as much as the rank: it named a player from the
shortlist on all 450 prompts, and `eval_agent.py` refuses to map near-misses
onto real players, so that is not charity.

It is below 6.00 in every held-out season, which makes it an effect rather than
one season's luck:

| season | n | mean rank | best of 12 |
|---|---|---|---|
| 2023 | 150 | 5.72 | 10.7% |
| 2024 | 150 | 5.97 | 9.3% |
| 2025 | 150 | 5.73 | 13.3% |

It is also strongest exactly where the decision matters most — on picks where
the gap between the best and second-best option was decisive (20+ VBD) it
scores 5.52, against 6.66 on marginal 1-5 point gaps.

**Why the first measurement was wrong.** A `--limit 60` run scored 5.67, but
`--limit` takes the FIRST 60 examples, which are all season 2023 rounds 1-6.
The full set gives 5.80. The subsample flattered it by 0.13, which is most of
the effect size — a reminder that `--limit` is for smoke-testing the plumbing,
never for producing a number.

### The bar this raised, and the fine-tune clearing it

§11g asks two questions. The un-tuned run answered the first — *can a language
model beat the tabular layer?* Yes, untrained — and in doing so moved the bar
for stage 4 from 6.00 to **5.80**. Landing between them would have meant
post-training made the model worse than leaving it alone.

**The fine-tuned adapter scores 5.41** (QLoRA rank 16, 1 epoch over 1,950
examples, `PROVENANCE.txt` in the adapter). That is 0.39 below the un-tuned
base and 0.59 below the board, at 100% validity, and it is below the base in
every held-out season:

| season | untuned | fine-tuned | Δ |
|---|---|---|---|
| 2023 | 5.72 | **5.31** | −0.41 |
| 2024 | 5.97 | **5.53** | −0.44 |
| 2025 | 5.73 | **5.40** | −0.33 |

Best-of-12 rose 11.1% → 15.1% against 8.3% chance, so the gain is finding the
right player rather than only avoiding the worst one.

**The gain is concentrated where the sort was useless, not where it was
already working.** By decision difficulty:

| margin (actual-VBD gap) | n | untuned | fine-tuned | Δ |
|---|---|---|---|---|
| marginal 1–5 | 50 | 6.66 | **5.38** | −1.28 |
| decisive 20+ | 269 | 5.52 | 5.58 | +0.06 |

The base model was worst on near-ties and best on obvious calls; fine-tuning
inverted that. Decisive picks were already solvable by sorting — that is what
VBD is — so a language layer had nothing to add there and, measurably, added
nothing. The coin flips are where it earned its keep.

**Significance: measured, and the answer is "suggestive, underpowered".**
`paired_test.py`, on the same 450 prompts answered by both models:

| | | |
|---|---|---|
| mean difference | **−0.393** | ranks of 12 |
| sd of one difference | 4.197 | |
| se of the mean | 0.198 | one "noise width" |
| 95% CI | **[−0.781, −0.005]** | excludes 0, barely |

| test | what it keeps | p |
|---|---|---|
| paired t | sizes, assumes normality | **0.047** |
| Wilcoxon signed-rank | ranks of sizes, no shape assumption | **0.054** |
| sign test | direction only | **0.090** |

Three tests resting on progressively weaker assumptions, straddling 0.05. **The
honest reading is not to pick the friendliest one.** It is that the effect sits
at the edge of what 450 examples can resolve.

Detecting a 0.393 shift with sd 4.197 at 80% power needs **895 paired
examples**. The test split has 450, so this experiment is about **2× short** of
the size that would settle it. That is the actionable number: the fix is more
held-out decisions, not a different test.

Direction is consistent — the tuned model picked better on 195, worse on 162,
identically on 93 — and so are the per-season and per-margin breakdowns. So
"probably real, not established" is the claim the data supports.

The 20.7% exact ties are also why the t-test and Wilcoxon differ. The
difference histogram is symmetric and unimodal, so skew is not the problem; a
spike of 93 zeros is simply not something a normal distribution produces.

**Context for the size of the win: the board beats chance by only 0.50 ranks
out of 12.** Once VBD has sorted twelve players into a narrow band, choosing
between them is close to a coin flip. Measured against the 6.50 that chance
gives by construction:

| | mean rank | gap under chance |
|---|---|---|
| deterministic board | 6.00 | 0.50 |
| fine-tuned adapter | 5.41 | **1.09** |

So the agent finds slightly more than twice the signal the tabular layer did.

Use 6.50 rather than the sampled `random` run (6.36) as the reference for this
comparison. 6.50 is exact by construction — the mean of 1..12 — whereas 6.36 is
one draw of a random chooser and carries its own sampling error; against it the
board's gap reads 0.35 and the ratio changes. The two references answer
different questions and mixing them is how a "0.52" that reproduces from
neither gets published.

A labelling error was caught and is documented in `step9_draft_examples.py`:
the first version labelled each pick with the best actual outcome over all ~200
available players, which returns whoever won the season rather than the right
pick — in 2021 the label was Cooper Kupp for ten consecutive picks, and a
150-pick draft carried only 4–11 distinct labels. Restricting the label to the
12 candidates actually shown fixed it (21–37 distinct labels), and an assert
guards the regression.

See `SERVING.md` for the conversion and serving runbook.

## Sit/start (stage 6)

A second, separate task on the same architecture: given the six flex-eligible
players on your roster this week, which do you start? Score is where your choice
finished among the six once the week was played, so chance is 3.50.

Weekly data did not exist in this project and now does: `step11_weekly_data.py`
pulls nflverse via `nfl_data_py` and joins it on identifiers
(`gsis_id` -> `pfr_id` -> `pid`), never names. It is validated by
reconstruction: summing weekly PPR reproduces the season `ppr` column that came
from Pro-Football-Reference, median difference **+0.00**, mean absolute 0.27,
93.3% within one point. Two independent sources scoring PPR independently agree,
which confirms the join and both implementations at once.

**278 held-out decisions, seasons 2023-2024:**

| | mean rank (of 6) | best of 6 |
|---|---|---|
| chance, by construction | 3.50 | 16.7% |
| random | 3.51 | 17.3% |
| `llama3.1:8b` **un-tuned** | 3.02 | 23.7% |
| start the best season average | **2.92** | 24.1% |
| `fantasy-sit` **fine-tuned** | 3.12 | 20.9% |

### The fine-tune changed nothing measurable

**3.12 against 3.02 un-tuned and 2.92 for the rule.** It clears neither bar.

But the 0.10 gap against the un-tuned base **is not distinguishable from
noise**, and unlike the draft result all three tests agree on that
(`paired_test.py`, same 278 prompts answered by both):

| | |
|---|---|
| mean difference | **+0.097** (worse) |
| 95% CI | **[−0.128, +0.323]** — includes 0 comfortably |
| paired t | p = 0.400 |
| Wilcoxon | p = 0.534 |
| sign test | p = 0.578 |

Detecting an effect that small would need **3,065 paired examples**; there are
278, so this is 11× short. The tuned model picked better on 75, worse on 83,
and identically on 120.

So the honest claim is not "post-training made it worse". It is **"post-training
did nothing detectable, and what it did do points the wrong way"**. Against the
deterministic rule the story is simpler: 2.92 stands, unbeaten by any model.

Best-of-6 fell 23.7% to 20.9% and validity stayed at 100%.

**83% of its answers never stopped.** 232 of 278 ran to the 300-token cap.
The adapter produces the trained sentence correctly and then keeps going,
inventing a rejection for every other candidate:

```
Start Isiah Pacheco (RB). He is averaging 14.5 ... larger sample.   <- trained format, exact
Do not start Marvin Jones (WR); ...                                 <- "Do not start" appears
Do not start George Kittle (TE); ...                                   NOWHERE in the training data
```

An adapter that loses its end-of-sequence token and enumerates the whole
shortlist has fit the surface form of its training data rather than the
decision inside it. That is the most likely explanation of the worse picks too,
and it is a result about post-training rather than about fantasy football.

Scoring is unaffected: `extract_pick` reads the first shortlisted name and the
recommendation is written first. The 300-token cap in `eval_agent.py` bounds
the runaway; without it the first example alone exceeded a 300-second timeout.

### The confound, stated rather than buried

**This ran `epochs=2.0`; the draft adapter ran `epochs=1.0`.** So two
explanations are not separated by this experiment:

| | |
|---|---|
| the task | sit/start leaves less for a language layer than the draft did |
| the hyperparameter | two epochs over 1,233 short templated examples over-trained it |

Both predict what was observed. The lost stop token points at the second, since
format collapse is what over-training on a fixed template looks like.

A second candidate cause sits in `train_adapter.py`: `tok.pad_token =
tok.eos_token`. When pad and EOS are the same token, a collator that masks pad
positions also masks the real EOS out of the loss, so the model never learns to
emit it. The draft adapter has identical code and does stop, which is why this
is a hypothesis and not a conclusion.

**The experiment that separates them is one run:** `--dataset sit --epochs 1`,
scored on the same 278. Roughly 20 minutes of GPU time. Until it exists, the
supportable claim is "this fine-tune made the model worse, and it is defective
in a way consistent with over-training" — not "an LLM cannot do sit/start".

### The base model does NOT beat the tabular layer here

This is the finding, and it is the opposite of the draft result. §11g's question
-- *can a language model beat the number it was handed?* -- turns out to have a
different answer per task, on the same base model, through the same harness:

| | tabular layer | un-tuned Llama | verdict |
|---|---|---|---|
| draft | 6.00 of 12 | **5.80** | LLM wins by 0.20 |
| sit/start | **2.92** of 6 | 3.02 | LLM loses by 0.10 |

It is worse in both held-out seasons (3.16 vs 3.13, 2.89 vs 2.72), so this is an
effect rather than one bad year. **The bar for the fine-tune is therefore 2.92,
the deterministic rule** -- unlike the draft, where the un-tuned base moved the
bar upward.

The two are close to mirror images by decision difficulty:

| margin | rule | un-tuned Llama |
|---|---|---|
| coin flip <1 (n=39) | **2.92** | 3.23 |
| marginal 1-5 (n=101) | **3.01** | 3.21 |
| real 5-20 (n=124) | **2.81** | 2.84 |
| decisive 20+ (n=14) | 3.36 | **2.71** |

The rule wins three of four bands. The band the model wins holds 14 decisions
and should not be leaned on.

### Recency loses, measured twice

| | r | r squared |
|---|---|---|
| season-to-date mean -> next week | **0.389** | 0.152 |
| last 3 weeks mean -> next week | 0.377 | 0.142 |
| this week alone -> next week | 0.308 | 0.095 |

Measured on 18,719 player-weeks, 2019-2024, restricted to players with 8+ games
and 5+ mean PPR. It reproduces at the decision level: starting the hot hand
scores 2.91 against 2.71 for the plain season average, over all 1,774 decisions.
"He is hot right now" is the most common advice in fantasy football and it loses
to an average.

`last3` is still shown in every prompt. Removing it would engineer the answer
into the input; leaving it in makes "does the fine-tune chase recency?" a
question that can be asked of the trained model.

### Weekly is a third as predictable as seasonal, but the rule is stronger

r squared 0.152 against 0.455 for season-level PPG. That predicted more room for
a language layer here than in the draft. **The prediction was wrong**, and the
measurement says why: what matters is not how noisy a single week is but how
much of the available signal the deterministic rule already captures.

| | chance | rule | signal captured |
|---|---|---|---|
| draft (VBD) | 6.50 | 6.00 | 9.1% |
| sit/start (season PPG) | 3.50 | 2.92 | 23.2% |

Averaging six players' noise still orders them usefully. Less is left over here,
not more.

### Known limits of this task

- **2025 is unavailable.** `nfl_data_py` 0.3.3 returns 404 for it, so weekly
  data ends at 2024. Verified against the current release, not assumed.
- **Coverage collapses before 2010** -- 12% of top-250 player-seasons in 2000
  against 99% from 2010, because early-2000s players have no `pfr_id` upstream.
  `USABLE_FROM = 2010`, so ten seasons of the file are unusable for this task.
- **Rosters are simulated,** by a snake draft over last season's PPR finish. No
  real roster exists in any of this data.
- **Only the flex slot.** A QB slot with one QB on the roster is not a decision.
- **1,774 decisions** against the draft task's 2,850, and 278 held out against
  450, so every number here carries wider error bars than the draft's.

---

## What is NOT done

- **Rookies.** 23.2% of a real top 250 cannot appear on the board. Needs PFR
  draft results, 26 seasons, joined on `pfr_id`.
- **A paired significance test** on the un-tuned vs fine-tuned per-example
  ranks. Both CSVs exist; the test does not. See above.
- **Ablations.** One adapter was trained, at one rank, for one epoch. Nothing
  here separates "post-training helps" from "these particular hyperparameters
  help", and no second seed was run.
- **In-season start/sit.** The built system replays completed seasons. Advising
  on a week of a season currently in progress is a different system and cannot
  be validated until that season ends.
- **Full-lineup optimisation.** Start/sit answers the flex slot only. Naming
  nine starters is a different decision shape and none of this scoring applies.
- **2025 weekly data.** Not published by `nfl_data_py` 0.3.3, so weekly work
  ends at 2024 while the season file runs to 2025.

### Resolved

- ~~**Sit/start blocked on weekly data.**~~ `step11_weekly_data.py` adds
  `weekly_ppr.csv` (84,909 player-weeks, nflverse, joined on identifiers,
  validated by reconstruction). See the stage 6 section above.
- ~~**The spec conflict.**~~ `CLAUDE.md` and `README.md` scoped v1 to the draft
  board while `PROJECT_CONTEXT.md` §1 wanted a weekly model and sit/start.
  Resolved in favour of §1: start/sit is now a built system, and all three
  documents describe three systems rather than two.
