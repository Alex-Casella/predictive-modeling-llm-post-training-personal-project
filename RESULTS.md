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
| Widened pool, returners (`step6`) | 42.1 | 51.2 | 61.4 | 65.2 | 67.8 | **69.1** |
| Target | | | | | | 75.0 |
| **Structural ceiling without rookie data** | | | | | | **76.8** |

Set overlap %, mean over 25 walk-forward season pairs. The baseline was
reproduced independently to 0.0 at all six tiers (`step4_race.py`).

**Rookies via draft position (`step17`) is the best method, and it is reported
on a different span**, so it sits in its own table rather than being compared
to rows above it. The boost was fitted on 2001–2015 and these seasons were
never shown to that choice:

| held out, 2016–2025 | @25 | @50 | @100 | @150 | @200 | **@250** |
|---|---|---|---|---|---|---|
| Carry-forward baseline | 42.0 | 50.8 | 61.4 | 64.9 | 67.6 | **68.6** |
| **+ rookies, boost 1.5** | 42.0 | 50.8 | 63.3 | 67.1 | 70.0 | **71.9** |
| delta | 0.0 | 0.0 | +1.9 | +2.2 | +2.4 | **+3.3** |

**+3.3 points on held-out seasons, against +0.3 from everything before it.**
That is 3.1 short of the 75.0 target, and the 76.8 "ceiling" no longer applies:
it assumed rookies were unreachable, and this reaches some of them.

**Read the elite tiers before celebrating.** @25 and @50 did not move at all.
Every point comes from tier 100 and below, which is exactly the pattern the
success-metric section warns about — rookies rarely finish top 50, so this
widens the pool where it was already easiest. The top of the board, where draft
picks are most expensive, is untouched by anything in this project.

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

### But the comparison that actually answers §11g was never run

The table above tests the fine-tune against the **un-tuned base**, which asks
"did post-training help". §11g asks something else — *can the language layer
beat the number it was handed?* — and that is the fine-tune against **the
board**. Both CSVs had been sitting in the repo for days:

| | | |
|---|---|---|
| board | **6.004** | the bar |
| `fantasy-draft` | **5.411** | |
| mean difference | **−0.593** | sd 4.366, se 0.206, 2.88 noise widths |
| 95% CI | **[−0.997, −0.190]** | entirely below zero |
| paired t / Wilcoxon / sign | **0.0041 / 0.0040 / 0.0068** | all three reject |
| n for 80% power | **425** | against 450 available |

**This is the only adequately powered comparison in the project, and it is
significant on all three tests.** The language layer beats the sort it was
handed, on held-out seasons, by 0.593 ranks of 12. Better on 204, worse on 152,
identical on 94.

**And it replicates twice.** Three adapters were trained from scratch on the
same data at the same 1 epoch, differing only in the pad token and the seed.
All three are tested against the same board:

| | score | gap vs board | paired t | 95% CI |
|---|---|---|---|---|
| `fantasy-draft` | 5.411 | **−0.593** | 0.0041 | [−0.997, −0.190] |
| `fantasy-draft-pad` | 5.418 | **−0.587** | 0.0066 | [−1.008, −0.166] |
| `fantasy-draft-pads1` | 5.249 | **−0.762** | **0.0003** | [−1.175, −0.348] |

Three separately trained adapters, three independent evaluation runs, every
confidence interval entirely below zero. Nothing else in this project
replicates even once.

**The draft noise floor is 0.160.** `pads1` differs from `pad` only in the
seed, so the gap between them is pure run-to-run variation: −0.160,
p = 0.105 / 0.323 / 0.235, CI [−0.354, +0.033]. That is the yardstick the
board result has to be read against:

    board gap    0.593 – 0.762
    noise floor  0.160
                 ─────────────
                 3.7x to 4.8x noise

Which is a weaker statement than the p-values alone imply, and the right one.
A p-value asks whether a different *test set* could have produced this; the
noise floor asks whether a different *training run* could have. Both had to be
answered and only one of them was, until now.

Normalised for K, draft is noisier per unit of scale than start/sit — 0.160/12
= 0.0133 against 0.043/6 = 0.0072. That is the opposite of what the bar's
season-to-season swing suggested (0.11 on draft, 0.43 on sit), and the two are
not the same quantity: one is variation in the target, the other in the model.
No mechanism is offered for the difference.

The two adapters differ from each other by **+0.007**, p = 0.933 / 0.644 /
0.545, CI [−0.148, +0.162] — so the pad-token change that eliminated the
start/sit runaway does not touch draft decisions either. Third independent
confirmation that the stop-token defect and the decision quality are separate
systems.

**An open question this run sharpened rather than settled.** The draft adapter
was trained with the *same* `pad_token = eos_token` line and has never shown
the runaway defect, on either version. Identical code, identical bug, fatal on
one dataset and harmless on the other. Nothing measured here explains that.

**Why it went unmeasured for days is the lesson.** `eval_agent.py`'s `round`
column was renamed `stage` when the harness grew a second task. The two draft
LLM runs predate that rename; every sit file postdates it. `paired_test.py`
correctly refuses to pair files from either side of the rename — so the
base-vs-tuned comparison still worked (both files were old and agreed) while
the board-vs-tuned comparison was silently unavailable. A guard against
comparing incompatible files hid the project's best result, and nothing
surfaced it because nothing ever failed. The two files have since had their
headers migrated, with every other column asserted byte-identical.

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

### GRPO — a second training method reaches the same place

`train_grpo.py`. Everything above is supervised fine-tuning: the model is shown
the right answer and trained to reproduce it. This is reinforcement learning,
and it is never shown a right answer at all. For each prompt it samples 8 picks
at temperature 1.1, scores each by `(K+1-rank)/K` — where the pick actually
finished among the 12 shown — and pushes toward the ones that beat *the group's
own average*. An invalid answer scores 0.

Trained on the first 600 rows of `sft_train.jsonl` (seasons 2007–2010), one
epoch, `beta=0.04`, seed 0, from the **base** model rather than from an SFT
checkpoint. Scored on the same 450 held-out picks as every adapter above;
overlap between the 600 trained prompts and the 450 test items is **0**, and
the two spans are thirteen years apart.

**Three instrumentation gates were read before the score, and the order was
fixed in advance.** This matters because two of them had already failed once:

| | smoke test (temp 0.9) | this run (temp 1.1) | |
|---|---|---|---|
| `invalid_pct` | 0.0 | **1.5** | higher temperature cost little |
| `board_agreement_pct` | 100.0 | **32.4** | no collapse onto the sort |
| `flat_group_pct` | (the counter lied) | **14.0** | 86% of groups carried gradient |

`frac_reward_zero_std` was **1.0 at temperature 0.9** — every sample of a prompt
named the same player, so every within-group advantage was exactly zero and the
gradient was zero with it. The loss curve looked fine. Two hours of that would
have trained nothing. Temperature 0.9 → 1.1 is the entire fix, and this time the
project's own flat-group counter agreed with TRL's metric instead of reporting a
reassuring 0.0%.

| | score | gap vs board | p (t / Wilcoxon / sign) | 95% CI | n for 80% power |
|---|---|---|---|---|---|
| `fantasy-draft-grpo` | **5.351** | **−0.653** | 0.0015 / 0.0027 / 0.0152 | [−1.053, −0.253] | **345** vs 450 |

Best draft score in the project, adequately powered, all three tests reject.
Better on 187, worse on 142, identical on 121.

**Against the SFT adapter it is nothing.** −0.067, p = 0.7445 / 0.7728 / 1.0,
CI [−0.467, +0.334], 149 better against 148 worse. `n` for 80% power at that
effect size is **33,212** against the 450 available — and 0.067 is well inside
the 0.160 seed noise floor either way. The claim is *matches*, not *beats*, and
the histogram says it without any test: the two adapters make **identical picks
on 34%** of decisions despite sharing no training objective.

| | method | trained on | score | gap vs board |
|---|---|---|---|---|
| `fantasy-draft` | SFT | 1,950 labelled | 5.411 | −0.593 |
| `fantasy-draft-pad` | SFT | 1,950 labelled | 5.418 | −0.587 |
| `fantasy-draft-pads1` | SFT, seed 1 | 1,950 labelled | 5.249 | −0.762 |
| `fantasy-draft-grpo` | **GRPO** | **600 unlabelled** | **5.351** | **−0.653** |

**Four adapters, two training methods, one conclusion.** Supervised learning on
1,950 labelled examples and reinforcement learning on 600 unlabelled prompts
land in the same place, from opposite directions, and the four gaps span
0.587–0.762 against a 0.160 noise floor. Every CI is entirely below zero.

The reading that follows is about the ceiling, not the methods: **if a labelled
approach and an unlabelled approach converge, the limit is upstream of
training** — in what the prompt can see, or in how much of a season is knowable
at draft time at all. Trying a third optimiser is not the experiment; changing
what the model is told is.

Not attempted, and the obvious next run: GRPO starting from the SFT adapter
rather than the base, which is how post-training pipelines normally stack. A
gain of half a rank there would be detectable at n=450 — 0.653 was found with
345 needed — whereas the 0.067 measured here never could be.

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
| `fantasy-sit-e1` **fine-tuned, 1 epoch** | 2.99 | 25.9% |
| `fantasy-sit-e1s1` same recipe, seed 1 | 3.04 | 24.5% |
| `fantasy-sit` **fine-tuned, 2 epochs** | 3.12 | 20.9% |

### The 2-epoch fine-tune changed nothing measurable

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

### The epoch ablation — three comparisons, three nulls

The 2-epoch run above left a confound: **it ran `epochs=2.0` while the draft
adapter ran `epochs=1.0`,** so "sit/start leaves less for a language layer" and
"two epochs over-trained it" both predicted what was seen. `--dataset sit
--epochs 1` on the same 278 items separates them. It scores **2.99**.

The point estimate moved the way over-training predicts. Nothing else did:

| the 1-epoch adapter, against | mean difference | reading |
|---|---|---|
| its own 2-epoch sibling | **−0.126** better | p = 0.111 / 0.192 / 0.111, CI **[−0.280, +0.029]** |
| the un-tuned base (3.02) | **−0.029** better | no test rejects at α = 0.05 |
| the board / the rule (2.92) | **+0.068** worse | no test rejects at α = 0.05 |

Every interval contains zero. **The 1-epoch adapter is not measurably different
from anything on this test set** — not from the base it was tuned from, not
from the sort it was meant to beat, not from its own sibling.

Reading the third row as "it matches the rule now" would be wrong, and it is
the easiest mistake here. A test that cannot resolve a 0.068 gap would also
miss a genuine 0.068 disadvantage; absence of evidence is not evidence of
absence. The supportable sentence is *"still behind the rule, by an amount
278 examples cannot resolve."*

### The seed replicate — what the noise floor actually is

Every gap above was measured against zero, because nothing in this project had
ever been trained twice. `--seed 1`, everything else identical to the 1-epoch
run, gives the first answer to *"how much does a rerun move on its own?"*

**3.04 against 2.99. Mean difference +0.043**, sd 1.123, CI
**[−0.089, +0.175]**, p = 0.522 / 0.671 / 0.659.

| pair | what varies | mean gap |
|---|---|---|
| seed 0 vs seed 1 | nothing but the draw | **0.043** |
| 1 epoch vs 2 epochs | the hyperparameter | **0.126** |

So the epoch gap is about **three times** the one measurement of run-to-run
noise available. It was not measuring the seed. It keeps whatever weight its
p-values allow — which is not much, but it is not nothing either.

**The aggregate hides how unstable the individual decisions are.** Two adapters
built from the same recipe disagree on one flex call in six:

| pair | items answered differently | split |
|---|---|---|
| seed only | **46 / 278 (16.5%)** | 21 better, 25 worse — a coin toss |
| 1 vs 2 epochs | 57 / 278 (20.5%) | 35 better, 22 worse — leaning |

The epoch change did not make the model disagree with itself much *more often*
(57 against 46). It biased *which way* the disagreements fell. That is a
different mechanism from what the mean gap alone suggests, and it is only
visible with a replicate to compare against.

Two practical consequences. A single flex recommendation from this system is
not a stable object — reseed and it changes about 17% of the time, so it should
never be presented as *the* answer. And detecting an effect the size of the
seed gap would need **5,315 paired examples**, 19× what exists, which sets a
floor on what any future comparison here can resolve.

Runaway generation also moves with the seed: 93% here against 96% at seed 0.
The 83% → 96% change between epoch counts is larger than that, so the
direction reported below is not a seed artefact.

**One replicate bounds the noise, it does not estimate it.** 0.043 is a single
draw, not a standard deviation. Two more seeds would give the spread; until
then "the epoch gap is 3× the noise" rests on one number.

### Why every comparison here is underpowered

Detecting the epoch gap at 80% power needs **856 paired examples** against the
278 available, 3.1× short. That is the third time this project has landed on
"suggestive, underpowered" — the draft gap needed 895 and the base-vs-tuned
gap 3,065. It is not bad luck. It is what a test set of this size buys, and it
is a property of the design rather than of any one result.

**What it does buy is an epoch-matched headline.** Both tasks now have a
1-epoch adapter, so the cross-task comparison no longer varies two things at
once — not because the epoch question was answered, but because it was
sidestepped:

| at 1 epoch | board | un-tuned | fine-tuned |
|---|---|---|---|
| draft (of 12) | 6.00 | 5.80 | **5.41** — beats both |
| sit/start (of 6) | 2.92 | 3.02 | **2.99** — beats neither, distinguishably |

**The lost stop token is not over-training.** Runaway generations went 83% at
two epochs to **96% at one** — 267 of 278. More training produced *more*
stopping, the opposite of what format collapse predicts, so that explanation is
out. The surviving candidate was the `tok.pad_token = tok.eos_token` line in
`train_adapter.py`: when pad and EOS are the same token, a collator masking pad
positions masks the real EOS out of the loss. **That was tested and confirmed —
see below.** Why one epoch stops *less* often than two is not explained by
anything measured here, and is recorded as an observation rather than a
mechanism.

Best-of-6 recovered to 25.9% from the 2-epoch run's 20.9%, above the un-tuned
23.7% and the rule's 24.1%. Validity stayed at 100%. Per-season means were
3.04 (2023) and 2.94 (2024), so this is not one year carrying it.

### The pad-token fix: a partial one, and a clean dissociation

`--pad-token auto` gives the trainer a reserved Llama pad id distinct from EOS,
so the real end-of-sequence token stays in the loss. Same seed, same epoch
count, same data — one line changed.

**It worked. Runaway went to zero.**

| | runaway | mean rank | picks identical to `e1` |
|---|---|---|---|
| pad = eos, seed 0 (`e1`) | 96% | 2.993 | — |
| pad = eos, seed 1 (`e1s1`) | 93% | 3.036 | 83.5% |
| **pad = reserved (`e1pad`)** | **0%** | **2.978** | **90.3%** |

> **Retraction.** This section first recorded **83%** and concluded the fix was
> partial and the cause still unidentified. That was wrong. Two direct probes —
> 8 individual prompts and a 40-item eval — produced **zero** capped answers,
> and the harness counts a cap through the same `ask_ollama` call in both
> paths, so there is no mechanism by which the rates could differ. 83% is also
> exactly the 2-epoch model's rate, which is the likely source. Nothing was
> re-measured to produce this correction; the original figure simply could not
> be reproduced. Commit `d2959f2` carries the wrong version.

One line — a pad id distinct from EOS — took an adapter that never stopped on
96% of answers to one that stops on all of them, in the exact trained format:

```
Start Isiah Pacheco (RB). He is averaging 14.5 points per game over 4 games and
faces MIN this week. His last three games average 16.2, above his season figure
of 14.5.
```

against `e1` on the same prompt, which recommends, rejects all five others, and
then **starts the whole answer again** until the token cap — a greedy-decoding
loop, not novel text.

**Stopping and picking came apart.** The decision score moved −0.014 — p =
0.764 / 0.746 / 0.701, CI [−0.108, +0.079], and 80% power would need **24,081**
paired examples against 278, the widest gap between what a question needs and
what this test set holds anywhere in this project.

More telling than the p-value: **changing the pad token perturbed the model's
decisions LESS than changing the seed did** — 90.3% of picks identical against
83.5% for the seed pair, sd 0.797 against 1.123 — while eliminating a defect
that seed changes barely touched. One line changed how the model *ends* an
answer without meaningfully changing which player it *names*.

That is worth more than the null it sits next to. It says the format defect and
the decision quality are **separable failures**, which retires an earlier
reading in this file: that the adapter "fit the surface form of its training
data rather than the decision inside it, and that is the most likely
explanation of the worse picks too". The surface form was fixed completely and
the picks did not move. Whatever explains the decisions, it is not the format.

It also means the defect was never evidence about the task. Every start/sit
score in this file was produced by a model that could not stop talking, and
fixing that changed the scores by 0.014.

**Cause, confirmed.** `tok.pad_token = tok.eos_token` in `train_adapter.py`:
when pad and EOS are the same token, masking pad positions masks the real EOS
out of the loss and the model never learns to emit it. Everything else was
ruled out first — over-training (runaway *rises* as epochs fall), serving and
prompt format (the un-tuned base has no runaway on the same prompts through the
same Ollama path), and the chat template (no literal `<|eot_id|>` in any
output). `--pad-token auto` is the fix and should be the default for any
future run.

### Giving the prompt something the sort cannot see

Every number in the start/sit prompt described the *player*, and the rule sorts
on one of them. The model was being asked to disagree with a sort using only
the sort's own inputs. `step15_matchup.py` adds the one thing available for
free that the rule cannot see — how many points this week's opponent has
allowed to the player's position, computed from prior weeks only:

```
K.J. Osborn (WR) vs KC - 7.9 points per game over 4 games, 8.5 over his last 3;
                         KC has allowed 28.7 per game to WRs, 21 most of 32
```

**Asked whether the field was usable before asking whether a model could use
it.** Otherwise a null result has two explanations and no way to separate them:
the model ignored a useful field, or the field was not useful. A deterministic
sort is the cleanest consumer of a feature — no prompt, no tokenizer, no
sampling — so `step16_matchup_signal.py` blends it into one
(`ppr + w × z(allowed)`, weight fitted on 2011–2022, applied unchanged to
2023–24).

| | mean rank of 6 | |
|---|---|---|
| the sort, ppg alone | 2.924 | the bar |
| the sort, ppg + matchup | 2.878 | p = 0.539 / 0.814 / 0.897 |
| `llama3.1:8b`, no matchup | 3.022 | |
| `llama3.1:8b`, **with matchup** | **2.924** | p = 0.273 / 0.247 / 0.228 |

**Correlations with what the player actually scored:** ppg **+0.320**, last-3
+0.278, points allowed **+0.070**. The field points the right way and is weak.

**The un-tuned base moved 3.02 → 2.92 and drew level with the rule** — the
first time any model has matched the sort on this task. Mean difference
**−0.097**, sd 1.475, CI **[−0.270, +0.076]**, and 80% power would need
**1,810** paired examples against 278. A fourth null.

Two independent probes of the same field land on the same side — the sort
gained 0.047, the model 0.097, neither significant. Two weak agreeing signals
are worth more than one, and are still not a result.

Where the model's gain sits is at least consistent:

| margin | without | with |
|---|---|---|
| coin flip <1 (n=39) | 3.23 | 3.15 |
| marginal 1–5 (n=101) | 3.21 | 3.15 |
| **real 5–20 (n=124)** | 2.84 | **2.69**, best-of-6 25.8% → 31.5% |
| decisive 20+ (n=14) | 2.71 | 2.79 |

Almost all of it comes from mid-margin decisions, where a matchup plausibly
decides between two comparable players; the near-ties and the blowouts barely
move. On n=124 that pattern is suggestive and nothing more.

**This was the cleanest comparison in the project and it still could not
resolve 0.097.** Same weights, same model, temperature 0, no training anywhere
— the seed noise floor of 0.043 does not even apply, and the only randomness
left is which 278 decisions were drawn. If *this* design cannot separate a 0.1
effect, the binding constraint is the size of the test set, not the models.

**A near miss worth recording.** Written the quick way, the sort's weight was
chosen by running a grid on the 2023–24 test set and keeping the best. That
gave **2.820**, a 0.10 improvement, and it was an artefact of picking the
winner after seeing the answers. Fitted on 2011–2022 instead, the same idea
gives 0.047 and fails every test. The fitting curve is also nearly flat below
w = 1.25 (a 0.014 spread), so the minimum is weak evidence for the weight it
selects. Same shape as the stale-adapter bug: a plausible number produced by a
broken procedure.

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

- **Rookies.** 23.2% of a real top 250 cannot appear on the board, which is
  what caps it at 76.8%. **The blocker is removable and the source is already
  installed**: `nfl_data_py.import_draft_picks()` returns NFL draft position
  back to 2000 with a `pfr_player_id` that joins the same way the weekly data
  did. `draft_2026_offense.csv` covers only 2026 — enough to APPLY a rookie
  model, not to train one.

  This is the only remaining lever on the headline metric. Finding #1 proved
  @250 cannot be moved by reordering, only by changing the candidate pool, and
  adding rookies is exactly that. Expect some of the 7.7 points, not all: draft
  position makes rookies rankable, not predictable, and the widened-pool
  precedent netted +0.36 from a smaller and likely higher-precision group.
- **Ablations.** One dimension has now been varied: sit at 1 epoch scores 2.99
  against 3.12 at 2, and the headline is epoch-matched at 1 (see *The epoch
  ablation* above). The epoch gap itself is not resolved — it needs 856 paired
  examples against 278. Rank was 16 everywhere and the learning rate never
  moved.

  One seed replicate now exists, on the sit 1-epoch config only: it puts the
  noise floor at **0.043**, roughly a third of the epoch gap. That is one draw
  rather than a spread, and **no replicate exists for the draft adapter at
  all**, so 5.41 and its 0.39 gap against the base still have nothing to be
  judged against.

  The first attempt at that ablation **produced no result.** The volume
  path is keyed on the dataset alone, so both runs targeted `/adapter/sit`; the
  second never overwrote it, `finish_adapter.sh` downloaded the 2-epoch files,
  served them as `fantasy-sit-1ep` and scored them. The eval was byte-identical
  to the 2-epoch run — same mean rank, same md5, same Ollama content ID
  `e69c6953baa1` — and it read as a plausible ablation until the identity was
  noticed.

  Logged here because it is a result about the harness rather than the model:
  at `temperature 0` decoding is deterministic, so **two identical eval CSVs
  are proof of identical weights, not of a reproducible finding.** The fix is
  `--tag` (a per-run volume path), `EXPECT_EPOCHS` (provenance asserted, not
  printed), and a content-ID collision check before any scoring runs.
  `run_ablation.sh` chains all three.
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
