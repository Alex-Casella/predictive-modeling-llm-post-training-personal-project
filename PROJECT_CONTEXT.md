# NFL Fantasy Prediction — Project Context

Merged handoff. Supersedes `ppg_feature_design_notes.md`; that document's content is
carried forward here in full, with new material from a subsequent session added and
three direct conflicts flagged in §2.

Self-contained. Assumes no prior conversation.

---

## 1. Project state

**Goal:** an LLM agent for fantasy football that delivers three things:

1. **End-of-season top 250 (PPR)** — a projected board
2. **Sit/start advice** for a user's specific roster
3. **Confidence ratings** — probability a player hits, booms, or busts against his
   projection

Updated weekly as new game data arrives. See §11 for the architecture, which splits
this across two layers; the weekly update applies to the numeric layer, not the LLM.

**Scope decisions (settled):**

| Decision | Value |
| --- | --- |
| Historical span | 2000–2025 (26 seasons) |
| Granularity | Weekly logs **and** season rollups |
| Positions | QB / RB / WR / TE |
| MVP approach | Build both board and weekly model; expect poor accuracy at first |

**Primary data:** `fantasy_top250.csv` — 6,500 rows, top 250 players by PPR points per
season, 2000–2025. Season totals only. Key columns: `ppr` (season PPR points), `g`
(games played). **No week column.**

**Supporting data:**

- `draft_2026_offense.csv` — 81 rows, 2026 offensive draft picks
  (`draft_year, rnd, pick, team, player, pos, age, college, pfr_id`).
  Composition: 36 WR, 22 TE, 12 RB, 10 QB, 1 FB.
  **Undecided:** whether to keep the single FB.
  **Role:** row-level injection so rookies appear in the 250 at all. It is *not* a
  feature join — rookies and veterans share almost no input columns.
- Instagram tier screenshots from `fantasyguides` — rookie tiered ranking, tiers 1–7,
  roughly 31 players. Third-party expert opinion, treated as an additional signal, not
  ground truth.

**Baseline to beat:** carrying forward last season's top 250 unchanged produces
**68.8% set overlap** with the actual next-season top 250. **Target: 75%.**

---

## 2. Conflicts to resolve before building

These three are real disagreements between the prior notes and later work. Do not paper
over them.

### 2a. pandas vs Polars

The prior notes are written entirely in pandas (`groupby().transform()`,
`.value_counts()`, `df.groupby('season')['g'].describe()`).

`nflreadpy` returns **Polars** DataFrames, not pandas. Its README states data loading is
"Fast data loading with Polars DataFrames" and shows `.to_pandas()` as an explicit
conversion step.

Pick one and be consistent:

- **Polars throughout** — every code pattern in §5 and §6 must be rewritten.
- **Convert at the boundary** — call `.to_pandas()` immediately on load and keep the
  existing patterns. Costs memory and time on 26 seasons of play-by-play.

Not resolved. Decide before writing the loader.

### 2b. Two different "75%" targets

The prior notes define 75% as **set overlap** between predicted and actual top 250.
That is a well-defined percentage and a valid target — **for the preseason board.**

The weekly model has no defined metric. "Top 250, week by week, at 75% accuracy" does
not resolve, because weekly PPR is a continuous quantity and accuracy is a metric for
labels. See §3.

**Status:** board target is defined. Weekly target is not.

### 2c. Build order

Prior notes §7: do the preseason board first, leave the weekly model alone.

Later decision: MVP does **both**, accepting that accuracy will likely be poor.

The later decision stands, but the prior reasoning was sound — the board uses data
already on disk and has a number to beat. Suggested reconciliation: build both, but
land the board's §8 sequence first so there is a working measurement loop before weekly
complexity is added.

---

## 3. Defining the weekly target

Weekly PPR points are continuous. Three framings, three metrics:

| Framing | Predicts | Metric | Yields a % |
| --- | --- | --- | --- |
| Points | e.g. 14.2 in Week 3 | MAE / RMSE | No |
| Rank | e.g. WR7 in Week 3 | Spearman | No |
| Bucket | start / flex / bench | Accuracy, F1 | Yes |

Same prediction, three verdicts. Predicted 14.0, actual 8.5: that is 5.5 points of
error, 19 places of rank error, or fully correct — depending on framing.

### 3a. Metric definitions

- **MAE** (mean absolute error) — average of `|predicted − actual|`. In points.
- **RMSE** (root mean squared error) — square the errors, average, take the root. Also
  in points, but larger. On errors of 2, 3, 2, 13: MAE = 5.0, RMSE = 6.8. MAE treats a
  13-point miss as 6.5× worse than a 2-point miss; RMSE treats it as 42× worse.
- **r** (Pearson) — do two lists of *values* move together. Range −1 to 1.
- **Spearman** — Pearson run on *rank order* instead of values. Range −1 to 1.
  Can score 1.0 while every point total is wrong, as long as the order is right.

Given the stated goal is the *ranking* being nearly accurate, Spearman is the metric
that agrees with the goal. Pearson would penalize magnitude errors that never change a
lineup decision.

### 3b. Decision-sensitivity bands

Established by working through McCaffrey's 2023 log: for a first-round pick, no
prediction changes the lineup call. He starts regardless, barring injury or bye.

Stated boundary: **top 50–75 are guaranteed starts.** Ranks ~75–250 are the contested
band where predictions actually flip decisions.

Two consequences:

1. **A bucket metric over all 250 is inflated.** The locked top of the list scores
   correct for free. 75% would be cleared on day one without building anything.
2. **The contested band is the hardest to predict.** Ranks 75–250 are low-volume,
   touchdown-dependent, with usage that swings on other players' injuries. The band
   that matters most is the band a model predicts worst.

Two ways out, not equivalent:

- **Narrow the population** — measure only inside 75–250. Makes 75% a real bar.
  Requires defending the boundary.
- **Keep all 250, change the baseline** — measure lift over the flat-average model.
  Free wins cancel. Loses the clean "75%" phrasing.

**Unresolved sub-question:** the 50–75 boundary is a ratio, not a rank. It depends on
starting slots, which depends on league format (team count, roster size, superflex).
Format has not been specified. Specify it to convert the intuition into a computable
line.

### 3c. Published benchmarks for calibration

- IBM Watson fantasy paper: ESPN weekly projections at **6.81 RMSE**; their best
  combined model **6.78**. A projected weekly score is off by roughly 6.8 points.
- Fantasy Projection Lab: competitive weekly half-PPR **MAE below 5.0** for skill
  positions; **above 7.0** the system likely adds noise over simpler baselines.
  Full-season **r above 0.65** is real signal; **below 0.45** is unreliable for
  decisions.
- Fantasy Football Analytics, 11 sources across 2014–2025: QB is the hardest position
  to project, best seasonal MAE **61.0** (FantasyPros).

**Caution:** these are computed over populations that include the easy top of the list.
Comparing a ranks-75-250 result against them will make a decent model look bad. Any
comparison must be recomputed on the same slice.

Nobody publishes a "75% accurate" weekly number. The industry standard target is the
points branch, where percentage is not the unit.

### 3d. Probability output resolves the metric problem

The boom/bust confidence rating is not a fourth framing bolted onto the three above —
it **replaces** the question. A confidence rating is a probability, and probabilities
have their own scoring rules.

The relevant property is **calibration**: when the model says 75%, does the thing
happen 75% of the time? That is a well-defined, measurable target, and it is what the
original "75%" was reaching for all along.

| Concept | Meaning |
| --- | --- |
| Calibration | Stated probabilities match observed frequencies |
| Brier score | Mean squared error of probability vs. 0/1 outcome. Lower is better |
| Log loss | Penalizes confident wrong answers much harder than Brier does |
| Reliability curve | Plot predicted probability against observed rate; diagonal = calibrated |

A model can be well-calibrated and still low-skill (predicting the base rate every time
is perfectly calibrated and useless), so track calibration *and* discrimination
together. Brier score decomposes into both.

**Consequence for the tabular layer:** a regression that outputs a single number cannot
produce a boom probability. Two routes, decide before building:

- **Predict a distribution** — quantile regression, or a model outputting mean and
  spread. Boom/bust probabilities fall out by integrating the tail.
- **Predict the event directly** — train a classifier on `P(points > boom_threshold)`
  and another on `P(points < bust_threshold)`. Simpler, but each threshold needs its
  own model.

Either way, `boom_threshold` and `bust_threshold` must be defined. Common convention is
relative to the player's own projection, not an absolute point total — a 12-point week
is a boom for a WR4 and a bust for a WR1. **Unresolved.**

---

## 4. PPG as a feature

Season total decomposes as:

```
season_total = ppg × games_played
```

`ppg` measures how good a player is *when he plays*. `games_played` measures
availability. Different things, very different predictability. 200 points in 9 games
and 200 points in 17 games are not the same player.

**Caveat:** the top-250 cut is ranked on the **total**, not PPG. A player averaging
18 PPG over 6 games does not make the list. PPG is a *feature*, not the target.

**Open question:** which half of the product is more predictable? Join season *t* to
*t+1* and compare year-over-year correlation of `ppr/g` against that of `g`. Whichever
is lower is the harder half and deserves more modeling effort.

---

## 5. Look-ahead bias

Also called **data leakage** or **target leakage**.

End-of-season PPG averages *every* game in the season. Using 2026 end-of-season PPG as
a feature for Week 9 of 2026 means the feature contains Week 9's answer plus nine
future weeks. Backtest looks excellent; live model is useless.

**Timing rule: at the moment of prediction, a feature may only use data that had
already occurred.**

Terminology fix: for a single week, "points per game" is just *points*. PPG only means
something across multiple games.

### 5a. Leakage-free pattern (pandas — see §2a)

```python
df = df.sort_values(["player_id", "game_num"])

df["pts_prior_mean"] = (
    df.groupby("player_id")["pts"]
      .transform(lambda s: s.shift(1).expanding().mean())
)
```

- `shift(1)` slides the column down one row so a game's own score never lands in its
  own feature.
- `expanding()` averages all rows above it — window grows 1, 2, 3... games. Contrast
  with plain `.mean()`, which averages the whole season including the future.

| game | points | prior mean |
|---|---|---|
| 1 | 24 | (no prior games) |
| 2 | 8 | 24.0 |
| 3 | 16 | 16.0 |
| 4 | 30 | 16.0 |
| 5 | 12 | 19.5 |

Row 4 reads 16.0 because it averages games 1–3 only; its own 30 is excluded.

**Close cousin:** `.rolling(4)` — fixed window of the last four games instead of
everything so far. Recent form vs. season-long form. Both legitimate; include both.

---

## 6. Cold start and shrinkage

At Week 1 there is zero current-season data.

```python
w = games_played / (games_played + k)
blended = w * current_ppg + (1 - w) * prior_ppg
```

- `k` is how many games of current-season evidence it takes before you trust it as
  much as the preseason estimate.
- The two weights cross at `g = k`.
- With `k = 4`: weight on current season is 0.20 at 1 game, 0.50 at 4, 0.67 at 8,
  0.81 at 17.

**`k` is not chosen by feel.** Backtest past seasons; pick the value minimizing error.
Likely differs by position — RB workload stabilizes faster than WR touchdown rate.

This unifies the two projects: the preseason board becomes the prior the weekly model
shrinks away from as the season accumulates.

Shrinkage also fixes the small-denominator problem: `g = 1` with 22 points gives a PPG
better than any full season in history, off one game. Shrinkage pulls that toward the
positional mean.

---

## 7. Filtering: training vs. output

**Drop low-game players from output only, never from training.**

- **Output filtering** is fine — declining to rank someone you lack information on.
  A product decision.
- **Training filtering** is dangerous. Low-game players are exactly the ~31% turnover
  you need to explain in order to beat 68.8%. Train only on healthy full-season players
  and the model learns a world where nobody gets hurt.

---

## 8. Data caveats

### 8a. `season_games` — derive, don't hardcode

`df.groupby('season')['g'].max()` returns season length per year and doubles as an
integrity check. Expect a step from 16 to 17 when the NFL expanded in 2021.

Then `games_missed = season_games - g` is a usable availability feature.

### 8b. Games ≠ weeks

A 17-game season runs across 18 weeks because of byes. If a weekly loader assumes week
number = game number, every player's running average shifts by one somewhere after his
bye. In nflverse weekly stats there is no row for a bye week, so **game number must be
counted from the rows present, not read off `week`.**

Confirmed in a real PFR game log: McCaffrey's 2023 shows 16 dated rows with a visible
gap between 10/29 and 11/12, and no entry after 12/31. Nothing in the file labels
either gap.

### 8c. `g` is ambiguous

For any team game a player is in one of three states:

1. **Did not dress** — inactive, injured, or on IR
2. **Dressed, near-zero snaps** — healthy but buried
3. **Dressed, real snaps** — genuine opportunity

`g` cannot distinguish 2 from 3. So `season_games - g` conflates injury, healthy
scratches, mid-season signings, and inactive rookies — one number, four causes.

**Unresolvable inside `fantasy_top250.csv`**, because the file only contains the top
250 by PPR, so by construction everyone in it produced. A player whose season was all
state-2 games never made the cut. The evidence was already filtered out.

### 8d. The fix: snap counts

nflverse `load_snap_counts()` provides game-level snap counts sourced from PFR,
**starting 2012**, including `offense_snaps` and `offense_pct` per player per game.

```python
# opportunity — share of the offense he was on the field for
df["snap_share"] = df["offense_snaps"] / df["team_offensive_plays"]

# efficiency — production per unit of opportunity
df["pts_per_snap"] = df["fantasy_points_ppr"] / df["offense_snaps"]
```

Season points ≈ **snap share × efficiency × games available.** Three factors, very
different stability:

- **Snap share** — coach-driven, fairly sticky week to week
- **Efficiency** — bounces a lot, especially touchdown rate
- **Availability** — close to noise

Modeling them separately usually beats modeling the product; each deserves a different
amount of regression to the mean.

**Tradeoff:** snaps start 2012; the season file goes back to 2000. 26 seasons of blunt
`g` versus ~14 of sharp `snap_share` (~3,500 player-seasons — thin for a many-featured
model, fine for a simple one).

**Middle path:** keep `g` in the season file for long history; use snaps only in the
weekly model, where 2012+ is plenty because each season contributes ~17 rows per player
instead of 1.

### 8e. Ragged column coverage

`load_snap_counts()` starting in 2012 is one instance of a general problem. Across
2000–2025, different columns begin in different years. Receiving detail, air yards /
aDOT, and snap counts each have a start year that must be **verified, not assumed.**

Audit before committing to a span. Read the output as a wall — the year a column flips
from `0.00` to a real number is its true start year:

```python
coverage = (
    df.groupby("season")
      .apply(lambda g: g.notna().mean())
      .round(2)
)
```

If the start years differ enough, "2000–2025" is not one dataset — it's three.

### 8f. PFR web-copy artifacts

Manually copied PFR tables carry display artifacts. Observed in a real paste:

- Player name duplicated: `Christian McCaffrey Christian McCaffrey`
- Team doubled: `SFSF`; opponent doubled: `@ WASWAS`
- **Footer rows** labelled `Average` and `Total` in the same shape as data rows — these
  become two phantom "games" if loaded naively
- No week column at all, only `DATE`

Strip footers and de-duplicate fields before any per-game math.

### 8g. Regular season vs. playoffs

Confirm whether PFR season totals are regular-season only or include playoffs. Do this
before any per-game math — it changes both numerator and denominator.

---

## 9. Joining the tier screenshots

The Instagram tier data has **names only** — no `pfr_id`. It must join to
`draft_2026_offense.csv` on player name before it can reach anything else.

Known hazards in the actual names: apostrophes (`Ja'Kobi Lane`, `De'Zhaun Stribling`),
plus whatever PFR does with suffixes.

Never use a bare `how="left"` — it returns a full-length frame with silent NaNs:

```python
merged = tiers.merge(master, on="name", how="left", indicator=True)

print(merged["_merge"].value_counts())
print(merged.loc[merged["_merge"] == "left_only", "name"].tolist())
```

`indicator=True` plus printing the misses is the whole technique.

**Related lead:** nflreadpy exposes `load_ff_playerids()` (ffverse/dynastyprocess player
ids). Inspect what that table contains before hand-rolling a name matcher.

---

## 10. Row-count discipline

Row counts must reconcile at every stack, merge, and filter. Use `assert`, not `print` —
a print inside a 26-season loop scrolls past unread; an assert stops the pipeline.

```python
def stack_seasons(frames, expected_total):
    combined = pl.concat(frames, how="vertical")
    actual = combined.height
    assert actual == expected_total, (
        f"row count mismatch: expected {expected_total}, got {actual}"
    )
    return combined
```

State the expected count *in advance*, then assert it.

---

## 11. Architecture and build environment

Two layers. Keeping them separate is the single most important structural decision in
the project.

```
Weekly nflverse pull
        |
   Tabular model      <- retrained weekly; minutes, near-free
   numbers + probabilities
        |
    LLM agent         <- calls the model as a tool; explains, never computes
        |
  board / sit-start / boom-bust %
```

### 11a. Numeric layer

- **Data:** Polars (native to nflreadpy) or pandas via `.to_pandas()` — see §2a
- **Modeling:** scikit-learn, XGBoost, or LightGBM
- **Retrained weekly.** This is cheap and is where the weekly update belongs.

### 11b. LLM layer

The LLM's job is what a tabular model cannot do: interpret a specific roster, weigh
matchup context, and explain why a number is what it is. It calls the numeric layer as
a tool and reports its outputs.

**Do not fine-tune the LLM weekly.** Two reasons:

1. **Wrong tool for the number.** Gradient updates spent teaching a language model
   arithmetic buy worse numeric accuracy than a tabular model trained in minutes.
2. **Catastrophic forgetting.** Repeated narrow updates on small weekly batches degrade
   prior learning. Weekly fine-tuning on a few thousand stat rows is a known failure
   mode, not a schedule.

New weekly information reaches the LLM through the tool call and its context, not
through weights.

### 11c. On Ollama

Revised from an earlier position of "no, not for this project." More precisely:

- **Agent layer — reasonable.** Ollama serves local models. If the goal is the whole
  system running locally with no API cost, this is a legitimate choice.
- **Numeric layer — no.** Nothing in the prediction pipeline needs a language model.

### 11d. nflreadpy caveat

**nflreadpy caveat, from its own README:** most of the first version was written by
Claude based on nflreadr, use at your own risk. Not a reason to avoid it; it *is* a
reason to sanity-check row counts against a known season rather than trusting output.

---

### 11e. Build order

Five stages. Sequential — each depends on the one above it.

| # | Stage | Where |
| --- | --- | --- |
| 1 | Build the data pipeline | Claude Code, local |
| 2 | Tabular model + backtest | Claude Code, local |
| 3 | Generate fine-tune examples | Claude Code, local |
| 4 | Fine-tune LoRA adapter | Modal, cloud GPU |
| 5 | Serve the adapter | Ollama, local |

**Currently at stage 1.**

Stage 4 cannot begin until stage 3 produces a dataset. Ollama does not train models —
per Unsloth's Ollama tutorial and the `Modelfile` `ADAPTER` directive, a LoRA adapter is
trained with PEFT/transformers elsewhere and then loaded into Ollama for serving.
Modal is that "elsewhere." GPU credits sitting idle during stages 1–3 cost nothing.

### 11f. Stage 3 constraints (read before generating any examples)

Fine-tuning needs `(situation → correct call)` pairs. No such dataset exists to buy; it
must be manufactured from historical seasons. Two traps:

**Label balance.** Generating examples uniformly across all 250 players floods the set
with players nobody would ever bench (§3b). Most of the dataset would teach the model to
say "start him" about locked starters. Weight generation toward the contested band.

**Point-in-time reconstruction.** §5's leakage trap returns, now in natural language and
harder to detect. Every fact in a generated situation description must have been
knowable before kickoff. A phrase like "in what would become his breakout stretch"
teaches the model to read the future. Any narrative written with hindsight is leakage
even when no numeric column leaks.

### 11g. Post-training the agent (stage 4)

**What is being post-trained:** the advice and explanation layer only. Never the numeric
prediction (§11a, §11b). The target capability is *reasoning about a sit/start decision
given the model's outputs*, not computing the outputs.

**Cadence:** not weekly (§11b). Accumulate examples across many weeks and post-train
periodically. Weekly gradient updates on a few thousand rows is a forgetting risk, not a
schedule.

#### SFT vs RL

Two routes, and the second is unusually available here.

**Supervised fine-tuning (SFT).** Train on `(situation → good advice)` pairs generated in
stage 3. Simpler, well-understood, and the standard first move. Requires that the target
advice text actually be good — which means it has to be generated carefully, since the
model will imitate its flaws.

**Reinforcement learning.** Sit/start is one of the rare language tasks with a
**ground-truth outcome that arrives automatically.** Recommend starting A over B, and one
week later the box score says whether that was right. No human labeling required, and
26 seasons of history means the environment can be replayed offline.

This is a genuine RL environment, not a stretch. It is also the harder path and should
not be attempted before SFT works.

#### Reward design (the hard part)

The outcome is verifiable but **not binary**, and treating it as binary will train the
model on noise:

- A scores 12.1, B scores 11.9 → technically "right," but the call was a coin flip.
  Full reward for this teaches the model that a lucky guess was skill.
- A scores 24.0, B scores 4.0 → the same "right," but a genuinely valuable call.
- The right decision can produce the wrong outcome. A well-reasoned start that busts is
  not a bad recommendation; it's variance. A reward function that punishes it teaches
  outcome-chasing.

Candidate shapes to test: reward proportional to the point margin; reward against the
model's own stated confidence (§3d) so overconfidence is penalized; or reward only
decisions where the margin exceeded some threshold, discarding coin flips from training
entirely.

**Unresolved.** Reward design should be treated as its own experiment with its own
backtest, not a detail settled while writing the training loop.

#### Practical setup

- **Technique:** LoRA / QLoRA rather than full fine-tune — far cheaper, and the adapter
  is what Ollama loads (§11e).
- **Where:** Modal, serverless GPU. Python functions with decorators; you write code
  locally and it executes on their GPUs.
- **Base model:** undecided. Must be one Ollama can serve, and the adapter must be
  trained against the *same* base or it will behave erratically.

#### Evaluating whether it helped

The failure mode is a fine-tune that produces more confident-sounding advice without
producing better advice. Guard against it:

- Hold out entire seasons, not random weeks — random splits leak, since adjacent weeks
  of the same player-season are near-duplicates.
- Compare against the un-tuned base model on the same held-out decisions.
- Compare against the tabular layer alone, no LLM. If the agent doesn't beat the number
  it was handed, the language layer is decoration.

#### Note on motivation

Post-training experience is a stated hiring criterion for at least one internship in
this project's files. That is a legitimate reason to build stage 4 even if it never
beats the tabular layer on accuracy — but the two goals must be kept distinct. Fine-
tuning the *advice* layer is defensible work. Fine-tuning an LLM to output point totals
is not, and would be the first thing challenged in an interview.

---

## 12. Next steps

Build both tracks per the MVP decision, but land this measurement loop first — it uses
data already on disk and has a number to beat.

1. **Profile the `g` column.** `groupby('season')['g'].describe()` and
   `value_counts().sort_index()`. Check max `g` steps 16 → 17 in 2021; count rows with
   `g < 4`.
2. **Add `ppg`, `season_games`, `games_missed`.** Three columns, no modeling. Save to a
   new file so the original stays clean.
3. **Build the year-over-year table.** Join each player's season *t* row to his *t+1*
   row on player id. Every row then reads: what was known going in, what actually
   happened. Verify a player appearing in 2019 and 2021 but not 2020 does not
   accidentally link across the gap.
4. **Race three rankings against the baseline.** For every season 2001–2025, produce a
   predicted top 250 three ways:
   - carry forward last year's top 250 by total points (reproduces 68.8%)
   - rank by last year's `ppg`
   - rank by last year's `ppg` × a simple games estimate

   Measure set overlap each time. **No machine learning — this is sorting and set
   intersection.**
5. **Read the result before writing more code.** If PPG-based ranking beats 68.8%, the
   instinct is confirmed and a model on top is justified. If not, weeks of building on
   a bad feature were avoided.

**Why no ML in that list:** a model can only find signal already present in the
features. If PPG doesn't beat the baseline when ranked on directly, no amount of
gradient boosting rescues it. Sorting a column answers the question faster and more
honestly.

---

## 13. Known ceiling: rookies

The 68.8% baseline can only predict players already in last year's top 250, so
**rookies are guaranteed misses.** Every year some portion of the top 250 are rookies,
putting a hard floor on the carry-forward approach.

`draft_2026_offense.csv` and the tier screenshots exist to address this, but that is a
separate feature source. Before building for it, quantify it: **once step 3's table
exists, count how many of each season's actual top 250 had no prior season at all.**
That number is how much of the 31% gap is rookies.

---

## 14. References

- nflreadpy (Python port of nflreadr): https://github.com/nflverse/nflreadpy
  - `load_player_stats()` — weekly *or* seasonal player stats (one function, both
    granularities; find the argument that switches them)
  - `load_snap_counts()` — game-level snap counts, 2012+
  - `load_rosters_weekly()` — roster status by season-week
  - `load_injuries()` — injury designations
  - `load_ff_playerids()` — ffverse/dynastyprocess player ids
  - `load_ff_opportunity()` — expected yards, touchdowns, fantasy points
  - `load_schedules()` — needed for bye weeks
- nflreadr `load_snap_counts()`: https://nflreadr.nflverse.com/reference/load_snap_counts.html
- Snap counts data dictionary: https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html
- nflreadpy docs: https://nflreadpy.nflverse.com
- nflverse pbp covers every play of every game back to 1999

**Post-training (stage 4)**

- Modal — serverless GPU compute for Python: https://modal.com
- Unsloth, LoRA fine-tune to Ollama walkthrough:
  https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama
- Ollama `Modelfile` `ADAPTER` directive is how a trained LoRA adapter gets served
  locally; Ollama itself does not train

---

## 15. Open questions

**Metric definition**

- Weekly target: points, rank, bucket, or probability? (§3) Board target is settled at
  75% set overlap. The confidence-rating product points at **probability + calibration**
  (§3d), which is likely the answer — confirm and close this out.
- Distribution model or direct-event classifier for boom/bust? (§3d)
- How are `boom_threshold` and `bust_threshold` defined — absolute points, or relative
  to the player's own projection? (§3d)
- Calibration and discrimination both need tracking. Which headline number goes on the
  README: Brier, log loss, or reliability-curve deviation?
- League format — team count, roster size, superflex? Needed to convert "top 50–75 are
  locked" into a computable line. (§3b)
- Narrow to ranks 75–250, or keep all 250 and measure lift over baseline? (§3b)
- **Open trap, unresolved:** ranking a *constant* prediction column for Spearman. A flat
  season-average prediction has no variance, so every value ties. Work out what that
  does to the rank correlation before relying on Spearman as the headline metric.

**Data**

- Which is more stable year over year, `ppr/g` or `g`?
- How many rows have `g < 4`, and does that vary by season?
- Does `g` credit a game where a player dressed but took zero snaps? (Needs snap counts.)
- Are PFR season totals regular-season only, or do they include playoffs?
- What fraction of each season's top 250 are rookies?
- What value of `k` minimizes shrinkage error, and does it differ by position?
- Verified start years for receiving detail, air yards / aDOT? (§8e)

**Engineering**

- Polars or pandas? (§2a)
- Keep or drop the single FB in `draft_2026_offense.csv`?
- Which LLM serves the agent layer, and local (Ollama) or hosted API? (§11c)
- Which base model for the LoRA adapter? Must be Ollama-servable, and the adapter must
  be trained against that same base. (§11g)
- SFT first, or straight to RL? (§11g — SFT first is the default recommendation)
- **Reward function shape** for the RL route: margin-proportional, confidence-weighted,
  or threshold-gated to discard coin flips? Needs its own backtest. (§11g)
- Post-training cadence — how many weeks of accumulated examples before a retrain?
- Does the fine-tuned agent beat (a) the base model and (b) the tabular layer alone on
  held-out seasons? If not, the language layer is decoration. (§11g)
- How does the agent read a user's roster — manual entry, league platform import, or
  paste?
- Tool-call contract between the layers: what exactly does the numeric model return per
  player per week?

---

## 16. Working note

If a loader runs clean on the first try across 26 seasons, be suspicious rather than
relieved. Ragged historical data doesn't usually cooperate that fast, and a pipeline
that doesn't complain is often one that's dropping rows quietly.
