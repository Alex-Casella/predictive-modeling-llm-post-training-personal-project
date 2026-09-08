# Fantasy football projection + draft board + start/sit

## What this project is

Three systems. Do not merge them.

1. **Projection model** — a tabular regression model that predicts each player's
   2026 PPR fantasy points. This is NOT an LLM. Use scikit-learn / XGBoost /
   LightGBM. LLMs are the wrong tool for numeric prediction over a table.
2. **Draft assistant** — an LLM-backed CLI that reads the finished board plus my
   current roster and recommends a pick with reasoning. This IS where an LLM
   belongs. Build it only after system 1 is validated. Done: `assistant.py`.
3. **Start/sit assistant** — a SECOND fine-tuned adapter, on a SECOND dataset,
   answering a different question: given the flex-eligible players on my roster
   this week, which do I start? Added deliberately, not as scope creep — the
   point is a second post-training run to compare against the first.

Systems 2 and 3 share one training script (`train_adapter.py --dataset
draft|sit`) and one eval harness (`eval_agent.py --test/--key`). That is not
convenience: copying either would fork the hyperparameters, and a difference
between two results could then be the task or a drifted learning rate, with no
way to tell which. Keep it that way.

## The weekly data (system 3)

`weekly_ppr.csv` — 84,909 player-weeks from nflverse via `nfl_data_py`, built
by `step11_weekly_data.py`. Separate from `fantasy_top250.csv`; neither
replaces the other.

- Joined on identifiers only: `gsis_id` → `pfr_id` → `pid`. The name rule below
  applies here too and is not relaxed because the data came from elsewhere.
- **Validated by reconstruction.** Summing weekly PPR reproduces the season
  `ppr` column from Pro-Football-Reference: median difference +0.00, mean
  absolute 0.27, 93.3% within one point from 2010. Two independent sources
  agreeing is what makes the join trustworthy. Re-run that check if the file is
  ever rebuilt.
- **`USABLE_FROM = 2010`.** Coverage is 12% in 2000 and 99% from 2010, because
  early-2000s players have no `pfr_id` upstream. Ten seasons of the season file
  are unusable for anything weekly.
- **2025 is not published** by `nfl_data_py` 0.3.3 (404). Weekly runs
  2000–2024. Verified against the current release, not assumed.
- 12 `pfr_id`s are claimed by more than one `gsis_id` and at least one pairs two
  different people. All are dropped, never tie-broken.
- Credit nflverse alongside Pro-Football-Reference in anything built from this.

## The data

`fantasy_top250.csv` — 6,500 rows, top 250 players per season, all 26 seasons
from 2000 through 2025 with no gaps.

Columns: `pid, season, rk, player, pos, pos_rk, team, age, g, gs, pro_bowl,
all_pro, ppr, pass_cmp, pass_att, pass_yds, pass_td, pass_int, rush_att,
rush_yds, rush_ypc, rush_td, tgt, rec, rec_yds, rec_ypr, rec_td, fmb, fl,
total_td, two_pt_made, two_pt_pass, pfr_id`

Notes:
**MVP scoring is full-PPR only.** One point per reception. Do not add standard
or half-PPR yet.

**But keep the scoring step separable.** The long-term goal is a board anyone
can use with their own league settings. That means league configuration is
always an argument, never a constant. There is one architectural consequence
worth understanding now: a model that predicts PPR points directly can only
ever serve PPR leagues, because you cannot recover receptions from a PPR total.
Serving other formats eventually requires predicting component stats
(receptions, yards, touchdowns) and applying scoring afterward. The MVP does
not need this, but the code should not make it a rewrite — keep prediction,
scoring, and valuation as separate steps.

- `ppr` is the target variable and the only scoring column in the file.
- `rk` is this season's rank by `ppr`, 1 through 250. `pos_rk` is the rank
  within position by `ppr` (RB1, WR1, ...).
- `vbd_10` / `vbd_rk_10` are points above the last startable player at that
  position. This is draft value, not raw production — it is what the board
  should be ordered on, not `rk`.
- Uses ESPN's standard lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE).
  ESPN also starts 1 D/ST and 1 K, which are absent from this data and do not
  affect skill-position baselines. FB is treated as RB.
- **VBD is league-specific.** The stored `vbd_10` column is valid ONLY for a
  10-team ESPN standard PPR league. It is a cached convenience for the MVP, not
  the source of truth. `vbd.py` is the source of truth — it takes a
  `LeagueConfig` and computes VBD for any team count or lineup.
- **Never hardcode league settings anywhere else.** Import from `vbd.py` and
  pass a config. The long-term goal is that any user can supply their own
  league, so settings must stay an argument.
- These are descriptive, computed from actual end-of-season points. The draft
  board needs VBD computed from *predicted* points using the same baseline
  logic. Do not confuse the two.
- The source's own `Rk` column was NOT used. It ranks by VBD, which
  Pro-Football-Reference computes from standard scoring. Using it would have
  excluded roughly 10 genuine PPR top-250 players per season — pass-catching
  backs and slot receivers that standard scoring undervalues. The rows were
  re-cut and re-ranked by `ppr`. Source columns `FantPt`, `DKPt`, `FDPt`,
  `VBD`, `PosRank` and `OvRank` were dropped for the same reason.
- `pid` is the primary key — this project's own identifier, format `P0001`.
  Use it for all internal joins and grouping.
- `pfr_id` is the Pro-Football-Reference identifier, kept as a foreign key for
  joining additional PFR data (draft results, rosters).
- **Never join or group on player name.** 1,692 distinct players share only
  1,681 distinct names — ten names belong to two different people (two Adrian
  Petersons, two Ricky Williamses, and more), five of those pairs at the same
  position. Name-based identity silently merges two careers into one.
- Data is from Pro-Football-Reference. See `ATTRIBUTION.md`. Credit the source
  in any output, chart, or write-up produced from it.
- Positions present: QB, RB, WR, TE, FB. Offensive line, defense and special
  teams are excluded everywhere in this project.
- Traded players: the source lists a player once as an aggregate (team "2TM" /
  "3TM") plus stub rows for teams he never played a game for. 12 such stubs
  were dropped. If new seasons are added, apply the same rule — keep the row
  with the most games played per player-season.
- Blanks are NOT zeros and have not been imputed. `vbd` blank means something
  different from `two_pt_made` blank. Decide per column and document the choice.
- Every row is an END-OF-SEASON result, not a preseason projection.

## The post-training goal

The fantasy football is the vehicle. **The thing being learned is post-training:
how to take a general model and make it better at one specific decision, and
how to know whether it worked.**

That last clause is the whole project. Making an LLM produce fantasy advice is
easy and worthless — it will produce fluent advice whether or not the advice is
good. So every design choice here exists to make the question answerable:

| choice | what it buys |
|---|---|
| K named candidates, one answer | the answer is checkable, not judged |
| labelled by hindsight from the K shown | a right answer exists per decision |
| score = rank of the pick, chance = (K+1)/2 | a scale with a known floor |
| held-out seasons, never random splits | adjacent decisions cannot leak |
| the un-tuned base scored FIRST | separates "tuning helped" from "the model could already do this" |
| a deterministic layer scored on the same items | separates the language layer from the sort underneath it |

**Two tasks exist so the answer is not one anecdote.** One result is a story;
two results on the same base model through the same harness is a finding — and
they already disagree, which is the most useful thing either of them produced:

    draft       tabular 6.00 of 12   un-tuned Llama 5.80   fine-tuned 5.41
    sit/start   tabular 2.92 of 6    un-tuned Llama 3.02   fine-tuned 2.99

So "can an LLM beat the spreadsheet" has no single answer. It is task-dependent,
and that was measured rather than assumed.

**On draft the fine-tune beats the board and it replicates.** −0.593 (p =
0.0041 / 0.0040 / 0.0068) and −0.587 on a second independently trained adapter
(p = 0.0066 / 0.0070 / 0.0112). Both CIs sit entirely below zero. This is the
only adequately powered comparison in the project and the only one that
replicates.

**On start/sit nothing is distinguishable from anything** — five paired
comparisons, five intervals containing zero. Before reading that as a fact
about the task, note that the board's own score swings **0.43** between the
test seasons while the effects being chased were 0.014–0.126. The bar is
noisier than the signal, so the sit results are a statement about the test set
at least as much as about language models.

### The rules that follow from this

1. **Score the un-tuned base before training, every time.** On the draft task
   this moved the bar from 6.00 to 5.80 and would otherwise have turned a
   failure into a reported success.
2. **Fix the bar before the result exists**, never after.
3. **One training script, one eval harness.** Forking either lets a drifted
   hyperparameter masquerade as a task difference.
4. **Report the negative and the underpowered.** Draft fine-tune vs un-tuned
   base is p = 0.047 / 0.054 / 0.090 and needs 895 against 450 — still
   "suggestive, underpowered", and still the honest claim for *that* pairing.
   Draft vs the board is a different question and is significant; do not let
   one stand in for the other.
5. **Run every pairing, not just the obvious one.** Draft-vs-board went
   unmeasured for days because a `round`→`stage` column rename left the two
   draft CSVs on the old side of a guard in `paired_test.py`. Nothing failed;
   the comparison was simply never available. It turned out to be the
   strongest result in the project. **A guard that hides a result is as costly
   as one that lets a wrong result through.**
6. **A defect in the model is a result, not a nuisance.** The start/sit adapter
   emitted no stop token and enumerated all six candidates on 96% of answers.
   Recorded rather than hidden behind a token cap — and then fixed:
   `tok.pad_token = tok.eos_token` masked the real EOS out of the loss.
   `--pad-token auto` took runaway to 0% and moved decisions by 0.014 on sit
   and 0.007 on draft. **Use `--pad-token auto` on every future run.**
7. **Check what the bar itself does before trusting a gap.** Season-to-season
   swing in the board's own score is 0.11 on draft and 0.43 on sit. An effect
   smaller than its bar's variation is not measurable no matter how the test
   is run.

## Success metric

**Set overlap at 250.** Of the 250 players I predict will finish in next
season's top 250, how many actually do? Reported as a percentage.

This is the only metric that counts. Do not substitute RMSE, R², or rank
correlation as the headline number. Report them as diagnostics if useful.

### The baseline to beat

Carrying last season's PPR top 250 forward unchanged scores **68.8%** mean
overlap across all 25 consecutive year-pairs (range 63.6%-73.6%).

Target is 75%. That is 6.2 points above doing nothing. Any model that does not
beat 68.8% on held-out seasons is worthless and should be discarded, not tuned.

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

Implement this baseline FIRST, before any model. Every experiment reports its
score next to the baseline score on the same held-out years, at every tier.

## Hard rules

1. **No look-ahead.** To predict season N, the model may only use data from
   seasons <= N-1. Any feature built from season N stats is a bug, including
   age-in-season-N if derived from the target row.
2. **Walk-forward validation only.** Train on seasons up to N-1, test on N, step
   forward. Never use a random train/test split — it leaks the future.
3. **Every claim gets checked against the data.** Do not assert a statistical
   relationship without running it. If something is unverifiable with the
   columns present, say so explicitly.
4. Never modify `fantasy_top250.csv`. Write derived data to new files.

## Version control

This project is under git. The commit log is the experiment log — it is how I
find out which change moved the score.

- Commit after every step that produces a verified result. Not at the end of a
  session, not after a batch of changes — after each result.
- **Put the overlap score in the commit message**, so `git log --oneline` reads
  as a history of what worked. Format:
  `<what changed> — @250 XX.X% / @50 XX.X% (baseline 68.8 / 51.2)`
  e.g. `add prior-3yr rolling ppr mean — @250 70.2% / @50 55.4% (baseline 68.8 / 51.2)`
  For commits with no score, prefix with `chore:` or `wip:`.
- If a change makes the score worse, commit it anyway with the worse number.
  Negative results are the useful ones and I do not want them silently dropped.
- One idea per commit. If a commit changes three features at once I cannot tell
  which one mattered.
- Run the git commands for me rather than telling me to run them, but show me
  the message before committing.
- Never `git add .` — stage files explicitly. Never commit anything containing
  credentials (ESPN `swid` / `espn_s2` cookies, API keys). Those go in a
  `.env` file that is listed in `.gitignore`.

## Known open problems (do not silently solve these)

- **Rookies.** This dataset contains nothing about a player before his first
  NFL snap, so rookies are structurally unpredictable from it. Solving this
  requires external data (NFL draft position, college production). Flag it,
  propose sources, but do not fabricate rookie rows.
- **Missing input features.** Predicting season N needs preseason-knowable
  inputs (depth chart, team changes, injury history) that this file lacks.

## How to work with me

I know pandas and notebooks. I have NOT used scikit-learn or any ML library
before, and I am learning this as I build it.

- Explain the *why* before the code. I want to understand the decision, not
  receive a finished script.
- Build in small steps I can run and inspect. One concept per step.
- When there is a choice to make (which model, how to handle a blank, which
  feature), lay out the options and ask me which I want. Do not pick silently.
- When I make a wrong call, tell me it's wrong and why. Do not just implement it.
- Prefer showing me a plot or a printed table over describing a result in prose.
- Do not build the draft assistant, the ESPN integration, or a web UI until I
  say the projection model is done.

## Deferred — revisit later

Decisions consciously postponed. If any of these come up, the answer is "not
yet", not "no".

- **Other league sizes.** `vbd.py` already handles any team count; only the
  cached `vbd_10` column is 10-team specific. Adding 12-team is a config
  change, not new logic. Do it once the 10-team board works end to end.
- **Half-PPR and standard scoring.** Deferred, not rejected. Blocked on the
  model predicting component stats rather than PPR points, and on re-cutting
  the top 250, since each scoring format has a different top 250.
- **User-supplied league settings.** The end goal. Everything above is a step
  toward it. Design for it now by keeping settings as arguments; build it
  later.
- **ESPN integration.** Manual board entry first.
- **In-season updating.** Start/sit is built, but it replays completed seasons.
  Predicting week N of a season currently in progress is a different system and
  is not built. It also cannot be tested until a season finishes.
- **Full-lineup optimisation.** Start/sit answers the flex slot only. Naming
  all nine starters is a different decision shape and none of the current
  scoring applies to it.

## Out of scope for v1

- ESPN API integration (unofficial, undocumented, requires swid + espn_s2
  cookies; revisit after the CLI works with manual entry)
- Live draft tracking
- Web app / anything others use
- Predicting a season already in progress (see Deferred)
