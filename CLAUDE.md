# Fantasy football projection + draft board

## What this project is

Two separate systems. Do not merge them.

1. **Projection model** — a tabular regression model that predicts each player's
   2026 PPR fantasy points. This is NOT an LLM. Use scikit-learn / XGBoost /
   LightGBM. LLMs are the wrong tool for numeric prediction over a table.
2. **Draft assistant** — an LLM-backed CLI that reads the finished board plus my
   current roster and recommends a pick with reasoning. This IS where an LLM
   belongs. Build it only after system 1 is validated.

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
- **Weekly / in-season modelling.** The dataset is season totals only.
- **ESPN integration.** Manual board entry first.

## Out of scope for v1

- ESPN API integration (unofficial, undocumented, requires swid + espn_s2
  cookies; revisit after the CLI works with manual entry)
- Live draft tracking
- Web app / anything others use
- Weekly projections (this data is season totals only)
