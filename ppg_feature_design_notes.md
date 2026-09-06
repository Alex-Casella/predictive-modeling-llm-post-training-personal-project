# PPG Feature Design — Handoff Notes

Context note for carrying this work into a new session. Self-contained; assumes no prior conversation.

---

## 1. Project state

**Goal:** predict the 2026 fantasy football top 250 (PPR).

**Primary data:** `fantasy_top250.csv` — 6,500 rows, the top 250 players by PPR points for each season from 2000 through 2025 (26 seasons). Season totals only. Relevant columns include `ppr` (season PPR points) and `g` (games played). **There is no week column.**

**Supporting data:**
- `draft_2026_offense.csv` — 81 rows, 2026 offensive draft picks (round, pick, team, player, pos, age, college, pfr_id).
- Instagram tier screenshots from `fantasyguides` — a rookie tiered ranking, tiers 1 through 7, roughly 31 players. Third-party expert opinion, treated as an additional signal, not ground truth.

**Baseline to beat:** carrying forward last season's top 250 unchanged produces **68.8% set overlap** with the actual next-season top 250. **Target: 75%.**

---

## 2. The idea that was evaluated

> Compute points per game (`ppr / g`) for each player-season, use historical patterns to better predict the 2026 top 250, and then use the same thing to predict a player's next-week points.

**Verdict: the first half is sound. The second half has a fatal timing flaw as described.**

### 2a. Why PPG is a good feature (keep this)

Season total decomposes as:

```
season_total = ppg × games_played
```

`ppg` measures how good a player is *when he plays*. `games_played` measures availability. These are different things with very different predictability, and lumping them into one number (season total) throws away that distinction. 200 points in 9 games and 200 points in 17 games are not the same player.

**Caveat:** the top-250 cut is ranked on the **total**, not on PPG. A player averaging 18 PPG over 6 games does not make the list. So PPG is a *feature*, not the target.

**Open question to resolve empirically:** which half of the product is more predictable? Test by joining season *t* to season *t+1* and comparing the year-over-year correlation of `ppr/g` against the year-over-year correlation of `g`. Whichever is lower is the harder half and deserves more modeling effort.

### 2b. Why the week-by-week half breaks — look-ahead bias

Also called **data leakage** or **target leakage**.

End-of-season PPG is an average over *every* game in the season. To use 2026 end-of-season PPG as a feature when predicting Week 9 of 2026 means the feature already contains Week 9's answer plus nine more future weeks. The backtest will look excellent; the live model is useless, because in the real Week 9 that number does not exist yet.

Timing rule: **at the moment of prediction, a feature may only use data that had already occurred.**

Also a terminology fix: for a single week, "points per game" is just *points*. PPG only means something across multiple games.

---

## 3. The leakage-free feature pattern

```python
df = df.sort_values(["player_id", "game_num"])

df["pts_prior_mean"] = (
    df.groupby("player_id")["pts"]
      .transform(lambda s: s.shift(1).expanding().mean())
)
```

- `shift(1)` slides the column down one row so a game's own score never lands in its own feature.
- `expanding()` averages all rows above it — a window that grows 1, 2, 3... games. Contrast with plain `.mean()`, which would average the whole season including the future.

Worked example:

| game | points | prior mean |
|---|---|---|
| 1 | 24 | (no prior games) |
| 2 | 8 | 24.0 |
| 3 | 16 | 16.0 |
| 4 | 30 | 16.0 |
| 5 | 12 | 19.5 |

Row 4 reads 16.0 because it averages games 1–3 only; its own 30 points are excluded.

**Close cousin:** `.rolling(4)` — same idea, fixed window of the last four games instead of everything so far. Recent form vs. season-long form. Both are legitimate features; include both.

---

## 4. Cold start and shrinkage

**Problem:** at Week 1 there is zero current-season data. Early-season players are unpredictable because there's nothing to average yet.

**Standard fix — shrinkage.** Instead of an arbitrary "drop players with under 4 games" cutoff, blend continuously:

```python
w = games_played / (games_played + k)
blended = w * current_ppg + (1 - w) * prior_ppg
```

- `k` is how many games of current-season evidence it takes before you trust it as much as the preseason estimate.
- The two weights cross at `g = k`.
- With `k = 4`: weight on current season is 0.20 at 1 game, 0.50 at 4 games, 0.67 at 8 games, 0.81 at 17 games.

**`k` is not chosen by feel.** Tune it by backtesting past seasons and picking the value that minimizes error. It will likely differ by position — RB workload stabilizes faster than WR touchdown rate.

This also unifies the two projects: the preseason board becomes the prior that the weekly model shrinks away from as the season accumulates.

Shrinkage also solves the small-denominator problem: a player with `g = 1` and 22 points has a PPG of 22.0, better than any full season in history, off one game. Shrinkage pulls that toward the positional mean instead of trusting it.

---

## 5. Filtering: training vs. output

Decided: **drop low-game players from output only, never from training.**

- **Output filtering** is fine — you're declining to rank someone you don't have enough information on. A product decision.
- **Training filtering** is dangerous. Low-game players are exactly the ~31% turnover you need to explain in order to beat 68.8%. Train only on healthy full-season players and the model learns a world where nobody gets hurt.

---

## 6. Data caveats to carry forward

### `season_games` — derive, don't hardcode

`df.groupby('season')['g'].max()` should return the season length per year, and doubles as a data-integrity check. Expect a step from 16 to 17 games when the NFL expanded the regular season in 2021.

Then `games_missed = season_games - g` is a usable availability feature.

### Games ≠ weeks

A 17-game season runs across 18 weeks because of bye weeks. If a weekly loader assumes week number = game number, every player's running average shifts by one somewhere after his bye. In nflverse weekly stats there is no row for a bye week, so **game number must be counted from the rows present, not read off `week`**.

### `g` is ambiguous — the unresolved problem

For any team game, a player is in one of three states:

1. **Did not dress** — inactive, injured, or on IR
2. **Dressed, near-zero snaps** — healthy but buried on the depth chart
3. **Dressed, real snaps** — genuine opportunity

`g` cannot distinguish state 2 from state 3. So `season_games - g` conflates injury, healthy scratches, mid-season signings, and inactive rookies — one number, four causes.

**This cannot be resolved inside `fantasy_top250.csv`**, because the file only contains the top 250 by PPR, so by construction everyone in it produced. A player whose season was all state-2 games never made the cut. The evidence needed was already filtered out.

### The fix: snap counts

nflverse `load_snap_counts()` provides game-level snap counts sourced from Pro Football Reference, **starting with the 2012 season**, including `offense_snaps` and `offense_pct` per player per game.

That replaces "did he play?" with "what share of his team's offense was he on the field for?" — the question that actually matters.

```python
# opportunity — share of the offense he was on the field for
df["snap_share"] = df["offense_snaps"] / df["team_offensive_plays"]

# efficiency — production per unit of opportunity
df["pts_per_snap"] = df["fantasy_points_ppr"] / df["offense_snaps"]
```

Season points ≈ **snap share × efficiency × games available**. Three factors with very different stability:
- **Snap share** — coach-driven, fairly sticky week to week
- **Efficiency** — bounces a lot, especially touchdown rate
- **Availability** — close to noise

Modeling them separately usually beats modeling the product, because each deserves a different amount of regression to the mean.

**Tradeoff:** snap counts start in 2012; the season file goes back to 2000. So it's 26 seasons of a blunt `g` versus ~14 seasons of a sharp `snap_share` (~3,500 player-seasons — thin for a many-featured model, fine for a simple one).

**Middle path:** keep `g` in the season file for the long history; use snaps only in the weekly model, where 2012+ is plenty because each season contributes ~17 rows per player instead of 1.

---

## 7. Agreed next steps

Do the **preseason board first**; leave the weekly model alone for now. The board uses data already on disk, has a number to beat, and every feature it needs (PPG, games, shrinkage) is the same feature the weekly model will need later.

1. **Profile the `g` column.** `df.groupby('season')['g'].describe()` and `df['g'].value_counts().sort_index()`. Check that max `g` steps 16 → 17 in 2021, and count rows with `g < 4`.

2. **Add `ppg`, `season_games`, `games_missed`.** Three columns, no modeling. Save to a new file so the original stays clean.

3. **Build the year-over-year table.** Join each player's season *t* row to his season *t+1* row on player id. Every row then reads: what was known going in, what actually happened. Verify a player appearing in 2019 and 2021 but not 2020 does not accidentally link across the gap.

4. **Race three rankings against the baseline.** For every season 2001–2025, produce a predicted top 250 three ways:
   - carry forward last year's top 250 by total points (reproduces the 68.8%)
   - rank by last year's `ppg`
   - rank by last year's `ppg` × a simple games estimate

   Measure set overlap against the actual top 250 each time. **No machine learning — this is sorting and set intersection.**

5. **Read the result before writing more code.** If PPG-based ranking beats 68.8%, the instinct is confirmed and a model on top of it is justified. If not, weeks of building on a bad feature were avoided.

**Why no ML in that list:** a model can only find signal that already exists in the features. If PPG doesn't beat the baseline when ranked on directly, no amount of gradient boosting rescues it. Sorting a column answers the question faster and more honestly.

---

## 8. Known ceiling: rookies

The 68.8% baseline can only predict players already in last year's top 250, so **rookies are guaranteed misses**. Every year some portion of the top 250 are rookies, which puts a hard floor on how far the carry-forward approach can go.

`draft_2026_offense.csv` and the tier screenshots exist to address this, but that's a separate feature source. Before building for it, quantify it: **once step 3's table exists, count how many of each season's actual top 250 had no prior season at all.** That number is how much of the 31% gap is rookies.

---

## 9. References

- nflreadpy (Python port of nflreadr): https://github.com/nflverse/nflreadpy
  - `load_player_stats()` — weekly or seasonal player stats
  - `load_snap_counts()` — game-level snap counts, 2012+
  - `load_rosters_weekly()` — roster status by season-week
  - `load_injuries()` — injury designations
- nflreadr `load_snap_counts()` reference: https://nflreadr.nflverse.com/reference/load_snap_counts.html
- Snap counts data dictionary: https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html
- nflreadpy docs: https://nflreadpy.nflverse.com

---

## 10. Open questions

- Which is more stable year over year, `ppr/g` or `g`?
- How many rows have `g < 4`, and does that vary by season?
- Does `g` credit a game where a player dressed but took zero snaps? (Unresolvable in the current file — needs snap counts.)
- Are PFR season totals regular-season only, or do they include playoffs? Confirm before any per-game math.
- What fraction of each season's top 250 are rookies?
- What value of `k` minimizes shrinkage error, and does it differ by position?
