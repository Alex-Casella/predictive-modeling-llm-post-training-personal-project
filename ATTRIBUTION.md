# Data attribution

## Pro-Football-Reference / Sports Reference LLC

The season-level fantasy statistics in `fantasy_top250.csv` were exported from
Pro-Football-Reference (https://www.pro-football-reference.com), a Sports
Reference LLC site.

Any additional NFL data added to this project (draft results, team rosters)
comes from the same source unless noted otherwise.

The `pfr_id` column contains Pro-Football-Reference player identifiers.

**Please credit Pro-Football-Reference in any output, visualization, or write-up
derived from this data.**

Sports Reference publishes data use terms here:
https://www.sports-reference.com/data_use.html

## nflverse

The per-player, per-week statistics in `weekly_ppr.csv` come from the
**nflverse** project (https://github.com/nflverse), retrieved through the
`nfl_data_py` Python package (https://github.com/nflverse/nfl_data_py). The
`fantasy_points_ppr` column is nflverse's own computation and is used as
published rather than reimplemented.

nflverse also supplies the `gsis_id` ↔ `pfr_id` crosswalk (`import_ids()`) that
lets weekly rows join to this project's `pid` without ever matching on a name.

**Please credit nflverse alongside Pro-Football-Reference in any output built
on the weekly data.** nflverse data is released under CC BY 4.0; see
https://github.com/nflverse/nflverse-data for current terms.

That the two sources agree is itself documented: summing weekly PPR reproduces
the season `ppr` column from Pro-Football-Reference to a median difference of
+0.00 (`step11_weekly_data.py`).

## THE MIT LICENSE COVERS THE CODE, NOT THE DATA

`LICENSE.txt` grants MIT terms over **the software in this repository** — the
scripts, the harnesses, the documentation this project wrote.

It does **not** and cannot grant any rights over the NFL statistics, which this
project does not own. `fantasy_top250.csv`, `fantasy_top250_derived.csv`,
`weekly_ppr.csv`, `board_2026.csv` and every file derived from them carry the
terms of their original sources, above. Anyone reusing this repository is bound
by those terms for the data regardless of what the MIT license says about the
code.

## A note on licensing

Attribution is not a license. Crediting the source is good practice and the
right thing to do, but it does not by itself establish that a given use is
permitted. Before publishing, distributing, or making any commercial use of
work built on this data, read the terms above and get proper advice.

Nothing in this file is legal advice.

## Identifiers used in this project

| Column | Origin | Purpose |
|---|---|---|
| `pid` | This project | Primary key. Format `P0001`. Stable, ours to control. |
| `pfr_id` | Pro-Football-Reference | Foreign key. Used to join additional PFR data. |

`pid` is assigned deterministically: players are ordered by the first season
they appear in the dataset, then by their best rank in that season, then by
`pfr_id` as a tiebreak, and numbered sequentially from `P0001`. Re-running the
build script on the same input reproduces the same assignments. Players added
later (e.g. a 2026 season) receive new numbers appended to the end without
disturbing existing ones.

`pfr_id` is retained rather than discarded because it is the only key that
distinguishes different players who share a name. This dataset contains 1,692
distinct players but only 1,681 distinct names — ten names belong to two
different people, including two Adrian Petersons and two Ricky Williamses, five
of which share a position. Deriving identity from names alone would silently
merge those careers.
