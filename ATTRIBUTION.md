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
