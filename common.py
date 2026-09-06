"""Shared loading and season-length logic.

The one non-obvious thing in here is `derive_season_games`. See step 1's finding:
`max(g)` per season is NOT season length, because a player traded mid-season can
play MORE games than either of his teams does.
"""
import pandas as pd

SRC = 'fantasy_top250.csv'

# Multi-team aggregate rows. PFR lists a mid-season-traded player once under
# '2TM' / '3TM' carrying his combined stats (README.md "Traded players").
MULTI_TEAM = ('2TM', '3TM')

# External fact, used ONLY to validate the derived value -- never to compute it.
# The NFL regular season was 16 games from 1978 through 2020 and expanded to
# 17 games beginning with the 2021 season.
NFL_REG_SEASON_LEN = {s: (16 if s <= 2020 else 17) for s in range(2000, 2026)}


def load(path=SRC):
    return pd.read_csv(path)


def is_multi_team(df):
    return df['team'].isin(MULTI_TEAM)


def derive_season_games(df):
    """Season length per season, derived from the data.

    Why not `df.groupby('season')['g'].max()` (PROJECT_CONTEXT.md §8a):

        A player traded mid-season plays for two teams. Each team plays a
        16-game schedule across 17 weeks, with a bye. If his old team's bye
        falls AFTER the trade and his new team's bye falls BEFORE it, he dodges
        both and plays 17 games in a 16-game season.

            weeks   1 . . . . . 8 | 9 . . . . . . 17
            team A  P P P P P P P |  (bye wk 12)        -> 8 games
            team B     (bye wk 5) | P P P P P P P P P   -> 9 games
                                                   sum  = 17

        Three such rows exist in this file, all team == '2TM':
        Jerry Rice 2004 (17), Emmanuel Sanders 2019 (17),
        Rashid Shaheed 2025 (18). The naive max is therefore wrong in
        2004, 2019 and 2025.

    Excluding multi-team rows removes exactly that failure mode, because a
    single-team player can never exceed his team's schedule. Verified correct
    in 26/26 seasons against NFL_REG_SEASON_LEN.
    """
    single = df[~is_multi_team(df)]
    return single.groupby('season')['g'].max()
