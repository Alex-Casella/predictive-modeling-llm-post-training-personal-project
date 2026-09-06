"""Feature table for the projection model.

One row per (player, season t). Every column is knowable BEFORE season t+1 is
played -- that is CLAUDE.md hard rule 1, and it is the only thing in this file
worth checking carefully.

Three groups:

  season t      what he did last year          ppr, ppg, g, usage, efficiency
  career <= t   what he has done overall       expanding means, seasons played
  lags          where he was going             t-1 and t-2, and the deltas

`age_next` is age-in-t plus one. That is derived from the PRIOR row, not read
off the target row, so it does not leak -- you know a 26-year-old will be 27.

Blanks are NOT filled (CLAUDE.md: "Blanks are NOT zeros"). rush_ypc is blank for
a player with no carries, which means "did not run", not "ran for zero yards".
HistGradientBoosting handles NaN natively by learning a split direction for
missing, so the distinction survives into the model instead of being erased by
an imputation.
"""
import pandas as pd

SEASON_T = [
    'age', 'g', 'gs', 'games_missed', 'ppr', 'ppg', 'rk', 'pos_rk',
    'pro_bowl', 'all_pro', 'total_td',
    'pass_att', 'pass_yds', 'pass_td', 'pass_int',
    'rush_att', 'rush_yds', 'rush_ypc', 'rush_td',
    'tgt', 'rec', 'rec_yds', 'rec_ypr', 'rec_td', 'fmb',
]

CAREER_SRC = ['ppr', 'ppg', 'g']
LAG_SRC = ['ppr', 'ppg', 'g', 'rk']


def build(df):
    """Return a feature frame, one row per player-season, plus keys."""
    d = df.sort_values(['pid', 'season']).copy()

    # -- career-to-date, INCLUSIVE of season t --------------------------
    # Row t carries the mean over seasons <= t. Row t predicts t+1, so nothing
    # from t+1 is inside it.
    g = d.groupby('pid')
    for col in CAREER_SRC:
        d[f'career_mean_{col}'] = g[col].transform(
            lambda s: s.expanding().mean())
        d[f'career_max_{col}'] = g[col].transform(
            lambda s: s.expanding().max())
    d['career_seasons'] = g.cumcount() + 1

    # -- lags: the previous rows IN THE DATA ----------------------------
    # NOTE: a lag is the player's previous *appearance*, which may be two
    # calendar years back if he fell out of the top 250 in between. The gap is
    # exposed explicitly so the model can tell the two situations apart rather
    # than silently treating them as the same.
    for col in LAG_SRC:
        d[f'{col}_lag1'] = g[col].shift(1)
        d[f'{col}_lag2'] = g[col].shift(2)
    d['season_lag1'] = g['season'].shift(1)
    d['gap_since_last'] = d['season'] - d['season_lag1']
    d['ppr_delta1'] = d['ppr'] - d['ppr_lag1']
    d['ppg_delta1'] = d['ppg'] - d['ppg_lag1']

    d['age_next'] = d['age'] + 1

    feat_cols = (SEASON_T + ['age_next', 'career_seasons', 'gap_since_last',
                             'ppr_delta1', 'ppg_delta1']
                 + [f'career_mean_{c}' for c in CAREER_SRC]
                 + [f'career_max_{c}' for c in CAREER_SRC]
                 + [f'{c}_lag{i}' for c in LAG_SRC for i in (1, 2)])
    return d, feat_cols, ['pos']


def attach_target(d, df):
    """Target = ppr in season t+1, plus the censoring threshold for that season.

    A player in season t's top 250 who is absent from t+1's did NOT score zero.
    He scored something below the 250th player's total and the file does not
    record it -- the target is right-censored. `cut_next` is that threshold, so
    the two imputation choices can both be tested rather than one being picked
    silently.
    """
    nxt = df[['pid', 'season', 'ppr']].copy()
    nxt.columns = ['pid', 'season_next', 'next_ppr']
    d = d.assign(season_next=d['season'] + 1).merge(
        nxt, on=['pid', 'season_next'], how='left', indicator=True)
    d['made_next'] = d['_merge'] == 'both'
    d = d.drop(columns='_merge')

    cut = df[df['rk'] == 250].set_index('season')['ppr']
    d['cut_next'] = d['season_next'].map(cut)
    return d
