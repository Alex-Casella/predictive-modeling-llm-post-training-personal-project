"""Layer 2 -- the draft board.

Takes the projection model's output and turns it into draft value. This layer is
deterministic arithmetic, no model (README.md "Architecture").

    predicted points  ->  VBD  ->  positional tiers  ->  board order

VBD comes from vbd.py with a LeagueConfig passed in, never hardcoded. The same
function that produced the historical `vbd_10` column runs here on PREDICTED
points -- that is exactly why it was written as a function rather than stored
as a column (CLAUDE.md).

WHAT THIS BOARD CANNOT DO -- read before drafting from it:

  Rookies are absent. The candidate pool is season n-1's top 250, and a rookie
  has no prior season, so he cannot appear. Measured over 25 seasons, 23.2% of
  each season's actual top 250 (57.9 players) are players with no prior season
  in this data. This board is blind to roughly one in four players who will
  finish in the money, including every first-round rookie RB and WR.

  Do not read a missing player as a low ranking. He is not ranked at all.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features
from vbd import ESPN_STANDARD, LeagueConfig, add_vbd, startable_counts

SRC = 'fantasy_top250_derived.csv'
PARAMS = dict(max_iter=300, learning_rate=0.06, max_depth=6,
              min_samples_leaf=25, l2_regularization=1.0, random_state=0)

# Tier break = a VBD gap inside a position bigger than this many points.
# Drafting is about tier breaks more than about ranks: the difference between
# RB4 and RB5 matters only if a tier boundary sits between them.
TIER_GAP = 15.0


def project(df, target_season):
    """Predict target_season points for everyone in target_season-1's top 250.

    Uses the M2 variant from step 8 -- regression with censored targets imputed
    at zero, which beat the cutoff imputation at every tier on held-out seasons.
    """
    d, feat_cols = _prepare(df)
    train = d[d['season_next'] <= target_season - 1]
    pool = d[d['season'] == target_season - 1].copy()
    assert len(pool) == 250, f'pool is {len(pool)}, expected 250'
    assert train['season_next'].max() <= target_season - 1, 'look-ahead'

    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(train[feat_cols], train['next_ppr'].fillna(0.0))
    pool['proj_ppr'] = model.predict(pool[feat_cols]).round(1)

    # `pos` was one-hot encoded away for the model; bring the readable version
    # back from the source rows. Joined on pid+season, never on name
    # (CLAUDE.md). `team` survived encoding, so it is not re-merged -- doing so
    # would silently produce team_x / team_y.
    raw = df.loc[df['season'] == target_season - 1, ['pid', 'season', 'pos']]
    pool = pool.merge(raw, on=['pid', 'season'], how='left', validate='1:1')
    assert pool['pos'].notna().all(), 'lost a position in the rejoin'
    return pool


def _prepare(df):
    d, feat_cols, cat_cols = features.build(df)
    d = features.attach_target(d, df)
    d = pd.get_dummies(d, columns=cat_cols, prefix=cat_cols, dtype=float)
    return d, feat_cols + [c for c in d.columns
                           if c.startswith('pos_') and c != 'pos_rk']


def add_tiers(board):
    """Number tiers within each position, breaking where VBD gaps exceed TIER_GAP."""
    board = board.sort_values('vbd', ascending=False).copy()
    board['tier'] = 0
    for pos, grp in board.groupby('pos'):
        tier, prev = 1, None
        for idx in grp.index:
            v = board.at[idx, 'vbd']
            if prev is not None and prev - v > TIER_GAP:
                tier += 1
            board.at[idx, 'tier'] = tier
            prev = v
    return board


def build(target_season=2026, league=None, src=SRC):
    """Return the board: one row per candidate, ordered by VBD."""
    league = league or LeagueConfig(teams=10, **ESPN_STANDARD)
    df = pd.read_csv(src)
    pool = project(df, target_season)

    keep = pool[['pid', 'player', 'pos', 'team', 'age_next', 'proj_ppr',
                 'ppr', 'ppg', 'g', 'rk']].copy()
    keep.columns = ['pid', 'player', 'pos', 'prev_team', 'age', 'proj_ppr',
                    'prev_ppr', 'prev_ppg', 'prev_g', 'prev_rk']

    rows = keep.to_dict('records')
    add_vbd(rows, league, points_key='proj_ppr')
    board = add_tiers(pd.DataFrame(rows))

    board = board.sort_values('vbd_rk').reset_index(drop=True)
    board['pos_rk'] = board.groupby('pos').cumcount() + 1
    board.attrs['league'] = league.label()
    board.attrs['startable'] = startable_counts(rows, league,
                                                points_key='proj_ppr')
    return board


def main():
    league = LeagueConfig(teams=10, **ESPN_STANDARD)
    board = build(2026, league)
    out = 'board_2026.csv'
    board.to_csv(out, index=False)

    print(f'=== 2026 draft board -- {league.label()} ===')
    print(f'projection: gradient-boosted regression trained through 2025')
    print(f'valuation:  vbd.py, league passed as an argument')
    print(f'startable by position: {board.attrs["startable"]}\n')

    cols = ['vbd_rk', 'player', 'pos', 'pos_rk', 'tier', 'age',
            'proj_ppr', 'vbd', 'prev_rk', 'prev_ppr']
    print('--- top 40 ---')
    print(board.head(40)[cols].to_string(index=False))

    print('\n--- tier 1 at each position (the players worth reaching for) ---')
    for pos in ['QB', 'RB', 'WR', 'TE']:
        t1 = board[(board['pos'] == pos) & (board['tier'] == 1)]
        names = ', '.join(f"{r.player} ({r.vbd:+.0f})" for r in t1.itertuples())
        print(f'  {pos}: {names if names else "(none)"}')

    # Two separate effects, kept separate. prev_rk -> vbd_rk mixes them and the
    # combined number is dominated by the second, which makes it read as though
    # the model hates every quarterback when really VBD is doing that.
    #
    #   prev_rk  --[ the model changed its mind ]-->  proj_rk
    #   proj_rk  --[ VBD reweights by scarcity  ]-->  vbd_rk
    board['proj_rk'] = board['proj_ppr'].rank(ascending=False,
                                              method='first').astype(int)
    board['model_move'] = board['prev_rk'] - board['proj_rk']
    board['vbd_move'] = board['proj_rk'] - board['vbd_rk']
    mcols = ['player', 'pos', 'prev_rk', 'proj_rk', 'proj_ppr', 'model_move']
    vcols = ['player', 'pos', 'proj_rk', 'vbd_rk', 'vbd', 'vbd_move']

    print('\n--- effect 1: where the MODEL disagrees with last season ---')
    print('  (points rank -> points rank, so position scarcity is not involved)')
    print('  up:')
    print(board.nlargest(6, 'model_move')[mcols].to_string(index=False))
    print('  down:')
    print(board.nsmallest(6, 'model_move')[mcols].to_string(index=False))

    print('\n--- effect 2: where VBD disagrees with raw points ---')
    print('  (this is scarcity, not opinion -- 10 startable QBs vs 30 WRs)')
    print('  up:')
    print(board.nlargest(6, 'vbd_move')[vcols].to_string(index=False))
    print('  down:')
    print(board.nsmallest(6, 'vbd_move')[vcols].to_string(index=False))
    print('\n  mean vbd_move by position (negative = VBD demotes the position):')
    print('    ' + '   '.join(
        f'{p}:{v:+.0f}' for p, v in
        board.groupby('pos')['vbd_move'].mean().round(0).items()))

    print(f'\nwrote {out}: {len(board)} players')
    print('\n!! ROOKIES ARE ABSENT FROM THIS BOARD !!')
    print('   23.2% of a typical top 250 (57.9 players) have no prior season')
    print('   and therefore cannot be projected from this data. A player who is')
    print('   missing is UNRANKED, not ranked low. See board.py docstring.')
    print('\nData source: Pro-Football-Reference. See ATTRIBUTION.md.')
    return board


if __name__ == '__main__':
    main()
