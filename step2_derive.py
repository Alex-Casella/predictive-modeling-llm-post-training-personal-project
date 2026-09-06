"""Step 2 -- add ppg, season_games, games_missed. Write a new file.

PROJECT_CONTEXT.md §12.2. Three columns, no modelling.

    ppg           = ppr / g          how good he was WHEN HE PLAYED
    season_games  = league schedule  16 (2000-2020) / 17 (2021+), derived
    games_missed  = season_games - g availability

fantasy_top250.csv is never modified (CLAUDE.md hard rule 4).
"""
import pandas as pd

from checks import Checks
from common import NFL_REG_SEASON_LEN, derive_season_games, is_multi_team, load

OUT = 'fantasy_top250_derived.csv'
EXPECTED_ROWS = 6500


def main():
    df = load()
    c = Checks('step 2 -- derived columns')
    before = len(df)

    # -- season_games ------------------------------------------------------
    # Derived (step 1), not hardcoded, then asserted against the known schedule.
    season_games = derive_season_games(df)
    df['season_games'] = df['season'].map(season_games)
    c.check('season_games matches the known NFL schedule',
            (df['season'].map(NFL_REG_SEASON_LEN) == df['season_games']).all())
    c.check('season_games has no blanks', df['season_games'].notna().all())

    # -- ppg ---------------------------------------------------------------
    # Step 1 established g is never 0, so this cannot divide by zero.
    df['ppg'] = (df['ppr'] / df['g']).round(2)
    c.check('ppg has no blanks/inf', df['ppg'].notna().all()
            and (df['ppg'].abs() != float('inf')).all())
    c.check('ppg reconstructs ppr', ((df['ppg'] * df['g'] - df['ppr']).abs()
                                     < 0.5).all())

    # -- games_missed ------------------------------------------------------
    # This goes NEGATIVE for the 3 traded players from step 1 (g > schedule).
    # Not an error -- he really did play more games than his team did. Left as
    # -1 rather than clipped, because clipping would hide it.
    df['games_missed'] = df['season_games'] - df['g']
    neg = df[df['games_missed'] < 0]
    print(f'--- games_missed < 0: {len(neg)} rows (traded players, step 1) ---')
    print(neg[['season', 'player', 'team', 'g', 'season_games', 'games_missed']]
          .to_string(index=False))
    c.check('every negative games_missed is a multi-team row',
            is_multi_team(neg).all() if len(neg) else True)
    c.check('games_missed >= -1 everywhere',
            (df['games_missed'] >= -1).all(),
            f"min {df['games_missed'].min()}")

    # -- row-count discipline (§10) ---------------------------------------
    c.check('row count unchanged', len(df) == before == EXPECTED_ROWS,
            f'{before} -> {len(df)}')
    c.check('no column was overwritten',
            len(df.columns) == len(set(df.columns)))

    # -- what the new columns look like -----------------------------------
    print('\n--- ppg by season ---')
    print(f"{'season':>7} {'sched':>6} {'ppg_min':>8} {'ppg_mean':>9} "
          f"{'ppg_max':>8} {'gm_mean':>8}")
    for s, grp in df.groupby('season'):
        print(f"{s:>7} {grp['season_games'].iloc[0]:>6} {grp['ppg'].min():>8.2f} "
              f"{grp['ppg'].mean():>9.2f} {grp['ppg'].max():>8.2f} "
              f"{grp['games_missed'].mean():>8.2f}")

    print('\n--- ppg by position (all seasons) ---')
    print(df.groupby('pos')['ppg'].describe()[['count', 'mean', '50%', 'max']]
          .round(2).to_string())

    # The §4 caveat, shown rather than asserted: high ppg does not mean top 250
    # rank, because the cut is on the TOTAL.
    print('\n--- §4 caveat: top 10 by ppg in 2025 vs their actual rk ---')
    y = df[df['season'] == 2025].nlargest(10, 'ppg')
    print(y[['rk', 'player', 'pos', 'g', 'ppr', 'ppg']].to_string(index=False))
    print('\n--- and the widest rk/ppg disagreement in 2025 ---')
    y = df[df['season'] == 2025].copy()
    y['ppg_rk'] = y['ppg'].rank(ascending=False, method='min').astype(int)
    y['gap'] = y['ppg_rk'] - y['rk']
    print(y.nsmallest(5, 'gap')[['rk', 'ppg_rk', 'player', 'pos', 'g', 'ppr',
                                 'ppg']].to_string(index=False))

    df.to_csv(OUT, index=False)
    reread = pd.read_csv(OUT)
    c.check('written file re-reads with the same shape',
            reread.shape == df.shape, f'{df.shape} vs {reread.shape}')
    c.check('fantasy_top250.csv untouched',
            len(load()) == EXPECTED_ROWS and 'ppg' not in load().columns)

    c.report()
    print(f'\nwrote {OUT}: {df.shape[0]} rows x {df.shape[1]} columns')
    print('step 2 complete.')
    return df


if __name__ == '__main__':
    main()
