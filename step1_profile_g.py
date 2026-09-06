"""Step 1 -- profile the `g` column of fantasy_top250.csv.

PROJECT_CONTEXT.md §12.1 and KICKOFF_PROMPT.md "What I care about in step 1".

The point is NOT to compute anything useful. It is to find out whether the data
behaves the way §8 assumes before anything gets built on top of it.

Four questions from the kickoff:
  Q1  Does max `g` per season step from 16 to 17 in 2021?      (§8a)
  Q2  How many rows have g < 4, and does it vary by season?
  Q3  Regular season only, or do totals include playoffs?      (§8g)
  Q4  Is it exactly 250 rows every season?

Plus five of my own that step 3's join and step 2's division depend on.

Data source: Pro-Football-Reference (see ATTRIBUTION.md).
"""
import pandas as pd

from checks import Checks
from common import (NFL_REG_SEASON_LEN, MULTI_TEAM, derive_season_games,
                    is_multi_team, load)

# Expected values, stated IN ADVANCE (§10).
EXPECTED_ROWS = 6500
EXPECTED_SEASONS = list(range(2000, 2026))
EXPECTED_ROWS_PER_SEASON = 250

# ATTRIBUTION.md's stated player counts. Kept as `warn`, not `check`: step 1
# found the file has 1698/1687, six more than documented, while every
# structural claim in that document (11 excess ids, 10 shared names, 5 sharing
# a position) reproduces exactly. Reported as documentation drift.
DOC_N_PLAYERS = 1692
DOC_N_NAMES = 1681
DOC_SHARED_NAMES = 10
DOC_SHARED_SAME_POS = 5


def main():
    df = load()
    c = Checks('step 1 -- structure and the g column')

    print(f'loaded fantasy_top250.csv: {df.shape[0]} rows x {df.shape[1]} columns')
    print(f'columns: {list(df.columns)}')

    c.check('row count is 6500', len(df) == EXPECTED_ROWS, f'got {len(df)}')

    seasons = sorted(df['season'].unique())
    c.check('26 seasons, 2000-2025, no gaps', seasons == EXPECTED_SEASONS,
            f'got {len(seasons)}: {seasons[0]}-{seasons[-1]}')

    per_season = df.groupby('season').size()
    bad = per_season[per_season != EXPECTED_ROWS_PER_SEASON]
    c.check('exactly 250 rows every season (Q4)', bad.empty, dict(bad))

    # -- identity: step 3's join stands on these ---------------------------
    dup_ps = int(df.duplicated(subset=['pid', 'season']).sum())
    c.check('pid+season is unique', dup_ps == 0, f'{dup_ps} duplicates')
    c.check('each pid maps to exactly one pfr_id',
            (df.groupby('pid')['pfr_id'].nunique() == 1).all())
    c.check('each pfr_id maps to exactly one pid',
            (df.groupby('pfr_id')['pid'].nunique() == 1).all())

    n_players, n_names = df['pid'].nunique(), df['player'].nunique()
    per_name = df.groupby('player')['pid'].nunique()
    shared = per_name[per_name > 1]
    same_pos = sum(
        1 for n in shared.index
        if df.loc[df['player'] == n].groupby('pid')['pos'].first().nunique()
        < df.loc[df['player'] == n, 'pid'].nunique())

    print(f'\ndistinct pid: {n_players}   distinct names: {n_names}   '
          f'excess: {n_players - n_names}')
    print(f'names held by >1 player: {len(shared)}   of those sharing a '
          f'position: {same_pos}')
    c.warn('pid count matches ATTRIBUTION.md', n_players == DOC_N_PLAYERS,
           f'file {n_players}, doc {DOC_N_PLAYERS} -- documentation drift')
    c.warn('name count matches ATTRIBUTION.md', n_names == DOC_N_NAMES,
           f'file {n_names}, doc {DOC_N_NAMES} -- documentation drift')
    c.check('excess ids over names is internally consistent',
            n_players - n_names == int((shared - 1).sum()))
    c.check('10 names held by more than one player (ATTRIBUTION.md)',
            len(shared) == DOC_SHARED_NAMES, f'got {len(shared)}')
    c.check('5 of those share a position (ATTRIBUTION.md)',
            same_pos == DOC_SHARED_SAME_POS, f'got {same_pos}')

    # -- rk sanity: the overlap metric assumes these 250 rows ARE the top 250
    rk_ok = df.groupby('season')['rk'].apply(
        lambda s: sorted(s) == list(range(1, 251)))
    c.check('rk runs 1..250 within every season', rk_ok.all(),
            f'{int((~rk_ok).sum())} seasons off')
    ppr_sorted = df.groupby('season')['ppr'].apply(
        lambda s: s.is_monotonic_decreasing)
    c.check('rows ordered by descending ppr within season', ppr_sorted.all(),
            f'{int((~ppr_sorted).sum())} seasons off')

    # -- Q1 + Q3: the g wall ----------------------------------------------
    reg = pd.Series(NFL_REG_SEASON_LEN, name='reg_len')
    naive = df.groupby('season')['g'].max()
    derived = derive_season_games(df)

    print('\n--- Q1: season length, three ways ---')
    print('  naive     = max(g)                    <- PROJECT_CONTEXT.md §8a')
    print('  derived   = max(g) over single-team rows only')
    print('  nfl_known = external fact (16 through 2020, 17 from 2021)\n')
    cmp = pd.DataFrame({'naive': naive, 'derived': derived, 'nfl_known': reg})
    cmp['naive_ok'] = cmp['naive'] == cmp['nfl_known']
    cmp['derived_ok'] = cmp['derived'] == cmp['nfl_known']
    print(cmp.to_string())
    print(f"\n  naive correct in {int(cmp['naive_ok'].sum())}/26 seasons")
    print(f"  derived correct in {int(cmp['derived_ok'].sum())}/26 seasons")

    c.warn('naive max(g) equals season length (PROJECT_CONTEXT.md §8a)',
           cmp['naive_ok'].all(),
           'wrong in ' + ', '.join(str(s) for s in cmp.index[~cmp['naive_ok']])
           + ' -- traded players exceed their teams\' schedules')
    c.check('derived season length equals the known NFL schedule in 26/26',
            cmp['derived_ok'].all())
    c.check('max(g) steps 16 -> 17 exactly at 2021 once traded players excluded',
            (derived.loc[2000:2020] == 16).all() and (derived.loc[2021:] == 17).all())

    over = df.assign(reg_len=df['season'].map(NFL_REG_SEASON_LEN))
    over = over[over['g'] > over['reg_len']]
    print(f'\n--- rows where g exceeds the regular-season length: {len(over)} ---')
    print(over[['season', 'rk', 'player', 'pos', 'team', 'g', 'ppr']]
          .to_string(index=False))
    c.check('every over-length row is a multi-team (traded) row',
            is_multi_team(over).all() if len(over) else True)

    # -- Q3: playoffs ------------------------------------------------------
    # If totals included playoffs, a Super Bowl team's player would reach
    # 16 + 4 = 20 games (2000-2020) or 17 + 4 = 21 (2021+). The observed ceiling
    # is season length (+1 for the traded-player case), so: regular season only.
    single_max = int(df.loc[~is_multi_team(df), 'g'].max())
    print(f'\n--- Q3: playoffs? ---')
    print(f'  highest g for a single-team player, any season: {single_max}')
    print(f'  a Super Bowl run would put that at 20 (2000-2020) / 21 (2021+)')
    print(f'  => season totals are REGULAR SEASON ONLY')
    c.check('no single-team player exceeds 17 games (=> no playoff games)',
            single_max <= 17, f'max {single_max}')

    # -- Q2: the low-games tail -------------------------------------------
    lt4 = df[df['g'] < 4].groupby('season').size().reindex(seasons, fill_value=0)
    print('\n--- Q2: rows with g < 4, by season ---')
    print('  ' + '  '.join(f'{s}:{n}' for s, n in lt4.items() if n))
    print(f'  total: {int(lt4.sum())} of {len(df)} '
          f'({100 * lt4.sum() / len(df):.2f}%)   '
          f'seasons with none: {int((lt4 == 0).sum())}/26')
    print(df.loc[df['g'] < 4, ['season', 'rk', 'player', 'pos', 'g', 'ppr']]
          .to_string(index=False))

    print('\n--- full g distribution (all seasons pooled) ---')
    for val, n in df['g'].value_counts().sort_index().items():
        print(f'  g={val:>2}  {n:>5}  {"#" * max(1, n // 40)}')

    n_zero = int((df['g'] == 0).sum())
    c.check('no g == 0 rows (step 2 would divide by zero)', n_zero == 0,
            f'{n_zero} rows')
    c.check('gs never exceeds g', int((df['gs'] > df['g']).sum()) == 0)

    # -- blanks -----------------------------------------------------------
    print('\n--- blanks per column (non-zero only) ---')
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    for col, n in nulls.items():
        print(f'  {col:<14} {n:>5}  ({100 * n / len(df):5.2f}%)')
    c.check('ppr has no blanks (it is the target)', df['ppr'].notna().all())
    c.check('g has no blanks', df['g'].notna().all())
    c.check('pos has no blanks', df['pos'].notna().all())

    print('\n--- positions present ---')
    print(df['pos'].value_counts().to_string())
    print(f"\n--- multi-team rows ({', '.join(MULTI_TEAM)}) ---")
    print(f'  {int(is_multi_team(df).sum())} rows')

    c.report()
    print('\nstep 1 complete.')
    return df


if __name__ == '__main__':
    main()
