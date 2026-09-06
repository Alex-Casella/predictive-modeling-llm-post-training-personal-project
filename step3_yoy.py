"""Step 3 -- build the year-over-year table.

PROJECT_CONTEXT.md §12.3. Join each player's season t row to his season t+1 row
on pid, so every row reads: what was known going in, what actually happened.

The trap the kickoff calls out: a player who appears in 2019 and 2021 but not
2020 must NOT link 2019 -> 2021. Guarded by joining on an explicit
`season + 1` key and then asserting the gap is exactly 1 on every matched row.

Also answers two open questions the table makes cheap:
  §4/§10  which is more stable year over year, ppg or g?
  §13     what fraction of each season's top 250 had no prior season at all?
"""
import pandas as pd

from checks import Checks

SRC = 'fantasy_top250_derived.csv'
OUT = 'yoy.csv'

# Stated in advance (§10): one row per player-season EXCEPT the final season,
# which has no t+1 to look at. 6500 - 250 = 6250.
EXPECTED_PAIR_ROWS = 6250

CARRY = ['pid', 'season', 'rk', 'player', 'pos', 'team', 'age', 'g', 'gs',
         'ppr', 'ppg', 'season_games', 'games_missed']


def main():
    df = pd.read_csv(SRC)
    c = Checks('step 3 -- year-over-year join')
    last_season = df['season'].max()

    left = df.loc[df['season'] < last_season, CARRY].copy()
    left['join_season'] = left['season'] + 1
    c.check('left side excludes the final season', len(left) == EXPECTED_PAIR_ROWS,
            f'got {len(left)}')

    right = df[['pid', 'season', 'rk', 'g', 'ppr', 'ppg', 'games_missed']].copy()
    right.columns = ['pid', 'season', 'next_rk', 'next_g', 'next_ppr',
                     'next_ppg', 'next_games_missed']

    # NEVER a bare how='left' (§9).
    yoy = left.merge(right, left_on=['pid', 'join_season'],
                     right_on=['pid', 'season'], how='left',
                     suffixes=('', '_r'), indicator=True)

    print('--- merge indicator ---')
    print(yoy['_merge'].value_counts().to_string())
    c.check('merge added no rows (join key is unique on the right)',
            len(yoy) == EXPECTED_PAIR_ROWS, f'{EXPECTED_PAIR_ROWS} -> {len(yoy)}')
    c.check('no right_only rows (left join cannot produce them)',
            (yoy['_merge'] == 'right_only').sum() == 0)

    yoy['made_next'] = yoy['_merge'] == 'both'

    # -- THE GAP GUARD -----------------------------------------------------
    matched = yoy[yoy['made_next']]
    gap = matched['season_r'] - matched['season']
    c.check('every matched pair is exactly one season apart',
            (gap == 1).all(), f'gaps seen: {sorted(gap.unique())}')

    # Show it working on a real player with a hole in his record.
    seasons_by_pid = df.groupby('pid')['season'].apply(set)
    holed = [p for p, s in seasons_by_pid.items()
             if 2019 in s and 2021 in s and 2020 not in s]
    print(f'\n--- gap guard: players in 2019 and 2021 but not 2020: {len(holed)} ---')
    if holed:
        demo = holed[0]
        print(df.loc[df['pid'] == demo,
                     ['pid', 'season', 'rk', 'player', 'pos', 'g', 'ppr']]
              .to_string(index=False))
        print('\nhis rows in the yoy table (2019 must NOT reach 2021):')
        print(yoy.loc[yoy['pid'] == demo,
                      ['season', 'rk', 'player', 'made_next', 'season_r',
                       'next_rk']].to_string(index=False))

    yoy = yoy.drop(columns=['_merge', 'season_r', 'join_season'])

    # -- §13: first appearances -------------------------------------------
    # NOTE: this is "no prior season in THIS FILE", which is an UPPER BOUND on
    # true rookies. The file only holds the top 250, so a fourth-year player
    # breaking out for the first time looks identical to a rookie here.
    # Separating the two needs external draft data (README.md "Known gaps").
    first = df.groupby('pid')['season'].min().rename('first_season')
    df = df.merge(first, on='pid', how='left')
    newcomers = (df[df['season'] > df['season'].min()]
                 .assign(is_new=lambda d: d['season'] == d['first_season'])
                 .groupby('season')['is_new'].agg(['sum', 'mean']))
    newcomers.columns = ['n_new', 'pct_new']
    print('\n--- §13: players with no prior season in this file ---')
    print('    (upper bound on rookies -- see note in source)')
    print((newcomers.assign(pct_new=(100 * newcomers['pct_new']).round(1))
           ).to_string())
    print(f"\n  mean per season 2001-2025: {newcomers['n_new'].mean():.1f} "
          f"of 250 ({100 * newcomers['pct_new'].mean():.1f}%)")
    print(f"  the carry-forward baseline misses ALL of these by construction,")
    print(f"  so its ceiling is about {100 - 100 * newcomers['pct_new'].mean():.1f}%")

    # -- §4/§10: which half of ppg x g is more stable? ---------------------
    print('\n--- §4/§10: year-over-year stability, ppg vs g ---')
    m = yoy[yoy['made_next']]
    rows = []
    for label, a, b in [('ppg', 'ppg', 'next_ppg'), ('g', 'g', 'next_g'),
                        ('ppr (total)', 'ppr', 'next_ppr')]:
        rows.append(dict(feature=label,
                         pearson_r=round(m[a].corr(m[b]), 3),
                         r2=round(m[a].corr(m[b]) ** 2, 3),
                         spearman=round(m[a].corr(m[b], method='spearman'), 3)))
    print(pd.DataFrame(rows).to_string(index=False))
    print('\n  by position:')
    per_pos = []
    for pos, grp in m.groupby('pos'):
        if len(grp) < 50:
            continue
        per_pos.append(dict(pos=pos, n=len(grp),
                            ppg_r=round(grp['ppg'].corr(grp['next_ppg']), 3),
                            g_r=round(grp['g'].corr(grp['next_g']), 3),
                            ppr_r=round(grp['ppr'].corr(grp['next_ppr']), 3)))
    print(pd.DataFrame(per_pos).to_string(index=False))

    print(f'\n--- survival: made the top 250 again the next season ---')
    surv = yoy.groupby('season')['made_next'].mean().mul(100).round(1)
    print('  ' + '  '.join(f'{s}:{v}' for s, v in surv.items()))
    print(f'  mean {surv.mean():.1f}%')

    yoy.to_csv(OUT, index=False)
    c.check('written file re-reads with same shape',
            pd.read_csv(OUT).shape == yoy.shape)
    c.report()
    print(f'\nwrote {OUT}: {yoy.shape[0]} rows x {yoy.shape[1]} columns')
    print('step 3 complete.')
    return yoy


if __name__ == '__main__':
    main()
