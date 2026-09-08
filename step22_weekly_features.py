"""Stage 10 step 3 -- weekly-shape features, because the model was not the problem.

step21's conclusion, pre-registered as its failure case and reached anyway:
prior-season box-score totals barely separate a player who stays top-25 from one
who falls to 26-100. A gradient-boosted classifier with a tier-matched target
captured 2% of the 49 available points; `ppg x games`, two columns of
arithmetic, captured 4%. The lever is features, not models.

WHAT THE SEASON FILE CANNOT SAY

`fantasy_top250.csv` records a season as one row of totals. Two players with
240 PPR over 15 games are identical in it, and completely different in reality:

    A   16, 16, 15, 16, 15, 16, ...        steady
    B    2,  3,  4, 38, 41, 44, ...        hurt, then a different player
    C   44, 41, 38,  4,  3,  2, ...        the reverse, and a trap

`weekly_ppr.csv` distinguishes them, and every one of these is computed from
season t alone -- so it is legal for predicting t+1 under hard rule 1.

    consistency     sd and coefficient of variation of weekly points
    floor/ceiling   worst and best week, share of weeks above a threshold
    availability    weeks missed, and whether they were early or late
    trajectory      second half minus first half, and the last four weeks

`trajectory` is the one to watch. step12 already measured that chasing recent
form LOSES on weekly decisions (last-3 at 2.91 against season-to-date at 2.71).
That was about picking a player for next WEEK. Whether an end-of-season surge
predicts next SEASON is a different question, and it is not answered by that
result.

COVERAGE, AND WHY THE SPAN SHRINKS ANYWAY

`weekly_ppr.csv` is usable from 2010 (12% coverage in 2000, 99% from 2010 --
see CLAUDE.md). Rows before that are left NaN rather than imputed, since
HistGradientBoosting learns a split direction for missing exactly as
`rush_ypc` is handled in features.py.

That is not enough on its own. Predicting 2006 trains on outcome seasons up to
2005, where every weekly feature is missing for every row, and sklearn's binner
cannot fit an all-NaN column at all -- it raises "window shape cannot be larger
than input array shape". So the walk-forward starts at 2014, by which point the
training window holds 2010-2012 with real weekly data.

**Carry-forward and the totals-only model are re-scored on THAT SAME SPAN
inside this file.** step21's numbers cover 2006-2025 and must not be compared
to these; that mistake has already been made three times today.

PREDICTION, BEFORE RUNNING

Small and positive at @25, smaller than the 49-point headroom makes it sound.
Availability and consistency are real information the totals genuinely lack.
But the elite tier turns over because of things no box score records -- a
coaching change, a rookie taking snaps, a contract year -- and none of that is
in here either.

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import numpy as np
import pandas as pd

from checks import Checks
from evaluate import carry_forward, evaluate, summarise
from step21_elite_ranker import (PARAMS, SRC, make_order_fn,
                                 prepare as base_prepare)

WEEKLY = 'weekly_ppr.csv'
OUT = 'step22_results.csv'
WEEKLY_FROM = 2010          # CLAUDE.md: 99% coverage from here, 12% in 2000
# Starts at 2014 so the training window (outcome seasons <= 2013) contains
# real weekly rows. Earlier targets train on all-NaN weekly columns, which
# sklearn's binner cannot fit. EVERY method below is scored on this span.
TARGETS = list(range(2014, 2026))
STARTABLE = 10.0            # PPR points that make a week worth having started


def weekly_features(weekly):
    """One row per (pid, season). Everything here is knowable at season end."""
    w = weekly[weekly['season'] >= WEEKLY_FROM].copy()
    g = w.sort_values(['pid', 'season', 'week']).groupby(['pid', 'season'])

    f = g['ppr'].agg(w_games='size', w_mean='mean', w_sd='std',
                     w_max='max', w_min='min', w_median='median').reset_index()
    # Coefficient of variation: spread relative to level, so a 15-point player
    # and a 5-point player are comparable. Guarded against a zero mean.
    f['w_cv'] = f['w_sd'] / f['w_mean'].replace(0, np.nan)
    f['w_startable_pct'] = (g['ppr'].apply(lambda s: (s >= STARTABLE).mean())
                            .reset_index(drop=True))

    # Trajectory. Split on the player's own median week, not week 9, so a
    # player who missed the first month is not scored as all-second-half.
    def halves(s):
        mid = len(s) // 2
        return pd.Series({'w_first': s.iloc[:mid].mean() if mid else np.nan,
                          'w_second': s.iloc[mid:].mean(),
                          'w_last4': s.iloc[-4:].mean()})
    h = g['ppr'].apply(halves).unstack().reset_index()
    f = f.merge(h, on=['pid', 'season'], how='left')
    f['w_trend'] = f['w_second'] - f['w_first']

    # Availability: weeks between his first and last appearance that he missed.
    # Counting from week 1 would charge a player for a late-season debut.
    span = g['week'].agg(['min', 'max']).reset_index()
    f = f.merge(span, on=['pid', 'season'], how='left')
    f['w_missed_within'] = (f['max'] - f['min'] + 1) - f['w_games']
    return f.drop(columns=['min', 'max'])


def main():
    df = pd.read_csv(SRC)
    weekly = pd.read_csv(WEEKLY)
    c = Checks('stage 10 step 3 -- weekly-shape features')

    wf = weekly_features(weekly)
    new_cols = [k for k in wf.columns if k.startswith('w_')]
    print(f'{len(wf):,} player-seasons of weekly shape, '
          f'{wf["season"].min()}-{wf["season"].max()}, {len(new_cols)} features')

    d, feat_cols = base_prepare(df)
    before = len(d)
    d = d.merge(wf, on=['pid', 'season'], how='left')
    c.check('the merge added no rows', len(d) == before,
            f'{before} -> {len(d)}; a duplicate (pid, season) in the weekly '
            f'aggregate would double-count a player')

    cov = d.groupby('season')['w_mean'].apply(lambda s: 100 * s.notna().mean())
    print(f'\n--- coverage of the new features, by season ---')
    print('  ' + '  '.join(f'{y}:{v:.0f}%' for y, v in cov.items()
                           if y in (2000, 2005, 2009, 2010, 2015, 2024, 2025)))
    print(f'  Left NaN before {WEEKLY_FROM} rather than imputed; '
          f'HistGradientBoosting\n  learns a split for missing.')
    # The last season in the season file has no weekly rows -- nfl_data_py
    # 0.3.3 does not publish 2025 (CLAUDE.md) -- and it also has no FOLLOWING
    # season, so those rows are never trained on and never form a test pool.
    # Excluding it here is not a convenience: including it would fail a check
    # on data the experiment does not use.
    last = int(df['season'].max())
    usable = cov[(cov.index >= WEEKLY_FROM) & (cov.index < last)]
    print(f'  Season {last} has no weekly rows and no following season, so it '
          f'is\n  neither trained on nor predicted from. Excluded from the '
          f'check below.')
    c.check(f'coverage is essentially complete {WEEKLY_FROM}-{last - 1}',
            usable.min() > 90, f'min {usable.min():.0f}% in '
            f'{int(usable.idxmin())}')

    # -- do these features relate to staying elite at all? -----------------
    # Asked before modelling, per the step16 rule.
    elite = d[(d['rk'] <= 50) & d['w_mean'].notna()]
    print(f'\n--- among players who finished top 50, do these separate the '
          f'ones\n    who stay top 25 next season? ({len(elite)} rows) ---')
    print(f'{"feature":<18} {"stayed":>9} {"did not":>9} {"diff":>8}')
    for col in new_cols:
        a = elite.loc[elite['made_next_25'], col].mean()
        b = elite.loc[~elite['made_next_25'], col].mean()
        print(f'{col:<18} {a:>9.2f} {b:>9.2f} {a - b:>+8.2f}')

    # -- run ----------------------------------------------------------------
    with_w = feat_cols + new_cols
    runs = [
        ('A_carry_forward', carry_forward),
        ('E1 P(top25), totals only', make_order_fn(d, feat_cols,
                                                   'made_next_25')),
        ('W1 P(top25) + weekly shape', make_order_fn(d, with_w,
                                                     'made_next_25')),
    ]
    res = pd.concat([evaluate(df, TARGETS, fn, label=lbl) for lbl, fn in runs])
    t = summarise(res).loc[[lbl for lbl, _ in runs]]
    print(f'\n--- set overlap, {len(TARGETS)} walk-forward seasons '
          f'({TARGETS[0]}-{TARGETS[-1]}) ---')
    print('  NOT comparable to step21, which covers 2006-2025.')
    print(t.to_string())
    print('\n--- delta vs carry-forward ---')
    print((t - t.loc['A_carry_forward']).drop(index='A_carry_forward')
          .round(1).to_string())

    gain = t.loc['W1 P(top25) + weekly shape', 25] - t.loc['E1 P(top25), totals only', 25]

    # The arithmetic sort that beat the model in step21, recomputed on THIS
    # span. step21 learned this the hard way: its published 43.5% covered
    # 2001-2025 and the comparison was invalid until it was re-derived.
    s4 = pd.read_csv('step4_results.csv')
    s4 = s4[s4['target'].between(TARGETS[0], TARGETS[-1])]
    simple = (s4.pivot_table(index='method', columns='tier', values='overlap')
              [25].drop('A_carry_forward'))

    cf = t.loc['A_carry_forward', 25]
    print(f'\n--- reading it, at @25 ({TARGETS[0]}-{TARGETS[-1]}) ---')
    print(f'  carry-forward                {cf:.1f}%')
    for m, v in simple.sort_values(ascending=False).items():
        print(f'  {m:<28} {v:.1f}%   ({v - cf:+.1f})   <- arithmetic sort')
    print(f'  E1 totals only               '
          f'{t.loc["E1 P(top25), totals only", 25]:.1f}%   '
          f'({t.loc["E1 P(top25), totals only", 25] - cf:+.1f})')
    print(f'  W1 + weekly shape            '
          f'{t.loc["W1 P(top25) + weekly shape", 25]:.1f}%   '
          f'({t.loc["W1 P(top25) + weekly shape", 25] - cf:+.1f})')
    print(f'  ORACLE ceiling               91.0%   (+49.0 available)')
    print(f'\n  weekly shape added {gain:+.1f} over the same model on totals '
          f'alone,')
    print(f'  and the pair captures '
          f'{100 * (t.loc["W1 P(top25) + weekly shape", 25] - cf) / 49:.0f}% of '
          f'the 49 points, against '
          f'{100 * (simple.max() - cf) / 49:.0f}% for the best arithmetic sort.')
    c.warn('the weekly features helped at @25', gain > 0,
           f'{gain:+.1f} -- pre-registered as possible; the totals already '
           f'carry most of what a box score can say')

    res.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    c.report()


if __name__ == '__main__':
    main()
