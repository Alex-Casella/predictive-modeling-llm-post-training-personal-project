"""Step 4 -- race three rankings against the 68.8% carry-forward baseline.

PROJECT_CONTEXT.md §12.4. No machine learning: this is sorting and set
intersection.

For every target season N in 2001..2025, build a predicted ordering using ONLY
seasons <= N-1 (CLAUDE.md hard rule 1), then measure set overlap at six tiers.

    A  carry-forward   order last season's 250 by last season's ppr  <- baseline
    B  ppg             order last season's 250 by last season's ppg
    C  ppg x games     order by last season's ppg x career-to-date mean g

Why career-to-date mean g for C, and not last season's g:

    ppg x g  =  (ppr / g) x g  =  ppr

    Using the SAME season's g makes C algebraically identical to A. The games
    estimate has to come from somewhere else or the third race is not a third
    race. Career-to-date mean g is the cheapest leakage-free alternative: it
    averages every season the player has had up to and including N-1.

Metric (CLAUDE.md "Success metric"): at tier T, take the top T of the predicted
ordering, intersect with the players who actually finished in season N's top T,
divide by T.
"""
import pandas as pd

from checks import Checks
from evaluate import TIERS, order_by

SRC = 'fantasy_top250_derived.csv'
OUT = 'step4_results.csv'

# CLAUDE.md's stated baseline, to be reproduced (README.md next-step 1).
DOC_BASELINE = {25: 42.1, 50: 51.2, 100: 61.4, 150: 65.1, 200: 67.6, 250: 68.8}


def add_career_games(df):
    """Career-to-date mean games, INCLUSIVE of the current season.

    Row for season N-1 carries the mean of every season through N-1. That row is
    then used to predict season N, so nothing from N reaches it.

        season   g    career_mean_g (inclusive)
          2018   16   16.00
          2019   12   14.00     <- (16+12)/2
          2020   16   14.67     <- (16+12+16)/3   used to predict 2021
    """
    df = df.sort_values(['pid', 'season']).copy()
    df['career_mean_g'] = (df.groupby('pid')['g']
                           .transform(lambda s: s.expanding().mean()))
    df['career_seasons'] = df.groupby('pid').cumcount() + 1
    return df


def overlap(predicted_order, actual_top, tier):
    """Share of the predicted top `tier` that really finished in the top `tier`."""
    pred = set(predicted_order[:tier])
    assert len(pred) == tier, f'ordering too short: {len(pred)} < {tier}'
    return len(pred & actual_top[tier]) / tier


def main():
    df = add_career_games(pd.read_csv(SRC))
    c = Checks('step 4 -- ranking race')
    seasons = sorted(df['season'].unique())
    targets = seasons[1:]                      # 2001..2025, 25 pairs
    c.check('25 target seasons', len(targets) == 25, f'got {len(targets)}')

    # rk is the final tiebreak on every ordering -- exact ppr ties exist and some
    # straddle a tier boundary. See evaluate.TIEBREAK.
    methods = {
        'A_carry_forward': lambda p: order_by(p, ['ppr'], [False]),
        'B_ppg':           lambda p: order_by(p, ['ppg'], [False]),
        'C_ppg_x_games':   lambda p: order_by(
            p.assign(score=lambda d: d['ppg'] * d['career_mean_g']),
            ['score'], [False]),
    }

    rows = []
    for target in targets:
        prior = df[df['season'] == target - 1]
        this = df[df['season'] == target]
        assert len(prior) == 250 and len(this) == 250

        actual_top = {t: set(this.loc[this['rk'] <= t, 'pid']) for t in TIERS}
        for t in TIERS:
            assert len(actual_top[t]) == t

        for name, order_fn in methods.items():
            order = order_fn(prior)
            for t in TIERS:
                rows.append(dict(target=target, method=name, tier=t,
                                 overlap=100 * overlap(order, actual_top, t)))

    res = pd.DataFrame(rows)
    pivot = (res.pivot_table(index='method', columns='tier', values='overlap')
             .round(1))

    print('--- mean set overlap %, 25 target seasons (2001-2025) ---\n')
    print(pivot.to_string())
    print('\n--- vs the baseline recorded in CLAUDE.md ---')
    doc = pd.Series(DOC_BASELINE, name='CLAUDE.md')
    comp = pd.DataFrame({'measured_A': pivot.loc['A_carry_forward'],
                         'CLAUDE.md': doc})
    comp['diff'] = (comp['measured_A'] - comp['CLAUDE.md']).round(1)
    print(comp.to_string())

    for t in TIERS:
        c.check(f'carry-forward reproduces CLAUDE.md at tier {t}',
                abs(pivot.loc['A_carry_forward', t] - DOC_BASELINE[t]) < 0.1,
                f"measured {pivot.loc['A_carry_forward', t]}, "
                f"doc {DOC_BASELINE[t]}")

    print('\n--- deltas vs carry-forward (positive = better) ---')
    delta = (pivot - pivot.loc['A_carry_forward']).round(1)
    print(delta.drop(index='A_carry_forward').to_string())

    # -- the @250 degeneracy ----------------------------------------------
    # All three methods rank the SAME 250 candidates -- last season's top 250.
    # Taking the top 250 of a 250-long ordering returns the whole pool, so the
    # headline tier cannot distinguish them. Proven, not assumed:
    at250 = pivot[250]
    c.check('all three methods are identical at tier 250',
            at250.nunique() == 1, dict(at250.round(3)))
    print(f'\n--- why tier 250 is identical for all three ---')
    print(f'  candidate pool  = last season\'s top 250 = 250 players')
    print(f'  predicted top 250 of 250 candidates = the whole pool')
    print(f'  => order is irrelevant at tier 250. Measured: '
          f'{sorted(at250.unique().round(3))}')
    print(f'  The headline metric can only move by CHANGING THE POOL,')
    print(f'  not by reordering it. Tiers 25-200 are where these three differ.')

    # -- per-season detail at the two tiers CLAUDE.md says to watch --------
    for t in (25, 50):
        print(f'\n--- per-season overlap at tier {t} ---')
        p = (res[res['tier'] == t].pivot_table(index='target', columns='method',
                                               values='overlap').round(1))
        p['B-A'] = (p['B_ppg'] - p['A_carry_forward']).round(1)
        p['C-A'] = (p['C_ppg_x_games'] - p['A_carry_forward']).round(1)
        print(p.to_string())
        print(f"  B beats A in {int((p['B-A'] > 0).sum())}/25 seasons, "
              f"C beats A in {int((p['C-A'] > 0).sum())}/25 seasons")

    print('\n--- who ppg promotes that carry-forward does not (2025 board) ---')
    prior = df[df['season'] == 2024].copy()
    prior['rk_A'] = prior['ppr'].rank(ascending=False, method='first').astype(int)
    prior['rk_B'] = prior['ppg'].rank(ascending=False, method='first').astype(int)
    prior['rk_C'] = ((prior['ppg'] * prior['career_mean_g'])
                     .rank(ascending=False, method='first').astype(int))
    actual25 = set(df.loc[(df['season'] == 2025) & (df['rk'] <= 25), 'pid'])
    promoted = prior[(prior['rk_B'] <= 25) & (prior['rk_A'] > 25)].copy()
    promoted['hit_2025_top25'] = promoted['pid'].isin(actual25)
    print(promoted[['player', 'pos', 'g', 'ppr', 'ppg', 'career_mean_g',
                    'rk_A', 'rk_B', 'rk_C', 'hit_2025_top25']]
          .sort_values('rk_B').to_string(index=False))

    res.to_csv(OUT, index=False)
    c.report()
    print(f'\nwrote {OUT}')
    print('step 4 complete.')
    return pivot


if __name__ == '__main__':
    main()
