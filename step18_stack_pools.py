"""Experiment #3 -- both widenings at once. Do the gains add?

Two experiments widened the candidate pool, at two disjoint groups:

    step6   returners   8.1% of a top 250, had a prior season but not n-1   +0.3
    step17  rookies    23.2% of a top 250, no prior season at all           +3.3

Disjoint by construction: a player either has a prior season or he does not. So
the arithmetic says +3.6. The arithmetic is almost certainly wrong, and the
reason is the thing worth measuring.

WHY THEY CAN FAIL TO ADD

The board has 250 slots. Both methods pay for their hits by evicting carried
players, and they evict from the SAME queue -- the bottom of last season's top
250. The second method to reach that queue finds it already thinned. Whether
+3.3 and +0.3 compose into +3.6 or into +3.3 is a question about how much
eviction the board can absorb, and only running it answers that.

    score(carried)   = ppr                       gap = 1
    score(returner)  = ppr x d**(gap - 1)        gap > 1     (step6)
    score(rookie)    = boost x expected(bucket)              (step17)

    d = 0 and boost = 0  ->  every non-carried player scores 0, ranks below
                             every carried player, and this reduces EXACTLY to
                             carry-forward. Asserted below.

BOTH PARAMETERS ARE FITTED ON 2001-2015 AND REPORTED ON 2016-2025.

step6 and step17 each swept one parameter; sweeping two on the same seasons you
then report doubles the winner's curse. The grid is searched on the early half
only, and the late half sees a single fixed pair.

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import pandas as pd

from checks import Checks
from evaluate import (carry_forward, evaluate, order_by, score_for_tuning,
                      summarise)
from step17_rookie_pool import DRAFT, SRC, bucket_values, prepare

OUT = 'step18_results.csv'

FIT_TARGETS = list(range(2001, 2016))
REPORT_TARGETS = list(range(2016, 2026))

# d centred on step6's optimum (0.3); boost on step17's (1.5).
D_GRID = [0.0, 0.2, 0.3, 0.4, 0.5]
BOOST_GRID = [0.0, 1.0, 1.5, 2.0, 3.0]


def stacked_order(d, boost, draft):
    """Carried + decayed returners + this year's draft class, one ranking."""
    def order_fn(df, n):
        prior = df[df['season'] < n]
        # Most recent season per player, strictly before n. Mirrors step6.
        latest = prior.loc[prior.groupby('pid')['season'].idxmax()].copy()
        latest['gap'] = n - latest['season']
        # gap 1 gives d**0 = 1, so carried players keep their full ppr for any
        # d -- which is what makes d=0 collapse to carry-forward rather than
        # zeroing the whole board.
        latest['score'] = latest['ppr'] * (d ** (latest['gap'] - 1))
        vets = latest[['pid', 'ppr', 'rk', 'score']]

        seen = set(prior['pid'])
        vals = bucket_values(draft, n)
        rk = draft[(draft['draft_year'] == n)
                   & (~draft['uid'].isin(seen))].copy()
        # Built column by column rather than renamed: `draft` already carries a
        # `pid` (null for players who never reached a top 250), so renaming
        # uid -> pid produces two columns of that name and concat then fails
        # with "Reindexing only valid with uniquely valued Index objects".
        rk = pd.DataFrame({
            'pid': rk['uid'].to_numpy(),
            'ppr': 0.0,
            'rk': 999,                      # sorts behind veterans on ties
            'score': boost * rk['bucket'].map(vals).fillna(0.0).to_numpy(),
        })

        return order_by(pd.concat([vets, rk], ignore_index=True),
                        ['score', 'ppr'], [False, False])
    return order_fn


def main():
    df = pd.read_csv(SRC)
    draft = prepare(pd.read_csv(DRAFT), df)
    c = Checks('experiment #3 -- rookies and returners together')

    # -- the reduction, first ---------------------------------------------
    base_fit = evaluate(df, FIT_TARGETS, carry_forward, label='A_carry_forward')
    zero = evaluate(df, FIT_TARGETS, stacked_order(0.0, 0.0, draft),
                    label='d=0,boost=0')
    m = base_fit.merge(zero, on=['target', 'tier'], suffixes=('_b', '_z'))
    c.check('d=0 and boost=0 reproduce carry-forward exactly',
            (m['overlap_b'] == m['overlap_z']).all(),
            f"{int((m['overlap_b'] != m['overlap_z']).sum())} mismatches")

    # -- search the grid on the EARLY seasons only -------------------------
    print(f'--- grid on {FIT_TARGETS[0]}-{FIT_TARGETS[-1]}, '
          f'mean overlap at movable tiers 25-200 ---')
    fits = []
    for d in D_GRID:
        for b in BOOST_GRID:
            fits.append(evaluate(df, FIT_TARGETS, stacked_order(d, b, draft),
                                 label=f'd={d}|boost={b}'))
    fit = pd.concat(fits)
    tune = score_for_tuning(fit)
    grid = pd.DataFrame(
        [[tune[f'd={d}|boost={b}'] for b in BOOST_GRID] for d in D_GRID],
        index=[f'd={d}' for d in D_GRID],
        columns=[f'boost={b}' for b in BOOST_GRID])
    print(grid.round(2).to_string())
    best = tune.idxmax()
    best_d, best_b = (float(x.split('=')[1]) for x in best.split('|'))
    print(f'\n  best pair: {best}  ({tune.max():.2f}%)')

    # -- report on seasons the search never saw ----------------------------
    print(f'\n--- HELD OUT {REPORT_TARGETS[0]}-{REPORT_TARGETS[-1]}, '
          f'd={best_d} boost={best_b} fixed in advance ---')
    # Each single-lever variant is scored too, so "did they add?" is answerable
    # from one table instead of by reading numbers off two other files that
    # were measured on a different span.
    runs = {
        'A_carry_forward': stacked_order(0.0, 0.0, draft),
        f'returners only (d={best_d})': stacked_order(best_d, 0.0, draft),
        f'rookies only (boost={best_b})': stacked_order(0.0, best_b, draft),
        f'BOTH (d={best_d}, boost={best_b})': stacked_order(best_d, best_b,
                                                            draft),
    }
    res = pd.concat([evaluate(df, REPORT_TARGETS, fn, label=k)
                     for k, fn in runs.items()])
    t = summarise(res).loc[list(runs)]
    print(t.to_string())

    print('\n--- delta vs carry-forward, held out ---')
    delta = (t - t.loc['A_carry_forward']).drop(index='A_carry_forward')
    print(delta.round(1).to_string())

    # -- the question this file exists to answer ---------------------------
    r = delta.loc[f'returners only (d={best_d})', 250]
    k = delta.loc[f'rookies only (boost={best_b})', 250]
    both = delta.loc[f'BOTH (d={best_d}, boost={best_b})', 250]
    print(f'\n--- do the gains add? tier 250 ---')
    print(f'  returners alone   {r:+.2f}')
    print(f'  rookies alone     {k:+.2f}')
    print(f'  sum if additive   {r + k:+.2f}')
    print(f'  actual, stacked   {both:+.2f}   '
          f'({100 * both / (r + k):.0f}% of additive)' if r + k else '')
    print(f'  Both pay for hits by evicting from the same queue -- the bottom')
    print(f'  of last season\'s 250. Anything short of additive is that queue')
    print(f'  running out.')

    c.warn('stacking beats the better single lever',
           both > max(r, k) + 1e-9,
           f'{both:+.2f} against {max(r, k):+.2f} -- the second widening found '
           f'the eviction queue already thinned')

    print(f'\n  @250 reached: {t.loc[f"BOTH (d={best_d}, boost={best_b})", 250]:.1f}%   '
          f'target 75.0   baseline {t.loc["A_carry_forward", 250]:.1f}%')
    print(f'  @25 and @50 moved '
          f'{delta.loc[f"BOTH (d={best_d}, boost={best_b})", 25]:+.1f} / '
          f'{delta.loc[f"BOTH (d={best_d}, boost={best_b})", 50]:+.1f} -- '
          f'CLAUDE.md asks that these be reported on every experiment.')

    res.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    c.report()


if __name__ == '__main__':
    main()
