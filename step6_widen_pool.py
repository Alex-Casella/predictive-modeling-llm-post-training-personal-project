"""Experiment #1 -- widen the candidate pool.

Step 4 proved the headline metric cannot move by reordering: every method drew
from the same 250-player bag, and the top 250 of a 250-player bag is the bag.
The only lever left is the bag itself.

Composition of each season's actual top 250, measured over 25 seasons:

    carried    171.9  (68.8%)   in season n-1's top 250   <- the current bag
    returned    20.2  ( 8.1%)   had a prior season, not n-1
    brand new   57.9  (23.2%)   no prior season at all    <- unreachable here

Rookies are three times the returner pool but cannot be backtested: the only
draft file covers 2026, so there is nothing to score against for 2001-2025.
Returners are the one pool widening this dataset can actually test.

    pool  = every player with any season < n
    score = ppr in his most recent season  x  d ** (gap - 1)
            gap = n - that season          d in [0, 1]

    d = 0    a returner scores 0 -> ranks below every carried player
             -> reduces EXACTLY to carry-forward. The baseline is a special
                case of this method, so the comparison is arithmetic, not
                apples-to-oranges.
    d = 1    a season three years old counts as much as last season.

Prediction, stated before running (§10 discipline applied to a hypothesis):
this loses. The board has exactly 250 slots, so every returner admitted evicts
a carried player. The marginal carried player still hits well over half the
time, and the ~20 real returners are buried in a candidate pool an order of
magnitude larger.
"""
import pandas as pd

from checks import Checks
from evaluate import TIERS, carry_forward, evaluate, order_by, summarise

SRC = 'fantasy_top250_derived.csv'
OUT = 'step6_results.csv'

D_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_GAP_GRID = [1, 2, 3, 5, 99]      # 1 == carry-forward pool only


def decay_order(d, max_gap=99):
    """Rank every previously-seen player by decayed most-recent ppr."""
    def order_fn(df, n):
        prior = df[df['season'] < n]
        # most recent season per player, strictly before n -> no look-ahead
        latest = prior.loc[prior.groupby('pid')['season'].idxmax()].copy()
        latest['gap'] = n - latest['season']
        latest = latest[latest['gap'] <= max_gap]
        latest['score'] = latest['ppr'] * (d ** (latest['gap'] - 1))
        # ppr then rk break ties, so d=0 is an EXACT reduction to carry-forward
        return order_by(latest, ['score', 'ppr'], [False, False])
    return order_fn


def main():
    df = pd.read_csv(SRC)
    c = Checks('experiment #1 -- widened pool')
    targets = list(range(2001, 2026))

    # -- how big does the bag get? ----------------------------------------
    print('--- candidate pool size by target season ---')
    rows = []
    for n in targets:
        prior = df[df['season'] < n]
        latest = prior.loc[prior.groupby('pid')['season'].idxmax()]
        gaps = n - latest['season']
        actual = set(df.loc[df['season'] == n, 'pid'])
        rows.append(dict(n=n, pool=len(latest),
                         gap1=int((gaps == 1).sum()),
                         gap2=int((gaps == 2).sum()),
                         gap3plus=int((gaps >= 3).sum()),
                         hits_in_pool=len(actual & set(latest['pid']))))
    pool = pd.DataFrame(rows)
    print(pool.to_string(index=False))
    print(f"\n  mean pool {pool['pool'].mean():.0f} candidates for 250 slots")
    print(f"  of which reachable hits: {pool['hits_in_pool'].mean():.1f} "
          f"({100 * pool['hits_in_pool'].mean() / 250:.1f}% -- the 76.8% ceiling)")
    print(f"  carry-forward searches only the gap1 column "
          f"({pool['gap1'].mean():.0f} candidates)")
    print(f"  full pool adds {pool['pool'].mean() - pool['gap1'].mean():.0f} "
          f"candidates to find ~20 extra hits")

    # -- d = 0 must reproduce carry-forward exactly ------------------------
    base = evaluate(df, targets, carry_forward, label='A_carry_forward')
    d0 = evaluate(df, targets, decay_order(0.0), label='d=0.0')
    merged = base.merge(d0, on=['target', 'tier'], suffixes=('_base', '_d0'))
    c.check('d=0 reproduces carry-forward exactly at every season and tier',
            (merged['overlap_base'] == merged['overlap_d0']).all(),
            f"{int((merged['overlap_base'] != merged['overlap_d0']).sum())} mismatches")

    # -- the decay sweep ---------------------------------------------------
    print('\n--- decay sweep, all 25 target seasons, full pool ---')
    res = [base] + [evaluate(df, targets, decay_order(d), label=f'd={d}')
                    for d in D_GRID]
    res = pd.concat(res)
    tbl = summarise(res)
    order = ['A_carry_forward'] + [f'd={d}' for d in D_GRID]
    print(tbl.loc[order].to_string())
    print('\n--- delta vs carry-forward ---')
    print((tbl.loc[order] - tbl.loc['A_carry_forward']).round(1)
          .drop(index='A_carry_forward').to_string())

    best_d = max(D_GRID, key=lambda d: tbl.loc[f'd={d}', 250])
    print(f"\n  best d at tier 250: {best_d}  "
          f"-> {tbl.loc[f'd={best_d}', 250]:.1f}%  "
          f"(carry-forward {tbl.loc['A_carry_forward', 250]:.1f}%)")
    c.check('@250 finally responds to the method (pool changed, not just order)',
            tbl.loc[order, 250].nunique() > 1,
            'if this fails the pool never actually widened')

    # -- does capping the gap help? ---------------------------------------
    print('\n--- gap cap sweep at d=0.7 (does old evidence hurt?) ---')
    gap_res = pd.concat(
        [evaluate(df, targets, decay_order(0.7, g), label=f'max_gap={g}')
         for g in MAX_GAP_GRID])
    print(summarise(gap_res).loc[[f'max_gap={g}' for g in MAX_GAP_GRID]]
          .to_string())

    # -- the trade, made explicit -----------------------------------------
    print(f'\n--- what the swap actually costs, d={best_d}, tier 250 ---')
    swap = []
    for n in targets:
        actual = set(df.loc[df['season'] == n, 'pid'])
        prev = set(df.loc[df['season'] == n - 1, 'pid'])
        picked = set(decay_order(best_d)(df, n)[:250])
        added = picked - prev            # returners promoted onto the board
        dropped = prev - picked          # carried players evicted to make room
        swap.append(dict(n=n, added=len(added),
                         added_hit=len(added & actual),
                         dropped=len(dropped),
                         dropped_hit=len(dropped & actual)))
    s = pd.DataFrame(swap)
    s['net'] = s['added_hit'] - s['dropped_hit']
    print(s.to_string(index=False))
    print(f"\n  mean per season: {s['added'].mean():.1f} returners added, "
          f"{s['added_hit'].mean():.1f} of them hit "
          f"({100 * s['added_hit'].sum() / max(s['added'].sum(), 1):.0f}% precision)")
    print(f"                   {s['dropped'].mean():.1f} carried players evicted, "
          f"{s['dropped_hit'].mean():.1f} of them would have hit "
          f"({100 * s['dropped_hit'].sum() / max(s['dropped'].sum(), 1):.0f}%)")
    print(f"  net players gained per season: {s['net'].mean():+.1f}")
    print(f'\n  break-even needs added precision > evicted precision.')

    pd.concat([res, gap_res]).to_csv(OUT, index=False)
    c.report()
    print(f'\nwrote {OUT}')
    return tbl


if __name__ == '__main__':
    main()
