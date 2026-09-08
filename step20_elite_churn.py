"""Stage 10 -- why @25 and @50 never move, before trying to move them.

Four experiments have widened the candidate pool and every one of them gained
at tiers 100-250 and exactly 0.0 at tiers 25 and 50. CLAUDE.md says the elite
tiers are the ones that matter -- more than half of them turn over every year,
and that is where draft picks are most expensive.

So this file does NOT propose a method. It asks whether there is anything to
model, the same way step16 asked whether the matchup field carried signal
before anyone tried to make a model use it. That check cost half an hour and
saved a fine-tune.

WHY THE POOL CANNOT BE THE PROBLEM HERE

@250 moves only when the candidate pool changes (finding #1), because the top
250 of a 250-player pool is the whole pool. @25 is the opposite: 250 candidates
compete for 25 slots, so ORDERING is all that matters and widening the pool is
irrelevant. That is the whole explanation for the 0.0s -- rookies and returners
land in the tail, not the top.

    tier 250    pool-limited     -> widen the pool          (done, +3.6)
    tier 25     order-limited    -> rank the top better     (untouched)

Two different problems. Nothing this project has built addresses the second.

WHAT THIS MEASURES

    1. how much of the top 25 survives, and where the leavers go
    2. whether any single prior-season feature separates stayers from leavers
    3. what a perfect ranker of last season's top 250 could score at @25 --
       the ceiling on ordering, which bounds every future attempt

(3) is the number that decides whether this is worth working on. If a perfect
ordering of the existing pool only reaches, say, 55% at @25, then the elite
tier is mostly unpredictable from prior-season data and the honest move is to
say so rather than to keep tuning.

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import numpy as np
import pandas as pd

from checks import Checks
from evaluate import TIERS, carry_forward, evaluate, order_by

SRC = 'fantasy_top250_derived.csv'
OUT = 'step20_results.csv'
TARGETS = list(range(2001, 2026))
ELITE = 25


def main():
    df = pd.read_csv(SRC)
    c = Checks('stage 10 -- what happens to the elite tier')

    # -- 1. where does last year's top 25 go? ------------------------------
    rows = []
    for n in TARGETS:
        prev = df[(df['season'] == n - 1) & (df['rk'] <= ELITE)]
        cur = df[df['season'] == n].set_index('pid')['rk']
        landed = prev['pid'].map(cur)
        rows.append(dict(
            n=n,
            stayed_top25=int((landed <= 25).sum()),
            fell_26_100=int(((landed > 25) & (landed <= 100)).sum()),
            fell_101_250=int(((landed > 100) & (landed <= 250)).sum()),
            gone=int(landed.isna().sum())))
    ch = pd.DataFrame(rows)
    print(f'--- of last season\'s top {ELITE}, where are they this season? ---')
    print(ch.to_string(index=False))
    m = ch[['stayed_top25', 'fell_26_100', 'fell_101_250', 'gone']].mean()
    print(f'\n  mean per season, of {ELITE}:')
    for k, v in m.items():
        print(f'    {k:<14} {v:5.1f}  ({100 * v / ELITE:4.1f}%)')
    print(f'  So the board keeps {100 * m["stayed_top25"] / ELITE:.0f}% by doing '
          f'nothing, and the other\n  {100 * (1 - m["stayed_top25"] / ELITE):.0f}% '
          f'have to be replaced from somewhere.')

    # -- 2. where do this season's top 25 COME from? -----------------------
    rows = []
    for n in TARGETS:
        cur = df[(df['season'] == n) & (df['rk'] <= ELITE)]
        prev = df[df['season'] == n - 1].set_index('pid')['rk']
        came = cur['pid'].map(prev)
        older = set(df.loc[df['season'] < n - 1, 'pid'])
        rows.append(dict(
            n=n,
            from_top25=int((came <= 25).sum()),
            from_26_100=int(((came > 25) & (came <= 100)).sum()),
            from_101_250=int((came > 100).sum()),
            from_older=int(sum(1 for p, v in zip(cur['pid'], came)
                               if pd.isna(v) and p in older)),
            brand_new=int(sum(1 for p, v in zip(cur['pid'], came)
                              if pd.isna(v) and p not in older))))
    src = pd.DataFrame(rows)
    s = src[['from_top25', 'from_26_100', 'from_101_250', 'from_older',
             'brand_new']].mean()
    print(f'\n--- and where does THIS season\'s top {ELITE} come from? ---')
    for k, v in s.items():
        print(f'    {k:<14} {v:5.1f}  ({100 * v / ELITE:4.1f}%)')
    reachable = s['from_top25'] + s['from_26_100'] + s['from_101_250']
    print(f'\n  reachable by REORDERING last season\'s 250: '
          f'{reachable:.1f} of {ELITE} ({100 * reachable / ELITE:.0f}%)')
    print(f'  the rest were not in last season\'s top 250 at all, so no '
          f'ordering\n  of that pool can reach them.')

    # -- 3. THE CEILING: what could a perfect ranker of the pool score? ----
    # Order last season's 250 by what they ACTUALLY did next season. This is
    # cheating by construction and that is the point: it is the best any
    # reordering of this pool could possibly do, so it bounds every method.
    def oracle(d, n):
        pool = d[d['season'] == n - 1][['pid']].copy()
        nxt = d[d['season'] == n].set_index('pid')['rk']
        # Players absent next season get a rank worse than any real one.
        pool['next_rk'] = pool['pid'].map(nxt).fillna(9_999)
        pool['rk'] = pool['next_rk']
        return order_by(pool, ['next_rk'], [True])

    base = evaluate(df, TARGETS, carry_forward, label='carry_forward')
    orc = evaluate(df, TARGETS, oracle, label='ORACLE reorder')
    both = pd.concat([base, orc])
    tbl = both.pivot_table(index='method', columns='tier', values='overlap')
    print(f'\n--- the ceiling on REORDERING last season\'s 250 ---')
    print(tbl.round(1).loc[['carry_forward', 'ORACLE reorder']].to_string())
    gap = tbl.loc['ORACLE reorder'] - tbl.loc['carry_forward']
    print('\n  headroom a perfect ranker would have:')
    print('  ' + '  '.join(f'@{t} {gap[t]:+.1f}' for t in TIERS))
    c.check('the oracle cannot beat carry-forward at tier 250',
            abs(gap[250]) < 0.05,
            f'{gap[250]:+.2f} -- finding #1 says @250 cannot move by '
            f'reordering, so a non-zero here would contradict it')

    print(f'\n--- reading it ---')
    head = gap[25]
    print(f'  A PERFECT ordering of last season\'s 250 scores '
          f'{tbl.loc["ORACLE reorder", 25]:.1f}% at @25, against '
          f'{tbl.loc["carry_forward", 25]:.1f}% for doing nothing.')
    print(f'  So the entire prize for better ranking at the elite tier is '
          f'{head:.1f} points,')
    print(f'  and every point of it requires knowing next season in advance.')
    # step4's best reordering at @25, measured there, not assumed here.
    best_reorder = 43.5
    got = best_reorder - tbl.loc['carry_forward', 25]
    print(f'  The best reordering this project has measured (step4, {best_reorder}%)')
    print(f'  captured {got:.1f} of those {head:.1f} points -- '
          f'{100 * got / head:.0f}% of the available headroom.')

    both.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    c.report()


if __name__ == '__main__':
    main()
