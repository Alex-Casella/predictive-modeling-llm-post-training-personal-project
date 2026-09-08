"""Experiment #2 -- put rookies in the candidate pool.

Finding #1: @250 cannot be moved by reordering, only by changing which players
are candidates. Finding #2: the composition of a real top 250 is

    carried    171.9  (68.8%)   in season n-1's top 250   <- carry-forward's bag
    returned    20.2  ( 8.1%)   had a prior season, not n-1
    brand new   57.9  (23.2%)   NO prior season at all    <- this file

Experiment #1 chased the 8.1% and netted +0.36. This chases the 23.2%, which is
what caps the whole project at 76.8%. `step14_draft_picks.py` supplied the only
input that can reach it: where a player was taken in the NFL draft, known years
before the season being predicted.

THE SELECTION TRAP, AND WHY EVERY DRAFTED PLAYER IS IN THE POOL

`draft_picks.csv` carries this project's `pid` only for players who eventually
reached a top 250. Ranking "drafted players who have a pid" would be ranking
players already known to have succeeded -- an outcome leaking backwards through
an identifier.

So every drafted player enters the pool. Those without a `pid` get a synthetic
id from their `pfr_id`, which by construction can never match an actual top-250
member. They occupy board slots and score nothing, which is exactly what a
busted draft pick does. The cost of being wrong about a rookie is paid, not
hidden.

HOW A ROOKIE IS SCORED, WITHOUT LOOKING AHEAD

A rookie has no statistics. What he has is a draft slot, and history says what
a slot like his has been worth. For target season n:

    expected(bucket) = mean first-season PPR of players drafted into that
                       bucket in years < n, counting 0 for the ones who never
                       reached a top 250 at all

Fitted per target season on drafts strictly before it, so the curve walks
forward with everything else. Counting the misses as 0 is the point: a bucket
where four in five bust must score lower than one where half hit.

Buckets are overall pick, never round -- the draft ran 12 rounds through 1992
and 7 from 1994, so "round 6" means different things in different eras
(step14's docstring).

History is restricted to drafts from 2000, the first season the file covers.
A player drafted in 1996 has no observable rookie season here, and counting
that as a zero is reading a blank as a number. It cost a full rewrite of this
paragraph: with 1994 the same top-ten slot came out worth 19.0 for target 2001
and 91.4 for 2015, an artefact of how much history each target could see.

    boost = 0    every rookie scores 0, ranks below every carried player,
                 and the method reduces EXACTLY to carry-forward. The baseline
                 is a special case, so the comparison is arithmetic.
    boost = 1    a rookie is worth exactly what his draft slot has historically
                 returned, on the same scale as a veteran's last-season PPR.

PREDICTION, STATED BEFORE RUNNING

Modest gain, smaller than the arithmetic suggests, and possibly negative. The
board has 250 slots and every rookie admitted evicts a carried player who hits
better than half the time. Experiment #1 predicted a loss and got +0.36; the
difference here is that rookies are three times the returner pool and draft
position is a real signal (Spearman -0.320 against debut PPR). Against that,
expected value per rookie is low and variance is enormous.

Run:
    python3 step17_rookie_pool.py

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import numpy as np
import pandas as pd

from checks import Checks
from evaluate import (TIERS, carry_forward, evaluate, order_by,
                      score_for_tuning, summarise)

SRC = 'fantasy_top250_derived.csv'
DRAFT = 'draft_picks.csv'
OUT = 'step17_results.csv'

TARGETS = list(range(2001, 2026))
# The boost is a fitted parameter, so it is chosen on the EARLY seasons and the
# late ones are reported. Sweeping a grid over all 25 targets and keeping the
# best is fitting to the evaluation set -- the same mistake step16 caught after
# it produced a 0.10 "improvement" that was really the winner's curse.
FIT_TARGETS = list(range(2001, 2016))
REPORT_TARGETS = list(range(2016, 2026))
BOOST_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
# How many draft classes are eligible. 1 = this year's rookies only; 2 also
# admits last year's undrafted-into-relevance players, who often break out in
# year two -- step14 measured only 57% of first top-250 seasons as rookie years.
CLASSES_GRID = [1, 2, 3]

# Overall pick, not round. Boundaries are the draft's own structure: top ten,
# rest of round one, day two, and then progressively thinner slices.
BUCKETS = [(1, 10), (11, 32), (33, 64), (65, 100), (101, 150), (151, 224),
           (225, 10_000)]
# A rookie-season PPR of 0 must mean "he did not reach that season's top 250",
# never "the file does not cover that season". fantasy_top250.csv starts in
# 2000, so a player drafted in 1996 scores 0 by construction and would drag the
# historical average down for a reason that has nothing to do with football.
#
# The first version of this file used 1994 (the 7-round era, per step14) and a
# top-ten pick came out worth 19.0 PPR for target 2001 against 91.4 for 2015 --
# the same slot, five times the value, purely because later targets had more
# observable history. CLAUDE.md: blanks are not zeros.
OBSERVABLE_FROM = 2000

# Below this many usable draft classes the bucket averages are noise, so no
# rookie is scored at all and the method degrades to plain carry-forward.
MIN_HISTORY = 3


def bucket_of(pick):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= pick <= hi:
            return i
    return len(BUCKETS) - 1


def prepare(draft, season):
    """One row per drafted player, with an id that is always safe to rank.

    `uid` is the project pid where one exists (so the player CAN match an
    actual top-250 member) and a synthetic pfr-derived id where it does not
    (so he cannot). Both occupy a board slot; only one can ever score.
    """
    d = draft.dropna(subset=['draft_pick', 'draft_year']).copy()
    d['bucket'] = d['draft_pick'].astype(int).map(bucket_of)
    d['uid'] = np.where(d['pid'].notna(), d['pid'], 'D_' + d['pfr_id'])

    # First-season PPR: what this player scored in his DRAFT year, or 0 if he
    # did not reach that season's top 250 at all. Zero is the honest value for
    # the metric being optimised -- he was not on the board.
    first = (season.merge(d[['pid', 'draft_year']].dropna(), on='pid')
                   .query('season == draft_year')[['pid', 'ppr']]
                   .rename(columns={'ppr': 'rookie_ppr'}))
    d = d.merge(first, on='pid', how='left')
    d['rookie_ppr'] = d['rookie_ppr'].fillna(0.0)
    return d


def bucket_values(d, n):
    """bucket -> mean rookie-year PPR, from drafts BEFORE season n only.

    Restricted to draft years whose rookie season the season file actually
    covers, so a 0 means "missed the top 250" and never "not in the data".
    """
    hist = d[(d['draft_year'] < n) & (d['draft_year'] >= OBSERVABLE_FROM)]
    if hist['draft_year'].nunique() < MIN_HISTORY:
        return {}
    return hist.groupby('bucket')['rookie_ppr'].mean().to_dict()


def rookie_order(d, boost, classes):
    """carry-forward, with this year's draft class merged in by expected value."""
    def order_fn(df, n):
        carried = df[df['season'] == n - 1][['pid', 'ppr', 'rk']].copy()
        carried = carried.rename(columns={'pid': 'uid'})
        carried['score'] = carried['ppr']

        # Anyone with ANY season before n is a veteran, not a rookie, and is
        # already handled by the carried pool. Ranking him twice would break
        # evaluate()'s duplicate assertion and double-count him.
        seen = set(df.loc[df['season'] < n, 'pid'])
        vals = bucket_values(d, n)
        # classes=1 means THIS year's class only: draft_year > n-1, i.e. == n.
        # An earlier off-by-one here (> n - classes - 1) quietly admitted two
        # classes whenever one was asked for.
        rk = d[(d['draft_year'] <= n) & (d['draft_year'] > n - classes)
               & (~d['uid'].isin(seen))].copy()
        rk['ppr'] = 0.0
        rk['score'] = boost * rk['bucket'].map(vals).fillna(0.0)
        # rk column: rookies sort after carried players on ties. The season
        # file's rk runs 1..250, so 999 puts every rookie behind every veteran
        # at equal score -- which is what makes boost=0 reduce exactly.
        rk['rk'] = 999

        pool = pd.concat([carried[['uid', 'ppr', 'rk', 'score']],
                          rk[['uid', 'ppr', 'rk', 'score']]],
                         ignore_index=True)
        pool = pool.rename(columns={'uid': 'pid'})
        return order_by(pool, ['score', 'ppr'], [False, False])
    return order_fn


def main():
    df = pd.read_csv(SRC)
    draft = pd.read_csv(DRAFT)
    d = prepare(draft, df)
    c = Checks('experiment #2 -- rookies in the pool')

    print(f'{len(d):,} drafted skill players, '
          f'{d["draft_year"].min()}-{d["draft_year"].max()}')
    print(f'  with a project pid (ever reached a top 250)  '
          f'{d["pid"].notna().sum():,}')
    print(f'  synthetic id only (never did)                '
          f'{d["pid"].isna().sum():,}')

    # -- what a draft slot has been worth ----------------------------------
    hist = d[d['draft_year'] >= OBSERVABLE_FROM]
    tbl = hist.groupby('bucket').agg(
        n=('uid', 'size'),
        hit_rate=('rookie_ppr', lambda s: 100 * (s > 0).mean()),
        mean_ppr=('rookie_ppr', 'mean'),
        mean_if_hit=('rookie_ppr', lambda s: s[s > 0].mean()))
    tbl.index = [f'{lo}-{hi if hi < 10000 else "end"}' for lo, hi in BUCKETS]
    print(f'\n--- what a draft slot returned in the ROOKIE season, '
          f'{OBSERVABLE_FROM}+ ---')
    print(tbl.round(1).to_string())
    print('  hit_rate = % that reached a top 250 as a rookie at all.')
    print('  mean_ppr counts the misses as 0, and is what the method ranks on.')
    c.check('earlier picks are worth more than later ones',
            tbl['mean_ppr'].iloc[0] > tbl['mean_ppr'].iloc[-1],
            f'{tbl["mean_ppr"].iloc[0]:.1f} vs {tbl["mean_ppr"].iloc[-1]:.1f}')

    # -- how many rookies are even reachable -------------------------------
    print('\n--- rookies available and how many actually land, by season ---')
    rows = []
    for n in TARGETS:
        seen = set(df.loc[df['season'] < n, 'pid'])
        actual = set(df.loc[df['season'] == n, 'pid'])
        new = actual - seen                      # true first-timers that season
        cls = d[(d['draft_year'] == n) & (~d['uid'].isin(seen))]
        rows.append(dict(n=n, brand_new=len(new), class_size=len(cls),
                         reachable=len(new & set(cls['uid']))))
    pool = pd.DataFrame(rows)
    print(pool.to_string(index=False))
    print(f"\n  mean {pool['brand_new'].mean():.1f} brand-new players per top "
          f"250, of which {pool['reachable'].mean():.1f} are in that year's "
          f"draft class")
    print(f"  the class averages {pool['class_size'].mean():.0f} players "
          f"competing for those slots -- a "
          f"{100 * pool['reachable'].mean() / pool['class_size'].mean():.1f}% "
          f"hit rate if you took the whole class")

    # -- boost = 0 must reproduce carry-forward exactly --------------------
    base = evaluate(df, TARGETS, carry_forward, label='A_carry_forward')
    b0 = evaluate(df, TARGETS, rookie_order(d, 0.0, 1), label='boost=0.0')
    m = base.merge(b0, on=['target', 'tier'], suffixes=('_base', '_b0'))
    c.check('boost=0 reproduces carry-forward exactly at every season and tier',
            (m['overlap_base'] == m['overlap_b0']).all(),
            f"{int((m['overlap_base'] != m['overlap_b0']).sum())} mismatches -- "
            f"the reduction is what makes this comparison arithmetic")

    # -- choose boost on the EARLY seasons only ----------------------------
    print(f'\n--- boost sweep on {FIT_TARGETS[0]}-{FIT_TARGETS[-1]} '
          f'(fitting half, 1 draft class) ---')
    fit = pd.concat([evaluate(df, FIT_TARGETS, rookie_order(d, b, 1),
                              label=f'boost={b}') for b in BOOST_GRID])
    print(summarise(fit).to_string())
    # Parameter choice uses the MOVABLE tiers, never @250 -- picking on the
    # headline is picking on the number being reported.
    tune = score_for_tuning(fit)
    best = tune.idxmax()
    best_b = float(best.split('=')[1])
    print(f'\n  best on movable tiers 25-200: {best} ({tune.max():.2f}%)')
    c.warn('the chosen boost is interior to the grid, not at its edge',
           0 < BOOST_GRID.index(best_b) < len(BOOST_GRID) - 1,
           f'{best_b} is an endpoint of {BOOST_GRID}, so the true optimum may '
           f'lie outside the grid and this result is a floor, not an estimate')

    # -- report it on seasons the choice never saw -------------------------
    print(f'\n--- HELD OUT {REPORT_TARGETS[0]}-{REPORT_TARGETS[-1]}, '
          f'boost={best_b} fixed in advance ---')
    res = pd.concat(
        [evaluate(df, REPORT_TARGETS, carry_forward, label='A_carry_forward'),
         evaluate(df, REPORT_TARGETS, rookie_order(d, best_b, 1),
                  label=f'boost={best_b}')])
    t = summarise(res)
    print(t.to_string())
    print('\n--- delta vs carry-forward, held out ---')
    print((t.loc[f'boost={best_b}'] - t.loc['A_carry_forward']).round(1)
          .to_frame('delta').T.to_string())

    print(f'\n--- for reference, the full sweep over all 25 seasons ---')
    full = pd.concat(
        [base] + [evaluate(df, TARGETS, rookie_order(d, b, 1),
                           label=f'boost={b}') for b in BOOST_GRID])
    print(summarise(full).to_string())
    print('  Read the HELD OUT block above as the result. This block includes '
          'the\n  seasons the boost was chosen on and will flatter it.')
    res = pd.concat([res, full.assign(method=lambda x: 'all25_' + x['method'])])

    # -- do older classes help? --------------------------------------------
    if best_b > 0:
        print(f'\n--- draft classes admitted, at boost={best_b} ---')
        cr = pd.concat([evaluate(df, REPORT_TARGETS, rookie_order(d, best_b, k),
                                 label=f'classes={k}') for k in CLASSES_GRID])
        print(summarise(cr).loc[[f'classes={k}' for k in CLASSES_GRID]]
              .to_string())
        res = pd.concat([res, cr])

    # -- the trade, made explicit ------------------------------------------
    print(f'\n--- what the swap actually costs, {best}, tier 250, held out ---')
    swap = []
    for n in REPORT_TARGETS:
        actual = set(df.loc[df['season'] == n, 'pid'])
        prev = set(df.loc[df['season'] == n - 1, 'pid'])
        picked = set(rookie_order(d, best_b, 1)(df, n)[:250])
        added, dropped = picked - prev, prev - picked
        swap.append(dict(n=n, added=len(added), added_hit=len(added & actual),
                         dropped=len(dropped),
                         dropped_hit=len(dropped & actual)))
    s = pd.DataFrame(swap)
    s['net'] = s['added_hit'] - s['dropped_hit']
    print(s.to_string(index=False))
    print(f"\n  mean per season: {s['added'].mean():.1f} rookies added, "
          f"{s['added_hit'].mean():.1f} hit; "
          f"{s['dropped'].mean():.1f} veterans evicted, "
          f"{s['dropped_hit'].mean():.1f} of them would have hit")
    print(f"  net {s['net'].mean():+.1f} players per season "
          f"({100 * s['net'].mean() / 250:+.2f} points at tier 250)")

    res.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    c.report()


if __name__ == '__main__':
    main()
