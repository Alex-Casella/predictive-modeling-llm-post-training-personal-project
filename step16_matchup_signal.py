"""Stage 8 step 2 -- is there anything IN the matchup field to find?

step15 put opponent strength into the start/sit prompt. Before asking whether a
language model can use it, ask whether it is usable at all. Otherwise a null
LLM result has two explanations and no way to separate them:

    the model ignored a useful field       <- a fact about the model
    the field was not useful               <- a fact about football

This file answers the second so the first becomes readable.

HOW THE QUESTION IS PUT

A deterministic sort is the cleanest possible consumer of a feature: no
prompting, no tokenizer, no sampling, no runaway generation. If a sort that
sees the matchup cannot beat a sort that does not, the information is not
reachable by any simpler means, and a language model failing to use it says
nothing about language models.

    board          rank by season points per game        the existing bar, 2.92
    blend          rank by ppg + w x (matchup, z-scored)

`allowed` is standardised WITHIN (season, week, position) before blending.
Raw figures are not comparable across positions -- defenses concede ~25 to WRs
and ~8 to TEs -- so adding them unstandardised would rank every WR above every
TE regardless of form.

THE WEIGHT IS FITTED ON TRAIN+VAL AND THE TEST SEASONS ARE NEVER CONSULTED

Writing this the quick way first gave **2.820 against the board's 2.924**, and
that number is worthless: the weight had been chosen by trying a grid on the
2023-24 test set and keeping the best. That is fitting to the test, and it is
exactly the failure the walk-forward rule exists to prevent. It looked like a
0.10 improvement and was an artefact of picking the winner after seeing the
answers.

Fitted honestly on 2011-2022 and applied unchanged to 2023-24, the same idea
gives 2.878 -- and does not survive a paired test.

Run:
    python3 step16_matchup_signal.py

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from checks import Checks
from step15_matchup import SRC, WEEKLY, defense_table

FIT_SEASONS = range(2011, 2023)      # train + val, per step13's splits
TEST_SEASONS = (2023, 2024)
GRID = np.arange(0.0, 3.01, 0.25)


def standardiser(table):
    """(season, week, pos) -> (mean, sd) of points allowed across defenses."""
    buckets = {}
    for (season, week, _opp, pos), (ppg, _rk, _n) in table.items():
        buckets.setdefault((season, week, pos), []).append(ppg)
    # sd of 0 happens when one defense is the only one ranked that week; 1.0
    # then makes the z-score 0 rather than a division by zero.
    return {k: (float(np.mean(v)), float(np.std(v)) or 1.0)
            for k, v in buckets.items()}


def zed(table, norm, e, c):
    d = table.get((e['season'], e['week'], c['opp'], c['pos']))
    if d is None:
        return 0.0                    # unknown matchup ranks as average
    mean, sd = norm[(e['season'], e['week'], c['pos'])]
    return (d[0] - mean) / sd


def ranks(items, w, table, norm):
    """Where the sort's pick landed in hindsight. Lower is better."""
    out = []
    for e in items:
        best = max(e['situation']['candidates'],
                   key=lambda c: c['ppg'] + w * zed(table, norm, e, c))
        ids = [r['pid'] for r in e['label']['ranking']]
        out.append(ids.index(best['pid']) + 1)
    return np.array(out)


def main():
    rows = [json.loads(l) for l in open(SRC)]
    table = defense_table(pd.read_csv(WEEKLY))
    norm = standardiser(table)
    fit = [e for e in rows if e['season'] in FIT_SEASONS]
    test = [e for e in rows if e['season'] in TEST_SEASONS]
    c = Checks('stage 8 step 2 -- is the matchup field usable at all?')

    c.check('no test season is in the fitting set',
            not ({e['season'] for e in fit} & set(TEST_SEASONS)))

    # -- 1. does the field relate to the outcome at all? --------------------
    recs = []
    for e in test:
        pts = {r['pid']: r['points'] for r in e['label']['ranking']}
        for cand in e['situation']['candidates']:
            d = table.get((e['season'], e['week'], cand['opp'], cand['pos']))
            recs.append(dict(ppg=cand['ppg'], last3=cand['last3'],
                             allowed=None if d is None else d[0],
                             actual=pts[cand['pid']]))
    r = pd.DataFrame(recs)
    print(f'{len(test)} held-out decisions, {len(r)} candidate-rows\n')
    print('correlation with what the player ACTUALLY scored that week:')
    for col in ('ppg', 'last3', 'allowed'):
        s = r[[col, 'actual']].dropna()
        print(f'  {col:<8} pearson {s.corr().iloc[0, 1]:+.3f}   '
              f'spearman {s.corr(method="spearman").iloc[0, 1]:+.3f}   '
              f'n={len(s)}')
    allowed_r = r[['allowed', 'actual']].dropna().corr().iloc[0, 1]
    c.warn('points allowed points the right way (more allowed, more scored)',
           allowed_r > 0, f'{allowed_r:+.3f} -- a negative here would mean the '
                          f'feature is inverted, not merely weak')

    # -- 2. fit the weight WITHOUT looking at the test seasons -------------
    curve = [(w, ranks(fit, w, table, norm).mean()) for w in GRID]
    best_w = min(curve, key=lambda t: t[1])[0]
    print(f'\nweight fitted on {len(fit)} decisions, '
          f'{min(FIT_SEASONS)}-{max(FIT_SEASONS)}:')
    for w, m in curve:
        bar = '#' * int(round((m - 2.60) * 400))
        print(f'   w={w:.2f}  {m:.3f}  {bar}'
              f'{"   <- chosen" if w == best_w else ""}')
    spread = max(m for _, m in curve) - min(m for _, m in curve)
    flat = max(m for w, m in curve if w <= 1.25) - min(m for _, m in curve)
    print(f'  the curve is nearly flat below w=1.25: {flat:.3f} across those '
          f'weights,\n  against {spread:.3f} over the whole grid. A minimum '
          f'that shallow is\n  weak evidence for the weight it picks.')

    # -- 3. apply it, unchanged, to seasons never consulted -----------------
    a = ranks(test, 0.0, table, norm)
    b = ranks(test, best_w, table, norm)
    d = b - a
    print(f'\nHELD OUT {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}, '
          f'w={best_w:.2f} fixed in advance:')
    print(f'  ppg alone (the board)   {a.mean():.3f}')
    print(f'  ppg + matchup           {b.mean():.3f}   '
          f'({b.mean() - a.mean():+.3f})')
    print(f'  changed the pick on     {(d != 0).sum()} of {len(d)} decisions')

    t = stats.ttest_rel(b, a)
    w_ = stats.wilcoxon(b, a, zero_method='zsplit')
    better, worse = int((d < 0).sum()), int((d > 0).sum())
    sign = stats.binomtest(better, better + worse, 0.5)
    se = d.std(ddof=1) / len(d) ** 0.5
    print(f'  paired t p={t.pvalue:.4g}   Wilcoxon p={w_.pvalue:.4g}   '
          f'sign {better}/{better + worse} p={sign.pvalue:.4g}')
    print(f'  95% CI [{d.mean() - 1.96 * se:+.3f}, '
          f'{d.mean() + 1.96 * se:+.3f}]')

    c.check('the board score reproduces the published 2.92',
            abs(a.mean() - 2.924) < 0.01, f'{a.mean():.3f}')

    print('\n--- what this means for the LLM run ---')
    if all(p >= 0.05 for p in (t.pvalue, w_.pvalue, sign.pvalue)):
        print('  A SORT THAT SEES THE MATCHUP DOES NOT RELIABLY BEAT ONE THAT')
        print('  DOES NOT. So if llama3.1:8b scores about 3.02 on')
        print('  sitm_test.jsonl, that is NOT evidence it ignored the field.')
        print('  It is consistent with there being almost nothing to extract.')
        print('  The prompt experiment cannot separate those two, and saying')
        print('  "the model failed to use the matchup" would be unsupported.')
    else:
        print('  The sort DOES beat the board using this field, so the field')
        print('  carries usable signal and an LLM that ignores it has failed')
        print('  at something a sort managed.')
    c.report()


if __name__ == '__main__':
    main()
