"""Experiment #2 -- shrunk PPG.

Step 4 ranked on raw ppg and it promoted five-game quarterbacks: Kyler Murray
finished 212th on the season but ranked 42nd by ppg off 5 games. That is the
small-denominator problem PROJECT_CONTEXT.md §6 exists to fix, so step 4 tested
ppg in a form already known to be broken.

The fix, §6:

    w = g / (g + k)
    shrunk_ppg = w * his_ppg + (1 - w) * positional_mean_ppg

    k = how many games of evidence it takes before his own number is trusted
        as much as the positional prior. w = 0.5 exactly at g = k.
    k = 0 -> w = 1 -> no shrinkage at all -> this IS step 4's method B.

`k` is a fitted parameter, so choosing it on all 25 seasons and reporting on
those same seasons is leakage of a different kind -- the seasons would be doing
double duty. k is chosen on 2001-2013 and reported on 2014-2025, which it never
saw. The baseline is recomputed on those same held-out seasons (CLAUDE.md:
"Every experiment reports its score next to the baseline score on the same
held-out years").

The positional mean is taken from season n-1 only, so it is knowable at
prediction time.
"""
import pandas as pd

from checks import Checks
from evaluate import (MOVABLE_TIERS, TIERS, carry_forward, evaluate, order_by,
                      score_for_tuning, summarise)

SRC = 'fantasy_top250_derived.csv'
OUT = 'step5_results.csv'

K_GRID = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
TRAIN = list(range(2001, 2014))     # 13 target seasons, used to pick k
TEST = list(range(2014, 2026))      # 12 target seasons, never seen by k


def shrunk_order(k_by_pos):
    """Ordering function: rank season n-1 by shrunk ppg. k may vary by position."""
    def order_fn(df, n):
        prior = df[df['season'] == n - 1].copy()
        pos_mean = prior.groupby('pos')['ppg'].transform('mean')
        k = prior['pos'].map(k_by_pos).astype(float)
        w = prior['g'] / (prior['g'] + k)
        prior['shrunk'] = w * prior['ppg'] + (1 - w) * pos_mean
        # ppr then rk break ties deterministically; at large k many players
        # collapse to their positional mean and `shrunk` stops discriminating.
        return order_by(prior, ['shrunk', 'ppr'], [False, False])
    return order_fn


def main():
    df = pd.read_csv(SRC)
    c = Checks('experiment #2 -- shrunk ppg')
    positions = sorted(df['pos'].unique())
    c.check('train and test seasons do not overlap', not set(TRAIN) & set(TEST))
    c.check('train + test covers all 25 targets',
            sorted(TRAIN + TEST) == list(range(2001, 2026)))

    # -- worked example, before any scoring ------------------------------
    print('--- what shrinkage does, k = 4, 2024 rows ---')
    prior = df[df['season'] == 2024].copy()
    pos_mean = prior.groupby('pos')['ppg'].transform('mean')
    w = prior['g'] / (prior['g'] + 4)
    prior['w'] = w.round(2)
    prior['pos_mean'] = pos_mean.round(2)
    prior['shrunk'] = (w * prior['ppg'] + (1 - w) * pos_mean).round(2)
    demo = prior[prior['player'].isin(
        ['Chris Godwin', 'Puka Nacua', 'Brock Purdy', 'Ja\'Marr Chase',
         'Saquon Barkley', 'Alvin Kamara'])]
    print(demo[['player', 'pos', 'g', 'ppg', 'w', 'pos_mean', 'shrunk']]
          .sort_values('ppg', ascending=False).to_string(index=False))

    # -- k sweep on TRAIN only -------------------------------------------
    print(f'\n--- k sweep, tuned on {TRAIN[0]}-{TRAIN[-1]} '
          f'({len(TRAIN)} seasons) ---')
    print(f'  selection score = mean overlap over tiers {MOVABLE_TIERS}')
    print(f'  (@250 cannot move -- step 4 -- so tuning on it would tune on nothing)\n')

    train_res = [evaluate(df, TRAIN, carry_forward, label='A_carry_forward')]
    for k in K_GRID:
        train_res.append(evaluate(df, TRAIN, shrunk_order({p: k for p in positions}),
                                  label=f'k={k}'))
    train_res = pd.concat(train_res)
    train_tbl = summarise(train_res)
    train_tbl['tune_score'] = score_for_tuning(train_res).round(2)
    order = ['A_carry_forward'] + [f'k={k}' for k in K_GRID]
    print(train_tbl.loc[order].to_string())

    # Two selection objectives, BOTH defined on train before looking at test.
    # They disagree, and the disagreement is the finding: shrinkage trades
    # tier-50 accuracy for tier-25 accuracy. Reporting only one would hide that.
    best_k = max(K_GRID, key=lambda k: train_tbl.loc[f'k={k}', 'tune_score'])
    best_k25 = max(K_GRID, key=lambda k: train_tbl.loc[f'k={k}', 25])
    print(f'\n  objective 1 -- mean of movable tiers: best k = {best_k}  '
          f'(k=0 is raw ppg, step 4 method B)')
    print(f'  objective 2 -- tier 25 alone:         best k = {best_k25}  '
          f"(CLAUDE.md: \"Watch the top 25 and top 50\")")
    print(f'  They disagree. Both are reported on the held-out split below.')

    # -- per-position k, also tuned on TRAIN only ------------------------
    print(f'\n--- per-position k sweep (§6: "likely differs by position") ---')
    per_pos_k = {}
    for pos in positions:
        scores = {}
        for k in K_GRID:
            trial = {p: (k if p == pos else best_k) for p in positions}
            r = evaluate(df, TRAIN, shrunk_order(trial), label=f'{pos}_k={k}')
            scores[k] = score_for_tuning(r).iloc[0]
        per_pos_k[pos] = max(scores, key=scores.get)
        line = '  '.join(f'{k}:{scores[k]:.2f}' for k in K_GRID)
        print(f'  {pos:<3} {line}   -> k={per_pos_k[pos]}')
    print(f'\n  per-position k: {per_pos_k}')

    # -- report on TEST, which k never saw --------------------------------
    print(f'\n=== held-out result, {TEST[0]}-{TEST[-1]} '
          f'({len(TEST)} seasons k never saw) ===\n')
    test_res = pd.concat([
        evaluate(df, TEST, carry_forward, label='A_carry_forward'),
        evaluate(df, TEST, shrunk_order({p: 0 for p in positions}),
                 label='B_raw_ppg (k=0)'),
        evaluate(df, TEST, shrunk_order({p: best_k for p in positions}),
                 label=f'D_shrunk k={best_k} (obj 1)'),
        evaluate(df, TEST, shrunk_order({p: best_k25 for p in positions}),
                 label=f'F_shrunk k={best_k25} (obj 2)'),
        evaluate(df, TEST, shrunk_order(per_pos_k),
                 label='E_shrunk per-position k'),
    ])
    test_tbl = summarise(test_res)
    row_order = ['A_carry_forward', 'B_raw_ppg (k=0)',
                 f'D_shrunk k={best_k} (obj 1)', f'F_shrunk k={best_k25} (obj 2)',
                 'E_shrunk per-position k']
    print(test_tbl.loc[row_order].to_string())
    print('\n--- delta vs carry-forward on the same held-out seasons ---')
    delta = (test_tbl.loc[row_order] - test_tbl.loc['A_carry_forward']).round(1)
    print(delta.drop(index='A_carry_forward').to_string())

    c.check('all methods still tie at tier 250 (same pool)',
            test_tbl[250].nunique() == 1, dict(test_tbl[250].round(3)))

    # -- did it fix the five-game problem? --------------------------------
    print('\n--- the 2025 board: who raw ppg promoted, and where shrinkage puts them ---')
    prior = df[df['season'] == 2024].copy()
    pm = prior.groupby('pos')['ppg'].transform('mean')
    for name, kk in [('raw k=0', 0), (f'k={best_k25}', best_k25)]:
        w = prior['g'] / (prior['g'] + kk)
        prior[name] = ((w * prior['ppg'] + (1 - w) * pm)
                       .rank(ascending=False, method='first').astype(int))
    prior['carry_rk'] = prior['ppr'].rank(ascending=False, method='first').astype(int)
    actual25 = set(df.loc[(df['season'] == 2025) & (df['rk'] <= 25), 'pid'])
    prior['hit'] = prior['pid'].isin(actual25)
    watch = prior[prior['player'].isin(
        ['Chris Godwin', 'Alvin Kamara', 'Puka Nacua', 'Tee Higgins',
         'Brock Purdy', 'CeeDee Lamb'])]
    print(watch[['player', 'pos', 'g', 'ppg', 'carry_rk', 'raw k=0',
                 f'k={best_k25}', 'hit']]
          .sort_values('raw k=0').to_string(index=False))

    print('\n--- per-season detail at tier 50, held-out seasons ---')
    p = (test_res[test_res['tier'] == 50]
         .pivot_table(index='target', columns='method', values='overlap').round(1))
    for m in row_order[1:]:
        p[f'{m[:1]}-A'] = (p[m] - p['A_carry_forward']).round(1)
    print(p.to_string())
    for m in row_order[1:]:
        wins = int((p[m] > p['A_carry_forward']).sum())
        print(f"  {m:<28} beats carry-forward in {wins}/{len(TEST)} held-out seasons")

    pd.concat([train_res.assign(split='train'),
               test_res.assign(split='test')]).to_csv(OUT, index=False)
    c.report()
    print(f'\nwrote {OUT}')
    return test_tbl


if __name__ == '__main__':
    main()
