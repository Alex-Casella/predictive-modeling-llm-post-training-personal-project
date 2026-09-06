"""Track A -- the first actual model. Everything before this was sorting.

Gradient-boosted trees over the prior-season / career / lag features in
features.py, validated walk-forward: to predict season N, train only on pairs
whose OUTCOME season is <= N-1. Never a random split (CLAUDE.md hard rule 2).

    predict 2015  ->  train on t+1 in 2001..2014
    predict 2016  ->  train on t+1 in 2001..2015     (one more season each step)

Three variants, because the target needs a decision and the honest way to make
it is to run all of them:

  M1  regression, censored rows imputed at the top-250 CUTOFF for that season
  M2  regression, censored rows imputed at ZERO
  M3  classifier on P(finishes in next season's top 250)

The censoring problem, stated plainly: a player in season t's top 250 who is
absent from t+1's did not score zero. He scored something below the 250th
player's total, and this file does not record what. M1 treats that as "just
below the line" (biased optimistic, but the right order of magnitude). M2
treats it as "scored nothing" (badly wrong for a WR3 who finished 260th,
correct-ish for someone who tore an ACL in week 1). M3 sidesteps it: set
membership is exactly what the metric measures, so predict membership directly.

Dropping censored rows was not offered. PROJECT_CONTEXT.md §7: train only on
survivors and the model learns a world where nobody falls out.

The candidate pool is still season n-1's top 250, so @250 stays pinned at the
baseline (step 4). This experiment tests one thing: can a model ORDER that pool
better than sorting a column? Tiers 25-200 are where the answer lives.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import r2_score

import features
from checks import Checks
from evaluate import TIERS, carry_forward, evaluate, order_by, summarise

SRC = 'fantasy_top250_derived.csv'
OUT = 'step8_results.csv'

TARGETS = list(range(2006, 2026))     # 20 seasons; 2006 trains on 2001-2005
SEED = 0
PARAMS = dict(max_iter=300, learning_rate=0.06, max_depth=6,
              min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)


def prepare(df):
    d, feat_cols, cat_cols = features.build(df)
    d = features.attach_target(d, df)
    d = pd.get_dummies(d, columns=cat_cols, prefix=cat_cols, dtype=float)
    feat_cols = feat_cols + [c for c in d.columns if c.startswith('pos_')
                             and c != 'pos_rk']
    return d, feat_cols


def make_order_fn(d, feat_cols, kind):
    """kind: 'cut' | 'zero' | 'clf'."""
    cache = {}

    def order_fn(df, n):
        if n not in cache:
            train = d[d['season_next'] <= n - 1].copy()
            test = d[d['season'] == n - 1].copy()
            assert train['season_next'].max() <= n - 1, 'look-ahead in training set'
            assert len(test) == 250, f'{n}: pool is {len(test)}, expected 250'

            X, Xt = train[feat_cols], test[feat_cols]
            if kind == 'clf':
                model = HistGradientBoostingClassifier(**PARAMS)
                model.fit(X, train['made_next'].astype(int))
                pred = model.predict_proba(Xt)[:, 1]
            else:
                y = train['next_ppr'].copy()
                fill = train['cut_next'] if kind == 'cut' else 0.0
                y = y.fillna(fill) if kind == 'cut' else y.fillna(0.0)
                model = HistGradientBoostingRegressor(**PARAMS)
                model.fit(X, y)
                pred = model.predict(Xt)
            test['pred'] = pred
            cache[n] = order_by(test, ['pred'], [False])
        return cache[n]
    return order_fn


def main():
    df = pd.read_csv(SRC)
    c = Checks('track A -- first model')
    d, feat_cols = prepare(df)

    print(f'feature table: {len(d)} rows x {len(feat_cols)} features')
    print(f'targets: {TARGETS[0]}-{TARGETS[-1]} ({len(TARGETS)} seasons)')

    # -- censoring, quantified before it is imputed -----------------------
    pairs = d[d['season'] < df['season'].max()]
    n_cens = int((~pairs['made_next']).sum())
    print(f'\n--- the censoring problem ---')
    print(f'  {len(pairs)} player-seasons with a following season')
    print(f'  {n_cens} ({100 * n_cens / len(pairs):.1f}%) absent from the next '
          f'top 250 -> target unknown, not zero')
    print(f'  cutoff (250th ppr) by season: '
          f"{df[df['rk'] == 250].set_index('season')['ppr'].describe()[['min', 'mean', 'max']].round(1).to_dict()}")
    c.check('censored rows are the majority-minority we expect (~31%)',
            0.25 < n_cens / len(pairs) < 0.35, f'{n_cens / len(pairs):.3f}')

    # -- no look-ahead, checked not assumed -------------------------------
    leak = d[d['season_next'] <= d['season']]
    c.check('every target season is strictly after its feature season', leak.empty)
    c.check('feature list excludes anything from the target row',
            not any(col.startswith('next_') or col == 'cut_next'
                    for col in feat_cols))

    # -- run --------------------------------------------------------------
    res = [evaluate(df, TARGETS, carry_forward, label='A_carry_forward')]
    for kind, label in [('cut', 'M1_reg (censored=cutoff)'),
                        ('zero', 'M2_reg (censored=0)'),
                        ('clf', 'M3_classifier P(top250)')]:
        print(f'  fitting {label} ...', flush=True)
        res.append(evaluate(df, TARGETS, make_order_fn(d, feat_cols, kind),
                            label=label))
    res = pd.concat(res)
    tbl = summarise(res)
    order = ['A_carry_forward', 'M1_reg (censored=cutoff)',
             'M2_reg (censored=0)', 'M3_classifier P(top250)']

    print(f'\n=== walk-forward result, {TARGETS[0]}-{TARGETS[-1]} ===\n')
    print(tbl.loc[order].to_string())
    print('\n--- delta vs carry-forward on the same seasons ---')
    print((tbl.loc[order] - tbl.loc['A_carry_forward']).round(1)
          .drop(index='A_carry_forward').to_string())

    c.check('all variants still tie at tier 250 (pool unchanged)',
            tbl.loc[order, 250].nunique() == 1, dict(tbl.loc[order, 250].round(2)))

    for m in order[1:]:
        p = res[(res['method'] == m) & (res['tier'] == 50)].set_index('target')
        b = res[(res['method'] == 'A_carry_forward') & (res['tier'] == 50)
                ].set_index('target')
        print(f'  {m:<28} beats baseline at tier 50 in '
              f"{int((p['overlap'] > b['overlap']).sum())}/{len(TARGETS)} seasons")

    # -- diagnostics (CLAUDE.md: report, do not substitute) ---------------
    print('\n--- diagnostics on the M1 regression, uncensored rows only ---')
    rows = []
    for n in TARGETS:
        train = d[d['season_next'] <= n - 1]
        test = d[(d['season'] == n - 1) & d['made_next']]
        if test.empty:
            continue
        m = HistGradientBoostingRegressor(**PARAMS)
        m.fit(train[feat_cols], train['next_ppr'].fillna(train['cut_next']))
        pr = m.predict(test[feat_cols])
        err = pr - test['next_ppr']
        rows.append(dict(n=n, n_test=len(test), mae=abs(err).mean(),
                         rmse=np.sqrt((err ** 2).mean()),
                         # Two different things, both routinely called "r2":
                         #   pearson_sq  = corr(pred, actual) ** 2. Scale- and
                         #     bias-blind: add 50 points to every prediction and
                         #     it does not move.
                         #   R2          = 1 - SSres/SStot. Penalises bias, and
                         #     goes NEGATIVE if you do worse than the mean.
                         # Reporting the first while calling it the second
                         # flatters the model. Both are printed.
                         pearson_sq=np.corrcoef(pr, test['next_ppr'])[0, 1] ** 2,
                         R2=r2_score(test['next_ppr'], pr),
                         bias=err.mean()))
    diag = pd.DataFrame(rows)
    print(diag.round(2).to_string(index=False))
    print(f"\n  mean MAE {diag['mae'].mean():.1f}   RMSE {diag['rmse'].mean():.1f}"
          f"   pearson_sq {diag['pearson_sq'].mean():.3f}"
          f"   R2 {diag['R2'].mean():.3f}"
          f"   bias {diag['bias'].mean():+.1f}")
    print('  diagnostics only -- set overlap is the metric that counts.')
    print('  NOTE: computed on SURVIVORS ONLY (players who made the next top')
    print('  250). That excludes the ~31% hardest cases and flatters every')
    print('  number here. It is not comparable to a published projection MAE.')

    res.to_csv(OUT, index=False)
    c.report()
    print(f'\nwrote {OUT}')
    return tbl


if __name__ == '__main__':
    main()
