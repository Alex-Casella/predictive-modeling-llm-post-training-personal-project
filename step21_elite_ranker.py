"""Stage 10 step 2 -- rank for the elite tier, which is where the headroom is.

step20 measured the two problems and found the project has only worked on one:

                  @25   @50   @100  @150  @200  @250
    carry_forward 42.1  51.2  61.4  65.1  67.6  68.8
    ORACLE        91.0  89.7  84.4  79.8  74.3  68.8
    headroom     +49.0 +38.5 +22.9 +14.7  +6.8   0.0

@250 cannot move by reordering and has been optimised anyway (+3.6 by widening
the pool). @25 has 49 points of reordering headroom and the best method here
has captured 1.4 of them.

WHY step8's CLASSIFIER MADE @25 WORSE

step8 ran three models. M3 predicted P(the player is in next season's top 250)
and scored 32.0 at @25 against carry-forward's 42.1 -- the worst result in the
file. That is not a failure of classification, it is a failure of TARGET
CHOICE: essentially every plausible top-25 candidate has a high probability of
making a top *250*, so the score is saturated exactly where the ordering has to
discriminate. It sorts confidently on a question whose answer is "yes" for
everyone who matters.

So this file keeps the classifier and changes the target to the tier being
ranked for: P(next season's top 25). Same features, same walk-forward, same
hyperparameters as step8 -- nothing is tuned here, which is what keeps this an
honest comparison rather than a search.

    E1   P(top 25)      ranks for the elite tier directly
    E2   P(top 50)      a wider, less sparse target
    E3   P(top 25) with predicted PPR as the tiebreak

E3 exists because a probability model over 250 candidates produces ties and
near-ties; breaking them on expected points rather than on row order is free
and is the kind of detail that quietly decided results earlier in this project.

WHAT WOULD MAKE THIS A FAILURE

If none of the three beats 42.1 at @25, the honest conclusion is that
prior-season box-score features do not separate a player who stays elite from
one who falls to 26-100 -- and the next lever is different features (injury,
depth chart, team change), not a different model. Recorded before running.

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

import features
from checks import Checks
from evaluate import TIERS, carry_forward, evaluate, order_by, summarise

SRC = 'fantasy_top250_derived.csv'
OUT = 'step21_results.csv'
TARGETS = list(range(2006, 2026))       # 2006 trains on 2001-2005, as step8
SEED = 0
# Identical to step8. Not tuned here: tuning on the same 25 seasons being
# reported is the winner's curse that step16 caught the hard way.
PARAMS = dict(max_iter=300, learning_rate=0.06, max_depth=6,
              min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)


def prepare(df):
    d, feat_cols, cat_cols = features.build(df)
    d = features.attach_target(d, df)

    # The new targets. `made_next` (top 250) already exists; these are the
    # tier-specific versions, and a player absent from next season's file is
    # False for all of them -- which is correct, not an imputation: he was not
    # in the top 25 either.
    nxt = df[['pid', 'season', 'rk']].rename(
        columns={'season': 'season_next', 'rk': 'next_rk'})
    d = d.merge(nxt, on=['pid', 'season_next'], how='left')
    for t in (25, 50):
        d[f'made_next_{t}'] = (d['next_rk'] <= t).fillna(False)

    d = pd.get_dummies(d, columns=cat_cols, prefix=cat_cols, dtype=float)
    feat_cols = feat_cols + [c for c in d.columns
                             if c.startswith('pos_') and c != 'pos_rk']
    return d, feat_cols


def make_order_fn(d, feat_cols, target, tiebreak_ppr=False):
    cache = {}

    def order_fn(df, n):
        if n not in cache:
            train = d[d['season_next'] <= n - 1]
            test = d[d['season'] == n - 1].copy()
            assert train['season_next'].max() <= n - 1, 'look-ahead in training'
            assert len(test) == 250, f'{n}: pool is {len(test)}'

            clf = HistGradientBoostingClassifier(**PARAMS)
            clf.fit(train[feat_cols], train[target].astype(int))
            test['p'] = clf.predict_proba(test[feat_cols])[:, 1]

            if tiebreak_ppr:
                reg = HistGradientBoostingRegressor(**PARAMS)
                reg.fit(train[feat_cols], train['next_ppr'].fillna(0.0))
                test['pred_ppr'] = reg.predict(test[feat_cols])
                cache[n] = order_by(test, ['p', 'pred_ppr'], [False, False])
            else:
                cache[n] = order_by(test, ['p'], [False])
        return cache[n]
    return order_fn


def main():
    df = pd.read_csv(SRC)
    d, feat_cols = prepare(df)
    c = Checks('stage 10 step 2 -- ranking for the elite tier')
    print(f'feature table {len(d)} rows x {len(feat_cols)} features, '
          f'targets {TARGETS[0]}-{TARGETS[-1]}')

    # -- how sparse is the target? -----------------------------------------
    pairs = d[d['season'] < df['season'].max()]
    print('\n--- how often is each target true, among the 250 candidates? ---')
    for t in (25, 50):
        r = pairs[f'made_next_{t}'].mean()
        print(f'  P(next top {t:>3}) = {100 * r:5.2f}%   '
              f'{int(pairs[f"made_next_{t}"].sum()):>4} positives')
    r250 = pairs['made_next'].mean()
    print(f'  P(next top 250) = {100 * r250:5.2f}%   '
          f'{int(pairs["made_next"].sum()):>4} positives   '
          f'<- step8 M3 ranked on this')
    print('  A target that is true for 69% of candidates cannot separate the')
    print('  top 25 from the next 75. That is why M3 scored 32.0 at @25.')

    c.check('no look-ahead: every target season is after its feature season',
            d[d['season_next'] <= d['season']].empty)
    c.check('no target column leaked into the features',
            not any(cl.startswith('next_') or cl.startswith('made_next')
                    or cl == 'cut_next' for cl in feat_cols))

    # -- run ----------------------------------------------------------------
    runs = [
        ('A_carry_forward', carry_forward),
        ('E1 P(top25)', make_order_fn(d, feat_cols, 'made_next_25')),
        ('E2 P(top50)', make_order_fn(d, feat_cols, 'made_next_50')),
        ('E3 P(top25) + ppr tiebreak',
         make_order_fn(d, feat_cols, 'made_next_25', tiebreak_ppr=True)),
    ]
    res = pd.concat([evaluate(df, TARGETS, fn, label=lbl) for lbl, fn in runs])
    t = summarise(res).loc[[lbl for lbl, _ in runs]]
    print(f'\n--- set overlap, {len(TARGETS)} walk-forward seasons ---')
    print(t.to_string())
    print('\n--- delta vs carry-forward ---')
    print((t - t.loc['A_carry_forward']).drop(index='A_carry_forward')
          .round(1).to_string())

    c.check('reordering leaves tier 250 exactly where it was',
            (t[250] == t.loc['A_carry_forward', 250]).all(),
            f'{t[250].to_dict()} -- these all rank the same 250 players, so '
            f'@250 must be identical; a difference means the pool changed')

    # -- against the ceiling -----------------------------------------------
    best = t.drop(index='A_carry_forward')[25].idxmax()
    got = t.loc[best, 25] - t.loc['A_carry_forward', 25]

    # step4's methods, RESTRICTED TO THIS SPAN. step4 ran 2001-2025 and this
    # runs 2006-2025, so its published 43.5% describes different seasons and
    # comparing to it directly would be the same span mismatch this project
    # has now made three times.
    s4 = pd.read_csv('step4_results.csv')
    s4 = s4[s4['target'].between(TARGETS[0], TARGETS[-1])]
    s4t = s4.pivot_table(index='method', columns='tier', values='overlap')
    simple = s4t[25].drop('A_carry_forward')

    print(f'\n--- reading it, at @25 ---')
    print(f'  carry-forward             {t.loc["A_carry_forward", 25]:.1f}%')
    for m, v in simple.sort_values(ascending=False).items():
        print(f'  {m:<25} {v:.1f}%   '
              f'({v - t.loc["A_carry_forward", 25]:+.1f})   <- arithmetic sort')
    print(f'  {best:<25} {t.loc[best, 25]:.1f}%   ({got:+.1f})   '
          f'<- this file')
    print(f'  ORACLE ceiling            91.0%   (+49.0 available)')
    if simple.max() > t.loc[best, 25]:
        print(f'\n  A TWO-COLUMN ARITHMETIC SORT BEATS THE MODEL at the tier')
        print(f'  the model was built for. {simple.idxmax()} needs no training,')
        print(f'  no features and no walk-forward, and it is '
              f'{simple.max() - t.loc[best, 25]:.1f} points ahead.')
        print(f'  Headroom captured: {100 * (simple.max() - t.loc["A_carry_forward", 25]) / 49.0:.0f}% '
              f'by arithmetic, {100 * got / 49.0:.0f}% by the model.')
    if got > 0:
        print(f'  The model captured {100 * got / 49.0:.0f}% of the 49 points '
              f'available.')
    else:
        print(f'  NOTHING CAPTURED. Prior-season box-score features do not')
        print(f'  separate a player who stays elite from one who falls to')
        print(f'  26-100, and the next lever is different features -- injury,')
        print(f'  depth chart, team change -- not a different model.')
    c.warn('some elite-tier headroom was captured', got > 0,
           f'{got:+.1f} at @25 -- see the note above, this was pre-registered '
           f'as the failure case and it is a result, not a bug')

    res.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    c.report()


if __name__ == '__main__':
    main()
