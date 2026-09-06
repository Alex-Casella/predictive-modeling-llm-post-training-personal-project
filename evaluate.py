"""Shared evaluation harness: set overlap at six tiers, walk-forward.

Every experiment goes through `evaluate` so the numbers are comparable across
commits. An ordering function receives the FULL frame and the target season, and
must return an ordered list of pids built only from seasons < target
(CLAUDE.md hard rule 1). It picks its own candidate pool -- that is the whole
point of experiment #1.
"""
import pandas as pd

TIERS = [25, 50, 100, 150, 200, 250]

# @250 cannot distinguish two orderings of the same 250-player pool (step 4).
# Anything tuned on a metric that cannot move is tuned on nothing, so parameter
# selection uses the tiers that CAN move. Reported separately from the headline.
MOVABLE_TIERS = [25, 50, 100, 150, 200]


def evaluate(df, targets, order_fn, tiers=TIERS, label='method'):
    """Mean set overlap % per tier over `targets`, plus the per-season detail."""
    rows = []
    for n in targets:
        actual = df[df['season'] == n]
        assert len(actual) == 250, f'{n}: expected 250 rows, got {len(actual)}'
        actual_top = {t: set(actual.loc[actual['rk'] <= t, 'pid']) for t in tiers}

        order = order_fn(df, n)
        assert len(order) >= max(tiers), (
            f'{n}: {label} produced only {len(order)} candidates, '
            f'need {max(tiers)}')
        assert len(set(order)) == len(order), f'{n}: {label} returned duplicate pids'

        for t in tiers:
            hit = len(set(order[:t]) & actual_top[t])
            rows.append(dict(method=label, target=n, tier=t,
                             overlap=100 * hit / t))
    return pd.DataFrame(rows)


def summarise(results):
    """method x tier table of mean overlap."""
    return (results.pivot_table(index='method', columns='tier', values='overlap')
            .round(1))


def score_for_tuning(results):
    """One number per method for parameter selection: mean over movable tiers.

    Deliberately NOT @250. See MOVABLE_TIERS.
    """
    sub = results[results['tier'].isin(MOVABLE_TIERS)]
    return sub.groupby('method')['overlap'].mean()


# Sorting on ppr alone is not deterministic: exact ties exist and some straddle
# a tier boundary, so which player counts as "top 100" depended on row order
# rather than on the method. Real cases: Myles Gaskin / T.Y. Hilton both 164.2
# in 2020 (rk 100 and 101), Breece Hall / Chris Moore both 115.1 in 2022,
# Kayshon Boutte / Tyler Conklin both 121.9 in 2024. The file's own `rk` already
# resolves these, so every ordering breaks ties on rk ascending.
TIEBREAK = 'rk'


def order_by(frame, keys, ascending):
    """Deterministic descending sort with rk as the final tiebreak."""
    return (frame.sort_values(list(keys) + [TIEBREAK],
                              ascending=list(ascending) + [True])['pid']
            .tolist())


def carry_forward(df, n):
    """The baseline ordering: last season's 250, ordered by last season's ppr."""
    return order_by(df[df['season'] == n - 1], ['ppr'], [False])
