"""Is the gap between two eval_agent.py runs real, or is it noise?

RESULTS.md reports the fine-tuned adapter at 5.41 mean rank against the
un-tuned base at 5.80. That is a 0.39 gap on 450 examples. This file decides
whether 0.39 is bigger than the noise.

WHY PAIRED AND NOT TWO-SAMPLE

Both models answered the SAME 450 prompts. That is not two independent
samples, it is one sample measured twice, and the pairing is information:

    example 1   base picked 8th-best,  tuned picked 3rd-best   ->  -5
    example 2   base picked 4th-best,  tuned picked 4th-best   ->   0
    example 3   base picked 11th-best, tuned picked 9th-best   ->  -2

A two-sample test compares two piles of numbers and has to absorb the fact
that some picks are just harder than others -- a round-1 pick with a clear
best option and a round-12 coin flip land in the same pile. Differencing
within an example cancels that difficulty out before the test ever sees it,
so the paired test is both the correct one and the more sensitive one.

    H0   the true mean difference is 0    (fine-tuning changed nothing)
    H1   the true mean difference is not 0
    x    the observed mean difference     (a number, not a hypothesis)

NEGATIVE IS BETTER HERE. `rank` is "where the chosen player finished among
the 12 shown", so a negative difference means the second model picked a
better player. The sign is checked and printed in words rather than left for
the reader to work out.

TWO TESTS, NOT ONE

The t-test assumes the differences are roughly normal. These differences are
between bounded integers in 1..12, so that assumption is worth doubting.
Wilcoxon signed-rank makes no such assumption -- it throws away the sizes and
keeps only the ranks of the absolute differences. Running both is the honest
move: if they agree the conclusion does not rest on the assumption, and if
they disagree that is itself the finding.

Usage:
    python3 paired_test.py                       # base vs fine-tuned
    python3 paired_test.py --a eval_board.csv --b eval_fantasy-draft.csv
"""
import argparse

import pandas as pd
from scipy import stats

# The columns that identify an example, independent of which model answered
# it. If these do not line up row-for-row the two runs are not comparable and
# every number below would be meaningless.
KEYS = ['season', 'round', 'margin']


def load_pair(path_a, path_b):
    """Two eval CSVs, aligned and checked.

    eval_agent.py writes one row per test example in file order, so row i of
    both files is the same example. That is an assumption, not a guarantee --
    a --limit run or a re-ordered test set would break it silently. Rather
    than trust it, assert it.
    """
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    if len(a) != len(b):
        raise SystemExit(
            f'different lengths: {path_a} has {len(a)} rows, {path_b} has '
            f'{len(b)}.\n  A --limit run cannot be compared against a full '
            f'one. Re-run the shorter model without --limit.')
    mismatched = (a[KEYS] != b[KEYS]).any(axis=1).sum()
    if mismatched:
        raise SystemExit(
            f'{mismatched} rows describe different examples in the two files. '
            f'They are not aligned and cannot be paired.')
    return a, b


def describe(name, frame):
    ok = frame[frame['valid']]
    return (f'{name:<28} n={len(frame):>4}  valid={100 * frame["valid"].mean():5.1f}%  '
            f'mean rank={ok["rank"].mean():.3f}')


def histogram(diffs, width=48):
    """Text histogram, because the t-test's assumption is about this shape.

    Printed rather than plotted so this runs anywhere -- matplotlib is not in
    requirements.txt, and a shape this coarse does not need it.
    """
    counts = diffs.value_counts().sort_index()
    peak = counts.max()
    print(f'\n  difference (tuned - base), one row per possible value')
    print(f'  negative = the tuned model picked better\n')
    for value, count in counts.items():
        bar = '#' * max(1, round(width * count / peak))
        marker = '  <- no change' if value == 0 else ''
        print(f'   {value:>+4}  {count:>4}  {bar}{marker}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='eval_llama3.1_8b.csv',
                    help='baseline run (the model being compared AGAINST)')
    ap.add_argument('--b', default='eval_fantasy-draft.csv',
                    help='the run being tested')
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    a, b = load_pair(args.a, args.b)

    print('=== the two runs ===')
    print(' ', describe(args.a, a))
    print(' ', describe(args.b, b))

    # An invalid answer has rank NaN and cannot be differenced. Drop the pair,
    # and say so -- silently dropping rows is how a comparison stops being
    # like-for-like without anyone noticing.
    both_valid = a['valid'] & b['valid']
    if not both_valid.all():
        print(f'\n  dropping {(~both_valid).sum()} example(s) where at least '
              f'one model gave an invalid answer')
    d = (b.loc[both_valid, 'rank'] - a.loc[both_valid, 'rank']).astype(int)

    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / n ** 0.5          # spread of the MEAN, not of one difference

    print(f'\n=== the difference column ===')
    print(f'  n                                  {n}')
    print(f'  mean difference                    {mean:+.3f}   <- x, what you measured')
    print(f'  H0 predicts                         0.000    <- if tuning did nothing')
    print(f'  sd of one difference               {sd:.3f}')
    print(f'  se of the mean  (sd / sqrt(n))     {se:.3f}   <- one "noise width"')
    print(f'  distance from H0, in noise widths  {abs(mean) / se:.2f}')

    better = (d < 0).sum()
    worse = (d > 0).sum()
    same = (d == 0).sum()
    print(f'\n  picked better on   {better:>4}  ({100 * better / n:.1f}%)')
    print(f'  picked worse on    {worse:>4}  ({100 * worse / n:.1f}%)')
    print(f'  identical rank on  {same:>4}  ({100 * same / n:.1f}%)')

    histogram(d)

    print(f'\n=== the tests ===')
    t = stats.ttest_rel(b.loc[both_valid, 'rank'], a.loc[both_valid, 'rank'])
    print(f'  paired t-test        t={t.statistic:+.3f}  p={t.pvalue:.4g}')
    print(f'    assumes the differences above are roughly normal')

    # zero_method='wilcox' (the default) DISCARDS the tied pairs entirely,
    # which for this data is a large share of the rows. 'zsplit' keeps them
    # and splits their signs, which is the conservative choice: ties are
    # evidence AGAINST a difference, so dropping them flatters the result.
    w = stats.wilcoxon(b.loc[both_valid, 'rank'], a.loc[both_valid, 'rank'],
                       zero_method='zsplit')
    print(f'  Wilcoxon signed-rank W={w.statistic:.1f}  p={w.pvalue:.4g}')
    print(f'    assumes nothing about the shape; ties kept, signs split')

    agree = (t.pvalue < args.alpha) == (w.pvalue < args.alpha)
    print(f'\n=== reading it ===')
    direction = 'BETTER' if mean < 0 else 'WORSE'
    print(f'  {args.b} picked {direction} than {args.a}')
    print(f'  by {abs(mean):.3f} ranks of 12, on average.')
    if agree:
        verdict = ('too large to explain as noise'
                   if t.pvalue < args.alpha else
                   'NOT distinguishable from noise')
        print(f'  Both tests agree at alpha={args.alpha}: the gap is {verdict}.')
    else:
        print(f'  THE TWO TESTS DISAGREE at alpha={args.alpha}. Trust Wilcoxon --')
        print(f'  it is the one that does not assume normality, and the')
        print(f'  histogram above is why that assumption is doubtful. Report')
        print(f'  the disagreement rather than picking the friendlier number.')

    print(f'\n  A p-value is P(seeing a gap this big | H0 is true). It is NOT')
    print(f'  the probability that fine-tuning worked, and a small p says')
    print(f'  nothing about whether 0.39 ranks of 12 is worth caring about.')
    print(f'  That is the effect size above, and it is a separate judgement.')


if __name__ == '__main__':
    main()
