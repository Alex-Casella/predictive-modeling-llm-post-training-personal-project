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

THREE TESTS, NOT ONE

    paired t-test    assumes the differences are roughly normal
    Wilcoxon         assumes nothing about shape; keeps ranks, drops sizes
    sign test        keeps direction only; drops sizes AND ranks

Each discards more than the last, and each therefore rests on less. If all
three agree the conclusion does not depend on which you trust.

If they STRADDLE the threshold -- as they do on this project's draft result,
p = 0.047 / 0.054 / 0.090 -- the answer is not to pick the friendliest one.
It is that the effect sits at the edge of what the sample can resolve, and the
fix is more examples, not a different test. So this file also prints the
confidence interval and the n that 80% power would require.

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

    # A third view that throws away the sizes entirely and keeps only the
    # direction. If three tests with three different assumptions land in the
    # same place, the reading does not depend on which one you trust.
    sign = stats.binomtest(int(better), int(better + worse), 0.5)
    print(f'  sign test           {better}/{better + worse} better  '
          f'p={sign.pvalue:.4g}')
    print(f'    ignores how big each difference was; direction only')

    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(f'\n  95% CI for the true mean difference  '
          f'[{lo:+.3f}, {hi:+.3f}]')

    # The question a borderline p-value should prompt is not "which side of
    # 0.05" but "was this study big enough to answer at all".
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    need = (za + zb) ** 2 * sd ** 2 / mean ** 2 if mean else float('inf')
    print(f'  n for 80% power at this effect size  {need:.0f}'
          f'   (you have {n}, {need / n:.1f}x short)')

    agree = (t.pvalue < args.alpha) == (w.pvalue < args.alpha)
    print(f'\n=== reading it ===')
    direction = 'BETTER' if mean < 0 else 'WORSE'
    print(f'  {args.b} picked {direction} than {args.a}')
    print(f'  by {abs(mean):.3f} ranks of 12, on average.')
    ps = [t.pvalue, w.pvalue, sign.pvalue]
    if all(x < args.alpha for x in ps):
        print(f'  All three tests reject H0 at alpha={args.alpha}: the gap is')
        print(f'  too large to explain as noise.')
    elif all(x >= args.alpha for x in ps):
        print(f'  No test rejects H0 at alpha={args.alpha}: the gap is not')
        print(f'  distinguishable from noise at this sample size.')
    else:
        # The honest reading of a straddle is NOT "pick the test you like".
        # Three tests with three different assumptions landing either side of
        # one threshold means the threshold is where the effect sits, and the
        # sample cannot resolve it. Say that, and say how much data would.
        print(f'  THE TESTS STRADDLE alpha={args.alpha} '
              f'(p = {", ".join(f"{x:.3f}" for x in ps)}).')
        print(f'  Do NOT pick the friendlier one. Three tests with three sets')
        print(f'  of assumptions landing either side of one threshold means')
        print(f'  the effect sits AT the threshold and {n} examples cannot')
        print(f'  resolve it. The finding is "suggestive, underpowered", and')
        print(f'  the fix is {need:.0f} paired examples, not a different test.')

    print(f'\n  A p-value is P(seeing a gap this big | H0 is true). It is NOT')
    print(f'  the probability that fine-tuning worked, and a small p says')
    print(f'  nothing about whether 0.39 ranks of 12 is worth caring about.')
    print(f'  That is the effect size above, and it is a separate judgement.')


if __name__ == '__main__':
    main()
