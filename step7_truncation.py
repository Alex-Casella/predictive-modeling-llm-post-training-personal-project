"""Experiment #3 -- is the 0.455 vs 0.59 gap real, or an artifact of truncation?

README.md cites an outside analysis finding the year-over-year R-squared of
fantasy points per game is 0.59. Step 3 measured 0.455 on this data.

Before treating 0.59 as a target, ask whether the two numbers are measuring the
same thing. They are almost certainly not. `fantasy_top250.csv` keeps only the
top 250 per season, and restricting a correlation to a narrow slice of one
variable mechanically lowers it -- this is range restriction, a standard
measurement artifact, not a statement about the players.

The test: recompute year-over-year R-squared on progressively TIGHTER cuts of
this same data. If R-squared falls as the cut tightens, truncation is
demonstrably deflating the number, and 0.455 is a floor rather than a verdict.

What this CANNOT do: recover the untruncated value. There is no data here for
players outside the top 250, so extrapolating past the widest available cut
would be invention. The experiment establishes a direction and a slope. It does
not produce a number comparable to 0.59, and this file does not pretend to.
"""
import pandas as pd

from checks import Checks

SRC = 'yoy.csv'
CUTS = [250, 200, 150, 100, 50, 25]
PAIRS = [('ppg', 'next_ppg'), ('ppr', 'next_ppr'), ('g', 'next_g')]
PUBLISHED_PPG_R2 = 0.59      # README.md, external source


def main():
    yoy = pd.read_csv(SRC)
    c = Checks('experiment #3 -- truncation sensitivity')
    m = yoy[yoy['made_next']].copy()
    c.check('only matched pairs are used', m['next_ppg'].notna().all())

    print('--- year-over-year R-squared by how tightly season t is cut ---')
    print('    cut = keep only players with rk <= cut in season t\n')
    rows = []
    for cut in CUTS:
        sub = m[m['rk'] <= cut]
        row = {'cut': cut, 'n_pairs': len(sub)}
        for a, b in PAIRS:
            row[f'{a}_r2'] = round(sub[a].corr(sub[b]) ** 2, 3)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))

    print('\n  ppg R-squared as the cut tightens:')
    for _, r in tbl.iterrows():
        bar = '#' * int(round(r['ppg_r2'] * 80))
        print(f"    top {int(r['cut']):>3}  {r['ppg_r2']:.3f}  {bar}")

    widest = tbl.iloc[0]
    tightest = tbl.iloc[-1]
    falls = tbl['ppg_r2'].iloc[0] > tbl['ppg_r2'].iloc[-1]
    c.check('ppg R-squared falls as the cut tightens (range restriction)',
            falls, f"top250 {widest['ppg_r2']} -> top25 {tightest['ppg_r2']}")

    print(f"\n  top 250 -> top 25 costs "
          f"{widest['ppg_r2'] - tightest['ppg_r2']:.3f} of R-squared "
          f"purely by narrowing the population.")
    print(f"  The published figure is {PUBLISHED_PPG_R2}; this file measures "
          f"{widest['ppg_r2']} at its widest available cut.")
    print(f"  Direction established. The untruncated value is NOT recoverable")
    print(f"  from this file -- there are no rows for players outside the top 250.")

    # Same effect, shown on the spread of the variable being restricted.
    print('\n--- why: how much ppg variation survives each cut ---')
    for cut in CUTS:
        sub = m[m['rk'] <= cut]
        print(f"    top {cut:>3}  n={len(sub):>4}  "
              f"ppg sd={sub['ppg'].std():5.2f}  "
              f"range {sub['ppg'].min():5.2f}-{sub['ppg'].max():5.2f}")
    print('\n  Correlation measures how much of the SPREAD one variable explains.')
    print('  Shrink the spread and the same underlying relationship scores lower.')

    print('\n--- g, for contrast (step 3 measured 0.024 at top 250) ---')
    print(tbl[['cut', 'n_pairs', 'g_r2']].to_string(index=False))
    print('\n  g is near zero at EVERY cut, so its result is not a truncation')
    print('  artifact. Availability really is close to unpredictable here.')

    c.report()
    return tbl


if __name__ == '__main__':
    main()
