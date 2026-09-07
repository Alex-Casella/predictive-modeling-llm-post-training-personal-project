"""Stage 7 step 1 -- NFL draft position, the one input that can reach rookies.

RESULTS.md finding #2: 23.2% of a real top 250 are players with NO prior season
anywhere in the data. They cannot be ranked, because there is nothing to rank.
That is what caps the board at 76.8%, and finding #1 proved it is the only cap
that matters: @250 cannot be moved by reordering, only by changing the
candidate pool.

This file adds the pool. It does nothing else -- no features, no model, no
board. One concept per step.

WHAT IT IS AND IS NOT

    IS      where a player was taken in the NFL draft, which is known
            YEARS before the season being predicted
    IS NOT  a prediction that he will be good

Draft position is a ranking signal, not a projection. Measured on this
project's own data, of skill-position players drafted 2001-2024:

    round 1   97.6% eventually reach a top 250    median debut 148.0 PPR
    round 4   55.9%                               median debut  93.6
    round 7   19.8%                               median debut  86.0

Monotone and steep at the top, which is all a candidate pool needs.

!! THIS SOURCE CARRIES CAREER OUTCOMES AND THEY ARE POISON !!

`import_draft_picks` returns `games`, `receptions`, `pass_yards`, `car_av`,
`probowls`, `allpro`, `hof`, `seasons_started`, `to` and more. Every one of
those describes what the player did over his WHOLE CAREER -- including seasons
after the one being predicted.

Using any of them as a feature is not a subtle leak, it is handing the model
the answer. A column named `probowls` would let it "predict" 2015 from facts
recorded in 2023. So the kept columns are an explicit WHITELIST rather than a
drop-list: a drop-list silently admits any column the source adds later, and
this source has 36 of them.

Kept: draft year, round, pick, team, position, college, age at draft.

ROUND NUMBERS ARE NOT COMPARABLE ACROSS ERAS

The NFL draft ran 12 rounds through 1992, 8 in 1993, and 7 from 1994. A
sixth-rounder in 1990 and a sixth-rounder in 2020 are different things, so
anything comparing across that boundary should use overall `pick` -- or the
pick's share of that year's draft -- rather than `round`.

Data sources: Pro-Football-Reference (season totals) and nflverse (weekly,
draft). See ATTRIBUTION.md.
"""
import sys

import pandas as pd

from checks import Checks

SEASON_FILE = 'fantasy_top250.csv'
OUT = 'draft_picks.csv'

# Back to 1985 because a player can debut in a top 250 years after being
# drafted -- 43% of first top-250 seasons are not the rookie year, and some are
# five years later. The earliest season this project predicts is 2000.
DRAFT_YEARS = list(range(1985, 2026))
POSITIONS = ['QB', 'RB', 'WR', 'TE', 'FB']

# WHITELIST. Everything not named here is discarded, including 20+ career-total
# columns that would leak the future. See the module docstring.
KEEP = {
    'season': 'draft_year',
    'round': 'draft_round',
    'pick': 'draft_pick',
    'team': 'draft_team',
    'position': 'draft_pos',
    'college': 'college',
    'age': 'draft_age',
    'pfr_player_id': 'pfr_id',
}

# Columns that must NEVER survive. Asserted, not assumed -- this is the check
# that fails loudly if someone widens the whitelist without thinking.
FORBIDDEN = ['games', 'receptions', 'rec_yards', 'rec_tds', 'rush_yards',
             'rush_tds', 'pass_yards', 'pass_tds', 'car_av', 'w_av', 'dr_av',
             'probowls', 'allpro', 'hof', 'seasons_started', 'to']


def fetch():
    try:
        import nfl_data_py as nfl
    except ImportError:
        sys.exit('nfl_data_py is not installed.  pip install nfl_data_py')
    print(f'downloading NFL draft picks, {DRAFT_YEARS[0]}-{DRAFT_YEARS[-1]} ...')
    frames, missing = [], []
    for yr in DRAFT_YEARS:
        try:
            frames.append(nfl.import_draft_picks([yr]))
        except Exception as e:
            missing.append((yr, type(e).__name__))
    if missing:
        print(f'  NOT AVAILABLE: '
              f'{", ".join(f"{y} ({e})" for y, e in missing)}')
    if not frames:
        sys.exit('no draft years downloaded -- check network access.')
    d = pd.concat(frames, ignore_index=True)
    print(f'  {len(d):,} picks, all positions, '
          f'{len(frames)} of {len(DRAFT_YEARS)} years')
    return d, [y for y, _ in missing]


def build(d, season):
    d = d[d['position'].isin(POSITIONS)].copy()
    d = d.dropna(subset=['pfr_player_id'])

    out = d[list(KEEP)].rename(columns=KEEP)

    # A player drafted twice (supplemental drafts, clerical duplicates) would
    # make the pool double-count him. Keep the EARLIEST, which is the one that
    # was knowable soonest, and count how many.
    dupes = out['pfr_id'].duplicated().sum()
    if dupes:
        print(f'  {dupes} pfr_id(s) appear in more than one draft; '
              f'keeping the earliest')
        out = out.sort_values('draft_year').drop_duplicates('pfr_id',
                                                            keep='first')

    # pid where this project already knows the player. MOST ROWS WILL HAVE NONE,
    # and that is the entire point: a drafted player who has never reached a top
    # 250 is exactly the candidate the board currently cannot see.
    key = season[['pid', 'pfr_id']].dropna().drop_duplicates('pfr_id')
    out = out.merge(key, on='pfr_id', how='left')

    for c in ('draft_year', 'draft_round', 'draft_pick'):
        out[c] = out[c].astype('Int64')
    return out.sort_values(['draft_year', 'draft_pick']).reset_index(drop=True)


def main():
    season = pd.read_csv(SEASON_FILE)
    raw, missing = fetch()
    d = build(raw, season)
    c = Checks('stage 7 step 1 -- NFL draft picks')

    known = d['pid'].notna()
    print(f'\n{len(d):,} skill-position picks, '
          f'{d["draft_year"].min()}-{d["draft_year"].max()}')
    print(f'  {known.sum():,} already appear in fantasy_top250.csv')
    print(f'  {(~known).sum():,} do NOT -- these are the candidates the board '
          f'currently cannot see')

    # -- the leak check, first because nothing else matters if it fails ------
    leaked = [col for col in FORBIDDEN if col in d.columns]
    c.check('no career-outcome column survived the whitelist', not leaked,
            f'LEAKED: {leaked}. These describe the whole career, including '
            f'seasons AFTER the one being predicted, and must never be '
            f'features.')
    c.check('every kept column is in the whitelist',
            set(d.columns) <= set(KEEP.values()) | {'pid'},
            f'not on the whitelist: '
            f'{sorted(set(d.columns) - set(KEEP.values()) - {"pid"})}')

    # -- structural ---------------------------------------------------------
    c.check('one row per player', not d['pfr_id'].duplicated().any(),
            f'{d["pfr_id"].duplicated().sum()} duplicate pfr_ids')
    c.check('every row has a draft year, round and pick',
            d[['draft_year', 'draft_round', 'draft_pick']].notna().all().all())
    c.warn('rounds run 1-7 (12 before 1993, 8 in 1993)',
           d[d['draft_year'] >= 1994]['draft_round'].max() <= 7,
           f'post-1994 max round is '
           f'{d[d["draft_year"] >= 1994]["draft_round"].max()}')

    # -- coverage of the players that matter --------------------------------
    # A "debut" is a player's FIRST top-250 season. Those are the rows the
    # board structurally cannot produce, so the question is what share of them
    # this file can now reach.
    debut = season.sort_values('season').groupby('pid').first().reset_index()
    debut = debut[debut['season'] >= 2001]
    hit = debut['pid'].isin(d.loc[known, 'pid'])
    print(f'\ndebut top-250 player-seasons 2001-2025   {len(debut):,}')
    print(f'  reachable via draft position           {hit.sum():,} '
          f'({100 * hit.mean():.1f}%)')
    print(f'  undrafted, so still invisible          {(~hit).sum():,} '
          f'({100 * (~hit).mean():.1f}%)')
    c.check('at least 70% of debut players are reachable', hit.mean() >= 0.70,
            f'{100 * hit.mean():.1f}%')

    # -- the signal check ---------------------------------------------------
    # If draft round did not order debut production at all, this file would be
    # 2,000 rows of noise and the whole idea would be dead. Verify before
    # building anything on it.
    dd = debut.merge(d[['pid', 'draft_round', 'draft_pick', 'draft_year']],
                     on='pid', how='inner')
    by_round = (dd[dd['draft_round'] <= 7]
                .groupby('draft_round')
                .agg(n=('ppr', 'size'), median_ppr=('ppr', 'median'),
                     median_rk=('rk', 'median')))
    print(f'\ndebut season by draft round -- does position carry signal?')
    print(by_round.round(1).to_string())
    r1 = by_round.loc[1, 'median_ppr']
    r7 = by_round.loc[7, 'median_ppr']
    c.check('round 1 debuts outscore round 7 debuts', r1 > r7,
            f'round 1 median {r1:.1f} vs round 7 {r7:.1f}')
    corr = dd[['draft_pick', 'ppr']].corr(method='spearman').iloc[0, 1]
    print(f'\n  Spearman(overall pick, debut PPR) = {corr:.3f}')
    print(f'  negative is expected: a lower pick number means a better player.')
    c.check('overall pick is negatively rank-correlated with debut PPR',
            corr < 0, f'{corr:.3f}')

    # -- when do they arrive? -----------------------------------------------
    dd['lag'] = dd['season'] - dd['draft_year']
    lag = dd['lag'].value_counts(normalize=True).sort_index()
    print(f'\nyears between being drafted and a first top-250 season:')
    for y in sorted(lag.index)[:6]:
        print(f'  {int(y)}  {100 * lag[y]:5.1f}%')
    print(f'  Only {100 * lag.get(0, 0):.0f}% are the rookie year, so the target')
    print(f'  is "first top-250 season", not "rookie season".')

    d.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}  ({len(d):,} rows)')
    c.report()


if __name__ == '__main__':
    main()
