"""Stage 6 step 1 -- acquire weekly data and prove it joins to this project.

`fantasy_top250.csv` is season totals with no week column, which is why
CLAUDE.md lists sit/start under "not yet" and RESULTS.md calls it blocked. This
file unblocks it, and does nothing else: no features, no model, no decisions.
One concept per step.

THE SOURCE

nflverse's `nfl_data_py`, which publishes per-player per-week fantasy points
back to 1999. It is the standard open source for this and it ships
`fantasy_points_ppr` already computed, so no scoring rules are reimplemented
here -- reimplementing them would be a second place for PPR to be defined and a
second place for it to be wrong.

    pip install nfl_data_py

THE JOIN IS ON IDENTIFIERS, NEVER NAMES

CLAUDE.md: "Never join or group on player name. 1,692 distinct players share
only 1,681 distinct names." That rule does not relax because the data came from
somewhere else. The path is:

    weekly.player_id  ==  ids.gsis_id  ->  ids.pfr_id  ==  top250.pfr_id  ->  pid

Two id joins and no string matching anywhere. `pid` stays the primary key.

THE CHECK THAT MATTERS

Summing a player's weekly PPR over a season should reproduce the `ppr` column
already in `fantasy_top250.csv`, which came from Pro-Football-Reference. Those
are two independent sources scored independently. If they agree, then the join
is right AND both scoring implementations are right, in one test. If they
disagree, everything downstream is built on sand and it is better to find out
now than after training an adapter.

Data sources: Pro-Football-Reference (season totals, see ATTRIBUTION.md) and
nflverse / nfl_data_py (weekly). Credit both in anything produced from this.
"""
import sys

import pandas as pd

from checks import Checks

SRC = 'fantasy_top250.csv'
OUT = 'weekly_ppr.csv'
SEASONS = list(range(2000, 2026))

# COVERAGE COLLAPSES BEFORE 2010 AND IT IS THE CROSSWALK, NOT THE WEEKLY DATA.
# Share of this project's top-250 player-seasons that have weekly rows, measured
# by this script:
#
#     2000  12%      2010  99%
#     2002  25%      2015 100%
#     2005  50%      2020 100%
#     2009  97%
#
# Early-2000s players largely have no pfr_id in nflverse's id table, so they
# cannot be matched however good the weekly file is. Season-total disagreement
# tracks the same curve (18.7% of rows off by >1 point in 2000-2004, 5.3% in
# 2020-2024). Anything built on weekly data starts here, and says so.
USABLE_FROM = 2010
POSITIONS = ['QB', 'RB', 'WR', 'TE', 'FB']

# A season total only reconstructs if every week is present. Postseason is
# excluded because `ppr` in fantasy_top250.csv is regular season only --
# step1_profile_g.py established that (no single-team player exceeds 17 games;
# a Super Bowl run would reach 20).
SEASON_TYPE = 'REG'


def fetch():
    try:
        import nfl_data_py as nfl
    except ImportError:
        sys.exit('nfl_data_py is not installed.\n'
                 '  pip install nfl_data_py\n'
                 '  (it is in requirements.txt)')
    # Year by year, not one bulk call. nfl_data_py 0.3.3 raises a bare 404 for
    # any season it cannot find, which kills the whole request -- and as of
    # writing 2025 is not published there, so a bulk call for 2000-2025 fetches
    # nothing at all. Skipping and REPORTING the gap is both more robust and
    # more honest than silently narrowing SEASONS to what happens to work.
    print(f'downloading weekly data, {SEASONS[0]}-{SEASONS[-1]} ...')
    frames, missing = [], []
    for yr in SEASONS:
        try:
            frames.append(nfl.import_weekly_data([yr]))
        except Exception as e:
            missing.append((yr, type(e).__name__))
    if missing:
        print(f'  NOT AVAILABLE: '
              f'{", ".join(f"{y} ({e})" for y, e in missing)}')
    if not frames:
        sys.exit('no seasons downloaded at all -- check network access.')
    w = pd.concat(frames, ignore_index=True)
    print(f'  {len(w):,} player-weeks before filtering, '
          f'{len(frames)} of {len(SEASONS)} seasons')
    ids = nfl.import_ids()
    return w, ids, [y for y, _ in missing]


def build(w, ids, season):
    w = w[(w['season_type'] == SEASON_TYPE)
          & w['position'].isin(POSITIONS)].copy()

    # gsis -> pfr. Drop rows where either side is missing rather than guessing:
    # a player with no pfr_id simply cannot be matched to this project's rows.
    xwalk = ids[['gsis_id', 'pfr_id']].dropna().drop_duplicates()

    # AMBIGUOUS pfr_ids ARE DROPPED, NOT RESOLVED. 12 pfr_ids in the crosswalk
    # are claimed by more than one gsis_id, and at least one pairs two
    # genuinely different people (CartKy01 is both Kyle Carter and David
    # Morgan). Keeping either would merge two careers into one player-week
    # series -- the exact failure CLAUDE.md forbids for names, arriving through
    # an id instead. There is no tie-break that is safe, so both go.
    ambiguous = set(xwalk['pfr_id'][xwalk.duplicated('pfr_id', keep=False)])
    if ambiguous:
        print(f'  dropping {len(ambiguous)} pfr_id(s) claimed by more than one '
              f'gsis_id: {sorted(ambiguous)[:4]}...')
    xwalk = xwalk[~xwalk['pfr_id'].isin(ambiguous)].drop_duplicates('gsis_id')
    w = w.merge(xwalk, left_on='player_id', right_on='gsis_id', how='inner')

    # pfr -> pid. One pid per pfr_id; season.pid is stable across seasons.
    key = (season[['pid', 'pfr_id']].dropna()
           .drop_duplicates('pfr_id'))
    w = w.merge(key, on='pfr_id', how='inner')

    out = (w[['pid', 'pfr_id', 'season', 'week', 'position',
              'recent_team', 'opponent_team', 'fantasy_points_ppr']]
           .rename(columns={'position': 'pos', 'recent_team': 'team',
                            'opponent_team': 'opp',
                            'fantasy_points_ppr': 'ppr'})
           .sort_values(['pid', 'season', 'week'])
           .reset_index(drop=True))
    out['ppr'] = out['ppr'].astype(float).round(2)

    # A player-week must be one row. Anything left after the ambiguity drop is
    # a duplicate inside the weekly source itself, not a join artifact, so it is
    # summed (both rows are that player's production in that game) and COUNTED,
    # never silently collapsed.
    dups = out.duplicated(['pid', 'season', 'week'], keep=False)
    if dups.any():
        print(f'  summing {dups.sum()} rows across '
              f'{out[dups].groupby(["pid", "season", "week"]).ngroups} '
              f'player-weeks that appear twice in the source')
        out = (out.groupby(['pid', 'pfr_id', 'season', 'week', 'pos'],
                           as_index=False)
               .agg(team=('team', 'first'), opp=('opp', 'first'),
                    ppr=('ppr', 'sum')))
    return out.sort_values(['pid', 'season', 'week']).reset_index(drop=True)


def main():
    season = pd.read_csv(SRC)
    w, ids, missing = fetch()
    wk = build(w, ids, season)
    c = Checks('stage 6 step 1 -- weekly data')

    print(f'\nkept {len(wk):,} player-weeks, '
          f'{wk["pid"].nunique():,} players, '
          f'seasons {wk["season"].min()}-{wk["season"].max()}')

    # -- structural ---------------------------------------------------------
    c.check('every row has a pid', wk['pid'].notna().all())
    c.check('no duplicate (pid, season, week)',
            not wk.duplicated(['pid', 'season', 'week']).any(),
            f'{wk.duplicated(["pid", "season", "week"]).sum()} duplicates')
    c.check('weeks are within the regular season',
            wk['week'].between(1, 18).all(),
            f'range {wk["week"].min()}-{wk["week"].max()}')
    # A season the SOURCE does not publish is a coverage fact to record, not a
    # bug to fail on -- but it must be loud, because every downstream split has
    # to respect it. A season that downloaded and then vanished IS a bug.
    c.warn('every requested season is published upstream', not missing,
           f'not available from nfl_data_py: {missing}')
    lost = set(SEASONS) - set(wk['season']) - set(missing)
    c.check('no downloaded season was lost in the join', not lost,
            f'downloaded but absent after joining: {sorted(lost)}')

    # -- coverage -----------------------------------------------------------
    # Coverage is only meaningful over seasons the source actually has.
    want = season[~season['season'].isin(missing)][['pid', 'season']]
    want = want.drop_duplicates()
    have = wk[['pid', 'season']].drop_duplicates()
    matched = want.merge(have, on=['pid', 'season'], how='inner')
    cov = 100 * len(matched) / len(want)
    print(f'\ncoverage: {len(matched):,} of {len(want):,} '
          f'top-250 player-seasons have weekly rows ({cov:.1f}%)')
    c.warn('at least 95% of top-250 player-seasons are covered, all seasons',
           cov >= 95,
           f'{cov:.1f}% over {SEASONS[0]}-2024 -- driven entirely by the '
           f'pre-{USABLE_FROM} crosswalk gap, see USABLE_FROM')

    usable = want[want['season'] >= USABLE_FROM]
    u_matched = usable.merge(have, on=['pid', 'season'], how='inner')
    u_cov = 100 * len(u_matched) / len(usable)
    print(f'  from {USABLE_FROM}: {len(u_matched):,} of {len(usable):,} '
          f'({u_cov:.1f}%)  <- the range downstream steps use')
    c.check(f'coverage from {USABLE_FROM} is at least 99%', u_cov >= 99,
            f'{u_cov:.1f}%')

    # -- THE RECONSTRUCTION CHECK -------------------------------------------
    # Two independent sources, two independent PPR implementations. If the
    # weekly sums land on the season totals, both are right.
    tot = wk.groupby(['pid', 'season'])['ppr'].sum().rename('weekly_sum')
    cmp = (season[['pid', 'season', 'player', 'ppr']]
           .merge(tot, on=['pid', 'season'], how='inner'))
    cmp['diff'] = cmp['weekly_sum'] - cmp['ppr']
    within = (cmp['diff'].abs() <= 1.0).mean()
    u_within = (cmp[cmp['season'] >= USABLE_FROM]['diff'].abs() <= 1.0).mean()
    print(f'\nreconstruction: weekly sums vs the season `ppr` column')
    print(f'  n            {len(cmp):,}')
    print(f'  median diff  {cmp["diff"].median():+.2f}')
    print(f'  mean |diff|  {cmp["diff"].abs().mean():.2f}')
    print(f'  within 1.0   {100 * within:.1f}%')
    print(f'  within 1.0, from {USABLE_FROM}   {100 * u_within:.1f}%')
    c.check('weekly sums reproduce the season totals for 90%+ of rows',
            within >= 0.90,
            f'{100 * within:.1f}% all seasons, {100 * u_within:.1f}% from '
            f'{USABLE_FROM}. Below 90% would mean the join or one of the two '
            f'PPR implementations is wrong and nothing downstream is safe.')

    worst = cmp.reindex(cmp['diff'].abs().sort_values(ascending=False).index)
    print(f'\n  largest disagreements (inspect before trusting the tail):')
    print(worst.head(5)[['player', 'season', 'ppr', 'weekly_sum', 'diff']]
          .to_string(index=False))

    # -- what the next step gets -------------------------------------------
    per = wk.groupby('season')['week'].max()
    print(f'\nweeks per season (17 from 2021, 16 before):')
    print('  ' + ', '.join(f'{s}:{n}' for s, n in per.items()))

    wk.to_csv(OUT, index=False)
    print(f'\nwrote {OUT}  ({len(wk):,} rows)')
    c.report()


if __name__ == '__main__':
    main()
