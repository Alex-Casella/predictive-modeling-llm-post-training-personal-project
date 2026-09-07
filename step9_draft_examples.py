"""Stage 3 -- generate (situation -> correct pick) examples for the agent.

PROJECT_CONTEXT.md §11e stage 3. Stages 1 and 2 are done, so this is unblocked.

No such dataset exists to buy, so it is manufactured by replaying history:

    for each season N:
        build the board for N using ONLY seasons <= N-1     (leakage-free)
        simulate a 10-team snake draft off that board
        at every pick, ask: knowing how season N ACTUALLY went,
                            which available player should have been taken?

That last question is answerable because the outcome is already in the data.
This is the property §11g calls out -- a language task with ground truth that
arrives for free, and 26 seasons of it to replay offline.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not write advice prose. §11g: "the target advice text actually has to be
good -- which means it has to be generated carefully, since the model will
imitate its flaws." Fabricating confident-sounding reasoning here would bake my
guesses into the training set permanently. So this emits the SITUATION (facts,
all knowable before the pick) and the LABEL (the pick that turned out best, plus
the numbers that justify it). Turning those into natural language is a separate
decision with its own trap, and it is deliberately left open.

THE TWO TRAPS FROM §11f
-----------------------
1. Label balance. Generating uniformly floods the set with picks nobody would
   question -- §3b argues the top 50-75 are automatic starts. Rather than guess
   a cutoff, every example carries a measured `margin`: the actual-VBD gap
   between the best and second-best legal choice. A tiny margin is a coin flip
   and teaches noise. Filter on it at training time.

2. Point-in-time reconstruction. Every number in `situation` comes from seasons
   <= N-1. Only `label` and `margin` use season N. The separation is asserted,
   not assumed.

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features
from board import PARAMS, TIER_GAP, add_tiers
from checks import Checks
from draft import SHOW_AVAILABLE, TEAMS, legal
from vbd import ESPN_STANDARD, POS_ALIAS, LeagueConfig, add_vbd, baselines

SRC = 'fantasy_top250_derived.csv'
OUT = 'draft_examples.jsonl'

SEASONS = list(range(2007, 2026))   # 2007 gives the model 6 training seasons
ROUNDS = 15
# TEAMS, SHOW_AVAILABLE and legal() are defined in draft.py and imported above:
# assistant.py needs them to build a live situation in this exact shape, and
# importing this module to get them would drag sklearn into an interactive CLI.


def prepare(df):
    d, feat_cols, cat_cols = features.build(df)
    d = features.attach_target(d, df)
    d = pd.get_dummies(d, columns=cat_cols, prefix=cat_cols, dtype=float)
    return d, feat_cols + [c for c in d.columns
                           if c.startswith('pos_') and c != 'pos_rk']


def project_board(d, feat_cols, df, n, league):
    """The board a drafter could have built before season n. Nothing from n."""
    train = d[d['season_next'] <= n - 1]
    pool = d[d['season'] == n - 1].copy()
    assert train['season_next'].max() <= n - 1, f'{n}: look-ahead in training'
    assert len(pool) == 250

    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(train[feat_cols], train['next_ppr'].fillna(0.0))
    pool['proj_ppr'] = model.predict(pool[feat_cols]).round(1)

    # Only `pos` is missing -- get_dummies consumed it. `player` survived, so
    # re-merging it would silently produce player_x / player_y (same trap as
    # board.py). Merge the one column that is actually gone.
    pos = df.loc[df['season'] == n - 1, ['pid', 'season', 'pos']]
    pool = pool.merge(pos, on=['pid', 'season'], how='left', validate='1:1')
    assert pool['pos'].notna().all(), 'lost a position in the rejoin'

    rows = pool[['pid', 'player', 'pos', 'proj_ppr', 'ppr', 'ppg', 'g',
                 'rk', 'age_next']].rename(
        columns={'ppr': 'prev_ppr', 'ppg': 'prev_ppg', 'g': 'prev_g',
                 'rk': 'prev_rk', 'age_next': 'age'}).to_dict('records')
    add_vbd(rows, league, points_key='proj_ppr')
    board = add_tiers(pd.DataFrame(rows)).sort_values('vbd_rk')
    board['pos_rk'] = board.groupby('pos').cumcount() + 1
    return board.reset_index(drop=True)


def actual_vbd(df, n, league, pids):
    """Actual season-n VBD for every pid on the board.

    A player who missed season n's top 250 has no row. He did not score zero --
    he scored below the cutoff and the file does not say what (the same
    censoring as step 8). He is credited with the CUTOFF, an upper bound.

    That direction matters: crediting the maximum he could possibly have scored
    means he can never outrank a player who actually made the list. So the
    hindsight label is never a player who fell out -- asserted below.
    """
    real = df[df['season'] == n]
    rows = real[['pid', 'pos', 'ppr']].to_dict('records')
    add_vbd(rows, league, points_key='ppr')
    got = {r['pid']: (r['ppr'], r['vbd']) for r in rows}

    base = baselines(real.to_dict('records'), league, points_key='ppr')
    cutoff = float(real.loc[real['rk'] == 250, 'ppr'].iloc[0])

    out = {}
    for pid, pos in pids.items():
        slot = POS_ALIAS.get(pos, pos)
        if pid in got:
            out[pid] = dict(actual_ppr=got[pid][0], actual_vbd=got[pid][1],
                            made_top250=True)
        else:
            out[pid] = dict(actual_ppr=cutoff,
                            actual_vbd=round(cutoff - base.get(slot, 0.0), 1),
                            made_top250=False)
    return out, cutoff


def snake_order(teams, rounds):
    for r in range(rounds):
        seats = range(teams) if r % 2 == 0 else reversed(range(teams))
        for seat in seats:
            yield r + 1, seat


def build_season(d, feat_cols, df, n, league):
    board = project_board(d, feat_cols, df, n, league)
    pids = dict(zip(board['pid'], board['pos']))
    actual, cutoff = actual_vbd(df, n, league, pids)
    board['actual_vbd'] = board['pid'].map(lambda p: actual[p]['actual_vbd'])
    board['actual_ppr'] = board['pid'].map(lambda p: actual[p]['actual_ppr'])
    board['made_top250'] = board['pid'].map(lambda p: actual[p]['made_top250'])

    rosters = {s: [] for s in range(TEAMS)}
    counts = {s: {} for s in range(TEAMS)}
    gone = set()
    examples = []

    for pick_no, (rnd, seat) in enumerate(snake_order(TEAMS, ROUNDS), 1):
        avail = board[~board['pid'].isin(gone)]
        ok = avail[avail['pos'].apply(lambda p: legal(counts[seat], p))]
        if ok.empty:
            break

        # THE ACTION SPACE IS THE K SHOWN CANDIDATES, NOT EVERY AVAILABLE PLAYER.
        #
        # The first version of this labelled each pick with the argmax of actual
        # VBD over all ~200 available players. That is not "the right pick", it
        # is "who won the season": in 2021 the label was Cooper Kupp for the
        # first ten picks in a row, and a whole 150-pick draft had only 4-11
        # distinct labels. 83% of labels simply repeated the previous pick's.
        # Training on that teaches a model to name the season's breakout in
        # advance -- unlearnable, and exactly the outcome-chasing §11g warns
        # about ("the right decision can produce the wrong outcome").
        #
        # Restricting the label to the same K options the situation shows makes
        # the question answerable -- "of these 12 realistic choices, which was
        # best?" -- and guarantees the answer is visible.
        shown = ok.head(SHOW_AVAILABLE)
        board_pick = shown.iloc[0]          # best PROJECTED vbd = the board's pick
        by_actual = shown.sort_values(['actual_vbd', 'vbd_rk'],
                                      ascending=[False, True])
        best = by_actual.iloc[0]
        second = by_actual.iloc[1] if len(by_actual) > 1 else best
        margin = round(float(best['actual_vbd'] - second['actual_vbd']), 1)

        # Where the board's own pick landed once the season was played. 1 = the
        # board was right; K = it chose the worst of the options it surfaced.
        # Random choice averages (K+1)/2, so this is the number that says
        # whether the deterministic layer has any skill at all.
        order = list(by_actual['pid'])
        board_rank = order.index(board_pick['pid']) + 1
        examples.append(dict(
            season=n, pick=pick_no, round=rnd, seat=seat,
            situation=dict(
                league=league.label(),
                roster=[dict(player=p['player'], pos=p['pos'],
                             proj_ppr=p['proj_ppr']) for p in rosters[seat]],
                roster_counts=dict(counts[seat]),
                available=[dict(player=r.player, pos=r.pos, tier=int(r.tier),
                                pos_rk=int(r.pos_rk), vbd_rk=int(r.vbd_rk),
                                proj_ppr=float(r.proj_ppr), vbd=float(r.vbd),
                                prev_rk=int(r.prev_rk), age=int(r.age))
                           for r in shown.itertuples()]),
            board_pick=dict(pid=board_pick['pid'], player=board_pick['player'],
                            pos=board_pick['pos']),
            label=dict(pid=best['pid'], player=best['player'], pos=best['pos'],
                       actual_ppr=float(best['actual_ppr']),
                       actual_vbd=float(best['actual_vbd']),
                       made_top250=bool(best['made_top250']),
                       # A single argmax throws away most of the signal and is
                       # the most luck-sensitive thing you can train on. The
                       # full ordering supports a ranking loss instead, which
                       # degrades gracefully when the top two were a coin flip.
                       ranking=[dict(player=r.player, pos=r.pos,
                                     actual_vbd=float(r.actual_vbd))
                                for r in by_actual.itertuples()]),
            margin=margin,
            board_rank=board_rank,
            n_candidates=len(shown),
            board_was_right=bool(board_pick['pid'] == best['pid'])))

        taken = board_pick
        gone.add(taken['pid'])
        rosters[seat].append(dict(player=taken['player'], pos=taken['pos'],
                                  proj_ppr=float(taken['proj_ppr'])))
        slot = POS_ALIAS.get(taken['pos'], taken['pos'])
        counts[seat][slot] = counts[seat].get(slot, 0) + 1

    return examples, board


def main():
    df = pd.read_csv(SRC)
    league = LeagueConfig(teams=TEAMS, **ESPN_STANDARD)
    c = Checks('stage 3 -- draft example generation')
    d, feat_cols = prepare(df)

    print(f'generating for {SEASONS[0]}-{SEASONS[-1]} '
          f'({len(SEASONS)} seasons), {TEAMS} teams x {ROUNDS} rounds')
    all_ex = []
    for n in SEASONS:
        ex, _ = build_season(d, feat_cols, df, n, league)
        all_ex += ex
        print(f'  {n}: {len(ex)} examples', flush=True)

    ex = pd.DataFrame(all_ex)
    c.check('every example is one pick', len(ex) == len(all_ex))
    c.check('no example uses a season outside the range',
            ex['season'].between(SEASONS[0], SEASONS[-1]).all())

    # -- §11f trap 2: point-in-time -----------------------------------------
    # The situation may only contain columns from <= n-1. Assert structurally.
    banned = {'actual_ppr', 'actual_vbd', 'made_top250'}
    leaked = [k for e in all_ex[:500] for a in e['situation']['available']
              for k in a if k in banned]
    c.check('no actual-outcome field appears in any situation', not leaked,
            f'{len(leaked)} leaks')

    # The generous-imputation property from actual_vbd()'s docstring.
    c.check('the hindsight label is never a player who missed the top 250',
            not ex['label'].apply(lambda l: not l['made_top250']).any(),
            f"{int(ex['label'].apply(lambda l: not l['made_top250']).sum())} cases")
    c.check('the label is always one of the candidates shown',
            all(e['label']['player'] in {a['player'] for a in
                                         e['situation']['available']}
                for e in all_ex))

    # Guard against the failure that produced version 1 of this file: a label
    # that is the season's breakout regardless of the pick. If a whole 150-pick
    # draft has only a handful of distinct answers, the labels are outcomes,
    # not decisions.
    distinct = ex.assign(p=ex['label'].apply(lambda l: l['player'])) \
                 .groupby('season')['p'].nunique()
    # Regression guard against version 1, which scored 4-11 here. Not a
    # quality bar: consecutive picks share most of their candidate set, so a
    # genuinely standout player legitimately stays the right answer for several
    # picks running. 15 means the label changes at least ~10% of the time.
    c.check('labels are decisions, not one repeated breakout (v1 scored 4-11)',
            distinct.min() >= 15,
            f'min {int(distinct.min())} distinct labels in a 150-pick draft')

    # -- what the agent has to beat ------------------------------------------
    k = ex['n_candidates'].median()
    chance = (k + 1) / 2
    print(f'\n--- the baseline the agent must beat ---')
    print(f'  Each pick offers {int(k)} candidates. After the season is played')
    print(f'  they can be ranked by what actually happened. Where does the')
    print(f'  deterministic board\'s own choice land in that ranking?\n')
    print(f'    random choice would average    {chance:.1f}')
    print(f'    the board averages             {ex["board_rank"].mean():.2f}'
          f'   <- lower is better')
    print(f'    board picks the best of {int(k)}      '
          f'{100 * ex["board_was_right"].mean():.1f}%  '
          f'(chance {100 / k:.1f}%)')
    by_round = ex.groupby('round').agg(
        n=('board_rank', 'size'),
        board_rank=('board_rank', 'mean'),
        best_pct=('board_was_right', 'mean'),
        median_margin=('margin', 'median'))
    by_round['best_pct'] = (100 * by_round['best_pct']).round(1)
    print()
    print(by_round.round(2).to_string())

    # -- §11f trap 1: label balance ------------------------------------------
    print(f'\n--- §11f trap 1: how many of these are real decisions? ---')
    print(f'  margin = actual-VBD gap between best and second-best legal pick')
    print(f'  a small margin is a coin flip; training on it teaches noise\n')
    for lo, hi, lbl in [(0, 1, 'coin flip   (<1 pt)'),
                        (1, 5, 'marginal    (1-5)'),
                        (5, 20, 'real        (5-20)'),
                        (20, 1e9, 'decisive    (20+)')]:
        m = ex[(ex['margin'] >= lo) & (ex['margin'] < hi)]
        if len(m):
            print(f'  {lbl:<22} {len(m):>5}  ({100 * len(m) / len(ex):4.1f}%)   '
                  f'board rank {m["board_rank"].mean():.2f}   '
                  f'best in {100 * m["board_was_right"].mean():4.1f}%')

    print(f'\n  distinct labels per 150-pick draft: '
          f'min {int(distinct.min())}, median {int(distinct.median())}, '
          f'max {int(distinct.max())}')
    print(f'  (version 1 of this file scored 4-11 here -- that was the bug)')

    with open(OUT, 'w') as f:
        for e in all_ex:
            f.write(json.dumps(e) + '\n')
    print(f'\nwrote {OUT}: {len(all_ex)} examples')
    c.report()
    return ex


if __name__ == '__main__':
    main()
