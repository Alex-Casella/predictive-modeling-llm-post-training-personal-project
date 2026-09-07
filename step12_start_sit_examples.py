"""Stage 6 step 2 -- turn weekly data into (situation -> correct start) examples.

The sit/start analogue of step9. Same decision shape, so eval_agent.py's whole
scoring harness applies unchanged:

    step9   "of these 12 available players, which should I DRAFT?"
    step12  "of these 6 players on my roster, which should I START at flex?"

Score is where your choice finished among the K shown, once the week was
played. Lower is better; chance is (K+1)/2.

WHY FLEX AND NOT THE WHOLE LINEUP

A QB slot with one QB on the roster is not a decision. The flex is where a real
manager actually agonises, and restricting to it keeps the action space to K
named options -- which is what makes the answer checkable and what lets
extract_pick, mean rank and the validity check carry over untouched.

WHERE THE ROSTERS COME FROM

There are no real rosters in any of this data, so they are simulated: a 10-team
snake draft over LAST SEASON'S PPR ranking, using this project's own snake_order
and position caps. Last season's finish is the carry-forward baseline
(RESULTS.md, 68.8% set overlap) -- a real and slightly naive thing a drafter
does, requiring no model and therefore no risk of leaking one.

The property that matters is not realism for its own sake. It is that the K
candidates are of SIMILAR QUALITY, because they came off one roster. A random
sample of six players would pit a league-winner against five waiver scraps and
the decision would be trivial.

NO LOOK-AHEAD, ENFORCED

Everything shown about a candidate is computed from weeks STRICTLY BEFORE the
week being decided, and asserted to be so. The opponent is included because a
manager genuinely knows it on Thursday; the points scored are the label and
appear nowhere in the situation.

THE LABEL IS THE BEST OF THE K SHOWN, NOT THE BEST PLAYER ALIVE

step9 documents at length why the action space must be the shown candidates:
labelling with an argmax over everyone returns "who won the season" rather than
"who should you have started". The same trap exists here in miniature and the
same fix applies.

Data sources: Pro-Football-Reference (season totals) and nflverse (weekly).
See ATTRIBUTION.md.
"""
import json

import pandas as pd

from checks import Checks
from draft import MAX_AT_POS, TEAMS, legal, snake_order
from step11_weekly_data import USABLE_FROM
from vbd import POS_ALIAS

SRC = 'weekly_ppr.csv'
NAMES = 'fantasy_top250.csv'      # pid -> display name; NEVER joined on
OUT = 'start_sit_examples.jsonl'

ROUNDS = 15                 # a standard roster: 9 starters plus a bench
SHOW = 6                    # candidates per decision -> chance is 3.5
FLEX_POS = {'RB', 'WR', 'TE'}

# Weeks 1-4 are excluded because season-to-date PPG does not exist yet in week 1
# and is one game of noise in week 2. FIRST_WEEK=5 gives every candidate at
# least four games of history, which is what makes the deterministic baselines
# meaningful rather than degenerate.
FIRST_WEEK = 5


def display_names(path=NAMES):
    """pid -> the name to PRINT. Identity stays on pid everywhere else.

    CLAUDE.md forbids joining on name and this does not join on name -- the
    prompt just has to say "Chase Brown", because a model cannot answer with
    "P1383" and extract_pick matches on names. The most recent spelling wins,
    since that is the one a reader recognises.
    """
    d = pd.read_csv(path)[['pid', 'season', 'player']]
    return (d.sort_values('season').groupby('pid')['player'].last().to_dict())


def season_positions(wk):
    """One position per (pid, season). A player's `pos` can wobble week to week
    in the source; the season mode is stable and is what a roster slot uses."""
    return (wk.groupby(['pid', 'season'])['pos']
            .agg(lambda s: s.mode().iat[0]).rename('pos'))


def draft_rosters(prev_totals, pos_map, teams=TEAMS, rounds=ROUNDS):
    """Ten rosters, snake-drafted off last season's PPR finish.

    Uses draft.snake_order and draft.legal so the roster shape obeys the same
    position caps as the rest of the project. Returns {seat: [pid, ...]}.
    """
    order = prev_totals.sort_values(ascending=False).index.tolist()
    rosters = {s: [] for s in range(teams)}
    counts = {s: {} for s in range(teams)}
    gone = set()
    for rnd, seat in snake_order(teams, rounds):
        for pid in order:
            if pid in gone:
                continue
            pos = pos_map.get(pid)
            if pos is None or not legal(counts[seat], pos):
                continue
            slot = POS_ALIAS.get(pos, pos)
            rosters[seat].append(pid)
            counts[seat][slot] = counts[seat].get(slot, 0) + 1
            gone.add(pid)
            break
    return rosters


def form_history(wk):
    """Pre-week form for every player-week, using only STRICTLY earlier weeks.

    shift() before every window is what makes that true. Computed once for the
    whole file rather than per decision, because doing it per decision is where
    an accidental <= creeps in.
    """
    wk = wk.sort_values(['pid', 'season', 'week']).copy()
    g = wk.groupby(['pid', 'season'])['ppr']
    prior = g.shift()
    wk['games_so_far'] = prior.groupby([wk['pid'], wk['season']]).cumcount() + 1
    wk['ppg_std'] = prior.groupby([wk['pid'], wk['season']]).expanding().mean() \
        .reset_index(level=[0, 1], drop=True)
    wk['last3'] = prior.groupby([wk['pid'], wk['season']]).rolling(3).mean() \
        .reset_index(level=[0, 1], drop=True)
    wk['games_so_far'] = (prior.notna()
                          .groupby([wk['pid'], wk['season']]).cumsum())
    return wk


def build(wk, seasons, names):
    """One example per (season, week, seat) where K candidates are available."""
    pos_map_all = season_positions(wk)
    examples = []

    for n in seasons:
        prev = wk[wk['season'] == n - 1]
        if prev.empty:
            continue
        prev_totals = prev.groupby('pid')['ppr'].sum()
        pos_prev = pos_map_all.xs(n - 1, level='season').to_dict()
        rosters = draft_rosters(prev_totals, pos_prev)

        cur = wk[wk['season'] == n]
        by_week = {w: d.set_index('pid') for w, d in cur.groupby('week')}
        pos_now = pos_map_all.xs(n, level='season').to_dict()

        for week in sorted(w for w in by_week if w >= FIRST_WEEK):
            wd = by_week[week]
            for seat, roster in rosters.items():
                # Flex-eligible, on the roster, and PLAYED this week. A player
                # on bye has no row, and you cannot start him.
                cand = [p for p in roster
                        if pos_now.get(p) in FLEX_POS and p in wd.index]
                rows = wd.loc[cand].dropna(subset=['ppg_std'])
                if len(rows) < SHOW:
                    continue

                # The K shown are the top K by season-to-date PPG. That makes
                # shown.iloc[0] the naive baseline's own pick, exactly as
                # step9's shown.iloc[0] was the board's pick.
                shown = rows.sort_values('ppg_std', ascending=False).head(SHOW)
                by_actual = shown.sort_values('ppr', ascending=False)
                best, second = by_actual.iloc[0], by_actual.iloc[1]
                order = list(by_actual.index)

                examples.append(dict(
                    season=int(n), week=int(week), seat=int(seat),
                    situation=dict(
                        slot='FLEX',
                        candidates=[
                            dict(pid=p, player=names.get(p, p),
                                 pos=str(r.pos), opp=str(r.opp),
                                 ppg=round(float(r.ppg_std), 1),
                                 last3=(None if pd.isna(r.last3)
                                        else round(float(r.last3), 1)),
                                 games=int(r.games_so_far))
                            for p, r in shown.iterrows()]),
                    baseline_pick=names.get(shown.index[0],
                                            str(shown.index[0])),
                    baseline_pid=str(shown.index[0]),
                    label=dict(
                        pid=str(by_actual.index[0]),
                        player=names.get(by_actual.index[0],
                                         str(by_actual.index[0])),
                        points=round(float(best['ppr']), 2),
                        ranking=[dict(pid=str(p), player=names.get(p, str(p)),
                                      points=round(float(r.ppr), 2))
                                 for p, r in by_actual.iterrows()]),
                    margin=round(float(best['ppr'] - second['ppr']), 2),
                    baseline_rank=order.index(shown.index[0]) + 1,
                    n_candidates=len(shown)))
    return examples


def rank_of(ex, pid):
    return [r['pid'] for r in ex['label']['ranking']].index(pid) + 1


def main():
    wk = pd.read_csv(SRC)
    wk = wk[wk['season'] >= USABLE_FROM]
    seasons = sorted(s for s in wk['season'].unique() if s > USABLE_FROM)
    print(f'{len(wk):,} player-weeks, seasons {wk["season"].min()}-'
          f'{wk["season"].max()}')
    print(f'building decisions for {seasons[0]}-{seasons[-1]} '
          f'(each needs the season before it for the draft)')

    wk = form_history(wk)
    ex = build(wk, seasons, display_names())
    c = Checks('stage 6 step 2 -- start/sit examples')

    print(f'\n{len(ex):,} decisions, '
          f'{len(ex) / len(seasons):.0f} per season')

    # -- no look-ahead, structurally ---------------------------------------
    # ppg is the mean of games_so_far games, all strictly before this week, so
    # games_so_far can never reach the week number itself.
    bad = [e for e in ex for cand in e['situation']['candidates']
           if cand['games'] >= e['week']]
    c.check('every candidate stat uses only earlier weeks', not bad,
            f'{len(bad)} candidates claim more games than weeks elapsed')
    c.check('no outcome field appears in any situation',
            not any(k in cand for e in ex
                    for cand in e['situation']['candidates']
                    for k in ('ppr', 'points', 'actual')))
    c.check('every decision offers exactly K candidates',
            all(e['n_candidates'] == SHOW for e in ex))
    c.check('the label is always one of the shown candidates',
            all(e['label']['pid'] in
                [x['pid'] for x in e['situation']['candidates']] for e in ex))
    c.check('no decision is before FIRST_WEEK',
            all(e['week'] >= FIRST_WEEK for e in ex))
    # Two players sharing a name inside ONE decision would make the answer
    # unresolvable -- extract_pick matches on the printed name, so the prompt
    # would offer the same string twice. Rare, but it must be zero, not rare.
    clash = [e for e in ex
             if len({c_['player'] for c_ in e['situation']['candidates']})
             != SHOW]
    c.check('no two candidates in one decision share a display name',
            not clash, f'{len(clash)} decisions have a duplicate name')
    c.check('every candidate has a display name',
            all(not str(c_['player']).startswith('P0') and
                not str(c_['player']).startswith('P1')
                for e in ex for c_ in e['situation']['candidates']),
            'some pids had no name in ' + NAMES)

    # -- the degeneracy step9 caught -----------------------------------------
    # If one player is the answer to most of a season's decisions, the task is
    # "name the season's breakout" rather than "make this week's call".
    per = {}
    for e in ex:
        per.setdefault(e['season'], []).append(e['label']['pid'])
    share = {s: max(pd.Series(v).value_counts()) / len(v) for s, v in per.items()}
    worst = max(share.values())
    c.check('no single player answers more than 10% of a season\'s decisions',
            worst <= 0.10, f'worst season: {100 * worst:.1f}%')
    print(f'\ndistinct answers per season: '
          f'{min(len(set(v)) for v in per.values())}-'
          f'{max(len(set(v)) for v in per.values())} '
          f'(of {min(len(v) for v in per.values())}-'
          f'{max(len(v) for v in per.values())} decisions)')

    # -- THE BARS ------------------------------------------------------------
    # Every deterministic rule scored on exactly these decisions, before any
    # model exists. This is the number the fine-tune has to beat, fixed in
    # advance rather than chosen afterwards to flatter a result.
    import random
    rng = random.Random(0)
    rows = []
    for e in ex:
        cands = e['situation']['candidates']
        by_ppg = max(cands, key=lambda c_: c_['ppg'])
        with3 = [c_ for c_ in cands if c_['last3'] is not None]
        by_l3 = max(with3, key=lambda c_: c_['last3']) if with3 else by_ppg
        rows.append(dict(
            season=e['season'], week=e['week'], margin=e['margin'],
            ppg=rank_of(e, by_ppg['pid']),
            last3=rank_of(e, by_l3['pid']),
            random=rank_of(e, rng.choice(cands)['pid'])))
    r = pd.DataFrame(rows)
    print(f'\n--- the bars, on all {len(r):,} decisions (rank of {SHOW}, '
          f'lower better) ---')
    print(f'  {"chance, by construction":<28} {(SHOW + 1) / 2:.2f}')
    for col, lbl in [('random', 'random choice'),
                     ('last3', 'start the hot hand (last 3)'),
                     ('ppg', 'start the best average (season-to-date)')]:
        print(f'  {lbl:<28} {r[col].mean():.2f}   '
              f'best-of-{SHOW} {100 * (r[col] == 1).mean():.1f}%')
    print(f'\n  by season (season-to-date rule):')
    print('   ' + '  '.join(f'{s}:{v:.2f}' for s, v in
                            r.groupby('season')['ppg'].mean().items()))

    with open(OUT, 'w') as f:
        for e in ex:
            f.write(json.dumps(e) + '\n')
    print(f'\nwrote {OUT}')
    c.report()


if __name__ == '__main__':
    main()
