"""Layer 3 -- the draft assistant CLI.

Reads the board, tracks who is gone and who is on your roster, and recommends a
pick with its reasoning shown rather than asserted.

The one concept worth internalising before using this:

    DRAFT by VBD.           Points above the last startable player at the
                            position. Answers "who helps me most relative to
                            what I could get instead?"

    START by projected points.  Once you own them, scarcity is irrelevant --
                            the flex slot just wants whoever scores most.

Those two orderings disagree constantly, which is why `pick` and `lineup` sort
on different columns. A QB can be the highest-scoring player on your roster and
still be a bad first-round pick, because the ninth-best QB is nearly as good and
the ninth-best RB is not.

Player lookup is by name because a human is typing, but names are NOT unique in
this dataset -- ten of them belong to more than one player. An ambiguous name
prints the candidates and refuses to guess; it never silently picks one
(CLAUDE.md: "Never join or group on player name").

Usage:
    python3 draft.py                      interactive
    python3 draft.py < commands.txt       scripted

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import json
import os
import re
import sys

import pandas as pd

from vbd import ESPN_STANDARD, POS_ALIAS, LeagueConfig

BOARD = 'board_2026.csv'
STATE = 'draft_state.json'

# Bonus added to VBD when a player fills a hole. Deliberately small: VBD is
# already the value estimate, and a need bonus large enough to reorder the board
# would be overriding the model with a guess.
NEED_BONUS = {'starter': 12.0, 'flex': 6.0, 'depth': 0.0}
# Past this many at a position, more of them is roster clog, not depth.
MAX_AT_POS = {'QB': 2, 'TE': 2, 'RB': 6, 'WR': 6}

# The shape of a draft: how many teams, and how many candidates a situation
# surfaces. These live HERE and not in step9_draft_examples.py, which owns the
# simulation, because step9 imports sklearn -- and a CLI that only needs to know
# "show twelve" should not have to load the modelling stack to find that out.
# step9 imports them back, so there is still one definition.
TEAMS = 10
SHOW_AVAILABLE = 12


def legal(roster_counts, pos):
    """May another player at this position still be drafted?

    `roster_counts` is keyed by SLOT, not raw position -- a FB occupies an RB
    slot. Callers must apply POS_ALIAS before counting or the cap silently
    stops applying to fullbacks.
    """
    slot = POS_ALIAS.get(pos, pos)
    return roster_counts.get(slot, 0) < MAX_AT_POS.get(slot, 6)


def normalise(s):
    """Fold a name to letters and spaces only.

    MUST be applied to the query as well as to the board, not just the board.
    PROJECT_CONTEXT.md §9 names apostrophes as a known hazard and it is real:
    normalising only one side means "Ja'Marr Chase" never matches the stored
    "jamarr chase", and the CLI then reports a top-15 player as missing.
    Hyphens do the same to "Amon-Ra St. Brown".
    """
    return re.sub(r'[^a-z ]', '', str(s).lower()).strip()


def load_board(path=BOARD):
    if not os.path.exists(path):
        sys.exit(f'{path} not found -- run `python3 board.py` first.')
    b = pd.read_csv(path)
    b['norm'] = b['player'].map(normalise)
    return b


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {'taken': [], 'mine': [], 'log': []}


def save_state(s):
    json.dump(s, open(STATE, 'w'), indent=1)


def resolve(board, query, state=None):
    """Name -> pid. Returns (pid, None) or (None, message)."""
    q = normalise(query)
    if not q:
        return None, 'give me a name.'
    hits = board[board['norm'].str.contains(q, regex=False)]
    if hits.empty:
        # Fall back to any single token before declaring him absent, so a
        # typo or a middle name does not get reported as "probably a rookie".
        toks = [t for t in q.split() if len(t) > 2]
        near = board[board['norm'].apply(
            lambda n: any(t in n for t in toks))] if toks else board.iloc[:0]
        if len(near):
            lines = [f'no exact match for "{query}". Did you mean:']
            lines += [f'   {r.player:<22} {r.pos}  vbd_rk {r.vbd_rk}'
                      for r in near.head(6).itertuples()]
            return None, '\n'.join(lines)
        return None, (f'no player matching "{query}" on the board.\n'
                      f'  If the name is spelled right he is probably a ROOKIE '
                      f'-- rookies are\n  not on this board at all '
                      f'(see board.py).')
    exact = hits[hits['norm'] == q]
    if len(exact) == 1:
        hits = exact
    if len(hits) > 1:
        lines = [f'"{query}" matches {len(hits)} players -- be more specific:']
        for r in hits.head(8).itertuples():
            lines.append(f'   {r.player:<22} {r.pos}  vbd_rk {r.vbd_rk:<4} '
                         f'pid {r.pid}')
        return None, '\n'.join(lines)
    return hits.iloc[0]['pid'], None


def available(board, state):
    return board[~board['pid'].isin(state['taken'] + state['mine'])]


def my_roster(board, state):
    r = board[board['pid'].isin(state['mine'])].copy()
    r['slot_pos'] = r['pos'].map(lambda p: POS_ALIAS.get(p, p))
    return r.sort_values('proj_ppr', ascending=False)


def best_lineup(roster, league):
    """Optimal starting lineup by PROJECTED POINTS, not VBD.

    This is the static sit/start answer this dataset can support: it never
    changes week to week, because the data has no weeks in it.
    """
    pool = roster.sort_values('proj_ppr', ascending=False)
    used, lineup = set(), []
    for pos, n in league.starters.items():
        picks = pool[(pool['slot_pos'] == pos)
                     & (~pool['pid'].isin(used))].head(n)
        for r in picks.itertuples():
            used.add(r.pid)
            lineup.append((pos, r))
        for _ in range(n - len(picks)):
            lineup.append((pos, None))
    flex_pool = pool[pool['slot_pos'].isin(league.flex_eligible)
                     & ~pool['pid'].isin(used)]
    for i in range(league.flex):
        if i < len(flex_pool):
            r = flex_pool.iloc[i]
            used.add(r['pid'])
            lineup.append(('FLEX', r))
        else:
            lineup.append(('FLEX', None))
    bench = pool[~pool['pid'].isin(used)]
    return lineup, bench


def need_level(roster, pos, league):
    """Does this position fill a starting slot, a flex slot, or only depth?"""
    slot = POS_ALIAS.get(pos, pos)
    have = int((roster['slot_pos'] == slot).sum()) if len(roster) else 0
    if have < league.starters.get(slot, 0):
        return 'starter'
    flex_elig = slot in league.flex_eligible
    surplus = sum(max(0, int((roster['slot_pos'] == p).sum())
                      - league.starters.get(p, 0))
                  for p in league.flex_eligible) if len(roster) else 0
    if flex_elig and surplus < league.flex:
        return 'flex'
    return 'depth'


def recommend(board, state, league, n=5):
    avail = available(board, state)
    roster = my_roster(board, state)
    rows = []
    for r in avail.head(40).itertuples():
        slot = POS_ALIAS.get(r.pos, r.pos)
        have = int((roster['slot_pos'] == slot).sum()) if len(roster) else 0
        lvl = need_level(roster, r.pos, league)
        clog = have >= MAX_AT_POS.get(slot, 6)
        same_tier = avail[(avail['pos'] == r.pos) & (avail['tier'] == r.tier)]
        rows.append(dict(
            pid=r.pid, player=r.player, pos=r.pos, tier=int(r.tier),
            vbd=r.vbd, proj=r.proj_ppr, need=lvl, have=have,
            tier_left=len(same_tier),
            score=round(r.vbd + NEED_BONUS[lvl] - (100 if clog else 0), 1)))
    return pd.DataFrame(rows).sort_values('score', ascending=False).head(n), roster


def show_reco(reco, roster, league):
    if reco.empty:
        print('  board exhausted.')
        return
    print(f'\n  {"":<2}{"player":<22}{"pos":<5}{"tier":<6}{"vbd":>7}'
          f'{"proj":>8}   why')
    for i, r in enumerate(reco.itertuples(), 1):
        why = []
        if r.need == 'starter':
            why.append(f'fills your open {r.pos}{r.have + 1} slot')
        elif r.need == 'flex':
            why.append('flex-eligible, flex open')
        else:
            why.append(f'depth ({r.have} already at {r.pos})')
        if r.tier_left == 1:
            why.append(f'LAST of {r.pos} tier {r.tier}')
        elif r.tier_left <= 3:
            why.append(f'only {r.tier_left} left in {r.pos} tier {r.tier}')
        mark = '>>' if i == 1 else '  '
        print(f'  {mark}{r.player:<22}{r.pos:<5}{r.tier:<6}{r.vbd:>7.1f}'
              f'{r.proj:>8.1f}   {"; ".join(why)}')
    print(f'\n  ranked on VBD + need bonus. Draft value, not weekly points --')
    print(f'  see `lineup` for the points ordering.')


def show_lineup(board, state, league):
    roster = my_roster(board, state)
    if roster.empty:
        print('  roster is empty. use `pick <name>` as you draft.')
        return
    lineup, bench = best_lineup(roster, league)
    print(f'\n  optimal starting lineup by PROJECTED POINTS ({league.label()})')
    total = 0.0
    for slot, r in lineup:
        if r is None:
            print(f'    {slot:<6} -- EMPTY --')
        else:
            total += r.proj_ppr if hasattr(r, 'proj_ppr') else r['proj_ppr']
            name = r.player if hasattr(r, 'player') else r['player']
            pos = r.pos if hasattr(r, 'pos') else r['pos']
            pp = r.proj_ppr if hasattr(r, 'proj_ppr') else r['proj_ppr']
            print(f'    {slot:<6} {name:<22} {pos:<4} {pp:>7.1f}')
    print(f'    {"TOTAL":<6} {"":<22} {"":<4} {total:>7.1f}')
    if len(bench):
        print(f'\n  bench:')
        for r in bench.itertuples():
            print(f'    {"":<6} {r.player:<22} {r.pos:<4} {r.proj_ppr:>7.1f}')
    print(f'\n  NOTE: this ordering is the same in week 1 and week 17. The data')
    print(f'  is season totals with no week column, so real sit/start -- which')
    print(f'  needs matchup, injury and bye -- is not buildable from it.')


HELP = """
  board [n]         top n available on the board          (default 15)
  pos <POS> [n]     top n available at a position
  pick <name>       YOU drafted him
  taken <name>      someone else drafted him
  rec [n]           recommend your next pick              (default 5)
  roster            your team
  lineup            your best starting lineup by points
  undo              undo the last pick/taken
  reset             clear all draft state
  help / quit
"""


def main():
    league = LeagueConfig(teams=TEAMS, **ESPN_STANDARD)
    board = load_board()
    state = load_state()
    print(f'=== 2026 draft assistant -- {league.label()} ===')
    print(f'  {len(board)} players on the board. ROOKIES ARE ABSENT '
          f'(~23% of a real top 250).')
    print(HELP)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(f'> {line}')
        cmd, _, arg = line.partition(' ')
        cmd, arg = cmd.lower(), arg.strip()

        if cmd in ('quit', 'exit', 'q'):
            break
        elif cmd == 'help':
            print(HELP)
        elif cmd == 'board':
            n = int(arg) if arg.isdigit() else 15
            av = available(board, state).head(n)
            print(av[['vbd_rk', 'player', 'pos', 'pos_rk', 'tier', 'age',
                      'proj_ppr', 'vbd']].to_string(index=False))
        elif cmd == 'pos':
            parts = arg.split()
            p = parts[0].upper() if parts else ''
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            av = available(board, state)
            av = av[av['pos'] == p].head(n)
            if av.empty:
                print(f'  nothing available at {p}')
            else:
                print(av[['vbd_rk', 'player', 'pos_rk', 'tier', 'age',
                          'proj_ppr', 'vbd']].to_string(index=False))
        elif cmd in ('pick', 'taken'):
            pid, err = resolve(board, arg, state)
            if err:
                print(f'  {err}')
                continue
            if pid in state['taken'] + state['mine']:
                print('  already off the board.')
                continue
            state['mine' if cmd == 'pick' else 'taken'].append(pid)
            state['log'].append([cmd, pid])
            row = board[board['pid'] == pid].iloc[0]
            who = 'YOU' if cmd == 'pick' else 'someone'
            print(f'  {who} drafted {row["player"]} ({row["pos"]}, '
                  f'vbd_rk {row["vbd_rk"]}, tier {row["tier"]})')
            save_state(state)
        elif cmd in ('rec', 'recommend'):
            n = int(arg) if arg.isdigit() else 5
            reco, roster = recommend(board, state, league, n)
            show_reco(reco, roster, league)
        elif cmd == 'roster':
            r = my_roster(board, state)
            print(r[['player', 'pos', 'tier', 'proj_ppr', 'vbd']]
                  .to_string(index=False) if len(r) else '  (empty)')
        elif cmd == 'lineup':
            show_lineup(board, state, league)
        elif cmd == 'undo':
            if not state['log']:
                print('  nothing to undo.')
                continue
            c, pid = state['log'].pop()
            state['mine' if c == 'pick' else 'taken'].remove(pid)
            print(f'  undid {c} {pid}')
            save_state(state)
        elif cmd == 'reset':
            state = {'taken': [], 'mine': [], 'log': []}
            save_state(state)
            print('  state cleared.')
        else:
            print(f'  unknown command "{cmd}" -- try `help`')


if __name__ == '__main__':
    main()
