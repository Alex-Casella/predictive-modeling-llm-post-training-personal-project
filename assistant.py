"""Layer 4 -- the draft assistant, backed by the fine-tuned model.

`draft.py` is the deterministic assistant: it sorts by VBD plus a need bonus and
shows you the top of the list. This file asks the language model the same
question and puts the two answers side by side.

WHY THE PROMPT IS IMPORTED, NOT REWRITTEN

A LoRA adapter learns the shape of the prompt it was trained on as much as the
task inside it. `step10_render_sft.render_user()` produced all 2,850 training
prompts; if this file rendered its own "equivalent" prompt, every drift -- a
different field order, "proj" instead of "projected", a missing age -- would
silently move the served input away from the trained distribution, and the
damage would look like a bad fine-tune rather than a formatting bug.

So the situation dict is built in exactly step9's shape and handed to step10's
own renderer. There is one prompt format in this repo and it lives in one file.
The same applies to `extract_pick`, imported from `eval_agent.py`: the rule that
decided the published 5.41 is the rule that decides what the model "said" here.

WHAT THIS CANNOT TELL YOU

`eval_agent.py` scores picks because 2023-2025 have been played and the answer
key exists. Nothing here can be scored -- board_2026.csv projects a season whose
outcome is not known. This tool shows you what the model recommends and where it
disagrees with the board. It cannot tell you which one is right, and it never
claims to.

Usage:
    python3 assistant.py                          # fine-tuned model
    python3 assistant.py --model llama3.1:8b      # the un-tuned base
    python3 assistant.py --compare llama3.1:8b    # both, side by side

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import argparse
import sys
import urllib.error

# Everything below is imported rather than reimplemented, so that this file
# cannot disagree with the pipeline that produced the published numbers.
from draft import (available, load_board, load_state, my_roster, resolve,
                   save_state, show_lineup)
from eval_agent import OLLAMA, ask_ollama, extract_pick
from step9_draft_examples import SHOW_AVAILABLE, TEAMS, legal
from step10_render_sft import SYSTEM, render_user
from vbd import ESPN_STANDARD, POS_ALIAS, LeagueConfig

DEFAULT_MODEL = 'fantasy-draft'


def roster_counts(roster):
    """Position counts keyed by SLOT, not raw position.

    step9 applies POS_ALIAS before counting (a FB occupies an RB slot), and
    render_assistant() looks the count up by slot. Counting raw positions here
    would produce a roster summary the model was never trained to read.
    """
    counts = {}
    for r in roster.itertuples():
        slot = POS_ALIAS.get(r.pos, r.pos)
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def shortlist(board, state, counts):
    """The K candidates the model is allowed to choose from.

    Same filter and same K as training: board order, minus anyone already gone,
    minus positions where you are already at the roster cap. The action space
    being exactly what is shown is the fix step9 documents at length -- widening
    it here would ask the model a question it was never trained on.
    """
    avail = available(board, state)
    ok = avail[avail['pos'].apply(lambda p: legal(counts, p))]
    return ok.head(SHOW_AVAILABLE)


def build_situation(board, state, league):
    """A live draft, in exactly the dict shape step9 wrote to disk."""
    roster = my_roster(board, state)
    counts = roster_counts(roster)
    shown = shortlist(board, state, counts)
    return dict(
        league=league.label(),
        roster=[dict(player=r.player, pos=r.pos, proj_ppr=float(r.proj_ppr))
                for r in roster.itertuples()],
        roster_counts=counts,
        available=[dict(player=r.player, pos=r.pos, tier=int(r.tier),
                        pos_rk=int(r.pos_rk), vbd_rk=int(r.vbd_rk),
                        proj_ppr=float(r.proj_ppr), vbd=float(r.vbd),
                        prev_rk=int(r.prev_rk), age=int(r.age))
                   for r in shown.itertuples()]), shown


def build_messages(board, state, league):
    """(messages, shown, pick_no, rnd) -- the exact payload the model sees."""
    situation, shown = build_situation(board, state, league)

    # The training prompts opened with "Round R, pick P overall", so something
    # has to fill those slots. Derived from how many players are off the board,
    # which is only correct if every rival pick was entered with `taken`. Stated
    # at startup rather than assumed silently.
    pick_no = len(state['taken']) + len(state['mine']) + 1
    rnd = (pick_no - 1) // league.teams + 1

    example = dict(round=rnd, pick=pick_no, situation=situation)
    return (dict(messages=[
        dict(role='system', content=SYSTEM.format(league=situation['league'])),
        dict(role='user', content=render_user(example))]),
        shown, pick_no, rnd)


def ask(model, item, names):
    """Send one situation to Ollama. Returns (pick_or_None, raw_text)."""
    try:
        text = ask_ollama(model, item, timeout=180)
    except urllib.error.URLError as e:
        print(f'\n  cannot reach Ollama at {OLLAMA}: {e}')
        print(f'  start it with `ollama serve`, and check `ollama list` shows '
              f'"{model}"')
        return None, None
    return extract_pick(text, names), text


def show_answer(model, pick, text, shown):
    print(f'\n  --- {model} ---')
    if pick is None:
        # Not a wrong pick, a broken one. eval_agent.py refuses to map a
        # near-miss onto a real player and so does this; softening it here
        # would hide the one failure mode a language layer has that a sort
        # does not.
        print('  INVALID: named no player from the shortlist.')
        print('  raw response:')
        for line in (text or '').strip().splitlines():
            print(f'    {line}')
        return
    row = shown[shown['player'] == pick].iloc[0]
    print(f'  picks {pick} ({row["pos"]})  -- board rank {row["vbd_rk"]}, '
          f'tier {row["tier"]}, VBD {row["vbd"]:.1f}')
    for line in (text or '').strip().splitlines():
        print(f'    {line}')


def recommend(board, state, league, models):
    item, shown, pick_no, rnd = build_messages(board, state, league)
    if shown.empty:
        print('  no legal candidates left -- board exhausted or roster capped.')
        return
    names = list(shown['player'])
    board_pick = shown.iloc[0]

    print(f'\n=== round {rnd}, pick {pick_no} overall '
          f'-- {len(shown)} candidates ===')
    print(f'\n  the board (draft.py, no LLM)')
    print(f'  picks {board_pick["player"]} ({board_pick["pos"]})  '
          f'-- board rank {board_pick["vbd_rk"]}, tier {board_pick["tier"]}, '
          f'VBD {board_pick["vbd"]:.1f}')

    picks = {}
    for m in models:
        p, text = ask(m, item, names)
        picks[m] = p
        show_answer(m, p, text, shown)

    valid = [p for p in picks.values() if p]
    if valid:
        print()
        if all(p == board_pick['player'] for p in valid):
            print('  everyone agrees with the board.')
        elif len(set(valid)) == 1 and len(valid) > 1:
            print(f'  the models agree with each other and DISAGREE with the '
                  f'board.')
        elif len(set(valid)) > 1:
            print(f'  the models disagree with each other.')
        else:
            print(f'  the model DISAGREES with the board.')
    print(f'\n  Neither answer can be scored: board_2026.csv projects a season')
    print(f'  whose outcome is unknown. eval_agent.py scores 2023-2025, where')
    print(f'  the answer key exists.')


HELP = """
  rec               ask the model for your next pick
  prompt            print the exact prompt that would be sent
  pick <name>       YOU drafted him
  taken <name>      someone else drafted him
  board [n]         top n available                        (default 12)
  roster            your team
  lineup            best starting lineup by projected points
  undo              undo the last pick/taken
  reset             clear all draft state
  help / quit
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help=f'Ollama model to ask (default {DEFAULT_MODEL})')
    ap.add_argument('--compare', default=None, metavar='MODEL',
                    help='also ask a second model and show both answers')
    args = ap.parse_args()
    models = [args.model] + ([args.compare] if args.compare else [])

    league = LeagueConfig(teams=TEAMS, **ESPN_STANDARD)
    board = load_board()
    state = load_state()

    print(f'=== 2026 draft assistant, LLM-backed -- {league.label()} ===')
    print(f'  asking: {", ".join(models)}   (local, via Ollama -- no API, '
          f'no cost)')
    print(f'  {len(board)} players on the board, {SHOW_AVAILABLE} shown per '
          f'question.')
    print(f'  ROOKIES ARE ABSENT (~23% of a real top 250).')
    print(f'  Round and pick number are derived from how many players are off')
    print(f'  the board, so enter rivals\' picks with `taken` to keep them right.')
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
        elif cmd in ('rec', 'recommend', 'ask'):
            recommend(board, state, league, models)
        elif cmd == 'prompt':
            item, shown, pick_no, rnd = build_messages(board, state, league)
            for m in item['messages']:
                print(f'\n[{m["role"].upper()}]\n{m["content"]}')
        elif cmd == 'board':
            n = int(arg) if arg.isdigit() else SHOW_AVAILABLE
            av = available(board, state).head(n)
            print(av[['vbd_rk', 'player', 'pos', 'pos_rk', 'tier', 'age',
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
