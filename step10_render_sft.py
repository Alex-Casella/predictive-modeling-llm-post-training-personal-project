"""Stage 3b -- render draft examples as text for supervised fine-tuning.

step9 produced structured facts. Nothing can train on a dict, so this turns each
example into a chat exchange. It is the last piece before Modal.

FORMAT: model-agnostic `messages` (system / user / assistant).

Deliberately NOT a specific chat template. §11g requires the adapter be trained
against the same base model that serves it, and that base is still undecided
(§15). Baking Llama's <|start_header_id|> or Mistral's [INST] in here would
force the choice now and make it expensive to change. The training script
applies the tokenizer's own chat template to these messages at load time, which
is what every fine-tuning stack expects anyway.

THE ASSISTANT TURN IS FACTUAL, NOT PERSUASIVE.

§11g: "the target advice text actually has to be good -- which means it has to
be generated carefully, since the model will imitate its flaws." A model trained
on confident prose learns to produce confident prose, which is the exact failure
mode §11g names: "a fine-tune that produces more confident-sounding advice
without producing better advice."

So the assistant turn states the pick and cites only numbers that appear
verbatim in the user turn. No adjectives, no narrative, no hindsight. That is
enforced, not intended:

  * every number in the completion must appear in the prompt  (asserted)
  * no hindsight vocabulary may appear at all                 (asserted)

The cost of this choice, stated plainly: the model will learn a template. It
will produce well-grounded, repetitive justifications. That is a floor, not a
ceiling -- a deliberate one, because a grounded template can be improved by a
human later, while hallucinated reasoning baked into 2,850 examples cannot be
removed.

SPLIT BY SEASON, NEVER RANDOMLY.

§11g: "Hold out entire seasons, not random weeks -- random splits leak, since
adjacent weeks of the same player-season are near-duplicates." The same applies
here even more strongly: consecutive picks in one draft share most of their
candidate list, so a random split would put near-identical rows on both sides.

  train  2007-2019    val  2020-2022    test  2023-2025

Test is the most recent seasons, which is the only honest direction for a
model that will be used on 2026.

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import json
import re

import pandas as pd

from checks import Checks

SRC = 'draft_examples.jsonl'
SPLITS = {'train': range(2007, 2020),
          'val': range(2020, 2023),
          'test': range(2023, 2026)}

# Words that could only be written by someone who already knows the outcome.
# §11f: "Any narrative written with hindsight is leakage even when no numeric
# column leaks."
HINDSIGHT = ['would become', 'turned out', 'went on to', 'in hindsight',
             'actually', 'breakout', 'ended up', 'finished the season',
             'we now know', 'famously', 'of course']

SYSTEM = (
    "You are a fantasy football draft assistant for a {league} PPR league. "
    "You are given the drafter's current roster and a shortlist of available "
    "players, each with a projected point total and a value-based-drafting "
    "score (VBD: projected points above the last startable player at that "
    "position). Recommend exactly one player from the shortlist. Cite only "
    "numbers given to you. Do not invent statistics."
)


def render_user(e):
    s = e['situation']
    lines = [f"Round {e['round']}, pick {e['pick']} overall.", '']
    if s['roster']:
        lines.append('Your roster so far:')
        for p in s['roster']:
            lines.append(f"  {p['player']} ({p['pos']}), projected "
                         f"{p['proj_ppr']}")
        counts = ', '.join(f'{v} {k}' for k, v in sorted(s['roster_counts'].items()))
        lines.append(f'  totals: {counts}')
    else:
        lines.append('Your roster is empty. This is your first pick.')
    lines += ['', 'Available players:']
    for a in s['available']:
        lines.append(
            f"  {a['player']} ({a['pos']}) - projected {a['proj_ppr']}, "
            f"VBD {a['vbd']}, tier {a['tier']}, {a['pos']}{a['pos_rk']} "
            f"available, board rank {a['vbd_rk']}, "
            f"finished {a['prev_rk']} last season, age {a['age']}")
    lines += ['', 'Which one should I take, and why?']
    return '\n'.join(lines)


def render_assistant(e):
    """Factual justification. Every number here is copied from the user turn."""
    pick = e['label']['player']
    facts = next(a for a in e['situation']['available']
                 if a['player'] == pick)
    counts = e['situation']['roster_counts']
    slot = 'RB' if facts['pos'] == 'FB' else facts['pos']
    have = counts.get(slot, 0)

    parts = [f"Take {pick} ({facts['pos']})."]
    parts.append(f"Projected {facts['proj_ppr']} points with VBD {facts['vbd']}, "
                 f"tier {facts['tier']}, board rank {facts['vbd_rk']}.")
    if have == 0:
        parts.append(f"You have no {slot} yet, so he fills a starting slot.")
    else:
        parts.append(f"You already have {have} at {slot}.")
    parts.append(f"He finished {facts['prev_rk']} last season and is "
                 f"{facts['age']} this year.")
    return ' '.join(parts)


def numbers(text):
    """Numeric tokens, without swallowing sentence-ending punctuation.

    `-?\\d+\\.?\\d*` looks right and is not: on "board rank 12." it captures
    "12." because the trailing period matches the optional decimal point, so it
    fails to match the "12" in the prompt. A decimal point only counts when a
    digit follows it.
    """
    return set(re.findall(r'-?\d+(?:\.\d+)?', text))


def main():
    rows = [json.loads(l) for l in open(SRC)]
    c = Checks('stage 3b -- SFT rendering')
    print(f'read {SRC}: {len(rows)} examples')

    out = {k: [] for k in SPLITS}
    for e in rows:
        split = next((k for k, yrs in SPLITS.items()
                      if e['season'] in yrs), None)
        assert split, f"season {e['season']} is in no split"
        user, assistant = render_user(e), render_assistant(e)
        out[split].append(dict(
            season=e['season'], pick=e['pick'], round=e['round'],
            margin=e['margin'], board_rank=e['board_rank'],
            board_pick=e['board_pick']['player'],
            answer=e['label']['player'],
            messages=[
                dict(role='system',
                     content=SYSTEM.format(league=e['situation']['league'])),
                dict(role='user', content=user),
                dict(role='assistant', content=assistant)]))

    # -- grounding: every number in the answer must be in the question -------
    bad_num, bad_word = [], []
    for split, items in out.items():
        for it in items:
            u = it['messages'][1]['content']
            a = it['messages'][2]['content']
            missing = numbers(a) - numbers(u)
            if missing:
                bad_num.append((split, it['pick'], missing))
            low = a.lower()
            if any(w in low for w in HINDSIGHT):
                bad_word.append((split, it['pick']))
    c.check('every number in the completion appears in the prompt',
            not bad_num, f'{len(bad_num)} violations: {bad_num[:3]}')
    c.check('no hindsight vocabulary in any completion',
            not bad_word, f'{len(bad_word)} violations')
    c.check('the recommended player is always on the shortlist',
            all(it['answer'] in it['messages'][1]['content']
                for items in out.values() for it in items))

    # -- split integrity ----------------------------------------------------
    seasons = {k: sorted({it['season'] for it in v}) for k, v in out.items()}
    c.check('no season appears in two splits',
            len(set().union(*[set(v) for v in seasons.values()]))
            == sum(len(v) for v in seasons.values()))
    c.check('test is the most recent seasons',
            min(seasons['test']) > max(seasons['train']))

    print('\n--- splits ---')
    for k in SPLITS:
        items = out[k]
        chars = [len(it['messages'][1]['content']) for it in items]
        print(f"  {k:<6} {len(items):>5} examples   seasons "
              f"{seasons[k][0]}-{seasons[k][-1]}   "
              f"prompt chars: median {int(pd.Series(chars).median())}, "
              f"max {max(chars)}")
        with open(f'sft_{k}.jsonl', 'w') as f:
            for it in items:
                f.write(json.dumps(it) + '\n')

    print('\n--- how often the answer differs from the board (i.e. there is')
    print('    something to learn beyond re-reading the top of the list) ---')
    for k in SPLITS:
        items = out[k]
        diff = sum(1 for it in items if it['answer'] != it['board_pick'])
        print(f'  {k:<6} {100 * diff / len(items):.1f}% of answers are NOT the '
              f'board\'s own pick')

    print('\n--- one rendered example ---')
    ex = out['train'][3]
    for m in ex['messages']:
        print(f"\n[{m['role'].upper()}]\n{m['content']}")

    print(f"\nwrote sft_train.jsonl / sft_val.jsonl / sft_test.jsonl")
    c.report()


if __name__ == '__main__':
    main()
