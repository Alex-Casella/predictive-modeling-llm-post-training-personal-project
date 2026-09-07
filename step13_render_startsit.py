"""Stage 6 step 3 -- render start/sit decisions as chat examples for SFT.

The sit/start analogue of step10. Everything that file argues for applies here
unchanged, so the reasoning is not repeated -- read step10 first. What follows
is only what is DIFFERENT, and why.

WHAT A TRAINING EXAMPLE IS

    system      you are a start/sit assistant, cite only what you are given
    user        week, slot, and K candidates with pre-week form
    assistant   "Start X. <numbers copied from the prompt>"

SPLIT BY SEASON, AND THE SPLIT IS TIGHTER HERE THAN IN step10

Ten decisions in one week come from ten rosters drawn from the same draft, so
they share candidates constantly -- Calvin Johnson sits on one roster and
appears in every week of that season. A random split would put week 6 in train
and week 7 in test for the same player with almost the same numbers.

    train  2011-2020    val  2021-2022    test  2023-2024

Test is the most recent seasons, the only honest direction.

LAST3 IS INCLUDED EVEN THOUGH IT IS THE WORSE SIGNAL

step12 measured season-to-date PPG at 2.71 and last-3 at 2.91, and the
correlation study agreed (0.389 vs 0.377). It would be tempting to drop last-3
so the model cannot be misled by it.

That would be cheating in a way worth naming: a real manager sees recent form,
and a model that has never been shown a hot streak cannot demonstrate that it
correctly ignores one. Leaving it in makes "does it chase recency?" an
answerable question about the trained model rather than a property engineered
into the prompt.

THE ASSISTANT TURN CITES ONLY WHAT THE PROMPT CONTAINS

Same two assertions as step10, same reason: a model trained on confident prose
learns to produce confident prose. Every number in the answer must appear in
the question, and no hindsight vocabulary is allowed anywhere.

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import json
import re

import pandas as pd

from checks import Checks
from step10_render_sft import HINDSIGHT, numbers
from step12_start_sit_examples import SHOW

SRC = 'start_sit_examples.jsonl'
SPLITS = {'train': range(2011, 2021),
          'val': range(2021, 2023),
          'test': range(2023, 2025)}

SYSTEM = (
    "You are a fantasy football start/sit assistant for a full-PPR league. "
    "You are given the week, the lineup slot being filled, and the players on "
    "the drafter's roster who are eligible for it. For each you get points per "
    "game so far this season, their average over the last three games, how "
    "many games they have played, and this week's opponent. Recommend exactly "
    "one player to start. Cite only numbers given to you. Do not invent "
    "statistics."
)


def render_user(e):
    s = e['situation']
    lines = [f"Week {e['week']}. Filling your {s['slot']} slot.", '',
             'Eligible players on your roster:']
    for c in s['candidates']:
        # "no data" rather than "None": a player with fewer than three games
        # has no three-game average, and the prompt should say so in words the
        # model can learn to read, not leak a Python repr.
        l3 = 'no 3-game average yet' if c['last3'] is None \
            else f"{c['last3']} over his last 3"
        lines.append(
            f"  {c['player']} ({c['pos']}) vs {c['opp']} - "
            f"{c['ppg']} points per game over {c['games']} games, {l3}")
    lines += ['', 'Which one should I start, and why?']
    return '\n'.join(lines)


def render_assistant(e):
    """Factual justification, every number copied from the user turn."""
    pick = e['label']['player']
    f = next(c for c in e['situation']['candidates'] if c['player'] == pick)
    parts = [f"Start {pick} ({f['pos']})."]
    parts.append(f"He is averaging {f['ppg']} points per game over "
                 f"{f['games']} games and faces {f['opp']} this week.")
    if f['last3'] is None:
        parts.append('He does not have three games yet, so the season average '
                     'is all there is to go on.')
    elif f['last3'] > f['ppg']:
        parts.append(f"His last three games average {f['last3']}, above his "
                     f"season figure of {f['ppg']}.")
    elif f['last3'] < f['ppg']:
        parts.append(f"His last three games average {f['last3']}, below his "
                     f"season figure of {f['ppg']}, but the season average is "
                     f"the larger sample.")
    else:
        parts.append(f"His last three games average {f['last3']}, the same as "
                     f"his season figure.")
    return ' '.join(parts)


def main():
    rows = [json.loads(l) for l in open(SRC)]
    c = Checks('stage 6 step 3 -- start/sit SFT rendering')
    print(f'read {SRC}: {len(rows)} decisions')

    out = {k: [] for k in SPLITS}
    for e in rows:
        split = next((k for k, yrs in SPLITS.items()
                      if e['season'] in yrs), None)
        if split is None:
            continue
        out[split].append(dict(
            season=e['season'], week=e['week'], seat=e['seat'],
            # eval_agent.py joins its answer key on (season, pick). Reusing
            # that harness means giving it a `pick` -- here a decision is
            # identified by week and seat, packed into one integer so the join
            # stays a single column and cannot half-match.
            pick=e['week'] * 100 + e['seat'],
            margin=e['margin'],
            board_rank=e['baseline_rank'],
            board_pick=e['baseline_pick'],
            answer=e['label']['player'],
            messages=[
                dict(role='system', content=SYSTEM),
                dict(role='user', content=render_user(e)),
                dict(role='assistant', content=render_assistant(e))]))

    # -- grounding ----------------------------------------------------------
    bad_num, bad_word = [], []
    for split, items in out.items():
        for it in items:
            u, a = it['messages'][1]['content'], it['messages'][2]['content']
            missing = numbers(a) - numbers(u)
            if missing:
                bad_num.append((split, it['pick'], missing))
            if any(w in a.lower() for w in HINDSIGHT):
                bad_word.append((split, it['pick']))
    c.check('every number in the completion appears in the prompt',
            not bad_num, f'{len(bad_num)} violations: {bad_num[:3]}')
    c.check('no hindsight vocabulary in any completion',
            not bad_word, f'{len(bad_word)} violations')
    c.check('the recommended player is always on the shortlist',
            all(it['answer'] in it['messages'][1]['content']
                for items in out.values() for it in items))

    # -- split integrity -----------------------------------------------------
    seasons = {k: sorted({it['season'] for it in v}) for k, v in out.items()}
    c.check('no season appears in two splits',
            len(set().union(*[set(v) for v in seasons.values()]))
            == sum(len(v) for v in seasons.values()))
    c.check('test is the most recent seasons',
            min(seasons['test']) > max(seasons['train']))
    c.check('(season, pick) is unique within every split',
            all(len({(i['season'], i['pick']) for i in v}) == len(v)
                for v in out.values()),
            'eval_agent.py joins on this pair; a collision silently mislabels')

    print('\n--- splits ---')
    for k in SPLITS:
        items = out[k]
        chars = [len(i['messages'][1]['content']) for i in items]
        print(f"  {k:<6} {len(items):>5} examples   seasons "
              f"{seasons[k][0]}-{seasons[k][-1]}   "
              f"prompt chars: median {int(pd.Series(chars).median())}, "
              f"max {max(chars)}")
        with open(f'sit_{k}.jsonl', 'w') as f:
            for i in items:
                f.write(json.dumps(i) + '\n')

    print('\n--- how often the answer differs from the naive rule (i.e. there')
    print('    is something to learn beyond sorting by season average) ---')
    for k in SPLITS:
        items = out[k]
        diff = sum(1 for i in items if i['answer'] != i['board_pick'])
        print(f'  {k:<6} {100 * diff / len(items):.1f}% of answers are NOT '
              f'"start the best average"')

    print('\n--- one rendered example ---')
    for m in out['train'][0]['messages']:
        print(f"\n[{m['role'].upper()}]\n{m['content']}")

    print('\nwrote sit_train.jsonl / sit_val.jsonl / sit_test.jsonl')
    c.report()


if __name__ == '__main__':
    main()
