"""Stage 8 -- put the opponent in the prompt, not just its name.

THE HOLE THIS FILLS

The start/sit prompt names the opponent and then says nothing about it:

    Calvin Johnson (WR) vs CHI - 26.0 points per game over 4 games, 25.8 ...
                        ^^^^^^ a label carrying no information

Every other number in that prompt describes the PLAYER, and the deterministic
rule sorts on one of them. So the model is being asked to disagree with a sort
using only the sort's own inputs. There is nothing to disagree ON.

Defensive strength is the cheapest thing available that the rule cannot see. It
lives entirely in weekly_ppr.csv, which already carries `opp`.

WHY A SEPARATE FILE AND A SEPARATE DATASET

step12 and step13 produced sit_train/val/test.jsonl, and 2.99, 3.04, 3.12 and
3.02 are all measured against them. Regenerating those files would leave every
one of those numbers describing a dataset that no longer exists. This writes
sitm_*.jsonl alongside them and touches nothing else.

start_sit_examples.jsonl is also left alone -- eval_agent.py joins the answer
key on (season, pick), so both datasets score against the same key and their
results are directly comparable.

THE LOOK-AHEAD RULE IS THE WHOLE GAME HERE

"How many points has CHI allowed to WRs" is a number that CHANGES as the season
goes on. Computing it over the full season and pasting it into a week-5 prompt
tells the model what CHI's defense did in weeks 6-17, which is hindsight
wearing a matchup's clothes.

    week 5 prompt  ->  weeks 1-4 ONLY
    week 12 prompt ->  weeks 1-11 ONLY

Asserted below, per decision, not assumed.

WHY A RANK AND NOT JUST A NUMBER

"CHI has allowed 28.4 points per game to WRs" means nothing without knowing
what average looks like. The prompt gives the rank alongside it so the figure
is interpretable from the prompt alone, which is the same standard every other
field here is held to.

Run:
    python3 step15_matchup.py            # writes sitm_train/val/test.jsonl

Then, BEFORE spending any GPU, find out whether the field is usable at all:
    python3 eval_agent.py --model llama3.1:8b \
        --test sitm_test.jsonl --key start_sit_examples.jsonl

Compare that against 3.02, the un-tuned score on the prompt without this field.
If the base model cannot use a clean numeric matchup signal handed to it
directly, a fine-tune is unlikely to, and no retrieval pipeline over messy
prose will either.

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import json

import numpy as np
import pandas as pd

from checks import Checks
from step10_render_sft import HINDSIGHT, numbers
from step13_render_startsit import SPLITS, render_assistant

SRC = 'start_sit_examples.jsonl'
WEEKLY = 'weekly_ppr.csv'
PREFIX = 'sitm'

SYSTEM = (
    "You are a fantasy football start/sit assistant for a full-PPR league. "
    "You are given the week, the lineup slot being filled, and the players on "
    "the drafter's roster who are eligible for it. For each you get points per "
    "game so far this season, their average over the last three games, how "
    "many games they have played, this week's opponent, and how many points "
    "that opponent has allowed to the player's position so far this season. "
    "Recommend exactly one player to start. Cite only numbers given to you. "
    "Do not invent statistics."
)


def defense_table(weekly):
    """(season, week, opp, pos) -> (ppg allowed, rank, teams ranked).

    Points allowed is summed over every player of that position who faced that
    defense, divided by the number of distinct weeks it has played. A defense
    with a bye has played fewer weeks, so dividing by the week number would
    quietly flatter it.

    The value at key `week` uses weeks STRICTLY BEFORE `week`. That is the
    no-look-ahead rule, applied at the point the number is built rather than
    checked for afterwards.
    """
    # Summed straight from the raw player-week rows, NOT from a pre-aggregated
    # per-week table. Adding floats in two different orders gives answers that
    # differ around 1e-15, which is invisible everywhere except on a rounding
    # boundary -- and 2 of 200 sampled values sat on one, so the two paths
    # displayed 34.8 and 34.9 for the same defense. One code path removes the
    # disagreement instead of widening the tolerance until it hides.
    w = weekly[weekly['opp'].notna()].copy()

    out = {}
    for (season, pos), grp in w.groupby(['season', 'pos']):
        weeks = sorted(grp['week'].unique())
        for target in weeks:
            prior = grp[grp['week'] < target]
            if prior.empty:
                continue
            # Per defense: total conceded / weeks that defense actually played.
            agg = prior.groupby('opp').agg(total=('ppr', 'sum'),
                                           played=('week', 'nunique'))
            # STORED UNROUNDED, rounded once in render_user.
            #
            # Rounding here made the table impossible to verify. Summing the
            # same floats in a different order moves the total by ~1e-15, which
            # is invisible until a value sits on a .x5 boundary -- then the two
            # orders round to 34.8 and 34.9 and a correctness check fails over
            # a display detail. Rounding at the last moment makes the check
            # compare arithmetic instead.
            #
            # Ranking on the unrounded figure is also better on its own terms:
            # two defenses that both display 25.4 still get distinct ranks.
            agg['ppg'] = agg['total'] / agg['played']
            # Rank 1 = most generous, which is the one a manager wants to
            # start against. Stated in the prompt so the direction is not
            # something the model has to infer.
            agg['rank'] = agg['ppg'].rank(ascending=False,
                                          method='min').astype(int)
            n = len(agg)
            for team, row in agg.iterrows():
                out[(season, target, team, pos)] = (row['ppg'],
                                                    int(row['rank']), n)
    return out


def render_user(e, table):
    s = e['situation']
    lines = [f"Week {e['week']}. Filling your {s['slot']} slot.", '',
             'Eligible players on your roster:']
    for c in s['candidates']:
        l3 = 'no 3-game average yet' if c['last3'] is None \
            else f"{c['last3']} over his last 3"
        d = table.get((e['season'], e['week'], c['opp'], c['pos']))
        if d is None:
            # Said in words, never dropped. A missing field the model cannot
            # see is a field it learns to assume is average.
            matchup = f"no {c['pos']} data on {c['opp']} yet this season"
        else:
            ppg, rank, n = d
            matchup = (f"{c['opp']} has allowed {ppg:.1f} per game to "
                       f"{c['pos']}s, {rank} most of {n}")
        lines.append(
            f"  {c['player']} ({c['pos']}) vs {c['opp']} - "
            f"{c['ppg']} points per game over {c['games']} games, {l3}; "
            f"{matchup}")
    lines += ['', 'Which one should I start, and why?']
    return '\n'.join(lines)


def main():
    rows = [json.loads(l) for l in open(SRC)]
    weekly = pd.read_csv(WEEKLY)
    c = Checks('stage 8 -- opponent strength in the start/sit prompt')
    print(f'read {SRC}: {len(rows)} decisions')
    print(f'read {WEEKLY}: {len(weekly):,} player-weeks')

    table = defense_table(weekly)
    print(f'built {len(table):,} (season, week, defense, position) entries')

    # -- THE LOOK-AHEAD CHECK, before anything is written -------------------
    # Rebuilt the slow, obvious way from the raw weekly file, over a SAMPLE
    # rather than one probe. A single probe cannot distinguish "correct" from
    # "coincidentally equal", and on the first run it did exactly that: the
    # value including the current week happened to match to one decimal, so
    # the leak test passed while proving nothing.
    #
    # Rounding is applied with numpy on both sides. Comparing numpy's
    # round-half-to-even against Python's built-in round produced a spurious
    # 34.8 vs 34.9 failure -- a disagreement about the rounder, not about the
    # aggregation the check exists to verify.
    keys = [(e['season'], e['week'], cd['opp'], cd['pos'])
            for e in rows for cd in e['situation']['candidates']]
    keys = [k for k in dict.fromkeys(keys) if k in table][:200]

    # Compared UNROUNDED, at 1e-6. Comparing the displayed one-decimal figure
    # tests the rounder, not the aggregation: two orders of summation differ
    # by ~1e-15, which only shows up when a value sits on a .x5 boundary, and
    # then a correct table fails a correctness check. 1e-6 is far below
    # anything that could reach the prompt and far above float noise.
    mismatch, would_change = [], 0
    for season, week, opp, pos in keys:
        w = weekly[(weekly['season'] == season) & (weekly['opp'] == opp)
                   & (weekly['pos'] == pos)]
        before = w[w['week'] < week]
        manual = before['ppr'].sum() / before['week'].nunique()
        if abs(manual - table[(season, week, opp, pos)][0]) > 1e-6:
            mismatch.append((season, week, opp, pos, round(manual, 4),
                             round(table[(season, week, opp, pos)][0], 4)))
        incl = w[w['week'] <= week]
        if abs(incl['ppr'].sum() / incl['week'].nunique() - manual) > 1e-6:
            would_change += 1

    c.check(f'all {len(keys)} sampled values match a hand-rebuilt figure',
            not mismatch, f'{len(mismatch)} mismatches: {mismatch[:3]}')
    # If including the current week changed almost nothing, the check above is
    # not actually detecting look-ahead and should not be trusted as if it
    # were. 60% is a floor on "this test has teeth", not a data property.
    c.check('the prior-weeks-only rule is doing real work',
            would_change >= 0.60 * len(keys),
            f'including the current week changed only {would_change} of '
            f'{len(keys)} sampled values, so a leak would be hard to see')
    print(f'  {would_change} of {len(keys)} sampled values would change if the '
          f'current week were included')

    out = {k: [] for k in SPLITS}
    for e in rows:
        split = next((k for k, yrs in SPLITS.items()
                      if e['season'] in yrs), None)
        if split is None:
            continue
        out[split].append(dict(
            season=e['season'], week=e['week'], seat=e['seat'],
            pick=e['pick'], margin=e['margin'],
            board_rank=e['baseline_rank'], board_pick=e['baseline_pick'],
            answer=e['label']['player'],
            messages=[
                dict(role='system', content=SYSTEM),
                dict(role='user', content=render_user(e, table)),
                # Unchanged from step13. The ANSWER TEXT is deliberately not
                # taught to cite the matchup: this dataset exists to test
                # whether a model can USE the field, and writing the field
                # into the target would train it to recite the field instead.
                dict(role='assistant', content=render_assistant(e))]))

    # -- the same grounding checks step13 runs ------------------------------
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

    # -- comparability with the dataset it will be measured against ---------
    base = {}
    for k in SPLITS:
        try:
            base[k] = [json.loads(l) for l in open(f'sit_{k}.jsonl')]
        except FileNotFoundError:
            base[k] = None
    if all(v is not None for v in base.values()):
        c.check('same example count as sit_*.jsonl',
                all(len(out[k]) == len(base[k]) for k in SPLITS),
                {k: (len(out[k]), len(base[k])) for k in SPLITS})
        c.check('same (season, pick) in the same order',
                all([(i['season'], i['pick']) for i in out[k]]
                    == [(i['season'], i['pick']) for i in base[k]]
                    for k in SPLITS),
                'paired_test.py aligns row-for-row; a reorder breaks pairing')

    covered = sum('has allowed' in it['messages'][1]['content']
                  for items in out.values() for it in items)
    total = sum(len(v) for v in out.values())
    print(f'\nprompts carrying at least one matchup figure: '
          f'{covered:,} of {total:,} ({100 * covered / total:.1f}%)')

    print('\n--- splits ---')
    for k in SPLITS:
        items = out[k]
        chars = [len(i['messages'][1]['content']) for i in items]
        seasons = sorted({i['season'] for i in items})
        print(f"  {k:<6} {len(items):>5} examples   seasons "
              f"{seasons[0]}-{seasons[-1]}   "
              f"prompt chars: median {int(pd.Series(chars).median())}, "
              f"max {max(chars)}")
        with open(f'{PREFIX}_{k}.jsonl', 'w') as f:
            for i in items:
                f.write(json.dumps(i) + '\n')

    print(f'\nwrote {PREFIX}_train/val/test.jsonl. sit_*.jsonl untouched.')
    print('\nsample prompt:')
    print('  ' + out['test'][0]['messages'][1]['content'].replace('\n', '\n  '))
    print('\nnext, and it needs NO GPU:')
    print(f'  python3 eval_agent.py --model llama3.1:8b \\')
    print(f'      --test {PREFIX}_test.jsonl --key {SRC}')
    print('  compare against 3.02, the un-tuned score without this field.')
    c.report()


if __name__ == '__main__':
    main()
