"""Stage 9 -- a wider test window, because 278 examples cannot resolve anything.

Every start/sit comparison in this project came back null, and the sample sizes
80% power would have required were:

    1 epoch vs 2 epochs          0.126 gap    needed   856    have 278
    matchup field in the prompt  0.097        needed 1,810    have 278
    pad token fixed              0.014        needed 24,081   have 278
    fine-tuned vs un-tuned base  0.097        needed 3,065    have 278

Five nulls is not five findings about football. It is one finding about the
test set, and this file is the only response to it that does not involve
pretending otherwise.

WHAT CHANGES AND WHAT DOES NOT

Only the split boundaries. Same decisions, same 6 candidates, same hindsight
labels, same generator -- `start_sit_examples.jsonl` is not regenerated, so the
answer key is untouched and old and new results describe the same underlying
data on different spans.

    current   train 2011-2020   val 2021-2022   test 2023-2024    278
    wider     train 2011-2018   val 2019-2020   test 2021-2024    541

Written to `sitw_*.jsonl`. `sit_*.jsonl` is left alone: 2.92, 3.02, 2.99, 3.04,
2.98 and 3.12 are all measured against it and would otherwise describe a file
that no longer exists.

THE HONEST ARITHMETIC, BEFORE ANYONE GETS EXCITED

Doubling n shrinks the standard error by sqrt(2), not by 2. The smallest effect
detectable at 80% power moves from about 0.19 ranks to about 0.135 -- computed
below from the standard deviations actually measured, not assumed. That brings
the epoch gap (0.126) close to resolvable and leaves the pad-token gap (0.014)
as far out of reach as it ever was.

It also costs two training seasons and every bar has to be re-measured on the
new span: board, random, un-tuned base, and each adapter. At roughly 6.5 s per
LLM call that is about an hour per model.

THE LEVER THAT LOOKS TEMPTING AND IS WRONG

step12 simulates one 10-team league per season. Simulating five leagues per
season with different draft orders would yield five times the decisions, and
almost none of the extra rows would be new information: the same players, the
same weeks, the same outcomes, re-dealt into different rosters. n would rise,
every standard error would shrink, and every p-value would improve for a reason
that has nothing to do with the world. The information here is bounded by the
number of distinct player-weeks, and no resampling scheme adds to it.

Run:
    python3 step19_widen_test.py

Data sources: Pro-Football-Reference and nflverse. See ATTRIBUTION.md.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from checks import Checks
from step10_render_sft import HINDSIGHT, numbers
from step13_render_startsit import SRC, SYSTEM, render_assistant, render_user

PREFIX = 'sitw'
SPLITS = {'train': range(2011, 2019),
          'val': range(2019, 2021),
          'test': range(2021, 2025)}

# Standard deviations of the paired differences actually observed, so the power
# table below is arithmetic on this project's own numbers rather than a
# textbook example. Sources are the paired_test.py runs in RESULTS.md.
OBSERVED = [
    ('1 epoch vs 2 epochs', 0.126, 1.314),
    ('matchup field in the prompt', 0.097, 1.475),
    ('fine-tuned vs un-tuned base', 0.097, 1.313),
    ('pad token fixed', 0.014, 0.797),
]


def need(effect, sd, power=0.80, alpha=0.05):
    """Paired-sample n for `power` at `alpha`. Same formula as paired_test.py."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) ** 2 * sd ** 2 / effect ** 2


def detectable(n, sd, power=0.80, alpha=0.05):
    """Smallest effect `n` examples can resolve -- the inverse question."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return (za + zb) * sd / n ** 0.5


def main():
    rows = [json.loads(l) for l in open(SRC)]
    c = Checks('stage 9 -- wider test window')
    print(f'read {SRC}: {len(rows)} decisions, '
          f'{min(r["season"] for r in rows)}-{max(r["season"] for r in rows)}')

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
                dict(role='user', content=render_user(e)),
                dict(role='assistant', content=render_assistant(e))]))

    # -- the same guarantees step13 asserts, re-asserted here ---------------
    bad_num, bad_word = [], []
    for split, items in out.items():
        for it in items:
            u, a = it['messages'][1]['content'], it['messages'][2]['content']
            if numbers(a) - numbers(u):
                bad_num.append((split, it['pick']))
            if any(w in a.lower() for w in HINDSIGHT):
                bad_word.append((split, it['pick']))
    c.check('every number in the completion appears in the prompt', not bad_num,
            f'{len(bad_num)} violations')
    c.check('no hindsight vocabulary in any completion', not bad_word,
            f'{len(bad_word)} violations')
    c.check('no season appears in two splits',
            len({s for v in out.values() for s in {i["season"] for i in v}})
            == sum(len({i['season'] for i in v}) for v in out.values()))
    c.check('test is the most recent seasons',
            min(i['season'] for i in out['test'])
            > max(i['season'] for i in out['train']))
    c.check('(season, pick) is unique within every split',
            all(len({(i['season'], i['pick']) for i in v}) == len(v)
                for v in out.values()),
            'eval_agent.py joins on this pair')
    c.check('every decision is used exactly once',
            sum(len(v) for v in out.values()) == len(rows))

    print('\n--- splits ---')
    for k in SPLITS:
        items = out[k]
        seasons = sorted({i['season'] for i in items})
        print(f'  {k:<6} {len(items):>5} examples   '
              f'seasons {seasons[0]}-{seasons[-1]}')
        with open(f'{PREFIX}_{k}.jsonl', 'w') as f:
            for i in items:
                f.write(json.dumps(i) + '\n')

    n_old, n_new = 278, len(out['test'])
    print(f'\ntest set {n_old} -> {n_new}  ({n_new / n_old:.2f}x)')
    print(f'  training loses {1233 - len(out["train"])} examples')

    # -- what that actually buys -------------------------------------------
    print(f'\n--- what {n_new} examples can and cannot resolve ---')
    print(f'{"comparison":<32} {"gap":>7} {"needed":>8} '
          f'{"@278":>7} {"@" + str(n_new):>7}')
    for name, eff, sd in OBSERVED:
        n_req = need(eff, sd)
        print(f'{name:<32} {eff:>7.3f} {n_req:>8.0f} '
              f'{100 * min(1, n_old / n_req):>6.0f}% {100 * min(1, n_new / n_req):>6.0f}%'
              )
    print('  last two columns are the share of the required sample you have, '
          'not power.')

    print(f'\n  smallest gap resolvable at 80% power, sd = 1.3 (typical here):')
    print(f'    at {n_old} examples   {detectable(n_old, 1.3):.3f} ranks')
    print(f'    at {n_new} examples   {detectable(n_new, 1.3):.3f} ranks')
    print(f'  The epoch gap was 0.126, so it moves from clearly out of reach '
          f'to\n  borderline. The pad-token gap was 0.014 and stays '
          f'unreachable by a\n  factor of ten. Doubling n buys sqrt(2), '
          f'not 2.')

    print(f'\n--- to use it, every bar must be re-measured on the new span ---')
    for m in ('board', 'random', 'llama3.1:8b'):
        print(f'  python3 eval_agent.py --model {m} \\')
        print(f'      --test {PREFIX}_test.jsonl --key {SRC}')
    print(f'  ...and each adapter. Roughly an hour per LLM at {n_new} calls.')
    print(f'  Existing sit_*.jsonl numbers stay valid on their own span; they '
          f'are\n  not comparable to these and should never be put in one '
          f'table.')
    c.report()


if __name__ == '__main__':
    main()
