"""Stage 4 evaluation -- does the agent actually beat the number it was handed?

PROJECT_CONTEXT.md §11g: "Compare against the tabular layer alone, no LLM. If
the agent doesn't beat the number it was handed, the language layer is
decoration." This file is that comparison, and it exists BEFORE any model is
trained so the bar is fixed in advance rather than chosen to flatter a result.

THE METRIC

Each test example offers 12 candidates. After the season played out they can be
ranked by what actually happened. Score = where the chosen player lands in that
ranking. Lower is better.

    random choice                 6.50  (chance, by construction)
    the deterministic board       6.00  <- THE BAR, on the 450-example test set
    a fine-tuned model            ?

The bar is recomputed on whatever slice is evaluated, never hardcoded -- see
RANDOM_BAR below. A model that scores worse than the board is decoration and
should be reported as such, not tuned until it isn't.

Also tracked, because a language model can fail in ways a sort cannot:

    validity   did it name a player that was actually on the shortlist?
               An invalid answer is not a wrong pick, it is a broken one.
    best_pct   how often it found the single best of the 12.

WHY THE ANSWER KEY LIVES IN A SEPARATE FILE

sft_test.jsonl carries only the prompt, the completion and light metadata. The
hindsight ranking needed to score an arbitrary pick stays in
draft_examples.jsonl and is joined here on (season, pick). Nothing that reveals
an outcome is ever in a file the trainer reads.

Usage:
    python3 eval_agent.py --model board       # the bar, no LLM needed
    python3 eval_agent.py --model random      # sanity check on the metric
    python3 eval_agent.py --model llama3.1:8b # base model, via local Ollama
    python3 eval_agent.py --model fantasy-draft
"""
import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request

import pandas as pd

TEST = 'sft_test.jsonl'
KEY = 'draft_examples.jsonl'
OLLAMA = 'http://localhost:11434/api/chat'

# The bar is RECOMPUTED on whatever slice is being evaluated, never hardcoded.
# Step 9's headline 5.98 is the board's score over all 2,850 examples; on the
# 450-example test split the board scores 6.00. Comparing a model measured on
# the test split against a bar measured on everything is the same apples-to-
# oranges error CLAUDE.md rules out for the projection model ("report its score
# next to the baseline score on the same held-out years"). Same rule here.
RANDOM_BAR = 6.5        # (12 + 1) / 2, by construction


def load():
    test = [json.loads(l) for l in open(TEST)]
    key = {}
    for e in (json.loads(l) for l in open(KEY)):
        key[(e['season'], e['pick'])] = {
            r['player']: i + 1
            for i, r in enumerate(e['label']['ranking'])}
    for t in test:
        t['ranking'] = key[(t['season'], t['pick'])]
        assert t['answer'] in t['ranking'], 'answer key mismatch'
    return test


def candidates(item):
    """Player names offered in the prompt, in the order shown."""
    user = item['messages'][1]['content']
    block = user.split('Available players:')[1].split('Which one')[0]
    return [re.match(r'\s*(.+?) \(', ln).group(1)
            for ln in block.strip().splitlines() if ln.strip()]


def ask_ollama(model, item, timeout=120):
    body = json.dumps({
        'model': model,
        'messages': item['messages'][:2],       # system + user, NOT the answer
        'stream': False,
        'options': {'temperature': 0},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())['message']['content']


def extract_pick(text, names):
    """Which shortlisted player did the model name first?

    Deliberately lenient about prose and strict about identity: it must be a
    name that was on the shortlist. Anything else counts as invalid rather than
    being charitably mapped onto a real player.
    """
    hits = [(text.find(n), n) for n in names if n in text]
    hits = [h for h in hits if h[0] >= 0]
    return min(hits)[1] if hits else None


def score(items, chooser, label, progress=False):
    """Score one chooser over `items`.

    `progress` prints a line per example. An LLM run is 60+ sequential
    generations at ~500-token prompts with the first one also paging ~5 GB of
    weights into memory, so several minutes of total silence looks exactly like
    a hang. It is not, but a user cannot tell the difference -- hence the
    per-item output and the running ETA.
    """
    rows = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        names = candidates(it)
        pick = chooser(it, names)
        valid = pick in it['ranking'] if pick else False
        rows.append(dict(
            season=it['season'], round=it['round'], margin=it['margin'],
            pick=pick, valid=valid,
            rank=it['ranking'][pick] if valid else None,
            best=bool(valid and it['ranking'][pick] == 1)))
        if progress:
            per = (time.time() - t0) / i
            eta = per * (len(items) - i)
            rk = rows[-1]['rank']
            print(f'  [{i:>3}/{len(items)}] {it["season"]} r{it["round"]:<2} '
                  f'{(pick or "NO VALID PICK"):<24} '
                  f'rank {rk if rk else "-":>3}   '
                  f'{per:.1f}s/ex, ~{eta / 60:.1f} min left', flush=True)
    r = pd.DataFrame(rows)
    ok = r[r['valid']]
    return dict(model=label, n=len(r),
                validity=round(100 * r['valid'].mean(), 1),
                mean_rank=round(ok['rank'].mean(), 2) if len(ok) else None,
                best_pct=round(100 * ok['best'].mean(), 1) if len(ok) else None), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='"board", "random", or an Ollama model name')
    ap.add_argument('--limit', type=int, default=0,
                    help='evaluate only the first N examples (LLM runs are slow)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    items = load()
    if args.limit:
        items = items[:args.limit]
    print(f'{len(items)} test examples, seasons '
          f'{min(i["season"] for i in items)}-{max(i["season"] for i in items)}')

    if args.model == 'board':
        chooser = lambda it, names: it['board_pick']
    elif args.model == 'random':
        rng = random.Random(args.seed)
        chooser = lambda it, names: rng.choice(names)
    else:
        def chooser(it, names, _m=args.model):
            try:
                return extract_pick(ask_ollama(_m, it), names)
            except urllib.error.URLError as e:
                raise SystemExit(
                    f'cannot reach Ollama at {OLLAMA}: {e}\n'
                    f'  start it with `ollama serve`, and make sure '
                    f'`ollama list` shows "{_m}"')

    is_llm = args.model not in ('board', 'random')
    if is_llm:
        print(f'\nquerying {args.model} via Ollama -- first call also loads\n'
              f'the model into memory, so expect a slow start.\n')
    summary, detail = score(items, chooser, args.model, progress=is_llm)
    # The bar, measured on exactly the examples just evaluated.
    bar, _ = score(items, lambda it, names: it['board_pick'], 'board')  # no LLM, instant

    print(f'\n=== {args.model} on {TEST} ===')
    print(f"  answered with a shortlisted player   {summary['validity']}%")
    print(f"  mean rank of its pick (of 12)        {summary['mean_rank']}"
          f"   <- lower is better")
    print(f"  picked the best of 12                {summary['best_pct']}%"
          f"   (chance 8.3%)")
    print(f'\n  reference points, on these same {len(items)} examples')
    print(f'    random choice                      {RANDOM_BAR:.2f}')
    print(f"    deterministic board (the bar)      {bar['mean_rank']:.2f}")
    if summary['mean_rank'] and args.model != 'board':
        d = bar['mean_rank'] - summary['mean_rank']
        verdict = ('BEATS the board' if d > 0 else
                   'does NOT beat the board -- decoration, per §11g')
        print(f'    this model                         '
              f'{summary["mean_rank"]:.2f}  ({d:+.2f})  {verdict}')

    ok = detail[detail['valid']]
    if len(ok):
        print('\n--- by decision difficulty (margin = actual-VBD gap) ---')
        bins = [(0, 1, 'coin flip  <1'), (1, 5, 'marginal  1-5'),
                (5, 20, 'real     5-20'), (20, 1e9, 'decisive   20+')]
        for lo, hi, lbl in bins:
            m = ok[(ok['margin'] >= lo) & (ok['margin'] < hi)]
            if len(m):
                print(f'  {lbl:<16} n={len(m):>4}  mean rank '
                      f'{m["rank"].mean():.2f}   best {100 * m["best"].mean():.1f}%')
        print('\n--- by season (held out, never trained on) ---')
        print(ok.groupby('season').agg(n=('rank', 'size'),
                                       mean_rank=('rank', 'mean'),
                                       best_pct=('best', 'mean'))
              .assign(best_pct=lambda d: (100 * d['best_pct']).round(1))
              .round(2).to_string())

    out = f'eval_{args.model.replace(":", "_").replace("/", "_")}.csv'
    detail.to_csv(out, index=False)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
