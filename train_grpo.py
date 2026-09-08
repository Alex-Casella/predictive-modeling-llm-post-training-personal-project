"""Stage 11 -- GRPO. Optimise the score instead of imitating an answer.

Everything before this was SUPERVISED fine-tuning: each example carried one
written answer and training pushed the model toward reproducing that text.

    SFT   prompt -> [one target answer]          "say this"
    GRPO  prompt -> [G sampled answers]          "here is a score for each;
                                                  make the better ones more
                                                  likely than the worse ones"

The difference matters when a task has many acceptable answers and no gold
string. It also matters here for a subtler reason: the SFT target was written
by a template, so the model was trained to imitate `step10_render_sft.py`, not
to pick well. The rank it eventually scored was a side effect. GRPO optimises
the rank directly.

THE REWARD IS THE EVAL, IMPORTED NOT REIMPLEMENTED

    reward(completion) = (K + 1 - rank) / K      in [0, 1], higher is better
    invalid answer     = 0.0

`rank` is where the named player finished among the K shown, by hindsight --
the same number `eval_agent.py` reports as the score. This file imports
`extract_pick` from that module rather than re-writing it. A reward that parses
completions differently from the scorer optimises something other than what is
measured, and the difference would be invisible: both would run, both would
produce numbers, and the training curve would climb while the eval did not
move.

REWARD HACKING, NAMED IN ADVANCE

The board's pick scores 6.00 of 12 on average. A model that learns to always
name the first player listed gets a mediocre-but-safe reward with no
understanding, and the mean rank would improve while the model got dumber. Two
defences:

  1. `beta` (the KL penalty) keeps the policy near the base model
  2. `board_agreement` is logged as a second reward function that returns 0.0
     and only records -- if agreement with the board climbs toward 100%, the
     model has collapsed onto the sort and the run is a hack, not a result

(2) costs nothing and is checkable after the fact, which is the point.

WHY THIS IS NOT IN train_adapter.py

CLAUDE.md rule 3 says one training script, so that two SFT tasks cannot drift
apart in hyperparameters. That rule is about comparing tasks. GRPO is a
different algorithm with a different config object and no shared surface, so
forking here does not create the failure that rule prevents. The EVAL harness
is still shared, which is where comparability actually lives.

RUN IT SMALL FIRST
    modal run train_grpo.py --limit 10 --steps 2 --generations 4
    # ~5 minutes. Proves TRL accepts the config and the reward is called.
    modal run train_grpo.py --limit 600
    EXPECT_EPOCHS=1 ./finish_adapter.sh fantasy-draft-grpo draft_grpo

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import modal

BASE = 'meta-llama/Llama-3.1-8B-Instruct'
OLLAMA_BASE = 'llama3.1:8b'
ADAPTER_DIR = '/adapter'

app = modal.App('fantasy-grpo')
volume = modal.Volume.from_name('fantasy-lora', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch==2.5.1',
        'transformers==4.46.3',
        'peft==0.13.2',
        # GRPOTrainer needs a newer TRL than the SFT runs used. Pinned, and
        # every kwarg is filtered against the real signature below, because
        # TRL renames arguments between releases -- that is what the same
        # filter in train_adapter.py exists for.
        'trl==0.12.1',
        'datasets==3.1.0',
        'bitsandbytes==0.44.1',
        'accelerate==1.1.1',
        'pandas==2.2.3',
    )
    .add_local_file('sft_train.jsonl', '/data/sft_train.jsonl')
    .add_local_file('draft_examples.jsonl', '/data/draft_examples.jsonl')
    # The scorer itself, so the reward cannot drift from the evaluation.
    .add_local_file('eval_agent.py', '/data/eval_agent.py')
)


@app.function(
    image=image,
    gpu='A10G',
    timeout=60 * 60 * 6,
    volumes={ADAPTER_DIR: volume},
    secrets=[modal.Secret.from_name('huggingface')],
)
def train(limit: int = 600, epochs: float = 1.0, generations: int = 8,
          rank: int = 16, lr: float = 1e-5, beta: float = 0.04,
          steps: int = 0, seed: int = 0, tag: str = 'grpo'):
    import datetime
    import inspect
    import json
    import os
    import sys

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, '/data')
    from eval_agent import candidates, extract_pick     # noqa: E402

    def accepted(cls, kwargs):
        valid = set(inspect.signature(cls.__init__).parameters)
        keep = {k: v for k, v in kwargs.items() if k in valid}
        dropped = sorted(set(kwargs) - set(keep))
        if dropped:
            print(f'  [{cls.__name__}] not supported here, dropped: {dropped}')
        return keep

    out_dir = f'{ADAPTER_DIR}/draft_{tag}'
    os.makedirs(out_dir, exist_ok=True)

    # -- the answer key, joined exactly as eval_agent.py joins it ----------
    key = {}
    for line in open('/data/draft_examples.jsonl'):
        r = json.loads(line)
        key[(r['season'], r['pick'])] = [x['player'] for x in
                                         r['label']['ranking']]

    rows = [json.loads(l) for l in open('/data/sft_train.jsonl')]
    if limit:
        rows = rows[:limit]

    data = []
    for r in rows:
        ranking = key.get((r['season'], r['pick']))
        if ranking is None:
            continue
        data.append({
            # Prompt only: the assistant turn is discarded. That IS the method
            # -- nothing here imitates the written answer.
            'prompt': r['messages'][:2],
            'names': candidates(r),
            'ranking': ranking,
            'board_pick': r['board_pick'],
        })
    ds = Dataset.from_list(data)
    k = len(data[0]['names'])
    print(f'{len(ds)} prompts, K={k} candidates each, '
          f'{generations} generations per prompt')
    print(f'writing to {out_dir}')

    # -- the reward ---------------------------------------------------------
    #
    # THE SCALE, measured on these prompts before the run:
    #
    #   always the hindsight best   1.000
    #   the SFT model already here  0.632   (mean rank 5.42 on test)
    #   the board                   0.586   (mean rank 5.97)
    #   uniform random of 12        0.542
    #   invalid                     0.000
    #
    # Almost all the mass is "did not name a nobody". The band GRPO has to
    # work in, between random and the current best, is 0.090 wide. That is
    # fine in itself -- GRPO uses the advantage WITHIN each group of G samples,
    # so a constant offset cancels and the scale divides out.
    #
    # What is not fine is a group whose G samples all score the same: the
    # advantage is then zero for every one of them and the step teaches
    # nothing. With a confident model and a short answer that is a real
    # possibility, so it is counted rather than assumed away.
    stats = {'invalid': 0, 'total': 0, 'board_agree': 0,
             'groups': 0, 'flat_groups': 0}

    def rank_reward(completions, names, ranking, **kw):
        out = []
        for comp, nm, rk in zip(completions, names, ranking):
            text = comp[-1]['content'] if isinstance(comp, list) else comp
            pick = extract_pick(text, nm)
            stats['total'] += 1
            if pick is None or pick not in rk:
                stats['invalid'] += 1
                out.append(0.0)                  # naming nobody scores nothing
                continue
            r = rk.index(pick) + 1               # 1 = best of K, by hindsight
            out.append((k + 1 - r) / k)
        # One call covers one group when the batch is exactly G wide, which is
        # how cfg is configured. If that ever stops holding the counter simply
        # stops incrementing rather than reporting something false.
        if len(out) == generations:
            stats['groups'] += 1
            if max(out) - min(out) < 1e-9:
                stats['flat_groups'] += 1
        return out

    def board_agreement(completions, names, board_pick, **kw):
        """Records only. Returns 0.0 so it cannot influence training.

        If this climbs toward 1.0 the model has collapsed onto the
        deterministic board's pick -- a safe, mediocre reward with no
        understanding. That is the hack this run is most likely to find.
        """
        for comp, nm, bp in zip(completions, names, board_pick):
            text = comp[-1]['content'] if isinstance(comp, list) else comp
            if extract_pick(text, nm) == bp:
                stats['board_agree'] += 1
        return [0.0] * len(completions)

    tok = AutoTokenizer.from_pretrained(BASE)
    # The lesson from the SFT runs: pad must not be EOS or the model never
    # learns to stop. Applied here from the start.
    for cand in ('<|finetune_right_pad_id|>', '<|reserved_special_token_0|>'):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid >= 0 and tid != tok.eos_token_id:
            tok.pad_token = cand
            break
    else:
        raise SystemExit('no reserved pad token; refusing to fall back to EOS')
    print(f'  pad_token={tok.pad_token!r} id={tok.pad_token_id}, '
          f'eos id={tok.eos_token_id}')

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        torch_dtype=torch.bfloat16, device_map='auto')
    model.config.use_cache = False

    cfg_kwargs = dict(
        output_dir='/tmp/grpo',
        num_train_epochs=epochs,
        max_steps=steps if steps else -1,
        per_device_train_batch_size=generations,
        gradient_accumulation_steps=2,
        num_generations=generations,
        # Answers are one short paragraph. Generation dominates GRPO's wall
        # clock, so this is the main speed lever -- 5x the length any real
        # answer needs, and no more.
        max_completion_length=96,
        max_prompt_length=2048,
        learning_rate=lr,           # far lower than SFT's 2e-4; RL is touchy
        beta=beta,                  # KL penalty toward the base model
        temperature=0.9,            # must be > 0 or every sample is identical
        lr_scheduler_type='constant',
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy='no',
        report_to='none',
        seed=seed,
    )
    cfg = GRPOConfig(**accepted(GRPOConfig, cfg_kwargs))

    peft_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'])

    trainer_kwargs = dict(
        model=model,
        reward_funcs=[rank_reward, board_agreement],
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tok,
    )
    trainer = GRPOTrainer(**accepted(GRPOTrainer, trainer_kwargs))
    trainer.train()

    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    inval = 100 * stats['invalid'] / max(stats['total'], 1)
    agree = 100 * stats['board_agree'] / max(stats['total'], 1)
    flat = 100 * stats['flat_groups'] / max(stats['groups'], 1)
    print(f'\ngenerations scored {stats["total"]}')
    print(f'  invalid (named nobody on the shortlist)   {inval:.1f}%')
    print(f'  agreed with the board                     {agree:.1f}%')
    print(f'  groups where all {generations} samples scored the same  '
          f'{flat:.1f}%')
    print('  Board agreement near 100% means the policy collapsed onto the')
    print('  sort -- a reward hack, not a result.')
    print('  Flat groups near 100% means the advantage was zero and the run')
    print('  taught the model nothing, whatever the loss curve looked like.')
    print('  Read both before reporting any score.')

    with open(f'{out_dir}/Modelfile', 'w') as f:
        f.write(f'FROM {OLLAMA_BASE}\nADAPTER ./adapter.gguf\n\n'
                f'PARAMETER temperature 0\n')
    with open(f'{out_dir}/PROVENANCE.txt', 'w') as f:
        f.write(f'method=GRPO\ndataset=draft\ntag={tag}\n'
                f'run_id={datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}\n'
                f'base={BASE}\nollama_base={OLLAMA_BASE}\n'
                f'rank={rank} alpha={rank * 2} lr={lr} epochs={epochs} '
                f'seed={seed}\n'
                f'generations={generations} beta={beta} prompts={len(ds)}\n'
                f'reward=(K+1-rank)/K, invalid=0, K={k}\n'
                f'pad_token={tok.pad_token!r} id={tok.pad_token_id} '
                f'eos_id={tok.eos_token_id}\n'
                f'invalid_pct={inval:.1f} board_agreement_pct={agree:.1f} '
                f'flat_group_pct={flat:.1f}\n')
    volume.commit()
    print(f'adapter written to {out_dir}')


@app.local_entrypoint()
def main(limit: int = 600, epochs: float = 1.0, generations: int = 8,
         rank: int = 16, lr: float = 1e-5, beta: float = 0.04,
         steps: int = 0, seed: int = 0, tag: str = 'grpo'):
    train.remote(limit=limit, epochs=epochs, generations=generations,
                 rank=rank, lr=lr, beta=beta, steps=steps, seed=seed, tag=tag)
    print('\nnext:')
    print(f'  modal volume ls fantasy-lora/draft_{tag}')
    print(f'  export HF_TOKEN=hf_...')
    print(f'  EXPECT_EPOCHS={epochs} ./finish_adapter.sh '
          f'fantasy-draft-{tag} draft_{tag}')
    print('')
    print('  Then the pairings that matter, in this order:')
    print('    python3 paired_test.py --a eval_board.csv '
          f'--b eval_fantasy-draft-{tag}.csv        # vs the sort')
    print('    python3 paired_test.py --a eval_fantasy-draft-pad.csv '
          f'--b eval_fantasy-draft-{tag}.csv   # vs SFT')
    print('  And read board_agreement_pct in PROVENANCE before either.')