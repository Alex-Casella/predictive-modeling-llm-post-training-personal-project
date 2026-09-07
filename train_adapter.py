"""Stage 4 -- QLoRA fine-tune of Llama 3.1 8B on Modal, for serving in Ollama.

PROJECT_CONTEXT.md §11e stage 4. Runs on Modal's GPU, not locally.

Executed three times: draft at 1 epoch (scored 5.41), sit at 2 epochs (3.12),
and a sit 1-epoch ablation that DID NOT PRODUCE A NUMBER -- see the --tag
comment in train() for what went wrong and what now prevents it.

WHY MODAL AND NOT THE MAC
    Apple Silicon has unified memory, and macOS lets the GPU address roughly
    two-thirds of it -- so a 24 GB Mac offers ~16 GB. That is enough to SERVE
    Llama 3.1 8B at 4-bit (~4.7 GB) and tight for training it. The bigger issue
    is toolchain: bitsandbytes, which QLoRA depends on, has been CUDA-first, and
    §14's own reference (the Unsloth -> Ollama walkthrough) assumes CUDA. MLX is
    the Apple-native alternative but produces a different adapter format and
    adds a conversion before Ollama will load it.
    Train on CUDA, serve on the Mac.

WHY THE BASE MODEL IS PINNED
    §11g: "the adapter must be trained against that same base or it will behave
    erratically." Ollama's own Modelfile docs say the same about ADAPTER. BASE
    below and the FROM line in the generated Modelfile are written from one
    constant so they cannot drift apart.

BEFORE THE FIRST RUN
    1. pip install modal && modal setup
    2. Accept the Llama 3.1 license on Hugging Face -- the weights are gated.
       https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    3. modal secret create huggingface HF_TOKEN=hf_...
    4. modal run train_adapter.py --dataset draft
       modal run train_adapter.py --dataset sit
    5. EXPECT_EPOCHS=2.0 ./finish_adapter.sh fantasy-draft draft

    Re-running a dataset with a different hyperparameter needs --tag, or the
    two runs land on the same volume path and become indistinguishable:
       modal run train_adapter.py --dataset sit --epochs 1 --tag e1
    or just  ./run_ablation.sh sit 1 e1  which chains all of it.

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import modal

BASE = 'meta-llama/Llama-3.1-8B-Instruct'
OLLAMA_BASE = 'llama3.1:8b'          # the SAME base, as Ollama names it
ADAPTER_DIR = '/adapter'

# TWO TASKS, ONE SCRIPT. The draft agent and the start/sit agent differ only in
# which pair of jsonl files they read and where the adapter lands. Copying this
# file for the second task would fork the hyperparameters, and then a difference
# in a result could be the task or could be a drifted learning rate -- with no
# way to tell which. One script, one set of defaults, one variable.
#
# Each writes into its OWN subdirectory of the volume, so the two adapters
# coexist and can be downloaded independently.
DATASETS = {
    'draft': dict(train='sft_train.jsonl', val='sft_val.jsonl',
                  seasons='train 2007-2019, val 2020-2022 '
                          '(test 2023-2025 never seen)'),
    'sit':   dict(train='sit_train.jsonl', val='sit_val.jsonl',
                  seasons='train 2011-2020, val 2021-2022 '
                          '(test 2023-2024 never seen)'),
}

app = modal.App('fantasy-draft-lora')
volume = modal.Volume.from_name('fantasy-lora', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install(
        'torch==2.5.1',
        'transformers==4.46.3',
        'peft==0.13.2',
        'trl==0.12.1',
        'datasets==3.1.0',
        'bitsandbytes==0.44.1',
        'accelerate==1.1.1',
    )
    # The training data ships with the image so the GPU container needs no
    # network access to read it. 5.1 MB -- small enough that this is simpler
    # than mounting a volume.
    .add_local_file('sft_train.jsonl', '/data/sft_train.jsonl')
    .add_local_file('sft_val.jsonl', '/data/sft_val.jsonl')
    .add_local_file('sit_train.jsonl', '/data/sit_train.jsonl')
    .add_local_file('sit_val.jsonl', '/data/sit_val.jsonl')
)


# NO GPU, NO TRAINING IMAGE. This is the credential check, and it has to be
# cheap enough to run before every training job.
#
# The container's Hugging Face token does NOT come from the shell that launched
# `modal run` -- it comes from the Modal secret above, set once and never
# re-read. So `export HF_TOKEN=...` locally can be correct while the container
# is still using a revoked token, and the failure surfaces on the FIRST LINE
# THAT TOUCHES THE HUB, minutes into a job that is already billing an A10G.
#
# That happened. A stale secret 401'd at AutoTokenizer.from_pretrained after
# the image had been built and the GPU allocated.
#
# The status code distinguishes the two causes, which need different fixes:
#     401  the token is missing, malformed or revoked  -> update the secret
#     403  the token is valid but the account has not accepted the Llama 3.1
#          license, or the grant lapsed                -> accept it on HF
@app.function(
    image=modal.Image.debian_slim(python_version='3.11'),
    timeout=120,
    secrets=[modal.Secret.from_name('huggingface')],
)
def check_access():
    import os
    import urllib.error
    import urllib.request

    url = f'https://huggingface.co/{BASE}/resolve/main/config.json'
    tok = os.environ.get('HF_TOKEN', '')
    if not tok:
        raise SystemExit(
            'the Modal secret "huggingface" does not set HF_TOKEN.\n'
            '  The key name matters -- huggingface_hub reads HF_TOKEN '
            'specifically.\n'
            '  Fix it at https://modal.com/secrets (not the CLI, so the token '
            'stays out\n  of your shell history).')

    # Length and prefix only. NEVER print the token, not even truncated -- a
    # log is a place a credential cannot be un-leaked from.
    print(f'  secret provides HF_TOKEN, {len(tok)} chars, '
          f'starts "{tok[:3]}"')

    req = urllib.request.Request(url, method='HEAD',
                                 headers={'Authorization': f'Bearer {tok}'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f'  HTTP {r.status} -- the container can read {BASE}')
            return 'ok'
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit(
                f'HTTP 401 from the Hub. The token IN THE MODAL SECRET is '
                f'rejected.\n'
                f'  Your local $HF_TOKEN is irrelevant here -- the container '
                f'never sees it.\n'
                f'  Replace the secret at https://modal.com/secrets, key '
                f'HF_TOKEN.')
        if e.code == 403:
            raise SystemExit(
                f'HTTP 403 from the Hub. The token is VALID but this account '
                f'may not\n  read {BASE}. Accept the license at\n'
                f'  https://huggingface.co/{BASE} and check the request was '
                f'granted.')
        raise SystemExit(f'HTTP {e.code} from the Hub: {e.reason}')
    except urllib.error.URLError as e:
        raise SystemExit(f'could not reach the Hub at all: {e.reason}')


@app.function(
    image=image,
    gpu='A10G',                       # 24 GB. QLoRA on 8B fits; H100 is waste.
    timeout=60 * 60 * 3,
    volumes={ADAPTER_DIR: volume},
    secrets=[modal.Secret.from_name('huggingface')],
)
def train(dataset: str = 'draft', epochs: float = 2.0, rank: int = 16,
          lr: float = 2e-4, tag: str = '', seed: int = 0,
          pad_token: str = ''):
    import datetime
    import inspect
    import json
    import os

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    def accepted(cls, kwargs):
        """Keep only kwargs this version of `cls` actually accepts.

        TRL renames things between releases -- max_seq_length became
        max_length, tokenizer became processing_class -- and pinning a version
        did not save me: the first smoke run died on
        `SFTConfig.__init__() got an unexpected keyword argument 'max_length'`.

        Filtering against the real signature works across versions instead of
        betting on one. Dropped keys are PRINTED, never silently discarded --
        some of them change what the model learns.
        """
        valid = set(inspect.signature(cls.__init__).parameters)
        keep = {k: v for k, v in kwargs.items() if k in valid}
        dropped = sorted(set(kwargs) - set(keep))
        if dropped:
            print(f'  [{cls.__name__}] not supported in this version, '
                  f'dropped: {dropped}')
        return keep

    def load(path):
        rows = [json.loads(l) for l in open(path)]
        # ONLY the messages. season / margin / board_rank / answer stay behind:
        # they are metadata for evaluation, and `answer` in particular is an
        # outcome. Nothing but the conversation reaches the trainer.
        return Dataset.from_list([{'messages': r['messages']} for r in rows])

    if dataset not in DATASETS:
        raise SystemExit(f'--dataset must be one of {sorted(DATASETS)}')
    spec = DATASETS[dataset]

    # --tag SEPARATES RUNS THAT DIFFER ONLY IN A HYPERPARAMETER.
    #
    # Without it every sit run writes to /adapter/sit, so a second run with
    # different epochs is indistinguishable from the first ON DISK. That is not
    # hypothetical: the 1-epoch ablation was launched, the volume was never
    # overwritten, and finish_adapter.sh happily downloaded the 2-epoch files
    # and scored them under the new model's name. The eval was byte-identical
    # to the previous run and nothing in the pipeline noticed.
    #
    # A tagged path makes that failure loud instead of silent: if the run did
    # not commit, /adapter/sit_e1 does not exist and the download fails.
    out_dir = f'{ADAPTER_DIR}/{dataset}{"_" + tag if tag else ""}'
    os.makedirs(out_dir, exist_ok=True)

    train_ds = load(f'/data/{spec["train"]}')
    val_ds = load(f'/data/{spec["val"]}')
    print(f'dataset={dataset}  train {len(train_ds)}  val {len(val_ds)}')
    print(f'writing to {out_dir}')

    tok = AutoTokenizer.from_pretrained(BASE)

    # THE PAD TOKEN DECIDES WHETHER THE MODEL EVER LEARNS TO STOP.
    #
    # The obvious line is `tok.pad_token = tok.eos_token`, and it is what this
    # file did. It produces an adapter that never emits EOS: 96% of the
    # start/sit model's answers ran to the token cap and enumerated every
    # candidate instead of stopping after the recommendation.
    #
    #   labels     ... Start Pacheco (RB). ... EOS PAD PAD PAD
    #   masking    label == pad_id  ->  -100          ^^^^^^^^^^^ intended
    #                                            ^^^ ALSO MASKED, because
    #                                                pad_id == eos_id
    #
    # The one position that teaches "stop here" is the one position removed
    # from the loss. No amount of training fixes it -- confirmed empirically:
    # runaway was WORSE at one epoch (96%) than at two (83%), the opposite of
    # what over-training would predict.
    #
    # The fix is a pad id that is not the eos id. Llama 3.x ships reserved
    # slots for exactly this, so no embedding resize is needed -- resizing
    # would change the vocabulary out from under `llama3.1:8b` and break
    # serving the adapter in Ollama.
    #
    # Which reserved token exists varies by release, so this searches rather
    # than asserting, PRINTS what it picked, and refuses to fall back to EOS
    # silently. A silent fallback is how the original bug survived.
    if not pad_token:
        tok.pad_token = tok.eos_token
        print(f'  pad_token = eos_token ({tok.eos_token!r}). THE ORIGINAL '
              f'BEHAVIOUR: real EOS is masked out of the loss and the model '
              f'will not learn to stop. Pass --pad-token to fix.')
    else:
        wanted = ([pad_token] if pad_token != 'auto' else
                  ['<|finetune_right_pad_id|>', '<|reserved_special_token_0|>',
                   '<|reserved_special_token_1|>'])
        chosen = None
        for cand in wanted:
            tid = tok.convert_tokens_to_ids(cand)
            if tid is not None and tid != tok.unk_token_id and tid >= 0:
                chosen = (cand, tid)
                break
        if chosen is None:
            raise SystemExit(
                f'none of {wanted} exists in this tokenizer, and falling back '
                f'to EOS is\n  the bug this flag exists to fix. Print '
                f'tok.additional_special_tokens to see what is available.')
        tok.pad_token = chosen[0]
        if tok.pad_token_id == tok.eos_token_id:
            raise SystemExit(
                f'{chosen[0]} resolves to the same id as EOS '
                f'({tok.eos_token_id}); it is not a usable pad token.')
        print(f'  pad_token = {chosen[0]!r} (id {chosen[1]}), '
              f'eos is id {tok.eos_token_id} -- distinct, so EOS stays in the '
              f'loss')

    # 4-bit base. The adapter itself trains in bf16 -- that is the QLoRA idea:
    # a frozen quantised backbone with a small high-precision delta on top.
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        torch_dtype=torch.bfloat16,
        device_map='auto')
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        bias='none', task_type='CAUSAL_LM',
        # Attention + MLP projections. Ollama converts adapters on these;
        # exotic target sets are where the GGUF conversion tends to break.
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'])

    cfg_kwargs = dict(
        output_dir='/tmp/out',
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,        # effective batch 16
        learning_rate=lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        report_to='none',
        # Controls BOTH the data order and the LoRA A-matrix init, so two runs
        # that differ only here are two independent draws of the same recipe.
        # That is what gives every other number in this project an error bar:
        # a 0.13 gap between epoch counts means nothing until you know what two
        # IDENTICAL configs score against each other.
        seed=seed,
        # Prompts are ~1,850 characters; 2048 tokens holds the whole exchange
        # with room to spare. Truncating here would silently cut the shortlist
        # and teach the model to answer from a partial list. The two names are
        # the same setting in different TRL versions -- exactly one survives
        # the filter below.
        max_length=2048,
        max_seq_length=2048,
        # Train on the answer, not on reciting the shortlist back. Only newer
        # TRL has this; if it is dropped the run is still valid, just less
        # efficient per token, so the warning below is a note not an alarm.
        completion_only_loss=True,
    )
    cfg = SFTConfig(**accepted(SFTConfig, cfg_kwargs))
    if not getattr(cfg, 'completion_only_loss', False):
        print('  NOTE: loss covers the whole exchange, not just the answer. '
              'Training will still work; the model spends capacity learning to '
              'echo the shortlist.')

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        args=cfg,
    )
    # `tokenizer` was renamed `processing_class`. Pass whichever exists, never
    # both -- versions that accept both treat one as deprecated and warn.
    sig = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs['processing_class' if 'processing_class' in sig
                   else 'tokenizer'] = tok

    trainer = SFTTrainer(**accepted(SFTTrainer, trainer_kwargs))
    trainer.train()

    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    modelfile = (f'FROM {OLLAMA_BASE}\n'
                 f'ADAPTER ./adapter.gguf\n\n'
                 f'PARAMETER temperature 0\n')
    with open(f'{out_dir}/Modelfile', 'w') as f:
        f.write(modelfile)
    # run_id is a wall-clock stamp written by THIS run. Two adapters that share
    # every hyperparameter still differ here, so "did the file I downloaded
    # come from the run I just launched" has an answer that does not depend on
    # remembering what was launched. finish_adapter.sh asserts on epochs; a
    # human reading a stale run_id is the backstop.
    with open(f'{out_dir}/PROVENANCE.txt', 'w') as f:
        f.write(f'dataset={dataset}\ntag={tag or "(none)"}\n'
                f'run_id={datetime.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}\n'
                f'base={BASE}\n'
                f'ollama_base={OLLAMA_BASE}\n'
                f'rank={rank} alpha={rank * 2} lr={lr} epochs={epochs} '
                f'seed={seed}\n'
                f'pad_token={tok.pad_token!r} id={tok.pad_token_id} '
                f'eos_id={tok.eos_token_id}\n'
                f'train={len(train_ds)} val={len(val_ds)}\n'
                f'seasons: {spec["seasons"]}\n')
    volume.commit()
    print(f'adapter written to {out_dir}')
    print(modelfile)


@app.local_entrypoint()
def main(dataset: str = 'draft', epochs: float = 2.0, rank: int = 16,
         lr: float = 2e-4, tag: str = '', seed: int = 0,
         pad_token: str = ''):
    # Credentials before the GPU, always. A CPU container costs seconds; the
    # same failure discovered inside train() costs the image build, the A10G
    # allocation, and the walk to the coffee machine.
    print('checking the container\'s Hugging Face access '
          '(CPU container, no GPU) ...')
    check_access.remote()

    train.remote(dataset=dataset, epochs=epochs, rank=rank, lr=lr, tag=tag,
                 seed=seed, pad_token=pad_token)
    subdir = f'{dataset}{"_" + tag if tag else ""}'
    print('\nnext:')
    # The volume is MOUNTED at /adapter, so its own root IS that directory.
    # `modal volume get fantasy-lora /adapter ...` fails with "no such file or
    # directory" -- copy from `/`, the volume root.
    print(f'  modal volume ls fantasy-lora/{subdir}   # confirm files exist')
    print(f'  export HF_TOKEN=hf_...              # gated base model config')
    print(f'  EXPECT_EPOCHS={epochs} ./finish_adapter.sh '
          f'fantasy-{subdir.replace("_", "-")} {subdir}')
    print('')
    print('  finish_adapter.sh does the rest: download, GGUF conversion,')
    print('  ollama create, and scoring against the right bar for this task.')
    print('  EXPECT_EPOCHS makes it ABORT on a provenance mismatch rather than')
    print('  print one and carry on -- that is how the last ablation scored the')
    print('  wrong adapter and reported the number with a straight face.')
