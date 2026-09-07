"""Stage 4 -- QLoRA fine-tune of Llama 3.1 8B on Modal, for serving in Ollama.

PROJECT_CONTEXT.md §11e stage 4. Runs on Modal's GPU, not locally.

!! THIS FILE HAS NEVER BEEN EXECUTED !!
Everything else in this repo was run and its output pasted into a commit
message. This was not: it needs a Modal account, a GPU, and a Hugging Face
token, none of which exist in the environment it was written in. Treat it as a
reviewed draft. The first run will surface something; that is expected.

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
    4. modal run train_adapter.py::train
    5. modal volume get fantasy-lora /adapter ./adapter

Data source: Pro-Football-Reference. See ATTRIBUTION.md.
"""
import modal

BASE = 'meta-llama/Llama-3.1-8B-Instruct'
OLLAMA_BASE = 'llama3.1:8b'          # the SAME base, as Ollama names it
ADAPTER_DIR = '/adapter'

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
)


@app.function(
    image=image,
    gpu='A10G',                       # 24 GB. QLoRA on 8B fits; H100 is waste.
    timeout=60 * 60 * 3,
    volumes={ADAPTER_DIR: volume},
    secrets=[modal.Secret.from_name('huggingface')],
)
def train(epochs: float = 2.0, rank: int = 16, lr: float = 2e-4):
    import inspect
    import json
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

    train_ds, val_ds = load('/data/sft_train.jsonl'), load('/data/sft_val.jsonl')
    print(f'train {len(train_ds)}  val {len(val_ds)}')

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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
        seed=0,
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

    trainer.model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)

    modelfile = (f'FROM {OLLAMA_BASE}\n'
                 f'ADAPTER ./adapter.gguf\n\n'
                 f'PARAMETER temperature 0\n')
    with open(f'{ADAPTER_DIR}/Modelfile', 'w') as f:
        f.write(modelfile)
    with open(f'{ADAPTER_DIR}/PROVENANCE.txt', 'w') as f:
        f.write(f'base={BASE}\nollama_base={OLLAMA_BASE}\n'
                f'rank={rank} alpha={rank * 2} lr={lr} epochs={epochs}\n'
                f'train={len(train_ds)} val={len(val_ds)}\n'
                f'seasons: train 2007-2019, val 2020-2022 '
                f'(test 2023-2025 never seen)\n')
    volume.commit()
    print(f'adapter written to {ADAPTER_DIR}')
    print(modelfile)


@app.local_entrypoint()
def main(epochs: float = 2.0, rank: int = 16, lr: float = 2e-4):
    train.remote(epochs=epochs, rank=rank, lr=lr)
    print('\nnext:')
    print('  modal volume get fantasy-lora /adapter ./adapter')
    print('  # convert the PEFT adapter to GGUF (llama.cpp '
          'convert_lora_to_gguf.py), then:')
    print('  ollama create fantasy-draft -f adapter/Modelfile')
    print('  python3 eval_agent.py --model fantasy-draft --limit 60')
    print(f'  # compare against: python3 eval_agent.py --model {OLLAMA_BASE}')
