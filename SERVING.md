# Stage 5 — from a trained adapter to a served model

> **Two adapters now.** The draft agent and the start/sit agent are separate
> adapters over the same base, written to separate subdirectories of the Modal
> volume. Everything below applies to both; `finish_adapter.sh` takes the
> subdirectory as its second argument:
>
> ```bash
> ./finish_adapter.sh fantasy-draft draft
> ./finish_adapter.sh fantasy-sit   sit
> ```
>
> Read the PROVENANCE it prints at step 2. It names the dataset, and building
> the wrong adapter then scoring it against the other task's bar is a mistake
> that produces a plausible-looking number.

> **Re-training a dataset you have already trained needs a `--tag`.**
> Both runs otherwise write to `/adapter/sit` and become indistinguishable on
> disk. This is not hypothetical — a 1-epoch start/sit ablation was launched,
> the volume was never overwritten, the 2-epoch adapter was downloaded,
> converted, served as `fantasy-sit-1ep` and scored. The result was
> byte-identical to the 2-epoch run and read as a finding.
>
> ```bash
> modal run train_adapter.py --dataset sit --epochs 1 --tag e1
> EXPECT_EPOCHS=1.0 ./finish_adapter.sh fantasy-sit-e1 sit_e1
> ```
>
> `EXPECT_EPOCHS` turns "read the provenance" into an assertion that aborts.
> `finish_adapter.sh` also refuses to score a model whose Ollama content ID
> matches one already installed — identical IDs are identical weights, so at
> `temperature 0` the eval could only reproduce the other model's CSV.
>
> `./run_ablation.sh sit 1 e1` chains all of it, including the paired test.

`PROJECT_CONTEXT.md` §11e stage 5. Everything here runs on your Mac.

`train_adapter.py` writes a **PEFT/safetensors** adapter. Ollama's `ADAPTER`
directive needs **GGUF**. Nothing in this repo produces that conversion — this
file is the missing step, written out rather than assumed.

> Ollama's own Modelfile docs: *"if the base model is not the same as the base
> model that the adapter was tuned from the behaviour will be erratic."*
> `train_adapter.py` pins `BASE` and the generated `FROM` line to one constant
> so they cannot drift. Do not edit one without the other.

---

## 0 · Do this WHILE training runs — it is the number that matters

§11g requires comparing against the un-tuned base, not only against the board.
Without it you cannot tell "fine-tuning helped" from "Llama could already do
this."

```bash
ollama pull llama3.1:8b
python3 eval_agent.py --model llama3.1:8b --limit 60
```

Two outcomes, both useful:

- It scores near or below the board (6.00) → the base model cannot do this
  unaided, and the fine-tune has room to prove something.
- It already beats the board → the interesting question becomes whether
  fine-tuning adds anything *on top of* a capable base, which is a different
  and more honest experiment.

This run is also the cheapest way to find a problem in the prompt format while
the adapter is still cooking.

---

## 1 · Pull the adapter down from Modal

The volume is MOUNTED at `/adapter` inside the container, so the volume's own
root *is* that directory. `modal volume get fantasy-lora /adapter ...` fails
with "no such file or directory" — there is no `/adapter` inside the volume.
Copy from the root:

```bash
modal volume ls fantasy-lora          # confirm the files are there first
modal volume get fantasy-lora / ./adapter
ls adapter/
```

An empty `modal volume ls` means the job errored after training but before
`volume.commit()` — check the Modal logs rather than re-running blind.

Expect `adapter_model.safetensors`, `adapter_config.json`, tokenizer files,
plus the `Modelfile` and `PROVENANCE.txt` the training job wrote.

```bash
cat adapter/PROVENANCE.txt
```
> Records the base model, LoRA rank, learning rate, epochs and the season
> splits. Read it before trusting any result — it is the only record of which
> run produced which adapter.

---

## 2 · Convert PEFT → GGUF

Ollama can convert some HuggingFace LoRA adapters directly; when it cannot,
`llama.cpp` has the converter.

### Direct safetensors loading does NOT work — tested, do not retry

Ollama 0.33.3 on macOS cannot consume the PEFT adapter this project produces.
Five variants were tried against a directory containing exactly what Ollama's
own docs require (`adapter_model.safetensors` + `adapter_config.json`):

| `ADAPTER` value | result |
|---|---|
| `.` (relative dir) | `no Modelfile or safetensors files found` |
| absolute path to the dir | `no Modelfile or safetensors files found` |
| `./adapter_model.safetensors` | finds and copies the file, then `open adapter_config.json: no such file or directory` |
| absolute path to the safetensors file | same as above |
| a minimal dir holding ONLY those two files | `no Modelfile or safetensors files found` |

The file-path variants get furthest: Ollama locates and copies the 160 MB
safetensors, reaches `converting adapter`, then fails looking for
`adapter_config.json` — which is sitting right beside it. It appears to resolve
that path against the server's working directory rather than the adapter's.

Matches the open upstream report:
https://github.com/ollama/ollama/issues/13314

**Go straight to the llama.cpp conversion below.** It is also the path
`PROJECT_CONTEXT.md` §14's own references (the Unsloth → Ollama walkthrough)
describe, so this is the mainline rather than a workaround.

<details>
<summary>The direct attempt, kept for the record</summary>


```bash
cd adapter
# point ADAPTER at the safetensors directory instead of a .gguf
sed -i '' 's|ADAPTER ./adapter.gguf|ADAPTER .|' Modelfile
ollama create fantasy-draft -f Modelfile
```
</details>

### The conversion that does work

VERIFIED WORKING. Produced a 167.8 MB GGUF with 448 tensors from a rank-16
adapter over all 32 layers.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# ISOLATED env, not the project env. llama.cpp pins numpy~=1.26 and torch==2.11;
# installing those into the project env downgrades numpy 2.5.3 -> 1.26.4 and
# pandas 3.0.5 -> 2.2.3 and breaks every script in this repo. Learned the hard
# way. A clean env also clears the "cannot import name 'GenerationMixin'"
# ImportError, which is a half-upgraded transformers, not a real conflict.
python3 -m venv gguf-convert-env && source gguf-convert-env/bin/activate
pip install -r requirements.txt

export HF_TOKEN=...   # the base model's config is gated; only the config is read

# --base-model-id, NOT --base.
#   --base           expects a LOCAL directory of base-model files, and fails
#                    with FileNotFoundError: 'meta-llama/Llama-3.1-8B-Instruct'
#                    after downloading config.json, because it then tries to
#                    listdir() the repo id as a path.
#   --base-model-id  fetches just the config from Hugging Face. A LoRA
#                    conversion needs the architecture and dimensions, not the
#                    16 GB of base weights.
python convert_lora_to_gguf.py \
  /abs/path/to/adapter_files \
  --outfile /abs/path/to/adapter_files/adapter.gguf \
  --base-model-id meta-llama/Llama-3.1-8B-Instruct

deactivate
```

Then, from the adapter directory:

```bash
cat > Modelfile <<'MODELFILE'
FROM llama3.1:8b
ADAPTER ./adapter.gguf

PARAMETER temperature 0
MODELFILE
ollama create fantasy-draft -f Modelfile
```

```bash
ollama list
```
> `fantasy-draft` should appear.

**If conversion errors on unsupported target modules:** `train_adapter.py`
targets the seven standard attention and MLP projections
(`q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`) precisely because
exotic target sets are where this step breaks. If it still fails, that is the
first thing to report, not to work around.

---

## 3 · Talk to it

```bash
ollama run fantasy-draft
```

Paste a prompt from the test set to see it in context:

```bash
python3 -c "import json; print(json.loads(open('sft_test.jsonl').readline())['messages'][1]['content'])"
```

---

## 4 · Score it against the bar

```bash
python3 eval_agent.py --model fantasy-draft --limit 60
```

`eval_agent.py` recomputes the board's score on exactly the examples it
evaluates, so the comparison is never apples-to-oranges. Fill this in:

| model | validity | mean rank (of 12) | best of 12 |
|---|---|---|---|
| random | 100% | 6.36 | 8.4% |
| deterministic board | 100% | 6.00 | 9.6% |
| `llama3.1:8b` (un-tuned base) — **the bar** | 100% | 5.80 | 11.1% |
| `fantasy-draft` (fine-tuned) | 100% | **5.41** | 15.1% |

The bar is the un-tuned base, not the board. A fine-tune that lands between
5.80 and 6.00 beats the tabular layer while being worse than the model it
started from, and `eval_agent.py` will still print "BEATS the board" for it.

Run the full 450 once the short runs look sane:

```bash
python3 eval_agent.py --model fantasy-draft
```

### How to read the result

**Validity first.** If the model names players that were never on the
shortlist, that is a broken answer, not a wrong pick, and no accuracy number
below it means anything. `eval_agent.py` reports it separately and deliberately
refuses to map near-misses onto real players.

**Then mean rank.** §11g: *"If the agent doesn't beat the number it was handed,
the language layer is decoration."* The board is 6.00. Under it is a real
result; over it is also a real result.

**A likely and publishable outcome is that the fine-tune does not beat 6.00.**
The board beats random by only 0.5 ranks of 12, which says most of the
separable signal is already in the VBD sort. If that is what happens, record
it — `CLAUDE.md` is explicit that negative results are the useful ones, and
"here is why the ceiling is where it is" is a stronger claim than a number
nobody can reproduce.

**Guard against the failure §11g names**: *"a fine-tune that produces more
confident-sounding advice without producing better advice."* Fluent output that
does not move mean rank is exactly that. Read a handful of completions
alongside the numbers.

---

## 5 · Record it

`CLAUDE.md`: commit after every verified result, score in the message, negative
results included.

```
agent: fine-tuned llama3.1-8b on 1950 draft decisions — mean rank X.XX of 12
(board 6.00, base X.XX, random 6.36) on 450 held-out picks, 2023-2025
```

Then update the table in `RESULTS.md`.
