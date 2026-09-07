# Stage 5 — from a trained adapter to a served model

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

**Try Ollama first** — if this works, skip to step 3:

```bash
cd adapter
# point ADAPTER at the safetensors directory instead of a .gguf
sed -i '' 's|ADAPTER ./adapter.gguf|ADAPTER .|' Modelfile
ollama create fantasy-draft -f Modelfile
```

**If that fails**, convert explicitly:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt
python3 convert_lora_to_gguf.py ../adapter --outfile ../adapter/adapter.gguf
cd ..
ollama create fantasy-draft -f adapter/Modelfile
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
| deterministic board — **the bar** | 100% | 6.00 | 9.6% |
| `llama3.1:8b` (un-tuned base) | | | |
| `fantasy-draft` (fine-tuned) | | | |

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
