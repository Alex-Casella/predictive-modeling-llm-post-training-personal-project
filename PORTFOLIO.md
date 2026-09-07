# Portfolio entry

Resume-ready summary of this repository. Every number here is reproducible from
a script in it; see `RESULTS.md` for the full log, including the failures.

---

## Resume block

**Fantasy Football Projection + LLM Post-Training** — personal project
*Python, pandas, scikit-learn, PyTorch, PEFT/TRL, Modal, Hugging Face, Ollama,
llama.cpp, SciPy, Git*

- Built a season-level projection model over **6,500 player-seasons (2000–2025,
  Pro-Football-Reference)** and **84,909 player-weeks (nflverse)**, validating
  the weekly/season join by reconstruction — summing weekly PPR reproduced the
  independently-sourced season totals to a **median difference of 0.00**.
- Fixed the success metric and the baseline **before** modelling: set overlap at
  250, against a 68.8% carry-forward baseline measured across 25 walk-forward
  season pairs. Proved the metric **cannot be improved by reordering** and that
  a **76.8% structural ceiling** comes from rookies absent from the data — so
  most candidate "improvements" were ruled out by analysis rather than by
  running them.
- Post-trained **two QLoRA adapters on Llama 3.1 8B** (Modal A10G, 4-bit
  bitsandbytes, PEFT + TRL) for two different decisions — draft pick and weekly
  start/sit — from **2,850 and 1,774** examples built with strict no-look-ahead
  windows and season-based (never random) splits.
- Designed the evaluation so the answer was checkable rather than judged: *K*
  named candidates, one answer, labelled by hindsight, scored as mean rank with
  a known chance floor. **Scored the un-tuned base model first, every time** —
  on the draft task that moved the bar from 6.00 to 5.80 and turned a reported
  success into an honest one.
- Ran **paired significance tests (t, Wilcoxon, sign), confidence intervals and
  power analysis** on every claim, and reported the underpowered and negative
  results rather than the flattering ones.
- Converted PEFT adapters to **GGUF via llama.cpp** and served them locally
  through **Ollama**, then shipped an interactive draft-assistant CLI backed by
  the fine-tuned model.

---

## What I found

**The interesting result is that the two tasks disagree.** Same base model, same
harness, same recipe — opposite answers to "can a language model beat the
spreadsheet?"

| at 1 epoch | tabular layer | un-tuned Llama | fine-tuned |
|---|---|---|---|
| draft (mean rank of 12) | 6.00 | 5.80 | **5.41** — beats both |
| start/sit (mean rank of 6) | 2.92 | 3.02 | **2.99** — beats neither |

So "does post-training help?" has no single answer; it is task-dependent, and
that was **measured rather than assumed**. Supporting findings:

- The projection model beats its baseline by **0.3 points (69.1% vs 68.8%)**.
  That is a small number and it is the honest one — the analysis above explains
  exactly why, and points at the one input (NFL draft position) that could move
  it.
- The draft gap is **suggestive but underpowered**: p = 0.047 / 0.054 / 0.090
  across three tests, needing 895 paired examples against the 450 available.
- The start/sit fine-tune **changed nothing measurable** at either epoch count —
  three paired comparisons, three intervals containing zero.
- A defect became a result: the start/sit adapter never emits its stop token and
  enumerates every candidate. Ruled out over-training as the cause (runaway went
  *up* at fewer epochs), leaving a data-collation hypothesis —
  `pad_token = eos_token` masks the real EOS out of the training loss.

---

## What I was actually learning

The fantasy football is a vehicle. The subject is **post-training: how to make a
general model better at one specific decision, and how to know whether it
worked.**

Making an LLM produce fluent fantasy advice is easy and worthless — it will
sound confident whether or not it is right. Every design decision here exists to
make the question answerable instead:

| choice | what it buys |
|---|---|
| K named candidates, one answer | the answer is checkable, not judged |
| labelled by hindsight from the K shown | a right answer exists per decision |
| held-out seasons, never random splits | adjacent decisions cannot leak |
| the un-tuned base scored **first** | separates "tuning helped" from "it could already do this" |
| a deterministic sort scored on the same items | separates the language layer from the logic under it |
| the bar fixed before the result exists | stops the goalposts moving |

The second lesson was about the harness rather than the model. An epoch ablation
"succeeded" twice and produced a plausible number both times — while actually
re-scoring the *previous* adapter, because a stale file download was silently
tolerated. At temperature 0, two identical eval outputs are proof of identical
weights, not of a reproducible finding. The pipeline now asserts provenance,
keys each run to its own path, and refuses to score a model whose content hash
matches one already installed.

---

## Skills, and what I actually did with each

| | |
|---|---|
| **Python / pandas / NumPy** | 25 scripts, 63 commits; every dataset built and validated in-repo |
| **scikit-learn** | `HistGradientBoostingRegressor` / `Classifier`, walk-forward evaluation harness |
| **SciPy** | paired t-test, Wilcoxon signed-rank, sign test, CIs, power analysis |
| **PyTorch / transformers / PEFT / TRL** | QLoRA SFT, LoRA rank/alpha/target-module config, 4-bit `bitsandbytes` |
| **Modal** | serverless A10G training, persistent volumes, secrets, image definitions, CPU pre-flight jobs |
| **Hugging Face** | gated model access, token scopes, hub auth failures (401 vs 403) |
| **llama.cpp** | PEFT safetensors → GGUF conversion in an isolated venv |
| **Ollama** | `Modelfile` / `ADAPTER` serving, HTTP API, content-hash model identity |
| **nflverse / `nfl_data_py`** | weekly stats, player ID crosswalks, NFL draft picks |
| **Data integrity** | identifier-only joins — 10 names in this data belong to two different players; joining on name silently merges two careers |
| **Experimental design** | walk-forward validation, pre-registered baselines, ablations, seed replication, leakage audits |
| **Bash / Git** | reproducible one-command pipelines; commit log used as the experiment log, score in every message |

---

Data: Pro-Football-Reference and nflverse. See `ATTRIBUTION.md`.
