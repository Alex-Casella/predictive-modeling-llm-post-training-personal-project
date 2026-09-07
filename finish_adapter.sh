#!/usr/bin/env bash
# Stage 5, end to end: Modal volume -> GGUF -> Ollama -> scored against the bar.
#
# Every step here was executed by hand first and the flags that cost round trips
# are baked in:
#
#   * the volume is MOUNTED at /adapter, so its own root IS that directory --
#     copy from `/`, not from `/adapter`
#   * `modal volume get` needs an explicit destination FILE per file, or it
#     writes the whole download as one file named after the destination
#   * convert_lora_to_gguf.py needs --base-model-id, NOT --base. `--base` wants
#     a local directory of base-model files and dies with FileNotFoundError on
#     a Hugging Face repo id
#   * the conversion MUST run in its own venv. llama.cpp pins numpy~=1.26 and
#     torch==2.11; installing that into the project env downgrades numpy and
#     pandas and breaks every script in this repo
#   * Ollama cannot load the PEFT safetensors directly (5 variants tried, see
#     SERVING.md), so GGUF is required rather than optional
#
# Usage:
#   export HF_TOKEN=hf_...            # gated base model config
#   ./finish_adapter.sh fantasy-draft draft
#   ./finish_adapter.sh fantasy-sit   sit
#
# $1 is the Ollama model name to create, $2 the subdirectory train_adapter.py
# wrote inside the volume (one per dataset). Passing no $2 reads the volume
# ROOT, which is where the very first run landed before the script grew a
# --dataset flag.
set -euo pipefail

MODEL_NAME="${1:-fantasy-draft}"
SUBDIR="${2:-}"
REMOTE="${SUBDIR:+$SUBDIR/}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP="${LLAMA_CPP:-$(dirname "$PROJECT")/llama.cpp}"
CONDA_PY="${CONDA_PY:-/opt/miniconda3/envs/pllmpp/bin/python}"
MODAL="${MODAL:-/opt/miniconda3/envs/pllmpp/bin/modal}"
OUT="$PROJECT/adapter_${SUBDIR:-real}"
BASE_ID="meta-llama/Llama-3.1-8B-Instruct"
OLLAMA_BASE="llama3.1:8b"

FILES=(adapter_model.safetensors adapter_config.json PROVENANCE.txt
       tokenizer.json tokenizer_config.json special_tokens_map.json)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set.
  The base model's config is gated. Get a Read token at
  https://huggingface.co/settings/tokens then:  export HF_TOKEN=hf_..."
[ -d "$LLAMA_CPP" ] || die "llama.cpp not found at $LLAMA_CPP
  git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP
  (or set LLAMA_CPP=/path/to/llama.cpp)"
command -v ollama >/dev/null || die "ollama not on PATH. brew install ollama"

say "1/5  downloading the adapter from the Modal volume"

# What is actually IN the volume, with timestamps. If these are older than the
# training run you just launched, the run did not write and everything below
# would faithfully re-measure the previous adapter.
"$MODAL" volume ls "fantasy-lora${SUBDIR:+/$SUBDIR}" || true

# WIPE FIRST. A download that fails leaves the previous run's files in place,
# and the rest of this script would then convert, serve and score the OLD
# adapter while reporting the new model's name. That happened: a 1-epoch rerun
# produced byte-identical results to the 2-epoch run because the volume was
# never overwritten and the stale local copy was still here.
rm -rf "$OUT"
mkdir -p "$OUT"

# These three MUST arrive. Tokenizer files are optional -- GGUF conversion
# reads the base model's config for those.
REQUIRED="adapter_model.safetensors adapter_config.json PROVENANCE.txt"
for f in "${FILES[@]}"; do
    # Explicit destination path per file. Passing just the directory makes
    # Modal write the download AS that directory name.
    if ! "$MODAL" volume get --force fantasy-lora "$REMOTE$f" "$OUT/$f"; then
        case " $REQUIRED " in
            *" $f "*) die "could not download $REMOTE$f from the volume.
  Nothing is left in $OUT, so no stale adapter can be scored by mistake.
  Check:  $MODAL volume ls fantasy-lora${SUBDIR:+/$SUBDIR}" ;;
            *) echo "  (optional file $f not in the volume, continuing)" ;;
        esac
    fi
done
[ -s "$OUT/adapter_model.safetensors" ] \
    || die "adapter_model.safetensors did not download.
  Check the volume:  $MODAL volume ls fantasy-lora/$SUBDIR
  An empty listing means training never reached volume.commit()."

say "2/5  provenance -- confirm this is the run you think it is"
cat "$OUT/PROVENANCE.txt"
ls -lh "$OUT"
echo
echo "  READ THE LINE ABOVE. dataset and epochs must be the run you just"
echo "  launched. Two adapters that differ only in a hyperparameter produce"
echo "  IDENTICAL eval output when the wrong one is scored, and nothing"
echo "  further down will notice."

say "3/5  converting PEFT -> GGUF in an isolated venv"
cd "$LLAMA_CPP"
if [ ! -d gguf-convert-env ]; then
    echo "  creating gguf-convert-env (one time, a few minutes)"
    "$CONDA_PY" -m venv gguf-convert-env
    # shellcheck disable=SC1091
    source gguf-convert-env/bin/activate
    pip install -q -r requirements.txt
else
    # shellcheck disable=SC1091
    source gguf-convert-env/bin/activate
fi
python convert_lora_to_gguf.py "$OUT" \
    --outfile "$OUT/adapter.gguf" \
    --base-model-id "$BASE_ID"
deactivate
[ -s "$OUT/adapter.gguf" ] || die "conversion produced no adapter.gguf"

say "4/5  building the Ollama model"
cd "$OUT"
cat > Modelfile <<MODELFILE
FROM $OLLAMA_BASE
ADAPTER ./adapter.gguf

PARAMETER temperature 0
MODELFILE
cat Modelfile
ollama create "$MODEL_NAME" -f Modelfile
ollama list | head -5

say "5/5  scoring against the bar"
cd "$PROJECT"
if [ "$SUBDIR" = "sit" ]; then
    echo "  278 held-out start/sit decisions, seasons 2023-2024."
    echo "  the bar is 2.92 (start the best season average) out of 6."
    echo "  also score llama3.1:8b the same way before believing any gain."
    "$CONDA_PY" eval_agent.py --model "$MODEL_NAME" \
        --test sit_test.jsonl --key start_sit_examples.jsonl
else
    echo "  450 held-out picks. The bar is 5.80 (un-tuned llama3.1:8b),"
    echo "  NOT 6.00 (the board) -- landing between them means post-training"
    echo "  made the model worse. Roughly 75 minutes; Ctrl+C and add"
    echo "  --limit 30 for a smoke test."
    "$CONDA_PY" eval_agent.py --model "$MODEL_NAME"
fi

say "done -- record the result"
echo "  CLAUDE.md: commit after every verified result, score in the message,"
echo "  negative results included. Then update the table in RESULTS.md."
