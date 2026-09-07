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
#   export HF_TOKEN=hf_...          # gated base model config
#   ./finish_adapter.sh             # or: ./finish_adapter.sh my-model-name
set -euo pipefail

MODEL_NAME="${1:-fantasy-draft}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP="${LLAMA_CPP:-$(dirname "$PROJECT")/llama.cpp}"
CONDA_PY="${CONDA_PY:-/opt/miniconda3/envs/pllmpp/bin/python}"
MODAL="${MODAL:-/opt/miniconda3/envs/pllmpp/bin/modal}"
OUT="$PROJECT/adapter_real"
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
mkdir -p "$OUT"
for f in "${FILES[@]}"; do
    # Explicit destination path per file. Passing just the directory makes
    # Modal write the download AS that directory name.
    "$MODAL" volume get --force fantasy-lora "$f" "$OUT/$f" \
        || echo "  (skipped $f -- not in the volume)"
done
[ -s "$OUT/adapter_model.safetensors" ] \
    || die "adapter_model.safetensors did not download.
  Check the volume:  $MODAL volume ls fantasy-lora
  An empty listing means training never reached volume.commit()."

say "2/5  provenance -- confirm this is the run you think it is"
cat "$OUT/PROVENANCE.txt" 2>/dev/null || echo "  (no PROVENANCE.txt)"
ls -lh "$OUT"

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

say "5/5  scoring against the bar -- all 450 held-out picks"
echo "  the bar is 5.80 (un-tuned llama3.1:8b), NOT 6.00 (the board)."
echo "  landing between them means post-training made the model worse."
echo "  this takes roughly 75 minutes. Ctrl+C and add --limit 30 for a smoke test."
cd "$PROJECT"
"$CONDA_PY" eval_agent.py --model "$MODEL_NAME"

say "done -- record the result"
echo "  CLAUDE.md: commit after every verified result, score in the message,"
echo "  negative results included. Then update the table in RESULTS.md."
