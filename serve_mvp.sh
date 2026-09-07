#!/usr/bin/env bash
# Make a scored adapter USABLE without retraining it or invalidating its score.
#
#   ./serve_mvp.sh fantasy-sit-e1
#   ./serve_mvp.sh fantasy-draft
#
# THE PROBLEM
#
# The start/sit adapter answers correctly and then does not stop. 96% of its
# answers ran to the token cap, continuing past the recommendation to invent a
# rejection for every other candidate:
#
#   Start Isiah Pacheco (RB). He is averaging 14.5 ... larger sample.   <- wanted
#   Do not start Marvin Jones (WR); ...          <- "Do not start" appears NOWHERE
#   Do not start George Kittle (TE); ...            in the training data
#   ...                                             until the cap cuts it off
#
# Scoring is unaffected -- eval_agent.py reads the first shortlisted name and
# the recommendation comes first -- so 2.99 is a real measurement of the
# decision. But six paragraphs of invented text is not a usable product.
#
# WHY A SEPARATE MODEL AND NOT AN EDIT
#
# `fantasy-sit-e1` and the number 2.99 are a matched pair. Adding parameters to
# that model would leave RESULTS.md quoting a score for a model that no longer
# exists, and nothing in the repo would record the difference. So this builds
# `<name>-mvp` alongside it: same weights, different generation settings. The
# measured model stays exactly as measured.
#
# THIS IS A SERVING WORKAROUND, NOT A FIX
#
# The model still has not learned its stop token; the stop string just hides
# that from the caller. The actual fix is `--pad-token auto` in
# train_adapter.py, which keeps EOS in the training loss. Use both: the
# workaround today, the retrain when it finishes. Do not let this script stop
# you recording the defect -- RESULTS.md keeps it as a finding.
set -euo pipefail

SRC="${1:?usage: ./serve_mvp.sh <existing ollama model>   e.g. fantasy-sit-e1}"
MVP="${SRC}-mvp"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PY="${CONDA_PY:-/opt/miniconda3/envs/pllmpp/bin/python}"

# Cuts the runaway at the string it was observed to start with. Every trained
# answer is ONE paragraph, so a paragraph break is also already past the
# recommendation -- but whether Ollama expands the \n in a quoted stop string
# or takes it literally is a version detail I have not verified, so it is
# listed LAST and the two literal-text stops carry the work.
#
# num_predict is the backstop for a continuation that starts some other way.
# 120 tokens comfortably holds the longest training answer.
STOPS=("Do not start" "Do not " '\n\n')
LIMIT="${LIMIT:-120}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v ollama >/dev/null || die "ollama not on PATH"
ollama list | awk 'NR>1 {print $1}' | grep -qx "$SRC:latest" \
    || die "$SRC is not installed. ollama list, then pass a name from it."

say "building $MVP from $SRC"
MF="$(mktemp -d)/Modelfile"
{
    echo "FROM $SRC"
    echo
    echo "PARAMETER temperature 0"
    echo "PARAMETER num_predict $LIMIT"
    for s in "${STOPS[@]}"; do
        # Double quotes, which is what the Modelfile grammar wants. Shell
        # quoting (%q) writes Do\ not\ start, and Ollama would take the
        # backslashes literally and never match.
        printf 'PARAMETER stop "%s"\n' "$s"
    done
} > "$MF"
cat "$MF"
ollama create "$MVP" -f "$MF"

say "one real prompt through both, so you can see the difference"
cd "$PROJECT"
"$CONDA_PY" - "$SRC" "$MVP" <<'PY'
import json
import sys

from eval_agent import ask_ollama

src, mvp = sys.argv[1], sys.argv[2]
test = 'sit_test.jsonl' if 'sit' in src else 'sft_test.jsonl'
item = json.loads(open(test).readline())

for name in (src, mvp):
    text, capped = ask_ollama(name, item, timeout=180)
    print(f'\n--- {name} ---')
    print(text.strip()[:900])
    print(f'  [{len(text.split())} words, '
          f'{"HIT THE CAP" if capped else "stopped on its own"}]')
PY

say "what you just saw"
cat <<NOTE
  The top answer is what the model does unaided. The bottom is the same
  weights with generation stopped at the point the runaway begins.

  If the bottom one is still long, the continuation starts with a string not in
  the STOPS list. Read it, add the string, re-run -- that is the whole
  maintenance loop for this workaround.

  Point the CLI at it:   python3 assistant.py --model $MVP

  And keep the un-suffixed $SRC installed. It is the model RESULTS.md quotes.
NOTE
