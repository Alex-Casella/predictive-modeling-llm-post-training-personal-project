#!/usr/bin/env bash
# One command per ablation: train -> download -> convert -> serve -> score ->
# paired test against the run it is an ablation OF.
#
#   ./run_ablation.sh sit 1 e1      # the 1-epoch start/sit ablation
#   ./run_ablation.sh draft 2 e2    # 2 epochs on draft, which was trained at 1
#
#   $1  dataset   draft | sit        (must match train_adapter.py DATASETS)
#   $2  epochs    the value being varied
#   $3  tag       short, unique, goes in the volume path and the model name
#
# WHY THIS EXISTS
#
# An ablation is only worth running if a wrong answer is impossible to mistake
# for a right one, and the last attempt failed that test: the volume was never
# overwritten, the previous adapter was downloaded, converted, served under a
# new name and scored, and it produced a plausible number. The eval was
# BYTE-IDENTICAL to the run it was supposedly being compared against and
# nothing in the pipeline said so.
#
# Four independent things now have to agree before a number is produced:
#
#   1. the volume path        /adapter/sit_e1 -- a run that did not commit
#                             leaves no directory, so the download fails
#   2. PROVENANCE epochs      asserted against $2, not printed for a human
#   3. the Ollama content ID  a hash of the WEIGHTS; matching an installed
#                             model means the same model, whatever the files say
#   4. the eval filename      eval_sit_fantasy-sit-e1.csv, its own file
#
# 1-3 are checks the previous version did not have. Any one of them would have
# caught what happened.
#
# WHAT IT DOES NOT DO
#
# It does not decide anything. It ends by running paired_test.py and printing
# what the three p-values would mean; reading them is yours. It also does not
# touch RESULTS.md -- CLAUDE.md wants the score in a commit message, and a
# script that writes results is a script that can write a wrong one.
#
# Roughly 40 min training + 30 min scoring for sit, 40 + 75 for draft.
set -euo pipefail

DATASET="${1:?usage: ./run_ablation.sh <dataset> <epochs> <tag>}"
EPOCHS="${2:?epochs, e.g. 1}"
TAG="${3:?a short tag, e.g. e1 -- it keys the volume path and the model name}"

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PY="${CONDA_PY:-/opt/miniconda3/envs/pllmpp/bin/python}"
MODAL="${MODAL:-/opt/miniconda3/envs/pllmpp/bin/modal}"

SUBDIR="${DATASET}_${TAG}"
MODEL_NAME="fantasy-${DATASET}-${TAG}"

# The run this is an ablation OF: the untagged adapter for the same dataset,
# and the eval CSV it wrote. eval_agent.py only prefixes non-default tasks, so
# draft keeps the bare filename.
BASE_MODEL="fantasy-${DATASET}"
case "$DATASET" in
    draft) BASE_CSV="eval_${BASE_MODEL}.csv" ;;
    *)     BASE_CSV="eval_${DATASET}_${BASE_MODEL}.csv" ;;
esac
NEW_CSV="eval_$([ "$DATASET" = draft ] || printf '%s_' "$DATASET")${MODEL_NAME}.csv"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "0/3  preflight -- everything that can fail cheaply, before the GPU bill"

# Each of these has cost a run. HF_TOKEN in particular fails 40 minutes in,
# at the conversion step, long after training has been paid for.
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set. The base model's config is
  gated, and finish_adapter.sh cannot convert without it. Get a Read token at
  https://huggingface.co/settings/tokens then:  export HF_TOKEN=hf_...
  Type it in THIS TERMINAL. Never paste a token into a chat window."
[ -x "$MODAL" ] || die "modal not found at $MODAL (set MODAL=/path/to/modal)"
[ -x "$CONDA_PY" ] || die "python not found at $CONDA_PY (set CONDA_PY=...)"
[ -f "$PROJECT/finish_adapter.sh" ] || die "finish_adapter.sh is missing"
command -v ollama >/dev/null || die "ollama not on PATH"
ollama list >/dev/null 2>&1 || die "ollama is not serving. Run 'ollama serve'
  in another terminal. (If it says 'address already in use' it is ALREADY
  running and you can ignore that terminal.)"

# An ablation with nothing to compare against is just another run. Check now,
# not after 70 minutes.
[ -f "$PROJECT/$BASE_CSV" ] || die "$BASE_CSV does not exist.
  An ablation is a COMPARISON. Score the un-ablated run first:
    EXPECT_EPOCHS=<its epochs> ./finish_adapter.sh $BASE_MODEL $DATASET"
[ ! -f "$PROJECT/$NEW_CSV" ] || die "$NEW_CSV already exists.
  Delete it or pick a different tag. Overwriting it would destroy the only
  record of the earlier run with this name."

printf '  dataset      %s\n  epochs       %s\n  volume path  fantasy-lora/%s\n' \
       "$DATASET" "$EPOCHS" "$SUBDIR"
printf '  ollama model %s\n  compare vs   %s\n  writes       %s\n' \
       "$MODEL_NAME" "$BASE_CSV" "$NEW_CSV"

# Installed models sharing a content ID are the same weights under two names.
# Printed rather than enforced here -- the stale fantasy-sit-1ep and
# fantasy-sit-2ep from the failed ablation are expected to show up as twins of
# fantasy-sit, and seeing them is the point.
say "installed models, grouped by content ID"
ollama list | awk 'NR==1 {next} {n[$2]=n[$2] " " $1}
                   END {for (i in n) printf "  %s %s\n", i, n[i]}'
echo "  Two names on one ID are ONE model. Nothing to fix; just do not score"
echo "  both and call it a comparison."

say "1/3  training on Modal -- $DATASET, $EPOCHS epoch(s), tag $TAG"
echo "  Do not Ctrl+C. output_dir is container-local and volume.commit() runs"
echo "  only after trainer.train() returns, so an interrupted run leaves"
echo "  NOTHING recoverable -- not a checkpoint, not a partial adapter."
cd "$PROJECT"
"$MODAL" run train_adapter.py \
    --dataset "$DATASET" --epochs "$EPOCHS" --tag "$TAG"

say "2/3  download, convert, serve and score"
EXPECT_EPOCHS="$EPOCHS" ./finish_adapter.sh "$MODEL_NAME" "$SUBDIR"

[ -f "$PROJECT/$NEW_CSV" ] || die "the eval did not write $NEW_CSV.
  Training and conversion succeeded, so the adapter is fine -- re-run just
  the scoring:
    $CONDA_PY eval_agent.py --model $MODEL_NAME \\
        --test ${DATASET}_test.jsonl --key start_sit_examples.jsonl"

say "3/3  is the difference real, or is it noise?"
echo "  Same test items, both models -> paired. See paired_test.py for why"
echo "  that is both the correct test and the more sensitive one."
"$CONDA_PY" paired_test.py --a "$BASE_CSV" --b "$NEW_CSV"

say "what this can and cannot tell you"
cat <<'NOTE'
  This compares TWO FINE-TUNES OF THE SAME BASE at different epoch counts.
  It answers "did the extra epoch help", and nothing else.

  It does NOT answer "is the fine-tune worth doing" -- that is the comparison
  against the un-tuned base, which for sit is eval_sit_llama3.1_8b.csv at
  3.02, and against the tabular bar at 2.92. Run those pairings too:

    python3 paired_test.py --a eval_sit_llama3.1_8b.csv --b <the new csv>
    python3 paired_test.py --a eval_sit_board.csv       --b <the new csv>

  Then commit, score in the message, per CLAUDE.md -- including if it is worse.
NOTE
