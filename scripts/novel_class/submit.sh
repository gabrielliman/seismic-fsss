#!/usr/bin/env bash
# Submits one manifest as a SLURM job array.
#   bash scripts/novel_class/submit.sh tier1            # run all tasks of tier1, MAX_PARALLEL at a time
#   bash scripts/novel_class/submit.sh tier1 0-3        # only some tasks (indices = data lines of the manifest, from 0)
#   bash scripts/novel_class/submit.sh reference
#   DRY_RUN=1 bash scripts/novel_class/submit.sh tier1  # print the tasks and the sbatch command only
# Extra sbatch options can be given in SBATCH_EXTRA (e.g. SBATCH_EXTRA="--mail-user=you@example.org").
# GPU request: GRES (default gpu:1). For partitions without GPU GRES, e.g. gorgonas:
#   GRES= SBATCH_EXTRA="--partition=gorgonas" MAX_PARALLEL=3 bash scripts/novel_class/submit.sh tier1
# The job then pins itself to the least-used GPU of the node (pick_gpu in config.sh).
set -euo pipefail

tier=${1:?usage: submit.sh <tier1|tier2|tier3|reference> [array range]}
CODE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST="$CODE_DIR/scripts/novel_class/manifests/$tier.tsv"
[ -f "$MANIFEST" ] || { echo "no manifest $MANIFEST" >&2; exit 1; }

ntasks=$(( $(wc -l < "$MANIFEST") - 1 ))
range=${2:-0-$(( ntasks - 1 ))}
MAX_PARALLEL=${MAX_PARALLEL:-8}
script=train_eval.sbatch
[ "$tier" = reference ] && script=reference_eval.sbatch

mkdir -p "$CODE_DIR/slurm_logs"
GRES=${GRES-gpu:1}
gres=()
[ -n "$GRES" ] && gres=(--gres="$GRES")
cmd=(sbatch --array="${range}%${MAX_PARALLEL}" "${gres[@]}" --export=ALL,MANIFEST="$MANIFEST",CODE_DIR="$CODE_DIR"
     ${SBATCH_EXTRA:-} --chdir="$CODE_DIR" "$CODE_DIR/scripts/novel_class/$script")

echo "$tier: $ntasks tasks in $MANIFEST, submitting array $range"
if [ -n "${DRY_RUN:-}" ]; then
    column -t "$MANIFEST" | awk 'NR == 1 {print "idx  " $0; next} {print NR - 2 "    " $0}'
    echo "${cmd[@]}"
else
    "${cmd[@]}"
fi
