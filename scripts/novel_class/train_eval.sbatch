#!/bin/bash
# One array task = one (dataset, partition, support mode, method, fold) of a manifest:
#   1) meta-train on the base classes of the fold (1-shot episodes, NITER epochs, resumes after a timeout);
#   2) evaluate the best checkpoint on the novel classes of the fold, test region, 1- and 5-shot.
# Submit through submit.sh, which sets MANIFEST, CODE_DIR and the array range.
#SBATCH --job-name=fsss_novel
#SBATCH --partition=medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -euo pipefail
: "${MANIFEST:?set by submit.sh}" "${CODE_DIR:?set by submit.sh}"
source "$CODE_DIR/scripts/novel_class/config.sh"
activate_env
cd "$CODE_DIR"

read -r dataset partition mode method fold <<< "$(manifest_line "$MANIFEST" "$SLURM_ARRAY_TASK_ID")"
bench=$(bench_args "$dataset" "$partition")
run="$LOGROOT/${dataset}_${partition}/fold${fold}/${method}_${mode}"
mkdir -p "$run"
common=(--datapath "$DATAPATH" $bench --method "$method" --support_mode "$mode"
        --fold "$fold" --val_classes "$VAL_CLASSES" --logpath "$run")
echo "task $SLURM_ARRAY_TASK_ID: $dataset $partition $mode $method fold=$fold -> $run"

# 1) Training on base classes (skipped when already finished; resumed from last.ckpt after a timeout)
if [ ! -f "$run/TRAINED" ]; then
    resume=()
    [ -f "$run/last.ckpt" ] && resume=(--resume)
    python main.py "${common[@]}" --niter "$NITER" "${resume[@]}" >> "$run/train.log" 2>&1
    touch "$run/TRAINED"
fi

# 2) Evaluation on the novel classes of the fold (1,000 fixed episodes per shot setting)
for shot in $SHOTS; do
    out="$run/eval_novel_${shot}shot.txt"
    grep -q "test/miou" "$out" 2>/dev/null && continue
    python main.py "${common[@]}" --shot "$shot" --eval > "$out" 2>&1
    grep "test/miou\|test/seg_macro_f1" "$out"
done
