#!/bin/bash
# One array task = one (dataset, partition, support mode, method, fold) of a manifest:
#   1) meta-train on the base classes of the fold (1-shot episodes, NITER epochs, resumes after a timeout);
#   2) evaluate the best checkpoint on the novel classes of the fold, test region, 1- and 5-shot.
# Submit through submit.sh, which sets MANIFEST, CODE_DIR and the array range.
#SBATCH --job-name=fsss_novel
#SBATCH --partition=medusas_shr
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -euo pipefail
: "${MANIFEST:?set by submit.sh}" "${CODE_DIR:?set by submit.sh}"
source "$CODE_DIR/scripts/novel_class/config.sh"
activate_env
pick_gpu
cd "$CODE_DIR"

read -r dataset partition mode method fold <<< "$(manifest_line "$MANIFEST" "$SLURM_ARRAY_TASK_ID")"
bench=$(bench_args "$dataset" "$partition")
run="$LOGROOT/${dataset}_${partition}/fold${fold}/${method}_${mode}"
timeout 120 mkdir -p "$run" && timeout 60 touch "$run/.write_test" && rm -f "$run/.write_test" \
    || give_up "cannot write to $run (filesystem unavailable?)"
common=(--datapath "$DATAPATH" $bench --method "$method" --support_mode "$mode"
        --fold "$fold" --val_classes "$VAL_CLASSES" --logpath "$run")
echo "task $SLURM_ARRAY_TASK_ID: $dataset $partition $mode $method fold=$fold -> $run"

# 1) Training on base classes. Skipped when already finished. A crash or a stall (watchdog) is retried
#    up to MAX_ATTEMPTS times, resuming from last.ckpt; see run_watched in config.sh.
attempt=1
while [ ! -f "$run/TRAINED" ]; do
    resume=()
    [ -f "$run/last.ckpt" ] && resume=(--resume)
    echo "$(date '+%F %T') training attempt $attempt/$MAX_ATTEMPTS ${resume[*]}"
    rc=0
    run_watched "$run/train.log" python main.py "${common[@]}" --niter "$NITER" "${resume[@]}" || rc=$?
    if [ "$rc" -eq 0 ]; then
        touch "$run/TRAINED"
        break
    fi
    echo "$(date '+%F %T') training attempt $attempt failed (exit $rc); last lines of train.log:"
    timeout 30 tail -c 3000 "$run/train.log" | tr '\r' '\n' | grep -av '^\s*$' | tail -4
    [ "$attempt" -ge "$MAX_ATTEMPTS" ] && give_up "training failed $attempt times (task $SLURM_ARRAY_TASK_ID, $run)"
    attempt=$((attempt + 1))
    sleep "$RETRY_WAIT_SEC"
done

# 2) Evaluation on the novel classes of the fold (1,000 fixed episodes per shot setting), same protection
for shot in $SHOTS; do
    out="$run/eval_novel_${shot}shot.txt"
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        grep -q "test/miou" "$out" 2>/dev/null && break
        rm -f "$out"
        run_watched "$out" python main.py "${common[@]}" --shot "$shot" --eval || sleep "$RETRY_WAIT_SEC"
    done
    grep "test/miou\|test/seg_macro_f1" "$out" || give_up "${shot}-shot evaluation failed $MAX_ATTEMPTS times ($out)"
done
