#!/bin/bash
# Seen-class reference (evaluation only, no training): the paper's checkpoints, trained with ALL classes,
# evaluated on exactly the same test episodes as the novel-class runs (same region, same fold classes,
# same seeds). The difference "reference - novel" isolates the cost of never having seen the class.
# Submit through submit.sh reference.
#SBATCH --job-name=fsss_ref
#SBATCH --partition=medusas_shr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -euo pipefail
: "${MANIFEST:?set by submit.sh}" "${CODE_DIR:?set by submit.sh}"
source "$CODE_DIR/scripts/novel_class/config.sh"
activate_env
cd "$CODE_DIR"

# Directory names of the original paper runs (from the ifsl/fs-cs SLURM files).
ckpt_dir_legacy() {  # <dataset> <partition> <mode> <method>
    local d=$1 p=$2 mode=$3 m=$4 suffix
    [ "$mode" = sssn ] && suffix=sssn || suffix=${mode}_support
    case "$p/$mode" in
        lrp/sssn)  echo "$PAPER_LOGDIR/logs_fs_cs_sssn/${d}_${m}_1way_1shot_400x400_200epoch_${suffix}" ;;
        lrp/*)     echo "$PAPER_LOGDIR/logs_fs_cs/${d}_${m}_1way_1shot_400x400_200epoch_${suffix}" ;;
        eds5/sssn) echo "$PAPER_LOGDIR/logs_fs_cs_eds_sssn/${d}_eds_${m}_1way_1shot_400x400_200epoch_${suffix}" ;;
        eds5/*)    echo "$PAPER_LOGDIR/logs_fs_cs_eds/${d}_eds_${m}_1way_1shot_400x400_200epoch_${suffix}" ;;
    esac
}

read -r dataset partition mode method fold <<< "$(manifest_line "$MANIFEST" "$SLURM_ARRAY_TASK_ID")"
bench=$(bench_args "$dataset" "$partition")
if [ "$PAPER_LAYOUT" = legacy ]; then
    ckpt=$(ckpt_dir_legacy "$dataset" "$partition" "$mode" "$method")
else
    ckpt="$PAPER_LOGDIR/${dataset}_${partition}/${method}_${mode}"
fi
out_dir="$LOGROOT/reference/${dataset}_${partition}/fold${fold}/${method}_${mode}"
mkdir -p "$out_dir"
echo "$ckpt" > "$out_dir/CHECKPOINT_SOURCE"
echo "task $SLURM_ARRAY_TASK_ID: $dataset $partition $mode $method fold=$fold <- $ckpt"

# Legacy PFENet checkpoints may contain a ResNet-50 (fs-cs built PFENet on ResNet-50 regardless of --backbone).
# Loading is strict, so a mismatch fails loudly; then resubmit with REF_PFENET_BACKBONE=resnet50.
backbone=resnet101
[ "$method" = pfenet ] && backbone=${REF_PFENET_BACKBONE:-resnet101}

for shot in $SHOTS; do
    out="$out_dir/eval_novel_${shot}shot.txt"
    grep -q "test/miou" "$out" 2>/dev/null && continue
    python main.py --datapath "$DATAPATH" $bench --method "$method" --support_mode "$mode" \
        --fold "$fold" --backbone "$backbone" --logpath "$ckpt" --shot "$shot" --eval > "$out" 2>&1
    grep "test/miou\|test/seg_macro_f1" "$out"
done
