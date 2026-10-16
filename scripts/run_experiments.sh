#!/usr/bin/env bash
# Reproduces the experiment grid of the paper:
#   datasets   : Parihaka, Penobscot
#   partitions : LRP, EDS (5 slices), EDS (1 slice)
#   support    : Anywhere, Same Slice, SSSN
#   models     : PANet, PFENet, HSNet, ASNet
# Every configuration is trained once (1-shot episodes) and evaluated with 1-shot and 5-shot
# support on the test partition of the same dataset (Tables 2 and 3) and, for LRP and EDS (5 slices),
# on the whole volume of the other dataset (transfer learning, Tables 4 and 5). Untrained models
# ("Baseline - Random" columns of Tables 4 and 5) are also saved and evaluated on both volumes.
#
# Usage: DATAPATH=/path/to/seismic [LOGDIR=logs] [METHODS="..."] [MODES="..."] bash scripts/run_experiments.sh
set -euo pipefail

DATAPATH=${DATAPATH:?Set DATAPATH to the directory with data_train.npz, labels_train.npz and penobscot.h5}
LOGDIR=${LOGDIR:-logs}
METHODS=${METHODS:-"panet pfenet hsnet asnet"}
MODES=${MODES:-"anywhere same_slice sssn"}

cd "$(dirname "$0")/.."

# evaluate <logpath> <method> <mode> <output tag> <benchmark args...>
evaluate() {
    local logpath=$1 method=$2 mode=$3 tag=$4
    shift 4
    for shot in 1 5; do
        python main.py --datapath "$DATAPATH" "$@" --method "$method" --support_mode "$mode" \
            --logpath "$logpath" --shot "$shot" --eval > "$logpath/eval_${tag}_${shot}shot.txt"
    done
}

for method in $METHODS; do
    for mode in $MODES; do

        for dataset in parihaka penobscot; do
            target=$([ "$dataset" = parihaka ] && echo penobscot || echo parihaka)

            for partition in lrp eds5 eds1; do
                case $partition in
                    lrp)  bench=(--benchmark "$dataset") ;;
                    eds5) bench=(--benchmark "${dataset}_eds" --eds_slices 5) ;;
                    eds1) bench=(--benchmark "${dataset}_eds" --eds_slices 1) ;;
                esac
                logpath=$LOGDIR/${dataset}_${partition}/${method}_${mode}

                # Train (1-shot episodes, 200 epochs) and evaluate on the test partition
                python main.py --datapath "$DATAPATH" "${bench[@]}" --method "$method" \
                    --support_mode "$mode" --logpath "$logpath" --niter 200
                evaluate "$logpath" "$method" "$mode" test "${bench[@]}"

                # Transfer learning: evaluate on the whole volume of the other dataset
                if [ "$partition" != eds1 ]; then
                    evaluate "$logpath" "$method" "$mode" "transfer_${target}" --benchmark "full_${target}"
                fi
            done
        done

        # Baseline - Random: ImageNet backbone + randomly initialised few-shot modules (no training)
        logpath=$LOGDIR/random/${method}_${mode}
        python main.py --datapath "$DATAPATH" --benchmark full_parihaka --method "$method" \
            --support_mode "$mode" --logpath "$logpath" --niter 0
        for dataset in parihaka penobscot; do
            evaluate "$logpath" "$method" "$mode" "full_${dataset}" --benchmark "full_${dataset}"
        done
    done
done
