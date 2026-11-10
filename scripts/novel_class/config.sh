# Cluster configuration for the novel-class experiments. Every value can be overridden from the
# environment (e.g. `LOGROOT=/other/dir bash submit.sh tier1`); submit.sh forwards the environment.

# Input volumes (data_train.npz, labels_train.npz, penobscot.h5)
: "${DATAPATH:=/snfs2/nunes/seismic}"
# Where the new runs are written: $LOGROOT/<dataset>_<partition>/fold<k>/<method>_<mode>/
: "${LOGROOT:=/snfs2/nunes/logs_novel_class}"
# Checkpoints of the paper (all classes seen), used only by reference_eval.sbatch.
#   PAPER_LAYOUT=legacy : directory names of the original /snfs2 runs (see ckpt_dir_legacy in reference_eval.sbatch)
#   PAPER_LAYOUT=repo   : layout written by scripts/run_experiments.sh ($PAPER_LOGDIR/<dataset>_<partition>/<method>_<mode>)
: "${PAPER_LAYOUT:=legacy}"
: "${PAPER_LOGDIR:=/snfs2/nunes}"

# Training schedule (paper, Sec. 4.1)
: "${NITER:=200}"
: "${SHOTS:=1 5}"
# Classes used for validation / best-checkpoint selection (see docs/novel_class_protocol.md)
: "${VAL_CLASSES:=base}"

# Python environment
: "${CONDA_ENV:=/sonic_home/nuneslima/envs/ifsl}"
activate_env() {
    [ -n "${SKIP_ENV_ACTIVATION:-}" ] && return 0   # environment already active (local runs)
    source /etc/profile.d/modules.sh
    module load anaconda3.2023.09-0
    conda activate "$CONDA_ENV"
}

# bench_args <dataset> <partition>  ->  main.py arguments selecting the partition
bench_args() {
    case $2 in
        lrp)  echo "--benchmark $1" ;;
        eds5) echo "--benchmark $1_eds --eds_slices 5" ;;
        eds1) echo "--benchmark $1_eds --eds_slices 1" ;;
        *) echo "unknown partition $2" >&2; return 1 ;;
    esac
}

# manifest_line <manifest> <array index>  ->  "dataset partition mode method fold" (line 1 is the header)
manifest_line() {
    awk -v n=$(( $2 + 2 )) 'NR == n' "$1"
}
