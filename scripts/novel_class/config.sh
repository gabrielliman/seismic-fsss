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

# ImageNet backbone weights: keep torchvision's cache on the shared filesystem. The default (~/.cache/torch)
# is the node's home directory, which can be full (gorgona7: "No space left on device"). Download them once
# on the login node so array tasks do not all download at the same time (see docs/novel_class_protocol.md).
: "${TORCH_HOME:=/sonic_home/nuneslima/.cache/torch}"
export TORCH_HOME

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

# pick_gpu: on partitions without GPU GRES (e.g. gorgonas), SLURM does not set CUDA_VISIBLE_DEVICES, and the
# trainer would use every GPU of the node. Pin the job to the single GPU with the least memory in use.
pick_gpu() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "GPU assigned by SLURM: $CUDA_VISIBLE_DEVICES"
        return 0
    fi
    # small random delay so array tasks starting together on one node do not all pick the same GPU
    sleep $(( RANDOM % 60 ))
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "no GPU GRES: using least-used GPU $CUDA_VISIBLE_DEVICES on $(hostname)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
}

# ---------------------------------------------------------------------------------------------------
# Robustness against transient /snfs2 failures (observed: "OSError: [Errno 121] Remote I/O error" while
# writing train.log, after which the Python process hung until the SLURM time limit).
: "${MAX_ATTEMPTS:=3}"     # attempts per job; each retry resumes from last.ckpt. Outages of /snfs2 can last
                           # hours, so only brief errors are retried; after that the job stops (see give_up)
: "${STALL_SEC:=2700}"     # kill a run whose log has not been written for this long (45 min)
: "${POLL_SEC:=60}"        # watchdog check interval
: "${RETRY_WAIT_SEC:=300}" # pause before retrying

# give_up <message>: stop this task and keep the rest of the array from starting (and failing) while the
# problem lasts. The pending tasks are held, not cancelled; release them with
#   scontrol release <array job id>
# The task ends as FAILED, so SLURM sends the FAIL e-mail when MAIL_USER was set at submission.
give_up() {
    echo "$(date '+%F %T') GIVING UP: $*"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
        scontrol hold "$SLURM_ARRAY_JOB_ID" 2>&1 | grep -v "already held\|Job is no longer pending" || true
        echo "pending tasks of array $SLURM_ARRAY_JOB_ID are on hold; when the problem is fixed run:"
        echo "    scontrol release $SLURM_ARRAY_JOB_ID"
    fi
    exit 1
}

# run_watched <log file> <command...>: append the command's output to the log; kill the whole process
# group (including data-loader workers) if the log stops changing for STALL_SEC. Returns the command's
# exit code, or 124 when it was killed by the watchdog.
run_watched() {
    local log=$1; shift
    setsid "$@" >> "$log" 2>&1 &
    local pid=$! now mtime
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL_SEC"
        kill -0 "$pid" 2>/dev/null || break
        now=$(date +%s)
        mtime=$(timeout 30 stat -c %Y "$log" 2>/dev/null || echo 0)   # a hung filesystem counts as stale
        if (( now - mtime > STALL_SEC )); then
            echo "$(date '+%F %T') watchdog: no output in $log for $((now - mtime)) s, killing pid $pid"
            kill -TERM -- "-$pid" 2>/dev/null; sleep 30; kill -KILL -- "-$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
    done
    wait "$pid"
}
