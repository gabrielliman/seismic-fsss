#!/usr/bin/env bash
# Status of a tier: SLURM state of every task (latest submission) and whether its results exist.
#   bash scripts/novel_class/status.sh            # tier1
#   bash scripts/novel_class/status.sh tier2
#   DAYS=30 bash scripts/novel_class/status.sh    # look further back in the SLURM history (default 14 days)
tier=${1:-tier1}
CODE_DIR=$(cd "$(dirname "$0")/../.." && pwd)
source "$CODE_DIR/scripts/novel_class/config.sh"
MANIFEST="$CODE_DIR/scripts/novel_class/manifests/$tier.tsv"
name=fsss_novel; [ "$tier" = reference ] && name=fsss_ref

echo "== Queue (yours)"
squeue -u "$USER" -o "%.18i %.10P %.10j %.2t %.10M %R" 2>/dev/null
echo
# latest SLURM state per array index: "idx state elapsed node"
declare -A st
while read -r jid state elapsed node; do
    idx=${jid##*_}; [[ $idx =~ ^[0-9]+$ ]] || continue
    st[$idx]="$state $elapsed $node"
done < <(sacct -u "$USER" --name="$name" -S "now-${DAYS:-14}days" -X -n -P -o JobID,State,Elapsed,NodeList 2>/dev/null \
         | sort -t_ -k1,1n | tr '|' ' ')

printf '%-3s %-9s %-5s %-10s %-6s %-5s %-24s %s\n' idx dataset part mode method fold "results" "last SLURM state"
done_n=0; total=0; failed=()
while read -r i ds part mode m f; do
    total=$((total+1))
    if [ "$tier" = reference ]; then d=$LOGROOT/reference/${ds}_${part}/fold$f/${m}_$mode; else d=$LOGROOT/${ds}_${part}/fold$f/${m}_$mode; fi
    if grep -q test/miou "$d/eval_novel_5shot.txt" 2>/dev/null; then r=DONE; done_n=$((done_n+1))
    elif [ -f "$d/last.ckpt" ]; then r="partial ($(grep -ao 'ep: *[0-9]*' "$d/train.log" 2>/dev/null | tail -1))"
    else r="not started"; fi
    s=${st[$i]:-"-"}
    printf '%-3s %-9s %-5s %-10s %-6s %-5s %-24s %s\n' "$i" "$ds" "$part" "$mode" "$m" "$f" "$r" "$s"
    [[ $r != DONE && $s =~ ^(FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|CANCELLED) ]] && failed+=("$i")
done < <(awk 'NR > 1' "$MANIFEST" | nl -v0 -w1 -s' ')

echo
echo "== $tier: $done_n / $total done"
if [ ${#failed[@]} -gt 0 ]; then
    list=$(IFS=,; echo "${failed[*]}")
    echo "== Not done and last SLURM run ended badly: $list"
    echo "   last error lines of the first one:"
    i=${failed[0]}; read -r ds part mode m f <<< "$(manifest_line "$MANIFEST" "$i")"
    tail -3 "$LOGROOT/${ds}_${part}/fold$f/${m}_$mode/train.log" 2>/dev/null | sed 's/^/   /'
    echo "   resubmit with: GRES= SBATCH_EXTRA=\"--partition=gorgonas\" MAX_PARALLEL=3 bash scripts/novel_class/submit.sh $tier $list"
fi
