"""Writes the job manifests of the novel-class experiments (one line = one SLURM array task).

Run from the repository root:  python scripts/novel_class/make_manifests.py
See docs/novel_class_protocol.md for why each tier exists and in which order to run them.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.class_split import NFOLDS  # noqa: E402

METHODS = ['panet', 'pfenet', 'hsnet', 'asnet']
DATASETS = ['parihaka', 'penobscot']

# tier -> (partitions, support modes)
TIERS = {
    # Tier 1: best partition of the paper (EDS, 5 slices) with the two parameter-free support modes.
    'tier1': (['eds5'], ['anywhere', 'same_slice']),
    # Tier 2: LRP with the same modes -> does EDS > LRP survive with novel classes?
    'tier2': (['lrp'], ['anywhere', 'same_slice']),
    # Tier 3: learned support selection on both partitions -> Same Slice vs SSSN with novel classes.
    'tier3': (['eds5', 'lrp'], ['sssn']),
    # Reference (evaluation only): paper checkpoints (all classes seen) on the novel classes of each fold.
    'reference': (['eds5', 'lrp'], ['anywhere', 'same_slice', 'sssn']),
}


def rows(partitions, modes):
    for dataset, partition, mode, method in itertools.product(DATASETS, partitions, modes, METHODS):
        for fold in range(NFOLDS[dataset]):
            yield dataset, partition, mode, method, fold


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'manifests')
    os.makedirs(out_dir, exist_ok=True)
    for tier, (partitions, modes) in TIERS.items():
        lines = ['dataset\tpartition\tmode\tmethod\tfold']
        lines += ['\t'.join(map(str, r)) for r in rows(partitions, modes)]
        path = os.path.join(out_dir, f'{tier}.tsv')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{path}: {len(lines) - 1} tasks')


if __name__ == '__main__':
    main()
