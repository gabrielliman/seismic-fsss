"""Collects the novel-class results into per-fold tables (HSNet/iFSL style: fold columns + mean).

    python scripts/novel_class/collect_results.py --logroot /snfs2/nunes/logs_novel_class [--out results_novel]

For every (dataset, partition, shot) it writes a table with one row per (method, support mode) and columns
fold0..k, Mean (novel classes), Ref (paper checkpoint, all classes seen, same test episodes) and
Gap = Ref - Mean. Cells are "Macro F1 | mIoU", as in the paper. Missing runs are left blank.
Outputs: <out>/novel_class_results.csv (long format) and <out>/novel_class_tables.md.
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.class_split import NFOLDS, describe  # noqa: E402

METHODS = {'panet': 'PANet', 'pfenet': 'PFENet', 'hsnet': 'HSNet', 'asnet': 'ASNet'}
MODES = {'anywhere': 'A', 'same_slice': 'SS', 'sssn': 'SSSN'}
PARTITIONS = {'lrp': 'LRP', 'eds5': 'EDS (5 Slices)', 'eds1': 'EDS (1 Slice)'}
PATTERN = re.compile(r'(?P<ds>parihaka|penobscot)_(?P<part>\w+?)/fold(?P<fold>\d+)/(?P<method>[a-z]+)_(?P<mode>\w+)/'
                     r'eval_novel_(?P<shot>\d+)shot\.txt$')


def parse_log(path):
    metrics = {}
    with open(path, errors='ignore') as f:
        for line in f:
            for key in ('miou', 'seg_macro_f1'):
                if f'test/{key}' in line:
                    metrics[key] = float(line.split()[-1])
    return metrics if len(metrics) == 2 else None


def collect(root, kind, exclude=None):
    rows = []
    for path in glob.glob(os.path.join(root, '**', 'eval_novel_*shot.txt'), recursive=True):
        if exclude and os.path.abspath(path).startswith(os.path.abspath(exclude) + os.sep):
            continue
        m = PATTERN.search(path.replace(os.sep, '/'))
        metrics = parse_log(path) if m else None
        if metrics:
            rows.append(dict(kind=kind, dataset=m['ds'], partition=m['part'], fold=int(m['fold']),
                             method=m['method'], mode=m['mode'], shot=int(m['shot']), **metrics))
    return rows


def mean(values):
    return sum(values) / len(values) if values else None


def cell(f1, miou):
    return '' if f1 is None else f'{f1:.2f} \\| {miou:.2f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logroot', required=True)
    parser.add_argument('--out', default='results_novel')
    args = parser.parse_args()

    ref_root = os.path.join(args.logroot, 'reference')
    novel = collect(args.logroot, 'novel', exclude=ref_root)
    reference = collect(ref_root, 'reference')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'novel_class_results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['kind', 'dataset', 'partition', 'fold', 'method', 'mode', 'shot',
                                               'miou', 'seg_macro_f1'])
        writer.writeheader()
        writer.writerows(sorted(novel + reference, key=lambda r: tuple(str(r[k]) for k in writer.fieldnames)))

    index = {(r['kind'], r['dataset'], r['partition'], r['fold'], r['method'], r['mode'], r['shot']): r
             for r in novel + reference}
    md = ['# Novel-class (intra-survey) results', '',
          'Cells: Macro F1 | mIoU over the novel classes of each fold (1,000 test episodes per fold). '
          '**Mean** = average over folds (only when every fold finished). **Ref** = paper checkpoint trained with '
          'all classes, evaluated on the same episodes. **Gap** = Ref - Mean (mIoU).', '']
    for dataset in NFOLDS:
        md += [f'## {dataset.capitalize()}', ''] + [f'- {describe(dataset, k)}' for k in range(NFOLDS[dataset])] + ['']
        folds = range(NFOLDS[dataset])
        for partition in PARTITIONS:
            for shot in (1, 5):
                lines = []
                for method in METHODS:
                    for mode in MODES:
                        per_fold = [index.get(('novel', dataset, partition, k, method, mode, shot)) for k in folds]
                        refs = [index.get(('reference', dataset, partition, k, method, mode, shot)) for k in folds]
                        if not any(per_fold) and not any(refs):
                            continue
                        done = all(per_fold)
                        mf1 = mean([r['seg_macro_f1'] for r in per_fold]) if done else None
                        mmiou = mean([r['miou'] for r in per_fold]) if done else None
                        rdone = all(refs)
                        rf1 = mean([r['seg_macro_f1'] for r in refs]) if rdone else None
                        rmiou = mean([r['miou'] for r in refs]) if rdone else None
                        gap = f'{rmiou - mmiou:+.2f}' if done and rdone else ''
                        cells = [cell(r['seg_macro_f1'], r['miou']) if r else '' for r in per_fold]
                        lines.append(f'| {METHODS[method]} | {MODES[mode]} | ' + ' | '.join(cells) +
                                     f' | ' + (f'**{cell(mf1, mmiou)}**' if done else '') + f' | {cell(rf1, rmiou)} | {gap} |')
                if lines:
                    head = ' | '.join(f'fold{k}' for k in folds)
                    md += [f'### {PARTITIONS[partition]}, {shot}-shot', '',
                           f'| Model | Support | {head} | Mean | Ref | Gap |',
                           '|' + '---|' * (NFOLDS[dataset] + 5)] + lines + ['']
    with open(os.path.join(args.out, 'novel_class_tables.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'{len(novel)} novel-class and {len(reference)} reference evaluations -> {args.out}/')


if __name__ == '__main__':
    main()
