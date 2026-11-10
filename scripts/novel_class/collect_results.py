"""Collects the novel-class results into per-fold tables (HSNet/iFSL style: fold columns + mean).

    python scripts/novel_class/collect_results.py --logroot /snfs2/nunes/logs_novel_class [--out results_novel]

For every (dataset, partition, shot) it writes a table with one row per (method, support mode) and columns
fold0..k, Mean (novel classes), Ref (paper checkpoint, all classes seen, same test episodes) and
Gap = Ref - Mean. Cells are "Macro F1 | mIoU", as in the paper. Missing runs are left blank.
Outputs: <out>/novel_class_results.csv (long format), <out>/novel_class_tables.md and, for the paper,
<out>/table_novel_{main,gap,folds}_{parihaka,penobscot}.tex (booktabs; needs \\usepackage{booktabs,multirow,graphicx}).
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.class_split import NFOLDS, CLASS_NAMES, describe, novel_classes  # noqa: E402

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
    for path in write_latex(index, args.out):
        print('LaTeX:', path)
    print(f'{len(novel)} novel-class and {len(reference)} reference evaluations -> {args.out}/')


# ----------------------------------------------------------------------------------------------------------------
# LaTeX tables in the format of the paper (booktabs, "Macro F1|mIoU", best per column and shot in bold)
# ----------------------------------------------------------------------------------------------------------------
TEX_PARTITIONS = ['lrp', 'eds5']
TEX_MODES = ['anywhere', 'same_slice', 'sssn']
TEX_NAMES = {'parihaka': 'Parihaka', 'penobscot': 'Penobscot'}


def fold_mean(index, kind, dataset, partition, method, mode, shot):
    """(Macro F1, mIoU) averaged over folds, or None unless every fold has finished."""
    runs = [index.get((kind, dataset, partition, k, method, mode, shot)) for k in range(NFOLDS[dataset])]
    if not all(runs):
        return None
    return mean([r['seg_macro_f1'] for r in runs]), mean([r['miou'] for r in runs])


def tex_cell(value, bold_f1=False, bold_miou=False):
    if value is None:
        return '--'
    f1, miou = f'{value[0]:.2f}', f'{value[1]:.2f}'
    return (rf'\textbf{{{f1}}}' if bold_f1 else f1) + '$|$' + (rf'\textbf{{{miou}}}' if bold_miou else miou)


def tex_main_table(index, dataset):
    """Same layout as Tables 2-3 of the paper, cells = mean over the folds of the novel-class runs."""
    name = TEX_NAMES[dataset]
    cols = [(p, m) for p in TEX_PARTITIONS for m in TEX_MODES]
    lines = [r'\begin{table*}[ht]', r'\centering',
             rf'\caption{{Intra-survey novel-class results for Macro F1$|$mIoU ({name}), mean over '
             rf'{NFOLDS[dataset]} class folds; test facies are never seen during training. A: Anywhere; SS: Same Slice; '
             r'SSSN: Siamese Support Selection Network. Best result per column and shot setting in bold}',
             rf'\label{{tab:novel_{dataset}}}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{ll' + 'c' * len(cols) + '}', r'\toprule',
             r' & & \multicolumn{3}{c}{LRP} & \multicolumn{3}{c}{EDS (5 Slices)} \\',
             r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
             r'Shot & Model & A & SS & SSSN & A & SS & SSSN \\', r'\midrule']
    for shot in (1, 5):
        values = {(meth, c): fold_mean(index, 'novel', dataset, c[0], meth, c[1], shot) for meth in METHODS for c in cols}
        best = {}
        for c in cols:
            col = [values[(meth, c)] for meth in METHODS if values[(meth, c)]]
            best[c] = (max(v[0] for v in col), max(v[1] for v in col)) if col else (None, None)
        for i, meth in enumerate(METHODS):
            cells = [tex_cell(values[(meth, c)],
                              values[(meth, c)] is not None and values[(meth, c)][0] == best[c][0],
                              values[(meth, c)] is not None and values[(meth, c)][1] == best[c][1]) for c in cols]
            lead = rf'\multirow{{4}}{{*}}{{{shot}-Shot}}' if i == 0 else ''
            lines.append(f'{lead} & {METHODS[meth]} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule' if shot == 1 else r'\bottomrule')
    lines += [r'\end{tabular}}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def tex_gap_table(index, dataset):
    """mIoU of models trained with all classes (Seen) vs trained without the test classes (Novel), same episodes."""
    name = TEX_NAMES[dataset]
    lines = [r'\begin{table}[ht]', r'\centering',
             rf'\caption{{Seen- vs novel-class mIoU on {name} (mean over {NFOLDS[dataset]} folds, identical test episodes). '
             r'Seen: models of Tables 2--3, trained with all facies; Novel: trained without the test facies; '
             r'$\Delta$ = Novel $-$ Seen}',
             rf'\label{{tab:novel_gap_{dataset}}}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lll' + 'ccc' * 2 + '}', r'\toprule',
             r' & & & \multicolumn{3}{c}{1-Shot} & \multicolumn{3}{c}{5-Shot} \\',
             r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
             r'Partition & Support & Model & Seen & Novel & $\Delta$ & Seen & Novel & $\Delta$ \\', r'\midrule']
    for p_i, partition in enumerate(TEX_PARTITIONS):
        for m_i, mode in enumerate(TEX_MODES):
            for i, meth in enumerate(METHODS):
                cells = []
                for shot in (1, 5):
                    seen = fold_mean(index, 'reference', dataset, partition, meth, mode, shot)
                    nov = fold_mean(index, 'novel', dataset, partition, meth, mode, shot)
                    cells += [f'{seen[1]:.2f}' if seen else '--', f'{nov[1]:.2f}' if nov else '--',
                              f'{nov[1] - seen[1]:+.2f}' if seen and nov else '--']
                part = PARTITIONS[partition] if m_i == 0 and i == 0 else ''
                sup = MODES[mode] if i == 0 else ''
                lines.append(f'{part} & {sup} & {METHODS[meth]} & ' + ' & '.join(cells) + r' \\')
            if m_i < len(TEX_MODES) - 1:
                lines.append(r'\cmidrule(lr){2-9}')
        lines.append(r'\midrule' if p_i < len(TEX_PARTITIONS) - 1 else r'\bottomrule')
    lines += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def tex_fold_table(index, dataset):
    """Per-fold mIoU (appendix), one block per partition and shot."""
    name = TEX_NAMES[dataset]
    nf = NFOLDS[dataset]
    names = CLASS_NAMES[dataset]
    folds_desc = '; '.join(f'fold {k}: ' + ', '.join(names[c] for c in novel_classes(dataset, k)) for k in range(nf))
    lines = [r'\begin{table}[ht]', r'\centering',
             rf'\caption{{Per-fold novel-class mIoU on {name}. Held-out facies per fold: {folds_desc}}}',
             rf'\label{{tab:novel_folds_{dataset}}}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lll' + 'c' * (nf + 1) + '}', r'\toprule',
             'Setting & Support & Model & ' + ' & '.join(f'Fold {k}' for k in range(nf)) + r' & Mean \\', r'\midrule']
    blocks = [(p, s) for p in TEX_PARTITIONS for s in (1, 5)]
    for b_i, (partition, shot) in enumerate(blocks):
        for m_i, mode in enumerate(TEX_MODES):
            for i, meth in enumerate(METHODS):
                runs = [index.get(('novel', dataset, partition, k, meth, mode, shot)) for k in range(nf)]
                cells = [f"{r['miou']:.2f}" if r else '--' for r in runs]
                cells.append(f"{mean([r['miou'] for r in runs]):.2f}" if all(runs) else '--')
                setting = f'{PARTITIONS[partition]}, {shot}-shot' if m_i == 0 and i == 0 else ''
                sup = MODES[mode] if i == 0 else ''
                lines.append(f'{setting} & {sup} & {METHODS[meth]} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule' if b_i < len(blocks) - 1 else r'\bottomrule')
    lines += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def write_latex(index, out_dir):
    paths = []
    for dataset in NFOLDS:
        for kind, fn in (('main', tex_main_table), ('gap', tex_gap_table), ('folds', tex_fold_table)):
            path = os.path.join(out_dir, f'table_novel_{kind}_{dataset}.tex')
            with open(path, 'w') as f:
                f.write(fn(index, dataset))
            paths.append(path)
    return paths


if __name__ == '__main__':
    main()
