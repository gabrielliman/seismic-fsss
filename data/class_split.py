r"""Base/novel class folds for the intra-survey novel-class protocol.

The paper's intra-survey regime trains and tests on the same facies (C_test = C_train). This module
implements the classic few-shot protocol inside each survey: the facies of a survey are split into
folds, the model is meta-trained on the base classes only and tested on the held-out (novel) classes
of the same survey (C_test ∩ C_train = ∅), as in PASCAL-5i (Shaban et al., 2017) and COCO-20i, the
benchmarks used by HSNet (Min et al., 2021) and iFSL/ASNet (Kang and Cho, 2022).

Fold rule: the interleaved COCO-20i rule of the iFSL code base, novel(fold) = {fold + nfolds*v + 1}.
  * Parihaka  (6 facies)    -> 3 folds x 2 novel facies (4 base facies for training)
  * Penobscot (8 intervals) -> 4 folds x 2 novel intervals (25% held out, as in PASCAL-5i/COCO-20i)
Interleaving keeps the two novel Penobscot intervals non-adjacent, so every novel interval is bounded
by base intervals and the two novel classes never merge into one background band during training.

Pixels of novel classes that appear inside training patches are treated as episodic background, the
convention of the OSLSM/HSNet/iFSL episodic masks (only the sampled target class is foreground).
"""

NCLASS = {'parihaka': 6, 'penobscot': 8}
NFOLDS = {'parihaka': 3, 'penobscot': 4}

CLASS_NAMES = {
    'parihaka': {1: 'Basement/Other', 2: 'Slope Mudstone A', 3: 'Mass Transport Deposit',
                 4: 'Slope Mudstone B', 5: 'Slope Valley', 6: 'Submarine Canyon System'},
    'penobscot': {1: 'Above H1', 2: 'H1-H2', 3: 'H2-H3', 4: 'H3-H4',
                  5: 'H4-H5', 6: 'H5-H6', 7: 'H6-H7', 8: 'Below H7'},
}


def survey_of(benchmark):
    for survey in NCLASS:
        if survey in benchmark:
            return survey
    raise ValueError(f'Unknown benchmark: {benchmark}')


def novel_classes(survey, fold):
    nfolds = NFOLDS[survey]
    if not 0 <= fold < nfolds:
        raise ValueError(f'{survey} has {nfolds} folds (0..{nfolds - 1}), got fold={fold}')
    return [fold + nfolds * v + 1 for v in range(NCLASS[survey] // nfolds)]


def base_classes(survey, fold):
    novel = novel_classes(survey, fold)
    return [c for c in range(1, NCLASS[survey] + 1) if c not in novel]


def split_classes(benchmark, fold, split, val_classes='base'):
    """Target classes allowed in episodes of `split` ('trn', 'val' or 'test')."""
    survey = survey_of(benchmark)
    if split == 'trn' or (split == 'val' and val_classes == 'base'):
        return base_classes(survey, fold)
    return novel_classes(survey, fold)


def describe(survey, fold):
    names = CLASS_NAMES[survey]
    fmt = lambda cs: ', '.join(f'{c}:{names[c]}' for c in cs)
    return (f'{survey} fold {fold} | base (train) = [{fmt(base_classes(survey, fold))}] '
            f'| novel (test) = [{fmt(novel_classes(survey, fold))}]')


if __name__ == '__main__':
    for survey in NCLASS:
        for fold in range(NFOLDS[survey]):
            print(describe(survey, fold))
