# Intra-survey novel-class protocol (classic few-shot inside each survey)

## Why

In the paper, the intra-survey regime (Tables 2–3) trains and tests on the **same facies** (C_test = C_train);
only space is held out. Unseen classes appear only in the inter-survey regime (Tables 4–5), where class novelty
is confounded with domain shift. The standard FSSS premise, segmenting classes never seen in meta-training *within the
same domain*, is therefore never tested on its own. This protocol adds that test.

## Design, and the reference behind each choice

All references are from the paper's bibliography.

| Decision | Choice | Basis |
|---|---|---|
| Class folds | Parihaka 6 facies → **3 folds × 2 novel**; Penobscot 8 intervals → **4 folds × 2 novel** | PASCAL-5i (Shaban et al. 2017): 20 classes → 4 folds × 5 (25% held out). Penobscot matches this ratio exactly; Parihaka needs 33% to keep ≥ 2 novel classes per fold and every facies novel exactly once |
| Assignment of classes to folds | Interleaved: novel(k) = {k + nfolds·v + 1} | COCO-20i rule of the iFSL code base (Kang and Cho 2022), on which this repo is built. Keeps the two novel Penobscot intervals non-adjacent, so each novel interval is bounded by base intervals |
| Episodes | 1-way, 1-shot training; 1- and 5-shot testing | Paper protocol; Shaban et al. 2017; Min et al. 2021 |
| Novel pixels inside training patches | Treated as episodic background | Episodic masks of OSLSM/HSNet/iFSL (Shaban et al. 2017; Min et al. 2021; Kang and Cho 2022) |
| Spatial split | Unchanged (LRP / EDS train–val–test regions) | Lima et al. 2025; paper Sec. 3.2. Test classes are novel **and** the test region is spatially disjoint |
| Validation / checkpoint selection | Base classes in the validation region (`--val_classes base`) | Stricter than PASCAL-5i, whose validation images are the test images: no novel-class information is used before testing. `--val_classes novel` reproduces the PASCAL/iFSL practice if needed |
| Test | 1,000 fixed episodes per fold (seed = episode index); report each fold and the **mean over folds** | HSNet (Min et al. 2021) and iFSL (Kang and Cho 2022) tables |
| Metrics | mIoU over the novel classes of the fold, Macro F1 | Paper Sec. 4.1 (computed only over the classes targeted in the test episodes) |
| Seen-class reference | Paper checkpoints (all classes seen), evaluated on **the same episodes** | Isolates the cost of class novelty (same region, classes, seeds, supports) |

Folds (`python data/class_split.py`):

| Survey | Fold | Novel (test) | Base (train) |
|---|---|---|---|
| Parihaka | 0 | Basement/Other, Slope Mudstone B | SM A, MTD, SV, SCS |
| Parihaka | 1 | Slope Mudstone A, Slope Valley | B/O, MTD, SM B, SCS |
| Parihaka | 2 | Mass Transport Deposit, Submarine Canyon System | B/O, SM A, SM B, SV |
| Penobscot | 0 | Above H1, H4–H5 | the other 6 |
| Penobscot | 1 | H1–H2, H5–H6 | the other 6 |
| Penobscot | 2 | H2–H3, H6–H7 | the other 6 |
| Penobscot | 3 | H3–H4, Below H7 | the other 6 |

Parihaka fold 2 holds out MTD and SCS together, the texturally similar pair identified as the main confusion
(paper Sec. 3.1). Expect it to be the hardest fold, and report it separately rather than only in the mean.

## Tests to run (in this order)

| Tier | What | Tasks | Question answered |
|---|---|---|---|
| **reference** | Paper checkpoints, eval only, on the novel classes of every fold (LRP + EDS5 × A/SS/SSSN × 4 models × 7 folds) | 168 evals | Seen-class score on exactly the test episodes of the novel runs |
| **tier1** (essential) | Train on base, test on novel: **EDS (5 slices)** × {Anywhere, Same Slice} × 4 models × 7 folds | 56 trainings | Main reviewer answer: do the models generalize to unseen facies, and does Same Slice still help? |
| **tier2** | Same on **LRP** | 56 trainings | Does EDS > LRP hold for novel classes? |
| **tier3** | **SSSN** on EDS5 and LRP | 56 trainings | Same Slice vs learned selection for novel classes |

EDS (1 slice) is left out: in the paper it is the degenerate case (best Parihaka mIoU 45.9), and removing classes
would leave too few training episodes for a meaningful comparison.

Cost: one PANet epoch took about 3 min on an RTX 4090 (1 GPU), so about 11 h per 200-epoch run. HSNet and ASNet are slower.
Each task asks for 48 h and resumes from `last.ckpt` if resubmitted after a timeout.

## How to run

```bash
# on the cluster, in a clone of this branch
git checkout novel-class-folds
# check paths in scripts/novel_class/config.sh (DATAPATH, LOGROOT, PAPER_LOGDIR, CONDA_ENV) or override by env vars
DRY_RUN=1 bash scripts/novel_class/submit.sh tier1         # list tasks without submitting
bash scripts/novel_class/submit.sh reference               # cheap, run first
MAX_PARALLEL=8 bash scripts/novel_class/submit.sh tier1
bash scripts/novel_class/submit.sh tier2
bash scripts/novel_class/submit.sh tier3
# resubmit only some tasks (e.g. after a timeout): bash scripts/novel_class/submit.sh tier1 3,17
python scripts/novel_class/collect_results.py --logroot $LOGROOT --out results_novel
```

Outputs: `$LOGROOT/<dataset>_<partition>/fold<k>/<method>_<mode>/{best_model.ckpt,train.log,eval_novel_{1,5}shot.txt}`;
the reference evals go to `$LOGROOT/reference/...`. `collect_results.py` writes `novel_class_tables.md` (fold columns, Mean,
Ref, Gap = Ref − Mean) and a long-format CSV.

Single run by hand:

```bash
python main.py --datapath $DATAPATH --benchmark parihaka_eds --eds_slices 5 --method asnet \
       --support_mode same_slice --fold 0 --logpath logs/novel/parihaka_eds5/fold0/asnet_same_slice
python main.py ... --fold 0 --shot 5 --eval
```

## Caveats to check before submitting `reference`

* `PAPER_LAYOUT=legacy` maps to the original `/snfs2/nunes/logs_fs_cs*` directories. Penobscot EDS5 SSSN exists in both
  `logs_fs_cs_eds_sssn` and `logs_fs_cs_eds5_sssn`; the script uses the former. Confirm which run produced Table 3.
* Legacy PFENet checkpoints may contain a ResNet-50. Loading is strict, so a mismatch fails loudly; resubmit those tasks with
  `REF_PFENET_BACKBONE=resnet50`.
* The reference is only valid for Penobscot if those checkpoints were trained with labels 1..8 (the corrected labeling).

## Suggested paper text (Sec. 3.3)

> *Intra-survey novel-class regime.* Following the fold protocol of PASCAL-5i (Shaban et al. 2017), as adopted by
> HSNet and iFSL (Min et al. 2021; Kang and Cho 2022), the facies of each survey are split into folds (3 for Parihaka,
> 4 for Penobscot, two novel classes each). Models are meta-trained on the base classes of the training partition and
> evaluated on the novel classes in the spatially disjoint test partition, so that C_test ∩ C_train = ∅ without domain
> shift. We report the mean over folds and compare with models trained on all classes, evaluated on the same episodes.
