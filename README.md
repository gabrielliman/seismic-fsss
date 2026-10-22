# A Broad Assessment of Episodic Construction, Partitioning, and Transferability in Few-Shot Seismic Facies Segmentation


The code evaluates four Few-Shot Semantic Segmentation (FSSS) architectures (PANet, PFENet, HSNet and ASNet) on
the Parihaka and Penobscot 3D seismic volumes, combining:

* **Spatial data partitioning**: Large Rectangular Prisms (**LRP**) and Equally Distant Slices (**EDS**);
* **Support selection**: unconstrained sampling (**Anywhere**), same inline/crossline as the query (**Same Slice**),
  and learned similarity-based selection with a Siamese Support Selection Network (**SSSN**);
* **Inter-dataset transfer**: models trained on one survey and evaluated, without fine-tuning, on the other.

## Acknowledgement of the original code base

This repository is based on the official implementation of
[**Integrative Few-Shot Learning for Classification and Segmentation (iFSL)**](https://github.com/dahyun-kang/ifsl)
by Dahyun Kang and Minsu Cho (CVPR 2022), specifically its `fs-cs` code. The PANet, PFENet, HSNet and ASNet
implementations, the episodic training/evaluation loop and the evaluation utilities come from that repository.
We adapted it for seismic facies segmentation and added what this work needed:

* episodic datasets for Parihaka and Penobscot, with LRP and EDS partitions and whole-volume datasets for transfer learning (`data/`);
* the **Same Slice** and **Anywhere** support selection modes, with guaranteed presence of the target facies in the support patch;
* the **SSSN** support selection (Shao et al., 2024), trained jointly with the few-shot model (`model/siamese.py`, `model/ifsl.py`);
* single-channel (grayscale) input by adapting the first convolution of the ImageNet ResNet backbones;
* loading the official iFSL PASCAL/COCO checkpoints on seismic data by converting their RGB input layer to one channel;
* Macro F1 and per-class metrics, and an update to recent PyTorch / PyTorch Lightning versions.

The original iFSL code is distributed under the MIT license (see [LICENSE](LICENSE)).
Please also cite iFSL if you use this code (BibTeX below).

## Installation

Tested on Ubuntu 24.04, Python 3.10, PyTorch 2.10 (CUDA 12.8) and an NVIDIA RTX 4090 (24 GB).

```bash
conda create -n seismic-fsss python=3.10
conda activate seismic-fsss
pip install -r requirements.txt
```

ImageNet weights for the ResNet backbones are downloaded automatically by torchvision on first use.

## Data

Put both volumes in the same directory and pass it with `--datapath`:

```
DATAPATH/
├── data_train.npz      # Parihaka seismic amplitudes, key "data",   shape (1006, 782, 590)
├── labels_train.npz    # Parihaka facies labels,     key "labels", values 1..6
└── penobscot.h5        # Penobscot, datasets "features" (601, 1501, 481, 1) and "label" (601, 1501, 481), values 0..7
```

* **Parihaka**: training files of the [SEG/AIcrowd Seismic Facies Identification Challenge](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge)
  (labeled region of the volume released by Chevron U.S.A. Inc. / New Zealand Petroleum and Minerals).
* **Penobscot**: interpreted Penobscot dataset of Baroni et al. (2019), [doi:10.5281/zenodo.1341774](https://doi.org/10.5281/zenodo.1341774), stored as an HDF5 file.

Penobscot labels are shifted to 1..8 when loaded, so that all eight facies can be target classes (0 is reserved for the episodic background).
Amplitudes are min-max scaled to [0, 255] and patches of 400×400 are extracted with a 200×200 stride.

## Usage

All experiments go through `main.py`. The main arguments are:

| Argument | Values | Description |
|---|---|---|
| `--benchmark` | `parihaka`, `penobscot` | **LRP** partition of the volume |
| | `parihaka_eds`, `penobscot_eds` | **EDS** partition (number of slices set by `--eds_slices`, default 5) |
| | `full_parihaka`, `full_penobscot` | whole volume, used as target for **transfer learning** (evaluation only) |
| `--method` | `panet`, `pfenet`, `hsnet`, `asnet` | FSSS architecture |
| `--support_mode` | `anywhere`, `same_slice`, `sssn` | support selection strategy |
| `--shot` | `1`, `5` | number of support patches at validation/test time (training is always 1-shot) |
| `--logpath` | directory | where checkpoints are written (training) or read from (`--eval`) |
| `--niter` | default `200` | training epochs (5,000 episodes each); `0` saves an untrained model |
| `--backbone` | `resnet101` (default), `resnet50` | ImageNet backbone (kept frozen) |

Other defaults follow the paper: 1-way episodes, batch size 12, Adam with learning rate 1e-3 (PANet and PFENet keep
the SGD optimizers of the original implementations), 1,000 fixed validation/test episodes, and an SSSN candidate pool
of 9 patches (`--support_pool_size`). The best checkpoint according to validation mIoU is kept as `best_model.ckpt`.
By default the Trainer uses every visible GPU (DDP); restrict it with `CUDA_VISIBLE_DEVICES`.

### Training

```bash
# ASNet, Parihaka, LRP partition, Same Slice support
python main.py --datapath $DATAPATH --benchmark parihaka --method asnet \
               --support_mode same_slice --logpath logs/parihaka_lrp/asnet_same_slice

# PFENet, Penobscot, EDS with 5 slices, SSSN support
python main.py --datapath $DATAPATH --benchmark penobscot_eds --eds_slices 5 --method pfenet \
               --support_mode sssn --logpath logs/penobscot_eds5/pfenet_sssn
```

Use `--eds_slices 1` for the "EDS (1 Slice)" experiments and `--resume` to continue an interrupted run.

### Evaluation (Tables 2 and 3)

Evaluate the best checkpoint of a run on the test partition of the same dataset, with the same partition and
support mode used for training:

```bash
python main.py --datapath $DATAPATH --benchmark parihaka --method asnet \
               --support_mode same_slice --logpath logs/parihaka_lrp/asnet_same_slice --shot 5 --eval
```

The output reports per-class IoU, accuracy and F1, and the episode-level metrics `test/miou` (mean IoU over the
facies classes) and `test/seg_macro_f1` (Macro F1), which are the values reported in the paper.

### Transfer learning (Tables 4 and 5)

Evaluate a model trained on one survey on the whole volume of the other one, keeping the support mode:

```bash
# trained on Parihaka (LRP), tested on Penobscot
python main.py --datapath $DATAPATH --benchmark full_penobscot --method asnet \
               --support_mode same_slice --logpath logs/parihaka_lrp/asnet_same_slice --shot 5 --eval
```

**Baseline – Random** (ImageNet backbone with randomly initialised few-shot modules, no seismic training):

```bash
python main.py --datapath $DATAPATH --benchmark full_parihaka --method asnet \
               --support_mode anywhere --logpath logs/random/asnet_anywhere --niter 0
python main.py --datapath $DATAPATH --benchmark full_parihaka --method asnet \
               --support_mode anywhere --logpath logs/random/asnet_anywhere --eval
```

### Natural-image pre-trained baselines (Table 6)

These baselines use the checkpoints released by the iFSL authors, trained on PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup>
(fold 0, 1-way 1-shot). Download them from the [iFSL repository](https://github.com/dahyun-kang/ifsl/tree/main/fs-cs)
and point `--logpath` to the folder containing `best_model.ckpt`. The RGB input layer is converted to a single
channel on the first evaluation (a `best_model_1channel.ckpt` file is written next to the original):

```bash
python main.py --datapath $DATAPATH --benchmark full_parihaka --method hsnet --backbone resnet101 \
               --support_mode same_slice --logpath logs/pascal/fold0/resnet101/hsnet --eval
```

**Note on PFENet:** the original iFSL implementation always builds PFENet on a ResNet-50, so the released
`pascal/fold0/resnet101/pfenet` checkpoint actually contains a ResNet-50 backbone. Evaluate it with
`--backbone resnet50` (checkpoints are loaded strictly, so a backbone mismatch raises an error).

If a checkpoint fails to load with a `weights_only` error, add `--no_safe_load` (only for checkpoints you trust).

### Full experiment grid

`scripts/run_experiments.sh` trains and evaluates every configuration of Tables 2–5
(2 datasets × 3 partitions × 3 support modes × 4 models, 1- and 5-shot, plus transfer and random baselines):

```bash
DATAPATH=/path/to/seismic LOGDIR=logs bash scripts/run_experiments.sh
# a subset can be selected, e.g. METHODS="asnet hsnet" MODES="same_slice"
```

Evaluation logs are written as `eval_<test|transfer_*|full_*>_<k>shot.txt` inside each run directory.

## Repository structure

```
main.py                  # entry point: training, evaluation, untrained baselines
data/
  dataset.py             # LightningDataModule and benchmark registry
  parihaka.py            # Parihaka, LRP partition
  parihaka_eds.py        # Parihaka, EDS partition
  full_parihaka.py       # Parihaka, whole volume (transfer target)
  penobscot.py           # Penobscot, LRP partition
  penobscot_eds.py       # Penobscot, EDS partition
  full_penobscot.py      # Penobscot, whole volume (transfer target)
model/
  ifsl.py                # shared episodic training / evaluation logic and SSSN filtering
  panet.py pfenet.py hsnet.py asnet.py
  siamese.py             # Siamese Support Selection Network
  base/ module/          # building blocks of the FSSS models (from iFSL)
  utils/backbone_patch.py  # 3-channel -> 1-channel backbone adaptation
common/                  # callbacks, metrics, utilities
scripts/run_experiments.sh
```

## Citation

If you use this code, please cite our paper (reference to be updated upon publication) and iFSL:

```BibTeX
@article{nunes2026seismicfsss,
  author  = {Nunes, Gabriel Lima and Cunha, Caio Teles and de Souza, Matheus and Amarante, Gabriel and Barreiros Jr., Willian and Ferreira, Renato and Teodoro, George},
  title   = {A Broad Assessment of Episodic Construction, Partitioning, and Transferability in Few-Shot Seismic Facies Segmentation},
  year    = {2026}
}

@inproceedings{kang2022ifsl,
  author    = {Kang, Dahyun and Cho, Minsu},
  title     = {Integrative Few-Shot Learning for Classification and Segmentation},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
