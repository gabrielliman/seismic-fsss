import os
import argparse

import torch
from pytorch_lightning import Trainer
from lightning_fabric.utilities.data import AttributeDict

from model.asnet import AttentiveSqueezeNetwork
from model.panet import PrototypeAlignmentNetwork
from model.pfenet import PriorGuidedFeatureEnrichmentNetwork
from model.hsnet import HypercorrSqueezeNetwork
from data.dataset import FSCSDatasetModule, DATASETS
from common.callbacks import MeterCallback, CustomProgressBar, CustomCheckpoint

# Register custom classes as safe for unpickling (PyTorch 2.6+ weights_only loading)
torch.serialization.add_safe_globals([AttributeDict])


def patch_checkpoint_if_needed(ckpt_path, safe_load=True):
    """
    Checks if the checkpoint has a 3-channel (RGB) input layer, e.g. the official iFSL
    PASCAL/COCO checkpoints, and patches it to 1 channel for seismic data.
    The patched checkpoint is saved next to the original so it is converted only once.
    """
    base, ext = os.path.splitext(ckpt_path)
    patched_path = f"{base}_1channel{ext}"

    # If we already patched it previously, just use the patched version
    if os.path.exists(patched_path):
        print(f"--- Found existing 1-channel checkpoint: {patched_path} ---")
        return patched_path

    print(f"--- Checking checkpoint channels at: {ckpt_path} ---")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=safe_load)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Input layer names across the different architectures
    possible_keys = ["backbone.conv1.weight", "backbone0.0.weight"]
    patched_any = False

    for weight_key in possible_keys:
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            # [out_channels, in_channels, H, W] with in_channels == 3
            if len(weight.shape) == 4 and weight.shape[1] == 3:
                print(f"--- Converting {weight_key} from 3 channels to 1 channel ---")
                # Collapse the 3 channels into 1 by summing across the channel dimension
                state_dict[weight_key] = weight.sum(dim=1, keepdim=True)
                patched_any = True

    if patched_any:
        torch.save(checkpoint, patched_path)
        print(f"--- Saved 1-channel checkpoint to: {patched_path} ---")
        return patched_path

    return ckpt_path


def main(args):

    # Method mapping
    modeldict = dict(panet=PrototypeAlignmentNetwork,
                     pfenet=PriorGuidedFeatureEnrichmentNetwork,
                     hsnet=HypercorrSqueezeNetwork,
                     asnet=AttentiveSqueezeNetwork)
    modelclass = modeldict[args.method]

    # Dataset initialization
    dm = FSCSDatasetModule(args)

    # Initialize callback to handle paths
    checkpoint_callback = CustomCheckpoint(args)

    # Trainer initialization
    trainer = Trainer(
        accelerator='gpu',
        devices="auto",
        strategy='ddp_find_unused_parameters_true',
        callbacks=[MeterCallback(args), checkpoint_callback, CustomProgressBar()],
        logger=False,
        max_epochs=args.niter,
        num_sanity_val_steps=0,
    )

    if args.eval:
        # Loading the best model checkpoint
        if checkpoint_callback.modelpath is None:
            raise FileNotFoundError(f"No best_model*.ckpt found in '{args.logpath}'")
        safe_load_flag = not args.no_safe_load
        modelpath = patch_checkpoint_if_needed(checkpoint_callback.modelpath, safe_load=safe_load_flag)

        print(f"--- Evaluating model from: {modelpath} ---")
        model = modelclass.load_from_checkpoint(
            modelpath,
            args=args,
            strict=True,
            weights_only=safe_load_flag
        )
        trainer.test(model, dataloaders=dm.test_dataloader())

    else:
        # Initialize model
        model = modelclass(args)

        if args.niter == 0:
            # "Baseline - Random": save the untrained model (ImageNet backbone + randomly initialised
            # few-shot modules) so that it can be evaluated with --eval
            os.makedirs(checkpoint_callback.dirpath, exist_ok=True)
            save_path = os.path.join(checkpoint_callback.dirpath, "best_model.ckpt")

            # Attach model to trainer strategy to allow saving without .fit()
            model.to(trainer.strategy.root_device)
            trainer.strategy.connect(model)

            print(f"--- Saving BASE WEIGHTS to: {save_path} ---")
            trainer.save_checkpoint(save_path)
            return

        trainer.fit(model, dm, ckpt_path=checkpoint_callback.lastmodelpath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Few-shot seismic facies segmentation (LRP/EDS partitions, Anywhere/Same Slice/SSSN support)')
    parser.add_argument('--datapath', type=str, required=True, help='Directory with data_train.npz, labels_train.npz (Parihaka) and penobscot.h5')
    parser.add_argument('--benchmark', type=str, required=True, choices=list(DATASETS), help='Dataset and spatial partition')
    parser.add_argument('--method', type=str, default='asnet', choices=['panet', 'pfenet', 'hsnet', 'asnet'], help='FSSS architecture')
    parser.add_argument('--logpath', type=str, required=True, help='Directory where checkpoints are saved (training) or loaded from (--eval)')
    parser.add_argument('--support_mode', type=str, default='same_slice', choices=['anywhere', 'same_slice', 'sssn'],
                        help='Support selection strategy: Anywhere, Same Slice or Siamese Support Selection Network')
    parser.add_argument('--eds_slices', type=int, default=5, help='Number of equally distant slices per orientation (EDS partitions only)')
    parser.add_argument('--way', type=int, default=1, help='N-way')
    parser.add_argument('--shot', type=int, default=1, help='K-shot (used at validation/test time; training is 1-shot)')
    parser.add_argument('--bsz', type=int, default=12, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--niter', type=int, default=200, help='Number of training epochs (0 saves an untrained model)')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone network')
    parser.add_argument('--img_size', type=int, default=400, help='Patch size')
    parser.add_argument('--support_pool_size', type=int, default=9, help='Number of candidate support patches scored by SSSN')
    parser.add_argument('--eval', action='store_true', help='Evaluate the best checkpoint in --logpath')
    parser.add_argument('--resume', action='store_true', help='Resume training from last.ckpt in --logpath')
    parser.add_argument('--no_safe_load', action='store_true', help='Load checkpoints with torch.load(weights_only=False)')

    args = parser.parse_args()

    # SSSN re-ranks a candidate pool sampled with the Anywhere strategy
    args.use_sssn = args.support_mode == 'sssn'
    if args.use_sssn:
        args.support_mode = 'anywhere'

    main(args)
