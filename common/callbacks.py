import os
from pprint import PrettyPrinter

from pytorch_lightning.callbacks import Callback, TQDMProgressBar, ModelCheckpoint

from common.evaluation import AverageMeter
from common import utils

class CustomProgressBar(TQDMProgressBar):
    """
    Simplified progress bar to avoid attribute errors in Lightning 2.x.
    """
    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True):
        super().__init__()
        self.global_progress = global_progress
        self.leave_global_progress = leave_global_progress

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        # Use the standard PL 2.x attribute name
        if self.train_progress_bar is not None:
            self.train_progress_bar.set_description(f"[trn] ep: {trainer.current_epoch:>3}")

class MeterCallback(Callback):
    """
    A class that initiates classification and segmentation metrics.
    Updated for Lightning 2.x DataLoader handling.
    """
    def __init__(self, args):
        super(MeterCallback, self).__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        if hasattr(pl_module, 'learner'):
            print(pl_module.learner)
        utils.print_param_count(pl_module)

    def on_test_start(self, trainer, pl_module):
        PrettyPrinter().pprint(vars(self.args))
        utils.print_param_count(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'\n\n----- ep: {trainer.current_epoch:>3}-----')
        utils.fix_randseed(None)
        
        train_loader = trainer.train_dataloader
        dataset = train_loader.dataset
        if hasattr(dataset, 'datasets'):
            dataset = dataset.datasets

        pl_module.average_meter = AverageMeter(dataset, self.args.way)
        pl_module.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        # FIX: Handle single vs list of dataloaders
        val_loaders = trainer.val_dataloaders
        dataloader = val_loaders[0] if isinstance(val_loaders, list) else val_loaders
        self._shared_eval_epoch_start(dataloader.dataset, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        # FIX: Handle single vs list of dataloaders
        test_loaders = trainer.test_dataloaders
        dataloader = test_loaders[0] if isinstance(test_loaders, list) else test_loaders
        self._shared_eval_epoch_start(dataloader.dataset, pl_module)

    def _shared_eval_epoch_start(self, dataset, pl_module):
        utils.fix_randseed(0)
        pl_module.average_meter = AverageMeter(dataset, self.args.way)
        pl_module.eval()


class CustomCheckpoint(ModelCheckpoint):
    """
    Checkpoint load & save logic.
    """
    def __init__(self, args):
        self.dirpath = args.logpath
        if not args.eval and not args.resume:
            if os.path.exists(self.dirpath):
                print(f"Warning: {self.dirpath} already exists.")

        self.filename = 'best_model'
        self.monitor = 'val/miou'

        super().__init__(dirpath=self.dirpath,
                         monitor=self.monitor,
                         filename=self.filename,
                         mode='max',
                         verbose=True,
                         save_last=True)

        if args.eval:
            self.modelpath = self.return_best_model_path(self.dirpath, self.filename)
            print('evaluating', self.modelpath)
        else:
            self.modelpath = os.path.join(self.dirpath, self.filename + '.ckpt')

        self.lastmodelpath = os.path.join(self.dirpath, 'last.ckpt') if args.resume else None

    def return_best_model_path(self, dirpath, filename):
        if not os.path.exists(dirpath):
             return None
        ckpt_files = [f for f in os.listdir(dirpath) if f.endswith('.ckpt')]
        vers = [ckpt_file for ckpt_file in ckpt_files if filename in ckpt_file]
        if not vers: return None
        vers.sort()
        # Return the latest version if multiple exist (e.g., best_model-v1.ckpt)
        best_model = vers[-1] if len(vers) == 1 else vers[-2]
        return os.path.join(dirpath, best_model)

