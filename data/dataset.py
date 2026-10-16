from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule

from data.parihaka import DatasetParihaka
from data.parihaka_eds import DatasetParihakaEDS
from data.full_parihaka import DatasetFullParihaka
from data.penobscot import DatasetPenobscot
from data.penobscot_eds import DatasetPenobscotEDS
from data.full_penobscot import DatasetFullPenobscot


DATASETS = {
    'parihaka': DatasetParihaka,              # LRP partition
    'parihaka_eds': DatasetParihakaEDS,       # EDS partition
    'full_parihaka': DatasetFullParihaka,     # whole volume (transfer-learning target)
    'penobscot': DatasetPenobscot,            # LRP partition
    'penobscot_eds': DatasetPenobscotEDS,     # EDS partition
    'full_penobscot': DatasetFullPenobscot,   # whole volume (transfer-learning target)
}


class FSCSDatasetModule(LightningDataModule):
    """
    A LightningDataModule for the seismic few-shot benchmarks
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.datapath = args.datapath
        self.img_size = args.img_size
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def build_dataset(self, split, shot):
        kwargs = {}
        if self.args.benchmark.endswith('_eds'):
            kwargs['num_train'] = self.args.eds_slices

        return DATASETS[self.args.benchmark](
            self.datapath,
            fold=0,
            transform=self.transform,
            split=split,
            way=self.args.way,
            shot=shot,
            shape=(self.img_size, self.img_size),
            stride=(self.img_size // 2, self.img_size // 2),
            support_mode=self.args.support_mode,
            ensure_class_presence=True,
            **kwargs
        )

    def train_dataloader(self):
        # Training episodes are 1-shot; with SSSN a pool of candidates is fetched and filtered to the top-k
        fetch_shot = self.args.support_pool_size if self.args.use_sssn else 1
        dataset = self.build_dataset('trn', fetch_shot)
        return DataLoader(dataset, batch_size=self.args.bsz, shuffle=True, num_workers=8)

    def val_dataloader(self):
        fetch_shot = self.args.support_pool_size if self.args.use_sssn else self.args.shot
        dataset = self.build_dataset('val', fetch_shot)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    def test_dataloader(self):
        fetch_shot = self.args.support_pool_size if self.args.use_sssn else self.args.shot
        dataset = self.build_dataset('test', fetch_shot)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
