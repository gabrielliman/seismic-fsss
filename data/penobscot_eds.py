import os
import numpy as np
import torch
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows
from PIL import Image

# Ajuste conforme seu projeto:
# from your_module import read_h5_file
def read_h5_file(file_path):
    f = h5py.File(file_path,'r')
    images=f['features']
    labels=f['label']
    # The 8 facies are stored as 0..7; shift them to 1..8 (0 is the episodic background)
    return np.squeeze(np.array(images)), np.array(labels, dtype=np.uint8) + 1


# ============================================================
# Helpers
# ============================================================

def scale_to_256(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)


def extract_patches(img, shape, stride):
    """Extrai patches 2D mesmo se a imagem for menor que o patch."""
    z, x = img.shape
    sh, sw = shape
    sz, sx = stride
    if z < sh or x < sw:
        return []
    patches = view_as_windows(img, (sh, sw), step=(sz, sx))
    patches = patches.reshape(-1, sh, sw)
    return [p for p in patches]


def get_limits(start, end, num_limits):
    """Equally Distant Slices (EDS) limits."""
    step_size = (end - start) / (num_limits + 1)
    return [int(start + i * step_size) for i in range(1, num_limits + 1)]


# ============================================================
# DatasetPenobscotEDS (FS-CS compatible)
# ============================================================
class DatasetPenobscotEDS(Dataset):
    """
    Few-shot dataset EDS (Equally Distant Slices) para Penobscot.
    """

    def __init__(
        self, datapath, fold, transform, split, way, shot,
        shape=(400, 400), stride=(128, 64), num_train=1,support_mode="same_slice",ensure_class_presence=True
    ):
        self.split = split
        self.fold = fold
        self.way = way
        self.shot = shot
        self.shape = shape
        self.stride = stride
        self.transform = transform
        self.num_train = num_train
        self.support_mode = support_mode
        self.nclass = 8  # Penobscot possui 8 classes, não existe fundo
        self.benchmark = "penobscot"
        self.class_ids = list(range(1, self.nclass + 1))
        self.ensure_class_presence = ensure_class_presence
    
        # === Carrega Penobscot (H5)
        seis_data, labels = read_h5_file(os.path.join(datapath, "penobscot.h5"))

        seis_data = scale_to_256(seis_data).astype(np.uint8)

        # rearranja para (z, y, x)
        seis_data = seis_data.transpose((1, 0, 2))
        labels = labels.transpose((1, 0, 2))

        # === Constrói regiões EDS seguindo exatamente o modelo Parihaka EDS
        self.regions = self.build_regions(seis_data, labels, num_train)

        self.class_to_patch_refs = None
        if self.ensure_class_presence:
            self.class_to_patch_refs = self.build_global_patch_index()
        #maybe this breaks test and train
        valid_classes = [
                c for c in self.class_ids 
                if len(self.class_to_patch_refs.get(c, [])) > 0
            ]
            
        if len(valid_classes) < len(self.class_ids):
            missing = set(self.class_ids) - set(valid_classes)
            print(f"Removing empty classes from sampling pool: {missing}")
        #end of change
        self.class_ids = valid_classes
        # === Metadados simulados (mesma estrutura)
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.sampling_stats = {
            "hit_slice": 0,
            "fallback_global": 0,
            "fatal_random": 0,
            "blind_random": 0
            
        }
    def build_global_patch_index(self):
        """
        Constrói índice:
            class_to_refs[c] = list of (split_name, region_idx, patch_idx)
        """

        class_to_refs = {c: [] for c in self.class_ids}

        for split_name in ["train", "val", "test"]:
            for region_idx, (data2d, mask2d) in enumerate(self.regions[split_name]):

                patches = extract_patches(data2d, self.shape, self.stride)
                masks   = extract_patches(mask2d, self.shape, self.stride)

                for p_idx, m in enumerate(masks):
                    for c in np.unique(m):
                        if 1 <= c <= self.nclass:
                            class_to_refs[c].append(
                                (split_name, region_idx, p_idx)
                            )

        return class_to_refs
    def get_patch_from_ref(self, ref):
        split_name, region_idx, patch_idx = ref

        data2d, mask2d = self.regions[split_name][region_idx]

        patches = extract_patches(data2d, self.shape, self.stride)
        masks   = extract_patches(mask2d, self.shape, self.stride)

        if patch_idx >= len(patches):
            raise IndexError("Patch index out of range")

        return patches[patch_idx], masks[patch_idx]
    # ----------------------------------------------------------------------
    def build_regions(self, seis_data, labels, num_train):

        x_limits = get_limits(0, seis_data.shape[1], num_train)
        y_limits = get_limits(0, seis_data.shape[2], num_train)

        train_regions, val_regions = [], []

        # -- Treino e Val seguem modelo do Parihaka EDS --
        for pos_x, pos_y in zip(x_limits, y_limits):

            # === Treino: crossline slice (z,x)
            if pos_x < seis_data.shape[1] - 1:
                train_regions.append((
                    seis_data[:, pos_x, :],        # (z, x)
                    labels[:, pos_x, :]
                ))

            # === Treino: inline slice (z,y)
            if pos_y < seis_data.shape[2] - 1:
                train_regions.append((
                    seis_data[:, :, pos_y],        # (z, x)
                    labels[:, :, pos_y]
                ))

            # === Validação: próxima slice
            if pos_x + 1 < seis_data.shape[1]:
                val_regions.append((
                    seis_data[:, pos_x + 1, :],
                    labels[:, pos_x + 1, :]
                ))

            if pos_y + 1 < seis_data.shape[2]:
                val_regions.append((
                    seis_data[:, :, pos_y + 1],
                    labels[:, :, pos_y + 1]
                ))

        # === Teste: slices que não entraram em train/val
        test_regions = []
        used_x = set(x_limits + [x + 1 for x in x_limits if x + 1 < seis_data.shape[1]])
        used_y = set(y_limits + [y + 1 for y in y_limits if y + 1 < seis_data.shape[2]])

        for x in range(seis_data.shape[1]):
            if x not in used_x:
                test_regions.append((seis_data[:, x, :], labels[:, x, :]))

        for y in range(seis_data.shape[2]):
            if y not in used_y:
                test_regions.append((seis_data[:, :, y], labels[:, :, y]))

        return {"train": train_regions, "val": val_regions, "test": test_regions}

    # ----------------------------------------------------------------------
    def __len__(self):
        return 5000 if self.split == "trn" else 1000

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        query_name, support_names, support_classes = self.sample_episode(idx)

        support_mode = self.support_mode
        query_img, query_mask, support_imgs, support_masks, org_size = \
            self.load_frame(query_name, support_names,support_classes, support_mode=support_mode)

        query_class_presence = [
            c in torch.unique(query_mask) for c in support_classes
        ]

        rename_class = lambda x: support_classes.index(x) + 1 if x in support_classes else 0

        # transforma query
        query_img = self.transform(query_img)
        query_mask, query_ignore = self.get_query_mask(query_img, query_mask, rename_class)

        # transforma suporte
        support_imgs = torch.stack([
            torch.stack([self.transform(im) for im in supps])
            for supps in support_imgs
        ])

        support_masks, support_ignores = self.get_support_masks(
            support_imgs, support_classes, support_masks, rename_class
        )

        return {
            "query_img": query_img,
            "query_mask": query_mask,
            "query_name": query_name,
            "query_ignore_idx": query_ignore,
            "org_query_imsize": org_size,
            "support_imgs": support_imgs,
            "support_masks": support_masks,
            "support_names": support_names,
            "support_ignore_idxs": support_ignores,
            "support_classes": torch.tensor(support_classes),
            "query_class_presence": torch.tensor(query_class_presence),
        }

    # ----------------------------------------------------------------------
    def build_img_metadata(self):
        n_items = 5000 if self.split == "trn" else 1000
        return [(f"patch_{i}", int(i % self.nclass) + 1) for i in range(n_items)]

    def build_img_metadata_classwise(self):
        d = {i: [] for i in range(1, self.nclass + 1)}
        for name, cid in self.img_metadata:
            d[cid].append(name)
        return d

    # ----------------------------------------------------------------------
    def sample_episode(self, idx):
        if self.split in ["val", "test"]:
            np.random.seed(idx)

        query_class = np.random.choice(self.class_ids)
        query_name = f"patch_{idx}_{query_class}"

        support_classes = np.random.choice(self.class_ids, self.way, replace=False).tolist()
        support_names = [
            [f"patch_{idx}_{query_class}_{s}_{i}" for i in range(self.shot)]
            for s in support_classes
        ]
        return query_name, support_names, support_classes

    # ----------------------------------------------------------------------
    def load_frame(self, query_name, support_names, support_classes, support_mode=None):

        shape  = self.shape
        stride = self.stride

        support_mode = support_mode

        # --------------------------------------------------
        # 1) Select region pool
        # --------------------------------------------------
        region_pool = (
            self.regions["train"] if self.split == "trn"
            else self.regions["val"] if self.split == "val"
            else self.regions["test"]
        )

        # --------------------------------------------------
        # 2) Select one slice
        # --------------------------------------------------
        region_idx = np.random.randint(len(region_pool))
        data2d, mask2d = region_pool[region_idx]

        patches = extract_patches(data2d, shape, stride)
        masks   = extract_patches(mask2d, shape, stride)

        if len(patches) == 0:
            raise RuntimeError(
                f"No valid patches for split={self.split}. Slice shape: {data2d.shape}"
            )

        # --------------------------------------------------
        # 3) QUERY
        # --------------------------------------------------
        qi = np.random.randint(len(patches))
        query_img  = Image.fromarray(np.uint8(patches[qi]))
        query_mask = torch.tensor(masks[qi], dtype=torch.int64)

        # --------------------------------------------------
        # 4) SUPPORT
        # --------------------------------------------------
        support_imgs  = []
        support_masks = []

        has_index = self.ensure_class_presence and self.class_to_patch_refs is not None

        for cid in support_classes:

            imgs_c  = []
            masks_c = []

            for _ in range(self.shot):

                # ==========================================
                # SAME SLICE
                # ==========================================
                if support_mode == "same_slice":

                    if has_index:
                        # try current slice first
                        valid = [
                            (p, m)
                            for p, m in zip(patches, masks)
                            if cid in np.unique(m)
                        ]

                        if len(valid) > 0:
                            self.sampling_stats["hit_slice"] += 1
                            p, m = valid[np.random.randint(len(valid))]
                        else:
                            refs = self.class_to_patch_refs.get(cid, [])
                            if len(refs) > 0:
                                self.sampling_stats["fallback_global"] += 1
                                ref = refs[np.random.randint(len(refs))]
                                p, m = self.get_patch_from_ref(ref)
                            else:
                                self.sampling_stats["fatal_random"] += 1
                                j = np.random.randint(len(patches))
                                p, m = patches[j], masks[j]
                    else:
                        self.sampling_stats["blind_random"] += 1
                        j = np.random.randint(len(patches))
                        p, m = patches[j], masks[j]

                # ==========================================
                # ANYWHERE (same split)
                # ==========================================
                elif support_mode == "anywhere":

                    if has_index:
                        refs = [
                            r for r in self.class_to_patch_refs[cid]
                            if r[0] == ("train" if self.split == "trn"
                                        else "val" if self.split == "val"
                                        else "test")
                        ]

                        if len(refs) > 0:
                            ref = refs[np.random.randint(len(refs))]
                            p, m = self.get_patch_from_ref(ref)
                        else:
                            j = np.random.randint(len(patches))
                            p, m = patches[j], masks[j]
                    else:
                        data_s, mask_s = region_pool[
                            np.random.randint(len(region_pool))
                        ]
                        p2 = extract_patches(data_s, shape, stride)
                        m2 = extract_patches(mask_s, shape, stride)
                        j = np.random.randint(len(p2))
                        p, m = p2[j], m2[j]

                else:
                    raise ValueError(f"Invalid support_mode={support_mode}")

                imgs_c.append(Image.fromarray(np.uint8(p)))
                masks_c.append(m)

            support_imgs.append(imgs_c)
            support_masks.append(masks_c)

        return query_img, query_mask, support_imgs, support_masks, query_img.size

    # ----------------------------------------------------------------------
    def get_query_mask(self, query_img, query_cmask, rename_class):
        query_cmask = F.interpolate(
            query_cmask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:], mode="nearest"
        ).squeeze()
        return self.generate_query_episodic_mask(query_cmask, rename_class)

    def get_support_masks(self, support_imgs, support_classes, support_cmasks, rename_class):
        out_masks, out_ignores = [], []
        for cid, masks_c in zip(support_classes, support_cmasks):
            masks_list, ignore_list = [], []
            for cm in masks_c:
                cm = torch.tensor(cm, dtype=torch.float32)
                smask, sidx = self.generate_support_episodic_mask(cm, cid, rename_class)
                masks_list.append(smask)
                ignore_list.append(sidx)
            out_masks.append(torch.stack(masks_list))
            out_ignores.append(torch.stack(ignore_list))
        return torch.stack(out_masks), torch.stack(out_ignores)

    def generate_query_episodic_mask(self, mask, rename_class):
        out = torch.zeros_like(mask)
        for c in torch.unique(mask):
            out[mask == c] = rename_class(int(c))
        boundary = (mask / 255).floor()
        return out, boundary

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        out = mask.clone()
        out[out != class_id] = 0
        out[out == class_id] = rename_class(class_id)
        boundary = (mask / 255).floor()
        return out, boundary
