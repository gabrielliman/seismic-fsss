import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows
from PIL import Image


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
    step_size = (end - start) / (num_limits + 1)
    return [int(start + i * step_size) for i in range(1, num_limits + 1)]


class DatasetParihakaEDS(Dataset):
    """
    Few-shot dataset para o esquema EDS (Equally Distant Slices) no Parihaka.
    Compatível com FS-CS, permitindo "same_slice" ou "anywhere".
    """

    def __init__(self, datapath, fold, transform, split, way, shot,
                 shape=(400, 400), stride=(128, 64), num_train=1, support_mode="same_slice", ensure_class_presence=True):
        self.split = split
        self.fold = fold
        self.way = way
        self.shot = shot
        self.shape = shape
        self.stride = stride
        self.transform = transform
        self.num_train = num_train
        self.support_mode = support_mode
        self.nclass = 6
        self.benchmark = 'parihaka'
        self.ensure_class_presence = ensure_class_presence
        self.class_to_patch_refs=None
        # === Carrega os dados
        self.data_path = os.path.join(datapath, 'data_train.npz')
        self.label_path = os.path.join(datapath, 'labels_train.npz')

        seis_data = np.load(self.data_path, allow_pickle=True, mmap_mode='r')['data']
        labels = np.load(self.label_path, allow_pickle=True, mmap_mode='r')['labels']

        seis_data = scale_to_256(seis_data)
        # labels[labels == 6] = 0  # normaliza classe

        # === Define regiões dinamicamente (Equally Distant Slices)
        self.regions = self.build_regions(seis_data, labels, num_train)
        if self.ensure_class_presence:
            self.class_to_patch_refs = self.build_global_patch_index()
        # === Metadados simulados
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids = list(range(1, self.nclass + 1))
        self.sampling_stats = {
            "hit_slice": 0,
            "fallback_global": 0,
            "fatal_random": 0,
            "blind_random": 0
            
        }

        #maybe this breaks tests
        valid_classes = [
                c for c in self.class_ids 
                if len(self.class_to_patch_refs.get(c, [])) > 0
            ]
            
        if len(valid_classes) < len(self.class_ids):
            missing = set(self.class_ids) - set(valid_classes)
            print(f"Removing empty classes from sampling pool: {missing}")
            
        self.class_ids = valid_classes
        #end of change 

    def build_global_patch_index(self):
        """
        Constrói:
            class_to_patch_refs[c] = lista de (split_name, region_idx, patch_idx)
        """
        class_to_refs = {c: [] for c in range(1, self.nclass + 1)}

        for split_name in ["train", "val", "test"]:
            for region_idx, (data2d, mask2d) in enumerate(self.regions[split_name]):
                patches = extract_patches(data2d, self.shape, self.stride)
                masks = extract_patches(mask2d, self.shape, self.stride)

                for p_idx, m in enumerate(masks):
                    unique = np.unique(m)
                    for c in unique:
                        if 1 <= c <= self.nclass:
                            class_to_refs[c].append(
                                (split_name, region_idx, p_idx)
                            )

        return class_to_refs
    def get_patch_from_ref(self, ref):
        split_name, region_idx, patch_idx = ref

        data2d, mask2d = self.regions[split_name][region_idx]

        patches = extract_patches(data2d, self.shape, self.stride)
        masks = extract_patches(mask2d, self.shape, self.stride)

        if patch_idx >= len(patches):
            raise IndexError("Patch index out of range")

        return patches[patch_idx], masks[patch_idx]
    # ----------------------------------------------------------------------
    def build_regions(self, seis_data, labels, num_train):
        """Define regiões de treino/val/test equivalentes à EDS_Parihaka."""
        x_limits = get_limits(0, seis_data.shape[1], num_train)
        y_limits = get_limits(0, seis_data.shape[2], num_train)

        train_regions, val_regions = [], []

        for pos_x, pos_y in zip(x_limits, y_limits):
            # --- Crossline e inline completas ---
            # Treino
            if pos_x < seis_data.shape[1] - 1:
                train_regions.append((
                    seis_data[:, pos_x, :],     # fatia crossline completa
                    labels[:, pos_x, :]
                ))
            if pos_y < seis_data.shape[2] - 1:
                train_regions.append((
                    seis_data[:, :, pos_y],     # fatia inline completa
                    labels[:, :, pos_y]
                ))

            # Validação (usa a próxima fatia)
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

        # --- Teste: regiões fora das slices de treino/val ---
        test_regions = []
        used_x = set(x_limits + [x + 1 for x in x_limits if x + 1 < seis_data.shape[1]])
        used_y = set(y_limits + [y + 1 for y in y_limits if y + 1 < seis_data.shape[2]])

        for x in range(seis_data.shape[1]):
            if x not in used_x:
                test_regions.append((seis_data[:, x, :], labels[:, x, :]))
        for y in range(seis_data.shape[2]):
            if y not in used_y:
                test_regions.append((seis_data[:, :, y], labels[:, :, y]))

        return {'train': train_regions, 'val': val_regions, 'test': test_regions}

    # ----------------------------------------------------------------------
    def __len__(self):
        return 5000 if self.split == 'trn' else 1000

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        query_name, support_names, support_classes = self.sample_episode(idx)
        support_mode = self.support_mode
        query_img, query_mask, support_imgs, support_masks, org_qry_imsize = self.load_frame(
            query_name, support_names,support_classes, support_mode=support_mode
        )

        query_class_presence = [
            s_c in torch.unique(query_mask) for s_c in support_classes
        ]
        rename_class = lambda x: support_classes.index(x) + 1 if x in support_classes else 0

        query_img = self.transform(query_img)
        query_mask, query_ignore = self.get_query_mask(query_img, query_mask, rename_class)

        support_imgs = torch.stack(
            [torch.stack([self.transform(im) for im in supps]) for supps in support_imgs]
        )
        support_masks, support_ignores = self.get_support_masks(
            support_imgs, support_classes, support_masks, rename_class
        )

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'query_ignore_idx': query_ignore,
            'org_query_imsize': org_qry_imsize,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'support_ignore_idxs': support_ignores,
            'support_classes': torch.tensor(support_classes),
            'query_class_presence': torch.tensor(query_class_presence),
        }
        return batch

    # ----------------------------------------------------------------------
    def build_img_metadata(self):
        n_items = 5000 if self.split == 'trn' else 1000
        return [(f"patch_{i}", int(i % self.nclass) + 1) for i in range(n_items)]

    def build_img_metadata_classwise(self):
        class_dict = {i: [] for i in range(1, self.nclass + 1)}
        for name, cid in self.img_metadata:
            class_dict[cid].append(name)
        return class_dict

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def sample_episode(self, idx):
        if self.split in ['val', 'test']:
            np.random.seed(idx)
            
        # FIX: Sample query_class from filtered class_ids
        query_class = np.random.choice(self.class_ids)
        query_name = f"patch_{idx}_{query_class}"
        
        # FIX: Sample support_classes from filtered class_ids
        support_classes = np.random.choice(
            self.class_ids, self.way, replace=False
        ).tolist()
        
        support_names = [
            [f"patch_{idx}_{query_class}_{s}_{i}" for i in range(self.shot)]
            for s in support_classes
        ]
        return query_name, support_names, support_classes

    # ----------------------------------------------------------------------
    def load_frame(self, query_name, support_names, support_classes, support_mode="same_slice"):

        shape = self.shape
        stride = self.stride

        # --------------------------------------------------
        # 1) Select region pool depending on split (query)
        # --------------------------------------------------
        region_pool = (
            self.regions["train"] if self.split == "trn"
            else self.regions["val"] if self.split == "val"
            else self.regions["test"]
        )

        # --------------------------------------------------
        # 2) Select one full 2D slice (region) for QUERY
        # --------------------------------------------------
        region_idx = np.random.randint(len(region_pool))
        data2d, mask2d = region_pool[region_idx]

        patches = extract_patches(data2d, shape, stride)
        masks   = extract_patches(mask2d, shape, stride)

        if len(patches) == 0:
            raise RuntimeError(
                f"No valid patches for split={self.split}. Section shape: {data2d.shape}"
            )

        # --------------------------------------------------
        # 3) QUERY
        # --------------------------------------------------
        q_idx = np.random.randint(len(patches))
        query_img_np  = patches[q_idx]
        query_mask_np = masks[q_idx]

        query_img  = Image.fromarray(np.uint8(query_img_np))
        query_mask = torch.tensor(query_mask_np, dtype=torch.int64)

        # --------------------------------------------------
        # 4) SUPPORT
        # --------------------------------------------------
        support_imgs  = []
        support_masks = []

        for cid in support_classes:

            imgs_c  = []
            masks_c = []

            for _ in range(self.shot):

                # ======================================================
                # SAME_SLICE
                # ======================================================
                if support_mode == "same_slice":

                    if self.ensure_class_presence:

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

                # ======================================================
                # ANYWHERE (same split only)
                # ======================================================
                elif support_mode == "anywhere":

                    split_name = (
                        "train" if self.split == "trn"
                        else "val" if self.split == "val"
                        else "test"
                    )

                    if self.ensure_class_presence:

                        refs = [
                            r for r in self.class_to_patch_refs.get(cid, [])
                            if r[0] == split_name
                        ]

                        if len(refs) > 0:
                            ref = refs[np.random.randint(len(refs))]
                            p, m = self.get_patch_from_ref(ref)
                        else:
                            data_s, mask_s = region_pool[
                                np.random.randint(len(region_pool))
                            ]
                            new_patches = extract_patches(data_s, shape, stride)
                            new_masks   = extract_patches(mask_s, shape, stride)
                            j = np.random.randint(len(new_patches))
                            p, m = new_patches[j], new_masks[j]

                    else:
                        data_s, mask_s = region_pool[
                            np.random.randint(len(region_pool))
                        ]
                        new_patches = extract_patches(data_s, shape, stride)
                        new_masks   = extract_patches(mask_s, shape, stride)
                        j = np.random.randint(len(new_patches))
                        p, m = new_patches[j], new_masks[j]

                else:
                    raise ValueError(
                        f"Invalid support_mode='{support_mode}'. "
                        f"Use 'same_slice' or 'anywhere'."
                    )

                imgs_c.append(Image.fromarray(np.uint8(p)))
                masks_c.append(m)

            support_imgs.append(imgs_c)
            support_masks.append(masks_c)

        return query_img, query_mask, support_imgs, support_masks, query_img.size



    # ----------------------------------------------------------------------
    def get_query_mask(self, query_img, query_cmask, rename_class):
        query_cmask = F.interpolate(
            query_cmask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:], mode='nearest'
        ).squeeze()
        query_mask, query_ignore_idx = self.generate_query_episodic_mask(query_cmask, rename_class)
        return query_mask, query_ignore_idx

    def get_support_masks(self, support_imgs, support_classes, support_cmasks, rename_class):
        support_masks, support_ignore_idxs = [], []
        for cid, scmask_c in zip(support_classes, support_cmasks):
            masks_c, ignores_c = [], []
            for scmask in scmask_c:
                scmask = torch.tensor(scmask, dtype=torch.float32)
                smask, sidx = self.generate_support_episodic_mask(scmask, cid, rename_class)
                masks_c.append(smask)
                ignores_c.append(sidx)
            support_masks.append(torch.stack(masks_c))
            support_ignore_idxs.append(torch.stack(ignores_c))
        return torch.stack(support_masks), torch.stack(support_ignore_idxs)

    def generate_query_episodic_mask(self, mask, rename_class):
        mask_renamed = torch.zeros_like(mask)
        for c in torch.unique(mask):
            mask_renamed[mask == c] = rename_class(int(c))
        boundary = (mask / 255).floor()
        return mask_renamed, boundary

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        mask_new = mask.clone()
        mask_new[mask_new != class_id] = 0
        mask_new[mask_new == class_id] = rename_class(class_id)
        boundary = (mask_new / 255).floor()
        return mask_new, boundary

