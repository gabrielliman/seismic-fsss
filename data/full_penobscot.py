r""" Penobscot few-shot classification and segmentation dataset (FS-CS compatible) """

import os
import numpy as np
import torch
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows
from PIL import Image


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def read_h5_file(file_path):
    f = h5py.File(file_path, 'r')
    images = f['features']
    labels = f['label']
    # The 8 facies are stored as 0..7; shift them to 1..8 (0 is the episodic background)
    return np.squeeze(np.array(images)), np.array(labels, dtype=np.uint8) + 1


def scale_to_256(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)


def extract_patches(img, shape, stride):
    z, x = img.shape
    sh, sw = shape
    sz, sx = stride

    if z < sh or x < sw:
        return []
    patches = view_as_windows(img, (sh, sw), step=(sz, sx))
    patches = patches.reshape(-1, sh, sw)
    return [p for p in patches]


# -------------------------------------------------------------------------
# DatasetPenobscot
# -------------------------------------------------------------------------
class DatasetFullPenobscot(Dataset):
    """
    Full upgrade version matching DatasetParihaka features:
    - support_mode
    - ensure_class_presence
    """

    def __init__(
        self,
        datapath,
        fold,
        transform,
        split,
        way,
        shot,
        shape=(400, 400),
        stride=(200, 200),
        support_mode="anywhere",
        ensure_class_presence=False,
    ):
        # split processing
        self.split = "test"
        self.fold = fold
        self.way = way
        self.shot = shot
        self.shape = shape
        self.stride = stride
        self.transform = transform

        # support modes
        self.support_mode = support_mode
        self.ensure_class_presence = ensure_class_presence

        # Penobscot class count
        self.nclass = 8
        self.class_ids = list(range(1, self.nclass + 1))
        self.benchmark = 'full_penobscot'

        # load H5
        data_path = os.path.join(datapath, 'penobscot.h5')
        images, masks = read_h5_file(data_path)

        # scale and reorder dims
        images = scale_to_256(images).astype(np.uint8)
        images = images.transpose((1, 0, 2))
        masks = masks.transpose((1, 0, 2))

        # crop x to 400
        test_data = images[:, :, :400]
        test_label = masks[:, :, :400]

        # FS-CS expects 4 keys
        self.regions = {
            'test': (test_data, test_label),
        }

        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.support_mode= support_mode
        self.ensure_class_presence = ensure_class_presence

        self.class_to_patch_refs = None
        if self.ensure_class_presence:
            self.class_to_patch_refs = self.build_global_patch_index()

    # -------------------------------------------------------------------------
    # Dataset length
    # -------------------------------------------------------------------------
    def __len__(self):
        return 5000 if self.split == "trn" else 1000

    # -------------------------------------------------------------------------
    # Episode Sampling
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        support_mode = self.support_mode

        query_name, support_names, support_classes = self.sample_episode(idx)

        query_img, query_mask, support_imgs, support_masks, org_size = \
            self.load_frame(query_name, support_names, support_classes, support_mode)

        # presence mask
        query_class_presence = [
            c in torch.unique(query_mask) for c in support_classes
        ]

        # rename to episodic class ids
        rename_class = lambda x: support_classes.index(x) + 1 if x in support_classes else 0

        # transforms
        query_img = self.transform(query_img)
        query_mask, query_ignore = self.get_query_mask(query_img, query_mask, rename_class)

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
            "query_ignore_idx": query_ignore,
            "support_imgs": support_imgs,
            "support_masks": support_masks,
            "support_ignore_idxs": support_ignores,
            "support_classes": torch.tensor(support_classes),
            "query_class_presence": torch.tensor(query_class_presence),
            "query_name": query_name,
            "support_names": support_names,
            "org_query_imsize": org_size,
        }
    def build_global_patch_index(self):
        """
        Varre todas as regiões do Penobscot, gera patches inline e crossline,
        e cria um índice global:
            class_to_refs[c] = list of (region_name, orientation, plane_idx, patch_idx)
        onde cada entrada aponta para um patch que contém aquela classe.
        """
        class_to_refs = {c: [] for c in self.class_ids}

        for region_name, (data, mask) in self.regions.items():
            Z, Y, X = data.shape

            # inline slices (fixa Y)
            for y in range(Y):
                sec_d = data[:, y, :]
                sec_m = mask[:, y, :]

                patches = extract_patches(sec_d, self.shape, self.stride)
                masks   = extract_patches(sec_m, self.shape, self.stride)

                for p_idx, m in enumerate(masks):
                    for c in np.unique(m):
                        if 1 <= c <= self.nclass:
                            class_to_refs[c].append((region_name, "inline", y, p_idx))

            # crossline slices (fixa X)
            for x in range(X):
                sec_d = data[:, :, x]
                sec_m = mask[:, :, x]

                patches = extract_patches(sec_d, self.shape, self.stride)
                masks   = extract_patches(sec_m, self.shape, self.stride)

                for p_idx, m in enumerate(masks):
                    for c in np.unique(m):
                        if 1 <= c <= self.nclass:
                            class_to_refs[c].append((region_name, "crossline", x, p_idx))

        return class_to_refs

    # -------------------------------------------------------------------------
    # FS-CS metadata
    # -------------------------------------------------------------------------
    def build_img_metadata(self):
        n = 5000 if self.split == "trn" else 1000
        return [(f"patch_{i}", (i % self.nclass) + 1) for i in range(n)]

    def build_img_metadata_classwise(self):
        d = {c: [] for c in self.class_ids}
        for name, cid in self.img_metadata:
            d[cid].append(name)
        return d

    # -------------------------------------------------------------------------
    # Episode sampler
    # -------------------------------------------------------------------------
    def sample_episode(self, idx):
        if self.split in ["val", "test"]:
            np.random.seed(idx)

        support_classes = np.random.choice(self.class_ids, self.way, replace=False).tolist()
        query_class = np.random.choice(support_classes)
        query_name = f"patch_{idx}_{query_class}"

        support_names = [
            [f"patch_{idx}_{query_class}_{s}_{i}" for i in range(self.shot)]
            for s in support_classes
        ]

        return query_name, support_names, support_classes


    def get_patch_from_ref(self, ref):
        region_name, orientation, plane_idx, patch_idx = ref
        data, mask = self.regions[region_name]

        if orientation == "inline":
            sec_d = data[:, plane_idx, :]
            sec_m = mask[:, plane_idx, :]
        else:
            sec_d = data[:, :, plane_idx]
            sec_m = mask[:, :, plane_idx]

        patches = extract_patches(sec_d, self.shape, self.stride)
        masks   = extract_patches(sec_m, self.shape, self.stride)

        return patches[patch_idx], masks[patch_idx]

    # -------------------------------------------------------------------------
    # Local class search
    # -------------------------------------------------------------------------
    def find_patches_with_class(self, data2d, mask2d, class_id):
        patches = extract_patches(data2d, self.shape, self.stride)
        masks = extract_patches(mask2d, self.shape, self.stride)
        out = []
        for p, m in zip(patches, masks):
            if class_id in np.unique(m):
                out.append((p, m))
        return out

    # -------------------------------------------------------------------------
    # load_frame
    # -------------------------------------------------------------------------
    #######################################################################
    # Frame loader with support modes + ensure_class_presence (Penobscot)
    #######################################################################
    def load_frame(self, query_name, support_names, support_classes, support_mode=None):
        """
        Extracts patches for query + support.

        support_mode:
            - "same_slice"           (rarely useful for Penobscot)
            - "anywhere"             (same region)
        """

        shape  = self.shape
        stride = self.stride
        support_mode = support_mode

        # 1) Choose region + slice for the QUERY -----------------------------------------
        region_name = "test"
        data, mask = self.regions["test"]

        # training supports both inline and crossline sampling
        if np.random.rand() < 0.5:
            # crossline slice
            x_idx = np.random.randint(data.shape[2])
            section_data = data[:, :, x_idx]     # (z, y)
            section_mask = mask[:, :, x_idx]
            orientation = "crossline"
        else:
            # inline slice
            y_idx = np.random.randint(data.shape[1])
            section_data = data[:, y_idx, :]     # (z, x)
            section_mask = mask[:, y_idx, :]
            orientation = "inline"

        # 2) Extract patches for QUERY ---------------------------------------------------
        patches = extract_patches(section_data, shape, stride)
        masks   = extract_patches(section_mask, shape, stride)

        if len(patches) == 0:
            raise RuntimeError(
                f"No patches found: section={section_data.shape}, split={self.split}, orientation={orientation}"
            )

        qi = np.random.randint(len(patches))
        query_img_np  = patches[qi]
        query_mask_np = masks[qi]

        query_img  = Image.fromarray(np.uint8(query_img_np))
        query_mask = torch.tensor(query_mask_np, dtype=torch.int64)

        # 3) Support patch sampling ------------------------------------------------------
        support_imgs  = []
        support_masks = []


        # Shortcuts to global index
        has_index = self.ensure_class_presence and (self.class_to_patch_refs is not None)

        # ------------------------- SUPPORT SAMPLERS -------------------------------------
        def sample_same_slice(cid):
            """Try same inline/crossline slice first; fallback to regional/global index."""
            if has_index:
                # find patches with target class in this slice
                valid = self.find_patches_with_class(section_data, section_mask, cid)
                if len(valid) > 0:
                    return valid[np.random.randint(len(valid))]

                # fallback same region
                region_refs = [r for r in self.class_to_patch_refs[cid] if r[0] == region_name]
                if len(region_refs) > 0:
                    ref = region_refs[np.random.randint(len(region_refs))]
                    return self.get_patch_from_ref(ref)

                # fallback global
                global_refs = self.class_to_patch_refs[cid]
                if len(global_refs) > 0:
                    ref = global_refs[np.random.randint(len(global_refs))]
                    return self.get_patch_from_ref(ref)

            # No ensure_class_presence -> random slice in same region
            idx = np.random.randint(len(patches))
            return patches[idx], masks[idx]

        def sample_anywhere(cid):
            """Same region only; uses class index if available."""
            if has_index:
                region_refs = [
                    ref for ref in self.class_to_patch_refs[cid] if ref[0] == region_name
                ]
                if len(region_refs) > 0:
                    ref = region_refs[np.random.randint(len(region_refs))]
                    return self.get_patch_from_ref(ref)

                # fallback global
                global_refs = self.class_to_patch_refs[cid]
                if len(global_refs) > 0:
                    ref = global_refs[np.random.randint(len(global_refs))]
                    return self.get_patch_from_ref(ref)

            # random Y slice sampling (Penobscot style)
            data_r, mask_r = self.regions[region_name]
            Yr = data_r.shape[1]

            for _ in range(20):
                y_idx_r = np.random.randint(Yr)
                sec_d = data_r[:, y_idx_r, :]
                sec_m = mask_r[:, y_idx_r, :]

                p2 = extract_patches(sec_d, shape, stride)
                m2 = extract_patches(sec_m, shape, stride)
                if len(p2) > 0:
                    j = np.random.randint(len(p2))
                    return p2[j], m2[j]

            # final fallback: same-slice random
            j = np.random.randint(len(patches))
            return patches[j], masks[j]

        # ------------------------- BUILD SUPPORT ----------------------------------------
        for cid in support_classes:
            imgs_c  = []
            masks_c = []

            for _ in range(self.shot):

                if support_mode == "same_slice":
                    p, m = sample_same_slice(cid)

                elif support_mode == "anywhere":
                    p, m = sample_anywhere(cid)

                else:
                    raise ValueError(f"Invalid support_mode={support_mode}")

                imgs_c.append(Image.fromarray(np.uint8(p)))
                masks_c.append(m)

            support_imgs.append(imgs_c)
            support_masks.append(masks_c)

        return query_img, query_mask, support_imgs, support_masks, query_img.size


    # -------------------------------------------------------------------------
    # Mask utilities
    # -------------------------------------------------------------------------
    def get_query_mask(self, query_img, query_cmask, rename_class):
        return self.generate_query_episodic_mask(query_cmask.float(), rename_class)

    def get_support_masks(self, support_imgs, support_classes, support_cmasks, rename_class):
        out_masks = []
        out_ignores = []

        for cid, scmask_list in zip(support_classes, support_cmasks):
            masks_c, ignores_c = [], []
            for scmask in scmask_list:
                scmask = torch.tensor(scmask, dtype=torch.float32)
                smask, sidx = self.generate_support_episodic_mask(scmask, cid, rename_class)
                masks_c.append(smask)
                ignores_c.append(sidx)

            out_masks.append(torch.stack(masks_c))
            out_ignores.append(torch.stack(ignores_c))

        return torch.stack(out_masks), torch.stack(out_ignores)

    def generate_query_episodic_mask(self, mask, rename_class):
        mask_renamed = torch.zeros_like(mask)
        boundary = (mask / 255).floor()

        for c in torch.unique(mask):
            mask_renamed[mask == c] = rename_class(int(c))

        return mask_renamed, boundary

    def generate_support_episodic_mask(self, mask, cid, rename_class):
        out = mask.clone()
        boundary = (out / 255).floor()
        out[out != cid] = 0
        out[out == cid] = rename_class(cid)
        return out, boundary
