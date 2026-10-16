r"""Parihaka few-shot classification and segmentation dataset (FS-CS compatible)."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.util.shape import view_as_windows
from PIL import Image


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def scale_to_256(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)


def extract_patches(img, shape, stride):
    """
    Extrai patches 2D de uma imagem (Z, X) usando view_as_windows.
    Robusta à imagem menor que o patch (retorna lista vazia).
    """
    z, x = img.shape
    sh, sw = shape
    sz, sx = stride

    if z < sh or x < sw:
        return []
    patches = view_as_windows(img, (sh, sw), step=(sz, sx))
    patches = patches.reshape(-1, sh, sw)
    return [p for p in patches]


# -------------------------------------------------------------------------
# DatasetParihaka
# -------------------------------------------------------------------------
class DatasetParihaka(Dataset):
    """
    Dataset few-shot (FS-CS) adaptado para o volume Parihaka.

    - Multi-way (way >= 1), multi-shot (shot >= 1).
    - Três modos de suporte:
        * "same_slice"
        * "anywhere"            (mesma região da query)
    - ensure_class_presence:
        * constrói um índice global de patches por classe (todas as regiões),
          usando SEMPRE o mesmo stride.
        * "same_slice": tenta primeiro na slice, se não houver vai para mesma região,
          se ainda não houver, cai para global.
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
        support_mode="same_slice",
        ensure_class_presence=False,
    ):
        # split interno: 'trn' / 'val' (val = val ou test)
        self.split = split
        self.fold = fold
        self.way = way
        self.shot = shot
        self.shape = shape
        self.stride = stride
        self.transform = transform

        # modos de suporte
        self.support_mode = support_mode
        # por padrão, teste usa o mesmo modo que treino

        self.ensure_class_presence = ensure_class_presence

        self.nclass = 6
        self.benchmark = "parihaka"
        self.nfolds = 1  # sem cross-validation formal

        # paths dos arquivos NPZ
        self.data_path = os.path.join(datapath, "data_train.npz")
        self.label_path = os.path.join(datapath, "labels_train.npz")

        # carrega dados
        read_seis_data = np.load(self.data_path, allow_pickle=True, mmap_mode="r")
        read_labels = np.load(self.label_path, allow_pickle=True, mmap_mode="r")

        seis_data = scale_to_256(read_seis_data["data"])
        labels = read_labels["labels"]

        # define regiões (Z, Y, X)
        self.regions = {
            "train": (seis_data[:, :622, :430], labels[:, :622, :430]),
            "val_cross": (seis_data[:, 622:702, :510], labels[:, 622:702, :510]),
            "val_inline": (seis_data[:, :702, 430:510], labels[:, :702, 430:510]),
            "test_cross": (seis_data[:, 702:782, :], labels[:, 702:782, :]),
            "test_inline": (seis_data[:, :, 510:590], labels[:, :, 510:590]),
        }

        # metadados sintéticos só para manter compatibilidade FS-CS
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids = list(range(1, self.nclass + 1))

        # índice global de patches por classe (se necessário)
        self.class_to_patch_refs = None
        if self.ensure_class_presence:
            self.class_to_patch_refs = self.build_global_patch_index()
        
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

        self.sampling_stats = {
            "hit_slice": 0,
            "fallback_region": 0,
            "fallback_global": 0,
            "fatal_random": 0
        }

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        # Episódios por época (ajuste se quiser)
        return len(self.img_metadata) if self.split == "trn" else 1000

    def __getitem__(self, idx):
        # Define modo de suporte dependendo do split
        support_mode = self.support_mode

        # sample_episode escolhe classes de suporte (FS-CS style)
        query_name, support_names, support_classes = self.sample_episode(idx)

        # extrai query + suporte com o modo desejado
        query_img, query_mask, support_imgs, support_masks, org_qry_imsize = self.load_frame(
            query_name,
            support_names,
            support_classes,
            support_mode=support_mode,
        )

        # presença de classes da support_classes na query_mask original
        query_class_presence = [
            s_c in torch.unique(query_mask) for s_c in support_classes
        ]

        # renomeia classes para o espaço local do episódio:
        #   suporte/query: classe s_c -> índice 1..way, outras -> 0
        rename_class = (
            lambda x: support_classes.index(x) + 1 if x in support_classes else 0
        )

        # aplica transform na query
        query_img = self.transform(query_img)

        # gera máscara da query no espaço episodic (0..way), + boundary/ignore
        query_mask, query_ignore = self.get_query_mask(
            query_img, query_mask, rename_class
        )

        # aplica transform nos suportes
        support_imgs = torch.stack(
            [torch.stack([self.transform(im) for im in supps]) for supps in support_imgs]
        )

        # gera máscaras de suporte (episódicas) e boundaries
        support_masks, support_ignores = self.get_support_masks(
            support_imgs, support_classes, support_masks, rename_class
        )

        batch = {
            "query_img": query_img,
            "query_mask": query_mask,
            "query_name": query_name,
            "query_ignore_idx": query_ignore,
            "org_query_imsize": org_qry_imsize,
            "support_imgs": support_imgs,
            "support_masks": support_masks,
            "support_names": support_names,
            "support_ignore_idxs": support_ignores,
            "support_classes": torch.tensor(support_classes),
            "query_class_presence": torch.tensor(query_class_presence),
        }
        return batch

    # ------------------------------------------------------------------
    # Construção de metadados
    # ------------------------------------------------------------------
    def build_img_metadata(self):
        """Placeholder simples — Parihaka não tem imagens explícitas,
        então simulamos índices numéricos como "nomes"."""
        n_items = 1000 if self.split != "trn" else 5000
        return [(f"patch_{i}", int(i % self.nclass) + 1) for i in range(n_items)]

    def build_img_metadata_classwise(self):
        class_dict = {i: [] for i in range(1, self.nclass + 1)}
        for name, cid in self.img_metadata:
            class_dict[cid].append(name)
        return class_dict

    # ------------------------------------------------------------------
    # Amostragem Episódica (FS-CS style)
    # ------------------------------------------------------------------
    def sample_episode(self, idx):
        """
        Inspiração no DatasetCOCO FS-CS:
        - sorteia support_classes (classes de suporte do episódio)
        - query_name aqui é simbólico (não é usado de fato para indexar arquivos),
          mas mantemos a interface.
        """
        if self.split in ["val", "test"]:
            np.random.seed(idx)

        # classes de suporte para o episódio
        support_classes = np.random.choice( 
            self.class_ids, self.way, replace=False
        ).tolist()

        # nome simbólico da query
        query_class = np.random.choice(support_classes)
        query_name = f"patch_{idx}_{query_class}"

        # nomes simbólicos dos suportes (não usados para acesso físico)
        support_names = [
            [f"patch_{idx}_{query_class}_{s}_{i}" for i in range(self.shot)]
            for s in support_classes
        ]

        return query_name, support_names, support_classes

    # ------------------------------------------------------------------
    # Índice global de patches por classe (para ensure_class_presence)
    # ------------------------------------------------------------------
    def build_global_patch_index(self):
        """
        Constrói um índice:
            class_to_patch_refs[c] = lista de (region_name, orientation, plane_idx, patch_idx)
        onde:
            - orientation ∈ {"inline", "crossline"}
            - plane_idx  : índice da inline/crossline naquela região
            - patch_idx  : índice do patch dentro daquela seção 2D
        Sempre usando o MESMO stride (self.stride).
        """
        class_to_refs = {c: [] for c in range(1, self.nclass + 1)}

        for region_name, (data, mask) in self.regions.items():
            Z, Y, X = data.shape

            # inlines: fixamos Y (eixo 1) → seção (Z, X)
            for y in range(Y):
                sec_d = data[:, y, :]
                sec_m = mask[:, y, :]
                patches = extract_patches(sec_d, self.shape, self.stride)
                masks = extract_patches(sec_m, self.shape, self.stride)
                for p_idx, m in enumerate(masks):
                    unique = np.unique(m)
                    for c in unique:
                        if c == 0:
                            continue
                        if 1 <= c <= self.nclass:
                            class_to_refs[c].append(
                                (region_name, "inline", y, p_idx)
                            )

            # crosslines: fixamos X (eixo 2) → seção (Z, Y)
            for x in range(X):
                sec_d = data[:, :, x]
                sec_m = mask[:, :, x]
                patches = extract_patches(sec_d, self.shape, self.stride)
                masks = extract_patches(sec_m, self.shape, self.stride)
                for p_idx, m in enumerate(masks):
                    unique = np.unique(m)
                    for c in unique:
                        if c == 0:
                            continue
                        if 1 <= c <= self.nclass:
                            class_to_refs[c].append(
                                (region_name, "crossline", x, p_idx)
                            )

        return class_to_refs

    def get_patch_from_ref(self, ref):
        """
        ref = (region_name, orientation, plane_idx, patch_idx)
        Retorna (patch_img, patch_mask) 2D (numpy arrays).
        """
        region_name, orientation, plane_idx, patch_idx = ref
        data, mask = self.regions[region_name]

        if orientation == "inline":
            sec_d = data[:, plane_idx, :]
            sec_m = mask[:, plane_idx, :]
        else:  # "crossline"
            sec_d = data[:, :, plane_idx]
            sec_m = mask[:, :, plane_idx]

        patches = extract_patches(sec_d, self.shape, self.stride)
        masks = extract_patches(sec_m, self.shape, self.stride)

        if len(patches) == 0:
            raise RuntimeError(
                f"No patches in get_patch_from_ref for {ref}, shape={sec_d.shape}"
            )
        if patch_idx >= len(patches):
            raise IndexError(
                f"patch_idx {patch_idx} out of range ({len(patches)}) in {ref}"
            )

        return patches[patch_idx], masks[patch_idx]

    # ------------------------------------------------------------------
    # Utilitários locais de classe na slice
    # ------------------------------------------------------------------
    def find_patches_with_class(self, data2d, mask2d, class_id):
        """
        data2d: seção 2D (Z, X)
        mask2d: máscara 2D (Z, X)
        Retorna lista de (patch_img, patch_mask) para patches que contêm `class_id`.
        Usa sempre self.shape, self.stride.
        """
        patches = extract_patches(data2d, self.shape, self.stride)
        masks = extract_patches(mask2d, self.shape, self.stride)
        valid = []
        for p, m in zip(patches, masks):
            if class_id in np.unique(m):
                valid.append((p, m))
        return valid

    # ------------------------------------------------------------------
    # Leitura de frames / episódios (Parihaka-style)
    # ------------------------------------------------------------------
    def load_frame(self, query_name, support_names, support_classes, support_mode="same_slice"):
        """
        Extrai patches verticais (inline ou crossline) de forma adaptativa.

        support_mode:
            "same_slice"          → suporte e query vêm da mesma fatia (prioridade)
            "anywhere"            → suporte vem de qualquer fatia na mesma região
        """
        shape = self.shape
        stride = self.stride

        # 1) Seleção da região/slice da QUERY
        if self.split == "trn":
            region_name = "train"
            data, mask = self.regions[region_name]

            if np.random.rand() < 0.5:
                # crossline
                x_idx = np.random.randint(data.shape[2])
                section_data = data[:, :, x_idx]
                section_mask = mask[:, :, x_idx]
                orientation = "crossline"
            else:
                # inline
                y_idx = np.random.randint(data.shape[1])
                section_data = data[:, y_idx, :]
                section_mask = mask[:, y_idx, :]
                orientation = "inline"
        else:
            # val/test → região de val/test (cross/inline)
            if np.random.rand() < 0.5:
                region_name = "val_cross" if self.split == "val" else "test_cross"
            else:
                region_name = "val_inline" if self.split == "val" else "test_inline"

            data, mask = self.regions[region_name]
            zdim, ydim, xdim = data.shape

            if ydim < shape[1] and xdim >= shape[1]:
                # força inline
                y_idx = np.random.randint(ydim)
                section_data = data[:, y_idx, :]
                section_mask = mask[:, y_idx, :]
                orientation = "inline (forced)"
            elif xdim < shape[1] and ydim >= shape[1]:
                # força crossline
                x_idx = np.random.randint(xdim)
                section_data = data[:, :, x_idx]
                section_mask = mask[:, :, x_idx]
                orientation = "crossline (forced)"
            else:
                if np.random.rand() < 0.5:
                    x_idx = np.random.randint(xdim)
                    section_data = data[:, :, x_idx]
                    section_mask = mask[:, :, x_idx]
                    orientation = "crossline"
                else:
                    y_idx = np.random.randint(ydim)
                    section_data = data[:, y_idx, :]
                    section_mask = mask[:, y_idx, :]
                    orientation = "inline"

        # 2) Extrair patches da slice da QUERY
        patches = extract_patches(section_data, shape, stride)
        masks = extract_patches(section_mask, shape, stride)
        if len(patches) == 0:
            raise RuntimeError(
                f"No valid patches found for split={self.split} ({orientation}). "
                f"Section shape: {section_data.shape}, patch size: {shape}"
            )

        # escolhe 1 patch aleatório para query
        q_idx = np.random.randint(len(patches))
        query_img_np = patches[q_idx]
        query_mask_np = masks[q_idx]

        query_img = Image.fromarray(np.uint8(query_img_np))
        query_mask = torch.tensor(query_mask_np, dtype=torch.int64)

        # 3) Preparar suporte
        support_imgs = []
        support_masks = []


        # sampling helpers por modo ---------------------
        def sample_support_same_slice(cid):
            """
            SAME_SLICE:
            - tenta pegar patch com classe cid na mesma slice
            - se não houver, tenta na mesma região (usando índice global)
            - se ainda não houver, cai para global
            """
            # primeiro, tenta na própria slice
            if self.ensure_class_presence:
                # 1. Tenta na própria slice
                valid = self.find_patches_with_class(section_data, section_mask, cid)
                if len(valid) > 0:
                    self.sampling_stats["hit_slice"] += 1  # <-- ADDED
                    p, m = valid[np.random.randint(len(valid))]
                    return p, m

                # 2. Tenta na mesma região
                if self.class_to_patch_refs is not None:
                    region_refs = [
                        ref for ref in self.class_to_patch_refs.get(cid, [])
                        if ref[0] == region_name
                    ]
                    if len(region_refs) > 0:
                        self.sampling_stats["fallback_region"] += 1  # <-- ADDED
                        ref = region_refs[np.random.randint(len(region_refs))]
                        return self.get_patch_from_ref(ref)

                    # 3. Tenta global
                    global_refs = self.class_to_patch_refs.get(cid, [])
                    if len(global_refs) > 0:
                        self.sampling_stats["fallback_global"] += 1  # <-- ADDED
                        ref = global_refs[np.random.randint(len(global_refs))]
                        return self.get_patch_from_ref(ref)

            # 4. Fallback final (cego)
            self.sampling_stats["fatal_random"] += 1  # <-- ADDED
            j = np.random.randint(len(patches))
            return patches[j], masks[j]

        def sample_support_anywhere(cid):
            """
            ANYWHERE (mesma região):
            - se ensure_class_presence: usa índice global filtrado pela região
            - senão: escolhe uma slice aleatória da região e pega patch aleatório
            """
            if self.ensure_class_presence and self.class_to_patch_refs is not None:
                region_refs = [
                    ref
                    for ref in self.class_to_patch_refs.get(cid, [])
                    if ref[0] == region_name
                ]
                if len(region_refs) > 0:
                    ref = region_refs[np.random.randint(len(region_refs))]
                    return self.get_patch_from_ref(ref)

                # não há na região → cai para global
                global_refs = self.class_to_patch_refs.get(cid, [])
                if len(global_refs) > 0:
                    ref = global_refs[np.random.randint(len(global_refs))]
                    return self.get_patch_from_ref(ref)

            # modo sem ensure_class_presence, ou caso não haja no índice:
            data_r, mask_r = self.regions[region_name]
            Zr, Yr, Xr = data_r.shape
            # tenta algumas slices aleatórias
            for _ in range(20):
                if np.random.rand() < 0.5:
                    y_idx_r = np.random.randint(Yr)
                    sec_d = data_r[:, y_idx_r, :]
                    sec_m = mask_r[:, y_idx_r, :]
                else:
                    x_idx_r = np.random.randint(Xr)
                    sec_d = data_r[:, :, x_idx_r]
                    sec_m = mask_r[:, :, x_idx_r]

                new_patches = extract_patches(sec_d, shape, stride)
                new_masks = extract_patches(sec_m, shape, stride)
                if len(new_patches) > 0:
                    j = np.random.randint(len(new_patches))
                    return new_patches[j], new_masks[j]

            # fallback para slice da query
            j = np.random.randint(len(patches))
            return patches[j], masks[j]

        # 4) Loop sobre classes de suporte (way)
        for cid in support_classes:
            imgs_c = []
            masks_c = []

            for _ in range(self.shot):
                if support_mode == "same_slice":
                    p, m = sample_support_same_slice(cid)
                elif support_mode == "anywhere":
                    p, m = sample_support_anywhere(cid)
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

    # ------------------------------------------------------------------
    # Geração de máscaras episódicas
    # ------------------------------------------------------------------
    def get_query_mask(self, query_img, query_cmask, rename_class):
        if self.split == "trn":
            query_cmask = F.interpolate(
                query_cmask.unsqueeze(0).unsqueeze(0).float(),
                query_img.size()[-2:],
                mode="nearest",
            ).squeeze()
        query_mask, query_ignore_idx = self.generate_query_episodic_mask(
            query_cmask.float(), rename_class
        )
        return query_mask, query_ignore_idx

    def get_support_masks(self, support_imgs, support_classes, support_cmasks, rename_class):
        support_masks = []
        support_ignore_idxs = []
        for cid, scmask_c in zip(support_classes, support_cmasks):
            masks_c = []
            ignores_c = []
            for scmask in scmask_c:
                scmask = torch.tensor(scmask, dtype=torch.float32)
                smask, sidx = self.generate_support_episodic_mask(
                    scmask, cid, rename_class
                )
                masks_c.append(smask)
                ignores_c.append(sidx)
            support_masks.append(torch.stack(masks_c))
            support_ignore_idxs.append(torch.stack(ignores_c))
        return torch.stack(support_masks), torch.stack(support_ignore_idxs)

    def generate_query_episodic_mask(self, mask, rename_class):
        """
        Renomeia toda a máscara da query:
          classe c -> rename_class(c)  (1..way), demais -> 0
        Também gera um "boundary"/ignore simples (placeholder aqui).
        """
        mask_renamed = torch.zeros_like(mask)
        boundary = (mask / 255).floor()  # placeholder; ajuste se tiver outra codificação

        classes = torch.unique(mask)
        for c in classes:
            mask_renamed[mask == c] = rename_class(int(c))

        return mask_renamed, boundary

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        """
        Máscara de suporte de 1 classe (cid):
          - zera tudo que não é class_id
          - renomeia class_id -> índice local (1..way)
        """
        mask_new = mask.clone()
        boundary = (mask_new / 255).floor()  # placeholder
        mask_new[mask_new != class_id] = 0
        mask_new[mask_new == class_id] = rename_class(class_id)
        return mask_new, boundary
