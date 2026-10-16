r""" Evaluation helpers """
import torch


class Evaluator:
    @classmethod
    def cls_prediction(cls, cls_score, gt):
        """
        Binary/multi-label classification:
        cls_score: probabilities/logits already thresholded at 0.5
        gt: ground-truth labels (0/1 or bool)
        """
        cls_pred = cls_score >= 0.5
        pred_correct = cls_pred == gt
        return pred_correct

    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def seg_prediction(cls, pred_mask, batch, ignore_index=255):
        """
        pred_mask: [B, H, W] (0/1 for foreground/background)
        batch['query_mask']: [B, H, W]
        batch['query_ignore_idx']: mask to ignore
        """
        gt_mask = batch.get('query_mask')

        # Apply ignore_index if provided
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx = query_ignore_idx * ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == ignore_index] = ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # torch.histc can't handle empty tensors
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device, dtype=torch.float32)
            else:
                _area_inter = torch.histc(_inter.float(), bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask.float(), bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask.float(), bins=2, min=0, max=1))

        area_inter = torch.stack(area_inter).t()
        area_pred  = torch.stack(area_pred).t()
        area_gt    = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union


class AverageMeter:
    """
    A class that logs and averages cls and seg metrics
    """
    def __init__(self, dataset, way, ignore_index=255):
        self.benchmark = dataset.benchmark
        self.way = way
        self.class_ids_interest = torch.tensor(dataset.class_ids)
        self.ignore_index = ignore_index

        if self.benchmark in ['parihaka', 'parihaka_eds', 'full_parihaka']:
            self.nclass = 6
        elif self.benchmark in ['penobscot', 'penobscot_eds', 'full_penobscot']:
            self.nclass = 8
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

        # +1 for background at index 0
        self.total_area_inter = torch.zeros((self.nclass + 1,), dtype=torch.float32)
        self.total_area_union = torch.zeros((self.nclass + 1,), dtype=torch.float32)
        self.ones = torch.ones((len(self.class_ids_interest),), dtype=torch.float32)

        # --- Segmentation loss tracking ---
        self.seg_loss_sum = 0.0
        self.seg_loss_count = 0.0

        # --- Classification loss / ER tracking ---
        self.cls_loss_sum = 0.0
        self.cls_er_sum = 0.0
        self.cls_loss_count = 0.0
        self.cls_er_count = 0.0

        # --- Classification counters (per class) ---
        self.cls_tp = torch.zeros(self.nclass, dtype=torch.float32)
        self.cls_fp = torch.zeros(self.nclass, dtype=torch.float32)
        self.cls_fn = torch.zeros(self.nclass, dtype=torch.float32)
        self.cls_total = torch.zeros(self.nclass, dtype=torch.float32)

        # --- Segmentation TP/FP/FN for F1 etc. (per class including bg) ---
        self.seg_tp = torch.zeros(self.nclass + 1, dtype=torch.float32)
        self.seg_fp = torch.zeros(self.nclass + 1, dtype=torch.float32)
        self.seg_fn = torch.zeros(self.nclass + 1, dtype=torch.float32)
        # support = number of GT pixels per class (used for weighted F1)
        self.seg_support = torch.zeros(self.nclass + 1, dtype=torch.float32)

        # --- Global pixel accuracy ---
        self.pixel_correct = 0.0
        self.pixel_labeled = 0.0

    # ---------------------------------------------------------------------
    #  SEGMENTATION UPDATES
    # ---------------------------------------------------------------------
    def update_seg(self, pred_mask, batch, loss=None):
        """
        pred_mask: [B, H, W] with values in [0..way] (0 = bg)
        batch['query_mask']: [B, H, W] with values [0..way] and ignore_index
        batch['support_classes']: [B, way] absolute class IDs
        """
        ignore_mask = batch.get('query_ignore_idx')
        gt_mask = batch.get('query_mask')
        support_classes = batch.get('support_classes')

        # Apply ignore mask, if any
        if ignore_mask is not None:
            pred_mask = pred_mask.clone()
            gt_mask = gt_mask.clone()
            pred_mask[ignore_mask == self.ignore_index] = self.ignore_index
            gt_mask[ignore_mask == self.ignore_index] = self.ignore_index

        # Move to CPU for metric aggregation
        pred_mask = pred_mask.cpu()
        gt_mask = gt_mask.cpu()
        support_classes = support_classes.cpu()

        class_dicts = self.return_class_mapping_dict(support_classes)

        samplewise_iou = []  # samplewise iou is for visualization only

        for class_dict, pred_mask_i, gt_mask_i in zip(class_dicts, pred_mask, gt_mask):
            # ----- IoU stats (relative classes 0..way) -----
            area_inter, area_union = self.intersect_and_union(pred_mask_i, gt_mask_i)

            # sample-wise IoU for visualization (foreground only)
            if torch.sum(gt_mask_i) == 0:  # no foreground (only bg)
                samplewise_iou.append(torch.tensor([float('nan')]))
            else:
                samplewise_iou.append(self.nanmean(area_inter[1:] / area_union[1:]))

            # Accumulate IoU in global arrays using absolute class IDs
            self.total_area_inter.scatter_reduce_(dim=0, index=class_dict, src=area_inter, reduce='sum', include_self=True)
            self.total_area_union.scatter_reduce_(dim=0, index=class_dict, src=area_union, reduce='sum', include_self=True)
            
            # ----- Pixel-accuracy and TP/FP/FN stats with ignore handling -----
            valid = gt_mask_i != self.ignore_index
            num_valid = valid.sum().item()
            if num_valid == 0:
                continue

            pred_valid = pred_mask_i[valid].float()
            gt_valid = gt_mask_i[valid].float()

            # Global pixel accuracy
            correct_pixels = (pred_valid == gt_valid).sum().item()
            self.pixel_correct += correct_pixels
            self.pixel_labeled += num_valid

            # Per class (relative indices 0..way)
            inter_valid = pred_valid[pred_valid == gt_valid]
            if inter_valid.numel() == 0:
                tp_rel = torch.zeros(self.way + 1, dtype=torch.float32)
            else:
                tp_rel = torch.histc(inter_valid, bins=self.way + 1, min=0, max=self.way)

            gt_count_rel = torch.histc(gt_valid,   bins=self.way + 1, min=0, max=self.way)
            pred_count_rel = torch.histc(pred_valid, bins=self.way + 1, min=0, max=self.way)

            fp_rel = pred_count_rel - tp_rel
            fn_rel = gt_count_rel - tp_rel
            support_rel = gt_count_rel  # GT pixels per relative class

            # Map relative indices [0..way] to absolute class ids via class_dict
            self.seg_tp.scatter_(dim=0, index=class_dict, src=tp_rel, reduce='add')
            self.seg_fp.scatter_(dim=0, index=class_dict, src=fp_rel, reduce='add')
            self.seg_fn.scatter_(dim=0, index=class_dict, src=fn_rel, reduce='add')
            self.seg_support.scatter_(dim=0, index=class_dict, src=support_rel, reduce='add')

        # Segmentation loss aggregation
        if loss is not None:
            bsz = float(pred_mask.shape[0])
            self.seg_loss_sum += loss * bsz
            self.seg_loss_count += bsz

        return torch.tensor(samplewise_iou) * 100.0

    def nanmean(self, v):
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        denom = (~is_nan).float().sum()
        if denom == 0:
            return torch.tensor(float('nan'))
        return v.sum() / denom

    def return_class_mapping_dict(self, support_classes):
        # [a, b] -> [0, a, b]
        # relative class index -> absolute class id
        bsz = support_classes.shape[0]
        bg_classes = torch.zeros(bsz, 1, dtype=support_classes.dtype, device=support_classes.device)
        class_dicts = torch.cat((bg_classes, support_classes), dim=1)
        return class_dicts

    def intersect_and_union(self, pred_mask, gt_mask):
        """
        pred_mask, gt_mask: [H, W] with values in [0..way] or ignore_index
        Ignore pixels with value == ignore_index (they won't appear in histc)
        """
        valid = gt_mask != self.ignore_index
        pred_valid = pred_mask[valid].float()
        gt_valid = gt_mask[valid].float()

        intersect = pred_valid[pred_valid == gt_valid]
        if intersect.numel() == 0:
            area_inter = torch.zeros(self.way + 1, dtype=torch.float32)
        else:
            area_inter = torch.histc(intersect, bins=(self.way + 1), min=0, max=self.way)

        area_pred_mask = torch.histc(pred_valid, bins=(self.way + 1), min=0, max=self.way)
        area_gt_mask   = torch.histc(gt_valid,   bins=(self.way + 1), min=0, max=self.way)
        area_union = area_pred_mask + area_gt_mask - area_inter
        return area_inter, area_union

    # ---------------------------------------------------------------------
    #  SEGMENTATION METRICS
    # ---------------------------------------------------------------------
    def compute_iou(self):
        """
        Mean IoU over classes of interest, dynamically ignoring classes
        that have no ground-truth support in the current evaluation split.
        """
        inter_interest = self.total_area_inter[self.class_ids_interest]
        union_interest = self.total_area_union[self.class_ids_interest]
        
        # Get ground-truth support specifically for the classes of interest
        support_interest = self.seg_support[self.class_ids_interest]
        
        # Calculate IoU for all classes (safeguarded against div by zero)
        iou_interest = inter_interest / torch.max(union_interest, self.ones)
        
        # Create a dynamic mask mirroring your Macro F1 logic
        valid_mask = support_interest > 0
        
        if valid_mask.any():
            miou = torch.mean(iou_interest[valid_mask])
        else:
            miou = torch.tensor(0.0)
            
        return miou * 100.0

    def compute_iou_per_class(self):
        """
        Return IoU for each class (including background).
        Returns a tensor of length nclass + 1, in percentage.
        """
        eps = 1e-6
        iou_per_class = self.total_area_inter / (self.total_area_union + eps)
        return iou_per_class * 100.0

    def compute_seg_accuracy(self):
        """
        Global pixel accuracy over all non-ignored pixels.
        """
        if self.pixel_labeled == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.pixel_correct / self.pixel_labeled * 100.0)

    def compute_seg_accuracy_per_class(self):
        """
        Per-class "recall" = TP / GT for each class (including background).
        """
        eps = 1e-6
        acc_pc = self.seg_tp / (self.seg_support + eps)
        return acc_pc * 100.0

    def compute_seg_f1_per_class(self):
        """
        Per-class F1 score based on TP/FP/FN (including background).
        """
        eps = 1e-6
        tp = self.seg_tp
        fp = self.seg_fp
        fn = self.seg_fn
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + eps)
        return f1 * 100.0

    def compute_seg_macro_f1(self):
        """
        Macro-F1 over classes that actually appear in the GT
        (seg_support > 0). Includes background if present.
        """
        f1_pc = self.compute_seg_f1_per_class()
        mask = self.seg_support > 0
        if mask.any():
            return f1_pc[mask].mean()
        else:
            return torch.tensor(0.0)

    def compute_seg_weighted_f1(self):
        """
        Weighted F1, weighted by GT support (seg_support).
        """
        f1_pc = self.compute_seg_f1_per_class()
        support = self.seg_support
        total_support = support.sum()
        if total_support == 0:
            return torch.tensor(0.0)
        return (f1_pc * support / total_support).sum()

    # ---------------------------------------------------------------------
    #  CLASSIFICATION METRICS
    # ---------------------------------------------------------------------
    def compute_cls_er(self):
        """
        Exact-match ratio over episodes: percentage of episodes where
        all class labels were predicted correctly.
        """
        if self.cls_er_count == 0:
            return 0.0
        return self.cls_er_sum / self.cls_er_count * 100.0

    def avg_seg_loss(self):
        if self.seg_loss_count == 0:
            return 0.0
        return self.seg_loss_sum / self.seg_loss_count

    def avg_cls_loss(self):
        if self.cls_loss_count == 0:
            return 0.0
        return self.cls_loss_sum / self.cls_loss_count

    def update_cls(self, pred_cls, gt_cls, loss=None):
        """
        Multi-label classification per episode.

        pred_cls: [B, way] (0/1 or bool)
        gt_cls:   [B, way] (0/1 or bool)

        We keep:
        - exact-match ER (all labels correct in the episode)
        - per-class TP/FP/FN and total positives
        """
        pred_cls = pred_cls.cpu().int()
        gt_cls = gt_cls.cpu().int()

        # Per-class counts
        self.cls_tp += (pred_cls & gt_cls).sum(dim=0).float()
        self.cls_fp += (pred_cls & (1 - gt_cls)).sum(dim=0).float()
        self.cls_fn += (((1 - pred_cls) & gt_cls)).sum(dim=0).float()
        self.cls_total += gt_cls.sum(dim=0).float()

        bsz = float(pred_cls.shape[0])

        # exact-match for ER
        samplewise_er = torch.all(pred_cls == gt_cls, dim=1).float()  # 1 if all labels match
        self.cls_er_sum += samplewise_er.sum().item()
        self.cls_er_count += bsz

        if loss is not None:
            self.cls_loss_sum += loss * bsz
            self.cls_loss_count += bsz

        # Return batch ER in percentage (for visualization)
        return samplewise_er.mean() * 100.0
