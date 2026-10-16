import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.siamese import SiameseNetwork


class iFSLModule(pl.LightningModule):
    def __init__(self, args):
        super(iFSLModule, self).__init__()
        self.args = args
        self.way = self.args.way
        self.range = torch.arange(args.way + 1, requires_grad=False).view(1, args.way + 1, 1, 1)
        self.learner = None

        if args.use_sssn:
            self.sssn = SiameseNetwork(args=args)
            self.sssn_bce = torch.nn.BCELoss()
            self.automatic_optimization = False

    def forward(self, batch):
        pass

    def train_mode(self):
        pass

    def configure_optimizers(self):
        pass

    def predict_mask_nshot(self, batch, nshot):
        pass

    def filter_supports_sssn(self, batch):
        """Scores candidate pool and overwrites batch with the top K shots."""
        B, way, pool_size, C, H, W = batch['support_imgs'].shape
        k_shot = self.args.shot
        
        # If pool size matches shot size, bypass filtering
        if pool_size <= k_shot:
            return 0.0 
        
        query_img = batch['query_img'] # [B, C, H, W]
        
        selected_imgs = []
        selected_masks = []
        selected_ignores = []
        sssn_loss = 0.0
        
        for b in range(B):
            b_imgs, b_masks, b_ignores = [], [], []
            for w in range(way):
                # Expand query to match candidate pool size for parallel scoring
                query_b = query_img[b].unsqueeze(0).expand(pool_size, -1, -1, -1)
                candidates = batch['support_imgs'][b, w] 
                
                # Squeeze(-1) to ensure shape is [pool_size]
                scores = self.sssn(query_b, candidates).squeeze(-1) 
                
                # Compute Siamese Loss (Only during training)
                if self.training:
                    # Target: 1.0 if query contains this support class, 0.0 otherwise
                    presence = batch['query_class_presence'][b, w].float()
                    targets = presence.expand(pool_size)
                    sssn_loss += self.sssn_bce(scores, targets)
                
                # Select Top K Candidates
                _, topk_idx = torch.topk(scores, k=k_shot, dim=0)
                
                b_imgs.append(candidates[topk_idx])
                b_masks.append(batch['support_masks'][b, w][topk_idx])
                if 'support_ignore_idxs' in batch and batch['support_ignore_idxs'] is not None:
                    b_ignores.append(batch['support_ignore_idxs'][b, w][topk_idx])
                    
            selected_imgs.append(torch.stack(b_imgs, dim=0))
            selected_masks.append(torch.stack(b_masks, dim=0))
            if 'support_ignore_idxs' in batch and batch['support_ignore_idxs'] is not None:
                selected_ignores.append(torch.stack(b_ignores, dim=0))
        
        # Overwrite the batch variables in-place with the top K winners
        batch['support_imgs'] = torch.stack(selected_imgs, dim=0)
        batch['support_masks'] = torch.stack(selected_masks, dim=0)
        if 'support_ignore_idxs' in batch and batch['support_ignore_idxs'] is not None:
            batch['support_ignore_idxs'] = torch.stack(selected_ignores, dim=0)
            
        # Average the loss over the batch and way
        if self.training:
            sssn_loss = sssn_loss / (B * way)
            
        return sssn_loss

    def training_step(self, batch, batch_idx):
        split = 'trn' if self.training else 'val'

        if self.args.use_sssn:
            
            if self.training:
                # --- MANUAL OPTIMIZATION (Train) ---
                opt_fss, opt_sssn = self.optimizers()

                # 1. Siamese Network Step
                opt_sssn.zero_grad()
                sssn_loss = self.filter_supports_sssn(batch)
                self.manual_backward(sssn_loss)
                opt_sssn.step()

                # 2. FSS Network Step
                opt_fss.zero_grad()
                shared_masks = self.forward(batch)
                pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
                loss = self.compute_seg_objective(logit_seg, batch['query_mask'])
                self.manual_backward(loss)
                opt_fss.step()
                
            else:
                # --- VALIDATION MODE (No Gradients) ---
                sssn_loss = self.filter_supports_sssn(batch)
                shared_masks = self.forward(batch)
                pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
                loss = self.compute_seg_objective(logit_seg, batch['query_mask'])

            # Logging
            with torch.no_grad():
                self.average_meter.update_cls(pred_cls, batch['query_class_presence'])
                self.average_meter.update_seg(pred_seg, batch, loss.item())
                self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
                
                # Only log SSSN loss if it's an actual tensor during training
                if self.training and isinstance(sssn_loss, torch.Tensor):
                    self.log(f'{split}/sssn_loss', sssn_loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)

            return None # Lightning requires returning None for manual optimization
            
        else:
            # --- AUTOMATIC OPTIMIZATION (Baseline) ---
            shared_masks = self.forward(batch)
            pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
            loss = self.compute_seg_objective(logit_seg, batch['query_mask'])

            with torch.no_grad():
                self.average_meter.update_cls(pred_cls, batch['query_class_presence'])
                self.average_meter.update_seg(pred_seg, batch, loss.item())
                self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
                
            return loss

    def on_train_epoch_end(self):
        self._shared_epoch_end()

    def validation_step(self, batch, batch_idx):
        # We call training_step but Lightning handles eval() and no_grad()
        self.training_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self._shared_epoch_end()

    def _shared_epoch_end(self):
        split = 'trn' if self.training else 'val'
        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        loss = self.average_meter.avg_seg_loss()

        metrics = {f'{split}/loss': loss,
                   f'{split}/miou': miou,
                   f'{split}/er': er}

        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, logger=(self.logger is not None))

        space = '\n\n' if split == 'val' else '\n'
        print(f'{space}[{split}] ep: {self.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er: {er:.3f}')

    def test_step(self, batch, batch_idx):

        if self.args.use_sssn:
            self.filter_supports_sssn(batch)

        pred_cls, pred_seg = self.predict_mask_nshot(batch, self.args.shot)
        self.average_meter.update_cls(pred_cls, batch['query_class_presence'], loss=None)
        self.average_meter.update_seg(pred_seg, batch, loss=None)

    def on_test_epoch_end(self):
        miou = self.average_meter.compute_iou()
        iou_per_class = self.average_meter.compute_iou_per_class()
        seg_acc = self.average_meter.compute_seg_accuracy()
        seg_acc_pc = self.average_meter.compute_seg_accuracy_per_class()
        seg_f1_pc = self.average_meter.compute_seg_f1_per_class()
        seg_macro_f1 = self.average_meter.compute_seg_macro_f1()
        seg_weighted_f1 = self.average_meter.compute_seg_weighted_f1()

        print("\n=== Segmentation per-class metrics ===")
        for i in range(len(iou_per_class)):
            print(f"Class {i:02d} | IoU: {iou_per_class[i]:6.2f} | Acc: {seg_acc_pc[i]:6.2f} | F1: {seg_f1_pc[i]:6.2f}")

        logs = {
            'test/miou': miou.item(),
            'test/seg_acc': seg_acc.item(),
            'test/seg_macro_f1': seg_macro_f1.item(),
            'test/seg_weighted_f1': seg_weighted_f1.item(),
        }

        for k, v in logs.items():
            # ljust is fine for printing, but for keys in logs, plain strings are better
            self.log(k, v, on_epoch=True, sync_dist=True)

    def predict_cls_and_mask(self, shared_masks, batch):
        logit_seg = self.merge_bg_masks(shared_masks)
        logit_seg = self.upsample_logit_mask(logit_seg, batch)
        with torch.no_grad():
            pred_cls = self.collect_class_presence(shared_masks)
            pred_seg = logit_seg.argmax(dim=1)
        return pred_cls, pred_seg, logit_seg

    def collect_class_presence(self, logit_mask):
        class_activation = logit_mask[:, :, 1].max(dim=-1)[0].max(dim=-1)[0] >= math.log(0.5)
        return class_activation.type(logit_mask.dtype).detach()

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def compute_seg_objective(self, logit_mask, gt_mask):
        return F.nll_loss(logit_mask, gt_mask.long())

    def merge_bg_masks(self, shared_fg_masks):
        logit_fg = shared_fg_masks[:, :, 1]
        logit_episodic_bg = shared_fg_masks[:, :, 0].mean(dim=1)
        logit_mask = torch.cat((logit_episodic_bg.unsqueeze(1), logit_fg), dim=1)
        return logit_mask

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
