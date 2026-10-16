from functools import reduce
from operator import add

import torch
from torchvision.models import resnet

from einops import rearrange

from model.base.feature import extract_feat_res
from model.base.correlation import Correlation
from model.ifsl import iFSLModule
from model.utils.backbone_patch import convert_resnet_to_grayscale
from model.module.asnet import AttentionLearner


class AttentiveSqueezeNetwork(iFSLModule):
    def __init__(self, args):
        super(AttentiveSqueezeNetwork, self).__init__(args)

        # 1. Backbone network initialization
        self.backbone_type = args.backbone
        if args.backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)

        # Seismic patches are single-channel: adapt conv1 to 1 input channel
        self.backbone = convert_resnet_to_grayscale(self.backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.learner = AttentionLearner(list(reversed(nbottlenecks[-3:])), args.way)
    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, shot, 3, H, W]  (ou [bsz, way, 3, H, W] no caso 1-shot)
        support_masks.shape : [bsz, way, shot, H, W]    (ou [bsz, way, H, W])
        '''

        support_imgs = batch['support_imgs']
        query_img = batch['query_img']

        # 🔧 Normaliza para sempre ter 6 dimensões
        if support_imgs.ndim == 4:              # [b, c, h, w]
            support_imgs = support_imgs.unsqueeze(1).unsqueeze(2)
        elif support_imgs.ndim == 5:            # [b, way, c, h, w]
            support_imgs = support_imgs.unsqueeze(2)

        # Agora garantimos 6D: [b, way, shot, c, h, w]
        b, w, s, c, h, w_dim = support_imgs.shape

        support_img = rearrange(support_imgs, 'b w s c h x -> (b w s) c h x')

        # Máscaras
        support_masks = batch['support_masks']
        if support_masks.ndim == 4:             # [b, way, h, w]
            support_masks = support_masks.unsqueeze(2)
        support_mask = rearrange(support_masks, 'b w s h x -> (b w s) h x')

        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids,
                                            self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids,
                                            self.bottleneck_ids, self.lids, pool=True)
            corr = Correlation.multilayer_correlation(query_feats, support_feats,
                                                    self.stack_ids, self.args.way)

        shared_masks = self.learner(corr, support_mask)
        shared_masks = torch.log_softmax(shared_masks, dim=2)
        return shared_masks


    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        cls_score_agg = 0
        support_imgs = batch['support_imgs'].clone()
        support_masks = batch['support_masks'].clone()
        for s_idx in range(nshot):
            batch['support_imgs'] = support_imgs[:, :, s_idx]
            batch['support_masks'] = support_masks[:, :, s_idx]
            shared_masks = self.forward(batch)
            pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
            cls_score_agg += pred_cls.clone()
            logit_mask_agg += logit_seg.clone()
            if nshot == 1:
                return pred_cls, pred_seg

        pred_cls = (cls_score_agg / float(nshot)) >= 0.5
        pred_seg = (logit_mask_agg / float(nshot)).argmax(dim=1)
        return pred_cls, pred_seg

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def configure_optimizers(self):
        # Filter out SSSN parameters so ASNet's optimizer only handles segmentation
        asnet_params = [p for n, p in self.named_parameters() if 'sssn' not in n]
        fss_opt = torch.optim.Adam([{"params": asnet_params, "lr": self.args.lr}])

        if self.args.use_sssn:
            sssn_opt = torch.optim.Adadelta(self.sssn.parameters(), lr=0.1)
            return [fss_opt, sssn_opt]

        return fss_opt
