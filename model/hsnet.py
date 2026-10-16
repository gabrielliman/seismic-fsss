r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn.functional as F
from torchvision.models import resnet

from model.base.feature import extract_feat_res
from model.base.correlation import Correlation
from model.ifsl import iFSLModule
from model.utils.backbone_patch import convert_resnet_to_grayscale
from model.module.hsnet import HPNLearner

from einops import rearrange


class HypercorrSqueezeNetwork(iFSLModule):
    def __init__(self, args):
        super(HypercorrSqueezeNetwork, self).__init__(args)

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
        self.learner = HPNLearner(list(reversed(nbottlenecks[-3:])), args.way)

    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, 3, H, W]
        support_masks.shape : [bsz, way, H, W]
        '''
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']
        query_img = batch['query_img']

        # ------------------------------------------------------
        # 1) Normalize dimensions
        # ------------------------------------------------------
        if support_imgs.ndim == 4:               # [b, c, h, w]
            support_imgs = support_imgs.unsqueeze(1).unsqueeze(2)
        elif support_imgs.ndim == 5:             # [b, way, c, h, w]
            support_imgs = support_imgs.unsqueeze(2)
        # Now guaranteed: [b, way, shot, c, h, w]

        if support_masks.ndim == 4:              # [b, way, h, w]
            support_masks = support_masks.unsqueeze(2)
        # Now guaranteed: [b, way, shot, h, w]

        b, w, s, c, h, w_dim = support_imgs.shape

        # ------------------------------------------------------
        # 2) Flatten way x shot to process them in parallel
        # ------------------------------------------------------
        support_img = rearrange(support_imgs, 'b w s c h x -> (b w s) c h x')
        support_mask = rearrange(support_masks, 'b w s h x -> (b w s) h x')

        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids, self.args.way)

        shared_masks = self.learner(corr)

        # B, N, 2, H, W
        shared_masks = torch.log_softmax(shared_masks, dim=2)
        return shared_masks

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

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
        # Filter out SSSN parameters so HSNet's optimizer only handles segmentation
        hsnet_params = [p for n, p in self.named_parameters() if 'sssn' not in n]
        fss_opt = torch.optim.Adam([{"params": hsnet_params, "lr": self.args.lr}])

        if self.args.use_sssn:
            sssn_opt = torch.optim.Adadelta(self.sssn.parameters(), lr=0.1)
            return [fss_opt, sssn_opt]

        return fss_opt
