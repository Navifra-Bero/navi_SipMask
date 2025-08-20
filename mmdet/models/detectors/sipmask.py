from .base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck

from mmdet.core import bbox2result
from mmcv.utils import print_log
from mmdet.utils import AvoidCUDAOOM
import os
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
@DETECTORS.register_module
class SipMask(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SipMask, self).__init__(init_cfg)
        if pretrained:
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
        bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    @AvoidCUDAOOM.retry_if_cuda_oom
    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        super(SipMask, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        for i in range(len(gt_labels)):
            gt_labels[i] = gt_labels[i] + 1

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_masks_list=gt_masks)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, aa in bbox_list
        ]

        segm_results = [
            aa
            for det_bboxes, det_labels, aa in bbox_list
        ]

        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError