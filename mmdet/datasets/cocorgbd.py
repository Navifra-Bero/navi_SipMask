from mmdet.datasets import CocoDataset
from .builder import DATASETS
import mmcv
import os.path as osp

@DATASETS.register_module()
class RGBDCocoDataset(CocoDataset):
    def __init__(self, img_prefix_rgb=None, img_prefix_depth=None, *args, **kwargs):
        super(RGBDCocoDataset, self).__init__(*args, **kwargs)
        self.img_prefix_rgb = img_prefix_rgb
        self.img_prefix_depth = img_prefix_depth

    def __getitem__(self, idx):
        results = dict(
            img_info=self.data_infos[idx], 
            img_prefix_rgb=self.img_prefix_rgb, 
            img_prefix_depth=self.img_prefix_depth
        )
        ann_info = self.get_ann_info(idx)
        results['ann_info'] = ann_info

        # 'bbox_fields'와 'mask_fields' 초기화
        results['bbox_fields'] = []
        results['mask_fields'] = []

        # 디버깅을 위한 각 파이프라인 변환 후의 출력 추가
        for t in self.pipeline.transforms:
            results = t(results)
            if 'masks' in results and results['masks'] is not None:
                print(f"After {t.__class__.__name__}: masks shape = {results['masks'].shape}")
        
        return results
