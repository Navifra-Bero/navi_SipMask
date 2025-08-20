import torch
import torch.nn as nn
from .resnet import ResNet
from ..builder import BACKBONES

@BACKBONES.register_module()
class RGBDResNet(nn.Module):
    def __init__(self, rgb_backbone_cfg, depth_backbone_cfg):
        super(RGBDResNet, self).__init__()
        self.rgb_backbone = ResNet(**rgb_backbone_cfg)
        self.depth_backbone = ResNet(**depth_backbone_cfg)
    
    def forward(self, img):
        # img를 RGB와 Depth 채널로 분리
        img_rgb = img[:, :3, :, :]   # RGB는 첫 3채널
        img_depth = img[:, 3:, :, :]  # Depth는 그 다음 1채널
        # RGB와 Depth 이미지를 각 백본에 전달하여 각 계층별 특징 맵 추출
        rgb_features = self.rgb_backbone(img_rgb)
        depth_features = self.depth_backbone(img_depth)

        # concat
        combined_features = [
            torch.cat((rgb_f, depth_f), dim=1)
            for rgb_f, depth_f in zip(rgb_features, depth_features)
        ]
        
        
        # addition
        # combined_features = [
        #     rgb_f + depth_f
        #     for rgb_f, depth_f in zip(rgb_features, depth_features)
        # ]
        
        # 결합된 특징 맵을 반환
        return combined_features
