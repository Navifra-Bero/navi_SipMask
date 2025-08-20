from mmcls.models import ConvNeXt
import torch
import torch.nn as nn
from ..builder import BACKBONES

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):  # 작은 reduction_ratio 설정
        super(ChannelAttention, self).__init__()
        assert in_channels > 0, "in_channels must be greater than 0"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, max(1, in_channels // reduction_ratio), kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, in_channels // reduction_ratio), in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

@BACKBONES.register_module()
class RGBDConvNeXt(nn.Module):
    def __init__(self, arch='small', out_indices=[0, 1, 2, 3], **kwargs):
        super(RGBDConvNeXt, self).__init__()
        # ConvNeXt 백본 초기화
        self.convnext = ConvNeXt(
            arch=arch,
            out_indices=out_indices,
            **kwargs
        )
        # 4채널 입력을 처리하는 채널 어텐션 레이어 추가
        self.attention = ChannelAttention(in_channels=4)
        # 4채널 입력을 3채널로 변환하는 1x1 컨볼루션 레이어
        self.fusion_layer = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, img):
        # print("Input shape to backbone:", img.shape)  # 디버깅용
        x = self.attention(img)
        x = self.fusion_layer(x)
        return self.convnext(x)

