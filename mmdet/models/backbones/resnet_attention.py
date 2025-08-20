import torch
import torch.nn as nn
from .resnet import ResNet
from ..builder import BACKBONES

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5  # Scale factor for stable training

    def forward(self, rgb_features, depth_features):
        B, N, C = rgb_features.size()  # B: batch size, N: sequence length, C: channels (embed_dim)
        if C != self.query.in_features:
            raise ValueError(f"Expected feature dimension {self.query.in_features}, but got {C}")

        Q = self.query(rgb_features)  # Query from RGB
        K = self.key(depth_features)  # Key from Depth
        V = self.value(depth_features)  # Value from Depth

        # Attention weights
        attention_weights = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        fused_features = attention_weights @ V
        return fused_features

@BACKBONES.register_module()
class ResnetAtt(nn.Module):
    def __init__(self, rgb_backbone_cfg, depth_backbone_cfg):
        super(ResnetAtt, self).__init__()
        # Initialize ResNet backbones
        self.rgb_backbone = ResNet(**rgb_backbone_cfg)
        self.depth_backbone = ResNet(**depth_backbone_cfg)

        # Initialize Attention modules with appropriate embed_dim for each stage
        self.attention_modules = nn.ModuleList([
            CrossAttention(embed_dim=256),
            CrossAttention(embed_dim=512),
            CrossAttention(embed_dim=1024),
            CrossAttention(embed_dim=2048)
        ])

    def forward(self, img):
        # Split img into img_rgb and img_depth
        img_rgb = img[:, :3, :, :]  # First 3 channels are RGB
        img_depth = img[:, 3:, :, :]  # Remaining channels are Depth

        # Extract features using the respective backbones
        rgb_features = self.rgb_backbone(img_rgb)
        depth_features = self.depth_backbone(img_depth)

        # Apply attention fusion at each stage
        fused_features = []
        for i, (rgb_f, depth_f) in enumerate(zip(rgb_features, depth_features)):
            # Reshape features to (batch, H*W, channels) for attention
            B, C, H, W = rgb_f.size()
            rgb_f = rgb_f.view(B, C, -1).permute(0, 2, 1)  # (batch, H*W, C)
            depth_f = depth_f.view(B, C, -1).permute(0, 2, 1)

            # Apply attention
            fused_feature = self.attention_modules[i](rgb_f, depth_f)
            fused_feature = fused_feature.permute(0, 2, 1).view(B, C, H, W)

            fused_features.append(fused_feature)
            
        print(len(fused_features))
        # Return fused features
        return fused_features
