import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import HyperNet  # 确保 model.py 在同级目录且包含 HyperNet

# =========================================================
# 1. 基础组件定义 (Base Components)
# =========================================================

class Conv3dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True, eps=1e-5),
            nn.ReLU(inplace=False)
        )

class Mamba(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels, affine=True, eps=1e-5)
        self.relu = nn.ReLU(inplace=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.relu(out)
        out = torch.clamp(out, -100, 100)
        return residual + self.gamma * out

class VSS(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv3dReLU(channels, channels)
        self.conv2 = Conv3dReLU(channels, channels)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.gamma * self.conv2(self.conv1(x))

class GateAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True, eps=1e-5)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True, eps=1e-5)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            Conv3dReLU(in_ch, out_ch),
            Conv3dReLU(out_ch, out_ch)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# =========================================================
# 2. 主网络定义 (Full HGMamba-Net)
# =========================================================

class UMambaBot3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base_ch=16, use_vss=True, use_hyper=True):
        super().__init__()
        self.use_vss = use_vss
        self.use_hyper = use_hyper 
        
        print(f"\n{'='*50}")
        print(f"Initializing HGMamba-Net (Full Architecture)")
        print(f"  > VSS Module:      {'[ENABLED]' if use_vss else '[DISABLED]'}")
        print(f"  > HyperNet Module: {'[ENABLED]' if use_hyper else '[DISABLED]'}")
        print(f"{'='*50}\n")

        # --- Encoder Layers ---
        self.enc1 = nn.Sequential(
            Conv3dReLU(in_channels, base_ch),
            Conv3dReLU(base_ch, base_ch)
        )
        self.enc2 = nn.Sequential(
            Conv3dReLU(base_ch, base_ch*2),
            Conv3dReLU(base_ch*2, base_ch*2)
        )
        self.enc3 = nn.Sequential(
            Conv3dReLU(base_ch*2, base_ch*4),
            Mamba(base_ch*4)
        )
        
        # Mamba VSS Integration
        if self.use_vss:
            self.enc4 = nn.Sequential(
                Conv3dReLU(base_ch*4, base_ch*8),
                VSS(base_ch*8)
            )
        else:
            self.enc4 = nn.Sequential(
                Conv3dReLU(base_ch*4, base_ch*8),
                Conv3dReLU(base_ch*8, base_ch*8)
            )

        self.pool = nn.MaxPool3d(2)

        # --- BottleNeck with Hypergraph ---
        if self.use_hyper:
            self.hyper = HyperNet(base_ch*8, k=8, window_size=(4, 4, 4))
            self.gate4 = GateAttention(F_g=base_ch*8, F_l=base_ch*8, F_int=base_ch*4)
        else:
            self.hyper = None
            self.gate4 = None

        # --- Decoder Layers ---
        self.dec3 = DecoderBlock3D(base_ch*8 + base_ch*4, base_ch*4)
        self.dec2 = DecoderBlock3D(base_ch*4 + base_ch*2, base_ch*2)
        self.dec1 = DecoderBlock3D(base_ch*2 + base_ch, base_ch)
        
        self.final = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.01)

        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Model Parameters: {total_params:,}\n")

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
        elif isinstance(m, (nn.InstanceNorm3d, nn.LayerNorm)):
            if m.weight is not None: nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        x = torch.clamp(x, -10, 10)
        
        try:
            # Encoder
            x1 = self.enc1(x)
            x2 = self.enc2(self.pool(x1))
            x3 = self.enc3(self.pool(x2))
            x4 = self.enc4(self.pool(x3)) 

            # BottleNeck
            if self.use_hyper and self.hyper is not None:
                hyper_skip = self.hyper(x4) 
                gskip = self.gate4(x4, hyper_skip)
            else:
                gskip = x4

            # Decoder
            d3 = self.dec3(gskip, x3)
            d2 = self.dec2(d3, x2)
            d1 = self.dec1(d2, x1)

            out = self.final(d1)
            out = out * self.output_scale
            
            return out
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.zeros((x.shape[0], self.final.out_channels, *x.shape[2:]), device=x.device)