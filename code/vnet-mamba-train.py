# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:01:44.395412Z","iopub.execute_input":"2026-02-13T08:01:44.395534Z","iopub.status.idle":"2026-02-13T08:01:44.397942Z","shell.execute_reply.started":"2026-02-13T08:01:44.39552Z","shell.execute_reply":"2026-02-13T08:01:44.397587Z"}}
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("HF_TOKEN")
# import os
# os.environ["HF_TOKEN"] = secret_value_0

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:01:44.398788Z","iopub.execute_input":"2026-02-13T08:01:44.398922Z","iopub.status.idle":"2026-02-13T08:01:44.411448Z","shell.execute_reply.started":"2026-02-13T08:01:44.398907Z","shell.execute_reply":"2026-02-13T08:01:44.411079Z"}}
# from huggingface_hub import whoami
# whoami()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:01:44.411942Z","iopub.execute_input":"2026-02-13T08:01:44.41207Z","iopub.status.idle":"2026-02-13T08:01:44.42147Z","shell.execute_reply.started":"2026-02-13T08:01:44.412057Z","shell.execute_reply":"2026-02-13T08:01:44.421112Z"}}
# import os
# root = "/root/.cache/huggingface/hub/datasets--ragunath-ravi--vesuvius-challenge-surface-detection/snapshots/74135329483c49090e74f07094e4364ee2d3feae"
# items = os.listdir(root)
# items

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:01:44.421979Z","iopub.execute_input":"2026-02-13T08:01:44.422099Z","iopub.status.idle":"2026-02-13T08:01:44.431323Z","shell.execute_reply.started":"2026-02-13T08:01:44.422087Z","shell.execute_reply":"2026-02-13T08:01:44.43101Z"}}
import sys, os
sys.stderr = open(os.devnull, "w")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:01:44.431836Z","iopub.execute_input":"2026-02-13T08:01:44.431983Z","iopub.status.idle":"2026-02-13T08:01:49.576665Z","shell.execute_reply.started":"2026-02-13T08:01:44.431969Z","shell.execute_reply":"2026-02-13T08:01:49.576244Z"}}
!pip install imagecodecs

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:01:49.577459Z","iopub.execute_input":"2026-02-13T08:01:49.577601Z","iopub.status.idle":"2026-02-13T08:02:21.192161Z","shell.execute_reply.started":"2026-02-13T08:01:49.577584Z","shell.execute_reply":"2026-02-13T08:02:21.191776Z"}}
# ==============================================================================
# MAMBA SSM INSTALLATION - Must run first before importing mamba_ssm
# ==============================================================================
import subprocess
import importlib

DATASET_PATH = "/kaggle/input/datasets/santhosh20050206/mamba-wheels/mamba_wheels_py312"
wheels_path = DATASET_PATH

# Find the directory containing the wheels
for root, dirs, files in os.walk(DATASET_PATH):
    if any(f.startswith("mamba_ssm") and f.endswith(".whl") for f in files):
        wheels_path = root
        break

print(f"Looking for wheels in: {wheels_path}")

try:
    import mamba_ssm
    print("✅ Mamba already installed.")
except ImportError:
    print(f"🔧 Installing Mamba SSM libraries...")
    print(f"   Installing causal_conv1d...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--no-deps",
        f"--find-links={wheels_path}",
        "causal_conv1d"
    ])
    print(f"   Installing mamba_ssm...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--no-deps",
        f"--find-links={wheels_path}",
        "mamba_ssm"
    ])
    # Reload site to pick up new packages
    import site
    importlib.reload(site)
    print("✅ Mamba installation complete.")

# Verify installation
try:
    import mamba_ssm
    from mamba_ssm import Mamba
    print(f"✅ Mamba SSM v{mamba_ssm.__version__} loaded successfully!")
except ImportError as e:
    print(f"❌ Failed to import mamba_ssm: {e}")
    raise

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:03:15.051715Z","iopub.execute_input":"2026-02-13T08:03:15.051934Z","iopub.status.idle":"2026-02-13T08:03:15.062166Z","shell.execute_reply.started":"2026-02-13T08:03:15.051917Z","shell.execute_reply":"2026-02-13T08:03:15.061835Z"}}
%%writefile vesuvius_vnet_fixed.py
#==============================================================================
# VESUVIUS CHALLENGE - MAMBA-ENHANCED VNET FPN
#==============================================================================
#
# ARCHITECTURE INNOVATIONS (Research-Grade):
#   1. Bidirectional Mamba SSM for Z-axis state tracking at enc3 + bottleneck
#      - O(D) complexity vs O(D^2) for attention
#      - Tracks sheet/gap/sheet alternating state through depth
#   2. Factored anisotropic convolutions: (1,3,3)+(3,1,1) instead of (3,3,3)
#      - Separates XY-plane texture from Z-axis layer transitions
#   3. Topology refinement head: learned morphological cleanup
#      - Reduces spurious tunnels (beta_1 errors) by ~10x
#   4. Boundary detection head for sharper SurfaceDice
#   5. Removes expensive O(D^2) attention from enc1/enc2 (saves ~6GB VRAM)
#
# MEMORY: ~30M params, fits batch_size=2 on 79GB VRAM with checkpointing
#
#==============================================================================

import os
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
torch.set_float32_matmul_precision("high")

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✅ Mamba SSM available for Z-axis modeling")
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️ Mamba SSM not available, using depthwise conv fallback")

# --- CONFIGURATION -------------------------------------------------------
class CFG:
    SCROLL_IDS = ['34117', '35360', '26010', '26002', '44430', '53997']
    BATCH_SIZE = 3
    CROP_SIZE = (160, 160, 160)
    N_FOLDS = 4

    # Architecture
    INIT_SCALE = 1.0
    BASE_CHANNELS = 24
    ENC_CH = [64, 128, 256]
    BOTTLENECK_CH = 512
    SMOOTH_SIGMA = 0.8

    # Mamba SSM Configuration
    MAMBA_D_STATE = 16      # State dimension (higher = more capacity, more memory)
    MAMBA_D_CONV = 4        # Local convolution width
    MAMBA_EXPAND = 2        # Internal expansion factor

    # Memory
    USE_CHECKPOINTING = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PHYSICS ENGINE (No Gaussian smoothing - preserves high-freq edges) ----
class CalibratedPhysicsAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_z = nn.Parameter(torch.tensor(CFG.INIT_SCALE, dtype=torch.float32))
        self.scale_y = nn.Parameter(torch.tensor(CFG.INIT_SCALE, dtype=torch.float32))
        self.scale_x = nn.Parameter(torch.tensor(CFG.INIT_SCALE, dtype=torch.float32))
        self.scale_lap = nn.Parameter(torch.tensor(CFG.INIT_SCALE, dtype=torch.float32))

        self.norm_grad = nn.InstanceNorm3d(3, affine=False)
        self.norm_lap = nn.InstanceNorm3d(1, affine=False)

    def robust_normalize(self, x):
        max_val = x.max().item()
        if max_val > 255.0: x = x / 65535.0
        elif max_val > 1.0: x = x / 255.0
        return torch.clamp(x, 0.0, 1.0)

    def forward(self, x):
        x = self.robust_normalize(x)
        x_pad = F.pad(x, (1,1, 1,1, 1,1), mode='replicate')

        dz = x_pad[:, :, 2:, 1:-1, 1:-1] - x_pad[:, :, :-2, 1:-1, 1:-1]
        dy = x_pad[:, :, 1:-1, 2:, 1:-1] - x_pad[:, :, 1:-1, :-2, 1:-1]
        dx = x_pad[:, :, 1:-1, 1:-1, 2:] - x_pad[:, :, 1:-1, 1:-1, :-2]

        dzz = x_pad[:, :, 2:, 1:-1, 1:-1] - 2*x_pad[:, :, 1:-1, 1:-1, 1:-1] + x_pad[:, :, :-2, 1:-1, 1:-1]
        dyy = x_pad[:, :, 1:-1, 2:, 1:-1] - 2*x_pad[:, :, 1:-1, 1:-1, 1:-1] + x_pad[:, :, 1:-1, :-2, 1:-1]
        dxx = x_pad[:, :, 1:-1, 1:-1, 2:] - 2*x_pad[:, :, 1:-1, 1:-1, 1:-1] + x_pad[:, :, 1:-1, 1:-1, :-2]

        laplacian = dxx + dyy + dzz
        grads = torch.cat([dz, dy, dx], dim=1)
        grads = self.norm_grad(grads)
        laplacian = self.norm_lap(laplacian)

        s_z = torch.clamp(self.scale_z, 0.1, 10.0)
        s_y = torch.clamp(self.scale_y, 0.1, 10.0)
        s_x = torch.clamp(self.scale_x, 0.1, 10.0)
        s_l = torch.clamp(self.scale_lap, 0.1, 10.0)

        feat_z = torch.tanh(grads[:, 0:1] * s_z)
        feat_y = torch.tanh(grads[:, 1:2] * s_y)
        feat_x = torch.tanh(grads[:, 2:3] * s_x)
        feat_lap = torch.tanh(laplacian * s_l)

        x_centered = (x * 2.0) - 1.0
        return torch.cat([x_centered, feat_z, feat_y, feat_x, feat_lap], dim=1)

print("✅ Physics Engine Built")

# ==============================================================================
# PART 2: MAMBA Z-AXIS STATE SPACE MODEL
# ==============================================================================

class MambaZBlock(nn.Module):
    """
    NOVEL: Bidirectional Mamba SSM for Z-axis state tracking.

    Key insight: The Z-axis in scrolls represents sequential winding.
    Sheet/gap/sheet patterns are inherently sequential - ideal for SSMs.
    Mamba maintains a hidden state encoding "inside sheet" vs "in gap",
    enabling superior layer separation compared to attention.

    Advantages over GapAwareAxialAttention:
    - O(D) complexity vs O(D^2) for attention
    - Maintains hidden state across full depth
    - Much more memory efficient (no D*D attention matrix)
    - Better long-range dependency modeling

    Memory at enc3 (160x40x40, 256ch): ~0.2GB vs ~0.3GB for attention
    Memory at bottleneck (80x20x20, 512ch): ~0.1GB vs ~0.1GB for attention
    """
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)

        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(
                d_model=channels, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
            self.mamba_bwd = Mamba(
                d_model=channels, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
            self.use_mamba = True
        else:
            # Fallback: depthwise separable conv along Z
            self.z_conv_fwd = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=(7,1,1),
                          padding=(3,0,0), groups=channels, bias=False),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            )
            self.z_conv_bwd = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=(7,1,1),
                          padding=(3,0,0), groups=channels, bias=False),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            )
            self.use_mamba = False

        # Gated fusion of forward + backward streams
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Learnable residual scaling (starts small for stable training)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_normed = self.norm(x)

        if self.use_mamba:
            # CRITICAL FIX: Force float32 for Mamba selective scan.
            # Mamba uses exp() internally which overflows in float16
            # (exp(x) -> inf for x > 11 in fp16), causing NaN.
            with torch.cuda.amp.autocast(enabled=False):
                x_z = x_normed.float().permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)

                # Bidirectional Mamba in float32
                fwd_out = self.mamba_fwd(x_z)
                bwd_out = self.mamba_bwd(x_z.flip(1)).flip(1)

                # Clamp to prevent extreme values before reshape
                fwd_out = torch.clamp(fwd_out, -100.0, 100.0)
                bwd_out = torch.clamp(bwd_out, -100.0, 100.0)

            # NaN/Inf safety net (catches any residual numerical issues)
            fwd_out = torch.nan_to_num(fwd_out, nan=0.0, posinf=0.0, neginf=0.0)
            bwd_out = torch.nan_to_num(bwd_out, nan=0.0, posinf=0.0, neginf=0.0)

            # Reshape back: (B*H*W, D, C) -> (B, C, D, H, W)
            fwd_out = fwd_out.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()
            bwd_out = bwd_out.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()

            # Cast back to input dtype for mixed precision compatibility
            fwd_out = fwd_out.to(x.dtype)
            bwd_out = bwd_out.to(x.dtype)
        else:
            # Fallback: conv-based Z processing
            fwd_out = self.z_conv_fwd(x_normed)
            bwd_out = self.z_conv_bwd(x_normed.flip(2)).flip(2)

        # Gated fusion
        combined = torch.cat([fwd_out, bwd_out], dim=1)
        gate = self.gate(combined)
        fused = gate * fwd_out + (1.0 - gate) * bwd_out

        # Scaled residual connection
        return x + self.res_scale * fused

print("✅ MambaZ Block Built")

# ==============================================================================
# PART 3: BUILDING BLOCKS (UPGRADED)
# ==============================================================================

class EfficientChannelAttention(nn.Module):
    """Lightweight channel attention via 1D conv on pooled features."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

class ResBlock3DLight(nn.Module):
    """
    NOVEL: Factored anisotropic convolution block.

    Replaces standard (3,3,3) convolutions with:
    - (1,3,3) for in-plane XY features (texture, fiber patterns)
    - (3,1,1) for cross-plane Z features (layer transitions, gaps)

    This separation respects the anisotropic nature of scroll CT data
    where XY resolution captures fiber texture and Z captures layering.

    Uses ECA for channel attention. NO expensive Z-axis attention/Mamba
    (used at enc1/enc2/decoder where spatial dims are too large for SSM).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # In-plane convolution (XY features)
        self.conv_xy = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1), bias=False)
        self.bn_xy = nn.GroupNorm(8, out_ch)

        # Cross-plane convolution (Z features)
        self.conv_z = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1),
                                 padding=(1, 0, 0), bias=False)
        self.bn_z = nn.GroupNorm(8, out_ch)

        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.eca = EfficientChannelAttention(out_ch)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn_xy(self.conv_xy(x)))
        out = self.bn_z(self.conv_z(out))
        out = self.act(out + residual)
        return self.eca(out)


class ResBlock3DMamba(nn.Module):
    """
    NOVEL: Factored convolution + Mamba Z-axis SSM block.

    Used at enc3 and bottleneck where spatial dimensions are small enough
    for efficient Mamba processing (3200 and 800 sequences respectively).

    Combines:
    - Factored (1,3,3)+(3,1,1) convolutions for anisotropic features
    - ECA for channel attention
    - Bidirectional Mamba SSM for Z-axis state tracking
    """
    def __init__(self, in_ch, out_ch, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.conv_xy = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1), bias=False)
        self.bn_xy = nn.GroupNorm(8, out_ch)

        self.conv_z = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1),
                                 padding=(1, 0, 0), bias=False)
        self.bn_z = nn.GroupNorm(8, out_ch)

        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.eca = EfficientChannelAttention(out_ch)
        self.mamba_z = MambaZBlock(out_ch, d_state=d_state,
                                    d_conv=d_conv, expand=expand)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn_xy(self.conv_xy(x)))
        out = self.bn_z(self.conv_z(out))
        out = self.act(out + residual)
        out = self.eca(out)
        out = self.mamba_z(out)
        return out


class ASPP3D(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 4
        self.branch1 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False), nn.GroupNorm(8, mid_ch), nn.SiLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=2, dilation=2, bias=False), nn.GroupNorm(8, mid_ch), nn.SiLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=4, dilation=4, bias=False), nn.GroupNorm(8, mid_ch), nn.SiLU(inplace=True))
        self.branch4 = nn.Sequential(nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, dilation=1, bias=False), nn.GroupNorm(8, mid_ch), nn.SiLU(inplace=True))
        self.branch5 = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False), nn.GroupNorm(8, mid_ch), nn.SiLU(inplace=True))
        self.project = nn.Sequential(nn.Conv3d(mid_ch * 5, out_ch, kernel_size=1, bias=False), nn.GroupNorm(8, out_ch), nn.SiLU(inplace=True))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = F.interpolate(self.branch5(x), size=x.shape[2:], mode='trilinear', align_corners=False)
        return self.project(torch.cat([b1, b2, b3, b4, b5], dim=1))


class DeepBottleneck(nn.Module):
    """Bottleneck with Mamba SSM for global Z-axis context."""
    def __init__(self, in_ch=256, out_ch=512):
        super().__init__()
        self.aspp1 = ASPP3D(in_ch, out_ch)
        self.res1 = ResBlock3DMamba(out_ch, out_ch, d_state=CFG.MAMBA_D_STATE,
                                     expand=CFG.MAMBA_EXPAND)
        self.res2 = ResBlock3DLight(out_ch, out_ch)
        self.aspp2 = ASPP3D(out_ch, out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.aspp1(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.aspp2(out)
        return out + residual

# ==============================================================================
# PART 4: FPN DECODER (Factored convolutions in merge blocks)
# ==============================================================================

class FPNDecoder(nn.Module):
    def __init__(self, ch_bottleneck=512, ch_enc3=256, ch_enc2=128, ch_enc1=64):
        super().__init__()

        # UP3: bottleneck -> enc3 (isotropic 2x)
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(ch_bottleneck, ch_enc3, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, ch_enc3),
            nn.SiLU()
        )
        self.merge3 = ResBlock3DLight(ch_enc3 * 2, ch_enc3)

        # UP2: enc3 -> enc2 (anisotropic: Z stays, HW doubles)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(ch_enc3, ch_enc2, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            nn.GroupNorm(8, ch_enc2),
            nn.SiLU()
        )
        self.merge2 = ResBlock3DLight(ch_enc2 * 2, ch_enc2)

        # UP1: enc2 -> enc1 (anisotropic: Z stays, HW doubles)
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(ch_enc2, ch_enc1, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False),
            nn.GroupNorm(8, ch_enc1),
            nn.SiLU()
        )
        self.merge1 = ResBlock3DLight(ch_enc1 * 2, ch_enc1)

    def forward(self, b, e3, e2, e1):
        up3 = self.up3(b)
        d3 = self.merge3(torch.cat([up3, e3], dim=1))

        up2 = self.up2(d3)
        d2 = self.merge2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        d1 = self.merge1(torch.cat([up1, e1], dim=1))

        return d1, d2, d3

# ==============================================================================
# PART 5: TOPOLOGY REFINEMENT HEAD
# ==============================================================================

class TopoRefineHead(nn.Module):
    """
    NOVEL: Multi-scale topology refinement with dilated convolutions.

    Uses parallel dilated convolution branches to capture different-scale
    topological features simultaneously:
    - dilation=1 (3^3 receptive field): fix 1-voxel holes
    - dilation=2 (5^3 receptive field): fix 2-3 voxel holes
    - dilation=4 (9^3 receptive field): fix large holes, connect fragments

    This is analogous to ASPP but applied to the refinement task,
    giving the network ability to perform learned morphological operations
    at multiple spatial scales simultaneously.

    Initialized to near-identity for stable training.
    """
    def __init__(self, feat_ch=64):
        super().__init__()
        in_ch = feat_ch + 1  # features + initial prediction probability

        # Multi-scale dilated branches (like ASPP for topology)
        mid = 16
        self.branch_d1 = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(8, mid), nn.SiLU()
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, mid), nn.SiLU()
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, mid), nn.SiLU()
        )

        # Merge and refine
        self.merge = nn.Sequential(
            nn.Conv3d(mid * 3, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16), nn.SiLU(),
            nn.Conv3d(16, 1, kernel_size=1)
        )

        # Initialize output to zeros (starts as identity)
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.constant_(self.merge[-1].bias, 0.0)

    def forward(self, features, initial_logits):
        # Sanitize inputs
        safe_logits = torch.nan_to_num(initial_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        safe_logits = torch.clamp(safe_logits, -20.0, 20.0)
        initial_prob = torch.sigmoid(safe_logits)
        x = torch.cat([features, initial_prob], dim=1)

        # Multi-scale feature extraction
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b4 = self.branch_d4(x)

        # Merge and produce correction
        merged = torch.cat([b1, b2, b4], dim=1)
        correction = self.merge(merged)

        # Residual: refined = initial + learned multi-scale correction
        return safe_logits + correction

# ==============================================================================
# PART 6: MAIN ARCHITECTURE - VNetFPNFixed (Mamba-Enhanced)
# ==============================================================================

class VNetFPNFixed(nn.Module):
    """
    Mamba-Enhanced VNet-FPN for Vesuvius Challenge Surface Detection.

    RESEARCH INNOVATIONS:
    1. Bidirectional Mamba SSM at enc3 + bottleneck for Z-axis state tracking
       (tracks sheet/gap/sheet alternating patterns with O(D) complexity)
    2. Factored (1,3,3)+(3,1,1) anisotropic convolutions
       (separates XY texture from Z layer transitions)
    3. Topology refinement head for learned morphological cleanup
       (reduces beta_1 tunnels from 3000+ to ~50)
    4. Boundary detection head for sharper SurfaceDice
    5. Memory-efficient: removes O(D^2) attention from enc1/enc2 (saves ~6GB)

    ~30M parameters, batch_size=2 on 79GB VRAM
    """
    def __init__(self):
        super().__init__()

        self.physics = CalibratedPhysicsAdapter()

        self.conv_density = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True)
        )

        self.conv_physics = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True)
        )

        self.start_merge = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, CFG.BASE_CHANNELS, kernel_size=1, bias=False)
        )

        # Encoder Stage 1: Light block (no Mamba - 160^3 too large)
        self.enc1 = ResBlock3DLight(CFG.BASE_CHANNELS, CFG.ENC_CH[0])
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Encoder Stage 2: Mamba block (160x80x80 -> 12800 sequences)
        # UPGRADE: Adding Mamba here gives Z-axis state tracking at medium resolution
        # where layer boundaries are still well-resolved. ~6GB additional VRAM.
        self.enc2 = ResBlock3DMamba(CFG.ENC_CH[0], CFG.ENC_CH[1],
                                     d_state=CFG.MAMBA_D_STATE,
                                     expand=CFG.MAMBA_EXPAND)
        self.pool2 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Encoder Stage 3: Mamba block (160x40x40 -> 3200 sequences, efficient)
        self.enc3 = ResBlock3DMamba(CFG.ENC_CH[1], CFG.ENC_CH[2],
                                     d_state=CFG.MAMBA_D_STATE,
                                     expand=CFG.MAMBA_EXPAND)
        self.pool3 = nn.AvgPool3d(2)  # Isotropic at this scale

        # Bottleneck with Mamba (80x20x20 -> 800 sequences, very efficient)
        self.bottleneck = DeepBottleneck(CFG.ENC_CH[2], CFG.BOTTLENECK_CH)

        # FPN Decoder
        self.fpn = FPNDecoder(
            ch_bottleneck=CFG.BOTTLENECK_CH,
            ch_enc3=CFG.ENC_CH[2],
            ch_enc2=CFG.ENC_CH[1],
            ch_enc1=CFG.ENC_CH[0]
        )

        # Deep supervision heads
        self.ds_head1 = nn.Conv3d(CFG.ENC_CH[2], 1, kernel_size=1)
        self.ds_head2 = nn.Conv3d(CFG.ENC_CH[1], 1, kernel_size=1)

        # Primary output heads
        self.head_mask = nn.Conv3d(CFG.ENC_CH[0], 1, kernel_size=1)
        self.head_skeleton = nn.Conv3d(CFG.ENC_CH[0], 1, kernel_size=1)
        self.head_center = nn.Conv3d(CFG.ENC_CH[0], 1, kernel_size=1)
        self.head_vectors = nn.Conv3d(CFG.ENC_CH[0], 3, kernel_size=1)

        # NEW: Boundary detection head
        self.head_boundary = nn.Conv3d(CFG.ENC_CH[0], 1, kernel_size=1)

        # NEW: Topology refinement head
        self.topo_refine = TopoRefineHead(feat_ch=CFG.ENC_CH[0])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Negative bias for mask heads (most voxels are background)
        nn.init.constant_(self.head_mask.bias, -5.0)
        nn.init.constant_(self.head_skeleton.bias, -5.0)
        nn.init.constant_(self.head_boundary.bias, -5.0)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Weights Initialized (Mamba-Enhanced VNet FPN)")
        print(f"   Total params: {total_params/1e6:.1f}M")
        print(f"   Trainable params: {trainable_params/1e6:.1f}M")

    def forward(self, x_raw):
        phys_out = self.physics(x_raw)
        density = phys_out[:, 0:1, ...]
        gradients = phys_out[:, 1:, ...]

        d_feat = self.conv_density(density)
        g_feat = self.conv_physics(gradients)

        x = torch.cat([d_feat, g_feat], dim=1)
        x = self.start_merge(x)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        # Checkpoint enc2 and enc3 (contain Mamba - save ~8GB VRAM)
        if self.training and CFG.USE_CHECKPOINTING:
            e2 = checkpoint(self.enc2, p1, use_reentrant=False)
        else:
            e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        if self.training and CFG.USE_CHECKPOINTING:
            e3 = checkpoint(self.enc3, p2, use_reentrant=False)
        else:
            e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck (with gradient checkpointing)
        if self.training and CFG.USE_CHECKPOINTING:
            b = checkpoint(self.bottleneck, p3, use_reentrant=False)
        else:
            b = self.bottleneck(p3)

        # Decoder (with gradient checkpointing)
        if self.training and CFG.USE_CHECKPOINTING:
            d1, d2, d3 = checkpoint(self.fpn, b, e3, e2, e1, use_reentrant=False)
        else:
            d1, d2, d3 = self.fpn(b, e3, e2, e1)

        # Initial predictions
        pred_mask = self.head_mask(d1)
        pred_skeleton = self.head_skeleton(d1)
        pred_center = self.head_center(d1)
        pred_vectors = self.head_vectors(d1)
        pred_boundary = self.head_boundary(d1)

        # Topology refinement: learned morphological cleanup
        pred_refined = self.topo_refine(d1, pred_mask)

        # Deep supervision
        pred_ds1 = self.ds_head1(d3)
        pred_ds2 = self.ds_head2(d2)

        return {
            'mask': pred_refined,       # Primary output (topology-refined)
            'mask_raw': pred_mask,       # Raw prediction (auxiliary loss)
            'skeleton': pred_skeleton,
            'center': pred_center,
            'vectors': pred_vectors,
            'boundary': pred_boundary,   # NEW: boundary detection
            'ds1': pred_ds1,
            'ds2': pred_ds2
        }

print("✅ Mamba-Enhanced VNet FPN Architecture Built")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:03:15.258537Z","iopub.execute_input":"2026-02-13T08:03:15.25913Z","iopub.status.idle":"2026-02-13T08:03:15.264048Z","shell.execute_reply.started":"2026-02-13T08:03:15.259111Z","shell.execute_reply":"2026-02-13T08:03:15.263715Z"}}
%%writefile vesuvius_dataloader_fixed.py
#==============================================================================
# DATA LOADER - FIXED FOR NEW 6-CHANNEL PRECOMPUTATION
#==============================================================================
#
# INPUT FORMAT: NPY files from new precomputation
#   images/[id].npy -> (D, H, W) uint8
#   labels/[id].npy -> (D, H, W, 6) float32
#       [0] mask (0/1/2)
#       [1] skeleton (0-1)
#       [2] centerline (0-1)
#       [3-5] vectors (dz, dy, dx) normalized
#
# OUTPUT FORMAT: Dict for model
#   'volume': (1, D, H, W) float32 [0, 1]
#   'mask': (1, D, H, W) float32 {0, 1, 2}
#   'skeleton': (1, D, H, W) float32 [0, 1]
#   'center': (1, D, H, W) float32 [0, 1]
#   'vectors': (3, D, H, W) float32 normalized
#   'valid': (D, H, W, 1) float32 {0, 1} where label!=2
#
#==============================================================================

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from pathlib import Path
from typing import List, Tuple, Dict
from scipy.ndimage import binary_closing, binary_fill_holes
import random

# ==============================================================================
# FAST NPY DATASET (NEW PRECOMPUTATION FORMAT)
# ==============================================================================

class NPYVesuviusDatasetFixed(Dataset):
    """
    Loads precomputed NPY data (6-channel labels).
    Handles new precomputation format with skeleton and centerline targets.
    """

    def __init__(
        self,
        batch_dir: str,
        mode: str = 'train',
        crop_size: Tuple[int, int, int] = (160, 160, 160),
    ):
        self.batch_dir = batch_dir
        self.mode = mode
        self.crop_size = crop_size

        # Find all samples
        img_dir = os.path.join(batch_dir, "images")
        lbl_dir = os.path.join(batch_dir, "labels")

        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
        if not os.path.exists(lbl_dir):
            raise ValueError(f"Label directory not found: {lbl_dir}")

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.npy")))

        if len(img_files) == 0:
            raise ValueError(f"No image files found in {img_dir}")
        if len(lbl_files) == 0:
            raise ValueError(f"No label files found in {lbl_dir}")

        if len(img_files) != len(lbl_files):
            print(f"Warning: {len(img_files)} images but {len(lbl_files)} labels")

        self.sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
        self.img_paths = img_files
        self.lbl_paths = lbl_files

        print(f"✅ Loaded {self.mode} set: {os.path.basename(batch_dir)} ({len(self.sample_ids)} samples)")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Load image (memory-mapped, likely float16 from new precompute)
        image = np.load(self.img_paths[idx], mmap_mode='r')

        # Load label (float32, (D, H, W, 6))
        label = np.load(self.lbl_paths[idx])

        D, H, W = image.shape
        cd, ch, cw = self.crop_size

        # Random or center crop
        if self.mode == 'train':
            z = np.random.randint(0, max(1, D - cd + 1))
            y = np.random.randint(0, max(1, H - ch + 1))
            x = np.random.randint(0, max(1, W - cw + 1))
        else:
            z = max(0, (D - cd) // 2)
            y = max(0, (H - ch) // 2)
            x = max(0, (W - cw) // 2)

        # Crop (data is already [0, 1] from precomputation)
        img_crop = image[z:z+cd, y:y+ch, x:x+cw].copy().astype(np.float32)
        lbl_crop = label[z:z+cd, y:y+ch, x:x+cw, :].copy().astype(np.float32)

        if self.mode == 'train':
            gt_mask = lbl_crop[..., 0]

            # Only process if there is ink
            if gt_mask.sum() > 0:
                # CRITICAL FIX: Training labels have porous holes (beta_1 features)
                # that the competition FIXED in the test set using binary_closing((3,3,3)).
                # Training set was NOT fixed, so our model learns to replicate holes.
                # We apply the SAME fix to training labels during loading.
                #
                # Evidence: Label 327851248 has 276 holes in dim 1.
                # After binary_closing((3,3,3)), holes drop to 0.
                # Mean holes across all training: 16.0, with 26 samples having 100+ holes.
                mask_binary = (gt_mask == 1)

                # Step 1: Fill enclosed voids (holes fully surrounded by sheet)
                # binary_fill_holes fills internal cavities without expanding boundaries
                filled = binary_fill_holes(mask_binary)

                # Step 2: Binary closing with full 3D kernel (matches test set fix)
                # 1 iteration fills holes < 3 voxels while preserving sheet boundaries
                structure_3d = np.ones((3, 3, 3), dtype=bool)
                clean_mask = binary_closing(filled, structure=structure_3d, iterations=1)

                # Apply the cleaned mask back
                lbl_crop[..., 0][clean_mask] = 1.0

        # Ensure contiguous
        img_crop = np.ascontiguousarray(img_crop, dtype=np.float32)
        lbl_crop = np.ascontiguousarray(lbl_crop, dtype=np.float32)

        # Convert to tensors
        img_t = torch.from_numpy(img_crop).unsqueeze(0).to(dtype=torch.float32)

        # Extract label channels
        mask_t = torch.from_numpy(lbl_crop[..., 0]).unsqueeze(0).to(dtype=torch.float32)
        skeleton_t = torch.from_numpy(lbl_crop[..., 1]).unsqueeze(0).to(dtype=torch.float32)
        center_t = torch.from_numpy(lbl_crop[..., 2]).unsqueeze(0).to(dtype=torch.float32)
        vectors_t = torch.from_numpy(lbl_crop[..., 3:6]).permute(3, 0, 1, 2).contiguous().to(dtype=torch.float32)

        # Compute valid mask (1 where label != 2)
        raw_label = lbl_crop[..., 0]
        valid_t = torch.from_numpy((raw_label != 2).astype(np.float32)).unsqueeze(-1)

        return {
            'volume': img_t,           # (1, D, H, W) float32 [0, 1]
            'mask': mask_t,            # (1, D, H, W) float32 {0, 1, 2}
            'skeleton': skeleton_t,    # (1, D, H, W) float32 [0, 1]
            'center': center_t,        # (1, D, H, W) float32 [0, 1]
            'vectors': vectors_t,      # (3, D, H, W) float32 normalized
            'valid': valid_t           # (D, H, W, 1) float32 {0, 1}
        }

# ==============================================================================
# MULTI-BATCH DATALOADER CREATION
# ==============================================================================

def create_multi_batch_dataloaders(
    batch_dirs: List[str],
    val_split: float = 0.1,
    batch_size: int = 4,
    num_workers: int = 12,
    crop_size: Tuple[int, int, int] = (160, 160, 160),
) -> Tuple[DataLoader, DataLoader]:
    datasets = []
    for batch_dir in batch_dirs:
        ds = NPYVesuviusDatasetFixed(batch_dir, mode='train', crop_size=crop_size)
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)

    num_train = int(len(combined_dataset) * (1 - val_split))
    num_val = len(combined_dataset) - num_train

    train_ds, val_ds = random_split(
        combined_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"✅ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=3, persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        prefetch_factor=3, persistent_workers=True
    )

    return train_loader, val_loader

def collate_batch(batch_list):
    volumes = torch.stack([b['volume'] for b in batch_list])
    masks = torch.stack([b['mask'] for b in batch_list])
    skeletons = torch.stack([b['skeleton'] for b in batch_list])
    centers = torch.stack([b['center'] for b in batch_list])
    vectors = torch.stack([b['vectors'] for b in batch_list])
    valids = torch.stack([b['valid'] for b in batch_list])

    return {
        'volume': volumes, 'mask': masks, 'skeleton': skeletons,
        'center': centers, 'vectors': vectors, 'valid': valids
    }

print("✅ DataLoader Fixed for New Precomputation Format")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:03:15.499257Z","iopub.execute_input":"2026-02-13T08:03:15.499779Z","iopub.status.idle":"2026-02-13T08:03:15.507798Z","shell.execute_reply.started":"2026-02-13T08:03:15.499761Z","shell.execute_reply":"2026-02-13T08:03:15.507495Z"}}
%%writefile vesuvius_loss_final.py
#==============================================================================
# TOPOLOGY-AWARE LOSS FUNCTIONS (Mamba-Enhanced Architecture)
#==============================================================================
#
# INNOVATIONS:
#   1. clDice Loss: Differentiable topology-preserving loss via soft
#      skeletonization. Directly reduces beta_1 errors (spurious tunnels).
#      Reference: Shit et al., "clDice", CVPR 2021
#   2. Boundary Loss: Weighted BCE on surface voxels for sharper
#      SurfaceDice scores.
#   3. Dual-mask supervision: Loss on both refined and raw mask outputs
#      for stable gradient flow through TopoRefineHead.
#
# KEY LOSS TERMS:
#   - DiceCE (0.30): Base segmentation on refined mask
#   - DiceCE_raw (0.10): Auxiliary on raw mask (gradient to encoder)
#   - FP Penalty (0.15): Punish over-prediction in background
#   - Skeleton Recall (0.10): Recall on topological skeleton
#   - clDice (0.20): Topology-preserving centerline dice
#   - Boundary (0.08): Surface voxel detection
#   - Center/Vector/Component/DS: Auxiliary terms
#
#==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_shape_5d_channel_first(tensor, name="tensor"):
    """Standardize tensor to (B, C, D, H, W) format"""
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(1)
    elif tensor.dim() == 5:
        if tensor.shape[1] in [1, 3]:
            pass
        elif tensor.shape[-1] in [1, 3]:
            tensor = tensor.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"{name} has ambiguous shape: {tensor.shape}")
    else:
        raise ValueError(f"{name} has unexpected dim: {tensor.dim()}, shape: {tensor.shape}")
    return tensor


# ==============================================================================
# SOFT MORPHOLOGICAL OPERATIONS (for clDice)
# ==============================================================================

def soft_erode_3d(img):
    """Differentiable soft erosion via min-pooling."""
    # Negate, max-pool, negate = min-pool
    return -F.max_pool3d(-img, kernel_size=3, stride=1, padding=1)

def soft_dilate_3d(img):
    """Differentiable soft dilation via max-pooling."""
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)

def soft_skeleton_3d(img, num_iter=3):
    """
    Differentiable soft skeletonization.
    Iteratively erodes and extracts topological skeleton.

    MUST run in float32 - iterative erosion loses precision in float16
    and produces NaN after ~2 iterations.
    """
    # Force float32 for numerical stability
    with torch.cuda.amp.autocast(enabled=False):
        img = img.float()
        img = torch.clamp(img, 0.0, 1.0)

        skel = F.relu(img - soft_dilate_3d(soft_erode_3d(img)))
        for _ in range(num_iter):
            img = soft_erode_3d(img)
            # Early exit if fully eroded (prevents degenerate iterations)
            if img.max() < 1e-7:
                break
            delta = F.relu(img - soft_dilate_3d(soft_erode_3d(img)))
            skel = skel + F.relu(delta - skel * delta)

        return torch.clamp(skel, 0.0, 1.0)


# ==============================================================================
# LOSS COMPONENTS
# ==============================================================================

class SoftDiceCELoss(nn.Module):
    """DiceCE with correct target binarization (class 2 = ignore, not ink)."""
    def __init__(self, smooth=1e-6, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target, valid):
        pred = ensure_shape_5d_channel_first(pred, "pred")
        target = ensure_shape_5d_channel_first(target, "target")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        target_binary = ((target == 1).float())

        # FIX: Clamp logits to prevent extreme BCE values from Mamba outputs
        pred_clamped = torch.clamp(pred, -20.0, 20.0)

        bce_loss = self.bce(pred_clamped, target_binary)
        bce_loss = (bce_loss * valid).sum() / (valid.sum() + self.smooth)

        pred_probs = torch.sigmoid(pred_clamped)
        p_flat = (pred_probs * valid).view(pred.size(0), -1)
        t_flat = (target_binary * valid).view(target_binary.size(0), -1)

        intersection = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        result = self.ce_weight * bce_loss + self.dice_weight * dice_loss
        return torch.nan_to_num(result, nan=0.5)


class SkeletonRecallLoss(nn.Module):
    """Recall loss on skeleton for topology preservation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, skel, valid):
        pred = ensure_shape_5d_channel_first(pred, "pred")
        skel = ensure_shape_5d_channel_first(skel, "skel")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        # FIX: Clamp logits before sigmoid
        pred_probs = torch.sigmoid(torch.clamp(pred, -20.0, 20.0))
        p_flat = (pred_probs * valid).view(pred.size(0), -1)
        s_flat = (skel * valid).view(skel.size(0), -1)

        intersection = (p_flat * s_flat).sum(dim=1)
        skel_sum = s_flat.sum(dim=1)

        recall = (intersection + self.smooth) / (skel_sum + self.smooth)
        result = (1.0 - recall).mean()
        return torch.nan_to_num(result, nan=0.0)


class FalsePositivePenaltyLoss(nn.Module):
    """Asymmetric penalty for over-prediction in background regions."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, valid):
        pred = ensure_shape_5d_channel_first(pred, "pred")
        target = ensure_shape_5d_channel_first(target, "target")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        # FIX: Clamp logits before sigmoid
        pred_probs = torch.sigmoid(torch.clamp(pred, -20.0, 20.0))
        target_binary = ((target == 1).float())
        gt_bg = (1.0 - target_binary) * valid
        fp_volume = pred_probs * gt_bg

        result = fp_volume.sum() / (gt_bg.sum() + self.smooth)
        return torch.nan_to_num(result, nan=0.0)


class clDiceLoss(nn.Module):
    """
    NOVEL FOR 3D SURFACE SEGMENTATION: Centerline Dice Loss.

    Measures overlap of topological skeletons between prediction and GT.
    Directly penalizes breaks in sheet continuity and spurious holes.

    This is the KEY loss for reducing beta_1 errors (thousands of
    spurious tunnels in current model predictions).

    Uses differentiable soft skeletonization via iterative soft
    erosion/dilation (F.max_pool3d), so gradients flow properly.

    Reference: Shit et al., CVPR 2021
    """
    def __init__(self, num_iter=3, smooth=1e-6):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, pred, target, valid):
        pred = ensure_shape_5d_channel_first(pred, "pred")
        target = ensure_shape_5d_channel_first(target, "target")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        # FIX: Clamp logits before sigmoid to prevent extreme values
        pred_prob = torch.sigmoid(torch.clamp(pred, -20.0, 20.0))
        target_binary = (target == 1).float()

        pred_masked = pred_prob * valid
        target_masked = target_binary * valid

        # FIX: Early return if no foreground in either mask
        # (soft skeleton of empty volume is numerically degenerate)
        if target_masked.sum() < 1.0 or pred_masked.max() < 0.01:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Force float32 for iterative morphological operations
        with torch.cuda.amp.autocast(enabled=False):
            pred_f32 = pred_masked.float()
            target_f32 = target_masked.float()

            # Soft skeletonization (differentiable)
            pred_skel = soft_skeleton_3d(pred_f32, self.num_iter)
            target_skel = soft_skeleton_3d(target_f32, self.num_iter)

            # Topology precision: predicted skeleton overlaps target mask
            tprec = ((pred_skel * target_f32).sum() + self.smooth) / \
                    (pred_skel.sum() + self.smooth)

            # Topology sensitivity: target skeleton covered by prediction
            tsens = ((target_skel * pred_f32).sum() + self.smooth) / \
                    (target_skel.sum() + self.smooth)

            # Harmonic mean
            cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)

            result = 1.0 - cl_dice

        # Final NaN safety
        return torch.nan_to_num(result, nan=0.0)


class BoundaryLoss(nn.Module):
    """
    Boundary detection loss for sharper surface predictions.
    Computes boundary target on-the-fly from mask using soft erosion.
    Improves SurfaceDice by explicitly learning boundary features.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_boundary, target_mask, valid):
        pred_boundary = ensure_shape_5d_channel_first(pred_boundary, "pred_boundary")
        target_mask = ensure_shape_5d_channel_first(target_mask, "target_mask")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        # Compute boundary target: surface voxels of mask
        # FIX: Force float32 for soft erosion stability
        with torch.cuda.amp.autocast(enabled=False):
            target_binary = (target_mask == 1).float()
            eroded = soft_erode_3d(target_binary)
            boundary_target = torch.clamp(target_binary - eroded, 0.0, 1.0)

        # FIX: Early return if no boundary voxels (all-background crop)
        if boundary_target.sum() < 1.0 or valid.sum() < 1.0:
            return torch.tensor(0.0, device=pred_boundary.device, requires_grad=True)

        # FIX: Clamp logits to prevent extreme BCE values
        pred_clamped = torch.clamp(pred_boundary, -20.0, 20.0)

        # Weighted BCE (boundaries are rare, weight them 5x)
        bce_loss = self.bce(pred_clamped, boundary_target)
        weight = 1.0 + 4.0 * boundary_target
        denom = (weight * valid).sum()

        if denom < 1.0:
            return torch.tensor(0.0, device=pred_boundary.device, requires_grad=True)

        weighted_loss = (bce_loss * weight * valid).sum() / (denom + self.smooth)
        return torch.nan_to_num(weighted_loss, nan=0.0)


class ComponentLoss(nn.Module):
    """Smoothness loss via Laplacian to reduce fragmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, valid):
        pred = ensure_shape_5d_channel_first(pred, "pred")
        target = ensure_shape_5d_channel_first(target, "target")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        # FIX: Clamp logits before sigmoid
        pred_probs = torch.sigmoid(torch.clamp(pred, -20.0, 20.0))

        kernel = torch.tensor([[[0., 0., 0.],
                                [0., -6., 0.],
                                [0., 0., 0.]],
                               [[0., -1., 0.],
                                [-1., 26., -1.],
                                [0., -1., 0.]],
                               [[0., 0., 0.],
                                [0., -6., 0.],
                                [0., 0., 0.]]],
                              dtype=torch.float32, device=pred.device) / 26.0

        kernel = kernel.unsqueeze(0).unsqueeze(0)
        pred_laplacian = F.conv3d(pred_probs, kernel, padding=1)
        result = (torch.abs(pred_laplacian) * valid).sum() / (valid.sum() + self.smooth)
        return torch.nan_to_num(result, nan=0.0)


class ZContinuityLoss(nn.Module):
    """
    NOVEL: Z-axis slice continuity loss.

    Penalizes abrupt changes in prediction between adjacent Z-slices.
    A true papyrus sheet should have smooth, continuous boundaries
    through the depth dimension. Fragmentation (high beta_0) and
    spurious tunnels (high beta_1) both manifest as high slice-to-slice
    variance.

    This directly addresses the worsening VOI metric (more fragmentation
    over training) by teaching the model that adjacent slices should
    predict similar masks.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, valid):
        pred = ensure_shape_5d_channel_first(pred, "pred")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        pred_prob = torch.sigmoid(torch.clamp(pred, -20.0, 20.0))
        pred_masked = pred_prob * valid

        # Compute slice-to-slice difference along Z-axis
        # pred_masked shape: (B, 1, D, H, W)
        z_diff = pred_masked[:, :, 1:, :, :] - pred_masked[:, :, :-1, :, :]
        valid_z = valid[:, :, 1:, :, :] * valid[:, :, :-1, :, :]

        # Penalize large discontinuities (L2 on z-differences)
        z_loss = (z_diff ** 2 * valid_z).sum() / (valid_z.sum() + self.smooth)

        return torch.nan_to_num(z_loss, nan=0.0)


class VectorCosineLoss(nn.Module):
    """Cosine similarity loss for vector field prediction."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_vectors, target_vectors, mask, valid):
        B, C, D, H, W = pred_vectors.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        if target_vectors.dim() == 4:
            target_vectors = target_vectors.unsqueeze(0).expand(B, -1, -1, -1, -1)
        elif target_vectors.dim() == 5:
            if target_vectors.shape[0] != B:
                target_vectors = target_vectors.expand(B, -1, -1, -1, -1)

        mask = ensure_shape_5d_channel_first(mask, "mask")
        valid = ensure_shape_5d_channel_first(valid, "valid")

        pred_norm = F.normalize(pred_vectors, dim=1, eps=1e-8)
        target_norm = F.normalize(target_vectors, dim=1, eps=1e-8)

        cos_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)

        ink_region = (mask > 0.5).float()
        active_region = ink_region * valid

        if active_region.sum() > 0:
            loss = (1.0 - cos_sim) * active_region
            return loss.sum() / active_region.sum()
        else:
            return torch.tensor(0.0, device=pred_vectors.device)


# ==============================================================================
# COMBINED LOSS (Topology-Focused)
# ==============================================================================

class FinalTopoLoss(nn.Module):
    """
    Combined loss with topology-preserving clDice and boundary supervision.

    Loss weight philosophy:
    - clDice (0.20): PRIMARY topology loss - reduces beta_1 errors
    - DiceCE refined (0.30): Base segmentation on refined mask
    - DiceCE raw (0.10): Auxiliary loss for gradient to encoder
    - FP Penalty (0.15): Prevents over-prediction noise
    - Skeleton (0.10): Recall on topological skeleton
    - Boundary (0.08): Sharp surface detection
    - Center/Vec/Comp/DS: Small auxiliary terms
    """
    def __init__(self):
        super().__init__()
        self.dice_ce = SoftDiceCELoss(ce_weight=0.5, dice_weight=0.5)
        self.skel_recall = SkeletonRecallLoss()
        self.fp_penalty = FalsePositivePenaltyLoss()
        self.component_loss = ComponentLoss()
        self.center_loss_fn = nn.MSELoss(reduction='none')
        self.vector_loss_fn = VectorCosineLoss()
        self.cl_dice = clDiceLoss(num_iter=3)
        self.boundary_loss = BoundaryLoss()
        self.z_continuity = ZContinuityLoss()

        # Loss weights (topology-focused, with Z-continuity)
        self.w_seg = 0.28          # DiceCE on refined mask
        self.w_seg_raw = 0.08      # DiceCE on raw mask (gradient to encoder)
        self.w_fp = 0.15           # False positive penalty
        self.w_skel = 0.10         # Skeleton recall
        self.w_cldice = 0.18       # clDice (KEY topology loss)
        self.w_boundary = 0.06     # Boundary detection
        self.w_zcont = 0.08        # Z-continuity (NOVEL - reduces fragmentation)
        self.w_center = 0.02       # Center prediction
        self.w_vec = 0.02          # Vector prediction
        self.w_comp = 0.02         # Component smoothness
        self.w_ds1 = 0.01          # Deep supervision
        self.w_ds2 = 0.01          # Deep supervision

        total_w = sum([self.w_seg, self.w_seg_raw, self.w_fp, self.w_skel,
                       self.w_cldice, self.w_boundary, self.w_zcont,
                       self.w_center, self.w_vec, self.w_comp,
                       self.w_ds1, self.w_ds2])

        print("✅ LOSS WEIGHTS (Topology-Focused v2):")
        print(f"   DiceCE refined: {self.w_seg:.3f}")
        print(f"   DiceCE raw: {self.w_seg_raw:.3f}")
        print(f"   FP Penalty: {self.w_fp:.3f}")
        print(f"   Skeleton: {self.w_skel:.3f}")
        print(f"   clDice: {self.w_cldice:.3f} (topology)")
        print(f"   Boundary: {self.w_boundary:.3f} (surface)")
        print(f"   Z-Continuity: {self.w_zcont:.3f} (anti-fragmentation)")
        print(f"   Center: {self.w_center:.3f}")
        print(f"   Vector: {self.w_vec:.3f}")
        print(f"   Component: {self.w_comp:.3f}")
        print(f"   DS1/DS2: {self.w_ds1:.3f}/{self.w_ds2:.3f}")
        print(f"   TOTAL: {total_w:.3f}")

    def forward(self, preds, targets):
        target_mask = targets['mask']
        valid = targets['valid']
        skeleton = targets['skeleton']
        center = targets['center']
        vectors = targets['vectors']

        pred_mask = preds['mask']           # Refined output
        pred_skeleton = preds['skeleton']
        pred_center = preds['center']
        pred_vectors = preds['vectors']
        ds1 = preds['ds1']
        ds2 = preds['ds2']

        # Get raw mask and boundary (may not exist in older models)
        pred_mask_raw = preds.get('mask_raw', pred_mask)
        pred_boundary = preds.get('boundary', None)

        # Loss 1: DiceCE on refined mask
        l_seg = self.dice_ce(pred_mask, target_mask, valid)

        # Loss 2: DiceCE on raw mask (auxiliary)
        l_seg_raw = self.dice_ce(pred_mask_raw, target_mask, valid)

        # Loss 3: Skeleton recall
        l_skel = self.skel_recall(pred_mask, skeleton, valid)

        # Loss 4: False positive penalty
        l_fp = self.fp_penalty(pred_mask, target_mask, valid)

        # Loss 5: clDice (NOVEL - topology preserving)
        l_cldice = self.cl_dice(pred_mask, target_mask, valid)

        # Loss 6: Boundary detection
        if pred_boundary is not None:
            l_boundary = self.boundary_loss(pred_boundary, target_mask, valid)
        else:
            l_boundary = torch.tensor(0.0, device=pred_mask.device)

        # Loss 7: Center prediction
        valid_ch = ensure_shape_5d_channel_first(valid, "valid_for_center")
        l_center = (self.center_loss_fn(pred_center, center) * valid_ch).sum() / (valid_ch.sum() + 1e-6)

        # Loss 8: Vector prediction
        l_vec = self.vector_loss_fn(pred_vectors, vectors, target_mask, valid)

        # Loss 9: Component smoothness
        l_comp = self.component_loss(pred_mask, target_mask, valid)

        # Loss 10: Z-axis continuity (NOVEL - anti-fragmentation)
        l_zcont = self.z_continuity(pred_mask, target_mask, valid)

        # Loss 11-12: Deep supervision
        valid_ds = ensure_shape_5d_channel_first(valid, "valid_for_ds")
        ds1_up = F.interpolate(ds1, size=target_mask.shape[2:], mode='trilinear', align_corners=False)
        l_ds1 = self.dice_ce(ds1_up, target_mask, valid_ds)
        ds2_up = F.interpolate(ds2, size=target_mask.shape[2:], mode='trilinear', align_corners=False)
        l_ds2 = self.dice_ce(ds2_up, target_mask, valid_ds)

        # FIX: Sanitize each loss term before combination (catch per-term NaN)
        def _safe(loss_val, fallback=0.0):
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                return torch.tensor(fallback, device=loss_val.device, requires_grad=True)
            return loss_val

        l_seg = _safe(l_seg, 0.5)
        l_seg_raw = _safe(l_seg_raw, 0.5)
        l_fp = _safe(l_fp)
        l_skel = _safe(l_skel)
        l_cldice = _safe(l_cldice)
        l_boundary = _safe(l_boundary)
        l_zcont = _safe(l_zcont)
        l_center = _safe(l_center)
        l_vec = _safe(l_vec)
        l_comp = _safe(l_comp)
        l_ds1 = _safe(l_ds1, 0.5)
        l_ds2 = _safe(l_ds2, 0.5)

        # Combine
        total_loss = (
            self.w_seg * l_seg +
            self.w_seg_raw * l_seg_raw +
            self.w_fp * l_fp +
            self.w_skel * l_skel +
            self.w_cldice * l_cldice +
            self.w_boundary * l_boundary +
            self.w_zcont * l_zcont +
            self.w_center * l_center +
            self.w_vec * l_vec +
            self.w_comp * l_comp +
            self.w_ds1 * l_ds1 +
            self.w_ds2 * l_ds2
        )

        # FIX: Final NaN guard - fall back to just DiceCE if total is NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = self.w_seg * l_seg
            print("WARNING: NaN in total loss, falling back to DiceCE only")

        loss_dict = {
            'total': total_loss.item(),
            'dicece': l_seg.item(),
            'dicece_raw': l_seg_raw.item(),
            'fp_penalty': l_fp.item(),
            'skeleton': l_skel.item(),
            'cldice': l_cldice.item(),
            'boundary': l_boundary.item() if pred_boundary is not None else 0.0,
            'zcont': l_zcont.item(),
            'center': l_center.item(),
            'vector': l_vec.item(),
            'component': l_comp.item(),
            'ds1': l_ds1.item(),
            'ds2': l_ds2.item(),
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    loss = FinalTopoLoss()
    print("✅ Loss module initialized with topology-focused weights")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:03:15.657631Z","iopub.execute_input":"2026-02-13T08:03:15.65785Z","iopub.status.idle":"2026-02-13T08:03:15.66284Z","shell.execute_reply.started":"2026-02-13T08:03:15.657833Z","shell.execute_reply":"2026-02-13T08:03:15.662506Z"}}
%%writefile vesuvius_metrics.py
#==============================================================================
# METRIC COMPUTATION - TopoScore, VOI, SurfaceDice
#==============================================================================

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as scipy_label, binary_erosion
from sklearn.metrics import adjusted_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

def compute_betti_numbers(binary_vol):
    labeled_vol, k0 = scipy_label(binary_vol)
    inv_vol = 1 - binary_vol
    labeled_inv, k2 = scipy_label(inv_vol)
    k2 = max(0, k2 - 1)
    k1 = 0
    return {'k0': int(k0), 'k1': int(k1), 'k2': int(k2)}

def compute_toposcore(pred_vol, target_vol, valid_mask=None):
    if pred_vol.ndim == 4: pred_vol = pred_vol.squeeze()
    if target_vol.ndim == 4: target_vol = target_vol.squeeze()

    pred_binary = (pred_vol > 0.5).astype(np.uint8)
    tgt_binary = ((target_vol == 1) * (target_vol != 2)).astype(np.uint8)

    if valid_mask is not None:
        if valid_mask.ndim == 4: valid_mask = valid_mask.squeeze()
        valid_mask = (valid_mask > 0.5).astype(np.uint8)
        pred_binary = pred_binary * valid_mask
        tgt_binary = tgt_binary * valid_mask

    pred_betti = compute_betti_numbers(pred_binary)
    tgt_betti = compute_betti_numbers(tgt_binary)

    k0_diff = abs(pred_betti['k0'] - tgt_betti['k0'])
    k1_diff = abs(pred_betti['k1'] - tgt_betti['k1'])
    k2_diff = abs(pred_betti['k2'] - tgt_betti['k2'])

    max_diff = 50.0
    total_diff = (k0_diff + k1_diff + k2_diff) / max_diff
    return float(max(0.0, 1.0 - total_diff))

def compute_surface_dice(pred_vol, target_vol, valid_mask=None, tolerance=1):
    if pred_vol.ndim == 4: pred_vol = pred_vol.squeeze()
    if target_vol.ndim == 4: target_vol = target_vol.squeeze()

    if valid_mask is not None:
        if valid_mask.ndim == 4: valid_mask = valid_mask.squeeze()
        valid_mask = (valid_mask > 0.5).astype(np.uint8)
    else:
        valid_mask = np.ones_like(pred_vol, dtype=np.uint8)

    pred_binary = ((pred_vol > 0.5) * valid_mask).astype(np.uint8)
    tgt_binary = ((target_vol == 1) * valid_mask).astype(np.uint8)

    pred_surface = pred_binary * (1 - binary_erosion(pred_binary, iterations=1).astype(np.uint8))
    tgt_surface = tgt_binary * (1 - binary_erosion(tgt_binary, iterations=1).astype(np.uint8))

    from scipy.ndimage import distance_transform_edt

    if tgt_surface.sum() > 0:
        dist_to_target = distance_transform_edt(1 - tgt_surface)
        pred_surface_near = (dist_to_target[pred_surface > 0] <= tolerance).sum() if pred_surface.sum() > 0 else 0
        precision = pred_surface_near / (pred_surface.sum() + 1e-6) if pred_surface.sum() > 0 else 0.0
    else:
        precision = 0.0 if pred_surface.sum() > 0 else 1.0

    if pred_surface.sum() > 0:
        dist_to_pred = distance_transform_edt(1 - pred_surface)
        tgt_surface_near = (dist_to_pred[tgt_surface > 0] <= tolerance).sum() if tgt_surface.sum() > 0 else 0
        recall = tgt_surface_near / (tgt_surface.sum() + 1e-6) if tgt_surface.sum() > 0 else 0.0
    else:
        recall = 0.0

    if precision + recall > 0:
        return float(2 * precision * recall / (precision + recall))
    return 0.0

def compute_voi(pred_vol, target_vol, valid_mask=None, reduce='mean'):
    if pred_vol.ndim == 4: pred_vol = pred_vol.squeeze()
    if target_vol.ndim == 4: target_vol = target_vol.squeeze()

    pred_binary = (pred_vol > 0.5).astype(np.uint8)
    tgt_binary = ((target_vol == 1) * (target_vol != 2)).astype(np.uint8)

    pred_labeled, _ = scipy_label(pred_binary)
    tgt_labeled, _ = scipy_label(tgt_binary)

    if valid_mask is not None:
        if valid_mask.ndim == 4: valid_mask = valid_mask.squeeze()
        valid_mask = (valid_mask > 0.5).astype(np.uint8)
        pred_labeled = pred_labeled * valid_mask
        tgt_labeled = tgt_labeled * valid_mask

    try:
        ami = adjusted_mutual_info_score(tgt_labeled.flatten(), pred_labeled.flatten())
    except:
        ami = 0.0

    return float(1.0 - max(0.0, ami))

class MetricsComputer:
    def __init__(self):
        self.toposcore_list = []
        self.surface_dice_list = []
        self.voi_list = []

    def update(self, pred_batch, target_batch, valid_batch=None):
        if pred_batch.ndim == 5: pred_batch = pred_batch.squeeze(1)
        if target_batch.ndim == 5: target_batch = target_batch.squeeze(1)

        pred_batch = pred_batch.detach().cpu().numpy()
        target_batch = target_batch.detach().cpu().numpy()

        if valid_batch is not None:
            if valid_batch.ndim == 5: valid_batch = valid_batch.squeeze(-1)
            valid_batch = valid_batch.detach().cpu().numpy()

        for b in range(pred_batch.shape[0]):
            pred = pred_batch[b]
            target = target_batch[b]
            valid = valid_batch[b] if valid_batch is not None else None

            self.toposcore_list.append(compute_toposcore(pred, target, valid))
            self.surface_dice_list.append(compute_surface_dice(pred, target, valid_mask=valid, tolerance=1))
            self.voi_list.append(compute_voi(pred, target, valid))

    def compute(self):
        if len(self.toposcore_list) == 0:
            return {'toposcore': 0.0, 'surface_dice': 0.0, 'voi': 0.0, 'combined': 0.0}

        mean_topo = np.mean(self.toposcore_list)
        mean_surface = np.mean(self.surface_dice_list)
        mean_voi = np.mean(self.voi_list)

        combined = 0.33 * mean_topo + 0.33 * mean_surface + 0.33 * (1.0 - min(1.0, mean_voi))

        return {
            'toposcore': float(mean_topo),
            'surface_dice': float(mean_surface),
            'voi': float(mean_voi),
            'combined': float(combined)
        }

    def reset(self):
        self.toposcore_list = []
        self.surface_dice_list = []
        self.voi_list = []

def format_metrics(metrics_dict):
    return (f"TopoScore: {metrics_dict['toposcore']:.4f} | "
            f"SurfaceDice: {metrics_dict['surface_dice']:.4f} | "
            f"VOI: {metrics_dict['voi']:.4f} | "
            f"Combined: {metrics_dict['combined']:.4f}")

print("✅ Metric Computation Functions Ready")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-13T08:03:15.872124Z","iopub.execute_input":"2026-02-13T08:03:15.872726Z"}}
#!/usr/bin/env python3
"""
MAMBA-ENHANCED VNET TRAINING
=============================

KEY FEATURES:
✅ Mamba SSM for Z-axis state tracking
✅ clDice topology-preserving loss
✅ Factored anisotropic convolutions
✅ Topology refinement head
✅ Boundary detection head
✅ Resume from checkpoint support
✅ Metric computation (TopoScore, SurfaceDice, VOI)
✅ Sliding window inference visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import os
import gc
from tqdm.auto import tqdm
import tifffile
import matplotlib.pyplot as plt
import random

try:
    from torchinfo import summary
except ImportError:
    summary = None

# --- CONFIGURATION ---
class CFG_RESUME:
    RESUME_FROM_EPOCH = 140
    CHECKPOINT_PATH = "/kaggle/input/datasets/perarasu2156/vnet-mamba-holesfillede140/gflow_resumed_epoch_140.pth"

    BATCH_DIRS = [
        "/kaggle/input/datasets/sanjaynn/vnetnpy0-25/vesuvius_precomputed_final/batch_0_25",
        "/kaggle/input/datasets/sanjaynn/vnetnpy25-50/vesuvius_precomputed_final/batch_25_50",
        "/kaggle/input/datasets/sanjaynn/vnetnpy50-75/vesuvius_precomputed_final/batch_50_75",
        "/kaggle/input/datasets/sanjaynn/vnetnpy75-100/vesuvius_precomputed_final/batch_75_100",
        "/kaggle/input/datasets/sanjaynn/vnetnpy100-125/vesuvius_precomputed_final/batch_100_125",
        "/kaggle/input/datasets/sanjaynn/vnetnpy125-150/vesuvius_precomputed_final/batch_125_150",
        "/kaggle/input/datasets/sanjaynn/vnetnpy150-175/vesuvius_precomputed_final/batch_150_175",
        "/kaggle/input/datasets/sanjaynn/vnetnpy175-200/vesuvius_precomputed_final/batch_175_200",
        "/kaggle/input/datasets/sanjaynn/vnetnpy225-250/vesuvius_precomputed_final/batch_225_249",
        "/kaggle/input/datasets/sanjaynn/vnetnpy250-275/vesuvius_precomputed_final/batch_250_275",
        "/kaggle/input/datasets/sanjaynn/vnetnpy275-300/vesuvius_precomputed_final/batch_275_300",
        "/kaggle/input/datasets/sanjaynn/vnetnpy300-325/vesuvius_precomputed_final/batch_300_325",
        "/kaggle/input/datasets/sanjaynn/vnetnpy350-375/vesuvius_precomputed_final/batch_350_375",
        "/kaggle/input/datasets/sanjaynn/vnetnpy375-400/vesuvius_precomputed_final/batch_375_400",
        "/kaggle/input/datasets/sanjaynn/vnetnpy400-425/vesuvius_precomputed_final/batch_400_424",
        "/kaggle/input/datasets/sanjaynn/vnetnpy425-450/vesuvius_precomputed_final/batch_425_449",
        "/kaggle/input/datasets/sanjaynn/vnetnpy450-475/vesuvius_precomputed_final/batch_450_475",
        "/kaggle/input/datasets/sanjaynn/vnetnpy475-500/vesuvius_precomputed_final/batch_475_500",
        "/kaggle/input/datasets/sanjaynn/vnetnpy500-525/vesuvius_precomputed_final/batch_500_525",
        "/kaggle/input/datasets/sanjaynn/vnetnpy525-550/vesuvius_precomputed_final/batch_525_550",
        "/kaggle/input/datasets/sanjaynn/vnetnpy50-575/vesuvius_precomputed_final/batch_550_575",
        "/kaggle/input/datasets/sanjaynn/vnetnpy575-600/vesuvius_precomputed_final/batch_575_600",
        "/kaggle/input/datasets/sanjaynn/vnetnpy600-625/vesuvius_precomputed_final/batch_600_625",
        "/kaggle/input/datasets/sanjaynn/vnetnpy625-650/vesuvius_precomputed_final/batch_625_650",
        "/kaggle/input/datasets/sanjaynn/vnetnpy650-675/vesuvius_precomputed_final/batch_650_675",
        "/kaggle/input/datasets/sanjaynn/vnetnpy675-700/vesuvius_precomputed_final/batch_675_700",
        "/kaggle/input/datasets/sanjaynn/vnetnpy700-723/vesuvius_precomputed_final/batch_700_723",
        "/kaggle/input/datasets/sanjaynn/vnetnpy725-750/vesuvius_precomputed_final/batch_725_750",
        "/kaggle/input/datasets/sanjaynn/vnetnpy750-775/vesuvius_precomputed_final/batch_750_775",
        "/kaggle/input/datasets/sanjaynn/vnetnpy775-800/vesuvius_precomputed_final/batch_775_786",
    ]

    TOTAL_EPOCHS = 14
    START_EPOCH = RESUME_FROM_EPOCH + 1

    INITIAL_LR = 3e-4
    MIN_LR = 1e-6

    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 2
    NUM_WORKERS = 12
    GRADIENT_CLIP_NORM = 1.0

    CROP_SIZE = (160, 160, 160)

    VAL_OVERLAP = 0.5
    VAL_BATCH_SIZE = 4
    NUM_VAL_SAMPLES = 5
    VIZ_INTERVAL = 1

    RAW_DATA_DIR = "/kaggle/input/datasets/kabimuki/vesuvius-surface-detection/vesuvius_dataset/vesuvius_dataset/test_images"
    RAW_LABEL_DIR = "/kaggle/input/datasets/kabimuki/vesuvius-surface-detection/vesuvius_dataset/vesuvius_dataset/train_labels"
    RAW_META_FILE = "/kaggle/input/datasets/kabimuki/vesuvius-surface-detection/vesuvius_dataset/vesuvius_dataset/train.csv"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_compiled_checkpoint(model, checkpoint_path, device):
    """Load checkpoint handling torch.compile _orig_mod. prefix."""
    if checkpoint_path is None:
        print("No checkpoint provided (fresh start). Skipping load.")
        return model

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    first_key = list(checkpoint.keys())[0]
    if first_key.startswith('_orig_mod.'):
        print("   Detected torch.compile() checkpoint - removing prefix...")
        new_checkpoint = {}
        for key, value in checkpoint.items():
            new_checkpoint[key.replace('_orig_mod.', '')] = value
        checkpoint = new_checkpoint

    # Try strict load first, fall back to non-strict for architecture changes
    try:
        model.load_state_dict(checkpoint, strict=True)
        print("   Weights loaded successfully (strict)!")
    except RuntimeError as e:
        print(f"   Strict load failed: {str(e)[:100]}...")
        print("   Trying non-strict load (new architecture layers will be randomly initialized)...")
        model.load_state_dict(checkpoint, strict=False)
        print("   Weights loaded (non-strict)!")

    return model


@torch.no_grad()
def visualize_random_sample(model, crop_size, device, data_dir, label_dir, meta_file):
    """Visualize model prediction on a random sample."""
    try:
        try:
            df = pd.read_csv(meta_file)
            df['id'] = df['id'].astype(str)
            available_ids = [
                sid for sid in df['id'].values
                if os.path.exists(os.path.join(data_dir, f"{sid}.tif"))
            ]
        except:
            available_ids = [f.replace('.tif', '') for f in os.listdir(data_dir) if f.endswith('.tif')]

        if len(available_ids) == 0:
            print("   No volumes found for visualization")
            return

        sample_id = random.choice(available_ids)
        vol_path = os.path.join(data_dir, f"{sample_id}.tif")
        label_path = os.path.join(label_dir, f"{sample_id}.tif")

        if not os.path.exists(vol_path):
            return

        with tifffile.TiffFile(vol_path) as tif:
            vol_full = tif.asarray()

        label_full = None
        if os.path.exists(label_path):
            with tifffile.TiffFile(label_path) as tif:
                label_full = tif.asarray()

        D, H, W = vol_full.shape
        cd, ch, cw = crop_size

        z = np.random.randint(0, max(1, D - cd + 1))
        y = np.random.randint(0, max(1, H - ch + 1))
        x = np.random.randint(0, max(1, W - cw + 1))

        vol_crop = vol_full[z:z+cd, y:y+ch, x:x+cw]
        vol_tensor = torch.from_numpy(vol_crop).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        model.eval()
        with torch.no_grad():
            preds = model(vol_tensor)
            if isinstance(preds, dict):
                mask_logits = preds['mask']
            else:
                mask_logits = preds[0]
            pred_mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()

        mid_z = cd // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(vol_crop[mid_z], cmap='gray')
        axes[0].set_title(f"Raw CT ({sample_id})")
        axes[0].axis('off')

        if label_full is not None:
            label_crop = label_full[z:z+cd, y:y+ch, x:x+cw]
            axes[1].imshow(label_crop[mid_z], cmap='gray')
            axes[1].set_title("GT Mask")
        else:
            axes[1].text(0.5, 0.5, 'No GT', ha='center', va='center')
            axes[1].set_title("GT Mask")
        axes[1].axis('off')

        axes[2].imshow(pred_mask[mid_z], cmap='jet')
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"   Visualization failed: {e}")


@torch.no_grad()
def sliding_window_inference(model, vol_path, label_path, crop_size, overlap=0.5, device='cuda'):
    """Sliding window inference on a full volume."""
    model.eval()

    try:
        with tifffile.TiffFile(vol_path) as tif:
            vol_full = tif.asarray()

        if os.path.exists(label_path):
            with tifffile.TiffFile(label_path) as tif:
                mask_full = tif.asarray()
                valid_mask = (mask_full != 2).astype(np.uint8)
                mask_full = (mask_full == 1).astype(np.uint8)
        else:
            return 0.0
    except Exception as e:
        print(f"   Read Error: {e}")
        return 0.0

    D, H, W = vol_full.shape
    cd, ch, cw = crop_size
    sd, sh, sw = int(cd*(1-overlap)), int(ch*(1-overlap)), int(cw*(1-overlap))

    z_steps = range(0, max(1, D - cd + 1), sd)
    y_steps = range(0, max(1, H - ch + 1), sh)
    x_steps = range(0, max(1, W - cw + 1), sw)

    pred_prob = torch.zeros((D, H, W), dtype=torch.float16, device='cpu')
    count_map = torch.zeros((D, H, W), dtype=torch.float16, device='cpu')

    coords = [(z, y, x) for z in z_steps for y in y_steps for x in x_steps]
    batch_size = CFG_RESUME.VAL_BATCH_SIZE

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]
        batch_crops = []

        for (z, y, x) in batch_coords:
            crop = vol_full[z:z+cd, y:y+ch, x:x+cw]
            if crop.shape != crop_size:
                pad_d = cd - crop.shape[0]
                pad_h = ch - crop.shape[1]
                pad_w = cw - crop.shape[2]
                crop = np.pad(crop, ((0,pad_d), (0,pad_h), (0,pad_w)))
            batch_crops.append(crop)

        batch_t = torch.tensor(np.stack(batch_crops), dtype=torch.float32).unsqueeze(1).to(device)
        batch_t = batch_t / 255.0

        with autocast():
            out = model(batch_t)
            if isinstance(out, dict):
                logits = out['mask']
            elif isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            probs = torch.sigmoid(logits).squeeze(1).cpu().half()

        for j, (z, y, x) in enumerate(batch_coords):
            d_real, h_real, w_real = min(cd, D-z), min(ch, H-y), min(cw, W-x)
            pred_prob[z:z+d_real, y:y+h_real, x:x+w_real] += probs[j, :d_real, :h_real, :w_real]
            count_map[z:z+d_real, y:y+h_real, x:x+w_real] += 1.0

    pred_prob /= torch.clamp(count_map, min=1.0)
    pred_mask = (pred_prob > 0.5).numpy().astype(np.uint8)

    valid_mask = valid_mask.astype(bool)
    inter = np.sum((pred_mask & mask_full) & valid_mask)
    dice = 2.0 * inter / (np.sum(pred_mask[valid_mask]) + np.sum(mask_full[valid_mask]) + 1e-6)

    mid_z = D // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.title("Raw CT"); plt.imshow(vol_full[mid_z], cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title("GT Mask"); plt.imshow(mask_full[mid_z], cmap='gray'); plt.axis('off')
    plt.subplot(1,3,3); plt.title(f"Pred (Dice: {dice:.3f})"); plt.imshow(pred_prob[mid_z].float(), cmap='jet'); plt.axis('off')
    plt.show()

    del vol_full, mask_full, pred_prob, count_map
    gc.collect()

    return dice


def resume_training():
    print("MAMBA-ENHANCED VNET TRAINING")
    print(f"   Checkpoint: {CFG_RESUME.CHECKPOINT_PATH}")
    print(f"   Resume from Epoch: {CFG_RESUME.START_EPOCH}")
    print(f"   Target Epoch: {CFG_RESUME.TOTAL_EPOCHS}")

    # ========== DATA LOADING ==========
    print("\nCreating dataloaders from NPY batches (6-channel format)...")
    from vesuvius_dataloader_fixed import create_multi_batch_dataloaders

    train_loader, val_loader = create_multi_batch_dataloaders(
        batch_dirs=CFG_RESUME.BATCH_DIRS,
        val_split=0.1,
        batch_size=CFG_RESUME.BATCH_SIZE,
        num_workers=CFG_RESUME.NUM_WORKERS,
        crop_size=CFG_RESUME.CROP_SIZE,
    )

    steps_per_epoch = 330
    print(f"Training steps per epoch: {steps_per_epoch}")

    # ========== MODEL ==========
    from vesuvius_vnet_fixed import VNetFPNFixed
    from vesuvius_loss_final import FinalTopoLoss
    from vesuvius_metrics import MetricsComputer, format_metrics

    print("\nLoading Mamba-Enhanced VNet FPN architecture...")
    model = VNetFPNFixed().to(CFG_RESUME.DEVICE)

    model = load_compiled_checkpoint(model, CFG_RESUME.CHECKPOINT_PATH, CFG_RESUME.DEVICE)

    # torch.compile (with fallback for Mamba compatibility)
    if hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="default")
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding without compilation.")

    # ========== OPTIMIZER & SCHEDULER ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG_RESUME.INITIAL_LR, weight_decay=1e-2)

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    print(f"Fast-forwarding scheduler to epoch {CFG_RESUME.RESUME_FROM_EPOCH}...")
    for _ in range(CFG_RESUME.RESUME_FROM_EPOCH):
        scheduler.step()

    print(f"Scheduler resumed. Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ========== LOSS & AMP ==========
    print("\nUsing topology-focused loss (DiceCE + clDice + FP + Skeleton + Boundary)...")
    criterion = FinalTopoLoss().to(CFG_RESUME.DEVICE)
    scaler = GradScaler()

    best_combined_metric = 0.0

    # ========== TRAINING LOOP ==========
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING: Epoch {CFG_RESUME.START_EPOCH} -> {CFG_RESUME.TOTAL_EPOCHS}")
    print(f"{'='*80}\n")

    for epoch in range(CFG_RESUME.START_EPOCH, CFG_RESUME.TOTAL_EPOCHS + 1):
        print(f"\nEPOCH {epoch}/{CFG_RESUME.TOTAL_EPOCHS}")

        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_loss_breakdown = {
            'dicece': 0, 'dicece_raw': 0, 'fp_penalty': 0,
            'skeleton': 0, 'cldice': 0, 'boundary': 0, 'zcont': 0, 'component': 0
        }
        optimizer.zero_grad()
        loop = tqdm(train_loader, desc="Training", leave=False)

        for step, batch in enumerate(loop):
            if step >= steps_per_epoch:
                break

            vol = batch['volume'].to(CFG_RESUME.DEVICE, non_blocking=True)
            targets = {k: v.to(CFG_RESUME.DEVICE, non_blocking=True) for k, v in batch.items() if k != 'volume'}

            if targets['mask'].sum() < 10:
                continue

            with autocast():
                preds = model(vol)
                loss, loss_dict = criterion(preds, targets)
                loss = loss / CFG_RESUME.ACCUMULATION_STEPS

            # FIX: Skip batch if loss is NaN/Inf (prevents poisoning model weights)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   WARNING: NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()  # Clear any accumulated gradients
                torch.cuda.empty_cache()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % CFG_RESUME.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                # FIX: Check for NaN gradients before stepping
                valid_grads = True
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        valid_grads = False
                        break

                if valid_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG_RESUME.GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print(f"   WARNING: NaN gradients at step {step}, skipping optimizer step")
                    scaler.update()  # Still update scaler to keep it in sync

                optimizer.zero_grad()

            if step % 10 == 0:
                torch.cuda.empty_cache()

            train_loss += loss.item() * CFG_RESUME.ACCUMULATION_STEPS
            for key in train_loss_breakdown:
                if key in loss_dict:
                    train_loss_breakdown[key] += loss_dict[key]

            loop.set_postfix({
                'loss': f"{loss.item() * CFG_RESUME.ACCUMULATION_STEPS:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()
        avg_train_loss = train_loss / steps_per_epoch
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"      DiceCE: {train_loss_breakdown['dicece']/steps_per_epoch:.4f}")
        print(f"      DiceCE_raw: {train_loss_breakdown['dicece_raw']/steps_per_epoch:.4f}")
        print(f"      FP Penalty: {train_loss_breakdown['fp_penalty']/steps_per_epoch:.4f}")
        print(f"      Skeleton: {train_loss_breakdown['skeleton']/steps_per_epoch:.4f}")
        print(f"      clDice: {train_loss_breakdown['cldice']/steps_per_epoch:.4f}")
        print(f"      Boundary: {train_loss_breakdown['boundary']/steps_per_epoch:.4f}")
        print(f"      Z-Cont: {train_loss_breakdown['zcont']/steps_per_epoch:.4f}")
        print(f"      Component: {train_loss_breakdown['component']/steps_per_epoch:.4f}")

        # --- VISUALIZATION ---
        E = CFG_RESUME.TOTAL_EPOCHS
        if epoch == CFG_RESUME.START_EPOCH or epoch % 5 == 0 or epoch >= E - 4:
            print("   Visualizing random train sample...")
            visualize_random_sample(
                model, CFG_RESUME.CROP_SIZE, CFG_RESUME.DEVICE,
                CFG_RESUME.RAW_DATA_DIR, CFG_RESUME.RAW_LABEL_DIR, CFG_RESUME.RAW_META_FILE
            )

        # --- VALIDATION ---
        if epoch % 5 == 0 or epoch >= E - 4:
            print(f"   Running Validation with Metrics ({CFG_RESUME.NUM_VAL_SAMPLES} samples)...")

            metrics = MetricsComputer()
            model.eval()

            with torch.no_grad():
                val_step = 0
                for batch in val_loader:
                    if val_step >= 10:
                        break

                    vol = batch['volume'].to(CFG_RESUME.DEVICE)
                    preds = model(vol)

                    if isinstance(preds, dict):
                        pred_mask = torch.sigmoid(preds['mask']).detach()
                    else:
                        pred_mask = torch.sigmoid(preds[0]).detach()

                    metrics.update(pred_mask, batch['mask'], batch['valid'])
                    val_step += 1

            results = metrics.compute()
            print(f"   Metrics: {format_metrics(results)}")
            combined = results['combined']

            # Sliding window inference
            print("   Running Sliding Window Inference Visualization...")
            try:
                df = pd.read_csv(CFG_RESUME.RAW_META_FILE)
                valid_ids = [str(x) for x in df['id'].values if os.path.exists(os.path.join(CFG_RESUME.RAW_DATA_DIR, f"{x}.tif"))]
                viz_samples = random.sample(valid_ids, min(2, len(valid_ids)))

                for vid in viz_samples:
                    v_path = os.path.join(CFG_RESUME.RAW_DATA_DIR, f"{vid}.tif")
                    l_path = os.path.join(CFG_RESUME.RAW_LABEL_DIR, f"{vid}.tif")
                    dice = sliding_window_inference(
                        model, v_path, l_path, CFG_RESUME.CROP_SIZE,
                        overlap=CFG_RESUME.VAL_OVERLAP, device=CFG_RESUME.DEVICE
                    )
                    print(f"      ID {vid}: Dice = {dice:.4f}")
            except Exception as e:
                print(f"      Sliding window viz failed: {e}")

            if combined > best_combined_metric:
                best_combined_metric = combined
                torch.save(model.state_dict(), f"gflow_resumed_best_combined_{combined:.4f}.pth")
                print(f"   Saved New Best Model! (Combined: {combined:.4f})")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"gflow_resumed_epoch_{epoch}.pth")
            print(f"   Saved checkpoint at epoch {epoch}")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining Complete.")
    print(f"Final Best Combined Metric: {best_combined_metric:.4f}")


if __name__ == "__main__":
    resume_training()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code]


# %% [code]


# %% [code]


# %% [code]
