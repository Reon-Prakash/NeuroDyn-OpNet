"""
Latent State Encoder (LSE)
--------------------------
NeuroDyn-OpNet Component

Input  : (Batch, Channels, Time_Steps, 1)
Output : (Batch, latent_dim)

Design:
- Channel-wise initial projection
- Multi-scale temporal residual blocks (dilated convs)
- Global temporal pooling
- Fully connected latent bottleneck

Author: NeuroDyn-OpNet Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# Residual Temporal Block with Dilation
# ────────────────────────────────────────────────────────────────

class TemporalResidualBlock(nn.Module):
    """
    1D Temporal Residual Block with dilation.
    Operates on shape: (B, C, T)
    """

    def __init__(self, channels, dilation=1, kernel_size=3):
        super().__init__()

        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ELU(inplace=True)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.act(out)

        return out


# ────────────────────────────────────────────────────────────────
# Latent State Encoder
# ────────────────────────────────────────────────────────────────

class LatentStateEncoder(nn.Module):
    """
    Latent State Encoder (LSE)

    Parameters
    ----------
    in_channels : int
        Number of EEG channels (17 for SEED-VIG, 30 for SADT)

    latent_dim : int
        Dimension of compressed latent vector z

    base_filters : int
        Width of temporal feature extractor
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 128,
        base_filters: int = 64
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # -------------------------------------------------------
        # 1) Initial Channel Projection
        # Treat EEG channels as input features
        # Converts C → base_filters
        # -------------------------------------------------------
        self.input_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=3,
            padding=1
        )

        self.input_bn = nn.BatchNorm1d(base_filters)
        self.act = nn.ELU(inplace=True)

        # -------------------------------------------------------
        # 2) Multi-Scale Temporal Feature Extractor
        # Dilations capture long-range dependencies over 384 T
        # -------------------------------------------------------
        self.res_block1 = TemporalResidualBlock(
            base_filters, dilation=1
        )
        self.res_block2 = TemporalResidualBlock(
            base_filters, dilation=2
        )
        self.res_block3 = TemporalResidualBlock(
            base_filters, dilation=4
        )
        self.res_block4 = TemporalResidualBlock(
            base_filters, dilation=8
        )

        # -------------------------------------------------------
        # 3) Global Temporal Aggregation
        # Compress time dimension
        # -------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # -------------------------------------------------------
        # 4) Latent Bottleneck
        # -------------------------------------------------------
        self.fc = nn.Linear(base_filters, latent_dim)

    # ────────────────────────────────────────────────────────────
    # Forward Pass
    # ────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        x shape: (B, C, T, 1)
        returns: (B, latent_dim)
        """

        # Remove trailing singleton dimension
        x = x.squeeze(-1)  # -> (B, C, T)

        # Initial projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.act(x)

        # Temporal residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Global temporal pooling
        x = self.global_pool(x)  # -> (B, base_filters, 1)

        x = x.squeeze(-1)  # -> (B, base_filters)

        # Latent compression
        z = self.fc(x)  # -> (B, latent_dim)

        return z


# ────────────────────────────────────────────────────────────────
# Dummy Test / Shape Verification
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BATCH_SIZE = 8
    TIME_STEPS = 384
    LATENT_DIM = 128

    print("=" * 60)
    print("Testing LatentStateEncoder with dummy input")
    print("=" * 60)

    # Test for SEED-VIG (17 channels)
    seed_channels = 17
    model_seed = LatentStateEncoder(
        in_channels=seed_channels,
        latent_dim=LATENT_DIM
    )

    dummy_seed = torch.randn(BATCH_SIZE, seed_channels, TIME_STEPS, 1)
    z_seed = model_seed(dummy_seed)

    print(f"SEED Input  shape : {dummy_seed.shape}")
    print(f"SEED Output shape : {z_seed.shape}")

    # Test for SADT (30 channels)
    sadt_channels = 30
    model_sadt = LatentStateEncoder(
        in_channels=sadt_channels,
        latent_dim=LATENT_DIM
    )

    dummy_sadt = torch.randn(BATCH_SIZE, sadt_channels, TIME_STEPS, 1)
    z_sadt = model_sadt(dummy_sadt)

    print(f"\nSADT Input  shape : {dummy_sadt.shape}")
    print(f"SADT Output shape : {z_sadt.shape}")

    print("\nExpected Output Shape: (Batch, Latent_Dim)")
    print("=" * 60)