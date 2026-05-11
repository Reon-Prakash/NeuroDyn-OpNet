"""
neural_operator.py
------------------
Fourier Neural Operator (FNO) Dynamics Block
NeuroDyn-OpNet Component

Implements the latent state transition:
    z_{t+1} = sigma( W*z_t + F^{-1}( R * F(z_t) ) )

Input  : (Batch, Latent_Dim)   <- latent state z_t
Output : (Batch, Latent_Dim)   <- predicted next state z_{t+1}

Components:
    - Lifting Layer        : latent_dim -> fno_width
    - SpectralConv1d Layer : frequency-domain weight R
    - Skip Connection      : linear W * x in spatial domain
    - Projection Layer     : fno_width -> latent_dim
    - GELU Activation

Author: NeuroDyn-OpNet Architecture
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 0.  WINDOWS CONSOLE ENCODING FIX
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SPECTRAL CONVOLUTION LAYER  (Core FNO building block)
# ──────────────────────────────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer.

    Implements the operator:
        F^{-1}( R * F(x) )

    where:
        F   = rfft  (real-to-complex FFT)
        R   = complex-valued learnable weight matrix (modes x width x width)
        F^{-1} = irfft (inverse FFT back to real domain)

    Parameters
    ----------
    in_channels  : int   Width of input feature space
    out_channels : int   Width of output feature space
    n_modes      : int   Number of Fourier modes to retain (rest are truncated)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_modes:      int = 16
    ):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes

        # ------------------------------------------------------------------ #
        # Complex-valued weight tensor R                                      #
        # Shape: (n_modes, in_channels, out_channels)                         #
        # Initialized as complex64 (real + imaginary parts learned jointly)   #
        # ------------------------------------------------------------------ #
        self.weights = nn.Parameter(
            torch.view_as_real(
                (1.0 / (in_channels * out_channels)) *
                torch.randn(
                    n_modes,
                    in_channels,
                    out_channels,
                    dtype=torch.complex64
                )
            )
        )   # stored as (..., 2) real tensor for nn.Parameter compatibility

    # ---------------------------------------------------------------------- #
    # Complex multiplication helper                                           #
    # x_hat : (B, n_modes, in_channels)   complex                            #
    # W     : (n_modes, in_channels, out_channels)  complex                  #
    # out   : (B, n_modes, out_channels)  complex                            #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def complex_mul(x_hat: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Batched complex matrix multiplication:
            out[b, m, o] = sum_i  x_hat[b, m, i] * W[m, i, o]
        """
        return torch.einsum("bmi, mio -> bmo", x_hat, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C)   — sequence in spatial / temporal domain

        Returns
        -------
        out : (B, T, C) — filtered sequence in same domain
        """
        B, T, C = x.shape

        # Step 1: FFT along time dimension
        x_ft = torch.fft.rfft(x, dim=1)          # (B, T//2+1, C)  complex64

        # Recover complex weights from view_as_real storage
        W_complex = torch.view_as_complex(self.weights)  # (n_modes, Cin, Cout)

        # Step 2: Truncate to n_modes and apply complex weight R
        n_freq = x_ft.shape[1]                    # T//2 + 1
        n_keep = min(self.n_modes, n_freq)

        out_ft = torch.zeros(
            B, n_freq, self.out_channels,
            dtype=torch.complex64,
            device=x.device
        )

        # Apply R to the kept low-frequency modes only
        out_ft[:, :n_keep, :] = self.complex_mul(
            x_ft[:, :n_keep, :],                  # (B, n_keep, Cin)
            W_complex[:n_keep, :, :]               # (n_keep, Cin, Cout)
        )

        # Step 3: Inverse FFT back to temporal domain
        out = torch.fft.irfft(out_ft, n=T, dim=1) # (B, T, Cout)

        return out


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SINGLE FNO LAYER  (Spectral path + Skip-connection path)
# ──────────────────────────────────────────────────────────────────────────────

class FNOLayer(nn.Module):
    """
    Single FNO update layer.

    Implements:
        h = sigma( F^{-1}(R * F(x)) + W*x )

    Both paths operate in (B, T, width) space.

    Parameters
    ----------
    width   : int   Hidden channel width
    n_modes : int   Fourier modes to keep
    """

    def __init__(self, width: int, n_modes: int = 16):
        super().__init__()

        # Spectral path
        self.spectral = SpectralConv1d(
            in_channels  = width,
            out_channels = width,
            n_modes      = n_modes
        )

        # Skip / residual path  W * x  (pointwise linear across channels)
        self.skip = nn.Linear(width, width, bias=False)

        # Normalization + Activation
        self.norm = nn.LayerNorm(width)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T, width)
        out : (B, T, width)
        """
        spectral_out = self.spectral(x)              # F^{-1}(R * F(x))
        skip_out     = self.skip(x)                  # W * x

        return self.act(self.norm(spectral_out + skip_out))


# ──────────────────────────────────────────────────────────────────────────────
# 3.  FNO DYNAMICS BLOCK  (Full operator: z_t -> z_{t+1})
# ──────────────────────────────────────────────────────────────────────────────

class FNODynamicsBlock(nn.Module):
    """
    Fourier Neural Operator Dynamics Block.

    Models latent state transition:
        z_{t+1} = Operator( z_t )

    Full equation per FNO layer:
        z_{t+1} = sigma( W*z_t + F^{-1}( R * F(z_t) ) )

    Architecture:
        z_t (B, latent_dim)
            |
            | Lift: Linear(latent_dim -> fno_width)
            | Unsqueeze T=1 -> (B, 1, fno_width)
            v
        FNO Layer 1  (B, 1, fno_width)
        FNO Layer 2  (B, 1, fno_width)
        FNO Layer 3  (B, 1, fno_width)
            |
            | Squeeze T -> (B, fno_width)
            | Project: Linear(fno_width -> latent_dim)
            v
        z_{t+1} (B, latent_dim)

    Parameters
    ----------
    latent_dim  : int   Latent state dimension (must match LSE output)
    fno_width   : int   Internal hidden width of FNO layers
    n_modes     : int   Fourier modes to retain in spectral conv
    n_layers    : int   Number of stacked FNO layers
    """

    def __init__(
        self,
        latent_dim: int = 128,
        fno_width:  int = 256,
        n_modes:    int = 16,
        n_layers:   int = 3
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.fno_width  = fno_width
        self.n_modes    = n_modes
        self.n_layers   = n_layers

        # ------------------------------------------------------------------ #
        # Lifting: latent_dim -> fno_width                                    #
        # ------------------------------------------------------------------ #
        self.lift = nn.Sequential(
            nn.Linear(latent_dim, fno_width),
            nn.GELU()
        )

        # ------------------------------------------------------------------ #
        # FNO Layers Stack                                                    #
        # ------------------------------------------------------------------ #
        self.fno_layers = nn.ModuleList([
            FNOLayer(width=fno_width, n_modes=n_modes)
            for _ in range(n_layers)
        ])

        # ------------------------------------------------------------------ #
        # Projection: fno_width -> latent_dim                                 #
        # ------------------------------------------------------------------ #
        self.project = nn.Sequential(
            nn.Linear(fno_width, fno_width // 2),
            nn.GELU(),
            nn.Linear(fno_width // 2, latent_dim)
        )

        # ------------------------------------------------------------------ #
        # Global residual: z_t directly added to z_{t+1}                     #
        # Helps the operator learn incremental corrections                    #
        # ------------------------------------------------------------------ #
        self.global_skip = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_t : (B, latent_dim)   current latent brain state

        Returns
        -------
        z_next : (B, latent_dim)   predicted next latent brain state
        """

        # Global residual branch
        z_skip = self.global_skip(z_t)              # (B, latent_dim)

        # Lift to FNO width
        h = self.lift(z_t)                          # (B, fno_width)

        # Unsqueeze sequence dimension T=1
        # (the latent vector IS the state; T=1 means one time-step token)
        h = h.unsqueeze(1)                          # (B, 1, fno_width)

        # Pass through stacked FNO layers
        for layer in self.fno_layers:
            h = layer(h)                            # (B, 1, fno_width)

        # Remove sequence dimension
        h = h.squeeze(1)                            # (B, fno_width)

        # Project back to latent space
        z_pred = self.project(h)                    # (B, latent_dim)

        # Add global residual (incremental dynamics)
        z_next = z_pred + z_skip                    # (B, latent_dim)

        return z_next


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SHAPE VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BATCH_SIZE = 8
    LATENT_DIM = 128
    FNO_WIDTH  = 256
    N_MODES    = 16
    N_LAYERS   = 3

    print("=" * 60)
    print("  FNO Dynamics Block — Shape Verification")
    print("=" * 60)

    model = FNODynamicsBlock(
        latent_dim = LATENT_DIM,
        fno_width  = FNO_WIDTH,
        n_modes    = N_MODES,
        n_layers   = N_LAYERS
    )

    # Dummy latent state z_t
    z_t = torch.randn(BATCH_SIZE, LATENT_DIM)

    # Forward pass
    z_next = model(z_t)

    print(f"\n  Input  z_t   shape : {z_t.shape}")
    print(f"  Output z_t+1 shape : {z_next.shape}")
    print(f"\n  Expected           : (Batch={BATCH_SIZE}, "
          f"Latent_Dim={LATENT_DIM})")

    # Verify correctness
    assert z_next.shape == (BATCH_SIZE, LATENT_DIM), (
        f"Shape mismatch: got {z_next.shape}, "
        f"expected ({BATCH_SIZE}, {LATENT_DIM})"
    )

    print("\n  [OK] Output shape verified")

    # ------------------------------------------------------------------ #
    # Parameter count                                                      #
    # ------------------------------------------------------------------ #
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Complex weights are stored as view_as_real -> double numel
    # Report actual complex param count for spectral layers
    complex_params = sum(
        p.numel() // 2
        for m in model.modules()
        if isinstance(m, SpectralConv1d)
        for p in m.parameters()
    )

    print("\n" + "-" * 60)
    print(f"  {'Total parameters':<30}: {total_params:>10,}")
    print(f"  {'Trainable parameters':<30}: {trainable_params:>10,}")
    print(f"  {'Complex spectral params':<30}: {complex_params:>10,}")
    print("-" * 60)

    # ------------------------------------------------------------------ #
    # Layer-by-layer summary                                              #
    # ------------------------------------------------------------------ #
    print("\n  Layer Summary:")
    print(f"  {'Layer':<35}  {'Output Shape'}")
    print("  " + "-" * 55)

    z_t_dbg  = torch.randn(BATCH_SIZE, LATENT_DIM)
    z_skip   = model.global_skip(z_t_dbg)
    h        = model.lift(z_t_dbg)
    print(f"  {'global_skip (z_t)':<35}  {z_skip.shape}")
    print(f"  {'lift (Linear+GELU)':<35}  {h.shape}")

    h = h.unsqueeze(1)
    print(f"  {'unsqueeze (T=1 token)':<35}  {h.shape}")

    for i, layer in enumerate(model.fno_layers):
        h = layer(h)
        print(f"  {'FNO Layer ' + str(i+1):<35}  {h.shape}")

    h = h.squeeze(1)
    print(f"  {'squeeze':<35}  {h.shape}")

    z_pred = model.project(h)
    print(f"  {'project (Linear->GELU->Linear)':<35}  {z_pred.shape}")

    z_final = z_pred + z_skip
    print(f"  {'global residual add':<35}  {z_final.shape}")

    print("\n" + "=" * 60)
    print("  FNO Dynamics Block ready for NeuroDyn-OpNet pipeline")
    print("=" * 60)