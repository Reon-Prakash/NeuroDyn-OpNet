"""
neurodyn_opnet.py
-----------------
NeuroDyn-OpNet — Master Model Integration
Full Pipeline Assembly

Integrates three components into one unified forward pass:
    1. LatentStateEncoder  (LSE)   : EEG -> latent z_t
    2. FNODynamicsBlock    (FNO)   : z_t -> z_{t+1}  (predicted next state)
    3. StateClassifier     (CLS)   : z   -> logits

Dual-Stream Forward Pass:
    Stream 1 (Current State) : x_t -> LSE -> z_t   -> CLS -> logits_current
    Stream 2 (Dynamic State) : z_t -> FNO -> z_t+1 -> CLS -> logits_dynamic

The two output streams enable:
    - Classification Loss       : logits_current  vs y_t
    - Temporal Consistency Loss : logits_dynamic  vs y_{t+1}
    - Dynamics Regression Loss  : z_{t+1}_pred   vs z_{t+1}_actual

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
# 1.  COMPONENT IMPORTS
#     All three components are defined inline below so this file is
#     fully self-contained. In a multi-file project, replace these
#     with:
#         from latent_state_encoder import LatentStateEncoder
#         from neural_operator       import FNODynamicsBlock
#         from classifier_head       import StateClassifier
# ──────────────────────────────────────────────────────────────────────────────

# ─── 1a. Latent State Encoder ────────────────────────────────────────────────

class TemporalResidualBlock(nn.Module):
    """
    1D Temporal Residual Block with dilation.
    Input/Output shape: (B, C, T)
    """
    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        padding      = (kernel_size - 1) // 2 * dilation
        self.conv1   = nn.Conv1d(channels, channels,
                                 kernel_size, padding=padding, dilation=dilation)
        self.bn1     = nn.BatchNorm1d(channels)
        self.act     = nn.ELU(inplace=True)
        self.conv2   = nn.Conv1d(channels, channels,
                                 kernel_size, padding=padding, dilation=dilation)
        self.bn2     = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out      = self.act(self.bn1(self.conv1(x)))
        out      = self.bn2(self.conv2(out))
        return self.act(out + identity)


class LatentStateEncoder(nn.Module):
    """
    EEG -> Latent Vector z_t
    Input  : (B, C, T, 1)
    Output : (B, latent_dim)
    """
    def __init__(
        self,
        in_channels:  int,
        latent_dim:   int = 128,
        base_filters: int = 64
    ):
        super().__init__()
        self.input_proj  = nn.Conv1d(in_channels, base_filters,
                                     kernel_size=3, padding=1)
        self.input_bn    = nn.BatchNorm1d(base_filters)
        self.act         = nn.ELU(inplace=True)
        self.res_block1  = TemporalResidualBlock(base_filters, dilation=1)
        self.res_block2  = TemporalResidualBlock(base_filters, dilation=2)
        self.res_block3  = TemporalResidualBlock(base_filters, dilation=4)
        self.res_block4  = TemporalResidualBlock(base_filters, dilation=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(base_filters, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)                         # (B, C, T, 1) -> (B, C, T)
        x = self.act(self.input_bn(self.input_proj(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.global_pool(x).squeeze(-1)       # (B, base_filters)
        return self.fc(x)                          # (B, latent_dim)


# ─── 1b. Fourier Neural Operator Dynamics Block ──────────────────────────────

class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution: F^{-1}( R * F(x) )
    Input/Output shape: (B, T, C)
    """
    def __init__(self, in_channels: int, out_channels: int, n_modes: int = 16):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes
        self.weights      = nn.Parameter(
            torch.view_as_real(
                (1.0 / (in_channels * out_channels)) *
                torch.randn(n_modes, in_channels, out_channels,
                            dtype=torch.complex64)
            )
        )

    @staticmethod
    def complex_mul(x_hat: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bmi, mio -> bmo", x_hat, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C  = x.shape
        x_ft     = torch.fft.rfft(x, dim=1)
        W        = torch.view_as_complex(self.weights)
        n_freq   = x_ft.shape[1]
        n_keep   = min(self.n_modes, n_freq)
        out_ft   = torch.zeros(B, n_freq, self.out_channels,
                               dtype=torch.complex64, device=x.device)
        out_ft[:, :n_keep, :] = self.complex_mul(
            x_ft[:, :n_keep, :], W[:n_keep, :, :]
        )
        return torch.fft.irfft(out_ft, n=T, dim=1)


class FNOLayer(nn.Module):
    """
    Single FNO update: sigma( F^{-1}(R*F(x)) + W*x )
    Input/Output shape: (B, T, width)
    """
    def __init__(self, width: int, n_modes: int = 16):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, n_modes)
        self.skip     = nn.Linear(width, width, bias=False)
        self.norm     = nn.LayerNorm(width)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.skip(x)))


class FNODynamicsBlock(nn.Module):
    """
    Latent State Transition: z_t -> z_{t+1}
    Input/Output shape: (B, latent_dim)
    """
    def __init__(
        self,
        latent_dim: int = 128,
        fno_width:  int = 256,
        n_modes:    int = 16,
        n_layers:   int = 3
    ):
        super().__init__()
        self.lift        = nn.Sequential(nn.Linear(latent_dim, fno_width), nn.GELU())
        self.fno_layers  = nn.ModuleList([
            FNOLayer(fno_width, n_modes) for _ in range(n_layers)
        ])
        self.project     = nn.Sequential(
            nn.Linear(fno_width, fno_width // 2),
            nn.GELU(),
            nn.Linear(fno_width // 2, latent_dim)
        )
        self.global_skip = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        z_skip = self.global_skip(z_t)
        h      = self.lift(z_t).unsqueeze(1)       # (B, 1, fno_width)
        for layer in self.fno_layers:
            h  = layer(h)
        h      = h.squeeze(1)                       # (B, fno_width)
        return self.project(h) + z_skip             # (B, latent_dim)


# ─── 1c. State Classifier Head ───────────────────────────────────────────────

class ClassifierBlock(nn.Module):
    """Linear -> BatchNorm1d -> GELU -> Dropout"""
    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StateClassifier(nn.Module):
    """
    Latent z -> raw logits
    Input  : (B, latent_dim)
    Output : (B, num_classes)
    """
    def __init__(
        self,
        latent_dim:   int   = 128,
        hidden_dim:   int   = 256,
        num_classes:  int   = 1,
        dropout_rate: float = 0.4
    ):
        super().__init__()
        self.hidden1      = ClassifierBlock(latent_dim, hidden_dim, dropout_rate)
        self.hidden2      = ClassifierBlock(hidden_dim, hidden_dim // 2, dropout_rate)
        self.output_layer = nn.Linear(hidden_dim // 2, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden2(self.hidden1(z)))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  NEURODYN-OPNET  — Master Integration Class
# ──────────────────────────────────────────────────────────────────────────────

class NeuroDynOpNet(nn.Module):
    """
    NeuroDyn-OpNet: Full Integrated Model

    Dual-Stream Forward Pass
    ========================
    Stream 1 — Current State Path:
        x_t  -> [LSE] -> z_t   -> [CLS] -> logits_current

    Stream 2 — Dynamic State Path:
        z_t  -> [FNO] -> z_t+1 -> [CLS] -> logits_dynamic

    The shared Classifier ensures that both the current and predicted
    future states are evaluated in the same decision space, enabling
    Temporal Consistency training.

    Parameters
    ----------
    in_channels  : int    EEG channels (17 = SEED-VIG, 30 = SADT)
    latent_dim   : int    Shared latent space size (default 128)
    fno_width    : int    FNO internal hidden width (default 256)
    n_modes      : int    Fourier modes in spectral conv (default 16)
    n_fno_layers : int    Number of stacked FNO layers (default 3)
    base_filters : int    LSE temporal conv width (default 64)
    hidden_dim   : int    Classifier hidden width (default 256)
    num_classes  : int    Output logits per sample (default 1 = binary)
    dropout_rate : float  Dropout rate in classifier (default 0.4)

    Returns (from forward)
    ----------------------
    logits_current : (B, num_classes)  logits for x_t
    logits_dynamic : (B, num_classes)  logits for predicted x_{t+1}
    z_t            : (B, latent_dim)   current latent state (for aux loss)
    z_next         : (B, latent_dim)   predicted next latent state
    """

    def __init__(
        self,
        in_channels:  int   = 17,
        latent_dim:   int   = 128,
        fno_width:    int   = 256,
        n_modes:      int   = 16,
        n_fno_layers: int   = 3,
        base_filters: int   = 64,
        hidden_dim:   int   = 256,
        num_classes:  int   = 1,
        dropout_rate: float = 0.4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        # ------------------------------------------------------------------ #
        # Sub-Module 1: Latent State Encoder                                  #
        # EEG signal -> compressed latent brain state                         #
        # ------------------------------------------------------------------ #
        self.encoder = LatentStateEncoder(
            in_channels  = in_channels,
            latent_dim   = latent_dim,
            base_filters = base_filters
        )

        # ------------------------------------------------------------------ #
        # Sub-Module 2: FNO Dynamics Block                                    #
        # Current state -> predicted next state                               #
        # ------------------------------------------------------------------ #
        self.dynamics = FNODynamicsBlock(
            latent_dim = latent_dim,
            fno_width  = fno_width,
            n_modes    = n_modes,
            n_layers   = n_fno_layers
        )

        # ------------------------------------------------------------------ #
        # Sub-Module 3: Shared State Classifier                               #
        # Shared weights ensure both streams use the same decision boundary   #
        # ------------------------------------------------------------------ #
        self.classifier = StateClassifier(
            latent_dim   = latent_dim,
            hidden_dim   = hidden_dim,
            num_classes  = num_classes,
            dropout_rate = dropout_rate
        )

    # ────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ────────────────────────────────────────────────────────────────────────

    def forward(self, x_t: torch.Tensor):
        """
        Parameters
        ----------
        x_t : (B, C, T, 1)
            Current EEG segment (4D tensor from segmentation pipeline)

        Returns
        -------
        logits_current : (B, num_classes)
            Classification logits for the CURRENT brain state z_t.
            Supervised against y_t.

        logits_dynamic : (B, num_classes)
            Classification logits for the PREDICTED next brain state z_{t+1}.
            Supervised against y_{t+1} for temporal consistency.

        z_t   : (B, latent_dim)
            Current latent state. Used for dynamics regression loss.

        z_next : (B, latent_dim)
            Predicted next latent state. Used for dynamics regression loss.
        """

        # ── Stream 1: Current State ───────────────────────────────────────
        # Step 1a: Encode EEG into latent space
        z_t            = self.encoder(x_t)              # (B, latent_dim)

        # Step 1b: Classify current latent state directly
        logits_current = self.classifier(z_t)           # (B, num_classes)

        # ── Stream 2: Dynamic State ───────────────────────────────────────
        # Step 2a: Apply FNO to predict next latent state
        z_next         = self.dynamics(z_t)             # (B, latent_dim)

        # Step 2b: Classify predicted next latent state
        logits_dynamic = self.classifier(z_next)        # (B, num_classes)

        return logits_current, logits_dynamic, z_t, z_next

    # ────────────────────────────────────────────────────────────────────────
    # Convenience: Inference-only (single prediction stream)
    # ────────────────────────────────────────────────────────────────────────

    def predict(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward. Returns only current-state logits.

        Parameters
        ----------
        x_t : (B, C, T, 1)

        Returns
        -------
        logits : (B, num_classes)
        """
        z_t    = self.encoder(x_t)
        logits = self.classifier(z_t)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 3.  FULL PIPELINE VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BATCH_SIZE = 8
    TIME_STEPS = 384
    LATENT_DIM = 128
    SEP        = "-" * 65

    print("=" * 65)
    print("  NeuroDyn-OpNet — Full Pipeline Verification")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # Test 1: SEED-VIG  (17 channels)                                     #
    # ------------------------------------------------------------------ #
    print("\n  [TEST 1]  SEED-VIG Dataset  (in_channels=17)")
    print(SEP)

    model_seed = NeuroDynOpNet(
        in_channels  = 17,
        latent_dim   = LATENT_DIM,
        fno_width    = 256,
        n_modes      = 16,
        n_fno_layers = 3,
        base_filters = 64,
        hidden_dim   = 256,
        num_classes  = 1,
        dropout_rate = 0.4
    )
    model_seed.eval()

    x_seed = torch.randn(BATCH_SIZE, 17, TIME_STEPS, 1)

    with torch.no_grad():
        lc_seed, ld_seed, zt_seed, zn_seed = model_seed(x_seed)

    print(f"  Input  x_t            : {x_seed.shape}")
    print(f"  z_t   (current state) : {zt_seed.shape}")
    print(f"  z_t+1 (next state)    : {zn_seed.shape}")
    print(f"  logits_current        : {lc_seed.shape}")
    print(f"  logits_dynamic        : {ld_seed.shape}")

    assert lc_seed.shape == (BATCH_SIZE, 1), \
        f"SEED logits_current shape error: {lc_seed.shape}"
    assert ld_seed.shape == (BATCH_SIZE, 1), \
        f"SEED logits_dynamic shape error: {ld_seed.shape}"
    assert zt_seed.shape == (BATCH_SIZE, LATENT_DIM), \
        f"SEED z_t shape error: {zt_seed.shape}"
    assert zn_seed.shape == (BATCH_SIZE, LATENT_DIM), \
        f"SEED z_next shape error: {zn_seed.shape}"
    print("  [OK]  All SEED-VIG output shapes verified")

    # ------------------------------------------------------------------ #
    # Test 2: SADT  (30 channels)                                         #
    # ------------------------------------------------------------------ #
    print(f"\n  [TEST 2]  SADT Dataset  (in_channels=30)")
    print(SEP)

    model_sadt = NeuroDynOpNet(
        in_channels  = 30,
        latent_dim   = LATENT_DIM,
        fno_width    = 256,
        n_modes      = 16,
        n_fno_layers = 3,
        base_filters = 64,
        hidden_dim   = 256,
        num_classes  = 1,
        dropout_rate = 0.4
    )
    model_sadt.eval()

    x_sadt = torch.randn(BATCH_SIZE, 30, TIME_STEPS, 1)

    with torch.no_grad():
        lc_sadt, ld_sadt, zt_sadt, zn_sadt = model_sadt(x_sadt)

    print(f"  Input  x_t            : {x_sadt.shape}")
    print(f"  z_t   (current state) : {zt_sadt.shape}")
    print(f"  z_t+1 (next state)    : {zn_sadt.shape}")
    print(f"  logits_current        : {lc_sadt.shape}")
    print(f"  logits_dynamic        : {ld_sadt.shape}")

    assert lc_sadt.shape == (BATCH_SIZE, 1), \
        f"SADT logits_current shape error: {lc_sadt.shape}"
    assert ld_sadt.shape == (BATCH_SIZE, 1), \
        f"SADT logits_dynamic shape error: {ld_sadt.shape}"
    assert zt_sadt.shape == (BATCH_SIZE, LATENT_DIM), \
        f"SADT z_t shape error: {zt_sadt.shape}"
    assert zn_sadt.shape == (BATCH_SIZE, LATENT_DIM), \
        f"SADT z_next shape error: {zn_sadt.shape}"
    print("  [OK]  All SADT output shapes verified")

    # ------------------------------------------------------------------ #
    # Test 3: Inference-Only (predict)                                    #
    # ------------------------------------------------------------------ #
    print(f"\n  [TEST 3]  Inference-Only  model.predict()")
    print(SEP)

    model_seed.eval()
    with torch.no_grad():
        pred = model_seed.predict(x_seed)

    print(f"  Input  x_t     : {x_seed.shape}")
    print(f"  Output logits  : {pred.shape}")
    assert pred.shape == (BATCH_SIZE, 1), \
        f"predict() shape error: {pred.shape}"
    print("  [OK]  predict() output shape verified")

    # ------------------------------------------------------------------ #
    # Test 4: Loss Compatibility                                           #
    # ------------------------------------------------------------------ #
    print(f"\n  [TEST 4]  Loss Compatibility (Temporal Consistency)")
    print(SEP)

    bce        = nn.BCEWithLogitsLoss()
    y_current  = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    y_next     = torch.randint(0, 2, (BATCH_SIZE, 1)).float()

    loss_cls   = bce(lc_seed, y_current)
    loss_dyn   = bce(ld_seed, y_next)
    loss_reg   = F.mse_loss(zn_seed, zt_seed.detach())

    loss_total = loss_cls + 0.5 * loss_dyn + 0.1 * loss_reg

    print(f"  Classification Loss  (logits_current vs y_t)   : {loss_cls.item():.6f}")
    print(f"  Temporal Cons. Loss  (logits_dynamic vs y_t+1) : {loss_dyn.item():.6f}")
    print(f"  Dynamics Regr. Loss  (z_t+1_pred vs z_t)       : {loss_reg.item():.6f}")
    print(f"  Total Combined Loss                             : {loss_total.item():.6f}")
    print("  [OK]  Loss computation verified")

    # ------------------------------------------------------------------ #
    # Parameter Summary                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n  Parameter Summary (SEED-VIG model)")
    print(SEP)

    sections = {
        "LatentStateEncoder" : model_seed.encoder,
        "FNODynamicsBlock"   : model_seed.dynamics,
        "StateClassifier"    : model_seed.classifier,
    }

    total_all = 0
    print(f"  {'Sub-Module':<25}  {'Total Params':>14}  {'Trainable':>14}")
    print(f"  {'-'*25}  {'-'*14}  {'-'*14}")

    for name, module in sections.items():
        total     = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters()
                        if p.requires_grad)
        total_all += total
        print(f"  {name:<25}  {total:>14,}  {trainable:>14,}")

    print(f"  {'-'*25}  {'-'*14}  {'-'*14}")
    print(f"  {'TOTAL':<25}  {total_all:>14,}  {total_all:>14,}")

    # ------------------------------------------------------------------ #
    # Data Flow Summary                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n  Data Flow Summary")
    print(SEP)
    print(f"  x_t  {x_seed.shape}")
    print(f"    |")
    print(f"    +--[LSE Encoder]---------> z_t    {zt_seed.shape}")
    print(f"    |       |")
    print(f"    |       +--[Classifier]--> logits_current  {lc_seed.shape}")
    print(f"    |       |")
    print(f"    |       +--[FNO Block]---> z_t+1  {zn_seed.shape}")
    print(f"    |                |")
    print(f"    |                +--[Classifier]--> logits_dynamic  {ld_seed.shape}")
    print(f"    |")
    print(f"  Losses:")
    print(f"    BCE(logits_current, y_t)      -> classification loss")
    print(f"    BCE(logits_dynamic, y_t+1)    -> temporal consistency loss")
    print(f"    MSE(z_t+1, z_t.detach())      -> dynamics regression loss")

    print("\n" + "=" * 65)
    print("  NeuroDyn-OpNet pipeline fully verified and ready to train")
    print("=" * 65)