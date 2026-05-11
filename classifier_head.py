"""
classifier_head.py
------------------
State Classifier Head
NeuroDyn-OpNet Component

Input  : (Batch, Latent_Dim)   <- latent vector z from LSE or FNO
Output : (Batch, num_classes)  <- raw logits (no sigmoid/softmax)

Use BCEWithLogitsLoss  for binary classification (num_classes=1)
Use CrossEntropyLoss   for multi-class classification (num_classes>1)

Design:
    - Linear expansion layer
    - BatchNorm1d + GELU + Dropout
    - Linear compression layer
    - BatchNorm1d + GELU + Dropout
    - Final linear projection to logits

Author: NeuroDyn-OpNet Architecture
"""

import os
import sys
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# 0.  WINDOWS CONSOLE ENCODING FIX
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CLASSIFIER BLOCK  (Reusable sub-unit)
# ──────────────────────────────────────────────────────────────────────────────

class ClassifierBlock(nn.Module):
    """
    Single Linear block:
        Linear -> BatchNorm1d -> GELU -> Dropout

    Parameters
    ----------
    in_dim       : int    Input feature dimension
    out_dim      : int    Output feature dimension
    dropout_rate : float  Dropout probability
    """

    def __init__(
        self,
        in_dim:       int,
        out_dim:      int,
        dropout_rate: float = 0.4
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  STATE CLASSIFIER HEAD
# ──────────────────────────────────────────────────────────────────────────────

class StateClassifier(nn.Module):
    """
    State Classifier Head for NeuroDyn-OpNet.

    Accepts a latent vector z and produces raw logits for
    binary or multi-class fatigue/drowsiness classification.

    Architecture:
        z (B, latent_dim)
            |
            | Linear(latent_dim -> hidden_dim)  + BN + GELU + Dropout
            v
        (B, hidden_dim)
            |
            | Linear(hidden_dim -> hidden_dim//2) + BN + GELU + Dropout
            v
        (B, hidden_dim // 2)
            |
            | Linear(hidden_dim//2 -> num_classes)
            v
        logits (B, num_classes)   <- raw, no activation

    Loss pairing:
        num_classes = 1  ->  BCEWithLogitsLoss  (binary)
        num_classes > 1  ->  CrossEntropyLoss   (multi-class)

    Parameters
    ----------
    latent_dim   : int    Must match LSE / FNO output dim (default 128)
    hidden_dim   : int    Expansion width of hidden layers  (default 256)
    num_classes  : int    1 = binary logit, >1 = multi-class logits
    dropout_rate : float  Dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        latent_dim:   int   = 128,
        hidden_dim:   int   = 256,
        num_classes:  int   = 1,
        dropout_rate: float = 0.4
    ):
        super().__init__()

        self.latent_dim   = latent_dim
        self.hidden_dim   = hidden_dim
        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        # ------------------------------------------------------------------ #
        # Hidden Layer 1: Expand  latent_dim -> hidden_dim                   #
        # ------------------------------------------------------------------ #
        self.hidden1 = ClassifierBlock(
            in_dim       = latent_dim,
            out_dim      = hidden_dim,
            dropout_rate = dropout_rate
        )

        # ------------------------------------------------------------------ #
        # Hidden Layer 2: Compress  hidden_dim -> hidden_dim // 2            #
        # ------------------------------------------------------------------ #
        self.hidden2 = ClassifierBlock(
            in_dim       = hidden_dim,
            out_dim      = hidden_dim // 2,
            dropout_rate = dropout_rate
        )

        # ------------------------------------------------------------------ #
        # Output Layer: Project to logits  hidden_dim//2 -> num_classes      #
        # No activation — raw logits returned                                 #
        # ------------------------------------------------------------------ #
        self.output_layer = nn.Linear(hidden_dim // 2, num_classes)

        # ------------------------------------------------------------------ #
        # Weight Initialization                                               #
        # ------------------------------------------------------------------ #
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming uniform for linear layers (works well with GELU/ReLU).
        Zero-init bias on output layer for stable early training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z      : (B, latent_dim)   latent brain state vector

        Returns
        -------
        logits : (B, num_classes)  raw logits
                 -> squeeze to (B,) outside if using BCEWithLogitsLoss
        """
        h = self.hidden1(z)          # (B, hidden_dim)
        h = self.hidden2(h)          # (B, hidden_dim // 2)
        logits = self.output_layer(h)  # (B, num_classes)

        return logits


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SHAPE VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BATCH_SIZE   = 8
    LATENT_DIM   = 128
    HIDDEN_DIM   = 256
    DROPOUT_RATE = 0.4

    SEP = "-" * 60

    print("=" * 60)
    print("  State Classifier Head — Shape Verification")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Test 1: Binary Classification  (num_classes = 1)                    #
    # ------------------------------------------------------------------ #
    print("\n  [TEST 1]  Binary Classification  (num_classes=1)")
    print(SEP)

    model_binary = StateClassifier(
        latent_dim   = LATENT_DIM,
        hidden_dim   = HIDDEN_DIM,
        num_classes  = 1,
        dropout_rate = DROPOUT_RATE
    )
    model_binary.eval()

    z_dummy = torch.randn(BATCH_SIZE, LATENT_DIM)
    with torch.no_grad():
        logits_binary = model_binary(z_dummy)

    print(f"  Input  z       shape : {z_dummy.shape}")
    print(f"  Output logits  shape : {logits_binary.shape}")
    print(f"  Expected             : ({BATCH_SIZE}, 1)")

    assert logits_binary.shape == (BATCH_SIZE, 1), (
        f"Binary test failed: got {logits_binary.shape}"
    )
    print("  [OK] Binary output shape verified")

    # ------------------------------------------------------------------ #
    # Test 2: Multi-Class  (num_classes = 3 — e.g. Alert/Drowsy/Sleep)   #
    # ------------------------------------------------------------------ #
    print(f"\n  [TEST 2]  Multi-Class Classification  (num_classes=3)")
    print(SEP)

    model_multi = StateClassifier(
        latent_dim   = LATENT_DIM,
        hidden_dim   = HIDDEN_DIM,
        num_classes  = 3,
        dropout_rate = DROPOUT_RATE
    )
    model_multi.eval()

    with torch.no_grad():
        logits_multi = model_multi(z_dummy)

    print(f"  Input  z       shape : {z_dummy.shape}")
    print(f"  Output logits  shape : {logits_multi.shape}")
    print(f"  Expected             : ({BATCH_SIZE}, 3)")

    assert logits_multi.shape == (BATCH_SIZE, 3), (
        f"Multi-class test failed: got {logits_multi.shape}"
    )
    print("  [OK] Multi-class output shape verified")

    # ------------------------------------------------------------------ #
    # Test 3: Loss Compatibility Check                                     #
    # ------------------------------------------------------------------ #
    print(f"\n  [TEST 3]  Loss Compatibility")
    print(SEP)

    # BCEWithLogitsLoss (binary)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    y_binary    = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    bce_loss    = bce_loss_fn(logits_binary, y_binary)
    print(f"  BCEWithLogitsLoss (binary)     : {bce_loss.item():.6f}")

    # CrossEntropyLoss (multi-class)
    ce_loss_fn = nn.CrossEntropyLoss()
    y_multi    = torch.randint(0, 3, (BATCH_SIZE,))
    ce_loss    = ce_loss_fn(logits_multi, y_multi)
    print(f"  CrossEntropyLoss  (multi)      : {ce_loss.item():.6f}")

    print("  [OK] Both loss functions accept raw logits correctly")

    # ------------------------------------------------------------------ #
    # Parameter Count                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n  Parameter Count")
    print(SEP)

    for name, model in [("Binary", model_binary), ("Multi-Class", model_multi)]:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:<15}  Total: {total:>8,}   Trainable: {trainable:>8,}")

    # ------------------------------------------------------------------ #
    # Layer-by-Layer Summary (Binary model)                               #
    # ------------------------------------------------------------------ #
    print(f"\n  Layer-by-Layer Summary (Binary Classifier)")
    print(SEP)
    print(f"  {'Layer':<40}  Output Shape")
    print("  " + "-" * 55)

    z_dbg = torch.randn(BATCH_SIZE, LATENT_DIM)
    model_binary.eval()

    with torch.no_grad():
        h1 = model_binary.hidden1(z_dbg)
        h2 = model_binary.hidden2(h1)
        out = model_binary.output_layer(h2)

    print(f"  {'Input z':<40}  {z_dbg.shape}")
    print(f"  {'hidden1 (Linear->BN->GELU->Drop)':<40}  {h1.shape}")
    print(f"  {'hidden2 (Linear->BN->GELU->Drop)':<40}  {h2.shape}")
    print(f"  {'output_layer (Linear, no activ)':<40}  {out.shape}")

    print("\n" + "=" * 60)
    print("  State Classifier Head ready for NeuroDyn-OpNet pipeline")
    print("=" * 60)