"""
predict_fatigue.py
------------------
Real-Time Fatigue Inference — NeuroDyn-OpNet

Pipeline:
    1. Load trained model weights
    2. Preprocess a single EEG sample -> (1, C, 384, 1)
    3. Run model.predict() -> raw logit
    4. Apply sigmoid -> probability
    5. Threshold at 0.5 -> FATIGUED / ALERT

Author: NeuroDyn-OpNet Inference Pipeline
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  WINDOWS CONSOLE ENCODING FIX
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

ROOT_PATH = r"C:\Users\KIIT0001\.vscode\Codes\NeuroDyn-OpNet"
os.chdir(ROOT_PATH)
sys.path.insert(0, ROOT_PATH)

print("=" * 65)
print("  NeuroDyn-OpNet — Real-Time Fatigue Inference")
print("=" * 65)
print(f"  [INFO] Working directory : {os.getcwd()}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  LOCAL MODEL IMPORT
# ──────────────────────────────────────────────────────────────────────────────

from neurodyn_opnet import NeuroDynOpNet   # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# 3.  DEVICE  (CPU default for inference)
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
print(f"  [INFO] Inference device  : {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  INFERENCE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

INFERENCE_CFG = {
    # Model architecture — must match the saved checkpoint exactly
    "in_channels"  : 17,      # SEED-VIG model (17 ch); switch to 30 for SADT
    "latent_dim"   : 128,
    "fno_width"    : 256,
    "n_modes"      : 16,
    "n_fno_layers" : 3,
    "base_filters" : 64,
    "hidden_dim"   : 256,
    "num_classes"  : 1,
    "dropout_rate" : 0.4,

    # Checkpoint
    "weights_path" : os.path.join(ROOT_PATH, "neurodyn_opnet_seed_best.pth"),

    # Data source for test sampling
    "data_X"       : os.path.join(ROOT_PATH, "seed_ready_X.npy"),
    "data_Y"       : os.path.join(ROOT_PATH, "seed_ready_Y.npy"),

    # Decision threshold
    "threshold"    : 0.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# 5.  MODEL LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, device: torch.device) -> NeuroDynOpNet:
    """
    Instantiate NeuroDynOpNet and load pre-trained weights.

    Parameters
    ----------
    cfg    : dict           INFERENCE_CFG dictionary
    device : torch.device

    Returns
    -------
    model : NeuroDynOpNet   in eval() mode on device
    """
    weights_path = cfg["weights_path"]

    print(f"\n  [LOAD] Weights path : {weights_path}")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"\n  [ERROR] Checkpoint not found: {weights_path}\n"
            f"          Run train.py first to generate the .pth file.\n"
            f"          (For testing, we will use random weights below.)"
        )

    # Build architecture
    model = NeuroDynOpNet(
        in_channels  = cfg["in_channels"],
        latent_dim   = cfg["latent_dim"],
        fno_width    = cfg["fno_width"],
        n_modes      = cfg["n_modes"],
        n_fno_layers = cfg["n_fno_layers"],
        base_filters = cfg["base_filters"],
        hidden_dim   = cfg["hidden_dim"],
        num_classes  = cfg["num_classes"],
        dropout_rate = cfg["dropout_rate"],
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)

    # Support both raw state_dict and wrapped checkpoint dict
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        saved_acc  = checkpoint.get("val_acc", "N/A")
        saved_f1   = checkpoint.get("val_f1",  "N/A")
        saved_ep   = checkpoint.get("epoch",   "N/A")
        print(f"  [INFO] Checkpoint epoch    : {saved_ep}")
        print(f"  [INFO] Saved val accuracy  : "
              f"{float(saved_acc)*100:.2f}%" if saved_acc != "N/A"
              else f"  [INFO] Saved val accuracy  : N/A")
        print(f"  [INFO] Saved val F1        : "
              f"{float(saved_f1):.4f}" if saved_f1 != "N/A"
              else f"  [INFO] Saved val F1        : N/A")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    # CRITICAL: disable dropout and batch norm updates
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [INFO] Model parameters    : {total_params:,}")
    print(f"  [OK]   Model loaded and set to eval() mode")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PREPROCESSING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_sample(
    data:   np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """
    Ensure a single EEG sample has the correct 4D shape for NeuroDynOpNet.

    Accepted input shapes:
        (C, T)       -> unsqueeze batch + trailing dim -> (1, C, T, 1)
        (C, T, 1)    -> unsqueeze batch               -> (1, C, T, 1)
        (1, C, T, 1) -> already correct, pass through
        (C,)         -> raise informative error

    Parameters
    ----------
    data   : np.ndarray   raw EEG sample
    device : torch.device

    Returns
    -------
    tensor : torch.Tensor  shape (1, C, 384, 1)  dtype float32
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"[ERROR] data must be np.ndarray or torch.Tensor, "
            f"got {type(data)}"
        )

    # Convert to numpy first for uniform handling
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    original_shape = data.shape

    # ── Shape normalisation ─────────────────────────────────────────────
    if data.ndim == 2:
        # (C, T) -> (1, C, T, 1)
        data = data[np.newaxis, :, :, np.newaxis]

    elif data.ndim == 3:
        if data.shape[-1] == 1:
            # (C, T, 1) -> (1, C, T, 1)
            data = data[np.newaxis, :, :, :]
        else:
            # (1, C, T) or (B, C, T) with B=1 -> (1, C, T, 1)
            data = data[:, :, :, np.newaxis]

    elif data.ndim == 4:
        # (1, C, T, 1) -> already correct
        pass

    else:
        raise ValueError(
            f"[ERROR] Cannot preprocess data with shape {original_shape}.\n"
            f"        Expected: (C,T) | (C,T,1) | (1,C,T,1)"
        )

    # ── Validate time steps ─────────────────────────────────────────────
    T = data.shape[2]
    if T != 384:
        raise ValueError(
            f"[ERROR] Time_Steps must be 384, got {T}.\n"
            f"        Ensure your EEG window is 384 samples."
        )

    # ── Cast to float32 tensor ──────────────────────────────────────────
    tensor = torch.tensor(data, dtype=torch.float32).to(device)

    print(f"  [PREPROCESS] Input shape  : {original_shape}")
    print(f"  [PREPROCESS] Output shape : {tensor.shape}  dtype={tensor.dtype}")

    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SINGLE SAMPLE INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def predict_fatigue(
    model:     NeuroDynOpNet,
    tensor:    torch.Tensor,
    threshold: float = 0.5,
    verbose:   bool  = True
) -> dict:
    """
    Run fatigue inference on a single preprocessed EEG tensor.

    Parameters
    ----------
    model     : NeuroDynOpNet   in eval() mode
    tensor    : torch.Tensor    shape (1, C, 384, 1)  float32
    threshold : float           decision boundary (default 0.5)
    verbose   : bool            print detailed output

    Returns
    -------
    result : dict with keys:
        logit       : float   raw model output
        probability : float   sigmoid(logit)
        label       : str     "FATIGUED" or "ALERT"
        confidence  : float   distance from 0.5 threshold
        latency_ms  : float   inference time in milliseconds
    """
    model.eval()

    with torch.no_grad():

        t_start = time.perf_counter()

        # ── Inference: current-state stream only ──────────────────────
        logit = model.predict(tensor)          # (1, 1)

        t_end   = time.perf_counter()
        latency = (t_end - t_start) * 1000    # ms

        # ── Post-processing ───────────────────────────────────────────
        prob       = torch.sigmoid(logit).item()
        raw_logit  = logit.item()
        label      = "FATIGUED" if prob > threshold else "ALERT"
        confidence = abs(prob - 0.5) * 2      # 0 = uncertain, 1 = certain

        # ── Dual-stream for additional context ────────────────────────
        lc, ld, z_t, z_next = model(tensor)
        prob_current = torch.sigmoid(lc).item()
        prob_dynamic = torch.sigmoid(ld).item()
        label_dynamic = "FATIGUED" if prob_dynamic > threshold else "ALERT"

    result = {
        "logit"          : raw_logit,
        "probability"    : prob,
        "label"          : label,
        "confidence"     : confidence,
        "latency_ms"     : latency,
        "prob_current"   : prob_current,
        "prob_dynamic"   : prob_dynamic,
        "label_dynamic"  : label_dynamic,
        "latent_norm"    : z_t.norm().item(),
    }

    if verbose:
        _print_result(result, threshold)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 8.  RESULT PRINTER
# ──────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict, threshold: float) -> None:
    """Print a formatted inference result panel."""

    SEP   = "=" * 65
    SEP2  = "-" * 65

    label = result["label"]
    prob  = result["probability"]
    conf  = result["confidence"]

    # Choose display style based on prediction
    if label == "FATIGUED":
        status_line = f"  >>>  PREDICTION  :  *** FATIGUED ***"
        alert_line  = f"  >>>  ACTION      :  ALERT DRIVER IMMEDIATELY"
    else:
        status_line = f"  >>>  PREDICTION  :  ALERT"
        alert_line  = f"  >>>  ACTION      :  No action required"

    # Confidence bar (ASCII)
    bar_len  = 30
    filled   = int(conf * bar_len)
    conf_bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

    print(f"\n{SEP}")
    print(f"  FATIGUE DETECTION RESULT")
    print(SEP2)
    print(f"  Raw Logit           :  {result['logit']:+.6f}")
    print(f"  Sigmoid Probability :  {prob:.6f}  ({prob*100:.2f}%)")
    print(f"  Decision Threshold  :  {threshold:.2f}")
    print(SEP2)
    print(status_line)
    print(alert_line)
    print(SEP2)
    print(f"  Confidence          :  {conf_bar}  {conf*100:.1f}%")
    print(f"  Inference Latency   :  {result['latency_ms']:.3f} ms")
    print(SEP2)
    print(f"  -- Dual-Stream Analysis --")
    print(f"  Stream 1 (Current)  :  {result['prob_current']*100:.2f}%  "
          f"-> {result['label']}")
    print(f"  Stream 2 (Dynamic)  :  {result['prob_dynamic']*100:.2f}%  "
          f"-> {result['label_dynamic']}")
    print(f"  Latent State Norm   :  {result['latent_norm']:.4f}")
    print(f"  Stream Agreement    :  "
          f"{'YES' if result['label'] == result['label_dynamic'] else 'NO (uncertain boundary)'}")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# 9.  BATCH INFERENCE  (multiple samples at once)
# ──────────────────────────────────────────────────────────────────────────────

def predict_batch(
    model:     NeuroDynOpNet,
    X:         np.ndarray,
    Y:         np.ndarray,
    indices:   list,
    threshold: float = 0.5,
    device:    torch.device = DEVICE
) -> None:
    """
    Run inference on a list of sample indices and compare with ground truth.

    Parameters
    ----------
    model     : NeuroDynOpNet
    X         : np.ndarray   (N, C, T, 1)  full dataset
    Y         : np.ndarray   (N,)          labels
    indices   : list         sample indices to evaluate
    threshold : float
    device    : torch.device
    """
    SEP  = "=" * 75
    SEP2 = "-" * 75

    print(f"\n{SEP}")
    print(f"  BATCH INFERENCE  ({len(indices)} samples)")
    print(SEP2)
    print(f"  {'Idx':>5}  {'True Label':>12}  {'Pred Label':>12}  "
          f"{'Prob':>8}  {'Confidence':>12}  {'Match':>7}")
    print(SEP2)

    correct = 0

    for idx in indices:
        sample     = X[idx]                        # (C, T, 1)
        true_label = int(Y[idx])
        true_str   = "FATIGUED" if true_label == 1 else "ALERT"

        tensor = preprocess_sample(sample, device)

        with torch.no_grad():
            logit = model.predict(tensor)
            prob  = torch.sigmoid(logit).item()

        pred_str   = "FATIGUED" if prob > threshold else "ALERT"
        confidence = abs(prob - 0.5) * 2
        match      = "OK" if pred_str == true_str else "MISS"

        if pred_str == true_str:
            correct += 1

        print(f"  {idx:>5}  {true_str:>12}  {pred_str:>12}  "
              f"{prob:>7.4f}  {confidence*100:>10.1f}%  {match:>7}")

    accuracy = correct / len(indices) * 100
    print(SEP2)
    print(f"  Batch Accuracy : {correct}/{len(indices)} = {accuracy:.2f}%")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# 10.  FALLBACK: RANDOM WEIGHT MODEL  (when no checkpoint exists)
# ──────────────────────────────────────────────────────────────────────────────

def load_model_safe(cfg: dict, device: torch.device) -> tuple:
    """
    Try to load trained weights. If .pth file missing, use random weights
    and flag it clearly so the demo still runs end-to-end.

    Returns
    -------
    model    : NeuroDynOpNet
    is_trained : bool
    """
    try:
        model      = load_model(cfg, device)
        is_trained = True
    except FileNotFoundError as e:
        print(f"\n  [WARN] {e}")
        print(f"  [WARN] Falling back to randomly initialized weights.")
        print(f"  [WARN] Predictions will be meaningless — run train.py first.")
        model = NeuroDynOpNet(
            in_channels  = cfg["in_channels"],
            latent_dim   = cfg["latent_dim"],
            fno_width    = cfg["fno_width"],
            n_modes      = cfg["n_modes"],
            n_fno_layers = cfg["n_fno_layers"],
            base_filters = cfg["base_filters"],
            hidden_dim   = cfg["hidden_dim"],
            num_classes  = cfg["num_classes"],
            dropout_rate = cfg["dropout_rate"],
        ).to(device)
        model.eval()
        is_trained = False

    return model, is_trained


# ──────────────────────────────────────────────────────────────────────────────
# 11.  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n  [INFO] PyTorch version : {torch.__version__}")
    print(f"  [INFO] NumPy   version : {np.__version__}")

    # ── Load Model ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 1 — Model Loading")
    print(f"{'='*65}")

    model, is_trained = load_model_safe(INFERENCE_CFG, DEVICE)

    if not is_trained:
        print("\n  [NOTE] Using random weights for demonstration only.")

    # ── Load Data ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 2 — Loading EEG Data")
    print(f"{'='*65}")

    data_X_path = INFERENCE_CFG["data_X"]
    data_Y_path = INFERENCE_CFG["data_Y"]

    if not os.path.isfile(data_X_path):
        raise FileNotFoundError(
            f"[ERROR] Data file not found: {data_X_path}\n"
            f"        Run segmentation.py first."
        )

    X = np.load(data_X_path, allow_pickle=False)   # (N, C, T, 1)
    Y = np.load(data_Y_path, allow_pickle=False)   # (N,)

    print(f"  [LOAD] X shape : {X.shape}   dtype={X.dtype}")
    print(f"  [LOAD] Y shape : {Y.shape}   dtype={Y.dtype}")

    n_total    = len(X)
    n_fatigued = int((Y == 1).sum())
    n_alert    = int((Y == 0).sum())

    print(f"  [INFO] Total samples   : {n_total}")
    print(f"  [INFO] FATIGUED (y=1)  : {n_fatigued}  "
          f"({n_fatigued/n_total*100:.1f}%)")
    print(f"  [INFO] ALERT    (y=0)  : {n_alert}  "
          f"({n_alert/n_total*100:.1f}%)")

    # ── TEST CASE 1: Single Random Sample ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 3 — Single Random Sample Inference")
    print(f"{'='*65}")

    rng        = np.random.default_rng(seed=42)
    random_idx = int(rng.integers(0, n_total))
    raw_sample = X[random_idx]                     # (C, T, 1)
    true_label = int(Y[random_idx])
    true_str   = "FATIGUED" if true_label == 1 else "ALERT"

    print(f"\n  Sample index       : {random_idx}")
    print(f"  Ground truth label : {true_str}  (y={true_label})")

    tensor = preprocess_sample(raw_sample, DEVICE)
    result = predict_fatigue(model, tensor,
                             threshold=INFERENCE_CFG["threshold"])

    match  = result["label"] == true_str
    print(f"\n  Ground Truth : {true_str}")
    print(f"  Prediction   : {result['label']}")
    print(f"  Correct      : {'YES' if match else 'NO'}")

    # ── TEST CASE 2: Known FATIGUED sample ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 4 — Targeted Test: Known FATIGUED Sample")
    print(f"{'='*65}")

    fatigued_indices = np.where(Y == 1)[0]
    if len(fatigued_indices) > 0:
        fat_idx    = int(fatigued_indices[0])
        fat_sample = X[fat_idx]
        print(f"\n  Sample index (FATIGUED) : {fat_idx}")
        tensor_fat = preprocess_sample(fat_sample, DEVICE)
        result_fat = predict_fatigue(model, tensor_fat,
                                     threshold=INFERENCE_CFG["threshold"])
        print(f"  Detected as  : {result_fat['label']}")
    else:
        print("  [WARN] No FATIGUED samples found in dataset.")

    # ── TEST CASE 3: Known ALERT sample ───────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 5 — Targeted Test: Known ALERT Sample")
    print(f"{'='*65}")

    alert_indices = np.where(Y == 0)[0]
    if len(alert_indices) > 0:
        alt_idx    = int(alert_indices[0])
        alt_sample = X[alt_idx]
        print(f"\n  Sample index (ALERT) : {alt_idx}")
        tensor_alt = preprocess_sample(alt_sample, DEVICE)
        result_alt = predict_fatigue(model, tensor_alt,
                                     threshold=INFERENCE_CFG["threshold"])
        print(f"  Detected as  : {result_alt['label']}")
    else:
        print("  [WARN] No ALERT samples found in dataset.")

    # ── TEST CASE 4: Batch Inference ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STEP 6 — Batch Inference  (10 random samples)")
    print(f"{'='*65}")

    batch_indices = rng.integers(0, n_total, size=10).tolist()
    predict_batch(model, X, Y, batch_indices,
                  threshold=INFERENCE_CFG["threshold"],
                  device=DEVICE)

    # ── TEST CASE 5: Custom numpy array (simulate real-time sensor input)
    print(f"\n{'='*65}")
    print(f"  STEP 7 — Simulated Real-Time Sensor Input")
    print(f"{'='*65}")

    C = INFERENCE_CFG["in_channels"]   # 17 for SEED-VIG
    simulated_eeg = np.random.randn(C, 384).astype(np.float64)

    print(f"\n  Simulated EEG shape (sensor output) : {simulated_eeg.shape}")
    tensor_sim = preprocess_sample(simulated_eeg, DEVICE)
    result_sim = predict_fatigue(model, tensor_sim,
                                 threshold=INFERENCE_CFG["threshold"])

    # ── Final Summary ─────────────────────────────────────────────────────
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  INFERENCE SUMMARY")
    print(f"{SEP}")
    print(f"  Model weights trained : {'YES' if is_trained else 'NO (random)'}")
    print(f"  Dataset               : seed_ready_X.npy")
    print(f"  Channels              : {INFERENCE_CFG['in_channels']}")
    print(f"  Time Steps            : 384")
    print(f"  Latent Dimension      : {INFERENCE_CFG['latent_dim']}")
    print(f"  Decision Threshold    : {INFERENCE_CFG['threshold']}")
    print(f"  Device                : {DEVICE}")
    print(f"\n  Test Case Results:")
    print(f"    Random sample [{random_idx}]   -> "
          f"{result['label']}  "
          f"({result['probability']*100:.2f}%)")
    if len(fatigued_indices) > 0:
        print(f"    Known FATIGUED [{fat_idx}] -> "
              f"{result_fat['label']}  "
              f"({result_fat['probability']*100:.2f}%)")
    if len(alert_indices) > 0:
        print(f"    Known ALERT    [{alt_idx}] -> "
              f"{result_alt['label']}  "
              f"({result_alt['probability']*100:.2f}%)")
    print(f"    Simulated EEG       -> "
          f"{result_sim['label']}  "
          f"({result_sim['probability']*100:.2f}%)")
    print(f"\n  Avg Inference Latency : ~{result['latency_ms']:.3f} ms/sample")
    print(SEP)
    print(f"  NeuroDyn-OpNet inference pipeline complete.")
    print(SEP)