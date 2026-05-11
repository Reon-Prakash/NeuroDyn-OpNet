"""
segmentation.py
---------------
Temporal segmentation and sequence preparation for the NeuroDyn-OpNet model.

Pipeline:
  1. Set working directory
  2. Load cleaned EEG arrays
  3. Reshape to 4D tensors  (Samples, Channels, Time_Steps, 1)
  4. Construct State-Pair sequences for dynamic learning
  5. Build SADT subject map for LOSO cross-validation
  6. Validate dimensionality alignment
  7. Save all ready-to-train artefacts

Author : NeuroDyn-OpNet Data Engineering Pipeline
"""

import os
import sys
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 0.  FIX WINDOWS CONSOLE ENCODING  (must happen before any print)
#     Forces stdout/stderr to UTF-8 so Unicode symbols render correctly
#     on all Windows terminals (cmd, PowerShell, VS Code integrated terminal).
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
# 0b.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

WORK_DIR = r"C:\Users\KIIT0001\.vscode\Codes\NeuroDyn-OpNet"

# Expected constants (used for sanity checks)
EXPECTED_TIME_STEPS   = 384
EXPECTED_SEED_SAMPLES = 4566
EXPECTED_SEED_CHANS   = 17
EXPECTED_SADT_SAMPLES = 2022
EXPECTED_SADT_CHANS   = 30
EXPECTED_N_SUBJECTS   = 11

# Reusable check/cross symbols that are safe on all platforms
OK   = "[OK]"
WARN = "[WARN]"
ERR  = "[ERR]"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  SET WORKING DIRECTORY
# ──────────────────────────────────────────────────────────────────────────────

os.chdir(WORK_DIR)
print("=" * 70)
print(f"[INFO] Working directory set to: {os.getcwd()}")
print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def load_npy(filename: str) -> np.ndarray:
    """Load a .npy file from the current working directory with logging."""
    path = os.path.join(os.getcwd(), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{ERR} Required file not found: {path}\n"
            f"       Ensure preprocessing has been completed first."
        )
    arr = np.load(path, allow_pickle=False)
    print(f"  [LOAD] {filename:40s}  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def validate_time_steps(arr: np.ndarray, name: str, axis: int = 2) -> None:
    """Assert that the time-steps dimension equals EXPECTED_TIME_STEPS."""
    t = arr.shape[axis]
    if t != EXPECTED_TIME_STEPS:
        raise ValueError(
            f"{ERR} {name}: expected Time_Steps={EXPECTED_TIME_STEPS} "
            f"on axis {axis}, got {t}."
        )
    print(f"  {OK} {name}: Time_Steps={t} — passed")


def reshape_to_4d(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Reshape  (Samples, Channels, Time_Steps)
          ->  (Samples, Channels, Time_Steps, 1)

    The trailing singleton dimension satisfies the Neural-Operator's
    expectation of a 4-D input tensor.
    """
    reshaped = arr[:, :, :, np.newaxis]
    print(
        f"  [RESHAPE] {name}: "
        f"{arr.shape}  ->  {reshaped.shape}"
    )
    return reshaped


# ──────────────────────────────────────────────────────────────────────────────
# 3.  LOAD RAW CLEANED DATA
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 1] Loading cleaned EEG arrays ...")

seed_x      = load_npy("seed_clean_x.npy")          # (4566, 17, 384)
seed_y      = load_npy("seed_clean_y.npy")           # (4566,)
sadt_x      = load_npy("sadt_clean_x.npy")           # (2022, 30, 384)
sadt_y      = load_npy("sadt_clean_y.npy")           # (2022,)
sadt_subidx = load_npy("sadt_clean_subindex.npy")    # (2022,)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SANITY CHECKS — raw shapes
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 2] Validating raw shapes ...")

# --- SEED-VIG ---
assert seed_x.shape == (EXPECTED_SEED_SAMPLES, EXPECTED_SEED_CHANS, EXPECTED_TIME_STEPS), \
    f"Unexpected SEED-VIG X shape: {seed_x.shape}"
assert seed_y.shape == (EXPECTED_SEED_SAMPLES,), \
    f"Unexpected SEED-VIG Y shape: {seed_y.shape}"

# --- SADT ---
assert sadt_x.shape == (EXPECTED_SADT_SAMPLES, EXPECTED_SADT_CHANS, EXPECTED_TIME_STEPS), \
    f"Unexpected SADT X shape: {sadt_x.shape}"
assert sadt_y.shape == (EXPECTED_SADT_SAMPLES,), \
    f"Unexpected SADT Y shape: {sadt_y.shape}"
assert sadt_subidx.shape == (EXPECTED_SADT_SAMPLES,), \
    f"Unexpected SADT SubIndex shape: {sadt_subidx.shape}"

print(f"  {OK} SEED-VIG X : {seed_x.shape} — passed")
print(f"  {OK} SEED-VIG Y : {seed_y.shape} — passed")
print(f"  {OK} SADT X     : {sadt_x.shape} — passed")
print(f"  {OK} SADT Y     : {sadt_y.shape} — passed")
print(f"  {OK} SADT SubIdx: {sadt_subidx.shape} — passed")
print(f"  {OK} All raw shapes match expected dimensions")

# Verify both datasets share the same temporal resolution
validate_time_steps(seed_x, "SEED-VIG X", axis=2)
validate_time_steps(sadt_x, "SADT X",     axis=2)
print(f"  {OK} Time_Steps dimensionality aligned across datasets")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  RESHAPE TO 4-D TENSORS  (Samples, Channels, Time_Steps, 1)
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Reshaping EEG tensors to 4-D ...")

seed_x_4d = reshape_to_4d(seed_x, "SEED-VIG X")   # (4566, 17, 384, 1)
sadt_x_4d = reshape_to_4d(sadt_x, "SADT X")        # (2022, 30, 384, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CAST LABELS TO int64
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 4] Casting labels to int64 ...")

seed_y_int = seed_y.astype(np.int64)
sadt_y_int = sadt_y.astype(np.int64)

print(f"  [CAST] SEED-VIG Y : {seed_y.dtype} -> int64  |  "
      f"unique labels = {np.unique(seed_y_int).tolist()}")
print(f"  [CAST] SADT Y     : {sadt_y.dtype} -> int64  |  "
      f"unique labels = {np.unique(sadt_y_int).tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  STATE-PAIR SEQUENCE CONSTRUCTION
#
#     NeuroDyn-OpNet learns the brain-state transition:
#         f : x_t  ->  x_{t+1}
#
#     We build (input_sequence, target_sequence) pairs by shifting the
#     sample index by one position along the Samples axis:
#
#         X_in   = samples[0 .. N-2]   <- "current state"
#         X_tgt  = samples[1 .. N-1]   <- "next state"
#
#     Labels are shifted consistently so that Y_in / Y_tgt align.
#
#     The PRIMARY files saved (seed_ready_X / sadt_ready_X) contain the
#     FULL ordered 4-D arrays so that the DataLoader can construct
#     consecutive pairs online during training.  The explicit pair slices
#     are saved as auxiliary files for offline inspection.
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 5] Constructing State-Pair sequences ...")


def build_state_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    name: str
):
    """
    Parameters
    ----------
    X    : (N, C, T, 1)  4-D feature tensor
    Y    : (N,)          int64 label vector

    Returns
    -------
    X_in, X_tgt, Y_in, Y_tgt  — each shifted by one sample index
    """
    N = X.shape[0]
    X_in  = X[: N - 1]     # (N-1, C, T, 1)  current brain state
    X_tgt = X[1 : N]       # (N-1, C, T, 1)  next    brain state
    Y_in  = Y[: N - 1]     # (N-1,)
    Y_tgt = Y[1 : N]       # (N-1,)

    print(f"\n  Dataset            : {name}")
    print(f"    Total samples N            : {N}")
    print(f"    State-pair count (N-1)     : {N - 1}")
    print(f"    X_in  shape (current state): {X_in.shape}")
    print(f"    X_tgt shape (next state)   : {X_tgt.shape}")
    print(f"    Y_in  shape (current label): {Y_in.shape}")
    print(f"    Y_tgt shape (next label)   : {Y_tgt.shape}")

    return X_in, X_tgt, Y_in, Y_tgt


# SEED-VIG
seed_X_in, seed_X_tgt, seed_Y_in, seed_Y_tgt = build_state_pairs(
    seed_x_4d, seed_y_int, "SEED-VIG"
)

# SADT
sadt_X_in, sadt_X_tgt, sadt_Y_in, sadt_Y_tgt = build_state_pairs(
    sadt_x_4d, sadt_y_int, "SADT"
)

# Primary "ready" tensors use the FULL ordered arrays; consecutive-pair
# sampling is done online by the NeuroDyn-OpNet DataLoader.
seed_ready_X = seed_x_4d       # (4566, 17, 384, 1)
seed_ready_Y = seed_y_int       # (4566,)
sadt_ready_X = sadt_x_4d        # (2022, 30, 384, 1)
sadt_ready_Y = sadt_y_int        # (2022,)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  SADT SUBJECT MAP  (LOSO Cross-Validation)
#
#     Output: sadt_subject_map.npy
#       - shape  : (2022,)  int64
#       - meaning: element i  = subject ID that sample i belongs to
#       - range  : 1 .. 11  (SADT subjects)
#
#     Also prints a per-subject sample-count table for verification.
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 6] Building SADT Subject Map for LOSO ...")

raw_subjects    = sadt_subidx.astype(np.int64)
unique_subjects = np.unique(raw_subjects)

print(f"\n  Unique subject IDs : {unique_subjects.tolist()}")
print(f"  Total subjects     : {len(unique_subjects)}")

if len(unique_subjects) != EXPECTED_N_SUBJECTS:
    print(
        f"  {WARN} Expected {EXPECTED_N_SUBJECTS} subjects, "
        f"found {len(unique_subjects)}. Proceeding."
    )

# Per-subject index dictionary
subject_to_indices: dict = {
    int(sid): np.where(raw_subjects == sid)[0].tolist()
    for sid in unique_subjects
}

# Pretty table
print()
print(f"  {'Subject ID':>12}  |  {'Sample Count':>12}  |  Index Range")
print("  " + "-" * 52)
for sid, idxs in subject_to_indices.items():
    idx_arr = np.array(idxs)
    print(
        f"  {'Subject ' + str(sid):>12}  |  {len(idxs):>12}  |"
        f"  [{idx_arr.min()}, {idx_arr.max()}]"
    )

# The subject map is a simple int64 array, index-aligned to SADT samples
sadt_subject_map = raw_subjects.copy()   # (2022,) int64

print()
print(f"  [MAP] sadt_subject_map shape  : {sadt_subject_map.shape}")
print(f"  [MAP] sadt_subject_map dtype  : {sadt_subject_map.dtype}")
print(f"  [MAP] First 11 entries        : {sadt_subject_map[:11].tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  FINAL SHAPE VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 7] Final shape verification ...")

# 4-D structure
assert seed_ready_X.ndim == 4,        "seed_ready_X must be 4-D"
assert sadt_ready_X.ndim == 4,        "sadt_ready_X must be 4-D"
assert seed_ready_X.shape[-1] == 1,   "seed_ready_X trailing dim must be 1"
assert sadt_ready_X.shape[-1] == 1,   "sadt_ready_X trailing dim must be 1"
print(f"  {OK} Both feature tensors are 4-D with trailing singleton dim=1")

# Time_Steps cross-dataset alignment
assert seed_ready_X.shape[2] == sadt_ready_X.shape[2], \
    "Time_Steps mismatch between SEED-VIG and SADT in final tensors!"
print(f"  {OK} Time_Steps consistent across datasets "
      f"(T={seed_ready_X.shape[2]})")

# Label dtype
assert seed_ready_Y.dtype == np.int64, "SEED-VIG labels must be int64"
assert sadt_ready_Y.dtype == np.int64, "SADT labels must be int64"
print(f"  {OK} All label arrays are int64")

# Subject map
assert sadt_subject_map.dtype == np.int64, "Subject map must be int64"
assert sadt_subject_map.shape[0] == EXPECTED_SADT_SAMPLES, \
    "Subject map length must equal number of SADT samples"
print(f"  {OK} sadt_subject_map is int64 and index-aligned to SADT samples")

# Full shape table
SEP = "-" * 70
print()
print(SEP)
print(f"  {'Tensor':<28}  {'Shape':<26}  {'Dtype'}")
print(SEP)
print(f"  {'seed_ready_X':<28}  {str(seed_ready_X.shape):<26}  {seed_ready_X.dtype}")
print(f"  {'seed_ready_Y':<28}  {str(seed_ready_Y.shape):<26}  {seed_ready_Y.dtype}")
print(f"  {'sadt_ready_X':<28}  {str(sadt_ready_X.shape):<26}  {sadt_ready_X.dtype}")
print(f"  {'sadt_ready_Y':<28}  {str(sadt_ready_Y.shape):<26}  {sadt_ready_Y.dtype}")
print(f"  {'sadt_subject_map':<28}  {str(sadt_subject_map.shape):<26}  {sadt_subject_map.dtype}")
print(SEP)
print(f"  {'[AUX] seed_pair_X_in':<28}  {str(seed_X_in.shape):<26}  {seed_X_in.dtype}")
print(f"  {'[AUX] seed_pair_X_tgt':<28}  {str(seed_X_tgt.shape):<26}  {seed_X_tgt.dtype}")
print(f"  {'[AUX] sadt_pair_X_in':<28}  {str(sadt_X_in.shape):<26}  {sadt_X_in.dtype}")
print(f"  {'[AUX] sadt_pair_X_tgt':<28}  {str(sadt_X_tgt.shape):<26}  {sadt_X_tgt.dtype}")
print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# 10. SAVE ALL ARTEFACTS
# ──────────────────────────────────────────────────────────────────────────────

print("\n[STEP 8] Saving output artefacts ...")

save_spec: dict = {
    # ---- primary files (consumed by training loop) ----
    "seed_ready_X.npy"     : seed_ready_X,
    "seed_ready_Y.npy"     : seed_ready_Y,
    "sadt_ready_X.npy"     : sadt_ready_X,
    "sadt_ready_Y.npy"     : sadt_ready_Y,
    "sadt_subject_map.npy" : sadt_subject_map,
    # ---- auxiliary state-pair slices (offline inspection) ----
    "seed_pair_X_in.npy"   : seed_X_in,
    "seed_pair_X_tgt.npy"  : seed_X_tgt,
    "seed_pair_Y_in.npy"   : seed_Y_in,
    "seed_pair_Y_tgt.npy"  : seed_Y_tgt,
    "sadt_pair_X_in.npy"   : sadt_X_in,
    "sadt_pair_X_tgt.npy"  : sadt_X_tgt,
    "sadt_pair_Y_in.npy"   : sadt_Y_in,
    "sadt_pair_Y_tgt.npy"  : sadt_Y_tgt,
}

total_bytes = 0
for fname, arr in save_spec.items():
    out_path  = os.path.join(os.getcwd(), fname)
    np.save(out_path, arr)
    size_mb    = arr.nbytes / (1024 ** 2)
    total_bytes += arr.nbytes
    print(f"  [SAVE] {fname:30s}  shape={str(arr.shape):<22s}  "
          f"dtype={str(arr.dtype):<10}  {size_mb:8.2f} MB")

print(f"\n  Total disk footprint : {total_bytes / (1024**2):.2f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# 11. SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  SEGMENTATION & SEQUENCE PREPARATION — COMPLETE")
print("=" * 70)
print()
print("  Primary outputs (consumed by NeuroDyn-OpNet training loop):")
rows = [
    ("seed_ready_X.npy",     seed_ready_X.shape,     seed_ready_X.dtype),
    ("seed_ready_Y.npy",     seed_ready_Y.shape,     seed_ready_Y.dtype),
    ("sadt_ready_X.npy",     sadt_ready_X.shape,     sadt_ready_X.dtype),
    ("sadt_ready_Y.npy",     sadt_ready_Y.shape,     sadt_ready_Y.dtype),
    ("sadt_subject_map.npy", sadt_subject_map.shape, sadt_subject_map.dtype),
]
for fname, shp, dt in rows:
    print(f"    {fname:<25}  shape={str(shp):<24}  dtype={dt}")

print()
print("  State-Pair convention:")
print("    Input  -> samples[0 .. N-2]   (current brain state)")
print("    Target -> samples[1 .. N-1]   (next brain state)")
print("    Online pair construction is performed by the DataLoader at")
print("    training time via consecutive index sampling.")
print()
print("  LOSO cross-validation subject IDs in sadt_subject_map:")
print(f"    {unique_subjects.tolist()}")
print()
print("  Channel note (datasets kept separate):")
print(f"    SEED-VIG  ->  {EXPECTED_SEED_CHANS} channels")
print(f"    SADT      ->  {EXPECTED_SADT_CHANS} channels")
print(f"    Time_Steps -> {EXPECTED_TIME_STEPS} (aligned across both datasets)")
print()
print("  All files written to:")
print(f"    {os.getcwd()}")
print("=" * 70)