"""
dataloader.py
-------------
Custom PyTorch Dataset and DataLoader for NeuroDyn-OpNet.

Handles:
    - SEED-VIG dataset (full dataset split)
    - SADT dataset     (LOSO cross-validation via subject map)

Each sample returns a triplet:
    x_t    : (C, T, 1)  float32  EEG segment at index i
    y_t    : scalar     float32  label at index i
    y_next : scalar     float32  label at index i+1  (next state target)

Author: NeuroDyn-OpNet Data Engineering Pipeline
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

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

print("=" * 65)
print(f"[INFO] Working directory : {os.getcwd()}")
print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FILE LOADING UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def load_npy(filename: str) -> np.ndarray:
    """Load a .npy file from ROOT_PATH with shape and dtype logging."""
    path = os.path.join(ROOT_PATH, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[ERROR] File not found: {path}\n"
            f"        Run segmentation.py first to generate ready files."
        )
    arr = np.load(path, allow_pickle=False)
    print(f"  [LOAD] {filename:<30}  shape={str(arr.shape):<26} dtype={arr.dtype}")
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BASE EEG DATASET
# ──────────────────────────────────────────────────────────────────────────────

class EEGStateDataset(Dataset):
    """
    Base EEG Dataset for NeuroDyn-OpNet.

    Returns triplets (x_t, y_t, y_next) for temporal consistency training.

    Parameters
    ----------
    X           : np.ndarray  shape (N, C, T, 1)  float64 EEG data
    Y           : np.ndarray  shape (N,)           int64  labels
    augment     : bool        Apply Gaussian noise augmentation
    noise_std   : float       Std of Gaussian noise added to x_t
    indices     : list/array  Optional subset of indices to use
                              (used internally by LOSO splitter)
    """

    def __init__(
        self,
        X:          np.ndarray,
        Y:          np.ndarray,
        augment:    bool  = False,
        noise_std:  float = 0.01,
        indices:    np.ndarray = None
    ):
        super().__init__()

        # Apply index subset if provided (LOSO splitting)
        if indices is not None:
            X = X[indices]
            Y = Y[indices]

        # Store as float32 tensors
        self.X         = torch.tensor(X, dtype=torch.float32)
        self.Y         = torch.tensor(Y, dtype=torch.float32)
        self.augment   = augment
        self.noise_std = noise_std
        self.N         = len(self.X)

    def __len__(self) -> int:
        # Last sample has no valid y_next partner — exclude it
        return self.N - 1

    def __getitem__(self, i: int):
        """
        Returns
        -------
        x_t    : (C, T, 1)  float32   EEG sample at index i
        y_t    : scalar     float32   label at index i
        y_next : scalar     float32   label at index i+1

        Boundary:
            i is always in [0, N-2] due to __len__ = N-1
            so i+1 is always valid — no explicit boundary check needed.
            The last sample (index N-1) is never returned as x_t;
            it only appears as y_next for i = N-2.
        """
        x_t    = self.X[i]          # (C, T, 1)
        y_t    = self.Y[i]          # scalar
        y_next = self.Y[i + 1]      # scalar  (always safe: i <= N-2)

        # Data augmentation: Gaussian noise on training samples only
        if self.augment:
            noise = torch.randn_like(x_t) * self.noise_std
            x_t   = x_t + noise

        return x_t, y_t, y_next


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SEED-VIG DATASET FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def get_seed_dataloaders(
    train_ratio:  float = 0.8,
    batch_size:   int   = 32,
    num_workers:  int   = 0,
    noise_std:    float = 0.01,
    seed:         int   = 42
) -> tuple:
    """
    Create train/test DataLoaders for SEED-VIG using random split.

    SEED-VIG has no subject metadata, so we use a fixed random split.

    Parameters
    ----------
    train_ratio : float   Fraction of data for training
    batch_size  : int     DataLoader batch size
    num_workers : int     DataLoader worker processes
    noise_std   : float   Gaussian noise std for training augmentation
    seed        : int     Random seed for reproducibility

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """
    print("\n[SEED-VIG] Building DataLoaders ...")

    seed_X = load_npy("seed_ready_X.npy")   # (4566, 17, 384, 1)
    seed_Y = load_npy("seed_ready_Y.npy")   # (4566,)

    N          = len(seed_X)
    rng        = np.random.default_rng(seed)
    all_idx    = np.arange(N)
    rng.shuffle(all_idx)

    n_train    = int(N * train_ratio)
    train_idx  = all_idx[:n_train]
    test_idx   = all_idx[n_train:]

    train_dataset = EEGStateDataset(
        seed_X, seed_Y,
        augment   = True,
        noise_std = noise_std,
        indices   = train_idx
    )
    test_dataset  = EEGStateDataset(
        seed_X, seed_Y,
        augment   = False,
        noise_std = noise_std,
        indices   = test_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False
    )

    print(f"\n  [OK] SEED-VIG Split:")
    print(f"       Train samples : {len(train_dataset)}")
    print(f"       Test  samples : {len(test_dataset)}")
    print(f"       Train batches : {len(train_loader)}")
    print(f"       Test  batches : {len(test_loader)}")

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SADT LOSO DATASET FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def get_sadt_loso_dataloaders(
    test_subject_id: int,
    batch_size:      int   = 32,
    num_workers:     int   = 0,
    noise_std:       float = 0.01
) -> tuple:
    """
    Leave-One-Subject-Out (LOSO) DataLoaders for SADT.

    Parameters
    ----------
    test_subject_id : int   Subject ID to hold out (1-11)
    batch_size      : int   DataLoader batch size
    num_workers     : int   DataLoader worker processes
    noise_std       : float Gaussian noise std for training augmentation

    Returns
    -------
    train_loader : DataLoader   (all subjects except test_subject_id)
    test_loader  : DataLoader   (only test_subject_id)
    subject_info : dict         Metadata about the split
    """
    print(f"\n[SADT LOSO] Building DataLoaders for Subject {test_subject_id} ...")

    sadt_X      = load_npy("sadt_ready_X.npy")          # (2022, 30, 384, 1)
    sadt_Y      = load_npy("sadt_ready_Y.npy")          # (2022,)
    subject_map = load_npy("sadt_subject_map.npy")      # (2022,)

    # Validate subject ID
    valid_subjects = np.unique(subject_map)
    if test_subject_id not in valid_subjects:
        raise ValueError(
            f"[ERROR] test_subject_id={test_subject_id} not found.\n"
            f"        Valid IDs: {valid_subjects.tolist()}"
        )

    # Index split by subject
    test_mask  = subject_map == test_subject_id
    train_mask = ~test_mask

    train_idx  = np.where(train_mask)[0]
    test_idx   = np.where(test_mask)[0]

    # Build datasets with subject-specific indices
    train_dataset = EEGStateDataset(
        sadt_X, sadt_Y,
        augment   = True,
        noise_std = noise_std,
        indices   = train_idx
    )
    test_dataset  = EEGStateDataset(
        sadt_X, sadt_Y,
        augment   = False,
        noise_std = noise_std,
        indices   = test_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False
    )

    subject_info = {
        "test_subject"      : test_subject_id,
        "train_subjects"    : [s for s in valid_subjects if s != test_subject_id],
        "train_samples"     : len(train_dataset),
        "test_samples"      : len(test_dataset),
        "train_batches"     : len(train_loader),
        "test_batches"      : len(test_loader),
        "train_label_dist"  : {
            int(k): int(v) for k, v in
            zip(*np.unique(sadt_Y[train_idx], return_counts=True))
        },
        "test_label_dist"   : {
            int(k): int(v) for k, v in
            zip(*np.unique(sadt_Y[test_idx], return_counts=True))
        },
    }

    print(f"\n  [OK] SADT LOSO Split:")
    print(f"       Test Subject    : {test_subject_id}")
    print(f"       Train Subjects  : {subject_info['train_subjects']}")
    print(f"       Train samples   : {subject_info['train_samples']}")
    print(f"       Test  samples   : {subject_info['test_samples']}")
    print(f"       Train batches   : {subject_info['train_batches']}")
    print(f"       Test  batches   : {subject_info['test_batches']}")
    print(f"       Train label dist: {subject_info['train_label_dist']}")
    print(f"       Test  label dist: {subject_info['test_label_dist']}")

    return train_loader, test_loader, subject_info


# ──────────────────────────────────────────────────────────────────────────────
# 6.  BATCH INSPECTION UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def inspect_batch(loader: DataLoader, loader_name: str = "Loader") -> None:
    """
    Pull one batch and print a detailed inspection.

    Verifies:
        - x_t  shape and dtype
        - y_t  shape and dtype
        - y_next shape and dtype
        - label shift correctness (y_t[i] != y_next[i] sometimes)
        - value ranges
    """
    SEP = "-" * 65

    print(f"\n  [{loader_name}] Batch Inspection")
    print(SEP)

    batch = next(iter(loader))
    x_t, y_t, y_next = batch

    print(f"  x_t    shape  : {x_t.shape}")
    print(f"  y_t    shape  : {y_t.shape}")
    print(f"  y_next shape  : {y_next.shape}")
    print()
    print(f"  x_t    dtype  : {x_t.dtype}")
    print(f"  y_t    dtype  : {y_t.dtype}")
    print(f"  y_next dtype  : {y_next.dtype}")
    print()
    print(f"  x_t    range  : [{x_t.min().item():.4f}, {x_t.max().item():.4f}]")
    print(f"  y_t    unique : {y_t.unique().tolist()}")
    print(f"  y_next unique : {y_next.unique().tolist()}")
    print()

    # Print first 10 label pairs to verify temporal shift
    print(f"  Label shift verification (first 12 pairs):")
    print(f"  {'idx':<6}  {'y_t':>6}  {'y_next':>8}  {'shifted?':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}")
    for i in range(min(12, len(y_t))):
        yt_val    = int(y_t[i].item())
        yn_val    = int(y_next[i].item())
        shifted   = "YES" if yt_val != yn_val else "---"
        print(f"  {i:<6}  {yt_val:>6}  {yn_val:>8}  {shifted:>10}")

    # Count how many pairs differ (proves next-state labeling works)
    n_different = (y_t != y_next).sum().item()
    n_total     = len(y_t)
    print(f"\n  Pairs where y_t != y_next : {n_different} / {n_total}")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VERIFICATION TEST BLOCK
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BATCH_SIZE = 32

    print("\n" + "=" * 65)
    print("  DataLoader Verification — NeuroDyn-OpNet")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # Test 1: SEED-VIG DataLoaders                                        #
    # ------------------------------------------------------------------ #
    print("\n[TEST 1]  SEED-VIG DataLoaders")

    seed_train_loader, seed_test_loader = get_seed_dataloaders(
        train_ratio = 0.8,
        batch_size  = BATCH_SIZE,
        num_workers = 0,
        noise_std   = 0.01,
        seed        = 42
    )

    inspect_batch(seed_train_loader, "SEED-VIG Train")
    inspect_batch(seed_test_loader,  "SEED-VIG Test")

    # ------------------------------------------------------------------ #
    # Test 2: SADT LOSO DataLoaders  (test on Subject 1)                  #
    # ------------------------------------------------------------------ #
    print("\n[TEST 2]  SADT LOSO DataLoaders  (test_subject=1)")

    sadt_train_loader, sadt_test_loader, info = get_sadt_loso_dataloaders(
        test_subject_id = 1,
        batch_size      = BATCH_SIZE,
        num_workers     = 0,
        noise_std       = 0.01
    )

    inspect_batch(sadt_train_loader, "SADT Train (excl. Subj 1)")
    inspect_batch(sadt_test_loader,  "SADT Test  (only Subj 1)")

    # ------------------------------------------------------------------ #
    # Test 3: Full LOSO Loop Preview                                      #
    # ------------------------------------------------------------------ #
    print("\n[TEST 3]  Full LOSO Loop — All 11 Subjects Preview")
    print("-" * 65)
    print(f"  {'Subject':<10}  {'Train N':>9}  {'Test N':>9}  "
          f"{'Train B':>9}  {'Test B':>9}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    for subj_id in range(1, 12):
        tr_loader, te_loader, meta = get_sadt_loso_dataloaders(
            test_subject_id = subj_id,
            batch_size      = BATCH_SIZE,
            num_workers     = 0,
            noise_std       = 0.01
        )
        print(f"  {('Subject ' + str(subj_id)):<10}  "
              f"{meta['train_samples']:>9}  "
              f"{meta['test_samples']:>9}  "
              f"{meta['train_batches']:>9}  "
              f"{meta['test_batches']:>9}")

    # ------------------------------------------------------------------ #
    # Test 4: dtype and shape assertion                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[TEST 4]  dtype and Shape Assertions")
    print("-" * 65)

    x_t, y_t, y_next = next(iter(seed_train_loader))

    assert x_t.dtype    == torch.float32, f"x_t must be float32, got {x_t.dtype}"
    assert y_t.dtype    == torch.float32, f"y_t must be float32, got {y_t.dtype}"
    assert y_next.dtype == torch.float32, f"y_next must be float32, got {y_next.dtype}"
    assert x_t.shape[1] == 17,            f"SEED-VIG must have 17 channels"
    assert x_t.shape[2] == 384,           f"Time steps must be 384"
    assert x_t.shape[3] == 1,             f"Trailing dim must be 1"
    assert y_t.shape    == (BATCH_SIZE,),  f"y_t must be (B,), got {y_t.shape}"
    assert y_next.shape == (BATCH_SIZE,),  f"y_next must be (B,), got {y_next.shape}"

    print(f"  [OK] x_t    dtype={x_t.dtype}   shape={x_t.shape}")
    print(f"  [OK] y_t    dtype={y_t.dtype}  shape={y_t.shape}")
    print(f"  [OK] y_next dtype={y_next.dtype}  shape={y_next.shape}")
    print(f"  [OK] All dtype and shape assertions passed")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("  DataLoader Summary")
    print("=" * 65)
    print(f"  SEED-VIG  Train batches : {len(seed_train_loader)}")
    print(f"  SEED-VIG  Test  batches : {len(seed_test_loader)}")
    print(f"  SADT      Train batches : {len(sadt_train_loader)}")
    print(f"  SADT      Test  batches : {len(sadt_test_loader)}")
    print(f"\n  Triplet per sample:")
    print(f"    x_t    : (B, C, 384, 1)  float32  EEG at time t")
    print(f"    y_t    : (B,)            float32  label at time t")
    print(f"    y_next : (B,)            float32  label at time t+1")
    print(f"\n  Augmentation: Gaussian noise std=0.01 on train only")
    print(f"  LOSO splits : 11 subjects (1-11) available")
    print("=" * 65)
    print("  DataLoader pipeline ready for NeuroDyn-OpNet training")
    print("=" * 65)