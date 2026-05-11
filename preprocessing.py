"""
EEG Preprocessing Pipeline for Driver Fatigue Detection
Supports: SEED-VIG and SADT datasets
Author: Biomedical Engineer / Python Developer
Libraries: MNE, SciPy, NumPy

PATCH v1.1 — Fixed SADT variable names to match actual .mat file structure.
             dataset.mat uses SEED-VIG naming convention:
             EEGsample / substate / subindex
"""

import numpy as np
import scipy.io
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
import mne
from mne.filter import resample
import warnings
import logging

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_seed_vig(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load SEED-VIG dataset from a .mat file.

    Parameters
    ----------
    filepath : str
        Path to the .mat file containing SEED-VIG data.

    Returns
    -------
    eeg_data : np.ndarray
        Raw EEG array of shape (Samples, Channels, Time_Steps).
    labels : np.ndarray
        Label array of shape (Samples,).

    Raises
    ------
    KeyError
        If expected variables are missing from the .mat file.
    ValueError
        If the data dimensions are not as expected.
    """
    logger.info(f"Loading SEED-VIG dataset from: {filepath}")
    mat = scipy.io.loadmat(filepath)

    # ── Validate expected keys ────────────────────────────────────────────────
    if "EEGsample" not in mat or "substate" not in mat:
        raise KeyError(
            "Expected keys 'EEGsample' and 'substate' not found. "
            f"Available keys: {[k for k in mat.keys() if not k.startswith('_')]}"
        )

    eeg_data = mat["EEGsample"]   # Shape: (Samples, Channels, Time_Steps)
    labels   = mat["substate"]    # Shape: (Samples, 1) or (Samples,)
    labels   = labels.squeeze()

    if eeg_data.ndim != 3:
        raise ValueError(
            f"EEGsample must be 3D (Samples, Channels, Time_Steps). "
            f"Got shape: {eeg_data.shape}"
        )

    logger.info(
        f"SEED-VIG loaded | EEG shape: {eeg_data.shape} | "
        f"Labels shape: {labels.shape} | Unique labels: {np.unique(labels)}"
    )
    return eeg_data.astype(np.float64), labels


def load_sadt(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the second EEG dataset (dataset.mat) which uses SEED-VIG variable
    naming conventions.

    IMPORTANT — Variable Name Correction (v1.1):
    ┌──────────────────────────────────────────────────────────────┐
    │  The file dataset.mat does NOT use 'dataset' / 'labels'.    │
    │  Actual keys discovered at runtime:                          │
    │    • EEGsample  → Raw EEG data                              │
    │    • substate   → Fatigue labels (0 / 1)                    │
    │    • subindex   → Subject index metadata (loaded, returned) │
    └──────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    filepath : str
        Path to the dataset.mat file.

    Returns
    -------
    eeg_data : np.ndarray
        Raw EEG array of shape (Samples, Channels, Time_Steps).
    labels : np.ndarray
        Fatigue label array of shape (Samples,).
    subindex : np.ndarray
        Subject index array of shape (Samples,). Useful for
        subject-independent cross-validation in later steps.

    Raises
    ------
    KeyError
        If expected variables are missing from the .mat file.
    ValueError
        If the data dimensions are not as expected.
    """
    logger.info(f"Loading dataset.mat (SADT) from: {filepath}")
    mat = scipy.io.loadmat(filepath)

    # ── Log all discovered keys for full transparency ─────────────────────────
    discovered_keys = [k for k in mat.keys() if not k.startswith('_')]
    logger.info(f"Keys found in dataset.mat: {discovered_keys}")

    # ── Validate corrected key names ──────────────────────────────────────────
    required_keys = ["EEGsample", "substate"]
    missing = [k for k in required_keys if k not in mat]
    if missing:
        raise KeyError(
            f"Required keys {missing} not found in dataset.mat. "
            f"Available keys: {discovered_keys}\n"
            "  → Update required_keys in load_sadt() to match your file."
        )

    # ── Extract arrays with CORRECTED variable names ──────────────────────────
    eeg_data = mat["EEGsample"]   # ← FIXED: was mat["dataset"]
    labels   = mat["substate"]    # ← FIXED: was mat["labels"]
    labels   = labels.squeeze()

    # ── Extract subindex if present (bonus metadata) ──────────────────────────
    if "subindex" in mat:
        subindex = mat["subindex"].squeeze()
        logger.info(
            f"subindex loaded | Shape: {subindex.shape} | "
            f"Unique subjects: {np.unique(subindex)}"
        )
    else:
        # Create a dummy subindex array of zeros if not present
        subindex = np.zeros(labels.shape[0], dtype=np.int32)
        logger.warning("'subindex' not found — filled with zeros as placeholder.")

    # ── Validate 3D structure ─────────────────────────────────────────────────
    if eeg_data.ndim != 3:
        raise ValueError(
            f"EEGsample must be 3D (Samples, Channels, Time_Steps). "
            f"Got shape: {eeg_data.shape}"
        )

    logger.info(
        f"dataset.mat loaded | EEG shape: {eeg_data.shape} | "
        f"Labels shape: {labels.shape} | Unique labels: {np.unique(labels)}"
    )
    return eeg_data.astype(np.float64), labels, subindex


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: NaN / OUTLIER HANDLING
# ═════════════════════════════════════════════════════════════════════════════

def handle_nan_and_outliers(
    data: np.ndarray,
    voltage_threshold: float = 100.0
) -> np.ndarray:
    """
    Replace NaN values with per-channel mean and clip extreme voltage spikes.

    Processing Order:
        1. Replace NaN  → compute honest channel mean before clipping.
        2. Clip outliers → ±voltage_threshold µV.

    Parameters
    ----------
    data : np.ndarray
        EEG array of shape (Samples, Channels, Time_Steps).
    voltage_threshold : float
        Clip threshold in µV. Default is 100 µV.

    Returns
    -------
    data_clean : np.ndarray
        Cleaned EEG array, same shape as input.
    """
    logger.info(
        f"Handling NaN / outliers | Shape: {data.shape} | "
        f"Threshold: ±{voltage_threshold} µV"
    )
    data_clean = data.copy()
    n_samples, n_channels, n_times = data_clean.shape
    total_nan = 0

    for s in range(n_samples):
        for c in range(n_channels):
            channel_signal = data_clean[s, c, :]
            nan_mask = np.isnan(channel_signal)
            n_nan = nan_mask.sum()

            if n_nan > 0:
                total_nan += n_nan
                channel_mean = np.nanmean(channel_signal)
                if np.isnan(channel_mean):
                    logger.warning(
                        f"Sample {s}, Channel {c}: entirely NaN. Filling with 0."
                    )
                    channel_mean = 0.0
                data_clean[s, c, nan_mask] = channel_mean

    logger.info(f"NaN values replaced: {total_nan} total across all samples/channels")

    pre_clip_extremes = np.sum(np.abs(data_clean) > voltage_threshold)
    data_clean = np.clip(data_clean, -voltage_threshold, voltage_threshold)
    logger.info(f"Clipped {pre_clip_extremes} data points beyond ±{voltage_threshold} µV")

    return data_clean


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: BAND-PASS FILTERING  (0.5 Hz – 45 Hz, Butterworth Order 4)
# ═════════════════════════════════════════════════════════════════════════════

def design_bandpass_filter(
    lowcut: float,
    highcut: float,
    sfreq: float,
    order: int = 4
) -> np.ndarray:
    """
    Design a zero-phase Butterworth band-pass filter in SOS format.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Filter order. Default is 4.

    Returns
    -------
    sos : np.ndarray
        Second-Order Sections filter coefficients.
    """
    nyquist = sfreq / 2.0
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(
            f"Frequencies must satisfy 0 < lowcut < highcut < Nyquist ({nyquist} Hz). "
            f"Got lowcut={lowcut}, highcut={highcut}."
        )
    low  = lowcut  / nyquist
    high = highcut / nyquist
    sos  = butter(order, [low, high], btype="band", output="sos")
    return sos


def apply_bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth band-pass filter to EEG data.

    Parameters
    ----------
    data : np.ndarray
        EEG array of shape (Samples, Channels, Time_Steps).
    sfreq : float
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    data_filtered : np.ndarray
        Band-pass filtered EEG, same shape as input.
    """
    logger.info(
        f"Applying band-pass filter | {lowcut}–{highcut} Hz | "
        f"Order: {order} | sfreq: {sfreq} Hz"
    )
    sos = design_bandpass_filter(lowcut, highcut, sfreq, order)
    n_samples, n_channels, n_times = data.shape

    min_length = 3 * (2 * order + 1)
    if n_times <= min_length:
        raise ValueError(
            f"Time dimension ({n_times}) too short for filter order {order}. "
            f"Minimum required: {min_length + 1} time points."
        )

    data_filtered = np.zeros_like(data)
    for s in range(n_samples):
        for c in range(n_channels):
            data_filtered[s, c, :] = sosfiltfilt(sos, data[s, c, :])

    logger.info(f"Band-pass filtering complete | Output shape: {data_filtered.shape}")
    return data_filtered


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: NOTCH FILTERING  (50 Hz power-line interference)
# ═════════════════════════════════════════════════════════════════════════════

def apply_notch_filter(
    data: np.ndarray,
    sfreq: float,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply a notch filter to remove power-line interference.

    Parameters
    ----------
    data : np.ndarray
        EEG array of shape (Samples, Channels, Time_Steps).
    sfreq : float
        Sampling frequency in Hz.
    notch_freq : float
        Frequency to attenuate in Hz. Default: 50 Hz.
    quality_factor : float
        Quality factor controlling notch bandwidth. Default: 30.0.

    Returns
    -------
    data_notched : np.ndarray
        Notch-filtered EEG, same shape as input.
    """
    logger.info(
        f"Applying notch filter | {notch_freq} Hz | "
        f"Q={quality_factor} | sfreq: {sfreq} Hz"
    )
    nyquist = sfreq / 2.0
    if notch_freq >= nyquist:
        logger.warning(
            f"Notch frequency ({notch_freq} Hz) ≥ Nyquist ({nyquist} Hz). "
            "Skipping notch filter."
        )
        return data.copy()

    b, a = iirnotch(notch_freq, quality_factor, fs=sfreq)
    data_notched = np.zeros_like(data)

    for s in range(data.shape[0]):
        for c in range(data.shape[1]):
            data_notched[s, c, :] = filtfilt(b, a, data[s, c, :])

    logger.info(f"Notch filtering complete | Output shape: {data_notched.shape}")
    return data_notched


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: RESAMPLING  (via MNE for anti-aliasing)
# ═════════════════════════════════════════════════════════════════════════════

def apply_resampling(
    data: np.ndarray,
    original_sfreq: float,
    target_sfreq: float
) -> np.ndarray:
    """
    Resample EEG data to target sampling frequency using MNE's resampler.

    Parameters
    ----------
    data : np.ndarray
        EEG array of shape (Samples, Channels, Time_Steps).
    original_sfreq : float
        Original sampling frequency in Hz.
    target_sfreq : float
        Target sampling frequency in Hz.

    Returns
    -------
    data_resampled : np.ndarray
        Resampled EEG array of shape (Samples, Channels, New_Time_Steps).
    """
    if original_sfreq == target_sfreq:
        logger.info(
            f"Resampling skipped | Already at target frequency: {target_sfreq} Hz"
        )
        return data.copy()

    logger.info(
        f"Resampling | {original_sfreq} Hz → {target_sfreq} Hz | "
        f"Input shape: {data.shape}"
    )
    expected_times    = int(np.round(data.shape[2] * target_sfreq / original_sfreq))
    data_resampled_list = []

    for s in range(data.shape[0]):
        resampled_sample = mne.filter.resample(
            data[s],
            up=target_sfreq,
            down=original_sfreq,
            npad="auto",
            verbose=False
        )
        data_resampled_list.append(resampled_sample)

    data_resampled = np.stack(data_resampled_list, axis=0)
    logger.info(
        f"Resampling complete | Output shape: {data_resampled.shape} | "
        f"Expected time steps: ~{expected_times}"
    )
    return data_resampled


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Z-SCORE NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def apply_zscore_normalization(
    data: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Apply Z-score normalization per channel across the time dimension.

    Parameters
    ----------
    data : np.ndarray
        EEG array of shape (Samples, Channels, Time_Steps).
    epsilon : float
        Numerical stability constant. Default: 1e-8.

    Returns
    -------
    data_normalized : np.ndarray
        Z-score normalized EEG, same shape as input.
    """
    logger.info(f"Applying Z-score normalization | Shape: {data.shape}")
    data_normalized    = np.zeros_like(data)
    flat_channel_count = 0

    for s in range(data.shape[0]):
        for c in range(data.shape[1]):
            signal = data[s, c, :]
            mean   = signal.mean()
            std    = signal.std()
            if std < epsilon:
                flat_channel_count += 1
            data_normalized[s, c, :] = (signal - mean) / (std + epsilon)

    if flat_channel_count > 0:
        logger.warning(
            f"{flat_channel_count} channel(s) had near-zero std — "
            "normalized with epsilon stabilization."
        )

    sample_means = data_normalized.mean(axis=2)
    sample_stds  = data_normalized.std(axis=2)
    logger.info(
        f"Normalization verification | "
        f"Mean range: [{sample_means.min():.4f}, {sample_means.max():.4f}] | "
        f"Std range:  [{sample_stds.min():.4f},  {sample_stds.max():.4f}]"
    )
    return data_normalized


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: MASTER PREPROCESSING FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_eeg(
    data: np.ndarray,
    sfreq: float,
    target_sfreq: float,
    bandpass_low: float       = 0.5,
    bandpass_high: float      = 45.0,
    filter_order: int         = 4,
    notch_freq: float         = 50.0,
    notch_quality: float      = 30.0,
    voltage_threshold: float  = 100.0
) -> np.ndarray:
    """
    Master EEG preprocessing pipeline.

    Pipeline order:
        1. NaN / Outlier Handling
        2. Band-pass Filtering  (0.5–45 Hz, Butterworth order 4)
        3. Notch Filtering      (50 Hz)
        4. Resampling           (via MNE anti-aliased)
        5. Z-Score Normalization

    Parameters
    ----------
    data : np.ndarray
        Raw EEG of shape (Samples, Channels, Time_Steps).
    sfreq : float
        Original sampling frequency in Hz.
    target_sfreq : float
        Target sampling frequency in Hz.
    bandpass_low : float
        Lower cutoff frequency in Hz.
    bandpass_high : float
        Upper cutoff frequency in Hz.
    filter_order : int
        Butterworth filter order.
    notch_freq : float
        Notch filter target frequency in Hz.
    notch_quality : float
        Notch filter quality factor.
    voltage_threshold : float
        Outlier clipping threshold in µV.

    Returns
    -------
    data : np.ndarray
        Preprocessed EEG, float64, shape (Samples, Channels, New_Time_Steps).
    """
    logger.info("=" * 65)
    logger.info("STARTING EEG PREPROCESSING PIPELINE")
    logger.info(f"Input shape    : {data.shape}")
    logger.info(f"Original sfreq : {sfreq} Hz → Target sfreq: {target_sfreq} Hz")
    logger.info("=" * 65)

    logger.info("─── STAGE 1: NaN / Outlier Handling ───")
    data = handle_nan_and_outliers(data, voltage_threshold)

    logger.info("─── STAGE 2: Band-Pass Filtering ───")
    data = apply_bandpass_filter(data, sfreq, bandpass_low, bandpass_high, filter_order)

    logger.info("─── STAGE 3: Notch Filtering ───")
    data = apply_notch_filter(data, sfreq, notch_freq, notch_quality)

    logger.info("─── STAGE 4: Resampling ───")
    data = apply_resampling(data, sfreq, target_sfreq)

    logger.info("─── STAGE 5: Z-Score Normalization ───")
    data = apply_zscore_normalization(data)

    logger.info("=" * 65)
    logger.info(f"PREPROCESSING COMPLETE | Output shape: {data.shape}")
    logger.info("=" * 65)
    return data.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: DATASET-SPECIFIC WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_seed_vig(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing wrapper for the SEED-VIG dataset.

    Specifications:
        Sampling rate  : 200 Hz (original = target, no resampling)
        EEG variable   : EEGsample
        Label variable : substate

    Parameters
    ----------
    filepath : str
        Path to SEED_VIG.mat.

    Returns
    -------
    preprocessed_eeg : np.ndarray  — shape (Samples, 17, 384)
    labels           : np.ndarray  — shape (Samples,)
    """
    SFREQ = 200.0

    eeg_data, labels = load_seed_vig(filepath)
    preprocessed_eeg = preprocess_eeg(
        data=eeg_data, sfreq=SFREQ, target_sfreq=SFREQ
    )

    assert preprocessed_eeg.shape[0] == labels.shape[0], (
        f"Sample/label count mismatch: {preprocessed_eeg.shape[0]} vs {labels.shape[0]}"
    )
    logger.info(
        f"SEED-VIG preprocessing complete | "
        f"EEG: {preprocessed_eeg.shape} | Labels: {labels.shape}"
    )
    return preprocessed_eeg, labels


def preprocess_sadt(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing wrapper for dataset.mat.

    PATCH v1.1 — Variable names corrected to match actual .mat file:
    ┌──────────────────────────────────────────────────────────────┐
    │  EEG data  : mat['EEGsample']  (was 'dataset')             │
    │  Labels    : mat['substate']   (was 'labels')               │
    │  Bonus     : mat['subindex']   subject IDs for CV splits    │
    │  Sfreq     : 200 Hz            (same source as SEED-VIG)    │
    └──────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    filepath : str
        Path to dataset.mat.

    Returns
    -------
    preprocessed_eeg : np.ndarray  — shape (Samples, Channels, Time_Steps)
    labels           : np.ndarray  — shape (Samples,)
    subindex         : np.ndarray  — shape (Samples,)  subject IDs
    """
    # ── CORRECTED: Both files run at 200 Hz ───────────────────────────────────
    SFREQ = 200.0    # ← FIXED: was 250.0 → 128.0

    eeg_data, labels, subindex = load_sadt(filepath)
    preprocessed_eeg = preprocess_eeg(
        data=eeg_data, sfreq=SFREQ, target_sfreq=SFREQ
    )

    assert preprocessed_eeg.shape[0] == labels.shape[0], (
        f"Sample/label count mismatch: {preprocessed_eeg.shape[0]} vs {labels.shape[0]}"
    )
    logger.info(
        f"dataset.mat preprocessing complete | "
        f"EEG: {preprocessed_eeg.shape} | Labels: {labels.shape} | "
        f"Subjects: {np.unique(subindex)}"
    )
    return preprocessed_eeg, labels, subindex


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: VALIDATION UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def validate_preprocessed_data(
    data: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "Dataset"
) -> dict:
    """
    Validate preprocessed EEG quality and print a structured report.

    Checks performed:
        ✓ Shape is 3D
        ✓ No NaN values
        ✓ No Inf values
        ✓ Sample count matches label count
        ✓ Channel means ≈ 0   (|mean| < 0.01)
        ✓ Channel stds  ≈ 1   (|std - 1| < 0.05)

    Parameters
    ----------
    data : np.ndarray
        Preprocessed EEG of shape (Samples, Channels, Time_Steps).
    labels : np.ndarray
        Labels of shape (Samples,).
    dataset_name : str
        Display name for the report header.

    Returns
    -------
    report : dict
        Full validation results dictionary.
    """
    logger.info(f"Validating preprocessed data: {dataset_name}")

    channel_means = data.mean(axis=2)
    channel_stds  = data.std(axis=2)

    report = {
        "dataset"           : dataset_name,
        "shape"             : data.shape,
        "dtype"             : str(data.dtype),
        "n_samples"         : data.shape[0],
        "n_channels"        : data.shape[1],
        "n_timesteps"       : data.shape[2],
        "n_labels"          : labels.shape[0],
        "unique_labels"     : np.unique(labels).tolist(),
        "has_nan"           : bool(np.any(np.isnan(data))),
        "has_inf"           : bool(np.any(np.isinf(data))),
        "global_min"        : float(data.min()),
        "global_max"        : float(data.max()),
        "channel_mean_mean" : float(channel_means.mean()),
        "channel_std_mean"  : float(channel_stds.mean()),
        "shape_ok"          : data.ndim == 3,
        "labels_aligned"    : data.shape[0] == labels.shape[0],
        "no_nan"            : not bool(np.any(np.isnan(data))),
        "no_inf"            : not bool(np.any(np.isinf(data))),
        "norm_mean_ok"      : bool(np.abs(channel_means.mean()) < 0.01),
        "norm_std_ok"       : bool(np.abs(channel_stds.mean() - 1.0) < 0.05),
    }
    report["all_checks_passed"] = all([
        report["shape_ok"],
        report["labels_aligned"],
        report["no_nan"],
        report["no_inf"],
        report["norm_mean_ok"],
        report["norm_std_ok"],
    ])

    logger.info(f"{'─' * 55}")
    logger.info(f"  VALIDATION REPORT: {dataset_name}")
    logger.info(f"{'─' * 55}")
    for key, val in report.items():
        status = ""
        if isinstance(val, bool):
            status = " ✓" if val else " ✗ WARNING"
        logger.info(f"  {key:<25}: {val}{status}")
    logger.info(f"{'─' * 55}")
    logger.info(
        f"  OVERALL STATUS: "
        f"{'✓ ALL CHECKS PASSED' if report['all_checks_passed'] else '✗ ISSUES DETECTED'}"
    )
    logger.info(f"{'─' * 55}")
    return report


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: MAIN ENTRY POINT — REAL DATA PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. File paths ─────────────────────────────────────────────────────────
    seed_path = r'C:\Users\KIIT0001\.vscode\Codes\NeuroDyn-OpNet\SEED_VIG.mat'
    sadt_path = r'C:\Users\KIIT0001\.vscode\Codes\NeuroDyn-OpNet\dataset.mat'

    # ── Placeholders (prevent NameError in save block on partial failure) ──────
    seed_eeg    = None
    seed_labels = None
    sadt_eeg    = None
    sadt_labels = None
    sadt_subidx = None

    # ── 2. Process SEED-VIG ───────────────────────────────────────────────────
    try:
        logger.info("Processing REAL SEED-VIG data...")
        seed_eeg, seed_labels = preprocess_seed_vig(seed_path)
        validate_preprocessed_data(seed_eeg, seed_labels, "SEED-VIG (Real)")
    except FileNotFoundError:
        logger.error(
            f"SEED-VIG file not found at: {seed_path}\n"
            "  → Check the path and confirm the file exists."
        )
    except KeyError as e:
        logger.error(f"SEED-VIG variable name mismatch: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing SEED-VIG: {e}", exc_info=True)

    # ── 3. Process dataset.mat (SADT) — FIXED variable names ─────────────────
    try:
        logger.info("Processing REAL dataset.mat (SADT) data...")
        sadt_eeg, sadt_labels, sadt_subidx = preprocess_sadt(sadt_path)
        validate_preprocessed_data(sadt_eeg, sadt_labels, "dataset.mat (Real)")
    except FileNotFoundError:
        logger.error(
            f"dataset.mat file not found at: {sadt_path}\n"
            "  → Check the path and confirm the file exists."
        )
    except KeyError as e:
        logger.error(f"dataset.mat variable name mismatch: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing dataset.mat: {e}", exc_info=True)

    # ── 4. Save all cleaned arrays ────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("SAVING CLEANED DATA TO DISK")
    logger.info("=" * 65)

    if seed_eeg is not None and seed_labels is not None:
        np.save('seed_clean_x.npy', seed_eeg)
        np.save('seed_clean_y.npy', seed_labels)
        logger.info(
            f"SEED-VIG saved | "
            f"seed_clean_x.npy {seed_eeg.shape} | "
            f"seed_clean_y.npy {seed_labels.shape}"
        )
    else:
        logger.warning("SEED-VIG arrays empty — .npy files NOT saved.")

    if sadt_eeg is not None and sadt_labels is not None:
        np.save('sadt_clean_x.npy', sadt_eeg)
        np.save('sadt_clean_y.npy', sadt_labels)
        # ── Save subindex for subject-independent CV in later steps ───────────
        if sadt_subidx is not None:
            np.save('sadt_clean_subindex.npy', sadt_subidx)
            logger.info(
                f"dataset.mat saved | "
                f"sadt_clean_x.npy {sadt_eeg.shape} | "
                f"sadt_clean_y.npy {sadt_labels.shape} | "
                f"sadt_clean_subindex.npy {sadt_subidx.shape}"
            )
    else:
        logger.warning("dataset.mat arrays empty — .npy files NOT saved.")

    logger.info("=" * 65)
    logger.info("STEP 2 COMPLETE: Cleaned data saved as .npy files.")
    logger.info("Load in Step 3 with:")
    logger.info("  seed_eeg    = np.load('seed_clean_x.npy')")
    logger.info("  seed_labels = np.load('seed_clean_y.npy')")
    logger.info("  sadt_eeg    = np.load('sadt_clean_x.npy')")
    logger.info("  sadt_labels = np.load('sadt_clean_y.npy')")
    logger.info("  sadt_subidx = np.load('sadt_clean_subindex.npy')")
    logger.info("=" * 65)