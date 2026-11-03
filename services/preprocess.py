import numpy as np
from scipy.signal import butter, sosfiltfilt, medfilt, savgol_filter

FS = 250
TARGET_LENGTH = 2500

def _remove_baseline(signal: np.ndarray) -> np.ndarray:
    """
    Two-pass median filter baseline wander removal.
    - Pass 1: 200ms window (50 samples) removes P/T wave influence
    - Pass 2: 600ms window (150 samples) estimates the slow baseline drift
    Subtracting the result physically removes baseline wander.
    """
    # kernel sizes must be odd
    win1 = int(0.2 * FS) | 1   # 50 -> 51 (odd)
    win2 = int(0.6 * FS) | 1   # 150 -> 151 (odd)
    baseline = medfilt(medfilt(signal, win1), win2)
    return signal - baseline

def _bandpass(signal: np.ndarray) -> np.ndarray:
    """
    Zero-phase SOS Butterworth bandpass 0.5–40Hz.
    Removes residual low-freq drift and high-freq muscle/EMI noise.
    40Hz preserves QRS, P, T morphology without aliasing risk.
    """
    sos = butter(4, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def _smooth_signal(signal: np.ndarray) -> np.ndarray:
    """
    Savitzky-Golay filter to smooth sharp R-peaks while preserving morphology.
    Window: 21 samples (~84ms), Polynomial order: 3
    """
    return savgol_filter(signal, window_length=21, polyorder=3)

def apply_filter(data: np.ndarray) -> np.ndarray:
    data = np.array(data, dtype=np.float64)

    # Stage 1: remove baseline wander
    detrended = _remove_baseline(data)

    # Stage 2: bandpass to clean residual noise
    cleaned = _bandpass(detrended)

    # Stage 3: smooth sharp R-peaks
    smoothed = _smooth_signal(cleaned)

    # Stage 4: Min-Max normalization to [-1, 1] range for better visualization
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)
    if max_val > min_val:
        normalized = 2 * (smoothed - min_val) / (max_val - min_val) - 1
    else:
        normalized = smoothed
    return normalized.astype(np.float32)

def clean_ecg_signal(signal) -> np.ndarray:
    return apply_filter(np.array(signal, dtype=np.float64))

def preprocess_signal(signal) -> np.ndarray:
    """For SVC model — raw signal only, no filtering. Matches Jupyter training pipeline."""
    signal = np.array(signal, dtype=np.float32)
    if signal.size != TARGET_LENGTH:
        raise ValueError(f"Invalid length: {signal.size}. Expected {TARGET_LENGTH}.")
    return signal.reshape(1, TARGET_LENGTH)
