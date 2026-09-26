"""
Biomedical PCG Signal Processing Pipeline for Smart Stethoscope
- 20 Hz - 400 Hz 4th-order Butterworth Bandpass Filtering
- Shannon Energy Envelope for S1/S2 Detection & Heartbeat Segmentation
- 32x32 Log-Mel Spectrogram Feature Extraction for TinyML Deployment
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample, stft

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

SAMPLE_RATE_ORIGINAL = 16000  # Original microphone sampling rate
TARGET_SAMPLE_RATE = 4000     # Decimated rate (20-400Hz Nyquist safe)
LOW_CUT = 20.0                # High-pass cut (motion artifact filter)
HIGH_CUT = 400.0              # Low-pass cut (high frequency noise)
FRAME_SIZE = 256              # FFT Window (64 ms at 4kHz)
HOP_LENGTH = 128              # Hop Size (32 ms overlap)
N_MELS = 32                   # 32 Mel Frequency Bins
TARGET_FRAMES = 32            # 32 Time steps (Total 32x32 matrix)

_MEL_FILTERBANK_CACHE = None

def get_mel_filterbank(n_mels=N_MELS, n_fft=FRAME_SIZE, fs=TARGET_SAMPLE_RATE, fmin=LOW_CUT, fmax=HIGH_CUT):
    """Generates and caches 32-bin triangular Mel Filterbank."""
    global _MEL_FILTERBANK_CACHE
    if _MEL_FILTERBANK_CACHE is not None:
        return _MEL_FILTERBANK_CACHE

    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    _MEL_FILTERBANK_CACHE = fb
    return _MEL_FILTERBANK_CACHE


def butter_bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=TARGET_SAMPLE_RATE, order=4):
    """4th-order Butterworth IIR Bandpass Filter (20-400 Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def shannon_energy_envelope(signal, window_len=50):
    """
    Computes Normalized Shannon Energy Envelope:
    Es(t) = - x_norm^2(t) * log(x_norm^2(t))
    Smooths using moving average window.
    """
    sig_norm = signal / (np.max(np.abs(signal)) + 1e-8)
    sig_sq = sig_norm ** 2
    # Avoid log(0)
    sig_sq_clip = np.clip(sig_sq, 1e-8, 1.0)
    shannon_energy = -sig_sq_clip * np.log(sig_sq_clip)
    
    # Smooth with moving average window (~50ms)
    window = np.ones(window_len) / window_len
    envelope = np.convolve(shannon_energy, window, mode='same')
    return envelope


def segment_cardiac_cycles(signal, fs=TARGET_SAMPLE_RATE, cycle_duration=1.0):
    """
    Segments signal into 1.0-second cardiac cycles based on S1 peaks.
    Target fixed output: int(fs * cycle_duration) samples = 4000 samples.
    """
    target_samples = int(fs * cycle_duration)
    envelope = shannon_energy_envelope(signal)
    
    # Peak detection threshold (at least 0.3 * max envelope)
    min_dist = int(fs * 0.4) # Minimum 0.4s between consecutive S1 pulses (~150 BPM max)
    peaks = []
    threshold = 0.35 * np.max(envelope)
    
    for i in range(1, len(envelope) - 1):
        if envelope[i] > threshold and envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
            if not peaks or (i - peaks[-1]) > min_dist:
                peaks.append(i)
                
    cycles = []
    for peak in peaks:
        start = peak - int(fs * 0.15) # Start 150ms before S1
        end = start + target_samples
        if start >= 0 and end <= len(signal):
            cycles.append(signal[start:end])
            
    if not cycles:
        # Fallback if no clean S1 peak found: frame fixed 1.0s window from middle
        mid = len(signal) // 2
        start = max(0, mid - target_samples // 2)
        end = start + target_samples
        if end <= len(signal):
            cycles.append(signal[start:end])
        else:
            padded = np.pad(signal, (0, max(0, target_samples - len(signal))), mode='constant')
            cycles.append(padded[:target_samples])
            
    return cycles


def compute_log_mel_spectrogram(audio_segment, fs=TARGET_SAMPLE_RATE):
    """
    Computes a 32x32 Log-Mel Spectrogram from a 1.0-second 4kHz audio segment.
    Output shape: (32, 32) float32 normalized to [0, 1].
    Ultra-fast execution (< 5 ms) via vectorized SciPy/NumPy STFT.
    """
    audio = np.asarray(audio_segment, dtype=np.float32)
    if len(audio) < FRAME_SIZE:
        audio = np.pad(audio, (0, FRAME_SIZE - len(audio)))

    fb = get_mel_filterbank(N_MELS, FRAME_SIZE, fs, LOW_CUT, HIGH_CUT)
    _, _, Zxx = stft(audio, fs=fs, nperseg=FRAME_SIZE, noverlap=FRAME_SIZE - HOP_LENGTH)
    power_spec = (np.abs(Zxx) ** 2).astype(np.float32)
    mel_spec = np.dot(fb, power_spec)
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-8))

    # Resize/trim time dimension to exactly TARGET_FRAMES (32)
    if log_mel.shape[1] > TARGET_FRAMES:
        log_mel = log_mel[:, :TARGET_FRAMES]
    elif log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])), mode='edge')

    # Min-Max Normalize to [0, 1]
    min_val = float(np.min(log_mel))
    max_val = float(np.max(log_mel))
    if max_val - min_val > 1e-6:
        log_mel_norm = (log_mel - min_val) / (max_val - min_val)
    else:
        log_mel_norm = np.zeros_like(log_mel)

    return log_mel_norm.astype(np.float32)


def process_raw_pcg_file(audio_data, original_fs=SAMPLE_RATE_ORIGINAL):
    """
    Full DSP pipeline for raw 16kHz audio input:
    1. Decimate 16kHz -> 4kHz
    2. Bandpass 20-400Hz
    3. Segment cardiac cycles
    4. Compute 32x32 Log-Mel Spectrograms
    """
    if original_fs != TARGET_SAMPLE_RATE:
        num_target_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / original_fs)
        audio_4k = resample(audio_data, num_target_samples)
    else:
        audio_4k = audio_data

    # Filter signal
    filtered_sig = butter_bandpass_filter(audio_4k, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=TARGET_SAMPLE_RATE)
    
    # Extract cycles
    cycles = segment_cardiac_cycles(filtered_sig, fs=TARGET_SAMPLE_RATE)
    
    # Compute Log-Mel Spectrograms
    features = [compute_log_mel_spectrogram(c, fs=TARGET_SAMPLE_RATE) for c in cycles]
    return np.array(features)


def estimate_bpm_autocorr(envelope, fs=TARGET_SAMPLE_RATE, min_bpm=45, max_bpm=160, last_bpm=72.0):
    """
    Estimates dynamic heart rate (BPM) from the Shannon energy envelope
    using autocorrelation and peak interval verification over physiological cardiac period lags.
    """
    if len(envelope) < int(fs * 0.4):
        return last_bpm

    # Downsample envelope for fast autocorrelation (4000Hz -> 200Hz, factor 20)
    ds_factor = 20
    env_ds = np.asarray(envelope[::ds_factor], dtype=np.float32)
    fs_ds = fs / ds_factor

    # Normalize envelope
    env_mean = float(np.mean(env_ds))
    env_max = float(np.max(env_ds))
    if env_max < 1e-6 or (env_max / (env_mean + 1e-8)) < 1.2:
        # Faint signal or ambient silence, retain previous valid BPM
        return last_bpm

    env_norm = env_ds - env_mean

    # Autocorrelation
    n = len(env_norm)
    autocorr = np.correlate(env_norm, env_norm, mode='full')
    autocorr = autocorr[n - 1:]

    min_lag = int(fs_ds * 60.0 / max_bpm)
    max_lag = int(fs_ds * 60.0 / min_bpm)
    max_lag = min(max_lag, len(autocorr) - 1)

    if min_lag >= max_lag:
        return last_bpm

    search_region = autocorr[min_lag:max_lag]
    if len(search_region) == 0:
        return last_bpm

    best_lag = min_lag + int(np.argmax(search_region))
    peak_val = float(autocorr[best_lag])
    zero_val = float(autocorr[0])

    # Check autocorrelation peak prominence
    if zero_val > 1e-8 and (peak_val / zero_val) > 0.10:
        period_sec = best_lag / fs_ds
        if period_sec > 0:
            measured_bpm = 60.0 / period_sec
            if 40.0 <= measured_bpm <= 180.0:
                # Smooth EMA update
                return float(round(0.65 * last_bpm + 0.35 * measured_bpm, 1))

    # Secondary peak-interval detection on envelope
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(env_ds, distance=int(fs_ds * 0.35), prominence=0.20 * (env_max - env_mean))
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / fs_ds
            valid_intervals = [dt for dt in intervals if 0.33 <= dt <= 1.5]
            if valid_intervals:
                med_dt = float(np.median(valid_intervals))
                measured_bpm = 60.0 / med_dt
                if 40.0 <= measured_bpm <= 180.0:
                    return float(round(0.65 * last_bpm + 0.35 * measured_bpm, 1))
    except Exception:
        pass

    return last_bpm


def process_live_pcg_chunk(audio_samples, rolling_audio=None, last_bpm=72.0, fs=TARGET_SAMPLE_RATE):
    """
    Optimized live processing for incoming streaming PCG audio from ESP32-S3:
    1. Bandpass filter 20-400Hz
    2. Shannon Energy Envelope
    3. Real-time dynamic BPM estimation (uses rolling audio for multi-cycle analysis)
    4. 32x32 Log-Mel Spectrogram extraction
    Returns a dictionary of processed signals and metrics.
    """
    audio = np.asarray(audio_samples, dtype=np.float32)
    max_amp = float(np.max(np.abs(audio)))
    if max_amp > 1e-6:
        audio = audio / max_amp

    # 1. Bandpass 20-400Hz
    try:
        filtered = butter_bandpass_filter(audio, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=fs)
    except Exception:
        filtered = audio

    # 2. Shannon Envelope
    envelope = shannon_energy_envelope(filtered)

    # 3. Dynamic BPM estimation (uses multi-second rolling audio if available)
    if rolling_audio is not None and len(rolling_audio) >= int(fs * 1.0):
        try:
            filt_roll = butter_bandpass_filter(rolling_audio, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=fs)
        except Exception:
            filt_roll = rolling_audio
        env_roll = shannon_energy_envelope(filt_roll)
        bpm = estimate_bpm_autocorr(env_roll, fs=fs, last_bpm=last_bpm)
    else:
        bpm = estimate_bpm_autocorr(envelope, fs=fs, last_bpm=last_bpm)

    # 4. 32x32 Log-Mel Spectrogram (fixed 1.0s window = 4000 samples)
    target_samples = int(fs * 1.0)
    if len(filtered) != target_samples:
        if len(filtered) < target_samples:
            padded = np.pad(filtered, (0, target_samples - len(filtered)), mode='constant')
            spec_audio = padded
        else:
            spec_audio = filtered[:target_samples]
    else:
        spec_audio = filtered

    spec = compute_log_mel_spectrogram(spec_audio, fs=fs)

    return {
        "filtered": filtered.tolist(),
        "envelope": envelope.tolist(),
        "bpm": round(bpm, 1),
        "spectrogram": spec.tolist()
    }

