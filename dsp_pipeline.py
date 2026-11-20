"""Biomedical PCG signal processing utilities."""
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, stft

SAMPLE_RATE_ORIGINAL = 16000
TARGET_SAMPLE_RATE = 4000
LOW_CUT = 20.0
HIGH_CUT = 400.0
FRAME_SIZE = 256
HOP_LENGTH = 128
N_MELS = 32
TARGET_FRAMES = 32

_MEL_FILTERBANK_CACHE = {}


def get_mel_filterbank(n_mels=N_MELS, n_fft=FRAME_SIZE, fs=TARGET_SAMPLE_RATE, fmin=LOW_CUT, fmax=HIGH_CUT):
    key = (int(n_mels), int(n_fft), float(fs), float(fmin), float(fmax))
    if key in _MEL_FILTERBANK_CACHE:
        return _MEL_FILTERBANK_CACHE[key]

    num_fft_bins = n_fft // 2 + 1
    nyquist = fs / 2.0
    fmax = min(float(fmax), nyquist)

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    bin_points = np.clip(bin_points, 0, num_fft_bins - 1)

    fb = np.zeros((n_mels, num_fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center > left:
            for k in range(left, center):
                if 0 <= k < num_fft_bins:
                    fb[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(center, right):
                if 0 <= k < num_fft_bins:
                    fb[m - 1, k] = (right - k) / (right - center)
    _MEL_FILTERBANK_CACHE[key] = fb
    return fb


def butter_bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=TARGET_SAMPLE_RATE, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, np.asarray(data, dtype=np.float32))


def shannon_energy_envelope(signal, window_len=50):
    signal = np.asarray(signal, dtype=np.float32)
    sig_norm = signal / (np.max(np.abs(signal)) + 1e-8)
    sig_sq = np.clip(sig_norm ** 2, 1e-8, 1.0)
    shannon_energy = -sig_sq * np.log(sig_sq)
    window = np.ones(window_len, dtype=np.float32) / window_len
    return np.convolve(shannon_energy, window, mode='same')


def segment_cardiac_cycles(signal, fs=TARGET_SAMPLE_RATE, cycle_duration=1.0):
    target_samples = int(fs * cycle_duration)
    envelope = shannon_energy_envelope(signal)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(envelope, distance=int(fs * 0.4), height=0.35 * np.max(envelope))
    cycles = []
    for peak in peaks:
        start = peak - int(fs * 0.15)
        end = start + target_samples
        if start >= 0 and end <= len(signal):
            cycles.append(signal[start:end])
    if not cycles:
        mid = len(signal) // 2
        start = max(0, mid - target_samples // 2)
        padded = np.pad(signal[start:start + target_samples], (0, max(0, target_samples - len(signal[start:start + target_samples])), mode='constant')
        cycles.append(padded[:target_samples])
    return cycles


def compute_log_mel_spectrogram(audio_segment, fs=TARGET_SAMPLE_RATE):
    audio = np.asarray(audio_segment, dtype=np.float32)
    if len(audio) < FRAME_SIZE:
        audio = np.pad(audio, (0, FRAME_SIZE - len(audio)))
    fb = get_mel_filterbank(N_MELS, FRAME_SIZE, fs, LOW_CUT, HIGH_CUT)
    _, _, Zxx = stft(audio, fs=fs, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
    power_spec = np.abs(Zxx[:FRAME_SIZE // 2 + 1, :]) ** 2
    mel_spec = np.dot(fb, power_spec)
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-8)).astype(np.float32)
    if log_mel.shape[1] > TARGET_FRAMES:
        log_mel = log_mel[:, :TARGET_FRAMES]
    elif log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])), mode='edge')
    min_val, max_val = float(np.min(log_mel)), float(np.max(log_mel))
    if max_val - min_val > 1e-6:
        log_mel = (log_mel - min_val) / (max_val - min_val)
    else:
        log_mel = np.zeros_like(log_mel)
    return log_mel.astype(np.float32)


def process_raw_pcg_file(audio_data, original_fs=SAMPLE_RATE_ORIGINAL):
    audio_data = np.asarray(audio_data, dtype=np.float32)
    if original_fs != TARGET_SAMPLE_RATE:
        if original_fs % TARGET_SAMPLE_RATE == 0:
            audio_4k = resample_poly(audio_data, TARGET_SAMPLE_RATE, original_fs)
        else:
            from math import gcd
            divisor = gcd(int(original_fs), TARGET_SAMPLE_RATE)
            audio_4k = resample_poly(audio_data, TARGET_SAMPLE_RATE // divisor, int(original_fs) // divisor)
    else:
        audio_4k = audio_data
    filtered_sig = butter_bandpass_filter(audio_4k, fs=TARGET_SAMPLE_RATE)
    cycles = segment_cardiac_cycles(filtered_sig, fs=TARGET_SAMPLE_RATE)
    features = [compute_log_mel_spectrogram(c, fs=TARGET_SAMPLE_RATE) for c in cycles]
    return np.asarray(features, dtype=np.float32)


def estimate_bpm_autocorr(envelope, fs=TARGET_SAMPLE_RATE, min_bpm=45, max_bpm=160, last_bpm=72.0):
    if len(envelope) < int(fs * 0.4):
        return last_bpm
    ds_factor = 20
    env_ds = np.asarray(envelope[::ds_factor], dtype=np.float32)
    fs_ds = fs / ds_factor
    env_mean = float(np.mean(env_ds))
    env_max = float(np.max(env_ds))
    if env_max < 1e-6 or (env_max / (env_mean + 1e-8)) < 1.2:
        return last_bpm
    env_norm = env_ds - env_mean
    n = len(env_norm)
    autocorr = np.correlate(env_norm, env_norm, mode='full')[n - 1:]
    min_lag = int(fs_ds * 60.0 / max_bpm)
    max_lag = min(int(fs_ds * 60.0 / min_bpm), len(autocorr) - 1)
    if min_lag >= max_lag:
        return last_bpm
    best_lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag]))
    if autocorr[0] > 1e-8 and float(autocorr[best_lag] / autocorr[0]) > 0.10:
        period_sec = best_lag / fs_ds
        measured_bpm = 60.0 / period_sec
        if 40.0 <= measured_bpm <= 180.0:
            return float(round(0.65 * last_bpm + 0.35 * measured_bpm, 1))
    return last_bpm


def process_live_pcg_chunk(audio_samples, rolling_audio=None, last_bpm=72.0, fs=TARGET_SAMPLE_RATE):
    audio = np.asarray(audio_samples, dtype=np.float32)
    try:
        filtered = butter_bandpass_filter(audio, fs=fs)
    except Exception:
        filtered = audio
    envelope = shannon_energy_envelope(filtered)

    if rolling_audio is not None and len(rolling_audio) >= int(fs):
        try:
            filt_roll = butter_bandpass_filter(np.asarray(rolling_audio, dtype=np.float32), fs=fs)
        except Exception:
            filt_roll = np.asarray(rolling_audio, dtype=np.float32)
        bpm = estimate_bpm_autocorr(shannon_energy_envelope(filt_roll), fs=fs, last_bpm=last_bpm)
    else:
        bpm = estimate_bpm_autocorr(envelope, fs=fs, last_bpm=last_bpm)

    target_samples = int(fs)
    if len(filtered) < target_samples:
        spec_audio = np.pad(filtered, (0, target_samples - len(filtered)))
    else:
        spec_audio = filtered[:target_samples]
    spec = compute_log_mel_spectrogram(spec_audio, fs=fs)
    return {"filtered": filtered.tolist(), "envelope": envelope.tolist(), "bpm": round(bpm, 1), "spectrogram": spec.tolist()}
