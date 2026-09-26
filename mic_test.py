"""
ESP32-S3 Microphone Hardware Test
=================================
Connects to the ESP32 via serial, captures raw binary audio packets,
saves them as a .wav file, and plots the waveform.

Usage:
    python python/mic_test.py --port COM11 --seconds 5
"""

import argparse
import struct
import time
import wave
import sys
import os
import numpy as np

try:
    import serial
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

# Binary protocol constants (must match ESP32 firmware)
MAGIC_HEADER = b'\xaa\x55\x43\x41'
HEADER_FORMAT = '<4sHIH'  # magic(4) + seq(2) + timestamp_ms(4) + sample_count(2)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def capture_audio(port: str, baud: int, duration_sec: float) -> np.ndarray:
    """Connects to ESP32 and captures raw PCM audio for the specified duration."""
    
    print(f"\n{'='*60}")
    print(f"  ESP32-S3 Microphone Hardware Test")
    print(f"  Port: {port} | Baud: {baud} | Duration: {duration_sec}s")
    print(f"{'='*60}\n")

    ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
    print(f"[OK] Serial port {port} opened at {baud} baud.")
    
    all_samples = []
    packets_received = 0
    byte_buffer = bytearray()
    ascii_lines = []
    
    start_time = time.time()
    target_samples = int(4000 * duration_sec)  # 4kHz sample rate
    
    print(f"[..] Listening for binary audio packets (need ~{target_samples} samples)...\n")
    
    while time.time() - start_time < duration_sec + 5:  # Extra 5s grace period
        try:
            raw = ser.read(ser.in_waiting or 1)
            if not raw:
                time.sleep(0.005)
                continue
            
            byte_buffer.extend(raw)
            
            while len(byte_buffer) > 0:
                magic_pos = byte_buffer.find(MAGIC_HEADER)
                
                if magic_pos > 0:
                    # ASCII text before magic header
                    text = byte_buffer[:magic_pos].decode('utf-8', errors='ignore').strip()
                    if text:
                        for line in text.splitlines():
                            line = line.strip()
                            if line:
                                ascii_lines.append(line)
                                if len(ascii_lines) <= 10:
                                    print(f"  [LOG] {line}")
                    del byte_buffer[:magic_pos]
                    continue
                
                elif magic_pos == 0:
                    if len(byte_buffer) < HEADER_SIZE:
                        break
                    
                    magic, seq, ts_ms, sample_count = struct.unpack(HEADER_FORMAT, byte_buffer[:HEADER_SIZE])
                    total_len = HEADER_SIZE + sample_count * 2 + 2
                    
                    if len(byte_buffer) < total_len:
                        break
                    
                    # Extract PCM samples
                    pcm_bytes = byte_buffer[HEADER_SIZE:HEADER_SIZE + sample_count * 2]
                    del byte_buffer[:total_len]
                    
                    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                    all_samples.extend(pcm_array.tolist())
                    packets_received += 1
                    
                    # Stats
                    rms = np.sqrt(np.mean(pcm_array.astype(np.float32) ** 2))
                    peak = np.max(np.abs(pcm_array))
                    elapsed = time.time() - start_time
                    
                    print(f"  [PKT #{seq:4d}] {sample_count} samples | "
                          f"RMS: {rms:8.1f} | Peak: {peak:6d} | "
                          f"Total: {len(all_samples):6d} samples | "
                          f"Time: {elapsed:.1f}s")
                    
                    if len(all_samples) >= target_samples:
                        break
                    continue
                
                else:
                    # No magic header found
                    nl = byte_buffer.find(b'\n')
                    if nl != -1:
                        text = byte_buffer[:nl + 1].decode('utf-8', errors='ignore').strip()
                        if text:
                            ascii_lines.append(text)
                            if len(ascii_lines) <= 10:
                                print(f"  [LOG] {text}")
                        del byte_buffer[:nl + 1]
                    else:
                        if len(byte_buffer) > 16384:
                            del byte_buffer[:8192]
                        break
            
            if len(all_samples) >= target_samples:
                break
                
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
            break
    
    ser.close()
    print(f"\n{'='*60}")
    print(f"  Capture Complete!")
    print(f"  Packets received: {packets_received}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Duration: {len(all_samples) / 4000:.2f} seconds at 4kHz")
    print(f"  ASCII log lines captured: {len(ascii_lines)}")
    print(f"{'='*60}\n")
    
    return np.array(all_samples, dtype=np.int16)


def save_wav(samples: np.ndarray, filename: str, sample_rate: int = 4000):
    """Saves PCM samples to a WAV file."""
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    print(f"[SAVED] WAV file: {filename}")
    print(f"        Duration: {len(samples) / sample_rate:.2f}s | "
          f"Sample Rate: {sample_rate} Hz | Channels: 1 | Bits: 16")


def plot_waveform(samples: np.ndarray, sample_rate: int = 4000, save_path: str = None):
    """Plots the captured audio waveform and spectrum."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot. Run: pip install matplotlib")
        return
    
    audio_float = samples.astype(np.float32) / 32768.0
    t = np.arange(len(samples)) / sample_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('ESP32-S3 INMP441 Microphone Hardware Test', fontsize=14, fontweight='bold')
    
    # 1. Full Waveform
    axes[0].plot(t, audio_float, linewidth=0.3, color='#00f5ff')
    axes[0].set_title(f'Raw PCG Waveform ({len(samples)} samples at {sample_rate} Hz)')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linewidth=0.5)
    
    # Stats annotation
    rms = np.sqrt(np.mean(audio_float ** 2))
    peak = np.max(np.abs(audio_float))
    axes[0].text(0.01, 0.95, f'RMS: {rms:.4f} | Peak: {peak:.4f}',
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                 color='white')
    
    # 2. Zoomed waveform (first 0.5 seconds)
    zoom_samples = min(2000, len(samples))
    t_zoom = np.arange(zoom_samples) / sample_rate
    axes[1].plot(t_zoom, audio_float[:zoom_samples], linewidth=0.5, color='#00ff88')
    axes[1].set_title(f'Zoomed View (first {zoom_samples / sample_rate:.2f}s)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Frequency Spectrum (FFT)
    if len(samples) > 256:
        try:
            from scipy.signal import welch
            freqs, psd = welch(audio_float, fs=sample_rate, nperseg=min(1024, len(samples)))
            axes[2].semilogy(freqs, psd, linewidth=0.8, color='#ff6b6b')
            axes[2].set_title('Power Spectral Density (Welch Method)')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('PSD')
            axes[2].set_xlim(0, sample_rate / 2)
            axes[2].grid(True, alpha=0.3)
            axes[2].axvspan(20, 400, alpha=0.1, color='green', label='Cardiac Band (20-400 Hz)')
            axes[2].legend()
        except ImportError:
            axes[2].text(0.5, 0.5, 'scipy not installed\nSkipping spectrum plot',
                        ha='center', va='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Plot: {save_path}")
    else:
        plt.show()


def analyze_signal(samples: np.ndarray, sample_rate: int = 4000):
    """Prints a diagnostic analysis of the captured audio."""
    audio_float = samples.astype(np.float32) / 32768.0
    
    rms = np.sqrt(np.mean(audio_float ** 2))
    peak = np.max(np.abs(audio_float))
    zero_pct = np.sum(samples == 0) / len(samples) * 100
    unique_vals = len(np.unique(samples))
    
    print(f"\n{'='*60}")
    print(f"  MICROPHONE SIGNAL ANALYSIS")
    print(f"{'='*60}")
    print(f"  RMS Level:       {rms:.6f}  {'(VERY LOW - check wiring!)' if rms < 0.001 else '(OK)' if rms < 0.1 else '(STRONG)'}")
    print(f"  Peak Level:      {peak:.6f}  {'(SILENCE - mic not working?)' if peak < 0.005 else '(OK)'}")
    print(f"  Zero samples:    {zero_pct:.1f}%    {'(ALL ZEROS - mic disconnected!)' if zero_pct > 99 else '(OK)' if zero_pct < 50 else '(HIGH - check L/R pin)'}")
    print(f"  Unique values:   {unique_vals}       {'(ONLY 1 VALUE - no signal!)' if unique_vals < 5 else '(OK)'}")
    print(f"  DC Offset:       {np.mean(audio_float):.6f}")
    print(f"  Min raw value:   {np.min(samples)}")
    print(f"  Max raw value:   {np.max(samples)}")
    print(f"{'='*60}")
    
    if zero_pct > 99:
        print("\n  DIAGNOSIS: Microphone is producing ALL ZEROS.")
        print("     -> Check INMP441 wiring (SD -> GPIO2, SCK -> GPIO41, WS -> GPIO42)")
        print("     -> Make sure L/R pin is tied to GND for left channel")
        print("     -> Verify VDD is connected to 3.3V")
    elif rms < 0.001:
        print("\n  DIAGNOSIS: Signal level is extremely low.")
        print("     -> Microphone may be connected but not picking up sound")
        print("     -> Try tapping or blowing on the mic while recording")
    elif rms > 0.01:
        print("\n  DIAGNOSIS: Microphone is WORKING! Audio signal detected.")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='ESP32-S3 Microphone Hardware Test')
    parser.add_argument('--port', type=str, default='COM11', help='Serial COM port (default: COM11)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--seconds', type=float, default=5.0, help='Recording duration in seconds (default: 5)')
    parser.add_argument('--output', type=str, default=None, help='Output WAV filename')
    args = parser.parse_args()
    
    # Output paths
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mic_test_output')
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    wav_path = args.output or os.path.join(out_dir, f'mic_capture_{timestamp}.wav')
    plot_path = os.path.join(out_dir, f'mic_waveform_{timestamp}.png')
    
    # Capture
    samples = capture_audio(args.port, args.baud, args.seconds)
    
    if len(samples) == 0:
        print("\n  NO AUDIO DATA RECEIVED!")
        print("     -> Is the ESP32-S3 powered on and running the firmware?")
        print("     -> Is the correct COM port selected?")
        print("     -> Try: python python/mic_test.py --port COM11 --baud 115200")
        sys.exit(1)
    
    # Save WAV
    save_wav(samples, wav_path)
    
    # Analyze
    analyze_signal(samples)
    
    # Plot
    plot_waveform(samples, save_path=plot_path)
    
    print(f"\n  Output files:")
    print(f"     WAV:  {wav_path}")
    print(f"     Plot: {plot_path}")
    print(f"\n  Play the WAV file to hear what the mic captured!")
    print(f"     You can open it with any audio player.\n")


if __name__ == '__main__':
    main()
