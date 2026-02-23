import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# =====================================
# (1) Read the WAV file
# =====================================
file_name = "WhatsApp Audio.wav"  
fs, signal = wavfile.read(file_name)

# If stereo, convert to mono
if len(signal.shape) > 1:
    signal = signal.mean(axis=1)

# Convert to float for accurate FFT
signal = signal.astype(float)

# =====================================
# (a) Basic Information
# =====================================
N = len(signal)                 # Number of samples
duration = N / fs               # Duration in seconds
max_amp = np.max(signal)
min_amp = np.min(signal)

print("Sampling Rate (Fs):", fs, "Hz")
print("Number of Samples (N):", N)
print("Duration:", duration, "seconds")
print("Maximum Amplitude:", max_amp)
print("Minimum Amplitude:", min_amp)

# =====================================
# (b) Plot Waveform
# =====================================
time = np.arange(N) / fs

plt.figure()
plt.plot(time, signal)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform of Speech Signal")
plt.grid(True)
plt.show()

# =====================================
# (c) Compute FFT and Plot Magnitude Spectrum
# =====================================
fft_values = np.fft.fft(signal)          # FFT computation
fft_magnitude = np.abs(fft_values) / N   # Normalized magnitude
freq = np.fft.fftfreq(N, d=1/fs)         # Frequency axis

# Take only positive frequencies
half = N // 2
freq = freq[:half]
fft_magnitude = fft_magnitude[:half]

# Print first 10 frequency components (to show computation)
print("\nFirst 10 Frequency Components:")
for i in range(10):
    print(f"Frequency = {freq[i]:.2f} Hz , Magnitude = {fft_magnitude[i]:.6f}")

# Plot Magnitude Spectrum
plt.figure()
plt.plot(freq, fft_magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum using FFT")
plt.grid(True)
plt.show()
