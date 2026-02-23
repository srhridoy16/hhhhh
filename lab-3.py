import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ===============================
# Step 1: Load speech signal
# ===============================
fs, audio = wavfile.read("WhatsApp Audio.wav")  # Replace with your file path

# Convert to float and normalize
audio = audio.astype(float)
audio = audio / np.max(np.abs(audio))

# ===============================
# Step 2: STFT using frame-wise FFT
# ===============================
frame_size = 1024           # Frame length
hop_size = 512              # 50% overlap
window = np.hanning(frame_size)

# Compute number of frames and pad signal if needed
num_frames = int(np.ceil((len(audio) - frame_size) / hop_size)) + 1
pad_length = (num_frames - 1) * hop_size + frame_size
audio_padded = np.append(audio, np.zeros(pad_length - len(audio)))

# Compute STFT
stft_matrix = []
for i in range(num_frames):
    start = i * hop_size
    frame = audio_padded[start:start+frame_size] * window
    fft_frame = np.fft.rfft(frame)
    stft_matrix.append(fft_frame)

stft_matrix = np.array(stft_matrix).T  # Shape: freq_bins x time_frames

# ===============================
# Step 3: Convert magnitude to dB
# ===============================
magnitude = np.abs(stft_matrix)
magnitude_db = 20 * np.log10(magnitude + 1e-6)  # add small value to avoid log(0)

# ===============================
# Step 4: Plot spectrogram
# ===============================
freqs = np.fft.rfftfreq(frame_size, 1/fs)
times = np.arange(num_frames) * hop_size / fs

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, freqs, magnitude_db, shading='gouraud', cmap='jet')
plt.colorbar(label='Magnitude (dB)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Speech Signal')
plt.tight_layout()
plt.show()

# ===============================
# Step 5: Practical Application
# ===============================
# Spectrograms are widely used in speech processing for:
# -> Speech Recognition: Extracts time-frequency features like phonemes and formants for ASR systems.
