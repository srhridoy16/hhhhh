import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# -------------------------------
# 1. Load Speech Signal
# -------------------------------
fs, signal = wavfile.read("WhatsApp Audio.wav")   # আপনার wav ফাইলের নাম দিন

# Mono করা যদি stereo হয়
if len(signal.shape) > 1:
    signal = signal[:, 0]

# -------------------------------
# 2. Frame Parameters
# -------------------------------
frame_size_ms = 25
frame_shift_ms = 10

frame_length = int(fs * frame_size_ms / 1000)
frame_step = int(fs * frame_shift_ms / 1000)

signal_length = len(signal)

# -------------------------------
# 3. Number of Frames
# -------------------------------
num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

pad_length = num_frames * frame_step + frame_length
pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

# -------------------------------
# 4. Framing
# -------------------------------
frames = []
for i in range(num_frames):
    start = i * frame_step
    end = start + frame_length
    frames.append(pad_signal[start:end])
frames = np.array(frames)

print("Total number of frames:", num_frames)

# -------------------------------
# 5. Apply Hamming Window
# -------------------------------
hamming_window = np.hamming(frame_length)
windowed_frames = frames * hamming_window

# -------------------------------
# 6. Plot Sample Frames with Labels
# -------------------------------
sample_index = 5  # যেকোনো একটি ফ্রেম দেখানোর জন্য
time_axis = np.arange(frame_length) / fs * 1000  # Time in milliseconds

plt.figure(figsize=(12, 5))

# Original Frame
plt.subplot(1, 2, 1)
plt.plot(time_axis, frames[sample_index], color='blue', marker='o', markersize=3)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title(f"Original Frame #{sample_index}")
plt.grid(True)

# Windowed Frame
plt.subplot(1, 2, 2)
plt.plot(time_axis, windowed_frames[sample_index], color='red', marker='o', markersize=3)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title(f"Hamming Windowed Frame #{sample_index}")
plt.grid(True)

plt.tight_layout()
plt.show()
