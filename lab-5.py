import numpy as np
import librosa
import matplotlib.pyplot as plt

# ==========================
# 1. Load speech signal
# ==========================
filename = 'WhatsApp Audio.wav'
y, sr = librosa.load(filename, sr=None)

# ==========================
# 2. Pre-emphasis
# ==========================
pre_emphasis = 0.97
y_preemph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

# ==========================
# 3. Framing
# ==========================
frame_size = 0.03  # 30 ms
frame_stride = 0.01  # 10 ms
frame_length = int(frame_size * sr)
frame_step = int(frame_stride * sr)
signal_length = len(y_preemph)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

# Pad signal to fit into frames
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(y_preemph, z)

# Slice frames
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

# Apply Hamming window
frames *= np.hamming(frame_length)

# ==========================
# 4. Energy per frame (for voiced/unvoiced)
# ==========================
frame_energy = np.sum(frames**2, axis=1)
energy_threshold = 0.1 * np.max(frame_energy)  # threshold

# ==========================
# 5. Autocorrelation function
# ==========================
def autocorr(frame):
    """Return autocorrelation of a frame"""
    frame = frame - np.mean(frame)  # zero-mean
    result = np.correlate(frame, frame, mode='full')
    return result[result.size//2:]

# ==========================
# 6. Pitch estimation
# ==========================
f0 = []
time_axis = np.arange(0, num_frames * frame_step, frame_step) / sr

for i in range(num_frames):
    frame = frames[i]
    if frame_energy[i] < energy_threshold:
        f0.append(0)  # unvoiced
        continue
    
    r = autocorr(frame)
    # Ignore the zero-lag peak
    r[:int(sr/400)] = 0  # maximum 400 Hz pitch
    peak_index = np.argmax(r)
    
    if peak_index == 0:
        f0.append(0)
    else:
        pitch = sr / peak_index
        f0.append(pitch)

f0 = np.array(f0)

# ==========================
# 7. Plot pitch contour
# ==========================
plt.figure(figsize=(12, 4))
plt.plot(time_axis, f0, color='r')
plt.xlabel('Time (s)')
plt.ylabel('Pitch F0 (Hz)')
plt.title('Pitch Contour (Voiced Frames)')
plt.ylim(0, 500)
plt.grid()
plt.show()
