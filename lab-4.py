import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack
import matplotlib.pyplot as plt

# ===============================
# ১. স্পিচ ফাইল লোড
# ===============================
rate, signal = wav.read("WhatsApp Audio.wav") 
signal = signal.astype(float)

# ===============================
# ২. Pre-emphasis
# ===============================
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# ===============================
# ৩. Framing
# ===============================
frame_size = 0.025  # 25ms
frame_stride = 0.010  # 10ms
frame_length = int(frame_size * rate)
frame_step = int(frame_stride * rate)
signal_length = len(emphasized_signal)
num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z)

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

# ===============================
# ৪. Hamming Window
# ===============================
frames *= np.hamming(frame_length)

# ===============================
# ৫. FFT and Power Spectrum
# ===============================
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude
pow_frames = (1.0 / NFFT) * (mag_frames ** 2)        # Power Spectrum

# ===============================
# ৬. Mel Filterbank
# ===============================
nfilt = 26
low_freq_mel = 0
high_freq_mel = 2595 * np.log10(1 + (rate / 2) / 700)  # Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
hz_points = 700 * (10**(mel_points / 2595) - 1)
bin = np.floor((NFFT + 1) * hz_points / rate).astype(int)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = bin[m - 1]   # left
    f_m = bin[m]             # center
    f_m_plus = bin[m + 1]    # right

    for k in range(f_m_minus, f_m):
        fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1])
    for k in range(f_m, f_m_plus):
        fbank[m-1, k] = (bin[m+1] - k) / (bin[m+1] - bin[m])

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Avoid log(0)
filter_banks = 20 * np.log10(filter_banks)  # dB

# ===============================
# ৭. DCT to get MFCC
# ===============================
num_ceps = 13
mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

# ===============================
# ৮. MFCC Heatmap Plot
# ===============================
plt.figure(figsize=(10, 6))
plt.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Frames')
plt.ylabel('MFCC Coefficients')
plt.title('MFCC Heatmap')
plt.colorbar(label='Amplitude (dB)')
plt.tight_layout()
plt.show()
