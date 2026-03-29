import numpy as np
import matplotlib.pyplot as plt

# Parameters
lam = 1
d = lam / 2
k = 2 * np.pi / lam
N = 3

# Angle range (full circle)
theta = np.linspace(0, 2*np.pi, 1000)

# Array Factor
AF = np.zeros_like(theta, dtype=complex)

for n in range(N):
    AF += np.exp(1j * n * k * d * np.cos(theta))

# Normalize
AF_mag = np.abs(AF)
AF_mag = AF_mag / np.max(AF_mag)

# Plot
plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
ax.plot(theta, AF_mag)

ax.set_title("2D Radiation Pattern of 3-Element Broadside Array")
plt.show()