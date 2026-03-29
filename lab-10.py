# lab-10
# Broadside array antenna 7-element (N=7)

import numpy as np
import matplotlib.pyplot as plt

# Parameters
lam = 1           # wavelength
d = lam / 2       # element spacing (λ/2)
k = 2 * np.pi / lam  # wave number
N = 7             # number of elements

# Angle range (0 to 180 degrees in radians)
theta = np.linspace(0, np.pi, 1000)

# Array Factor (AF) initialization
AF = np.zeros_like(theta, dtype=complex)

# Calculate Array Factor
for n in range(N):
    AF += np.exp(1j * n * k * d * np.cos(theta))

# Normalize magnitude
AF_mag = np.abs(AF)
AF_mag = AF_mag / np.max(AF_mag)

# Plot 2D Radiation Pattern
plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
ax.plot(theta, AF_mag, color='b', linewidth=2)
ax.set_title("2D Radiation Pattern (7-Element Broadside Array)\n", fontsize=12)
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])  # radial ticks
plt.show()