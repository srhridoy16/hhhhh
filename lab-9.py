# Lab-09
# Broadside array antenna 5-element (N=5)

import numpy as np
import matplotlib.pyplot as plt

# Parameters
lam = 1         # Wavelength
d = lam / 2     # Element spacing
k = 2 * np.pi / lam  # Wave number
N = 5           # Number of elements

# Angle range (0 to pi)
theta = np.linspace(0, np.pi, 1000)

# Array Factor (AF)
AF = np.zeros_like(theta, dtype=complex)
for n in range(N):
    AF += np.exp(1j * n * k * d * np.cos(theta))

# Normalize magnitude
AF_mag = np.abs(AF)
AF_mag = AF_mag / np.max(AF_mag)

# Plot 2D Radiation Pattern
plt.figure(figsize=(6,6))
ax = plt.subplot(111, projection='polar')
ax.plot(theta, AF_mag)
ax.set_title("2D Radiation Pattern (5-Element Broadside Array)")
plt.show()