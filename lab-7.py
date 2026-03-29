import numpy as np
import matplotlib.pyplot as plt

# Parameters
lam = 1.0                 # wavelength
d = lam / 2               # spacing (λ/2)
k = 2 * np.pi / lam       # wave number

# Angle range (0 to 2π)
theta = np.linspace(0, 2*np.pi, 1000)

# Array Factor (Two sources, equal amplitude, opposite phase)
# Phase shift beta = pi, so AF = 2j * sin( (kd cos(theta) + beta)/2 )
AF = 2 * np.sin((k * d / 2) * np.cos(theta))

# Normalized magnitude
AF_norm = np.abs(AF) / np.max(np.abs(AF))

# Plotting
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, AF_norm, color='blue', linewidth=2)

# Title fixing (pad or y parameter use korle upore jayga hoy)
ax.set_title("Far-Field Pattern\nTwo Isotropic Point Sources\nEqual Amplitude & Opposite Phase (d = λ/2)", 
             va='bottom', pad=30, fontsize=12)

ax.grid(True)
plt.tight_layout()
plt.show()