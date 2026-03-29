import numpy as np
import matplotlib.pyplot as plt

# Parameters
lam = 1.0
d = lam / 2
k = 2 * np.pi / lam

A1 = 1.0
A2 = 0.6

theta = np.linspace(0, 2*np.pi, 1000)
psi = (k * d / 2) * np.cos(theta)

# Array Factor calculation
AF = A1 * np.exp(1j * psi) - A2 * np.exp(-1j * psi)
AF_norm = np.abs(AF) / np.max(np.abs(AF))

# Plotting
fig = plt.figure(figsize=(7, 8)) # Fig size ektu boro kora hoyeche
ax = fig.add_subplot(111, polar=True)

ax.plot(theta, AF_norm, color='r', linewidth=2)

# Title logic fixing
# 'pad' barale title chobi theke dure shore jay
plt.title("Far-Field Pattern\nTwo Isotropic Point Sources\nUnequal Amplitude & Opposite Phase", 
          fontsize=12, pad=40) 

# Margin adjust korar jonno
plt.subplots_adjust(top=0.80) 

plt.show()