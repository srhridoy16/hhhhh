# Linear Antenna-2D 
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, np.pi, 800)
N = 10
d = 0.5
k = 2*np.pi

num = np.sin(N * k * d * np.cos(theta) / 2)
den = N * np.sin(k * d * np.cos(theta) / 2)

den = np.where(den == 0, 1e-10, den)  # avoid division by zero

AF = np.abs(num / den)

ax = plt.subplot(111, projection="polar")
ax.plot(theta, AF)
ax.set_title("2D Radiation Pattern of Linear Antenna Array")

plt.show()