#Horn Antenna-3D
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, np.pi, 250)
phi = np.linspace(0, 2*np.pi, 250)
theta, phi = np.meshgrid(theta, phi)

n = 6
R = np.abs(np.cos(theta))**n

X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap="inferno")
ax.set_title("Horn Antenna (3D)")
plt.show()