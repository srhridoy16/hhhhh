# 3D Radiation Pattern of Dipole Antenna
import numpy as np
import matplotlib.pyplot as plt

# Angular values
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Radiation pattern formula for half-wave dipole
R = np.abs(np.sin(theta))

# Convert to Cartesian coordinates
X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

# Plotting 3D pattern
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_title("3D Radiation Pattern of Dipole Antenna")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()