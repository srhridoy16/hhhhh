#Isotropic Antenna (3D)    
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2*np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

# Isotropic → constant radiation
R = np.ones_like(theta)

X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap="plasma")
ax.set_title("Isotropic Antenna (3D)")
plt.show()