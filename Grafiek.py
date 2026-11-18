
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Instellingen
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

fig.patch.set_facecolor("#101827")
ax.set_facecolor("#101827")

ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_zlim(0, 350)

ax.set_xlabel("Afstand X (km)", color="white")
ax.set_ylabel("Afstand Y (km)", color="white")
ax.set_zlabel("Hoogte (m)", color="white")

ax.set_title(
    "ALBATROSS AIRSPACE 3D",
    color="white",
    fontsize=18
)

ax.tick_params(colors="white")
ax.view_init(elev=25, azim=-55)

# Hoogtelagen
heights = [120, 180, 240, 300]

for i, h in enumerate(heights):
    x = np.linspace(-25, 25, 100)
    y = np.zeros_like(x)
    z = np.full_like(x, h)

    ax.plot(
        x, y, z,
        color="cyan",
        alpha=0.3,
        linestyle="--"
    )

    ax.text(22, 0, h, f" Luchtlaag {i+1}",
            color="white")

# Dronegegevens
drones = [
    {"x": 0, "y": 0, "z": 120,
     "color": "cyan", "speed": 0.12},

    {"x": 0, "y": -5, "z": 180,
     "color": "orange", "speed": 0.10},

    {"x": -5, "y": 0, "z": 240,
     "color": "lime", "speed": 0.08},

    {"x": -5, "y": -5, "z": 300,
     "color": "magenta", "speed": 0.09}
]

# Maak zichtbare drones
objects = []

for d in drones:
    obj = ax.scatter(
        [d["x"]],
        [d["y"]],
        [d["z"]],
        color=d["color"],
        s=140,
        edgecolors="white"
    )
    objects.append(obj)

# Bewegingsanimatie
def update(frame):

    for i, d in enumerate(drones):

        angle = frame * d["speed"]

        x = 20 * np.cos(angle + i)
        y = 15 * np.sin(angle + i)

        z = d["z"] + 10 * np.sin(angle)

        objects[i]._offsets3d = (
            [x], [y], [z]
        )

    return objects

ani = FuncAnimation(
    fig,
    update,
    frames=1000,
    interval=40,
    blit=False
)

plt.tight_layout()
plt.show()

