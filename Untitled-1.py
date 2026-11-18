
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# =========================================================
# ALBATROSS AIRSPACE NETHERLANDS
# 3D DEMONSTRATIEMODEL - 2000 PASSAGIERSDRONES
# =========================================================

np.random.seed(42)

TOTAL_DRONES = 2000
TOTAL_LAYERS = 8
MAX_RANGE_KM = 200

# Abstracte kaartcoordinaten.
# Dit zijn GEEN geografische vliegcoordinaten.
CITIES = {
    "ROTTERDAM": np.array([0.0, 0.0]),
    "DELFT": np.array([5.0, 7.0]),
    "AMSTERDAM": np.array([22.0, 14.0])
}

# Illustratieve visualisatielagen, geen vlieghoogteadvies.
ALTITUDES = np.array([
    60, 100, 140, 180,
    220, 260, 300, 340
])

COLORS = [
    "#00d9ff",
    "#ff9d00",
    "#a4ff00",
    "#ff39c8",
    "#ffffff",
    "#786bff",
    "#00ff99",
    "#ff5151"
]

# =========================================================
# FIGUUR
# =========================================================

plt.rcParams["toolbar"] = "toolbar2"

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#07111e")

ax = fig.add_axes([0.05, 0.19, 0.72, 0.74],
                  projection="3d")

ax.set_facecolor("#07111e")

ax.set_xlim(-5, 28)
ax.set_ylim(-5, 20)
ax.set_zlim(0, 370)

ax.set_xlabel("Netwerk X", color="white")
ax.set_ylabel("Netwerk Y", color="white")
ax.set_zlabel("Illustratieve hoogte", color="white")

ax.tick_params(colors="white")
ax.view_init(elev=30, azim=-60)

fig.suptitle(
    "ALBATROSS AIRSPACE | NETHERLANDS 3D",
    color="white",
    fontsize=19,
    fontweight="bold"
)

# =========================================================
# 3D RASTER
# =========================================================

for x in np.arange(-5, 30, 2):
    ax.plot(
        [x, x], [-5, 20], [0, 0],
        color="#24415a", alpha=0.4,
        linewidth=0.5
    )

for y in np.arange(-5, 22, 2):
    ax.plot(
        [-5, 28], [y, y], [0, 0],
        color="#24415a", alpha=0.4,
        linewidth=0.5
    )

# Verticale rasterlijnen
for x in np.arange(-4, 29, 4):
    for y in np.arange(-4, 21, 4):
        ax.plot(
            [x, x], [y, y], [0, 350],
            color="#24415a",
            alpha=0.10,
            linewidth=0.6
        )

# Horizontale hoogtelagen
for altitude in ALTITUDES:
    xx, yy = np.meshgrid(
        np.linspace(-5, 28, 15),
        np.linspace(-5, 20, 12)
    )

    zz = np.full_like(xx, altitude)

    ax.plot_wireframe(
        xx, yy, zz,
        color="#28516c",
        alpha=0.12,
        linewidth=0.4
    )

# =========================================================
# STEDEN
# =========================================================

for name, pos in CITIES.items():
    ax.scatter(
        [pos[0]], [pos[1]], [5],
        color="white",
        s=130,
        marker="^"
    )

    ax.text(
        pos[0], pos[1], 15,
        name,
        color="white",
        fontsize=10,
        fontweight="bold"
    )

# =========================================================
# ROUTE COMBINATIES
# =========================================================

city_names = list(CITIES.keys())

ROUTES = [
    ("ROTTERDAM", "AMSTERDAM"),
    ("AMSTERDAM", "ROTTERDAM"),
    ("ROTTERDAM", "DELFT"),
    ("DELFT", "ROTTERDAM"),
    ("DELFT", "AMSTERDAM"),
    ("AMSTERDAM", "DELFT")
]

# =========================================================
# DRONE DATA GENEREREN
# =========================================================

starts = []
ends = []
layers = []
phases = []
speeds = []
route_ids = []

for i in range(TOTAL_DRONES):

    route_id = np.random.randint(len(ROUTES))

    start_name, end_name = ROUTES[route_id]

    start = CITIES[start_name].copy()
    end = CITIES[end_name].copy()

    # Kleine spreiding binnen het abstracte netwerk
    start += np.random.normal(0, 0.3, 2)
    end += np.random.normal(0, 0.3, 2)

    starts.append(start)
    ends.append(end)

    layers.append(np.random.randint(TOTAL_LAYERS))

    phases.append(np.random.random())

    speeds.append(np.random.uniform(0.0008, 0.0018))

    route_ids.append(route_id)

starts = np.array(starts)
ends = np.array(ends)
layers = np.array(layers)
phases = np.array(phases)
speeds = np.array(speeds)

heights = ALTITUDES[layers]

# =========================================================
# DRONE KLEUREN
# =========================================================

drone_colors = np.array([
    COLORS[layer] for layer in layers
])

# =========================================================
# ROUTE CURVES
# =========================================================

def calculate_positions(frame):

    t = (phases + frame * speeds) % 1.0

    # Smooth interpolation
    smooth_t = t * t * (3 - 2 * t)

    x = starts[:, 0] + (
        ends[:, 0] - starts[:, 0]
    ) * smooth_t

    y = starts[:, 1] + (
        ends[:, 1] - starts[:, 1]
    ) * smooth_t

    # Kleine visuele spreiding voor het raster.
    # Dit is geen veiligheids- of afstandsregeling.
    offset = np.sin(t * np.pi * 2) * 0.35

    x += offset * np.cos(layers)
    y += offset * np.sin(layers)

    z = heights + 3 * np.sin(t * np.pi * 2)

    return x, y, z


# =========================================================
# 2000 DRONES IN EEN SCATTER OBJECT
# =========================================================

x, y, z = calculate_positions(0)

scatter = ax.scatter(
    x, y, z,
    c=drone_colors,
    s=5,
    alpha=0.85,
    depthshade=False
)

# =========================================================
# STATUSPANEEL
# =========================================================

info = fig.text(
    0.80, 0.83,
    "",
    color="white",
    fontsize=11,
    verticalalignment="top",
    family="monospace"
)

def update_status(frame):

    text = (
        "ALBATROSS CONTROL\n"
        "----------------------\n"
        f"Actieve drones: {TOTAL_DRONES}\n"
        f"Luchtlagen: {TOTAL_LAYERS}\n"
        f"Bereikmodel: {MAX_RANGE_KM} km\n"
        "----------------------\n"
        f"Rotterdam: {np.sum(np.arange(TOTAL_DRONES) % 3 == 0)}\n"
        f"Delft:     {np.sum(np.arange(TOTAL_DRONES) % 3 == 1)}\n"
        f"Amsterdam: {np.sum(np.arange(TOTAL_DRONES) % 3 == 2)}\n"
        "----------------------\n"
        "SIMULATIESTATUS: ACTIEF\n"
        "Netwerk: DEMONSTRATIE\n"
        "Verkeersveiligheid:\n"
        "Niet gevalideerd"
    )

    info.set_text(text)

# =========================================================
# LEGENDA
# =========================================================

legend_text = (
    "HOOGTELAGEN (VISUEEL)\n"
    "---------------------\n"
    + "\n".join(
        f"Laag {i+1}: {h} m"
        for i, h in enumerate(ALTITUDES)
    )
)

fig.text(
    0.80, 0.49,
    legend_text,
    color="#00d9ff",
    fontsize=10,
    family="monospace"
)

# =========================================================
# ANIMATIE
# =========================================================

paused = False
animation_speed = 1.0

def update(frame):

    if not paused:
        x, y, z = calculate_positions(
            frame * animation_speed
        )

        scatter._offsets3d = (x, y, z)

        update_status(frame)

    return scatter, info


ani = FuncAnimation(
    fig,
    update,
    frames=100000,
    interval=40,
    blit=False,
    cache_frame_data=False
)

# =========================================================
# CAMERA
# =========================================================

def rotate_view(event):
    ax.view_init(
        elev=25 + 15 * np.sin(event * 0.01),
        azim=event * 0.2
    )

# =========================================================
# START
# =========================================================

print("ALBATROSS AIRSPACE GESTART")
print("Aantal drones:", TOTAL_DRONES)
print("Aantal lagen:", TOTAL_LAYERS)
print("Bestemmingen: Rotterdam, Delft, Amsterdam")
print("Maximaal conceptueel bereik:", MAX_RANGE_KM, "km")
print("Druk op het venster om de 3D-wereld te bekijken.")

plt.show()
