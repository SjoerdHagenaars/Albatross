
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================================
# ALBATROSS AIRSPACE NETHERLANDS 3.0
# 2000 PASSAGIERSDRONES - 3D SIMULATIE
# ==========================================================

np.random.seed(42)

TOTAL_DRONES = 2000
TOTAL_LAYERS = 8
MAX_RANGE_KM = 200

# Abstracte netwerkcoordinaten (geen echte GPS-coordinaten)
CITIES = {
    "ROTTERDAM": np.array([0.0, 0.0]),
    "DELFT": np.array([7.0, 5.0]),
    "AMSTERDAM": np.array([25.0, 15.0])
}

CITY_COLORS = {
    "ROTTERDAM": "#00d9ff",
    "DELFT": "#ffcc00",
    "AMSTERDAM": "#ff66cc"
}

ROUTES = [
    ("ROTTERDAM", "AMSTERDAM"),
    ("AMSTERDAM", "ROTTERDAM"),
    ("ROTTERDAM", "DELFT"),
    ("DELFT", "ROTTERDAM"),
    ("DELFT", "AMSTERDAM"),
    ("AMSTERDAM", "DELFT")
]

# Illustratieve visualisatielagen
ALTITUDES = np.array([
    60, 100, 140, 180,
    220, 260, 300, 340
])

LAYER_COLORS = [
    "#00d9ff",
    "#ff9900",
    "#aaff00",
    "#ff33cc",
    "#ffffff",
    "#8877ff",
    "#00ffaa",
    "#ff4444"
]

# ==========================================================
# DRONES AANMAKEN
# ==========================================================

route_ids = np.random.randint(
    0, len(ROUTES), TOTAL_DRONES
)

layers = np.random.randint(
    0, TOTAL_LAYERS, TOTAL_DRONES
)

phases = np.random.random(TOTAL_DRONES)

speeds = np.random.uniform(
    0.0007, 0.0015, TOTAL_DRONES
)

start_positions = np.zeros((TOTAL_DRONES, 2))
end_positions = np.zeros((TOTAL_DRONES, 2))

for i in range(TOTAL_DRONES):

    start_name, end_name = ROUTES[route_ids[i]]

    start = CITIES[start_name].copy()
    end = CITIES[end_name].copy()

    # Kleine spreiding voor de visuele verkeersstromen
    start += np.random.normal(0, 0.12, 2)
    end += np.random.normal(0, 0.12, 2)

    start_positions[i] = start
    end_positions[i] = end

drone_heights = ALTITUDES[layers]

drone_colors = [
    LAYER_COLORS[i] for i in layers
]

# ==========================================================
# POSITIES BEREKENEN
# ==========================================================

def calculate_positions(frame):

    t = (phases + frame * speeds) % 1.0

    # Vloeiende beweging
    smooth = t * t * (3 - 2 * t)

    dx = end_positions[:, 0] - start_positions[:, 0]
    dy = end_positions[:, 1] - start_positions[:, 1]

    x = start_positions[:, 0] + dx * smooth
    y = start_positions[:, 1] + dy * smooth

    # Kleine visuele afwijking binnen de verkeersbaan
    offset = 0.20 * np.sin(t * 2 * np.pi)

    x += offset * np.sin(layers * 0.8)
    y += offset * np.cos(layers * 0.8)

    z = drone_heights + 2 * np.sin(t * 2 * np.pi)

    return x, y, z


# ==========================================================
# FIGUUR MAKEN
# ==========================================================

fig = plt.figure(figsize=(16, 10))

fig.patch.set_facecolor("#07111e")

ax = fig.add_axes(
    [0.04, 0.10, 0.72, 0.82],
    projection="3d"
)

ax.set_facecolor("#07111e")

ax.set_xlim(-4, 29)
ax.set_ylim(-5, 21)
ax.set_zlim(0, 370)

ax.set_xlabel("Netwerk X (km)", color="white")
ax.set_ylabel("Netwerk Y (km)", color="white")
ax.set_zlabel("Visualisatiehoogte (m)", color="white")

ax.tick_params(colors="white")

ax.set_title(
    "ALBATROSS AIRSPACE NETHERLANDS",
    fontsize=17,
    color="white",
    pad=20
)

ax.view_init(elev=28, azim=-55)

# ==========================================================
# 3D GRONDRaster
# ==========================================================

for x_grid in np.arange(-4, 30, 2):

    ax.plot(
        [x_grid, x_grid],
        [-5, 21],
        [0, 0],
        color="#31516b",
        alpha=0.45,
        linewidth=0.5
    )

for y_grid in np.arange(-5, 22, 2):

    ax.plot(
        [-4, 29],
        [y_grid, y_grid],
        [0, 0],
        color="#31516b",
        alpha=0.45,
        linewidth=0.5
    )

# ==========================================================
# VERTICALE RASTERLIJNEN
# ==========================================================

for x_grid in np.arange(-4, 30, 4):

    for y_grid in np.arange(-4, 22, 4):

        ax.plot(
            [x_grid, x_grid],
            [y_grid, y_grid],
            [0, 350],
            color="#31516b",
            alpha=0.12,
            linewidth=0.5
        )

# ==========================================================
# HOOGTELAGEN
# ==========================================================

for i, height in enumerate(ALTITUDES):

    xx, yy = np.meshgrid(
        np.linspace(-4, 29, 10),
        np.linspace(-5, 21, 9)
    )

    zz = np.full_like(xx, height)

    ax.plot_wireframe(
        xx,
        yy,
        zz,
        color=LAYER_COLORS[i],
        alpha=0.09,
        linewidth=0.5
    )

    ax.text(
        26,
        -4,
        height,
        f"LAAG {i+1}",
        color=LAYER_COLORS[i],
        fontsize=8
    )

# ==========================================================
# STEDEN WEERGEVEN
# ==========================================================

for name, position in CITIES.items():

    x, y = position

    ax.scatter(
        [x],
        [y],
        [5],
        color=CITY_COLORS[name],
        s=180,
        marker="o",
        edgecolors="white"
    )

    ax.text(
        x,
        y,
        20,
        name,
        color="white",
        fontsize=10,
        fontweight="bold"
    )

# ==========================================================
# VIRTUELE SNELWEGEN
# ==========================================================

for route in ROUTES:

    start = CITIES[route[0]]
    end = CITIES[route[1]]

    for height in ALTITUDES:

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [height, height],
            color="#00aaff",
            linewidth=0.7,
            alpha=0.10
        )

# ==========================================================
# PARKEERPLAATS ALS SCHAALREFERENTIE
# ==========================================================

# Afmetingen in kilometers
parking_length = 0.005
parking_width = 0.0025

px = [
    0,
    parking_length,
    parking_length,
    0,
    0
]

py = [
    0,
    0,
    parking_width,
    parking_width,
    0
]

pz = [8] * 5

ax.plot(
    px,
    py,
    pz,
    color="yellow",
    linewidth=3
)

ax.text(
    0,
    0,
    30,
    "PARKEERPLAATS\n2,5 x 5 M",
    color="yellow",
    fontsize=7
)

# ==========================================================
# DRONES WEERGEVEN
# ==========================================================

x, y, z = calculate_positions(0)

drones = ax.scatter(
    x,
    y,
    z,
    c=drone_colors,
    s=7,
    alpha=0.9,
    depthshade=False
)

# ==========================================================
# DASHBOARD
# ==========================================================

dashboard = fig.text(
    0.79,
    0.84,
    "",
    color="white",
    fontsize=10,
    family="monospace",
    verticalalignment="top"
)

layer_dashboard = fig.text(
    0.79,
    0.42,
    "",
    color="#00d9ff",
    fontsize=9,
    family="monospace",
    verticalalignment="top"
)

# ==========================================================
# DASHBOARD UPDATEN
# ==========================================================

def update_dashboard(frame):

    dashboard.set_text(
        "ALBATROSS CONTROL\n"
        "====================\n"
        f"Vliegtuigen: {TOTAL_DRONES}\n"
        f"Hoogtelagen: {TOTAL_LAYERS}\n"
        f"Max. bereik: {MAX_RANGE_KM} km\n"
        "====================\n"
        "BESTEMMINGEN\n"
        "Rotterdam\n"
        "Delft\n"
        "Amsterdam\n"
        "====================\n"
        "NETWERK: ACTIEF\n"
        "MODUS: SIMULATIE\n"
        "VEILIGHEID: NIET GEVALIDEERD"
    )

    layer_counts = np.bincount(
        layers,
        minlength=TOTAL_LAYERS
    )

    layer_text = "DRONES PER VISUELE LAAG\n"
    layer_text += "====================\n"

    for i in range(TOTAL_LAYERS):

        layer_text += (
            f"Laag {i+1}: "
            f"{layer_counts[i]} drones\n"
        )

    layer_dashboard.set_text(layer_text)

# ==========================================================
# ANIMATIE
# ==========================================================

def animate(frame):

    x, y, z = calculate_positions(frame)

    drones._offsets3d = (x, y, z)

    update_dashboard(frame)

    # Camera blijft stabiel en kan met de muis gedraaid worden.
    return drones, dashboard, layer_dashboard


animation = FuncAnimation(
    fig,
    animate,
    frames=100000,
    interval=40,
    blit=False,
    cache_frame_data=False
)

# ==========================================================
# START
# ==========================================================

print("======================================")
print(" ALBATROSS AIRSPACE NETHERLANDS 3.0")
print("======================================")
print("Drones:", TOTAL_DRONES)
print("Hoogtelagen:", TOTAL_LAYERS)
print("Steden: Rotterdam, Delft, Amsterdam")
print("3D raster: actief")
print("Animatie: actief")
print("======================================")
print("Gebruik de muis om de camera te draaien.")
print("Sluit het venster om de simulatie te stoppen.")

plt.show()
