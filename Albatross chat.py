
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random

# =================================
# ALBATROSS DELFT - LUCHTRUIM SIMULATIE
# =================================

AANTAL_AUTO = 20
AANTAL_LAGEN = 5
RASTER_GROOTTE = 20

# Hoogtes van de vlieglagen (meters)
HOOGTES = [100, 150, 200, 250, 300]

# Kleuren per vliegl aag
KLEUREN = ['red', 'blue', 'green', 'orange', 'purple']

# Opstijgpunten
STEDEN = {
    "Delft": (3, 3),
    "Amsterdam": (17, 17),
    "Rotterdam": (3, 17)
}

# =================================
# AUTO'S AANMAKEN
# =================================

autos = []

for i in range(AANTAL_AUTO):

    stad_naam = list(STEDEN.keys())[i % 3]
    start_x, start_y = STEDEN[stad_naam]

    laag = i % AANTAL_LAGEN

    auto = {
        "x": start_x,
        "y": start_y,
        "z": HOOGTES[laag],
        "laag": laag,
        "kleur": KLEUREN[laag],
        "stad": stad_naam,
        "richting": random.choice([-1, 1])
    }

    autos.append(auto)

# =================================
# FIGUUR INSTELLEN
# =================================

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

ax.set_title(
    "ALBATROSS DELFT - FUTURE AIRSPACE",
    fontsize=16
)

ax.set_xlabel("X - Oost / West")
ax.set_ylabel("Y - Noord / Zuid")
ax.set_zlabel("Hoogte (meters)")

ax.set_xlim(0, RASTER_GROOTTE)
ax.set_ylim(0, RASTER_GROOTTE)
ax.set_zlim(50, 350)

# =================================
# RASTER TEKENEN
# =================================

for x in range(RASTER_GROOTTE + 1):
    ax.plot(
        [x, x],
        [0, RASTER_GROOTTE],
        [50, 50],
        color='lightgray',
        alpha=0.5
    )

for y in range(RASTER_GROOTTE + 1):
    ax.plot(
        [0, RASTER_GROOTTE],
        [y, y],
        [50, 50],
        color='lightgray',
        alpha=0.5
    )

# =================================
# OPSTIJGPUNTEN TEKENEN
# =================================

for stad, positie in STEDEN.items():

    ax.scatter(
        positie[0],
        positie[1],
        50,
        color='black',
        s=100,
        marker='^'
    )

    ax.text(
        positie[0],
        positie[1],
        65,
        stad,
        fontsize=10
    )

# =================================
# VLIEGLAGEN TEKENEN
# =================================

for hoogte, kleur in zip(HOOGTES, KLEUREN):

    ax.plot(
        [0, RASTER_GROOTTE],
        [0, 0],
        [hoogte, hoogte],
        color=kleur,
        alpha=0.15
    )

# =================================
# AUTO'S WEERGEVEN
# =================================

punten = ax.scatter(
    [auto["x"] for auto in autos],
    [auto["y"] for auto in autos],
    [auto["z"] for auto in autos],
    c=[auto["kleur"] for auto in autos],
    s=100,
    marker='o'
)

# =================================
# ANIMATIE
# =================================

def update(frame):

    nieuwe_x = []
    nieuwe_y = []
    nieuwe_z = []

    for auto in autos:

        # Auto beweegt door het raster
        auto["x"] += 0.035 * auto["richting"]

        # Keer om bij de rand van het luchtruim
        if auto["x"] >= 19:
            auto["richting"] = -1

        if auto["x"] <= 1:
            auto["richting"] = 1

        # Kleine beweging in noord-zuidrichting
        auto["y"] += random.choice([-1, 0, 1]) * 0.01

        # Houd auto binnen het raster
        auto["y"] = max(0, min(20, auto["y"]))

        nieuwe_x.append(auto["x"])
        nieuwe_y.append(auto["y"])
        nieuwe_z.append(auto["z"])

    punten._offsets3d = (
        np.array(nieuwe_x),
        np.array(nieuwe_y),
        np.array(nieuwe_z)
    )

    ax.set_title(
        f"ALBATROSS DELFT - FUTURE AIRSPACE | Tijdstap: {frame}"
    )

    return punten,

# =================================
# START SIMULATIE
# =================================

animatie = FuncAnimation(
    fig,
    update,
    frames=10000,
    interval=40,
    blit=False
)

plt.show()
