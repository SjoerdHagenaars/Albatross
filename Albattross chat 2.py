
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==========================================
# ALBATROSS DELFT - LUCHTRUIM SIMULATIE
# ==========================================

AANTAL_AUTOS = 20
HOOGTES = [100, 150, 200, 250, 300]

# Afmetingen van het luchtruim in meters
LENGTE = 20000
BREEDTE = 20000
MAX_HOOGTE = 350

# Snelheid van de animatie
SNELHEID = 0.012

KLEUREN = [
    "red", "blue", "green", "orange", "purple"
]

# ==========================================
# OPSTIJG- EN LANDINGSPUNTEN
# ==========================================

STEDEN = {
    "Delft": (3000, 3000),
    "Amsterdam": (17000, 17000),
    "Rotterdam": (3000, 17000)
}

LANDINGSPUNTEN = list(STEDEN.values())

# ==========================================
# AUTO'S AANMAKEN
# ==========================================

autos = []

def nieuwe_auto():

    stad = random.choice(list(STEDEN.keys()))
    start = STEDEN[stad]

    bestemming = random.choice(LANDINGSPUNTEN)

    while bestemming == start:
        bestemming = random.choice(LANDINGSPUNTEN)

    laag = random.randint(0, len(HOOGTES)-1)

    return {
        "x": start[0],
        "y": start[1],
        "z": 0,

        "start": start,
        "bestemming": bestemming,

        "doel_hoogte": HOOGTES[laag],
        "kleur": KLEUREN[laag],

        "fase": "stijgen"
    }


for i in range(AANTAL_AUTOS):
    autos.append(nieuwe_auto())

# ==========================================
# FIGUUR INSTELLEN
# ==========================================

fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection="3d")

ax.set_title(
    "ALBATROSS DELFT - 3D LUCHTRUIM",
    fontsize=16
)

ax.set_xlabel("LENGTE X (meter)")
ax.set_ylabel("BREEDTE Y (meter)")
ax.set_zlabel("HOOGTE Z (meter)")

ax.set_xlim(0, LENGTE)
ax.set_ylim(0, BREEDTE)
ax.set_zlim(0, MAX_HOOGTE)

# Verhoudingen van de assen
ax.set_box_aspect((2, 2, 0.5))

# ==========================================
# RASTER OP DE GROND
# ==========================================

for x in np.linspace(0, LENGTE, 11):
    ax.plot(
        [x, x],
        [0, BREEDTE],
        [0, 0],
        color="gray",
        alpha=0.4
    )

for y in np.linspace(0, BREEDTE, 11):
    ax.plot(
        [0, LENGTE],
        [y, y],
        [0, 0],
        color="gray",
        alpha=0.4
    )

# ==========================================
# VLIEGLAGEN
# ==========================================

for hoogte, kleur in zip(HOOGTES, KLEUREN):

    ax.plot(
        [0, LENGTE],
        [0, 0],
        [hoogte, hoogte],
        color=kleur,
        alpha=0.35,
        linewidth=2
    )

    ax.text(
        0,
        0,
        hoogte,
        f"{hoogte} m",
        color=kleur
    )

# ==========================================
# OPSTIJGPUNTEN
# ==========================================

for stad, positie in STEDEN.items():

    ax.scatter(
        positie[0],
        positie[1],
        0,
        color="black",
        marker="^",
        s=150,
        label=stad
    )

    ax.text(
        positie[0],
        positie[1],
        10,
        stad,
        color="black",
        fontsize=10
    )

# ==========================================
# AUTO'S WEERGEVEN
# ==========================================

punten = ax.scatter(
    [a["x"] for a in autos],
    [a["y"] for a in autos],
    [a["z"] for a in autos],
    c=[a["kleur"] for a in autos],
    s=90,
    edgecolors="black"
)

# ==========================================
# ANIMATIE
# ==========================================

def update(frame):

    for auto in autos:

        if auto["fase"] == "stijgen":

            auto["z"] += 4

            if auto["z"] >= auto["doel_hoogte"]:
                auto["z"] = auto["doel_hoogte"]
                auto["fase"] = "vliegen"

        elif auto["fase"] == "vliegen":

            dx = auto["bestemming"][0] - auto["x"]
            dy = auto["bestemming"][1] - auto["y"]

            afstand = np.sqrt(dx**2 + dy**2)

            if afstand > 100:

                auto["x"] += (dx / afstand) * 140
                auto["y"] += (dy / afstand) * 140

            else:
                auto["x"] = auto["bestemming"][0]
                auto["y"] = auto["bestemming"][1]
                auto["fase"] = "landen"

        elif auto["fase"] == "landen":

            auto["z"] -= 4

            if auto["z"] <= 0:

                auto["z"] = 0

                # Nieuwe willekeurige start en bestemming
                nieuwe = nieuwe_auto()

                auto.update(nieuwe)

    # Posities bijwerken
    punten._offsets3d = (
        [a["x"] for a in autos],
        [a["y"] for a in autos],
        [a["z"] for a in autos]
    )

    ax.set_title(
        f"ALBATROSS DELFT | Simulatietijd: {frame}"
    )

    return punten,

# ==========================================
# START SIMULATIE
# ==========================================

animatie = FuncAnimation(
    fig,
    update,
    interval=30,
    frames=10000,
    blit=False
)

plt.show()
