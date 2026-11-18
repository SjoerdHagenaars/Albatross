
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==================================================
# ALBATROSS DELFT - 3D LUCHTRUIM SIMULATIE
# ==================================================

AANTAL_AUTOS = 20

# Afmetingen luchtruim in meters
LENGTE = 20000
BREEDTE = 20000
MAX_HOOGTE = 350

# Hoogtes van de vlieglagen
VLIEGLAGEN = [100, 150, 200, 250, 300]

# Kleuren per vliegl aag
KLEUREN = [
    "red",
    "blue",
    "green",
    "orange",
    "purple"
]

# Bewegingssnelheid in meters per animatiestap
VLIEGSNELHEID = 180
STIJGSNELHEID = 8

# ==================================================
# OPSTIJG- EN LANDINGSPUNTEN
# ==================================================

STEDEN = {
    "Delft": (3000, 3000),
    "Amsterdam": (17000, 17000),
    "Rotterdam": (3000, 17000),
    "Oudenbosch": (10000, 5000)
}

STAD_KLEUREN = {
    "Delft": "blue",
    "Amsterdam": "red",
    "Rotterdam": "green",
    "Oudenbosch": "orange"
}

# ==================================================
# NIEUWE AUTO MAKEN
# ==================================================

def maak_auto():

    vertrek = random.choice(list(STEDEN.keys()))

    bestemming = random.choice(
        [stad for stad in STEDEN if stad != vertrek]
    )

    laag = random.randrange(len(VLIEGLAGEN))

    x, y = STEDEN[vertrek]

    return {
        "x": float(x),
        "y": float(y),
        "z": 0.0,

        "vertrek": vertrek,
        "bestemming": bestemming,

        "doelhoogte": VLIEGLAGEN[laag],
        "laag": laag,
        "kleur": KLEUREN[laag],

        "fase": "stijgen"
    }


# ==================================================
# 20 AUTO'S AANMAKEN
# ==================================================

autos = []

for i in range(AANTAL_AUTOS):
    autos.append(maak_auto())


# ==================================================
# 3D FIGUUR MAKEN
# ==================================================

fig = plt.figure(figsize=(14, 10))

ax = fig.add_subplot(111, projection="3d")

ax.set_title(
    "ALBATROSS DELFT - FUTURE AIRSPACE",
    fontsize=17,
    fontweight="bold"
)

ax.set_xlabel("LENGTE X (meter)")
ax.set_ylabel("BREEDTE Y (meter)")
ax.set_zlabel("HOOGTE Z (meter)")

ax.set_xlim(0, LENGTE)
ax.set_ylim(0, BREEDTE)
ax.set_zlim(0, MAX_HOOGTE)

# Verhouding van de 3D-weergave
ax.set_box_aspect((2, 2, 0.7))

# ==================================================
# RASTER TEKENEN
# ==================================================

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

# ==================================================
# VLIEGLAGEN TEKENEN
# ==================================================

for i, hoogte in enumerate(VLIEGLAGEN):

    ax.plot(
        [0, LENGTE],
        [0, 0],
        [hoogte, hoogte],
        color=KLEUREN[i],
        alpha=0.5,
        linewidth=2
    )

    ax.text(
        0,
        0,
        hoogte,
        f" Laag {i+1}: {hoogte} m",
        color=KLEUREN[i],
        fontsize=9
    )

# ==================================================
# STEDEN WEERGEVEN
# ==================================================

for stad, positie in STEDEN.items():

    ax.scatter(
        positie[0],
        positie[1],
        0,
        color=STAD_KLEUREN[stad],
        marker="^",
        s=180,
        edgecolors="black"
    )

    ax.text(
        positie[0],
        positie[1],
        15,
        stad,
        fontsize=11,
        fontweight="bold"
    )

# ==================================================
# AUTO'S WEERGEVEN
# ==================================================

punten = ax.scatter(
    [a["x"] for a in autos],
    [a["y"] for a in autos],
    [a["z"] for a in autos],
    c=[a["kleur"] for a in autos],
    s=100,
    edgecolors="black",
    depthshade=True
)

# ==================================================
# ANIMATIE
# ==================================================

def update(frame):

    for auto in autos:

        # ------------------------------------------
        # FASE 1: OPSTIJGEN
        # ------------------------------------------

        if auto["fase"] == "stijgen":

            auto["z"] += STIJGSNELHEID

            if auto["z"] >= auto["doelhoogte"]:

                auto["z"] = auto["doelhoogte"]
                auto["fase"] = "vliegen"

        # ------------------------------------------
        # FASE 2: VLIEGEN NAAR BESTEMMING
        # ------------------------------------------

        elif auto["fase"] == "vliegen":

            doel_x, doel_y = STEDEN[auto["bestemming"]]

            dx = doel_x - auto["x"]
            dy = doel_y - auto["y"]

            afstand = np.sqrt(dx**2 + dy**2)

            if afstand > VLIEGSNELHEID:

                auto["x"] += (
                    dx / afstand
                ) * VLIEGSNELHEID

                auto["y"] += (
                    dy / afstand
                ) * VLIEGSNELHEID

            else:

                auto["x"] = doel_x
                auto["y"] = doel_y

                auto["fase"] = "landen"

        # ------------------------------------------
        # FASE 3: LANDEN
        # ------------------------------------------

        elif auto["fase"] == "landen":

            auto["z"] -= STIJGSNELHEID

            if auto["z"] <= 0:

                auto["z"] = 0

                # Nieuwe willekeurige vlucht
                nieuwe = maak_auto()

                auto.update(nieuwe)

    # Posities van alle auto's aanpassen
    punten._offsets3d = (
        [a["x"] for a in autos],
        [a["y"] for a in autos],
        [a["z"] for a in autos]
    )

    # Teller bijwerken
    ax.set_title(
        f"ALBATROSS DELFT - FUTURE AIRSPACE | "
        f"Simulatietijd: {frame}"
    )

    return punten,


# ==================================================
# LEGENDA
# ==================================================

for i, hoogte in enumerate(VLIEGLAGEN):

    ax.scatter(
        [],
        [],
        [],
        color=KLEUREN[i],
        label=f"Vlieglaag {i+1} ({hoogte} m)"
    )

ax.legend(loc="upper left")

# ==================================================
# SIMULATIE STARTEN
# ==================================================

animatie = FuncAnimation(
    fig,
    update,
    interval=30,
    frames=10000,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
