
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==================================================
# ALBATROSS DELFT - NEDERLANDSE LUCHTRUIM SIMULATIE
# ==================================================

AANTAL_AUTOS = 20

VLIEGSNELHEID = 0.0009
STIJGSNELHEID = 8

VLIEGLAGEN = [100, 150, 200, 250, 300]

KLEUREN = [
    "red", "blue", "green", "orange", "purple"
]

# ==================================================
# ECHTE GEOGRAFISCHE COORDINATEN
# Lengtegraad, breedtegraad
# ==================================================

STEDEN = {
    "Delft": (4.3571, 52.0116),
    "Amsterdam": (4.9041, 52.3676),
    "Rotterdam": (4.4777, 51.9244),
    "Oudenbosch": (4.5290, 51.5890)
}

STAD_KLEUREN = {
    "Delft": "blue",
    "Amsterdam": "red",
    "Rotterdam": "green",
    "Oudenbosch": "orange"
}

# ==================================================
# SCHEMATISCHE KAART VAN NEDERLAND
# Vereenvoudigde landsgrens, geen officiële kaartdata
# ==================================================

NEDERLAND_X = [
    3.36, 3.42, 3.50, 3.60, 3.72, 3.83,
    3.90, 4.00, 4.08, 4.20, 4.30, 4.42,
    4.55, 4.68, 4.82, 4.95, 5.10, 5.20,
    5.35, 5.50, 5.65, 5.75, 5.78, 5.65,
    5.48, 5.30, 5.12, 4.95, 4.80, 4.65,
    4.48, 4.30, 4.10, 3.95, 3.80, 3.65,
    3.50, 3.40, 3.36
]

NEDERLAND_Y = [
    51.30, 51.45, 51.60, 51.75, 51.85,
    51.95, 52.10, 52.25, 52.40, 52.55,
    52.70, 52.85, 53.00, 53.15, 53.25,
    53.30, 53.25, 53.15, 53.00, 52.85,
    52.70, 52.50, 52.30, 52.10, 51.95,
    51.80, 51.70, 51.60, 51.50, 51.42,
    51.38, 51.35, 51.33, 51.32, 51.31,
    51.30, 51.30, 51.30, 51.30
]

# ==================================================
# AUTO MAKEN
# ==================================================

def maak_auto():

    vertrek = random.choice(list(STEDEN.keys()))

    bestemming = random.choice([
        stad for stad in STEDEN
        if stad != vertrek
    ])

    laag = random.randrange(len(VLIEGLAGEN))

    lon, lat = STEDEN[vertrek]

    return {
        "lon": lon,
        "lat": lat,
        "hoogte": 0,

        "vertrek": vertrek,
        "bestemming": bestemming,

        "doelhoogte": VLIEGLAGEN[laag],
        "kleur": KLEUREN[laag],

        "fase": "stijgen",
        "route_x": [lon],
        "route_y": [lat]
    }


autos = [maak_auto() for _ in range(AANTAL_AUTOS)]

# ==================================================
# FIGUUR MAKEN
# ==================================================

fig = plt.figure(figsize=(15, 9))

kaart = fig.add_subplot(121)
dashboard = fig.add_subplot(122)

fig.suptitle(
    "ALBATROSS DELFT - NEDERLANDS LUCHTRUIM",
    fontsize=18,
    fontweight="bold"
)

# ==================================================
# KAART INSTELLEN
# ==================================================

kaart.set_facecolor("#B9DDF2")

kaart.fill(
    NEDERLAND_X,
    NEDERLAND_Y,
    color="#DDE8CB",
    edgecolor="black",
    linewidth=2,
    zorder=1
)

kaart.set_xlim(3.3, 5.9)
kaart.set_ylim(51.3, 53.4)

kaart.set_aspect(1.0 / np.cos(np.radians(52)))

kaart.set_title(
    "Nederland - Vliegende auto's",
    fontsize=14
)

kaart.set_xlabel("Lengtegraad (X)")
kaart.set_ylabel("Breedtegraad (Y)")

kaart.set_xticks(np.arange(3.5, 6.0, 0.25))
kaart.set_yticks(np.arange(51.3, 53.5, 0.25))

kaart.grid(
    True,
    linestyle=":",
    alpha=0.5
)

# ==================================================
# STEDEN TEKENEN
# ==================================================

for stad, positie in STEDEN.items():

    lon, lat = positie

    kaart.scatter(
        lon,
        lat,
        color=STAD_KLEUREN[stad],
        marker="^",
        s=160,
        edgecolors="black",
        zorder=5
    )

    kaart.annotate(
        stad,
        (lon, lat),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

# ==================================================
# VLIEGROUTES
# ==================================================

route_lijnen = []

for auto in autos:

    lijn, = kaart.plot(
        [],
        [],
        color=auto["kleur"],
        alpha=0.35,
        linewidth=1.2,
        zorder=2
    )

    route_lijnen.append(lijn)

# ==================================================
# AUTO'S TEKENEN
# ==================================================

punten = kaart.scatter(
    [a["lon"] for a in autos],
    [a["lat"] for a in autos],
    s=65,
    c=[a["kleur"] for a in autos],
    edgecolors="black",
    zorder=10
)

# ==================================================
# DASHBOARD
# ==================================================

dashboard.axis("off")

dashboard.set_title(
    "VLUCHTINFORMATIE",
    fontsize=15,
    fontweight="bold"
)

status = dashboard.text(
    0.02,
    0.97,
    "",
    fontsize=9,
    verticalalignment="top",
    family="monospace",
    transform=dashboard.transAxes
)

# ==================================================
# ANIMATIE
# ==================================================

def update(frame):

    for auto in autos:

        # OPSTIJGEN
        if auto["fase"] == "stijgen":

            auto["hoogte"] += STIJGSNELHEID

            if auto["hoogte"] >= auto["doelhoogte"]:

                auto["hoogte"] = auto["doelhoogte"]
                auto["fase"] = "vliegen"

        # VLIEGEN
        elif auto["fase"] == "vliegen":

            doel_lon, doel_lat = STEDEN[
                auto["bestemming"]
            ]

            dx = doel_lon - auto["lon"]
            dy = doel_lat - auto["lat"]

            # Corrigeer de lengtegraad voor de breedtegraad
            afstand = np.sqrt(
                (dx * np.cos(np.radians(auto["lat"])))**2
                + dy**2
            )

            if afstand > VLIEGSNELHEID:

                stap = VLIEGSNELHEID / afstand

                auto["lon"] += (
                    dx * stap /
                    np.cos(np.radians(auto["lat"]))
                )

                auto["lat"] += dy * stap

                auto["route_x"].append(auto["lon"])
                auto["route_y"].append(auto["lat"])

            else:

                auto["lon"] = doel_lon
                auto["lat"] = doel_lat

                auto["route_x"].append(doel_lon)
                auto["route_y"].append(doel_lat)

                auto["fase"] = "landen"

        # LANDEN
        elif auto["fase"] == "landen":

            auto["hoogte"] -= STIJGSNELHEID

            if auto["hoogte"] <= 0:

                auto["hoogte"] = 0

                nieuwe = maak_auto()

                auto.update(nieuwe)

    # AUTO-POSITIES BIJWERKEN
    punten.set_offsets(
        np.column_stack((
            [a["lon"] for a in autos],
            [a["lat"] for a in autos]
        ))
    )

    punten.set_color([
        a["kleur"] for a in autos
    ])

    # ROUTES BIJWERKEN
    for i, auto in enumerate(autos):

        route_lijnen[i].set_data(
            auto["route_x"],
            auto["route_y"]
        )

    # DASHBOARD BIJWERKEN
    stijgen = sum(
        a["fase"] == "stijgen" for a in autos
    )

    vliegen = sum(
        a["fase"] == "vliegen" for a in autos
    )

    landen = sum(
        a["fase"] == "landen" for a in autos
    )

    regels = [
        "ALBATROSS DELFT",
        "================================",
        "",
        f"Totaal aantal auto's: {AANTAL_AUTOS}",
        f"Opstijgen: {stijgen}",
        f"Onderweg:  {vliegen}",
        f"Landingen: {landen}",
        "",
        "AFMETINGEN",
        "--------------------------------",
        "Lengte:  geografisch",
        "Breedte: geografisch",
        "Hoogte:  0-300 meter",
        "",
        "VLIEGGEGEVENS",
        "================================"
    ]

    for i, auto in enumerate(autos):

        regels.append(
            f"Auto {i+1:02d}: "
            f"{auto['vertrek'][:8]} -> "
            f"{auto['bestemming'][:8]}"
        )

        regels.append(
            f" Hoogte: {auto['hoogte']:3.0f} m "
            f"| {auto['fase'].upper()}"
        )

    status.set_text("\n".join(regels))

    return punten, status, *route_lijnen


# ==================================================
# LEGENDA
# ==================================================

for i, hoogte in enumerate(VLIEGLAGEN):

    kaart.scatter(
        [],
        [],
        color=KLEUREN[i],
        label=f"Vlieglaag {hoogte} meter"
    )

kaart.legend(
    loc="lower left",
    fontsize=8
)

# ==================================================
# STARTEN
# ==================================================

animatie = FuncAnimation(
    fig,
    update,
    interval=40,
    frames=10000,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()

