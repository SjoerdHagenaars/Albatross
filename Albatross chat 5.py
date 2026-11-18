
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patheffects import withStroke
from urllib.request import urlopen
import json
import random

# =====================================================
# ALBATROSS DELFT - 3D NEDERLAND AIRSPACE
# =====================================================

AANTAL_AUTOS = 20

VLIEGSNELHEID = 4.0       # Kilometer per animatiestap
STIJGSNELHEID = 12       # Meter per animatiestap
MAX_HOOGTE = 350

VLIEGLAGEN = [100, 150, 200, 250, 300]

KLEUREN = [
    "red", "blue", "limegreen", "orange", "purple"
]

# =====================================================
# NEDERLANDSE STEDEN
# Coördinaten: lengtegraad, breedtegraad
# Gewicht: relatieve hoeveelheid vliegverkeer
# =====================================================

STEDEN = {
    "Amsterdam":  (4.9041, 52.3676, 20),
    "Rotterdam":  (4.4777, 51.9244, 17),
    "Utrecht":    (5.1214, 52.0907, 14),
    "Den Haag":   (4.3007, 52.0705, 13),
    "Eindhoven":  (5.4697, 51.4416, 11),
    "Groningen":  (6.5665, 53.2194, 8),
    "Tilburg":    (5.0913, 51.5555, 7),
    "Breda":      (4.7759, 51.5719, 7),
    "Nijmegen":   (5.8528, 51.8126, 6),
    "Enschede":   (6.8958, 52.2215, 5),
    "Apeldoorn":  (5.9699, 52.2112, 5),
    "Zwolle":     (6.0944, 52.5168, 5),
    "Maastricht": (5.6900, 50.8514, 5),
    "Leiden":     (4.4970, 52.1601, 5),
    "Delft":      (4.3571, 52.0116, 4),
    "Oudenbosch": (4.5290, 51.5890, 2),
    "Alkmaar":    (4.7486, 52.6324, 3),
    "Leeuwarden": (5.7999, 53.2012, 3),
    "Den Bosch":  (5.3037, 51.6978, 4),
    "Dordrecht":  (4.6900, 51.8133, 3)
}

STAD_NAMEN = list(STEDEN.keys())

# Gewichten voor willekeurige vertrek- en bestemmingskeuze
GEWICHTEN = [
    STEDEN[stad][2] for stad in STAD_NAMEN
]

# =====================================================
# ECHTE LANDKAART VAN NEDERLAND OPHALEN
# =====================================================

def laad_nederland():

    print("Nederlandse landkaart ophalen...")

    url = (
        "https://raw.githubusercontent.com/"
        "datasets/geo-countries/master/data/countries.geojson"
    )

    with urlopen(url, timeout=30) as response:
        gegevens = json.loads(response.read().decode("utf-8"))

    for feature in gegevens["features"]:

        naam = feature["properties"].get("ADMIN", "")

        if naam == "Netherlands":

            geometrie = feature["geometry"]

            if geometrie["type"] == "Polygon":
                return [geometrie["coordinates"][0]]

            elif geometrie["type"] == "MultiPolygon":
                return [
                    polygoon[0]
                    for polygoon in geometrie["coordinates"]
                ]

    raise ValueError("Nederland niet gevonden in kaartgegevens")


# =====================================================
# AUTO MAKEN
# =====================================================

def kies_stad():

    return random.choices(
        STAD_NAMEN,
        weights=GEWICHTEN,
        k=1
    )[0]


def maak_auto(start_stad=None):

    if start_stad is None:
        vertrek = kies_stad()
    else:
        vertrek = start_stad

    bestemming = kies_stad()

    while bestemming == vertrek:
        bestemming = kies_stad()

    lon, lat, _ = STEDEN[vertrek]

    laag = random.randrange(len(VLIEGLAGEN))

    return {
        "lon": lon,
        "lat": lat,
        "z": 0,

        "vertrek": vertrek,
        "bestemming": bestemming,

        "doelhoogte": VLIEGLAGEN[laag],
        "kleur": KLEUREN[laag],

        "fase": "stijgen",

        "route_lon": [lon],
        "route_lat": [lat]
    }


# =====================================================
# 20 AUTO'S MAKEN
# =====================================================

autos = [
    maak_auto()
    for _ in range(AANTAL_AUTOS)
]

# =====================================================
# LANDKAART LADEN
# =====================================================

POLYGONEN = laad_nederland()

# =====================================================
# FIGUUR INSTELLEN
# =====================================================

fig = plt.figure(figsize=(17, 11))

ax = fig.add_subplot(121, projection="3d")
dashboard = fig.add_subplot(122)

dashboard.axis("off")

fig.suptitle(
    "ALBATROSS DELFT - 3D NEDERLAND AIRSPACE",
    fontsize=19,
    fontweight="bold"
)

# =====================================================
# NEDERLAND ALS 3D LANDOPPERVLAK
# =====================================================

for polygoon in POLYGONEN:

    coordinaten = np.array(polygoon)

    # Alleen het Europese deel van Nederland
    if not (
        3 < np.mean(coordinaten[:, 0]) < 8
        and 50 < np.mean(coordinaten[:, 1]) < 54
    ):
        continue

    punten = [
        (lon, lat, 0)
        for lon, lat in coordinaten
    ]

    vlak = Poly3DCollection(
        [punten],
        facecolor="#B8D7A0",
        edgecolor="#344C32",
        linewidth=0.8,
        alpha=0.95
    )

    ax.add_collection3d(vlak)

# =====================================================
# KAARTGRENZEN
# =====================================================

ax.set_xlim(3.2, 7.4)
ax.set_ylim(50.7, 53.7)
ax.set_zlim(0, MAX_HOOGTE)

ax.set_box_aspect((4.2, 3.0, 1.1))

ax.set_xlabel("LENGTEGRAAD (X)")
ax.set_ylabel("BREEDTEGRAAD (Y)")
ax.set_zlabel("HOOGTE (meter)")

ax.set_title(
    "Nederland - 3D Luchtruim",
    fontsize=14
)

ax.view_init(elev=32, azim=-65)

# =====================================================
# STEDEN WEERGEVEN
# =====================================================

stad_punten = {}

for stad, gegevens in STEDEN.items():

    lon, lat, gewicht = gegevens

    # Grotere marker voor steden met meer verkeer
    grootte = 25 + gewicht * 3

    punt = ax.scatter(
        lon,
        lat,
        0,
        color="darkred",
        s=grootte,
        edgecolors="black",
        marker="o",
        depthshade=False
    )

    ax.text(
        lon,
        lat,
        8,
        stad,
        fontsize=7,
        fontweight="bold",
        color="black"
    )

    stad_punten[stad] = punt

# =====================================================
# VLIEGLAGEN ALS HORIZONTALE VLAKKEN
# =====================================================

for i, hoogte in enumerate(VLIEGLAGEN):

    # Een subtiele lijn op de gekozen hoogte
    ax.plot(
        [3.2, 7.4],
        [50.7, 50.7],
        [hoogte, hoogte],
        color=KLEUREN[i],
        alpha=0.65,
        linewidth=1.5
    )

# =====================================================
# AUTO'S WEERGEVEN
# =====================================================

punten = ax.scatter(
    [a["lon"] for a in autos],
    [a["lat"] for a in autos],
    [a["z"] for a in autos],
    c=[a["kleur"] for a in autos],
    s=65,
    edgecolors="black",
    depthshade=True
)

# Vluchtroutes
route_lijnen = []

for auto in autos:

    lijn, = ax.plot(
        [],
        [],
        [],
        color=auto["kleur"],
        linewidth=1,
        alpha=0.6
    )

    route_lijnen.append(lijn)

# =====================================================
# DASHBOARD
# =====================================================

dashboard.set_title(
    "VLUCHTINFORMATIE",
    fontsize=16,
    fontweight="bold"
)

status = dashboard.text(
    0.02,
    0.97,
    "",
    fontsize=8.5,
    family="monospace",
    verticalalignment="top",
    transform=dashboard.transAxes
)

# =====================================================
# AFSTAND BEREKENEN
# =====================================================

def afstand(lon1, lat1, lon2, lat2):

    dx = (
        (lon2 - lon1)
        * 111.32
        * np.cos(np.radians(lat1))
    )

    dy = (lat2 - lat1) * 111.32

    return dx, dy


# =====================================================
# ANIMATIE
# =====================================================

def update(frame):

    for auto in autos:

        # ---------------------------------------------
        # OPSTIJGEN
        # ---------------------------------------------

        if auto["fase"] == "stijgen":

            auto["z"] += STIJGSNELHEID

            if auto["z"] >= auto["doelhoogte"]:

                auto["z"] = auto["doelhoogte"]
                auto["fase"] = "vliegen"

        # ---------------------------------------------
        # VLIEGEN
        # ---------------------------------------------

        elif auto["fase"] == "vliegen":

            doel_lon, doel_lat, _ = STEDEN[
                auto["bestemming"]
            ]

            dx, dy = afstand(
                auto["lon"],
                auto["lat"],
                doel_lon,
                doel_lat
            )

            afstand_tot_doel = np.sqrt(dx**2 + dy**2)

            if afstand_tot_doel > VLIEGSNELHEID:

                stap = VLIEGSNELHEID / afstand_tot_doel

                auto["lon"] += (
                    (dx * stap) /
                    (
                        111.32 *
                        np.cos(np.radians(auto["lat"]))
                    )
                )

                auto["lat"] += (
                    dy * stap / 111.32
                )

                auto["route_lon"].append(auto["lon"])
                auto["route_lat"].append(auto["lat"])

            else:

                auto["lon"] = doel_lon
                auto["lat"] = doel_lat

                auto["fase"] = "landen"

        # ---------------------------------------------
        # LANDEN
        # ---------------------------------------------

        elif auto["fase"] == "landen":

            auto["z"] -= STIJGSNELHEID

            if auto["z"] <= 0:

                auto["z"] = 0

                oude_bestemming = auto["bestemming"]

                nieuwe = maak_auto(
                    start_stad=oude_bestemming
                )

                auto.update(nieuwe)

    # =================================================
    # POSITIES AUTO'S BIJWERKEN
    # =================================================

    punten._offsets3d = (
        [a["lon"] for a in autos],
        [a["lat"] for a in autos],
        [a["z"] for a in autos]
    )

    punten.set_color([
        a["kleur"] for a in autos
    ])

    # =================================================
    # ROUTES BIJWERKEN
    # =================================================

    for i, auto in enumerate(autos):

        hoogte = auto["z"]

        route_lijnen[i].set_data_3d(
            auto["route_lon"],
            auto["route_lat"],
            [hoogte] * len(auto["route_lon"])
        )

    # =================================================
    # DASHBOARD BIJWERKEN
    # =================================================

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
        f"Vliegende auto's: {AANTAL_AUTOS}",
        f"Stijgen:          {stijgen}",
        f"Onderweg:         {vliegen}",
        f"Landen:           {landen}",
        "",
        "VLIEGHOOGTES",
        "--------------------------------",
        "Laag 1: 100 meter",
        "Laag 2: 150 meter",
        "Laag 3: 200 meter",
        "Laag 4: 250 meter",
        "Laag 5: 300 meter",
        "",
        "VERKEERSVERDELING",
        "--------------------------------",
        "Amsterdam:  hoogste verkeer",
        "Rotterdam:  veel verkeer",
        "Utrecht:    veel verkeer",
        "Den Haag:   veel verkeer",
        "Delft:      minder verkeer",
        "Oudenbosch: weinig verkeer",
        "",
        "VLUCHTEN PER AUTO",
        "================================"
    ]

    for i, auto in enumerate(autos):

        regels.append(
            f"Auto {i+1:02d}: "
            f"{auto['vertrek'][:8]:8s}"
            f" -> {auto['bestemming'][:8]:8s}"
        )

        regels.append(
            f" Hoogte: {auto['z']:3.0f} m "
            f"| {auto['fase'].upper()}"
        )

    status.set_text("\n".join(regels))

    return punten, status, *route_lijnen


# =====================================================
# LEGENDA
# =====================================================

for i, hoogte in enumerate(VLIEGLAGEN):

    ax.scatter(
        [],
        [],
        [],
        color=KLEUREN[i],
        label=f"Hoogte {hoogte} meter"
    )

ax.legend(
    loc="upper left",
    fontsize=8
)

# =====================================================
# SIMULATIE STARTEN
# =====================================================

animatie = FuncAnimation(
    fig,
    update,
    interval=30,
    frames=100000,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
