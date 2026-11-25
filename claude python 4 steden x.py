"""
Team Albatross - luchtruim boven Nederland in 3D, met kaart, 4 steden en verboden zones.
Alleen numpy en matplotlib nodig. Plakken in een leeg .py bestand en op Run drukken.

Bediening in het venster:  linkermuisknop slepen = draaien | rechtermuisknop slepen = zoomen
                           spatie = pauze | pijl omhoog/omlaag = sneller/langzamer
                           z / x = hoogte meer / minder uitvergroten
"""
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================ INSTELLINGEN ============================
ADOPTIE = 0.01            # aandeel autoreizigers dat vliegt (0.01 = 1%)
AUTOS_TOTAAL_PER_DAG = 60000  # autoritten tussen deze steden samen, beide richtingen (PLACEHOLDER!)
SNELHEID_KMH = 160        # kruissnelheid
RANGE_KM = 200            # actieradius volle batterij
RESERVE = 0.20            # veiligheidsreserve: vluchten mogen max. RANGE*(1-RESERVE) lang zijn
PERSONEN = 2              # personen per voertuig
LAGEN_M = [150, 210, 270, 330, 390, 450]   # hoogtelagen; heen = even, terug = oneven
OFFSET_M = 400            # zijdelingse afstand van de rijbaan tot de hartlijn
ZONE_MARGE_M = 1500       # extra afstand die routes tot een verboden zone houden
H_SEP, V_SEP = 300, 50    # minimale scheiding horizontaal / verticaal [m]
LANE_HEADWAY_S = 60       # minimale tijd tussen twee voertuigen op dezelfde laag
VP_SLOT_S = 30            # minimale tijd tussen twee starts/landingen per stad
MAX_WACHT_S = 1200        # langer wachten op de grond = vraag valt weg
KLIM_MS = 5.0             # stijg- en daalsnelheid [m/s]
HOOGTE_UITVERGROTING = 60 # alleen voor het plaatje (450 m is niets naast 300 km)
SPEED = 60                # afspeelsnelheid (60 = 1 minuut per seconde)
START_UUR = 8.5           # starttijd van de animatie (8.5 = 08:30)
SEED = 1

# 4 steden ver uit elkaar: naam: (lat, lon, inwoners). Wissel gerust, bijv. Maastricht (50.851, 5.690, 120000)
CITIES = {
    "Groningen": (53.2194, 6.5665, 235000),
    "Amsterdam": (52.3792, 4.9003, 920000),
    "Eindhoven": (51.4400, 5.5000, 235000),
    "Enschede":  (52.2215, 6.8937, 160000)}

# Verboden zones (luchthavens en vliegbases), INDICATIEF: naam, lat, lon, straal [m]
NOFLY = [("Schiphol", 52.309, 4.764, 8000), ("Rotterdam Airport", 51.957, 4.437, 4000),
         ("Lelystad", 52.460, 5.527, 5000), ("Eindhoven AB", 51.450, 5.375, 5000),
         ("Eelde", 53.119, 6.579, 4000), ("Maastricht", 50.912, 5.770, 5000),
         ("Twente", 52.276, 6.874, 3000), ("Volkel", 51.657, 5.707, 5000),
         ("Gilze-Rijen", 51.567, 4.932, 5000), ("Woensdrecht", 51.449, 4.342, 5000),
         ("Leeuwarden", 53.229, 5.761, 5000), ("De Kooy", 52.924, 4.781, 4000),
         ("Deelen", 52.060, 5.890, 4000)]

# Vereenvoudigde omtrek van Nederland (lon, lat), ruwe schets, geen officiele kaart
NL = [(3.37, 51.37), (3.45, 51.53), (3.55, 51.62), (3.69, 51.72), (3.85, 51.80), (3.98, 51.95), (4.12, 51.99),
      (4.27, 52.11), (4.40, 52.20), (4.53, 52.37), (4.58, 52.47), (4.62, 52.60), (4.66, 52.78), (4.70, 52.85),
      (4.79, 52.96), (5.05, 52.93), (5.33, 53.07), (5.42, 53.17), (5.65, 53.30), (5.89, 53.38), (6.20, 53.41),
      (6.45, 53.40), (6.83, 53.45), (6.93, 53.33), (7.03, 53.30), (7.20, 53.19), (7.19, 53.00), (7.07, 52.88),
      (7.05, 52.80), (7.03, 52.65), (7.05, 52.45), (7.00, 52.39), (7.20, 52.30), (7.10, 52.22), (6.97, 52.12),
      (6.95, 51.98), (6.83, 51.88), (6.45, 51.85), (6.12, 51.85), (5.95, 51.78), (6.05, 51.65), (6.17, 51.40),
      (6.15, 51.20), (6.05, 51.05), (6.08, 50.87), (6.02, 50.75), (5.70, 50.76), (5.76, 50.88), (5.74, 51.03),
      (5.70, 51.16), (5.56, 51.22), (5.48, 51.30), (5.20, 51.32), (5.10, 51.30), (4.93, 51.44), (4.78, 51.43),
      (4.55, 51.42), (4.45, 51.47), (4.38, 51.43), (4.25, 51.37), (4.05, 51.28), (3.82, 51.33), (3.55, 51.40)]
IJSSELMEER = [(5.05, 52.94), (5.33, 53.07), (5.37, 52.90), (5.50, 52.82), (5.70, 52.80), (5.62, 52.66), (5.77, 52.60),
              (5.87, 52.57), (5.55, 52.50), (5.43, 52.52), (5.30, 52.70), (5.12, 52.77)]
MARKERMEER = [(5.30, 52.70), (5.43, 52.52), (5.35, 52.40), (5.15, 52.35), (5.00, 52.38), (5.02, 52.50),
              (5.06, 52.64), (5.20, 52.70)]
def eiland(lon, lat, a, b, n=16):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(lon + a * math.cos(t), lat + b * math.sin(t)) for t in th]
WADDEN = [eiland(4.80, 53.08, .10, .12), eiland(5.03, 53.26, .08, .03), eiland(5.35, 53.38, .20, .03),
          eiland(5.75, 53.45, .20, .02), eiland(6.20, 53.48, .08, .015)]

V = SNELHEID_KMH / 3.6    # m/s
REF = (52.20, 5.30)
def xy(lat, lon):         # lat/lon -> meters
    return np.array([(lon - REF[1]) * math.cos(math.radians(REF[0])) * 111320.0,
                     (lat - REF[0]) * 110574.0])

VP = {k: xy(v[0], v[1]) for k, v in CITIES.items()}
CIRK = [(xy(la, lo), r) for _, la, lo, r in NOFLY]          # (midden, straal)
for naam, p in VP.items():
    for (c, r), z in zip(CIRK, NOFLY):
        if np.hypot(*(p - c)) < r + ZONE_MARGE_M:
            print(f"WAARSCHUWING: {naam} ligt te dicht bij verboden zone {z[0]}")

# ============================ ROUTES OM DE ZONES HEEN ============================
def kortste_pad(a, b):
    """Kortste pad van a naar b dat om alle (opgeblazen) cirkels heen loopt (zichtbaarheidsgraaf + Dijkstra)."""
    nodes = [a, b]
    for c, r in CIRK:
        rv = (r + ZONE_MARGE_M) * 1.03 / math.cos(math.pi / 24)
        nodes += [c + rv * np.array([math.cos(2 * math.pi * k / 24), math.sin(2 * math.pi * k / 24)]) for k in range(24)]
    N = np.array(nodes)
    C = np.array([c for c, _ in CIRK]); R = np.array([r + ZONE_MARGE_M for _, r in CIRK])
    ok = np.array([bool(np.all(np.hypot(*(n - C).T) >= R * 0.999)) for n in N])
    dist, vorige, klaar, heap = {0: 0.0}, {}, set(), [(0.0, 0)]
    while heap:
        d, i = heapq.heappop(heap)
        if i in klaar:
            continue
        klaar.add(i)
        if i == 1:
            break
        J = np.array([j for j in range(len(N)) if ok[j] and j != i and j not in klaar])
        q = N[J] - N[i]; ll = (q ** 2).sum(1)
        vrij = np.ones(len(J), dtype=bool)
        for c, r in zip(C, R):
            t = np.clip(((c - N[i]) @ q.T) / ll, 0, 1)
            dd = np.hypot(*(N[i] + t[:, None] * q - c).T)
            vrij &= dd >= r * 0.999
        for j, l in zip(J[vrij], np.sqrt(ll[vrij])):
            if d + l < dist.get(j, 1e18):
                dist[j] = d + l; vorige[j] = i; heapq.heappush(heap, (d + l, j))
    if 1 not in vorige:
        raise SystemExit("Geen route gevonden: een stad ligt (te dicht) bij een verboden zone. Verplaats de stad of verklein de zone.")
    pad, i = [1], 1
    while i != 0:
        i = vorige[i]; pad.append(i)
    return N[pad[::-1]]

def rechts_verschuiven(p, d):      # lijn d meter naar rechts verschuiven (rijden aan de rechterkant)
    seg = np.diff(p, axis=0); t = seg / np.linalg.norm(seg, axis=1)[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]]); out = p.copy()
    out[0] += d * n[0]; out[-1] += d * n[-1]
    for i in range(1, len(p) - 1):
        m = n[i - 1] + n[i]; m /= np.linalg.norm(m); out[i] += d * m / max(m @ n[i], .5)
    return out

ROUTES, namen = {}, list(CITIES)
print("\nRoutes tussen de steden (om de verboden zones heen):")
for i, a in enumerate(namen):
    for j, b in enumerate(namen):
        if j <= i:
            continue
        pad = kortste_pad(VP[a], VP[b])
        L = np.linalg.norm(np.diff(pad, axis=0), axis=1).sum()
        recht = np.linalg.norm(VP[a] - VP[b])
        if L > RANGE_KM * (1 - RESERVE) * 1000:
            print(f"  {a:10s} - {b:10s}: {L/1000:5.0f} km  TE VER voor {RANGE_KM*(1-RESERVE):.0f} km bruikbare range, geen verbinding")
            continue
        print(f"  {a:10s} - {b:10s}: {L/1000:5.0f} km  (omweg {100*(L/recht-1):.0f}% t.o.v. rechte lijn)")
        for o, d, p, richting in ((a, b, pad, 0), (b, a, pad[::-1], 1)):
            lane = rechts_verschuiven(p, OFFSET_M)
            pts = np.vstack([VP[o], lane, VP[d]])
            cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            ROUTES[(o, d)] = dict(pts=pts, cum=cum, L=cum[-1], richting=richting, s_in=cum[1], lane=lane)
if not ROUTES:
    raise SystemExit("Geen enkele verbinding binnen de range, kies steden dichter bij elkaar.")
def afstand_tot_zones(r):
    s = np.arange(0, r["L"], 200.0)
    x, y = np.interp(s, r["cum"], r["pts"][:, 0]), np.interp(s, r["cum"], r["pts"][:, 1])
    return min(np.hypot(x - c[0], y - c[1]).min() - rad for c, rad in CIRK)
minafst = min(afstand_tot_zones(r) for r in ROUTES.values())
print(f"Kleinste afstand van een rijbaan tot een verboden zone: {minafst:.0f} m")

def hoogte(alt, s, L):
    return np.maximum(np.minimum(np.minimum(alt, KLIM_MS / V * s), KLIM_MS / V * (L - s)), 0)

# ============================ VRAAG (zwaartekrachtmodel) ============================
rng = np.random.default_rng(SEED)
UURPROFIEL = np.array([.4, .25, .2, .2, .4, 1.2, 3.5, 6.8, 7.2, 5.5, 4.6, 4.8,
                       5.2, 5.0, 5.3, 6.2, 7.6, 8.0, 6.4, 4.6, 3.2, 2.3, 1.5, 0.9])
UURPROFIEL = UURPROFIEL / UURPROFIEL.sum()
bezetting = {1: .70, 2: .22, 3: .05, 4: .03}                  # personen per auto
ritten_per_auto = sum(p * math.ceil(n / PERSONEN) for n, p in bezetting.items())

gew = {(o, d): CITIES[o][2] * CITIES[d][2] / ROUTES[(o, d)]["L"] for (o, d) in ROUTES}   # inwoners x inwoners / afstand
som = sum(gew.values())
aanvragen = []
for (o, d), w in gew.items():
    per_dag = AUTOS_TOTAAL_PER_DAG * w / som
    for uur in range(24):
        n = rng.poisson(per_dag * UURPROFIEL[uur] * ADOPTIE * ritten_per_auto)
        aanvragen += [(t, o, d) for t in uur * 3600 + rng.uniform(0, 3600, n)]
aanvragen.sort()

# ============================ PLANNING ============================
# Elke vlucht krijgt hoogtelaag + vertrektijd zodat slots vrij zijn en hij nergens te dicht bij een ander komt.
agenda = {}
def vrij(res, t, gap):
    return all(abs(t - x) >= gap for x in agenda.get(res, []))

DT, MAXV = 4.0, 150
STAPPEN = int(100000 / DT)
OX = np.full((STAPPEN, MAXV), np.nan); OY = OX.copy(); OZ = OX.copy(); AANTAL = np.zeros(STAPPEN, dtype=int)
H_PLAN, V_PLAN = 2 * H_SEP, 1.5 * V_SEP          # planningsmarge boven de minimale scheiding

def pad_in_tijd(r, alt, t0):
    ks = np.arange(math.ceil(t0 / DT), math.floor((t0 + r["L"] / V) / DT) + 1)
    s = (ks * DT - t0) * V
    return ks, np.interp(s, r["cum"], r["pts"][:, 0]), np.interp(s, r["cum"], r["pts"][:, 1]), hoogte(alt, s, r["L"])

def geen_conflict(r, alt, t0):
    ks, x, y, z = pad_in_tijd(r, alt, t0)
    if (AANTAL[ks] >= MAXV).any():            # tabel vol = luchtruim te druk op dat moment
        return False
    bots = ((OX[ks] - x[:, None]) ** 2 + (OY[ks] - y[:, None]) ** 2 < H_PLAN ** 2) & (np.abs(OZ[ks] - z[:, None]) < V_PLAN)
    return not bots.any()

def boek(r, alt, t0):
    ks, x, y, z = pad_in_tijd(r, alt, t0)
    plek = AANTAL[ks]; OX[ks, plek] = x; OY[ks, plek] = y; OZ[ks, plek] = z; AANTAL[ks] += 1

vluchten, afgewezen = [], 0
for t_req, o, d in aanvragen:
    r, beste = ROUTES[(o, d)], None
    for laag in range(r["richting"], len(LAGEN_M), 2):
        t = t_req
        while t - t_req <= MAX_WACHT_S:
            t_in, t_aan = t + r["s_in"] / V, t + r["L"] / V
            if (vrij(o, t, VP_SLOT_S) and vrij(d, t_aan, VP_SLOT_S) and vrij((o, d, laag), t_in, LANE_HEADWAY_S)
                    and geen_conflict(r, LAGEN_M[laag], t)):
                break
            t += 5
        else:
            continue
        if beste is None or t_aan < beste[0]:
            beste = (t_aan, t, laag)
    if beste is None:
        afgewezen += 1
        continue
    t_aan, t, laag = beste
    boek(r, LAGEN_M[laag], t)
    agenda.setdefault(o, []).append(t); agenda.setdefault(d, []).append(t_aan)
    agenda.setdefault((o, d, laag), []).append(t + r["s_in"] / V)
    vluchten.append(dict(t0=t, t1=t_aan, r=r, alt=LAGEN_M[laag], req=t_req))

t0s = np.array([f["t0"] for f in vluchten]); t1s = np.array([f["t1"] for f in vluchten])

def positie(f, t):
    r = f["r"]; s = (t - f["t0"]) * V
    return np.interp(s, r["cum"], r["pts"][:, 0]), np.interp(s, r["cum"], r["pts"][:, 1]), float(hoogte(f["alt"], s, r["L"]))

def posities(t):
    """x, y [km], z [m], richting van alle voertuigen die op tijd t in de lucht zijn"""
    X, Y, Z, R = [], [], [], []
    for i in np.nonzero((t0s <= t) & (t <= t1s))[0]:
        x, y, z = positie(vluchten[i], t)
        X.append(x / 1000); Y.append(y / 1000); Z.append(z); R.append(vluchten[i]["r"]["richting"])
    return np.array(X), np.array(Y), np.array(Z), np.array(R)

# ============================ VEILIGHEIDSCONTROLE ============================
def controle(dt=2.0):
    """Zet elk voertuig op tijdstappen van dt seconden en telt paren die te dicht bij elkaar komen (+ zones)."""
    paren, zone = set(), set()
    C = np.array([c for c, _ in CIRK]); R = np.array([r for _, r in CIRK])
    for k in range(int(t0s.min() // dt), int(t1s.max() // dt) + 1):
        t = k * dt
        idx = np.nonzero((t0s <= t) & (t <= t1s))[0]
        if len(idx) == 0:
            continue
        P = np.array([positie(vluchten[i], t) for i in idx])
        for n in np.nonzero((np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]) < R[None]).any(1))[0]:
            zone.add(idx[n])
        if len(idx) < 2:
            continue
        dh = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        dz = np.abs(P[:, None, 2] - P[None, :, 2])
        a, b = np.nonzero(np.triu((dh < H_SEP) & (dz < V_SEP), 1))
        paren.update((idx[i], idx[j]) for i, j in zip(a, b))
    return len(paren), len(zone)

wacht = np.array([f["t0"] - f["req"] for f in vluchten])
print(f"\nAanvragen: {len(aanvragen)}   gepland: {len(vluchten)}   afgewezen: {afgewezen}")
print(f"Gemiddelde wachttijd op de grond: {wacht.mean():.0f} s   (max {wacht.max():.0f} s)")
print(f"Gemiddelde vluchtduur: {np.mean(t1s - t0s) / 60:.0f} min")
print("Veiligheidscontrole bezig...")
nc, nz = controle()
print(f"Voertuigparen te dicht bij elkaar (<{H_SEP} m en <{V_SEP} m): {nc}   |   vluchten in een verboden zone: {nz}")

# ============================ 3D ANIMATIE ============================
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection="3d")
ax.computed_zorder = False        # kaart onderaan, dan lijnen, dan voertuigen
BLAUW, ORANJE, ROOD = "#1f5fa8", "#d2561b", "#b3261e"
km = lambda lonlat: [tuple(xy(la, lo) / 1000) for lo, la in lonlat]

def vlak(punten, kleur, z, rand=None, alpha=1.0):
    pts = [(x, y, z) for x, y in km(punten)]
    ax.add_collection3d(Poly3DCollection([pts], facecolors=kleur, edgecolors=rand or kleur, linewidths=.8, alpha=alpha, zorder=z / 10))

vlak(NL, "#e3e9d6", 0, "#7d8b6a")
for eiland_ in WADDEN:
    vlak(eiland_, "#e3e9d6", 0, "#7d8b6a")
vlak(IJSSELMEER, "#bcd6ee", 3, "#8fb3d6"); vlak(MARKERMEER, "#bcd6ee", 3, "#8fb3d6")

for (o, d), r in ROUTES.items():
    kleur = BLAUW if r["richting"] == 0 else ORANJE
    for laag in range(r["richting"], len(LAGEN_M), 2):
        ax.plot(r["lane"][:, 0] / 1000, r["lane"][:, 1] / 1000, np.full(len(r["lane"]), LAGEN_M[laag]), color=kleur, lw=.8, alpha=.6)
    ax.plot(r["lane"][:, 0] / 1000, r["lane"][:, 1] / 1000, np.zeros(len(r["lane"])), color=kleur, lw=.6, alpha=.25)
for naam, la, lo, straal in NOFLY:
    c = xy(la, lo) / 1000; th = np.linspace(0, 2 * np.pi, 60); rr = straal / 1000
    for z in (0, 1000):
        ax.plot(c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), np.full(60, z), color=ROOD, lw=1)
    for t_ in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([c[0] + rr * np.cos(t_)] * 2, [c[1] + rr * np.sin(t_)] * 2, [0, 1000], color=ROOD, lw=.4, alpha=.5)
    ax.text(c[0], c[1], 1100, naam, color=ROOD, ha="center", fontsize=6)
for naam, p in VP.items():
    ax.scatter(p[0] / 1000, p[1] / 1000, 0, marker="s", color="black", s=40, depthshade=False, zorder=15)
    ax.plot([p[0] / 1000] * 2, [p[1] / 1000] * 2, [0, 500], color="black", lw=1)
    ax.text(p[0] / 1000, p[1] / 1000, 600, naam, fontsize=10, fontweight="bold", ha="center", zorder=30)

allp = np.array([xy(la, lo) / 1000 for lo, la in NL])
ax.set_xlim(allp[:, 0].min() - 5, allp[:, 0].max() + 5); ax.set_ylim(allp[:, 1].min() - 5, allp[:, 1].max() + 5)
ax.set_zlim(0, 1200)
state = dict(t=START_UUR * 3600, speed=SPEED, play=True, exag=HOOGTE_UITVERGROTING)
def aspect():
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), 1.2 * state["exag"]), zoom=1.35)
aspect()
ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("hoogte [m]")
ax.view_init(elev=42, azim=-88)
sc = ax.scatter([], [], [], s=26, depthshade=False, zorder=20)
titel = ax.set_title("")

def update(_):
    if state["play"]:
        state["t"] = (state["t"] + state["speed"] * 0.04) % 86400
    X, Y, Z, R = posities(state["t"])
    sc._offsets3d = (X, Y, Z)
    sc.set_color([BLAUW if r == 0 else ORANJE for r in R] if len(R) else [BLAUW])
    h, m = int(state["t"] // 3600), int(state["t"] % 3600 // 60)
    titel.set_text(f"{h:02d}:{m:02d}   |   {len(X)} voertuigen in de lucht   |   {state['speed']}x   "
                   f"|   hoogte {state['exag']}x uitvergroot")
    return sc,

def toets(e):
    if e.key == " ":
        state["play"] = not state["play"]
    elif e.key == "up":
        state["speed"] = min(state["speed"] * 2, 1200)
    elif e.key == "down":
        state["speed"] = max(state["speed"] // 2, 5)
    elif e.key == "z":
        state["exag"] = min(state["exag"] + 10, 300); aspect()
    elif e.key == "x":
        state["exag"] = max(state["exag"] - 10, 1); aspect()
fig.canvas.mpl_connect("key_press_event", toets)

anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
plt.show()