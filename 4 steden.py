#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 TEAM ALBATROSS - TU DELFT - O&O
 Luchtruimindeling Rotterdam <-> Amsterdam voor 2-persoons personal air vehicles
===============================================================================

Wat doet dit script?
  1. Bouwt een luchtruim-netwerk: vertiports (parkeerplaatsen), "snelwegen in de
     lucht" (corridors), hoogtelagen per rijrichting en aan/afvliegroutes.
  2. Rekent uit hoeveel personen/voertuigen er op een tijdstip gebruik van
     kunnen maken: autoverkeer Rotterdam<->Amsterdam x adoptiepercentage.
  3. Plant elke vlucht met een 4D-slotsysteem (zoals ATFM / U-space):
     vertrektijd, corridor en hoogtelaag worden zo gekozen dat er altijd
     minimale scheiding is. Is het te druk, dan wacht het voertuig op de grond.
  4. Controleert achteraf ONAFHANKELIJK met een positiesimulatie (tijdstap dt)
     of er toch voertuigen te dicht bij elkaar of in een no-fly zone komen.
  5. Maakt grafieken, een CSV met alle vluchten en (optioneel) een animatie.

Gebruik (VS Code: gewoon op "Run Python File" drukken, of in de terminal):
    python albatross_luchtruim_sim.py                      # = live weergave (gewoon Run in VS Code)
    python albatross_luchtruim_sim.py --geen-venster       # alle grafieken als PNG's opslaan
    python albatross_luchtruim_sim.py --adoptie 0.05       # 5% van de auto's
    python albatross_luchtruim_sim.py --sweep              # ook adoptie-sweep 0.5%..15%
    python albatross_luchtruim_sim.py --live               # LIVE: zie ze vliegen (spatie = pauze)
    python albatross_luchtruim_sim.py --live --adoptie 0.05 --start 17:00 --snelheid 120
    python albatross_luchtruim_sim.py --animatie           # GIF van het piekuur
    python albatross_luchtruim_sim.py --wind 0.03          # snelheidsafwijking 3% (stresstest)
    python albatross_luchtruim_sim.py --fato 3             # 3 landingspads per vertiport
    python albatross_luchtruim_sim.py --verkeer-csv verkeer.csv   # echte NDW-data
    python albatross_luchtruim_sim.py --geen-venster       # alleen PNG's opslaan

Benodigd:  pip install numpy scipy matplotlib

LET OP - dit zijn AANNAMES die jullie moeten vervangen/valideren:
  * car_trips_per_day_per_direction : placeholder, gebruik NDW/KiM-data
    (of een CSV via --verkeer-csv).
  * Schiphol (6 km) en Rotterdam The Hague Airport (3 km) zijn vereenvoudigde
    cirkels, geen echte CTR/TMA-vlakken. Vervang door LVNL/AIP-data.
  * Coordinaten van vertiports en corridors zijn indicatief.
  * Vlucht = constante grondsnelheid, rechte stukken, geen wind (tenzij --wind),
    geen accelereren/hoveren; opstijgen/landen is vereenvoudigd.
  * Separatienormen (300 m / 50 m) zijn gekozen waarden, geen regelgeving.
"""
from __future__ import annotations

# --- Automatisch ontbrekende pakketten installeren (gewoon op Run drukken is genoeg) ---
import importlib.util as _ilu
import subprocess as _sp
import sys as _sys
for _mod, _pip in (("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib")):
    if _ilu.find_spec(_mod) is None:
        print(f"Pakket '{_pip}' ontbreekt, wordt nu geinstalleerd...")
        _sp.check_call([_sys.executable, "-m", "pip", "install", _pip])

import argparse
import bisect
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np

import matplotlib
if "--geen-venster" in sys.argv:          # geen pop-up vensters, alleen bestanden
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (nodig voor 3D-plot)
from matplotlib.animation import FuncAnimation as FuncAnimation_
from scipy.spatial import cKDTree


# =============================================================================
# 1. CONFIGURATIE  (pas hier aan!)
# =============================================================================
@dataclass
class Config:
    # --- Voertuig --------------------------------------------------------
    cruise_speed_kmh: float = 160.0      # kruissnelheid (grondsnelheid)
    range_km: float = 200.0              # actieradius volle batterij
    reserve_frac: float = 0.20           # veiligheidsreserve (van de range)
    seats: int = 2                       # personen per voertuig
    climb_rate_ms: float = 5.0           # stijgsnelheid [m/s]
    descent_rate_ms: float = 5.0         # daalsnelheid [m/s]
    turnaround_min: float = 20.0         # grondtijd/laden (alleen voor vlootschatting)

    # --- Luchtruim-ontwerp -------------------------------------------------
    # Hoogtelagen [m boven grond]. Heen (Rdam->Adam) = even indexen,
    # terug (Adam->Rdam) = oneven indexen  ->  richtingen zitten op andere hoogte.
    layer_alts_m: tuple = (150.0, 210.0, 270.0, 330.0, 390.0, 450.0)
    lane_offset_m: float = 400.0         # zijdelingse offset t.o.v. hartlijn ("rechts houden")
    h_sep_min_m: float = 300.0           # minimale horizontale scheiding
    v_sep_min_m: float = 50.0            # minimale verticale scheiding
    alert_factor: float = 2.0            # "bijna-conflict" = factor x de minima
    lane_headway_s: float = 60.0         # min. tijd tussen 2 voertuigen op dezelfde laag
    gate_sep_s: float = 15.0             # min. tijd tussen 2 voertuigen door dezelfde poort (alle lagen)
    plan_margin: float = 2.0             # planning houdt 2 x de scheidingsminima aan (buffer)
    vp_sep_s: float = 40.0               # min. tijd tussen 2 bewegingen (start/landing) per vertiport
    fato_service_s: float = 120.0        # bezettingstijd van een landingspad per beweging
    n_fato: int = 3                      # landingspads per vertiport
    ray_angle_deg: float = 15.0          # aan- en afvliegroute wijken +-15 graden af (V-vorm)
    ray_len_m: float = 1500.0            # lengte van dat eerste stukje
    max_ground_hold_s: float = 1200.0    # max wachttijd op de grond (daarna: vraag verloren)
    nofly_margin_m: float = 300.0        # extra afstand tot no-fly zone
    vp_clear_m: float = 600.0            # route mag niet binnen deze afstand langs een ANDER vertiport

    # --- Vraagmodel --------------------------------------------------------
    car_trips_per_day_per_direction: float = 20000.0   # PLACEHOLDER! (NDW/KiM)
    adoption: float = 0.02               # aandeel autoreizigers dat voor lucht kiest
    occupancy_dist: dict = field(default_factory=lambda: {1: 0.70, 2: 0.22, 3: 0.05, 4: 0.03})
    pooling: float = 0.0                 # aandeel alleenreizigers dat met een ander wil delen
    wind_sigma: float = 0.0              # std. afwijking snelheid (0.03 = 3%) -> stresstest
    traffic_csv: str = ""                # optioneel: hour,cars_rdam_ams,cars_ams_rdam

    # --- Simulatie -----------------------------------------------------------
    dt_s: float = 2.0                    # tijdstap positiesimulatie
    seed: int = 42
    out_dir: str = "uitvoer"

    @property
    def v(self) -> float:                # m/s
        return self.cruise_speed_kmh / 3.6


# --- Geografische gegevens (indicatief!) -------------------------------------
# key, naam, stad (R/A), lat, lon, gewicht binnen de stad
VERTIPORTS = [
    ("RC", "Rotterdam Centrum",     "R", 51.9225, 4.4792, 0.45),
    ("RA", "Rotterdam Alexander",   "R", 51.9526, 4.5532, 0.30),
    ("RZ", "Rotterdam Zuid",        "R", 51.8870, 4.4876, 0.25),
    ("AC", "Amsterdam Centrum",     "A", 52.3792, 4.9003, 0.35),
    ("AZ", "Amsterdam Zuid",        "A", 52.3389, 4.8730, 0.30),
    ("AS", "Amsterdam Sloterdijk",  "A", 52.3890, 4.8370, 0.15),
    ("AO", "Amsterdam Zuidoost",    "A", 52.3144, 4.9466, 0.20),
]

# Corridors ("snelwegen in de lucht"): hartlijn van Rotterdam- naar Amsterdam-kant.
# 'access' = welke vertiports op deze corridor mogen aansluiten. Sectorontwerp:
# West = noordwestelijke pads, Oost = zuidoostelijke pads; RA en AC zitten op beide.
# Zo lopen aan-/afvliegroutes niet frontaal langs elkaars vertiport.
CORRIDORS = {
    "West": dict(  # via Zoetermeer / Leiden / Haarlem, ten westen van Schiphol
        waypoints=[(51.985, 4.515), (52.07, 4.51), (52.20, 4.52), (52.33, 4.60), (52.44, 4.76)],
        access={"RC", "RA", "AS", "AC"}),
    "Oost": dict(  # via Gouda / Nieuwkoop, ten oosten/zuiden van Schiphol
        waypoints=[(51.985, 4.63), (52.015, 4.71), (52.16, 4.78), (52.235, 4.93)],
        access={"RA", "RZ", "AO", "AZ", "AC"}),
}

# No-fly cirkels: naam, lat, lon, straal [m], plafond [m]
NOFLY = [
    ("Schiphol",                       52.3105, 4.7683, 6000.0, 1000.0),
    ("Rotterdam The Hague Airport",    51.9569, 4.4372, 3000.0, 1000.0),
]

# Autoverkeer: relatief uurprofiel (werkdag, 00:00 .. 23:00), wordt genormaliseerd
BASE_PROFILE = np.array([0.4, 0.25, 0.2, 0.2, 0.4, 1.2, 3.5, 6.8, 7.2, 5.5, 4.6, 4.8,
                         5.2, 5.0, 5.3, 6.2, 7.6, 8.0, 6.4, 4.6, 3.2, 2.3, 1.5, 0.9])

REF_LAT, REF_LON = 52.10, 4.70


def ll_to_xy(lat: float, lon: float) -> np.ndarray:
    """lat/lon -> lokale x,y in meters (vlakke-aarde benadering, ruim voldoende hier)."""
    x = (lon - REF_LON) * math.cos(math.radians(REF_LAT)) * 111320.0
    y = (lat - REF_LAT) * 110574.0
    return np.array([x, y])


def hhmm(t: float) -> str:
    t = int(t)
    return f"{(t // 3600) % 24:02d}:{(t % 3600) // 60:02d}"


# =============================================================================
# 2. NETWERK: vertiports, corridors, routes
# =============================================================================
@dataclass
class Vertiport:
    key: str
    name: str
    city: str
    xy: np.ndarray
    weight: float


@dataclass
class Corridor:
    name: str
    center: np.ndarray          # N x 2, van Rotterdam- naar Amsterdam-kant
    access: set


@dataclass
class Route:
    key: tuple                  # (origin, dest, corridor)
    direction: str              # 'NB' (Rdam->Adam) of 'SB'
    pts: np.ndarray
    cum: np.ndarray
    L: float
    s_in: float                 # afstand tot corridor-ingang (poort)
    s_out: float                # afstand tot corridor-uitgang (poort)
    clearance_m: float          # kleinste afstand tot no-fly zone


def offset_polyline(pts: np.ndarray, d: float) -> np.ndarray:
    """Verschuif een polylijn d meter naar rechts t.o.v. de looprichting (met 'miter' bij knikken)."""
    seg = np.diff(pts, axis=0)
    t = seg / np.linalg.norm(seg, axis=1)[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]])            # rechter normaal
    out = np.zeros_like(pts)
    out[0] = pts[0] + d * n[0]
    out[-1] = pts[-1] + d * n[-1]
    for i in range(1, len(pts) - 1):
        m = n[i - 1] + n[i]
        m /= np.linalg.norm(m)
        scale = 1.0 / max(float(np.dot(m, n[i])), 0.5)
        out[i] = pts[i] + d * scale * m
    return out


def seg_point_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    tt = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + tt * ab)))


def seg_intersect(p1, p2, p3, p4):
    """Snijpunt van twee lijnstukken (zonder eindpunten) of None. Geeft (punt, t, u)."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / den
    if 1e-6 < t < 1 - 1e-6 and 1e-6 < u < 1 - 1e-6:
        return p1 + t * d1, t, u
    return None


def ray_point(p: np.ndarray, g: np.ndarray, ang_deg: float, length: float) -> np.ndarray:
    u = (g - p) / np.linalg.norm(g - p)
    a = math.radians(ang_deg)
    ur = np.array([u[0] * math.cos(a) - u[1] * math.sin(a), u[0] * math.sin(a) + u[1] * math.cos(a)])
    return p + length * ur


class Network:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vp = {k: Vertiport(k, n, c, ll_to_xy(la, lo), w) for k, n, c, la, lo, w in VERTIPORTS}
        self.corridors = {n: Corridor(n, np.array([ll_to_xy(*p) for p in d["waypoints"]]), d["access"])
                          for n, d in CORRIDORS.items()}
        self.nofly = [(n, ll_to_xy(la, lo), r, ce) for n, la, lo, r, ce in NOFLY]
        self.routes: dict = {}
        self.rejected: list = []
        self.warnings: list = []
        self.partners: dict = {}
        self._build_routes()
        self._build_conflict_table()

    # -- hulpfuncties -----------------------------------------------------
    def direction_of(self, origin: str) -> str:
        return "NB" if self.vp[origin].city == "R" else "SB"

    def layers_for(self, direction: str) -> list:
        n = len(self.cfg.layer_alts_m)
        par = 0 if direction == "NB" else 1
        return [i for i in range(n) if i % 2 == par]

    def lane_polyline(self, corr: str, direction: str) -> np.ndarray:
        c = self.corridors[corr].center
        base = c if direction == "NB" else c[::-1]
        return offset_polyline(base, self.cfg.lane_offset_m)

    def altitude(self, route: Route, layer_alt: float, s: np.ndarray) -> np.ndarray:
        """Hoogteprofiel langs de route: stijgen, kruisen op laagniveau, dalen."""
        v = self.cfg.v
        gc = self.cfg.climb_rate_ms / v
        gd = self.cfg.descent_rate_ms / v
        return np.maximum(np.minimum(np.minimum(layer_alt, gc * s), gd * (route.L - s)), 0.0)

    # -- routes bouwen -------------------------------------------------------
    def _build_routes(self):
        cfg = self.cfg
        max_alt = max(cfg.layer_alts_m)
        climb_dist = max_alt * cfg.v / cfg.climb_rate_ms
        desc_dist = max_alt * cfg.v / cfg.descent_rate_ms
        for cname, cor in self.corridors.items():
            for o, vo in self.vp.items():
                for d, vd in self.vp.items():
                    if vo.city == vd.city or o not in cor.access or d not in cor.access:
                        continue
                    direction = "NB" if vo.city == "R" else "SB"
                    poly = self.lane_polyline(cname, direction)
                    g_in, g_out = poly[0], poly[-1]
                    r_dep = min(cfg.ray_len_m, 0.4 * np.linalg.norm(g_in - vo.xy))
                    r_arr = min(cfg.ray_len_m, 0.4 * np.linalg.norm(g_out - vd.xy))
                    m_dep = ray_point(vo.xy, g_in, -cfg.ray_angle_deg, r_dep)
                    m_arr = ray_point(vd.xy, g_out, +cfg.ray_angle_deg, r_arr)
                    pts = np.vstack([vo.xy, m_dep, poly, m_arr, vd.xy])
                    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                    n = len(poly)
                    route = Route((o, d, cname), direction, pts, cum, float(cum[-1]),
                                  float(cum[2]), float(cum[2 + n - 1]), 0.0)
                    # no-fly check
                    clear = 1e12
                    for _, c, rad, _ceil in self.nofly:
                        for i in range(len(pts) - 1):
                            clear = min(clear, seg_point_dist(c, pts[i], pts[i + 1]) - rad)
                    route.clearance_m = clear
                    if clear < cfg.nofly_margin_m:
                        self.rejected.append((route.key, f"no-fly zone (marge {clear:.0f} m)"))
                        continue
                    near = None
                    for k2, v2 in self.vp.items():
                        if k2 in (o, d):
                            continue
                        dm = min(seg_point_dist(v2.xy, pts[i], pts[i + 1]) for i in range(len(pts) - 1))
                        if dm < cfg.vp_clear_m:
                            near = (k2, dm)
                            break
                    if near:
                        self.rejected.append((route.key, f"komt {near[1]:.0f} m langs vertiport {near[0]}"))
                        continue
                    if route.s_in < climb_dist or (route.L - route.s_out) < desc_dist:
                        self.rejected.append((route.key, "aanvliegroute te kort voor klim/daling naar hoogste laag"))
                        continue
                    self.routes[(o, d, cname)] = route
        if not self.routes:
            self.warnings.append("Geen enkele geldige route!")

    def _build_conflict_table(self):
        """Exact 4D-model: voor elk paar (route, hoogtelaag) bepaal welke verschillen in vertrektijd
        tot een (bijna-)conflict leiden. De planner vermijdt precies die tijdvensters.
        lag = vertrektijd(B) - vertrektijd(A); verboden als er een moment is met horizontaal
        < plan_margin*h_min EN verticaal < plan_margin*v_min."""
        cfg = self.cfg
        dt = 2.0
        h_plan = cfg.h_sep_min_m * cfg.plan_margin
        v_plan = cfg.v_sep_min_m * cfg.plan_margin
        keys = list(self.routes.keys())
        smp = {}
        for k in keys:
            r = self.routes[k]
            t = np.arange(0.0, r.L / cfg.v + dt, dt)
            sv = np.minimum(t * cfg.v, r.L)
            x = np.interp(sv, r.cum, r.pts[:, 0]); y = np.interp(sv, r.cum, r.pts[:, 1])
            z = {l: self.altitude(r, cfg.layer_alts_m[l], sv) for l in self.layers_for(r.direction)}
            smp[k] = (x, y, z)
        raw = {}
        self.hotspots = []
        for ia in range(len(keys)):
            xa, ya, za = smp[keys[ia]]
            for ib in range(ia, len(keys)):
                xb, yb, zb = smp[keys[ib]]
                dh = np.hypot(xa[:, None] - xb[None, :], ya[:, None] - yb[None, :])
                ii, jj = np.nonzero(dh < h_plan)
                if len(ii) == 0:
                    continue
                kk = ii - jj
                for la, zA in za.items():
                    for lb, zB in zb.items():
                        m = np.abs(zA[ii] - zB[jj]) < v_plan
                        if not m.any():
                            continue
                        lags = np.unique(kk[m]) * dt
                        ivs, lo, prev = [], lags[0], lags[0]
                        for lg in lags[1:]:
                            if lg - prev > 2 * dt:
                                ivs.append((lo - dt, prev + dt)); lo = lg
                            prev = lg
                        ivs.append((lo - dt, prev + dt))
                        raw.setdefault((keys[ia], la), []).append(((keys[ib], lb), ivs))
                        if ia != ib:
                            raw.setdefault((keys[ib], lb), []).append(((keys[ia], la), [(-h, -l) for l, h in ivs]))
                        width = sum(h - l for l, h in ivs)
                        self.hotspots.append((width, keys[ia], la, keys[ib], lb))
        for k, lst in raw.items():
            self.partners[k] = [(g, ivs, min(i[0] for i in ivs), max(i[1] for i in ivs)) for g, ivs in lst]
        self.hotspots.sort(key=lambda h: -h[0])

    # -- rapportage -----------------------------------------------------------
    def report(self):
        cfg = self.cfg
        print("\n=== NETWERKVALIDATIE ===")
        print(f"Geldige routes: {len(self.routes)}   afgekeurd: {len(self.rejected)}")
        for k, why in self.rejected:
            print(f"  afgekeurd {k}: {why}")
        npairs = sum(len(v) for v in self.partners.values()) // 2
        print(f"Route/laag-combinaties met mogelijk conflict (tijdvensters worden geblokkeerd): {npairs}")
        cross = [h for h in self.hotspots if (h[1], h[2]) != (h[3], h[4]) and h[1][:2] != h[3][:2]][:4]
        if cross:
            print("  Grootste conflictvensters tussen verschillende routes (plekken om het ontwerp te verbeteren):")
            for w, ka, la, kb, lb in cross:
                print(f"    {ka[0]}>{ka[1]}/{ka[2]}/L{la}  x  {kb[0]}>{kb[1]}/{kb[2]}/L{lb}: {w:.0f} s verboden vertrekverschil")
        mn = min(r.clearance_m for r in self.routes.values())
        print(f"Kleinste afstand van een route tot een no-fly zone: {mn / 1000:.1f} km")
        for w in self.warnings:
            print("WAARSCHUWING:", w)
        print("\n=== ENERGIE / RANGE (eenrichting) ===")
        usable = cfg.range_km * (1 - cfg.reserve_frac)
        print(f"Bruikbare range = {cfg.range_km:.0f} km x (1-{cfg.reserve_frac:.0%} reserve) = {usable:.0f} km")
        shown = [("RC", "AC", "West"), ("RC", "AS", "West"), ("RA", "AO", "Oost"), ("RZ", "AZ", "Oost")]
        for o, d, c in shown:
            r = self.routes.get((o, d, c))
            if r:
                km = r.L / 1000
                ok = "JA" if 2 * km <= usable else "NEE (laden op bestemming nodig)"
                print(f"  {o}->{d} via {c:<5}: {km:5.1f} km = {km / cfg.range_km:4.0%} accu | retour op 1 lading: {ok}")


# =============================================================================
# 3. SLOTSYSTEEM (agenda's per resource)
# =============================================================================
class Calendar:
    """Agenda voor een resource: elke boeking duurt `dur` seconden, gecentreerd rond t."""

    def __init__(self, dur: float):
        self.d = dur
        self.starts: list = []
        self.ends: list = []

    def earliest(self, t: float) -> float:
        h = self.d / 2.0
        lo = t - h
        i = bisect.bisect_right(self.starts, lo) - 1
        if i >= 0 and self.ends[i] > lo:
            lo = self.ends[i]
        i += 1
        n = len(self.starts)
        while i < n and self.starts[i] < lo + self.d:
            lo = max(lo, self.ends[i])
            i += 1
        return lo + h

    def reserve(self, t: float):
        lo = t - self.d / 2.0
        idx = bisect.bisect_left(self.starts, lo)
        self.starts.insert(idx, lo)
        self.ends.insert(idx, lo + self.d)


class MultiCalendar:
    """n identieke servers (bijv. landingspads van een vertiport)."""

    def __init__(self, n: int, dur: float):
        self.cals = [Calendar(dur) for _ in range(n)]

    def earliest(self, t: float) -> float:
        return min(c.earliest(t) for c in self.cals)

    def reserve(self, t: float):
        best = min(self.cals, key=lambda c: abs(c.earliest(t) - t))
        best.reserve(t)


@dataclass
class Flight:
    fid: int
    t_req: float
    origin: str
    dest: str
    direction: str
    corridor: str
    layer: int
    alt: float
    route: Route
    t0: float
    t_gin: float
    t_gout: float
    t_arr: float
    speed_factor: float = 1.0

    @property
    def delay(self) -> float:
        return self.t0 - self.t_req


class Scheduler:
    """Plant vluchten: kiest corridor + hoogtelaag + vertrektijd die conflictvrij is."""

    def __init__(self, cfg: Config, net: Network):
        self.cfg, self.net = cfg, net
        self.fato = {k: MultiCalendar(cfg.n_fato, cfg.fato_service_s) for k in net.vp}
        self.vpcal = {k: Calendar(cfg.vp_sep_s) for k in net.vp}
        self.lane = {}
        self.gate_in = {}
        self.gate_out = {}
        self.group_t0 = {}
        self.lane_use = {}
        for c in net.corridors:
            for dr in ("NB", "SB"):
                self.gate_in[(c, dr)] = Calendar(cfg.gate_sep_s)
                self.gate_out[(c, dr)] = Calendar(cfg.gate_sep_s)
                for l in net.layers_for(dr):
                    self.lane[(c, dr, l)] = Calendar(cfg.lane_headway_s)
                    self.lane_use[(c, dr, l)] = 0
        self.next_fid = 0

    def _pair_push(self, t: float, key: tuple) -> float:
        """Schuif vertrektijd t op tot hij in geen enkel verboden tijdvenster (conflict met al geplande vluchten) valt."""
        parts = self.net.partners.get(key, ())
        while True:
            moved = False
            for gkey, intervals, lo_all, hi_all in parts:
                lst = self.group_t0.get(gkey)
                if not lst:
                    continue
                a = bisect.bisect_left(lst, t + lo_all)
                b = bisect.bisect_right(lst, t + hi_all)
                for tg in lst[a:b]:
                    lag = tg - t
                    for lo, hi in intervals:
                        if lo <= lag <= hi:
                            t = tg - lo + 1e-3
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                return t

    def _evaluate(self, t_req, o, d, corr, direction, layer, route):
        cfg = self.cfg
        v = cfg.v
        a_in, a_out, a_end = route.s_in / v, route.s_out / v, route.L / v
        lane = self.lane[(corr, direction, layer)]
        gin = self.gate_in[(corr, direction)]
        gout = self.gate_out[(corr, direction)]
        t0 = t_req
        for _ in range(500):
            prev = t0
            t0 = max(t0, self.vpcal[o].earliest(t0))
            t0 = max(t0, self.fato[o].earliest(t0))
            t0 = max(t0, lane.earliest(t0 + a_in) - a_in)
            t0 = max(t0, gin.earliest(t0 + a_in) - a_in)
            t0 = max(t0, gout.earliest(t0 + a_out) - a_out)
            t0 = self._pair_push(t0, (route.key, layer))
            t0 = max(t0, self.vpcal[d].earliest(t0 + a_end) - a_end)
            t0 = max(t0, self.fato[d].earliest(t0 + a_end) - a_end)
            if t0 - t_req > cfg.max_ground_hold_s:
                return None
            if abs(t0 - prev) < 1e-7:
                break
        return dict(t0=t0, t_gin=t0 + a_in, t_gout=t0 + a_out, t_arr=t0 + a_end,
                    corr=corr, layer=layer, route=route, direction=direction)

    def plan(self, t_req: float, o: str, d: str):
        direction = self.net.direction_of(o)
        best = None
        for corr in self.net.corridors:
            route = self.net.routes.get((o, d, corr))
            if route is None:
                continue
            for layer in self.net.layers_for(direction):
                cand = self._evaluate(t_req, o, d, corr, direction, layer, route)
                if cand is None:
                    continue
                key = (round(cand["t_arr"] / 5.0), self.lane_use[(corr, direction, layer)], layer)
                if best is None or key < best[0]:
                    best = (key, cand)
        if best is None:
            return None
        c = best[1]
        v = self.cfg.v
        self.fato[o].reserve(c["t0"])
        self.vpcal[o].reserve(c["t0"])
        self.fato[d].reserve(c["t_arr"])
        self.vpcal[d].reserve(c["t_arr"])
        self.lane[(c["corr"], direction, c["layer"])].reserve(c["t_gin"])
        self.gate_in[(c["corr"], direction)].reserve(c["t_gin"])
        self.gate_out[(c["corr"], direction)].reserve(c["t_gout"])
        bisect.insort(self.group_t0.setdefault((c["route"].key, c["layer"]), []), c["t0"])
        self.lane_use[(c["corr"], direction, c["layer"])] += 1
        f = Flight(self.next_fid, t_req, o, d, direction, c["corr"], c["layer"],
                   self.cfg.layer_alts_m[c["layer"]], c["route"], c["t0"], c["t_gin"], c["t_gout"], c["t_arr"])
        self.next_fid += 1
        return f


# =============================================================================
# 4. VRAAGMODEL
# =============================================================================
def trips_per_car(cfg: Config) -> float:
    """Gemiddeld aantal vliegtuigritten dat 1 autorit oplevert (2 zitplaatsen)."""
    base = sum(p * math.ceil(n / cfg.seats) for n, p in cfg.occupancy_dist.items())
    solo = cfg.occupancy_dist.get(1, 0.0)
    return base - 0.5 * cfg.pooling * solo


def mean_occupancy(cfg: Config) -> float:
    return sum(n * p for n, p in cfg.occupancy_dist.items())


def hourly_profile(direction: str) -> np.ndarray:
    p = BASE_PROFILE.copy()
    if direction == "NB":
        p[6:10] *= 1.10
        p[16:20] *= 0.90
    else:
        p[6:10] *= 0.90
        p[16:20] *= 1.10
    return p / p.sum()


def car_traffic(cfg: Config) -> dict:
    """Auto's per uur per richting (24 waarden). Uit CSV of uit placeholder-model."""
    if cfg.traffic_csv and os.path.exists(cfg.traffic_csv):
        nb, sb = np.zeros(24), np.zeros(24)
        with open(cfg.traffic_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                h = int(row["hour"])
                nb[h] = float(row["cars_rdam_ams"])
                sb[h] = float(row["cars_ams_rdam"])
        return {"NB": nb, "SB": sb}
    n = cfg.car_trips_per_day_per_direction
    return {"NB": n * hourly_profile("NB"), "SB": n * hourly_profile("SB")}


def build_requests(cfg: Config, net: Network, rng: np.random.Generator):
    tpc = trips_per_car(cfg)
    traffic = car_traffic(cfg)
    reqs = []
    for direction in ("NB", "SB"):
        oc, dc = ("R", "A") if direction == "NB" else ("A", "R")
        ok = [k for k, v in net.vp.items() if v.city == oc]
        dk = [k for k, v in net.vp.items() if v.city == dc]
        # alleen verbindingen waarvoor minstens 1 geldige route bestaat; gewicht = product van pad-gewichten
        pairs = [(o, d) for o in ok for d in dk if any((o, d, c) in net.routes for c in net.corridors)]
        w = np.array([net.vp[o].weight * net.vp[d].weight for o, d in pairs])
        w /= w.sum()
        for h in range(24):
            lam = traffic[direction][h] * cfg.adoption * tpc
            n = rng.poisson(lam)
            for t in h * 3600 + rng.uniform(0, 3600, n):
                o, d = pairs[rng.choice(len(pairs), p=w)]
                reqs.append((float(t), o, d))
    reqs.sort()
    return reqs, traffic


# =============================================================================
# 5. POSITIESIMULATIE + CONFLICTDETECTIE (onafhankelijke controle)
# =============================================================================
def flight_position(net: Network, f: Flight, t: np.ndarray):
    """x, y, z van vlucht f op tijden t (array)."""
    v = net.cfg.v * f.speed_factor
    s = np.clip((t - f.t0) * v, 0.0, f.route.L)
    x = np.interp(s, f.route.cum, f.route.pts[:, 0])
    y = np.interp(s, f.route.cum, f.route.pts[:, 1])
    z = net.altitude(f.route, f.alt, s)
    return x, y, z, s


def build_trajectories(net: Network, flights: list, dt: float) -> dict:
    cols = {k: [] for k in ("step", "fid", "x", "y", "z", "phase")}
    v0 = net.cfg.v
    for f in flights:
        v = v0 * f.speed_factor
        t_end = f.t0 + f.route.L / v
        i0, i1 = math.ceil(f.t0 / dt), math.floor(t_end / dt)
        if i1 < i0:
            continue
        steps = np.arange(i0, i1 + 1)
        x, y, z, s = flight_position(net, f, steps * dt)
        cols["step"].append(steps.astype(np.int64))
        cols["fid"].append(np.full(len(steps), f.fid, dtype=np.int64))
        cols["x"].append(x); cols["y"].append(y); cols["z"].append(z)
        cols["phase"].append(((s > f.route.s_in).astype(np.int8) + (s > f.route.s_out).astype(np.int8)))
    if not cols["step"]:
        return {k: np.array([]) for k in cols}
    return {k: np.concatenate(v) for k, v in cols.items()}


def detect_conflicts(net: Network, rows: dict, flights_by_id: dict) -> dict:
    cfg = net.cfg
    out = dict(n_los=0, n_alert=0, events=[], min_h=None, nofly_rows=0, nofly_flights=0, by_phase={})
    if len(rows["step"]) == 0:
        return out
    x, y, z, step, fid, ph = (rows[k] for k in ("x", "y", "z", "step", "fid", "phase"))
    # no-fly zones
    bad = np.zeros(len(x), dtype=bool)
    for _n, c, rad, ceil in net.nofly:
        bad |= (np.hypot(x - c[0], y - c[1]) < rad) & (z < ceil)
    out["nofly_rows"] = int(bad.sum())
    out["nofly_flights"] = int(len(np.unique(fid[bad])))
    # paarsgewijs: KD-tree op (x, y, tijdstap*1e6) -> alleen buren in dezelfde tijdstap
    r_h = cfg.h_sep_min_m * cfg.alert_factor
    r_v = cfg.v_sep_min_m * cfg.alert_factor
    tree = cKDTree(np.column_stack([x, y, step.astype(np.float64) * 1e6]))
    pairs = tree.query_pairs(r_h, output_type="ndarray")
    if len(pairs) == 0:
        return out
    i, j = pairs[:, 0], pairs[:, 1]
    dh = np.hypot(x[i] - x[j], y[i] - y[j])
    dz = np.abs(z[i] - z[j])
    alert = dz < r_v
    los = alert & (dh < cfg.h_sep_min_m) & (dz < cfg.v_sep_min_m)

    def uniq(mask):
        if not mask.any():
            return np.zeros((0, 2), dtype=np.int64), np.array([], dtype=np.int64)
        a, b = fid[i[mask]], fid[j[mask]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        arr = np.column_stack([lo, hi])
        u = np.unique(arr, axis=0)
        return u, np.where(mask)[0]

    ua, _ = uniq(alert)
    ul, idx_los = uniq(los)
    out["n_alert"], out["n_los"] = int(len(ua)), int(len(ul))
    if len(idx_los):
        metric = np.maximum(dh[idx_los] / cfg.h_sep_min_m, dz[idx_los] / cfg.v_sep_min_m)
        out["min_h"] = float(dh[idx_los].min())
        seen = set()
        events = []
        for k in np.argsort(metric):
            ii, jj = i[idx_los[k]], j[idx_los[k]]
            pair = tuple(sorted((int(fid[ii]), int(fid[jj]))))
            if pair in seen:
                continue
            seen.add(pair)
            phs = tuple(sorted((int(ph[ii]), int(ph[jj]))))
            out["by_phase"][phs] = out["by_phase"].get(phs, 0) + 1
            if len(events) < 15:
                fa, fb = flights_by_id[pair[0]], flights_by_id[pair[1]]
                events.append(dict(t=hhmm(step[ii] * cfg.dt_s), a=f"{fa.origin}>{fa.dest}/{fa.corridor}/L{fa.layer}",
                                   b=f"{fb.origin}>{fb.dest}/{fb.corridor}/L{fb.layer}",
                                   dh=float(dh[idx_los[k]]), dz=float(dz[idx_los[k]]),
                                   phases=phs, xy=(float(x[ii]), float(y[ii]))))
        out["events"] = events
    return out


# =============================================================================
# 6. EEN SCENARIO DOORREKENEN
# =============================================================================
def run_scenario(cfg: Config, net: Network, verbose: bool = True) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    reqs, traffic = build_requests(cfg, net, rng)
    sched = Scheduler(cfg, net)
    flights, lost = [], []
    for t_req, o, d in reqs:
        f = sched.plan(t_req, o, d)
        if f is None:
            lost.append((t_req, o, d))
        else:
            if cfg.wind_sigma > 0:
                f.speed_factor = float(np.clip(rng.normal(1.0, cfg.wind_sigma), 0.7, 1.3))
            flights.append(f)
    by_id = {f.fid: f for f in flights}
    rows = build_trajectories(net, flights, cfg.dt_s)
    conf = detect_conflicts(net, rows, by_id)

    # statistieken
    delays = np.array([f.delay for f in flights]) if flights else np.array([0.0])
    n_steps = int(rows["step"].max()) + 1 if len(rows["step"]) else 1
    conc = np.bincount(rows["step"].astype(np.int64), minlength=n_steps) if len(rows["step"]) else np.zeros(1, dtype=int)
    peak_i = int(conc.argmax())
    n_h = 26
    req_h = np.zeros(n_h); dep_h = np.zeros(n_h)
    for t, _o, _d in reqs:
        req_h[min(int(t // 3600), n_h - 1)] += 1
    for f in flights:
        dep_h[min(int(f.t0 // 3600), n_h - 1)] += 1
    vp_keys = list(net.vp.keys())
    mov = np.zeros((len(vp_keys), n_h))
    for f in flights:
        mov[vp_keys.index(f.origin), min(int(f.t0 // 3600), n_h - 1)] += 1
        mov[vp_keys.index(f.dest), min(int(f.t_arr // 3600), n_h - 1)] += 1
    layer_use = {}
    for f in flights:
        layer_use[(f.corridor, f.direction, f.layer)] = layer_use.get((f.corridor, f.direction, f.layer), 0) + 1
    avg_dur = float(np.mean([f.route.L / cfg.v for f in flights])) if flights else 0.0
    peak_flights_h = float(dep_h.max())
    fleet = math.ceil(1.2 * peak_flights_h * (avg_dur + cfg.turnaround_min * 60) / 3600) if flights else 0
    persons = len(flights) * mean_occupancy(cfg) / trips_per_car(cfg)

    res = dict(cfg=cfg, reqs=reqs, flights=flights, lost=lost, rows=rows, conf=conf, traffic=traffic,
               delays=delays, conc=conc, peak_conc=int(conc.max()), peak_t=peak_i * cfg.dt_s,
               req_h=req_h, dep_h=dep_h, mov=mov, vp_keys=vp_keys, layer_use=layer_use,
               avg_dur_min=avg_dur / 60, peak_flights_h=peak_flights_h, fleet=fleet,
               persons=persons, runtime=time.time() - t_start)
    if verbose:
        print_summary(res)
    return res


def print_summary(res: dict):
    cfg: Config = res["cfg"]
    fl, lost, d, conf = res["flights"], res["lost"], res["delays"], res["conf"]
    n_req = len(res["reqs"])
    print("\n" + "=" * 70)
    print(f" RESULTATEN  |  adoptie {cfg.adoption:.1%}  |  {cfg.n_fato} pad(en) per vertiport  |  wind-sigma {cfg.wind_sigma:.0%}")
    print("=" * 70)
    print(f" Autoritten/dag beide richtingen (placeholder) : {2 * cfg.car_trips_per_day_per_direction:,.0f}")
    print(f" Vliegaanvragen/dag                            : {n_req}")
    print(f"   geplande vluchten                           : {len(fl)}   ({len(fl) / max(n_req, 1):.1%})")
    print(f"   verloren vraag (wachttijd > {cfg.max_ground_hold_s / 60:.0f} min)     : {len(lost)}   ({len(lost) / max(n_req, 1):.1%})")
    print(f" Passagiers/dag (geschat)                      : {res['persons']:.0f}")
    print(f" Piek vluchten per uur (vertrek)               : {res['peak_flights_h']:.0f}")
    print(f" Piek tegelijk in de lucht                     : {res['peak_conc']}  (om {hhmm(res['peak_t'])})")
    print(f" Gem. vliegtijd                                : {res['avg_dur_min']:.1f} min")
    print(f" Wachttijd op de grond  gemiddeld / p95 / max  : {d.mean():.0f} / {np.percentile(d, 95):.0f} / {d.max():.0f} s")
    print(f" Ruwe vlootschatting (taxi-model)              : {res['fleet']} voertuigen")
    print("-" * 70)
    print(" VEILIGHEIDSCONTROLE (onafhankelijk, dt = %.0f s)" % cfg.dt_s)
    print(f"   Loss of separation (<{cfg.h_sep_min_m:.0f} m EN <{cfg.v_sep_min_m:.0f} m)   : {conf['n_los']} vliegtuigparen")
    print(f"   Bijna-conflicten (<{cfg.h_sep_min_m * cfg.alert_factor:.0f} m EN <{cfg.v_sep_min_m * cfg.alert_factor:.0f} m)  : {conf['n_alert']} vliegtuigparen")
    print(f"   No-fly zone schendingen                     : {conf['nofly_flights']} vluchten")
    if conf["events"]:
        names = {0: "opstijgen", 1: "kruisen", 2: "dalen"}
        print("   Ergste conflicten (tijd, vlucht A, vlucht B, h, v, fasen):")
        for e in conf["events"][:8]:
            print(f"     {e['t']}  {e['a']:<22} {e['b']:<22} h={e['dh']:.0f} m v={e['dz']:.0f} m  "
                  f"{names[e['phases'][0]]}/{names[e['phases'][1]]}")
    print(f" (rekentijd {res['runtime']:.1f} s)")
    print("=" * 70)


def capacity_analysis(cfg: Config, net: Network):
    print("\n=== THEORETISCHE CAPACITEIT (ontwerp) ===")
    lane_cap = 3600.0 / cfg.lane_headway_s
    per_dir_layers = len(net.layers_for("NB"))
    funnel = 3600.0 / cfg.gate_sep_s
    corr_dir = min(per_dir_layers * lane_cap, funnel)
    n_corr = len(net.corridors)
    air_cap = n_corr * corr_dir
    vp_cap = min(cfg.n_fato * 3600.0 / cfg.fato_service_s, 3600.0 / cfg.vp_sep_s)
    r_cap = vp_cap * sum(1 for v in net.vp.values() if v.city == "R")
    a_cap = vp_cap * sum(1 for v in net.vp.values() if v.city == "A")
    print(f" Per laag              : {lane_cap:.0f} vluchten/uur   ({cfg.lane_headway_s:.0f} s tussen voertuigen)")
    print(f" Per corridor/richting : {corr_dir:.0f} vluchten/uur   ({per_dir_layers} lagen, poort-limiet {funnel:.0f}/uur)")
    print(f" Alle corridors/richting: {air_cap:.0f} vluchten/uur")
    print(f" Per vertiport          : {vp_cap:.0f} bewegingen/uur  ({cfg.n_fato} pads x {3600 / cfg.fato_service_s:.0f}/uur, slotlimiet {3600 / cfg.vp_sep_s:.0f}/uur)")
    print(f" Rotterdam totaal / Amsterdam totaal: {r_cap:.0f} / {a_cap:.0f} bewegingen/uur")
    per_dir = min(air_cap, r_cap / 2, a_cap / 2)
    bott = min([("luchtcorridors", air_cap), ("Rotterdamse vertiports", r_cap / 2), ("Amsterdamse vertiports", a_cap / 2)],
               key=lambda x: x[1])[0]
    print(f" => Max. gebalanceerde stroom: ~{per_dir:.0f} vluchten/uur per richting  (bottleneck: {bott})")
    traffic = car_traffic(cfg)
    peak_cars = max(traffic["NB"].max(), traffic["SB"].max())
    tpc = trips_per_car(cfg)
    print(f" => Dat komt overeen met een adoptie van ~{per_dir / (peak_cars * tpc):.1%} van het autoverkeer in het spitsuur "
          f"({peak_cars:.0f} auto's/uur/richting).")


# =============================================================================
# 7. GRAFIEKEN
# =============================================================================
C_NB = ["#1f5fa8", "#4c8fd6", "#8fbbea"]        # blauw  : Rotterdam -> Amsterdam
C_SB = ["#c4501b", "#e8824a", "#f4b894"]        # oranje : Amsterdam -> Rotterdam
ZONE = "#b3261e"


def lane_color(direction: str, layer: int, net: Network) -> str:
    layers = net.layers_for(direction)
    k = layers.index(layer) % 3
    return (C_NB if direction == "NB" else C_SB)[k]


def draw_map(ax, net: Network, flights_at=None, title="Luchtruimindeling (bovenaanzicht)"):
    cfg = net.cfg
    for n, c, rad, _ in net.nofly:
        ax.add_patch(Circle((c[0] / 1000, c[1] / 1000), rad / 1000, color=ZONE, alpha=0.15, lw=0))
        ax.add_patch(Circle((c[0] / 1000, c[1] / 1000), rad / 1000, fill=False, ec=ZONE, lw=1.2, ls="--"))
        ax.text(c[0] / 1000, c[1] / 1000, n.replace(" ", "\n", 1), ha="center", va="center", fontsize=7, color=ZONE)
    drawn = set()
    for r in net.routes.values():
        feeders = [r.pts[:3], r.pts[-3:]]
        for fd in feeders:
            key = tuple(np.round(fd.flatten(), -1))
            if key in drawn:
                continue
            drawn.add(key)
            ax.plot(fd[:, 0] / 1000, fd[:, 1] / 1000, color="#9aa0a6", lw=0.6, alpha=0.7, zorder=1)
    for cname in net.corridors:
        c = net.corridors[cname].center
        ax.plot(c[:, 0] / 1000, c[:, 1] / 1000, color="#cfd3d7", lw=7, alpha=0.6, zorder=0, solid_capstyle="round")
        for dr, col in (("NB", C_NB[0]), ("SB", C_SB[0])):
            p = net.lane_polyline(cname, dr)
            ax.plot(p[:, 0] / 1000, p[:, 1] / 1000, color=col, lw=1.6, zorder=2)
            ax.plot(p[[0, -1], 0] / 1000, p[[0, -1], 1] / 1000, "o", color=col, ms=4, zorder=3)
        mid = c[len(c) // 2]
        ax.text(mid[0] / 1000 + (1.2 if cname == "Oost" else -1.2), mid[1] / 1000, f"corridor {cname}",
                fontsize=8, ha="left" if cname == "Oost" else "right", color="#444", style="italic")
    for k, v in net.vp.items():
        ax.plot(v.xy[0] / 1000, v.xy[1] / 1000, "s", color="#111", ms=6, zorder=5)
        ax.text(v.xy[0] / 1000 + 0.8, v.xy[1] / 1000 - 0.3, v.name.split(" ", 1)[1], fontsize=7, zorder=6)
    ax.plot([], [], color=C_NB[0], label="Rotterdam -> Amsterdam (rijbaan rechts)")
    ax.plot([], [], color=C_SB[0], label="Amsterdam -> Rotterdam")
    ax.plot([], [], color="#9aa0a6", lw=0.8, label="aan-/afvliegroutes")
    ax.plot([], [], "s", color="#111", label="vertiport (parkeerplaats)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)


def fig_map(net: Network, path: str):
    fig, ax = plt.subplots(figsize=(8.5, 9))
    draw_map(ax, net)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def fig_layers(net: Network, path: str):
    cfg = net.cfg
    r_nb = net.routes.get(("RC", "AC", "West")) or next(r for r in net.routes.values() if r.direction == "NB")
    r_sb = net.routes.get(("AC", "RC", "West")) or next(r for r in net.routes.values() if r.direction == "SB")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for dr, r in (("NB", r_nb), ("SB", r_sb)):
        s = np.linspace(0, r.L, 600)
        for layer in net.layers_for(dr):
            z = net.altitude(r, cfg.layer_alts_m[layer], s)
            xs = s / 1000 if dr == "NB" else (r.L - s) / 1000
            ax.plot(xs, z, color=lane_color(dr, layer, net), lw=2,
                    label=f"{'Rdam->Adam' if dr == 'NB' else 'Adam->Rdam'} laag {layer} ({cfg.layer_alts_m[layer]:.0f} m)")
    ax.set_xlabel("afstand vanaf Rotterdam langs de route [km]")
    ax.set_ylabel("hoogte boven grond [m]")
    ax.set_title(f"Hoogtelagen: richtingen op andere hoogte (verticale scheiding {min(np.diff(cfg.layer_alts_m)):.0f} m, "
                 f"zijdelings {2 * cfg.lane_offset_m:.0f} m)", fontsize=10)
    ax.set_ylim(0, max(cfg.layer_alts_m) * 1.15)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="lower center")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def fig_demand(res: dict, path: str):
    cfg: Config = res["cfg"]
    hours = np.arange(24)
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    tr = res["traffic"]
    w = 0.4
    axs[0].bar(hours - w / 2, tr["NB"], w, color=C_NB[0], label="Rotterdam -> Amsterdam")
    axs[0].bar(hours + w / 2, tr["SB"], w, color=C_SB[0], label="Amsterdam -> Rotterdam")
    axs[0].set_ylabel("auto's per uur"); axs[0].set_title("Autoverkeer tussen Rotterdam en Amsterdam (placeholder-profiel)")
    axs[0].legend(fontsize=8); axs[0].set_xticks(hours); axs[0].grid(alpha=0.25, axis="y")
    axs[1].bar(hours, res["req_h"][:24], color="#c9ccd0", label="aanvragen")
    axs[1].bar(hours, res["dep_h"][:24], color="#2a7f62", width=0.55, label="vluchten uitgevoerd (vertrek)")
    axs[1].set_ylabel("vluchten per uur (beide richtingen)")
    axs[1].set_title(f"Vraag naar luchtvoertuigen bij {cfg.adoption:.1%} adoptie  ({cfg.seats} personen per voertuig)")
    axs[1].legend(fontsize=8); axs[1].set_xticks(hours); axs[1].set_xlabel("uur van de dag"); axs[1].grid(alpha=0.25, axis="y")
    conc = res["conc"]
    tt = np.arange(len(conc)) * cfg.dt_s / 3600
    axs[2].fill_between(tt, conc, color="#2a7f62", alpha=0.35)
    axs[2].plot(tt, conc, color="#2a7f62", lw=0.8)
    axs[2].set_xlim(0, 24); axs[2].set_xticks(range(0, 25, 2))
    axs[2].set_xlabel("tijd [uur]"); axs[2].set_ylabel("tegelijk in de lucht")
    axs[2].set_title(f"Aantal voertuigen in de lucht (piek {res['peak_conc']} om {hhmm(res['peak_t'])})")
    axs[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def snapshot_positions(net: Network, flights: list, t: float):
    pos = []
    for f in flights:
        v = net.cfg.v * f.speed_factor
        if f.t0 <= t <= f.t0 + f.route.L / v:
            x, y, z, _ = flight_position(net, f, np.array([t]))
            pos.append((x[0], y[0], z[0], f))
    return pos


def fig_snapshot(net: Network, res: dict, path: str, t: float | None = None):
    cfg = net.cfg
    t = res["peak_t"] if t is None else t
    pos = snapshot_positions(net, res["flights"], t)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 2, 1)
    draw_map(ax, net, title=f"Momentopname {hhmm(t)}: {len(pos)} voertuigen in de lucht")
    if pos:
        xs = [p[0] / 1000 for p in pos]; ys = [p[1] / 1000 for p in pos]; zs = [p[2] for p in pos]
        sc = ax.scatter(xs, ys, c=zs, cmap="viridis", vmin=0, vmax=max(cfg.layer_alts_m), s=34,
                        edgecolor="white", lw=0.6, zorder=8)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01); cb.set_label("hoogte [m]")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    a3 = fig.add_subplot(1, 2, 2, projection="3d")
    for cname in net.corridors:
        for dr in ("NB", "SB"):
            p = net.lane_polyline(cname, dr)
            for layer in net.layers_for(dr):
                a3.plot(p[:, 0] / 1000, p[:, 1] / 1000, np.full(len(p), cfg.layer_alts_m[layer]),
                        color=lane_color(dr, layer, net), lw=1.0, alpha=0.8)
    if pos:
        a3.scatter([p[0] / 1000 for p in pos], [p[1] / 1000 for p in pos], [p[2] for p in pos],
                   c=[p[2] for p in pos], cmap="viridis", vmin=0, vmax=max(cfg.layer_alts_m), s=22, depthshade=False)
    allx = np.concatenate([c.center[:, 0] for c in net.corridors.values()]) / 1000
    ally = np.concatenate([c.center[:, 1] for c in net.corridors.values()]) / 1000
    a3.set_box_aspect((np.ptp(allx), np.ptp(ally), 9))
    a3.set_zlim(0, max(cfg.layer_alts_m) * 1.05)
    a3.set_xlabel("x [km]"); a3.set_ylabel("y [km]"); a3.set_zlabel("hoogte [m]")
    a3.view_init(elev=24, azim=-62)
    a3.set_title("3D: hoogtelagen per corridor (blauw heen, oranje terug)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def fig_vertiports(net: Network, res: dict, path: str):
    cfg: Config = res["cfg"]
    mov = res["mov"][:, :24]
    names = [net.vp[k].name for k in res["vp_keys"]]
    cap = min(cfg.n_fato * 3600.0 / cfg.fato_service_s, 3600.0 / cfg.vp_sep_s)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    im = axs[0].imshow(mov, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(cap, mov.max()))
    axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names, fontsize=8)
    axs[0].set_xticks(range(0, 24, 2)); axs[0].set_xlabel("uur van de dag")
    axs[0].set_title(f"Bewegingen (start + landing) per vertiport per uur  (capaciteit ~{cap:.0f}/uur)", fontsize=10)
    for i in range(mov.shape[0]):
        for j in range(24):
            if mov[i, j] > 0:
                axs[0].text(j, i, f"{int(mov[i, j])}", ha="center", va="center", fontsize=6,
                            color="white" if mov[i, j] > 0.55 * cap else "#333")
    fig.colorbar(im, ax=axs[0], shrink=0.8)
    d = res["delays"]
    axs[1].hist(d / 60, bins=30, color="#2a7f62")
    axs[1].set_xlabel("wachttijd op de grond [min]"); axs[1].set_ylabel("aantal vluchten")
    axs[1].set_title(f"Wachttijd: gem. {d.mean() / 60:.1f} min, p95 {np.percentile(d, 95) / 60:.1f} min", fontsize=10)
    axs[1].set_yscale("log"); axs[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def fig_sweep(sweep: list, path: str):
    ad = [s["adoption"] * 100 for s in sweep]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(ad, [s["peak_conc"] for s in sweep], "o-", color="#2a7f62")
    axs[0, 0].set_title("Piek: voertuigen tegelijk in de lucht")
    axs[0, 1].plot(ad, [s["peak_flights_h"] for s in sweep], "o-", color="#1f5fa8")
    axs[0, 1].set_title("Piek: vluchten per uur (vertrek)")
    axs[1, 0].plot(ad, [s["lost_pct"] for s in sweep], "o-", color=ZONE, label="verloren vraag [%]")
    axs[1, 0].plot(ad, [s["delay_p95_min"] for s in sweep], "s--", color="#c4501b", label="wachttijd p95 [min]")
    axs[1, 0].set_title("Prijs van veiligheid: wachten of afgewezen worden"); axs[1, 0].legend(fontsize=8)
    axs[1, 1].plot(ad, [s["los"] for s in sweep], "o-", color=ZONE, label="loss of separation (paren)")
    axs[1, 1].plot(ad, [s["alerts"] for s in sweep], "s--", color="#e8824a", label="bijna-conflicten (paren)")
    axs[1, 1].set_title("Veiligheid (onafhankelijke controle)"); axs[1, 1].legend(fontsize=8)
    for ax in axs.flat:
        ax.set_xlabel("adoptie [% van de auto's]"); ax.grid(alpha=0.25)
        ax.set_xscale("log")
        ax.set_xticks(ad); ax.set_xticklabels([f"{a:g}" for a in ad])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def make_animation(net: Network, res: dict, path: str, minutes: float = 20.0, step_s: float = 8.0):
    from matplotlib.animation import FuncAnimation, PillowWriter
    t_mid = res["peak_t"]
    t0, t1 = max(0.0, t_mid - minutes * 30), t_mid + minutes * 30
    times = np.arange(t0, t1, step_s)
    fig, ax = plt.subplots(figsize=(7, 8))
    draw_map(ax, net, title="")
    ax.legend(loc="lower right", fontsize=7)
    sc = ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=max(net.cfg.layer_alts_m), s=36,
                    edgecolor="white", lw=0.6, zorder=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01); cb.set_label("hoogte [m]")
    ttl = ax.set_title("")

    def update(k):
        pos = snapshot_positions(net, res["flights"], times[k])
        if pos:
            sc.set_offsets(np.array([[p[0] / 1000, p[1] / 1000] for p in pos]))
            sc.set_array(np.array([p[2] for p in pos]))
        else:
            sc.set_offsets(np.empty((0, 2)))
        ttl.set_text(f"{hhmm(times[k])}  -  {len(pos)} voertuigen in de lucht")
        return sc, ttl

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)



def live_viewer(net: Network, res: dict, start_s: float, speed: float):
    """Interactieve live weergave: kaart + 3D, met afspelen/pauze, tijd- en snelheidsregelaar."""
    from matplotlib.widgets import Slider, Button
    cfg = net.cfg
    zmax = max(cfg.layer_alts_m)
    fig = plt.figure(figsize=(15, 8.5))
    fig.subplots_adjust(bottom=0.17, top=0.93, left=0.05, right=0.97, wspace=0.05)
    ax = fig.add_subplot(1, 2, 1)
    draw_map(ax, net, title="")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    sc = ax.scatter([], [], c=[], cmap="viridis", vmin=0, vmax=zmax, s=55, edgecolor="white", lw=0.8, zorder=9)
    a3 = fig.add_subplot(1, 2, 2, projection="3d")
    for cname in net.corridors:
        for dr in ("NB", "SB"):
            pl = net.lane_polyline(cname, dr)
            for layer in net.layers_for(dr):
                a3.plot(pl[:, 0] / 1000, pl[:, 1] / 1000, np.full(len(pl), cfg.layer_alts_m[layer]),
                        color=lane_color(dr, layer, net), lw=0.9, alpha=0.7)
    for n_, c, rad, _ in net.nofly:
        th = np.linspace(0, 2 * np.pi, 80)
        a3.plot((c[0] + rad * np.cos(th)) / 1000, (c[1] + rad * np.sin(th)) / 1000, np.zeros(80), color=ZONE, lw=1.2)
    allx = np.concatenate([c.center[:, 0] for c in net.corridors.values()]) / 1000
    ally = np.concatenate([c.center[:, 1] for c in net.corridors.values()]) / 1000
    a3.set_box_aspect((np.ptp(allx), np.ptp(ally), 9)); a3.set_zlim(0, zmax * 1.05)
    a3.set_xlabel("x [km]"); a3.set_ylabel("y [km]"); a3.set_zlabel("hoogte [m]")
    a3.view_init(elev=24, azim=-62)
    sc3 = a3.scatter([], [], [], c=[], cmap="viridis", vmin=0, vmax=zmax, s=30, depthshade=False)
    title = fig.suptitle("", fontsize=13)
    state = dict(t=float(start_s), playing=True, speed=float(speed))
    ax_t = fig.add_axes([0.12, 0.08, 0.60, 0.025])
    ax_s = fig.add_axes([0.12, 0.03, 0.60, 0.025])
    ax_b = fig.add_axes([0.80, 0.045, 0.10, 0.06])
    sl_t = Slider(ax_t, "tijd [uur]", 0, 24, valinit=state["t"] / 3600)
    sl_s = Slider(ax_s, "snelheid (x)", 1, 300, valinit=state["speed"])
    btn = Button(ax_b, "Pauze")

    def on_t(v): state["t"] = v * 3600
    def on_s(v): state["speed"] = v
    def toggle(_=None):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pauze" if state["playing"] else "Afspelen")
    sl_t.on_changed(on_t); sl_s.on_changed(on_s); btn.on_clicked(toggle)
    fig.canvas.mpl_connect("key_press_event", lambda e: toggle() if e.key == " " else None)
    interval_ms = 40

    def draw(t):
        pos = snapshot_positions(net, res["flights"], t)
        if pos:
            xs = np.array([p[0] / 1000 for p in pos]); ys = np.array([p[1] / 1000 for p in pos])
            zs = np.array([p[2] for p in pos])
            sc.set_offsets(np.column_stack([xs, ys])); sc.set_array(zs)
            sc3._offsets3d = (xs, ys, zs); sc3.set_array(zs)
        else:
            sc.set_offsets(np.empty((0, 2))); sc3._offsets3d = ([], [], [])
        title.set_text(f"{hhmm(t)}   -   {len(pos)} voertuigen in de lucht   (spatie = pauze)")

    def update(_frame):
        if state["playing"]:
            state["t"] = (state["t"] + state["speed"] * interval_ms / 1000.0) % 86400.0
            sl_t.eventson = False
            sl_t.set_val(state["t"] / 3600)
            sl_t.eventson = True
        draw(state["t"])
        return ()

    draw(state["t"])
    anim = FuncAnimation_(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    fig._anim = anim
    return fig, update, state


# =============================================================================
# 8. OPSLAAN
# =============================================================================
def save_outputs(res: dict, net: Network, out: str):
    cfg: Config = res["cfg"]
    with open(os.path.join(out, "vluchten.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["id", "aanvraag", "vertrek", "wachttijd_s", "van", "naar", "corridor", "laag", "hoogte_m",
                    "afstand_km", "duur_min", "aankomst"])
        for f in res["flights"]:
            w.writerow([f.fid, hhmm(f.t_req), hhmm(f.t0), round(f.delay), f.origin, f.dest, f.corridor, f.layer,
                        f.alt, round(f.route.L / 1000, 1), round(f.route.L / cfg.v / 60, 1), hhmm(f.t_arr)])
    c = res["conf"]
    summary = dict(adoptie=cfg.adoption, aanvragen=len(res["reqs"]), vluchten=len(res["flights"]),
                   verloren=len(res["lost"]), piek_tegelijk_in_lucht=res["peak_conc"], piek_vluchten_per_uur=res["peak_flights_h"],
                   wachttijd_gem_s=float(res["delays"].mean()), wachttijd_p95_s=float(np.percentile(res["delays"], 95)),
                   loss_of_separation_paren=c["n_los"], bijna_conflicten_paren=c["n_alert"],
                   nofly_schendingen_vluchten=c["nofly_flights"], vloot_schatting=res["fleet"],
                   passagiers_per_dag=round(res["persons"]))
    with open(os.path.join(out, "samenvatting.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


# =============================================================================
# 9. MAIN
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Team Albatross - luchtruimsimulatie Rotterdam <-> Amsterdam")
    p.add_argument("--adoptie", type=float, help="aandeel autoreizigers dat vliegt, bijv. 0.02 = 2%%")
    p.add_argument("--auto-per-dag", type=float, help="autoritten per dag per richting (placeholder 20000)")
    p.add_argument("--fato", type=int, help="landingspads per vertiport")
    p.add_argument("--wind", type=float, help="std. snelheidsafwijking, bijv. 0.03 (stresstest)")
    p.add_argument("--pooling", type=float, help="aandeel alleenreizigers dat deelt (0..1)")
    p.add_argument("--verkeer-csv", type=str, help="CSV met kolommen hour,cars_rdam_ams,cars_ams_rdam")
    p.add_argument("--dt", type=float, help="tijdstap positiesimulatie [s]")
    p.add_argument("--seed", type=int)
    p.add_argument("--uitvoer", type=str, help="uitvoermap")
    p.add_argument("--sweep", action="store_true", help="draai ook de adoptie-sweep")
    p.add_argument("--animatie", action="store_true", help="maak GIF van het piekmoment")
    p.add_argument("--live", action="store_true", help="LIVE weergave: zie de voertuigen door de lucht vliegen")
    p.add_argument("--start", type=str, default="08:30", help="starttijd live weergave, bijv. 08:30")
    p.add_argument("--snelheid", type=float, default=60.0, help="afspeelsnelheid live (60 = 1 min per seconde)")
    p.add_argument("--geen-venster", action="store_true", help="geen vensters tonen, alleen opslaan")
    return p.parse_args()


def main():
    a = parse_args()
    if len(sys.argv) == 1:          # gewoon op "Run" gedrukt in VS Code -> direct de live weergave
        a.live = True
    cfg = Config()
    if a.adoptie is not None: cfg.adoption = a.adoptie
    if a.auto_per_dag is not None: cfg.car_trips_per_day_per_direction = a.auto_per_dag
    if a.fato is not None: cfg.n_fato = a.fato
    if a.wind is not None: cfg.wind_sigma = a.wind
    if a.pooling is not None: cfg.pooling = a.pooling
    if a.verkeer_csv: cfg.traffic_csv = a.verkeer_csv
    if a.dt is not None: cfg.dt_s = a.dt
    if a.seed is not None: cfg.seed = a.seed
    if a.uitvoer: cfg.out_dir = a.uitvoer
    os.makedirs(cfg.out_dir, exist_ok=True)

    net = Network(cfg)
    if a.live:                                   # snel naar de live weergave, zonder alle grafieken
        res = run_scenario(cfg, net, verbose=False)
        hh, mm = a.start.split(":")
        print(f"Live weergave: {len(res['flights'])} vluchten gepland. Sluit het venster om te stoppen.")
        fig, _u, _st = live_viewer(net, res, int(hh) * 3600 + int(mm) * 60, a.snelheid)
        plt.show()
        return
    net.report()
    capacity_analysis(cfg, net)

    res = run_scenario(cfg, net)
    save_outputs(res, net, cfg.out_dir)

    figs = [fig_map(net, os.path.join(cfg.out_dir, "01_luchtruimkaart.png")),
            fig_layers(net, os.path.join(cfg.out_dir, "02_hoogtelagen.png")),
            fig_demand(res, os.path.join(cfg.out_dir, "03_vraag_en_verkeer.png")),
            fig_snapshot(net, res, os.path.join(cfg.out_dir, "04_momentopname_piek.png")),
            fig_vertiports(net, res, os.path.join(cfg.out_dir, "05_vertiports_en_wachttijd.png"))]

    if a.sweep:
        print("\n=== ADOPTIE-SWEEP ===")
        sweep = []
        for ad in (0.005, 0.01, 0.02, 0.05, 0.10, 0.15):
            r = run_scenario(replace(cfg, adoption=ad), net, verbose=False)
            n = max(len(r["reqs"]), 1)
            s = dict(adoption=ad, requests=len(r["reqs"]), flights=len(r["flights"]), lost_pct=100 * len(r["lost"]) / n,
                     peak_conc=r["peak_conc"], peak_flights_h=r["peak_flights_h"],
                     delay_mean_min=float(r["delays"].mean() / 60), delay_p95_min=float(np.percentile(r["delays"], 95) / 60),
                     los=r["conf"]["n_los"], alerts=r["conf"]["n_alert"], fleet=r["fleet"])
            sweep.append(s)
            print(f" adoptie {ad:5.1%}: {s['flights']:5d} vluchten/dag | piek {s['peak_flights_h']:4.0f}/uur | "
                  f"{s['peak_conc']:3d} tegelijk in lucht | verloren {s['lost_pct']:5.1f}% | wachttijd p95 {s['delay_p95_min']:5.1f} min | "
                  f"LoS {s['los']} | bijna {s['alerts']} | vloot ~{s['fleet']}")
        with open(os.path.join(cfg.out_dir, "sweep.json"), "w", encoding="utf-8") as fh:
            json.dump(sweep, fh, indent=2)
        figs.append(fig_sweep(sweep, os.path.join(cfg.out_dir, "06_adoptie_sweep.png")))

    if a.animatie:
        print("\nAnimatie maken (even geduld)...")
        make_animation(net, res, os.path.join(cfg.out_dir, "animatie_piekuur.gif"))

    print(f"\nKlaar. Bestanden staan in: {os.path.abspath(cfg.out_dir)}")
    if not a.geen_venster:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


if __name__ == "__main__":
    main()