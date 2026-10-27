"""Mobilier, marquages et véhicules.

- OpenStreetMap : passages piétons, stops, cédez-le-passage, arrêts de bus (abri si
  `shelter=yes`), feux, lampadaires, bancs, potelets, portails, piscines ;
- lampadaires **indicatifs** le long des voies quand OSM n'en recense aucun dans la zone ;
- véhicules : ceux mesurés par le LiDAR (`detections`), modèle stylisé.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
import shapely.ops
from numpy.typing import NDArray
from shapely.geometry import LineString, Point, Polygon

from maquette.detections import Vehicule
from maquette.maillage import (
    Maillage,
    boite,
    cylindre,
    fusionner,
    plaque,
    polygones,
    repere,
    surface,
)
from maquette.terrain import Terrain
from maquette.voirie import largeur

Tableau = NDArray[np.float64]
VERTICAL = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Element:
    nature: str
    x: float
    y: float
    tags: dict[str, str] = field(default_factory=dict)


def lire_osm(
    elements: list[dict[str, Any]],
    vers_local: Callable[[float, float], tuple[float, float]],
    zone: shapely.Geometry,
) -> tuple[list[Element], list[Polygon]]:
    """Éléments ponctuels utiles et piscines (chemins fermés `leisure=swimming_pool`)."""
    noeuds = {e["id"]: e for e in elements if e["type"] == "node"}
    dans = zone.buffer(2.0)
    resultat = []
    for e in noeuds.values():
        t = e.get("tags") or {}
        nature = None
        if (
            t.get("highway") == "crossing"
            and t.get("crossing") not in ("no", "unmarked")
            and (t.get("crossing:markings", "yes") != "no")
        ):
            nature = "passage_pieton"
        elif t.get("highway") in ("stop", "give_way", "traffic_signals", "street_lamp"):
            nature = {
                "stop": "stop",
                "give_way": "cedez",
                "traffic_signals": "feux",
                "street_lamp": "lampadaire",
            }[t["highway"]]
        elif t.get("highway") == "bus_stop" or (
            t.get("public_transport") == "platform" and t.get("bus") == "yes"
        ):
            nature = "bus"
        elif t.get("amenity") == "bench":
            nature = "banc"
        elif t.get("barrier") in ("bollard", "gate"):
            nature = {"bollard": "potelet", "gate": "portail"}[t["barrier"]]
        if nature is None:
            continue
        x, y = vers_local(e["lon"], e["lat"])
        if dans.contains(Point(x, y)):
            resultat.append(Element(nature, x, y, t))
    piscines = []
    for e in elements:
        t = e.get("tags") or {}
        if e["type"] != "way" or t.get("leisure") != "swimming_pool":
            continue
        ids = e.get("nodes", [])
        if len(ids) < 4 or ids[0] != ids[-1] or not all(i in noeuds for i in ids):
            continue
        poly = Polygon([vers_local(noeuds[i]["lon"], noeuds[i]["lat"]) for i in ids])
        if poly.is_valid and dans.contains(poly.representative_point()):
            piscines.append(poly)
    return resultat, piscines


def _route_proche(
    x: float,
    y: float,
    routes: list[tuple[LineString, dict[str, Any]]],
    cfg: dict[str, Any],
    portee: float = 10.0,
) -> tuple[Tableau, Tableau, float] | None:
    """Point le plus proche sur l'axe d'une voie, sa direction et la largeur de chaussée."""
    pt = Point(x, y)
    meilleure = None
    for ligne, p in routes:
        d = ligne.distance(pt)
        if d <= portee and (meilleure is None or d < meilleure[0]):
            meilleure = (d, ligne, p)
    if meilleure is None:
        return None
    _, ligne, p = meilleure
    s = ligne.project(pt)
    a = ligne.interpolate(max(s - 1.0, 0))
    b = ligne.interpolate(min(s + 1.0, ligne.length))
    t = np.array([b.x - a.x, b.y - a.y])
    t /= max(float(np.hypot(*t)), 1e-9)
    q = ligne.interpolate(s)
    return np.array([q.x, q.y]), t, largeur(p, cfg)


def _bord(x: float, y: float, chaussee: shapely.Geometry) -> tuple[Tableau, Tableau]:
    """Point du trottoir au bord de la chaussée, et direction vers l'extérieur de la voie."""
    p = Point(x, y)
    if chaussee.is_empty:
        return np.array([x, y]), np.array([1.0, 0.0])
    q = shapely.ops.nearest_points(chaussee.boundary, p)[0]
    dedans = chaussee.contains(p)
    ext = np.array([q.x - x, q.y - y]) if dedans else np.array([x - q.x, y - q.y])
    n = float(np.hypot(*ext))
    ext = ext / n if n > 1e-6 else np.array([1.0, 0.0])
    base = np.array([q.x, q.y]) + ext * 0.5 if dedans or n < 0.5 else np.array([x, y])
    return base, ext


def _poteau(base: Tableau, z: float, hauteur: float, rayon: float = 0.045) -> Maillage:
    m = cylindre(8)
    return repere(
        m,
        np.array([rayon, 0, 0]),
        np.array([0, rayon, 0]),
        VERTICAL * hauteur,
        np.array([base[0], base[1], z]),
    )


def _panneau(contour: Tableau, centre: Tableau, face: Tableau) -> tuple[Maillage, Maillage]:
    """Plaque verticale (avant texturé, dos gris) centrée sur `centre`, tournée vers `face`."""
    f = np.array([face[0], face[1], 0.0])
    f /= np.linalg.norm(f)
    cote = np.array([-f[1], f[0], 0.0])
    avant = repere(plaque(contour), cote, VERTICAL, f, centre + f * 0.06)
    dos = repere(plaque(contour), -cote, VERTICAL, -f, centre + f * 0.05)
    return avant, dos


def _octogone(rayon: float) -> Tableau:
    a = np.radians(22.5) + np.arange(8) * np.pi / 4
    return np.column_stack([np.cos(a), np.sin(a)]) * rayon


def construire(
    elements: list[Element],
    vehicules: list[Vehicule],
    routes: list[tuple[LineString, dict[str, Any]]],
    chaussee: shapely.Geometry,
    zone: shapely.Geometry,
    batiments: shapely.Geometry,
    terrain: Terrain,
    cfg: dict[str, Any],
) -> tuple[dict[str, list[Maillage]], dict[str, int]]:
    cm = cfg["mobilier"]
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    comptes: dict[str, int] = defaultdict(int)

    def z_sol(p: Tableau) -> float:
        return float(terrain.hauteur(np.asarray(p, dtype=np.float64)[None, :2])[0])

    zone_route = chaussee.buffer(-0.3)
    for e in elements:
        comptes[e.nature] += 1
        if e.nature == "passage_pieton":
            proche = _route_proche(e.x, e.y, routes, cfg)
            if proche is None:
                continue
            centre, t, larg = proche
            n = np.array([-t[1], t[0]])
            for k in range(max(int(larg // 1.0), 2)):
                c = centre + n * (-larg / 2 + 0.5 + k * 1.0)
                bande = Polygon(
                    [
                        c - t * 1.5 - n * 0.25,
                        c + t * 1.5 - n * 0.25,
                        c + t * 1.5 + n * 0.25,
                        c - t * 1.5 + n * 0.25,
                    ]
                )
                for p in polygones(bande.intersection(zone_route)):
                    sortie["marquage"].append(
                        surface(p, lambda xy: terrain.hauteur(xy) + 0.056, None, lambda xy: xy)
                    )
            continue
        base, ext = _bord(e.x, e.y, chaussee)
        z = z_sol(base)
        proche = _route_proche(e.x, e.y, routes, cfg)
        t = proche[1] if proche else np.array([-ext[1], ext[0]])
        if e.nature in ("stop", "cedez", "bus"):
            hauteur = 2.6 if e.nature == "bus" else 2.3
            sortie["metal_mobilier"].append(_poteau(base, z, hauteur))
            if e.nature == "stop":
                contour = _octogone(0.35)
            elif e.nature == "cedez":
                contour = np.array([[0.0, -0.35], [0.4, 0.35], [-0.4, 0.35]])
            else:
                contour = np.array([[-0.25, -0.35], [0.25, -0.35], [0.25, 0.35], [-0.25, 0.35]])
            avant, dos = _panneau(contour, np.array([base[0], base[1], z + hauteur - 0.4]), t)
            sortie[f"panneau_{e.nature}"].append(avant)
            sortie["panneau_dos"].append(dos)
            if e.nature == "bus" and e.tags.get("shelter") == "yes":
                _abribus(base + ext * 1.2, t, z, sortie)
        elif e.nature == "feux":
            sortie["metal_mobilier"].append(_poteau(base, z, 3.2, 0.06))
            centre = np.array([base[0], base[1], z + 2.3]) - np.array([*ext, 0]) * 0.0
            sortie["feu_boitier"].append(boite(centre, -ext, 0.32, 0.25, 0.95))
            f = np.array([t[0], t[1], 0.0])
            for i, couleur in enumerate(("feu_rouge", "feu_orange", "feu_vert")):
                c = centre + f * 0.13 + VERTICAL * (0.75 - i * 0.28)
                sortie[couleur].append(boite(c, np.array([-t[1], t[0]]), 0.16, 0.03, 0.16))
        elif e.nature == "lampadaire":
            _lampadaire(base, -ext, z, cm, sortie)
        elif e.nature == "banc":
            _banc(np.array([e.x, e.y]), t, z_sol(np.array([e.x, e.y])), sortie)
        elif e.nature == "potelet":
            sortie["metal_mobilier"].append(
                _poteau(np.array([e.x, e.y]), z_sol(np.array([e.x, e.y])), 0.9, 0.07)
            )

    if comptes["lampadaire"] == 0:
        comptes["lampadaire_indicatif"] = _lampadaires_indicatifs(
            routes, chaussee, zone, batiments, terrain, cfg, sortie
        )
    for v in vehicules:
        couleur = f"carrosserie_{int(abs(v.x * 7.3 + v.y * 3.1)) % 6}"
        for mat, m in _vehicule(v, z_sol(np.array([v.x, v.y])), couleur).items():
            sortie[mat].append(m)
    comptes["vehicule"] = len(vehicules)
    return sortie, dict(comptes)


def _lampadaire(
    base: Tableau,
    vers_route: Tableau,
    z: float,
    cm: dict[str, Any],
    sortie: dict[str, list[Maillage]],
) -> None:
    h = cm["lampadaire_hauteur"]
    sortie["metal_mobilier"].append(_poteau(base, z, h, 0.07))
    bras = np.array([*(base + vers_route * 0.6), z + h - 0.1])
    sortie["metal_mobilier"].append(boite(bras, vers_route, 1.2, 0.08, 0.08))
    tete = np.array([*(base + vers_route * 1.15), z + h - 0.22])
    sortie["lampe"].append(boite(tete, vers_route, 0.55, 0.25, 0.12))


def _lampadaires_indicatifs(
    routes: list[tuple[LineString, dict[str, Any]]],
    chaussee: shapely.Geometry,
    zone: shapely.Geometry,
    batiments: shapely.Geometry,
    terrain: Terrain,
    cfg: dict[str, Any],
    sortie: dict[str, list[Maillage]],
) -> int:
    cm = cfg["mobilier"]
    pas = cm["lampadaire_espacement"]
    interdit = shapely.union_all([chaussee, batiments.buffer(0.5)])
    poses: list[Tableau] = []
    for ligne, p in routes:
        if p.get("nature") in ("Sentier", "Chemin", "Escalier"):
            continue
        larg = largeur(p, cfg)
        s, cote = pas / 2, 1
        while s < ligne.length:
            a = ligne.interpolate(max(s - 0.5, 0))
            b = ligne.interpolate(min(s + 0.5, ligne.length))
            t = np.array([b.x - a.x, b.y - a.y])
            t /= max(float(np.hypot(*t)), 1e-9)
            n = np.array([-t[1], t[0]]) * cote
            q = ligne.interpolate(s)
            base = np.array([q.x, q.y]) + n * (larg / 2 + 0.6)
            pt = Point(*base)
            if (
                zone.buffer(-0.3).contains(pt)
                and not interdit.contains(pt)
                and all(np.hypot(*(base - autre)) > pas * 0.6 for autre in poses)
            ):
                z = float(terrain.hauteur(base[None, :])[0])
                _lampadaire(base, -n, z, cm, sortie)
                poses.append(base)
            s += pas
            cote = -cote
    return len(poses)


def _abribus(base: Tableau, t: Tableau, z: float, sortie: dict[str, list[Maillage]]) -> None:
    n = np.array([-t[1], t[0]])
    c = np.array([base[0], base[1], z])
    sortie["verre_garde_corps"].append(boite(c + np.array([*(n * 0.65), 0]), t, 3.2, 0.03, 2.3))
    for cote in (-1, 1):
        sortie["verre_garde_corps"].append(
            boite(c + np.array([*(t * 1.6 * cote), 0]), n, 1.3, 0.03, 2.3)
        )
    sortie["metal_mobilier"].append(boite(c + VERTICAL * 2.3, t, 3.4, 1.5, 0.08))
    sortie["metal_mobilier"].append(
        boite(c + np.array([*(n * 0.4), 0]) + VERTICAL * 0.45, t, 2.4, 0.4, 0.05)
    )


def _banc(p: Tableau, t: Tableau, z: float, sortie: dict[str, list[Maillage]]) -> None:
    c = np.array([p[0], p[1], z])
    n = np.array([-t[1], t[0]])
    sortie["bois_banc"].append(boite(c + VERTICAL * 0.42, t, 1.6, 0.45, 0.05))
    sortie["bois_banc"].append(
        boite(c + np.array([*(n * 0.2), 0]) + VERTICAL * 0.5, t, 1.6, 0.05, 0.4)
    )
    for cote in (-1, 1):
        sortie["metal_mobilier"].append(
            boite(c + np.array([*(t * 0.7 * cote), 0]), t, 0.06, 0.4, 0.42)
        )


def _vehicule(v: Vehicule, z: float, couleur: str) -> dict[str, Maillage]:
    """Voiture stylisée à l'échelle mesurée, orientée selon son grand axe."""
    lg, w = v.longueur, v.largeur
    h = float(np.clip(v.hauteur, 1.3, 2.1))
    axe = np.array([1.0, 0.0])
    caisse = boite(np.array([0.0, 0.0, 0.25]), axe, lg, w, 0.62)
    hab_h = max(h - 0.92, 0.35)
    habitacle = boite(np.array([-0.08 * lg, 0.0, 0.87]), axe, 0.52 * lg, w - 0.14, hab_h)
    toit = boite(np.array([-0.08 * lg, 0.0, 0.87 + hab_h]), axe, 0.5 * lg, w - 0.18, 0.04)
    roues = [
        boite(np.array([sx * 0.32 * lg, sy * (w / 2 - 0.12), 0.0]), axe, 0.64, 0.24, 0.64)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    c, s = np.cos(v.angle), np.sin(v.angle)
    ex, ey = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    o = np.array([v.x, v.y, z])
    return {
        couleur: repere(fusionner([caisse, toit]), ex, ey, VERTICAL, o),
        "vitre_auto": repere(habitacle, ex, ey, VERTICAL, o),
        "pneu": repere(fusionner(roues), ex, ey, VERTICAL, o),
    }
