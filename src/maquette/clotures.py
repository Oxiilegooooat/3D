"""Clôtures le long des limites de parcelles.

Détection mesurée : sur chaque limite, la hauteur du sursol LiDAR (MNS − MNT) entre 0,7 et
3,2 m sur au moins 2,5 m signale une clôture ; l'orthophoto tranche (vert : haie). Puis :

- **sur rue** : mur plein (≥ 1,3 m) avec chaperon, muret surmonté d'une grille (plus bas),
  haie ; une ouverture de largeur de portail entre deux clôtures reçoit un portail ;
- **entre parcelles** : mur ou haie détectés, sinon grillage **indicatif** (le LiDAR ne voit
  pas un grillage).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import shapely
import shapely.ops
from numpy.typing import NDArray
from PIL import Image
from shapely.geometry import LineString, Point

from maquette.geoplateforme import Grille
from maquette.maillage import Maillage, anneaux, boite, polygones, quad_mur, surface
from maquette.terrain import Terrain

H_MIN, H_MAX, LONGUEUR_MIN = 0.7, 3.2, 2.5


@dataclass(frozen=True)
class Cloture:
    ligne: LineString
    hauteur: float
    nature: str  # mur, muret_grille, haie, portail, grillage
    rue: bool = False


def _lignes(geometrie: shapely.Geometry) -> list[LineString]:
    if geometrie.is_empty:
        return []
    fusion = shapely.line_merge(shapely.union_all(geometrie))
    return [g for g in getattr(fusion, "geoms", [fusion]) if isinstance(g, LineString)]


def detecter(
    limites_rue: shapely.Geometry,
    limites_internes: shapely.Geometry,
    mns: Grille,
    mnt: Grille,
    ortho: Image.Image,
    emprise_ortho: tuple[float, float, float, float],
    interdit: shapely.Geometry,
    portails_osm: list[tuple[float, float]],
    cfg: dict[str, Any],
) -> list[Cloture]:
    cc = cfg["clotures"]
    tableau = np.asarray(ortho)
    hp, wp = tableau.shape[:2]
    xmin, ymin, xmax, ymax = emprise_ortho
    shapely.prepare(interdit)
    resultat: list[Cloture] = []
    for rue, geometrie in ((True, limites_rue), (False, limites_internes)):
        for ligne in _lignes(geometrie):
            if ligne.length < 1.0:
                continue
            pts = np.asarray(shapely.segmentize(ligne, 0.5).coords)[:, :2]
            if len(pts) < 2:
                continue
            abscisses = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
            d = np.gradient(pts, axis=0)
            n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
            h = np.max(
                [
                    mns.echantillonner(*(pts + n * o).T) - mnt.echantillonner(*(pts + n * o).T)
                    for o in (-0.3, 0.0, 0.3)
                ],
                axis=0,
            )
            libre = ~shapely.contains_xy(interdit, pts[:, 0], pts[:, 1])
            ok = (h >= H_MIN) & (h <= H_MAX) & libre
            pistes = _pistes(ok, abscisses)
            detectees = []
            for a, b in pistes:
                if abscisses[b - 1] - abscisses[a] < LONGUEUR_MIN:
                    continue
                morceau = pts[a:b]
                px = np.clip(((morceau[:, 0] - xmin) / (xmax - xmin) * wp).astype(int), 0, wp - 1)
                py = np.clip(((ymax - morceau[:, 1]) / (ymax - ymin) * hp).astype(int), 0, hp - 1)
                r, g, bl = np.median(tableau[py, px].astype(float), axis=0)
                hauteur = float(np.median(h[a:b]))
                if g > 1.06 * max(r, bl):
                    nature = "haie"
                elif rue and hauteur < cc["mur_plein_min"]:
                    nature = "muret_grille"
                else:
                    nature = "mur"
                detectees.append((abscisses[a], abscisses[b - 1]))
                resultat.append(Cloture(LineString(morceau).simplify(0.15), hauteur, nature, rue))
            # Intervalles sans clôture détectée.
            bornes = [(0.0, 0.0), *detectees, (ligne.length, ligne.length)]
            for (_, fin), (debut, _) in pairwise(bornes):
                longueur = debut - fin
                if longueur < 1.0:
                    continue
                trou = shapely.ops.substring(ligne, fin, debut)
                if (
                    not isinstance(trou, LineString)
                    or not libre[
                        np.searchsorted(abscisses, (fin + debut) / 2).clip(0, len(libre) - 1)
                    ]
                ):
                    continue
                entre_deux = fin > 0 and debut < ligne.length
                if rue and entre_deux and cc["portail"][0] <= longueur <= cc["portail"][1]:
                    resultat.append(Cloture(trou, cc["portail_hauteur"], "portail", True))
                elif not rue:
                    resultat.append(Cloture(trou, cc["grillage_hauteur"], "grillage", False))
    for x, y in portails_osm:
        lignes_rue = _lignes(limites_rue)
        if not lignes_rue:
            break
        ligne = min(lignes_rue, key=lambda li: li.distance(Point(x, y)))
        if ligne.distance(Point(x, y)) > 3.0:
            continue
        s = ligne.project(Point(x, y))
        trou = shapely.ops.substring(ligne, max(s - 1.75, 0), min(s + 1.75, ligne.length))
        if isinstance(trou, LineString) and not any(
            c.ligne.distance(Point(x, y)) < 1.0 for c in resultat if c.nature != "grillage"
        ):
            resultat.append(Cloture(trou, cc["portail_hauteur"], "portail", True))
    return resultat


def _pistes(ok: NDArray[np.bool_], abscisses: NDArray[np.float64]) -> list[tuple[int, int]]:
    pistes = []
    debut = None
    for i, v in enumerate([*ok, False]):
        if v and debut is None:
            debut = i
        elif not v and debut is not None:
            if i - debut >= 2:
                pistes.append((debut, i))
            debut = None
    _ = abscisses
    return pistes


def _prisme(
    poly: shapely.Geometry,
    terrain: Terrain,
    bas: float,
    haut: float,
    materiau: str,
    sortie: dict[str, list[Maillage]],
    echelle: float = 3.0,
) -> None:
    """Volume vertical posé sur le terrain, de `bas` à `haut` au-dessus du sol."""
    for p in polygones(poly):

        def dessus(xy: NDArray[np.float64], h: float = haut) -> NDArray[np.float64]:
            return terrain.hauteur(xy) + h

        sortie[materiau].append(surface(p, dessus, 2.0, lambda xy: xy / echelle))
        for i, ring in enumerate(anneaux(p, 2.0)):
            sens = shapely.LinearRing(ring).is_ccw
            ring = ring if sens == (i == 0) else ring[::-1]
            z = terrain.hauteur(ring)
            for a in range(len(ring)):
                b = (a + 1) % len(ring)
                seg = float(np.hypot(*(ring[b] - ring[a])))
                if seg < 1e-3:
                    continue
                quad = [
                    (0.0, z[a] + bas),
                    (seg, z[b] + bas),
                    (seg, z[b] + haut),
                    (0.0, z[a] + haut),
                ]
                sortie[materiau].append(
                    quad_mur(ring[a], ring[b], quad, lambda s, zz: (s / echelle, zz / echelle))
                )


def _barreaux(
    ligne: LineString,
    terrain: Terrain,
    bas: float,
    haut: float,
    pas: float,
    sortie: dict[str, list[Maillage]],
) -> None:
    """Grille à barreaux verticaux et deux lisses, de `bas` à `haut` au-dessus du sol."""
    s = pas / 2
    while s < ligne.length:
        p = ligne.interpolate(s)
        z = float(terrain.hauteur(np.array([[p.x, p.y]]))[0])
        a = ligne.interpolate(max(s - 0.1, 0))
        b = ligne.interpolate(min(s + 0.1, ligne.length))
        t = np.array([b.x - a.x, b.y - a.y])
        sortie["grille"].append(boite(np.array([p.x, p.y, z + bas]), t, 0.025, 0.025, haut - bas))
        s += pas
    for niveau in (bas + 0.08, haut - 0.08):
        for i in range(len(ligne.coords) - 1):
            p0, p1 = np.asarray(ligne.coords[i][:2]), np.asarray(ligne.coords[i + 1][:2])
            m = (p0 + p1) / 2
            z = float(terrain.hauteur(m[None, :])[0])
            sortie["grille"].append(
                boite(
                    np.array([m[0], m[1], z + niveau]),
                    p1 - p0,
                    float(np.hypot(*(p1 - p0))),
                    0.04,
                    0.04,
                )
            )


def maillages(
    clotures: list[Cloture], terrain: Terrain, cfg: dict[str, Any]
) -> dict[str, list[Maillage]]:
    cc = cfg["clotures"]
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    for c in clotures:
        if c.nature == "haie":
            _prisme(
                c.ligne.buffer(0.4, cap_style="flat"),
                terrain,
                -0.2,
                c.hauteur,
                "massif",
                sortie,
                2.0,
            )
        elif c.nature == "mur":
            _prisme(
                c.ligne.buffer(0.11, cap_style="flat"),
                terrain,
                -0.2,
                c.hauteur - 0.06,
                "mur_cloture",
                sortie,
            )
            _prisme(
                c.ligne.buffer(0.16, cap_style="flat"),
                terrain,
                c.hauteur - 0.06,
                c.hauteur,
                "chaperon",
                sortie,
            )
        elif c.nature == "muret_grille":
            muret = cc["muret_hauteur"]
            _prisme(
                c.ligne.buffer(0.11, cap_style="flat"),
                terrain,
                -0.2,
                muret - 0.05,
                "mur_cloture",
                sortie,
            )
            _prisme(
                c.ligne.buffer(0.15, cap_style="flat"),
                terrain,
                muret - 0.05,
                muret,
                "chaperon",
                sortie,
            )
            _barreaux(c.ligne, terrain, muret, max(c.hauteur, muret + 0.9), 0.12, sortie)
        elif c.nature == "portail":
            for extremite in (c.ligne.coords[0], c.ligne.coords[-1]):
                pilier = Point(extremite[:2]).buffer(0.22, cap_style="square")
                _prisme(pilier, terrain, -0.2, c.hauteur + 0.25, "mur_cloture", sortie)
                _prisme(
                    Point(extremite[:2]).buffer(0.27, cap_style="square"),
                    terrain,
                    c.hauteur + 0.25,
                    c.hauteur + 0.32,
                    "chaperon",
                    sortie,
                )
            interieur = shapely.ops.substring(c.ligne, 0.25, max(c.ligne.length - 0.25, 0.3))
            if isinstance(interieur, LineString):
                _barreaux(interieur, terrain, 0.08, c.hauteur, 0.11, sortie)
        elif c.nature == "grillage":
            _grillage(c.ligne, terrain, c.hauteur, sortie)
    return sortie


def _grillage(
    ligne: LineString, terrain: Terrain, hauteur: float, sortie: dict[str, list[Maillage]]
) -> None:
    pts = np.asarray(shapely.segmentize(ligne, 2.0).coords)[:, :2]
    z = terrain.hauteur(pts)
    s = 0.0
    for i in range(len(pts) - 1):
        seg = float(np.hypot(*(pts[i + 1] - pts[i])))
        if seg < 1e-3:
            continue
        quad = [(0.0, z[i]), (seg, z[i + 1]), (seg, z[i + 1] + hauteur), (0.0, z[i] + hauteur)]

        def uv(ss: float, zz: float, s: float = s, zi: float = float(z[i])) -> tuple[float, float]:
            return (s + ss, zz - zi)

        sortie["grillage"].append(quad_mur(pts[i], pts[i + 1], quad, uv))
        s += seg
    s = 0.0
    while s <= ligne.length:
        p = ligne.interpolate(s)
        zp = float(terrain.hauteur(np.array([[p.x, p.y]]))[0])
        sortie["poteau_grillage"].append(
            boite(np.array([p.x, p.y, zp - 0.1]), np.array([1.0, 0.0]), 0.05, 0.05, hauteur + 0.12)
        )
        s += 2.5


def infos(clotures: list[Cloture]) -> dict[str, Any]:
    natures = ("mur", "muret_grille", "haie", "portail", "grillage")
    return {n: round(sum(c.ligne.length for c in clotures if c.nature == n)) for n in natures}
