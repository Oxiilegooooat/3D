"""Murs de clôture et haies le long des limites de parcelles.

Détection mesurée : sur chaque limite, la hauteur du sursol LiDAR (MNS − MNT) entre 0,7 et
3,2 m sur au moins 2,5 m signale un mur ou une haie ; la couleur de l'orthophoto tranche
(vert : haie). Hors bâtiments et chaussées.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import LineString

from maquette.geoplateforme import Grille
from maquette.maillage import Maillage, anneaux, polygones, quad_mur, surface
from maquette.terrain import Terrain

H_MIN, H_MAX, LONGUEUR_MIN = 0.7, 3.2, 2.5


@dataclass(frozen=True)
class Cloture:
    ligne: LineString
    hauteur: float
    nature: str  # mur, haie


def detecter(
    limites: shapely.Geometry,
    mns: Grille,
    mnt: Grille,
    ortho: Image.Image,
    terrain: Terrain,
    interdit: shapely.Geometry,
) -> list[Cloture]:
    lignes = shapely.line_merge(shapely.union_all(limites)) if not limites.is_empty else limites
    tableau = np.asarray(ortho)
    hauteur_px, largeur_px = tableau.shape[:2]
    shapely.prepare(interdit)
    resultat = []
    for ligne in getattr(lignes, "geoms", [lignes]):
        if not isinstance(ligne, LineString) or ligne.length < LONGUEUR_MIN:
            continue
        pts = np.asarray(shapely.segmentize(ligne, 0.5).coords)[:, :2]
        if len(pts) < 3:
            continue
        d = np.gradient(pts, axis=0)
        n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
        h = np.max(
            [
                mns.echantillonner(*(pts + n * o).T) - mnt.echantillonner(*(pts + n * o).T)
                for o in (-0.3, 0.0, 0.3)
            ],
            axis=0,
        )
        ok = (h >= H_MIN) & (h <= H_MAX) & ~shapely.contains_xy(interdit, pts[:, 0], pts[:, 1])
        debut = None
        for i, v in enumerate([*ok, False]):
            if v and debut is None:
                debut = i
            elif not v and debut is not None:
                morceau = pts[debut:i]
                if len(morceau) >= 2 and LineString(morceau).length >= LONGUEUR_MIN:
                    uv = terrain.uv_ortho(morceau)
                    px = np.clip((uv[:, 0] * largeur_px).astype(int), 0, largeur_px - 1)
                    py = np.clip(((1 - uv[:, 1]) * hauteur_px).astype(int), 0, hauteur_px - 1)
                    r, g, b = np.median(tableau[py, px].astype(float), axis=0)
                    nature = "haie" if g > 1.06 * max(r, b) else "mur"
                    resultat.append(
                        Cloture(
                            LineString(morceau).simplify(0.15), float(np.median(h[debut:i])), nature
                        )
                    )
                debut = None
    return resultat


def maillages(clotures: list[Cloture], terrain: Terrain) -> dict[str, list[Maillage]]:
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    for c in clotures:
        epaisseur = 0.22 if c.nature == "mur" else 0.8
        materiau = "mur_cloture" if c.nature == "mur" else "feuillage_1"
        for poly in polygones(c.ligne.buffer(epaisseur / 2, cap_style="flat")):

            def dessus(xy: np.ndarray, h: float = c.hauteur) -> np.ndarray:
                return terrain.hauteur(xy) + h

            sortie[materiau].append(surface(poly, dessus, 2.0, lambda xy: xy / 3.0))
            for ring in anneaux(poly, 2.0):
                ring = ring if shapely.LinearRing(ring).is_ccw else ring[::-1]
                z = terrain.hauteur(ring)
                for i in range(len(ring)):
                    j = (i + 1) % len(ring)
                    seg = float(np.hypot(*(ring[j] - ring[i])))
                    if seg < 1e-3:
                        continue
                    quad = [
                        (0.0, z[i] - 0.2),
                        (seg, z[j] - 0.2),
                        (seg, z[j] + c.hauteur),
                        (0.0, z[i] + c.hauteur),
                    ]
                    sortie[materiau].append(
                        quad_mur(ring[i], ring[j], quad, lambda s, zz: (s / 3.0, zz / 3.0))
                    )
    return sortie


def infos(clotures: list[Cloture]) -> dict[str, Any]:
    return {
        n: round(sum(c.ligne.length for c in clotures if c.nature == n)) for n in ("mur", "haie")
    }
