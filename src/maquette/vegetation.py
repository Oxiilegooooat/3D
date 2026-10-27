"""Arbres détectés dans la hauteur de canopée LiDAR HD (MNS − MNT) : position, hauteur et
rayon de houppier mesurés ; la forme du houppier est stylisée."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from scipy import ndimage

from maquette.geoplateforme import Grille
from maquette.maillage import Maillage, cylindre, icosphere, transformer
from maquette.terrain import Terrain


@dataclass(frozen=True)
class Arbre:
    x: float
    y: float
    z_sol: float
    hauteur: float
    rayon: float


def detecter(
    mns: Grille, mnt: Grille, exclusion: shapely.Geometry, terrain: Terrain, cfg: dict[str, Any]
) -> list[Arbre]:
    cv = cfg["vegetation"]
    res = mns.resolution
    mnh = ndimage.gaussian_filter((mns.valeurs - mnt.valeurs).astype(np.float64), 1.0)
    h, w = mnh.shape
    xmin, _, _, ymax = mns.emprise
    xs = xmin + (np.arange(w) + 0.5) * res
    ys = ymax - (np.arange(h) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    shapely.prepare(exclusion)
    exclu = shapely.contains_xy(exclusion, gx, gy)
    vegetation = (mnh > cv["hauteur_min"]) & ~exclu
    taille = int(cv["distance_min"] / res) * 2 + 1
    sommets = (mnh == ndimage.maximum_filter(mnh, size=taille)) & vegetation
    rs, cs = np.nonzero(sommets)
    # Sommets du plus haut au plus bas ; un plateau (pixels ex aequo) ne donne qu'un arbre.
    gardes: list[int] = []
    min_px = cv["distance_min"] / res
    for k in np.argsort(-mnh[rs, cs], kind="stable"):
        if len(gardes) >= cv["arbres_max"]:
            break
        if gardes and np.min(np.hypot(rs[gardes] - rs[k], cs[gardes] - cs[k])) < min_px:
            continue
        gardes.append(int(k))
    ordre = gardes
    arbres = []
    directions = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    for k in ordre:
        r, c = rs[k], cs[k]
        sommet = mnh[r, c]
        distances = []
        for dx, dy in directions:
            d = 0.0
            for pas in np.arange(1, 16):
                rr = round(r - dy * pas)
                cc = round(c + dx * pas)
                if (
                    not (0 <= rr < h and 0 <= cc < w)
                    or not vegetation[rr, cc]
                    or mnh[rr, cc] < 0.55 * sommet
                ):
                    break
                d = pas * res
            distances.append(d)
        rayon = float(np.clip(np.median(distances) + res, 1.2, 7.0))
        x, y = float(xs[c]), float(ys[r])
        z = float(terrain.hauteur(np.array([[x, y]]))[0])
        arbres.append(Arbre(x, y, z, float(sommet), rayon))
    return arbres


def maillages(
    arbres: list[Arbre], proche: tuple[float, float, float], graine: int = 7
) -> dict[str, list[Maillage]]:
    """Tronc et houppier (sphère bosselée, trois teintes) pour chaque arbre ; houppiers plus
    fins dans le rayon `proche` = (x, y, rayon) autour du site."""
    rng = np.random.default_rng(graine)
    modeles: dict[int, list[Maillage]] = {1: [], 2: []}
    for subdivisions in (1, 2):
        for _ in range(4):
            sphere = icosphere(subdivisions)
            bosses = 1 + 0.16 * rng.standard_normal(len(sphere.sommets))
            sphere.sommets = sphere.sommets * np.clip(bosses, 0.75, 1.3)[:, None]
            modeles[subdivisions].append(sphere)
    tronc = cylindre(6)
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    for a in arbres:
        h_houppier = float(np.clip(a.hauteur * 0.62, 1.5, a.hauteur - 0.8))
        rayon = min(a.rayon, max(h_houppier * 0.9, 1.0))
        centre = np.array([a.x, a.y, a.z_sol + a.hauteur - h_houppier / 2])
        fin = np.hypot(a.x - proche[0], a.y - proche[1]) < proche[2]
        liste = modeles[2 if fin else 1]
        modele = liste[int(rng.integers(len(liste)))]
        sortie[f"feuillage_{int(rng.integers(3))}"].append(
            transformer(
                modele,
                np.array([rayon, rayon, h_houppier / 2]),
                float(rng.uniform(0, 2 * np.pi)),
                centre,
            )
        )
        r_tronc = 0.1 + 0.018 * a.hauteur
        h_tronc = a.hauteur - h_houppier + 0.5
        sortie["tronc"].append(
            transformer(
                tronc,
                np.array([r_tronc, r_tronc, h_tronc + 0.3]),
                0.0,
                np.array([a.x, a.y, a.z_sol - 0.3]),
            )
        )
    return sortie
