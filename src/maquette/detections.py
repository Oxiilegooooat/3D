"""Objets mesurés par le LiDAR mais absents du cadastre : petites constructions (abris de
jardin, vérandas, carports) et véhicules présents lors du survol.

Un objet est un amas de sursol (MNS − MNT) non végétal (infrarouge), hors bâtiments connus ;
sa taille et sa forme décident de sa nature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy import ndimage
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.polygon import orient

from maquette.geoplateforme import Grille


@dataclass(frozen=True)
class Vehicule:
    x: float
    y: float
    angle: float
    longueur: float
    largeur: float
    hauteur: float


def _composantes(
    masque: NDArray[np.bool_], gx: NDArray[np.float64], gy: NDArray[np.float64], res: float
) -> list[tuple[Polygon, NDArray[np.bool_]]]:
    etiquettes, n = ndimage.label(masque)
    resultat = []
    for i, tranche in enumerate(ndimage.find_objects(etiquettes), start=1):
        if tranche is None:
            continue
        sous = etiquettes[tranche] == i
        pts = np.column_stack([gx[tranche][sous], gy[tranche][sous]])
        if len(pts) < 3:
            continue
        enveloppe = MultiPoint(pts).convex_hull.buffer(res / 2, join_style="mitre")
        complet = np.zeros_like(masque)
        complet[tranche] = sous
        resultat.append((enveloppe, complet))
    _ = n
    return resultat


def detecter(
    mnh: Grille,
    ndvi: NDArray[np.float64],
    zone: shapely.Geometry,
    interdit: shapely.Geometry,
    seuil_vegetal: float,
    cfg: dict[str, Any],
) -> tuple[list[Polygon], list[Vehicule]]:
    """`ndvi` est échantillonné sur la grille de `mnh` ; un objet est nettement minéral
    (indice sous `seuil_vegetal` moins `marge_objets`)."""
    cd = cfg["detections"]
    res = mnh.resolution
    h, w = mnh.valeurs.shape
    xmin, _, _, ymax = mnh.emprise
    gx, gy = np.meshgrid(xmin + (np.arange(w) + 0.5) * res, ymax - (np.arange(h) + 0.5) * res)
    shapely.prepare(zone)
    shapely.prepare(interdit)
    libre = shapely.contains_xy(zone, gx, gy) & ~shapely.contains_xy(interdit, gx, gy)
    hauteur = mnh.valeurs.astype(np.float64)
    mineral = ndvi < seuil_vegetal - cfg["sol"]["marge_objets"]

    cv = cd["vehicule"]
    vehicules = []
    occupe = np.zeros_like(libre)
    masque = libre & (hauteur >= cv["hauteur"][0]) & (hauteur <= cv["hauteur"][1]) & mineral
    for enveloppe, pixels in _composantes(masque, gx, gy, res):
        surface = pixels.sum() * res * res
        if not cv["surface"][0] <= surface <= cv["surface"][1]:
            continue
        rect = enveloppe.minimum_rotated_rectangle
        coins = np.asarray(rect.exterior.coords)[:4]
        cotes = np.diff(np.vstack([coins, coins[:1]]), axis=0)
        longueurs = np.hypot(*cotes.T)
        i = int(np.argmax(longueurs[:2]))
        longueur, largeur = float(longueurs[i]), float(longueurs[1 - i])
        if not (
            cv["longueur"][0] <= longueur <= cv["longueur"][1]
            and cv["largeur"][0] <= largeur <= cv["largeur"][1]
        ):
            continue
        c = rect.centroid
        vehicules.append(
            Vehicule(
                c.x,
                c.y,
                float(np.arctan2(cotes[i, 1], cotes[i, 0])),
                min(longueur, 4.8),
                min(largeur, 1.9),
                float(np.percentile(hauteur[pixels], 90)),
            )
        )
        occupe |= pixels

    ca = cd["annexe"]
    annexes = []
    masque = (
        libre & ~occupe & (hauteur >= ca["hauteur"][0]) & (hauteur <= ca["hauteur"][1]) & mineral
    )
    # Ouverture : les murs de clôture (minces) ne forment pas d'amas.
    masque = ndimage.binary_opening(masque, structure=np.ones((3, 3)))
    for enveloppe, pixels in _composantes(masque, gx, gy, res):
        surface = pixels.sum() * res * res
        if not ca["surface"][0] <= surface <= ca["surface"][1]:
            continue
        rect = enveloppe.minimum_rotated_rectangle
        if surface / max(rect.area, 1e-9) < ca["compacite"]:
            continue
        annexes.append(orient(rect, sign=1.0))
    return annexes, vehicules
