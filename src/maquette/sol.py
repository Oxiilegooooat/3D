"""Sol des parcelles : classé d'après la photo aérienne, l'infrarouge et le LiDAR, puis rendu en
matériaux nets (l'orthophoto drapée est floue de près).

Classes : pelouse, massif (végétation basse, extrudée à sa hauteur mesurée), gravier, terre,
enrobé, dalles, eau (piscines et bassins, creusés avec margelle). Sous les houppiers, dans les
ombres et sous les objets, le sol est inconnu : il prend la classe la plus proche.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure

from maquette.geoplateforme import Emprise, Grille
from maquette.maillage import Maillage, anneaux, fusionner, polygones, quad_mur, surface
from maquette.terrain import Terrain

CLASSES = ("pelouse", "massif", "gravier", "terre", "enrobe", "dalles", "eau")
# Surélévation de chaque classe au-dessus du terrain (évite le scintillement aux raccords).
DECALAGE = {"pelouse": 0.02, "gravier": 0.024, "terre": 0.018, "enrobe": 0.028, "dalles": 0.032}
ECHELLE_UV = {"pelouse": 1.5, "gravier": 2.0, "terre": 3.0, "enrobe": 4.0, "dalles": 2.0}


def echantillonner_image(
    image: Image.Image, emprise: Emprise, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Valeurs RVB (0-1) d'une image géoréférencée aux points (x, y), au plus proche pixel."""
    tableau = np.asarray(image, dtype=np.float64) / 255
    h, w = tableau.shape[:2]
    xmin, ymin, xmax, ymax = emprise
    c = np.clip(((x - xmin) / (xmax - xmin) * w).astype(int), 0, w - 1)
    r = np.clip(((ymax - y) / (ymax - ymin) * h).astype(int), 0, h - 1)
    return np.asarray(tableau[r, c])


def ndvi(irc: NDArray[np.float64]) -> NDArray[np.float64]:
    """Indice de végétation d'une image infrarouge couleur (canaux PIR, rouge, vert)."""
    pir, rouge = irc[..., 0], irc[..., 1]
    return np.asarray((pir - rouge) / (pir + rouge + 1e-6))


def seuil_vegetal(indices: NDArray[np.float64], cfg: dict[str, Any]) -> float:
    """Seuil végétal / minéral : fixé dans la configuration, sinon seuil d'Otsu borné."""
    cs = cfg["sol"]
    if cs["ndvi_vegetal"] is not None:
        return float(cs["ndvi_vegetal"])
    from skimage.filters import threshold_otsu

    valeurs = indices[np.isfinite(indices)]
    if len(valeurs) < 100:
        return 0.0
    return float(np.clip(threshold_otsu(valeurs), *cs["ndvi_bornes"]))  # type: ignore[no-untyped-call]


def classer(
    region: shapely.Geometry,
    ortho: Image.Image,
    irc: Image.Image,
    emprise: Emprise,
    mnh: Grille,
    cfg: dict[str, Any],
) -> dict[str, shapely.Geometry]:
    """Polygones de chaque classe de sol, partition (approchée) de `region`."""
    cs = cfg["sol"]
    res = cs["resolution"]
    x0, y0, x1, y1 = region.bounds
    xs = np.arange(x0, x1, res) + res / 2
    ys = np.arange(y1, y0, -res) - res / 2
    gx, gy = np.meshgrid(xs, ys)
    rgb = echantillonner_image(ortho, emprise, gx, gy)
    indice = ndvi(echantillonner_image(irc, emprise, gx, gy))
    hauteur = mnh.echantillonner(gx, gy)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + 1e-6)

    k = {nom: i for i, nom in enumerate(CLASSES)}
    etiquettes = np.full(gx.shape, -1, dtype=int)
    seuil = seuil_vegetal(indice[shapely.contains_xy(region, gx, gy)], cfg)
    vegetal = indice > seuil
    h_massif = cs["hauteur_massif"]
    etiquettes[vegetal & (hauteur < h_massif[0])] = k["pelouse"]
    etiquettes[vegetal & (hauteur >= h_massif[0]) & (hauteur <= h_massif[1])] = k["massif"]
    # Eau claire (piscines) : bleu-vert lumineux ; les ombres, bleutées, sont sombres.
    eau = (b > r + 0.12) & (g > r + 0.06) & (luminance > 0.38) & (saturation > 0.25) & ~vegetal
    mineral = ~vegetal & ~eau & (hauteur < 0.8) & (luminance >= cs["ombre_luminance"])
    clair = (luminance >= 0.55) & (saturation < 0.22)
    etiquettes[mineral & (luminance < 0.3)] = k["enrobe"]
    etiquettes[mineral & clair] = k["dalles"]
    moyen = mineral & (luminance >= 0.3) & ~clair
    brun = (saturation >= 0.22) & (r >= g)
    etiquettes[moyen & brun] = k["terre"]
    etiquettes[moyen & ~brun] = k["gravier"]
    etiquettes[eau] = k["eau"]

    inconnu = etiquettes < 0
    if inconnu.all():
        etiquettes[:] = k["gravier"]
    elif inconnu.any():
        _, (ri, ci) = ndimage.distance_transform_edt(inconnu, return_indices=True)
        etiquettes = etiquettes[ri, ci]
    sigma = cs["lissage"] / res
    probabilites = np.stack(
        [
            ndimage.gaussian_filter((etiquettes == i).astype(np.float32), sigma)
            for i in range(len(CLASSES))
        ]
    )
    etiquettes = probabilites.argmax(axis=0)

    resultat: dict[str, shapely.Geometry] = {}
    for nom, i in k.items():
        masque = etiquettes == i
        if masque.sum() * res * res < cs["surface_min"]:
            continue
        geometrie = _polygoniser(masque, x0, y1, res)
        geometrie = geometrie.intersection(region)
        parts = [p for p in polygones(geometrie) if p.area >= cs["surface_min"]]
        if parts:
            resultat[nom] = shapely.union_all(parts).simplify(0.08)
    return resultat


def _polygoniser(masque: NDArray[np.bool_], x0: float, y1: float, res: float) -> shapely.Geometry:
    """Contours d'un masque binaire (carrés marchants), remplissage pair-impair."""
    cadre = np.pad(masque.astype(np.float32), 1)
    anneaux_xy = []
    for contour in measure.find_contours(cadre, 0.5):  # type: ignore[no-untyped-call]
        if len(contour) < 4:
            continue
        x = x0 + (contour[:, 1] - 1 + 0.5) * res
        y = y1 - (contour[:, 0] - 1 + 0.5) * res
        poly = shapely.make_valid(Polygon(np.column_stack([x, y])))
        if not poly.is_empty and poly.area > 0:
            anneaux_xy.append(poly)
    if not anneaux_xy:
        return Polygon()
    # Remplissage pair-impair : profondeur d'imbrication de chaque contour.
    points = [a.representative_point() for a in anneaux_xy]
    pairs: list[Polygon] = []
    impairs: list[Polygon] = []
    for i, a in enumerate(anneaux_xy):
        profondeur = sum(1 for j, b in enumerate(anneaux_xy) if j != i and b.contains(points[i]))
        (pairs if profondeur % 2 == 0 else impairs).append(a)
    return (
        shapely.union_all(pairs).difference(shapely.union_all(impairs))
        if impairs
        else shapely.union_all(pairs)
    )


def piscine(poly: Polygon) -> Polygon:
    """Bassin rectangulaire quand la forme détectée s'en approche (cas courant)."""
    rect = poly.minimum_rotated_rectangle
    return rect if poly.area / max(rect.area, 1e-9) > 0.7 else poly


def maillages(
    classes: dict[str, shapely.Geometry], terrain: Terrain, mnh: Grille, cfg: dict[str, Any]
) -> tuple[dict[str, list[Maillage]], shapely.Geometry]:
    """Surfaces drapées par classe, massifs extrudés, piscines creusées.

    Renvoie aussi l'emprise des bassins (à retirer du sol de base).
    """
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    bassins = []
    for nom, geometrie in classes.items():
        for poly in polygones(geometrie):
            if nom == "eau":
                bassin = piscine(poly)
                bassins.append(bassin.buffer(0.3, join_style="mitre"))
                for mat, m in _bassin(bassin, terrain).items():
                    sortie[mat].append(m)
            elif nom == "massif":
                # Hauteur mesurée au cœur du massif ; un massif bas reste un couvre-sol.
                pts = shapely.get_coordinates(shapely.segmentize(poly.buffer(-0.3), 1.0))
                pts = pts if len(pts) else np.asarray(poly.exterior.coords)[:, :2]
                h = float(np.median(mnh.echantillonner(pts[:, 0], pts[:, 1])))
                if h >= cfg["sol"]["massif_extrude"]:
                    sortie["massif"].extend(_extruder(poly, terrain, float(np.clip(h, 0.8, 2.5))))
                else:
                    sortie["massif"].append(
                        surface(
                            poly, lambda xy: terrain.hauteur(xy) + 0.03, 1.5, lambda xy: xy / 2.0
                        )
                    )
            else:
                decalage = DECALAGE[nom]
                echelle = ECHELLE_UV[nom]

                def hauteur(xy: NDArray[np.float64], d: float = decalage) -> NDArray[np.float64]:
                    return terrain.hauteur(xy) + d

                def uv(xy: NDArray[np.float64], e: float = echelle) -> NDArray[np.float64]:
                    return xy / e

                sortie[f"sol_{nom}"].append(surface(poly, hauteur, 1.5, uv))
    return sortie, shapely.union_all(bassins) if bassins else Polygon()


def _extruder(poly: Polygon, terrain: Terrain, hauteur: float) -> list[Maillage]:
    """Volume à dessus plat posé sur le terrain (massifs, haies)."""
    poly = poly.buffer(0)
    if poly.is_empty or not isinstance(poly, Polygon):
        return []

    def dessus(xy: NDArray[np.float64]) -> NDArray[np.float64]:
        return terrain.hauteur(xy) + hauteur

    morceaux = [surface(poly, dessus, 1.5, lambda xy: xy / 2.0)]
    for i, ring in enumerate(anneaux(poly, 1.5)):
        sens = shapely.LinearRing(ring).is_ccw
        ring = ring if sens == (i == 0) else ring[::-1]
        z = terrain.hauteur(ring)
        for a in range(len(ring)):
            b = (a + 1) % len(ring)
            seg = float(np.hypot(*(ring[b] - ring[a])))
            if seg < 1e-3:
                continue
            quad = [
                (0.0, z[a] - 0.1),
                (seg, z[b] - 0.1),
                (seg, z[b] + hauteur),
                (0.0, z[a] + hauteur),
            ]
            morceaux.append(quad_mur(ring[a], ring[b], quad, lambda s, zz: (s / 2.0, zz / 2.0)))
    return morceaux


def _bassin(bassin: Polygon, terrain: Terrain) -> dict[str, Maillage]:
    contour = np.asarray(bassin.exterior.coords)[:, :2]
    z = float(terrain.hauteur(contour).min())
    margelle = bassin.buffer(0.3, join_style="mitre").difference(bassin)
    resultat = {
        "margelle": fusionner(
            [
                surface(p, lambda xy: np.full(len(xy), z + 0.05), None, lambda xy: xy / 0.5)
                for p in polygones(margelle)
            ]
        ),
        "eau_piscine": surface(
            bassin, lambda xy: np.full(len(xy), z - 0.12), None, lambda xy: xy / 2.0
        ),
    }
    parois = []
    ring = np.asarray(bassin.exterior.coords)[:-1, :2]
    ring = ring if shapely.LinearRing(ring).is_ccw else ring[::-1]
    for a in range(len(ring)):
        b = (a + 1) % len(ring)
        seg = float(np.hypot(*(ring[b] - ring[a])))
        # Parois vues de l'intérieur du bassin : parcours inverse.
        quad = [(0.0, z - 0.4), (seg, z - 0.4), (seg, z + 0.05), (0.0, z + 0.05)]
        parois.append(quad_mur(ring[b], ring[a], quad, lambda s, zz: (s / 0.5, zz / 0.5)))
    resultat["carrelage_piscine"] = fusionner(parois)
    return resultat


def sol_de_base(zone: Polygon, trous: shapely.Geometry, terrain: Terrain) -> Maillage:
    """Sol neutre de toute la zone, sous les classes (comble les raccords)."""
    base = zone.difference(trous) if not trous.is_empty else zone
    return fusionner(
        [surface(p, terrain.hauteur, 1.5, lambda xy: xy / 2.0) for p in polygones(base)]
    )


def socle(zone: Polygon, terrain: Terrain, epaisseur: float) -> Maillage:
    """Socle de maquette découpé selon la zone : flancs verticaux et dessous."""
    zbas = float(terrain.z.min()) - epaisseur
    morceaux = []
    for i, ring in enumerate(anneaux(zone, 1.0)):
        sens = shapely.LinearRing(ring).is_ccw
        ring = ring if sens == (i == 0) else ring[::-1]
        z = terrain.hauteur(ring)
        s = 0.0
        for a in range(len(ring)):
            b = (a + 1) % len(ring)
            seg = float(np.hypot(*(ring[b] - ring[a])))
            if seg < 1e-3:
                continue
            quad = [(0.0, zbas), (seg, zbas), (seg, z[b]), (0.0, z[a])]

            def uv(ss: float, zz: float, s: float = s) -> tuple[float, float]:
                return ((s + ss) / 4.0, zz / 4.0)

            morceaux.append(quad_mur(ring[a], ring[b], quad, uv))
            s += seg
    dessous = surface(zone, lambda xy: np.full(len(xy), zbas), None, lambda xy: xy / 4.0)
    morceaux.append(Maillage(dessous.sommets, dessous.faces[:, [0, 2, 1]], dessous.uv))
    return fusionner(morceaux)
