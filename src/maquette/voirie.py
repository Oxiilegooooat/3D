"""Voirie et eau : chaussées (BD TOPO), trottoirs pris dans le domaine non cadastré, bordures,
marquage axial, surfaces en eau. Tout est drapé sur le terrain."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, Polygon

from maquette.maillage import Maillage, Tableau, fusionner, polygones, quad_mur, surface
from maquette.terrain import Terrain

NATURES_IGNOREES = {"Escalier", "Bac ou liaison maritime", "Type autoroutier"}


def largeur(proprietes: dict[str, Any], cfg: dict[str, Any]) -> float:
    valeur = proprietes.get("largeur_de_chaussee")
    if valeur:
        return float(valeur)
    defauts: dict[str, float] = cfg["voirie"]["largeur_defaut"]
    return float(defauts.get(proprietes.get("nature") or "", 4.0))


def voirie(
    troncons: list[tuple[LineString, dict[str, Any]]],
    parcelles: shapely.Geometry,
    batiments: shapely.Geometry,
    fenetre: Polygon,
    terrain: Terrain,
    cfg: dict[str, Any],
) -> tuple[dict[str, list[Maillage]], shapely.Geometry]:
    cv = cfg["voirie"]
    garder = []
    for ligne, p in troncons:
        if str(p.get("position_par_rapport_au_sol") or "0") == "-1":
            continue
        if p.get("nature") in NATURES_IGNOREES:
            continue
        garder.append((ligne, p))
    if not garder:
        return {}, shapely.Polygon()
    domaine_public = fenetre.difference(parcelles)
    interieur = fenetre.buffer(-0.05)
    sentiers = shapely.union_all(
        [ligne.buffer(largeur(p, cfg) / 2) for ligne, p in garder if p.get("nature") == "Sentier"]
    )
    chaussee = shapely.union_all(
        [
            ligne.buffer(largeur(p, cfg) / 2, quad_segs=4)
            for ligne, p in garder
            if p.get("nature") != "Sentier"
        ]
    )
    chaussee = chaussee.intersection(domaine_public.buffer(0.3)).intersection(interieur)
    bande = shapely.union_all(
        [
            ligne.buffer(largeur(p, cfg) / 2 + cv["bande_trottoir"])
            for ligne, p in garder
            if p.get("nature") not in ("Sentier", "Chemin", "Route empierrée")
        ]
    )
    trottoir = (
        bande.intersection(domaine_public)
        .difference(chaussee)
        .difference(batiments)
        .intersection(interieur)
    )
    trottoir = trottoir.buffer(-0.25).buffer(0.25)
    sentiers = (
        sentiers.difference(chaussee)
        .difference(trottoir)
        .difference(batiments)
        .intersection(interieur)
    )

    sortie: dict[str, list[Maillage]] = defaultdict(list)
    h_ch = cv["surelevation_chaussee"]
    h_tr = cv["hauteur_bordure"]

    def uv(echelle: float):  # type: ignore[no-untyped-def]
        return lambda xy: xy / echelle

    for poly in polygones(chaussee):
        sortie["asphalte"].append(
            surface(poly, lambda xy: terrain.hauteur(xy) + h_ch, 2.0, uv(4.0))
        )
    for poly in polygones(sentiers):
        sortie["chemin"].append(surface(poly, lambda xy: terrain.hauteur(xy) + h_ch, 2.0, uv(3.0)))
    for poly in polygones(trottoir):
        if poly.area < 1.0:
            continue
        sortie["trottoir"].append(
            surface(poly, lambda xy: terrain.hauteur(xy) + h_tr, 2.0, uv(2.0))
        )
        for i, ring in enumerate([poly.exterior, *poly.interiors]):
            sortie["bordure"].append(
                _bordure(np.asarray(ring.coords)[:, :2], terrain, h_tr, ring.is_ccw == (i == 0))
            )
    sortie["marquage"].extend(_marquage(garder, chaussee, terrain, h_ch + 0.012, cfg))
    return sortie, chaussee


def _bordure(pts: Tableau, terrain: Terrain, hauteur: float, sens_direct: bool) -> Maillage:
    """Face verticale de la bordure, tournée vers l'extérieur du trottoir."""
    if not sens_direct:
        pts = pts[::-1]
    z = terrain.hauteur(pts)
    morceaux = []
    s = 0.0
    for i in range(len(pts) - 1):
        seg = float(np.hypot(*(pts[i + 1] - pts[i])))
        if seg < 1e-3:
            continue
        poly = [
            (0.0, z[i] - 0.05),
            (seg, z[i + 1] - 0.05),
            (seg, z[i + 1] + hauteur),
            (0.0, z[i] + hauteur),
        ]

        def uv(ss: float, zz: float, s: float = s) -> tuple[float, float]:
            return (s + ss, zz / 0.2)

        morceaux.append(quad_mur(pts[i], pts[i + 1], poly, uv))
        s += seg
    return fusionner(morceaux)


def _marquage(
    troncons: list[tuple[LineString, dict[str, Any]]],
    chaussee: shapely.Geometry,
    terrain: Terrain,
    hauteur: float,
    cfg: dict[str, Any],
) -> list[Maillage]:
    """Ligne axiale discontinue sur les chaussées d'au moins 5 m à double sens."""
    zone = chaussee.buffer(-1.0)
    resultat = []
    for ligne, p in troncons:
        if p.get("nature") not in ("Route à 1 chaussée", "Route à 2 chaussées"):
            continue
        if largeur(p, cfg) < 5 or (p.get("nombre_de_voies") or 2) < 2:
            continue
        if p.get("sens_de_circulation") not in (None, "Double sens"):
            continue
        d = 0.0
        while d < ligne.length - 3.0:
            tiret = shapely.ops.substring(ligne, d, d + 3.0)
            d += 7.0
            bande = tiret.buffer(0.06, cap_style="flat").intersection(zone)
            for poly in polygones(bande):
                resultat.append(
                    surface(poly, lambda xy: terrain.hauteur(xy) + hauteur, 1.0, lambda xy: xy)
                )
    return resultat


def eau(surfaces: list[Polygon], fenetre: Polygon, terrain: Terrain) -> list[Maillage]:
    resultat = []
    for s in surfaces:
        for poly in polygones(s.intersection(fenetre.buffer(-0.05))):
            resultat.append(
                surface(poly, lambda xy: terrain.hauteur(xy) + 0.03, 2.0, lambda xy: xy / 8)
            )
    return resultat


def limite_site(site: Polygon, terrain: Terrain) -> dict[str, list[Maillage]]:
    """Limite de propriété : liseré au sol et voile translucide de 1 m."""
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    for poly in polygones(site):
        lisere = poly.exterior.buffer(0.18)
        for p in polygones(lisere):
            sortie["limite_site"].append(
                surface(p, lambda xy: terrain.hauteur(xy) + 0.09, 1.0, lambda xy: xy)
            )
        pts = np.asarray(shapely.segmentize(poly.exterior, 1.0).coords)[:, :2]
        z = terrain.hauteur(pts)
        for i in range(len(pts) - 1):
            seg = float(np.hypot(*(pts[i + 1] - pts[i])))
            if seg < 1e-3:
                continue
            quad = [(0.0, z[i]), (seg, z[i + 1]), (seg, z[i + 1] + 1.0), (0.0, z[i] + 1.0)]

            def uv(s: float, zz: float, zi: float = float(z[i])) -> tuple[float, float]:
                return (s, zz - zi)

            sortie["voile_site"].append(quad_mur(pts[i], pts[i + 1], quad, uv))
    return sortie
