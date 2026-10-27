"""Bâtiments : hauteurs et toits réels (LiDAR HD), façades par étage, ouvertures en 3D.

Murs percés de vraies ouvertures (tableaux, vitrages, menuiseries, appuis, volets, portes),
murs mitoyens aveugles, toit paramétrique avec débords et couverture déduite de l'orthophoto.

Les hauteurs et les toits sont mesurés. La disposition des fenêtres et des balcons est
**indicative** : aucune source ouverte ne décrit les façades.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from maquette.geoplateforme import Grille
from maquette.maillage import (
    Maillage,
    Tableau,
    boite,
    densifier,
    fusionner,
    orienter_vers_le_haut,
    quad_mur,
    trianguler_anneaux,
)
from maquette.terrain import Terrain
from maquette.toits import ModeleToit, ajuster, plis_dans

CODES_MURS = {"1": "pierre", "2": "meuliere", "4": "brique", "6": "bois"}


@dataclass
class Batiment:
    cle: str
    emprise: Polygon  # repère local, extérieur trigonométrique
    proprietes: dict[str, Any]
    detail: bool = False
    projet: bool = False
    # Mesures
    z_sol: float = 0.0
    z_base: float | None = None  # pied des murs imposé (attique posé sur un toit)
    z_toit_plat: float | None = None
    modele: ModeleToit | None = None
    hauteur_source: str = "lidar"
    h_gouttiere: float = 3.0
    etages: int = 1
    h_etage: float = 3.0
    style: str = "maison"
    revetement: str = "enduit"
    variante: int = 0
    teinte_toit: tuple[float, float, float] = (0.6, 0.6, 0.6)
    type_toit: str = "tuiles"
    acrotere: float = 0.0
    balcons: bool = False
    infos: dict[str, Any] = field(default_factory=dict)

    def hauteur_toit(self, xy: Tableau) -> Tableau:
        if self.modele is not None:
            return np.maximum(self.modele.z(xy), self.z_sol + 2.0)
        z = self.z_toit_plat if self.z_toit_plat is not None else self.z_sol + self.h_gouttiere
        return np.full(len(xy), z)


def _hash(cle: str, n: int) -> int:
    return int(hashlib.sha1(cle.encode()).hexdigest()[:8], 16) % n


def preparer_emprise(geometrie: Polygon, tolerance: float) -> Polygon | None:
    poly = geometrie.simplify(tolerance, preserve_topology=True)
    if poly.is_empty or not isinstance(poly, Polygon) or poly.area < 4:
        return None
    return orient(poly, sign=1.0)


def mesurer(bat: Batiment, mns: Grille, terrain: Terrain, cfg: dict[str, Any]) -> None:
    """Sol de référence, toit (plat ou maillé sur le MNS), hauteur à la gouttière, étages."""
    cb = cfg["batiments"]
    p = bat.proprietes
    contour = np.asarray(shapely.segmentize(bat.emprise.exterior, 1.0).coords)[:, :2]
    bat.z_sol = float(terrain.hauteur(contour).min())

    xmin, _, _, ymax = mns.emprise
    res = mns.resolution
    bx0, by0, bx1, by1 = bat.emprise.bounds
    c0 = max(int((bx0 - xmin) / res) - 2, 0)
    c1 = min(int((bx1 - xmin) / res) + 3, mns.valeurs.shape[1])
    r0 = max(int((ymax - by1) / res) - 2, 0)
    r1 = min(int((ymax - by0) / res) + 3, mns.valeurs.shape[0])
    sous = mns.valeurs[r0:r1, c0:c1].astype(np.float64)
    xs = xmin + (np.arange(c0, c1) + 0.5) * res
    ys = ymax - (np.arange(r0, r1) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)

    masque = np.zeros(sous.shape, dtype=bool)
    for erosion in (cb["erosion_lidar"], cb["erosion_lidar"] / 2, 0.0):
        zone = bat.emprise.buffer(-erosion) if erosion else bat.emprise
        if zone.is_empty:
            continue
        masque = shapely.contains_xy(zone, gx, gy)
        if masque.sum() >= 4:
            break

    hauteur_bdtopo = p.get("hauteur")
    valeurs = sous[masque]
    mediane = float(np.median(valeurs)) if len(valeurs) >= 4 else np.nan
    if len(valeurs) < 4 or mediane - bat.z_sol < 1.8:
        # Pas de mesure LiDAR exploitable (bâtiment minuscule, ou absent lors du survol).
        if hauteur_bdtopo:
            bat.hauteur_source, h = "bdtopo", float(hauteur_bdtopo)
        elif p.get("nombre_d_etages"):
            bat.hauteur_source, h = "etages", float(p["nombre_d_etages"]) * cb["hauteur_etage"]
        else:
            bat.hauteur_source, h = "defaut", cb["hauteur_defaut"]
        bat.z_toit_plat = bat.z_sol + max(h, 2.2)
    else:
        bas, haut = np.percentile(valeurs, [5, 95])
        coeur = valeurs[(valeurs >= bas) & (valeurs <= haut)]
        modele = ajuster(bat.emprise, gx[masque], gy[masque], valeurs)
        if len(coeur) < 4 or float(np.std(coeur)) < cb["ecart_toit_plat"]:
            bat.z_toit_plat = mediane
        elif modele is not None and modele.nature == "plat":
            bat.z_toit_plat = modele.params[0]
        elif modele is not None:
            bat.modele = modele
        else:
            bat.z_toit_plat = float(np.percentile(valeurs, 70))

    bord = np.asarray(shapely.segmentize(bat.emprise.exterior, 0.5).coords)[:, :2]
    z_bord = bat.hauteur_toit(bord)
    bat.h_gouttiere = max(float(np.percentile(z_bord, 10)) - bat.z_sol, 2.2)
    interieur = np.column_stack([gx[masque], gy[masque]]) if masque.any() else bord
    z_max = max(float(z_bord.max()), float(bat.hauteur_toit(interieur).max()))
    bat.infos["hauteur_faitage"] = round(z_max - bat.z_sol, 1)
    bat.infos["hauteur_gouttiere"] = round(bat.h_gouttiere, 1)
    bat.infos["toit_plat"] = bat.z_toit_plat is not None
    bat.infos["forme_toit"] = bat.modele.nature if bat.modele else "plat"
    bat.infos["hauteur_source"] = bat.hauteur_source


def classer(bat: Batiment, cfg: dict[str, Any]) -> None:
    """Style de façade, nombre d'étages et hauteur d'étage."""
    cb = cfg["batiments"]
    p = bat.proprietes
    usage = p.get("usage_1") or "Indifférencié"
    surface = bat.emprise.area
    etages_bd = p.get("nombre_d_etages")
    logements = p.get("nombre_de_logements") or 0
    if usage == "Annexe" or p.get("construction_legere") or surface < 20 or bat.h_gouttiere < 2.6:
        style = "annexe"
    elif (
        usage in ("Industriel", "Agricole", "Commercial et services")
        and (surface > 150 or bat.h_gouttiere > 4.5)
    ) or (usage == "Indifférencié" and surface > 250 and bat.h_gouttiere < 8):
        style = "activite"
    elif usage == "Religieux":
        style = "maison"
    elif logements >= 3 or bat.h_gouttiere > 8.5 or (surface > 300 and usage == "Résidentiel"):
        style = "collectif"
    else:
        style = "maison"
    bat.style = style

    if style in ("activite", "annexe"):
        bat.etages, bat.h_etage = 1, bat.h_gouttiere
    else:
        etages = None
        if etages_bd and 2.3 <= bat.h_gouttiere / etages_bd <= 4.5:
            etages = int(etages_bd)
        if etages is None:
            etages = max(1, round(bat.h_gouttiere / cb["hauteur_etage"]))
        bat.etages = etages
        bat.h_etage = float(
            np.clip(bat.h_gouttiere / etages, cb["hauteur_etage_min"], cb["hauteur_etage_max"])
        )
    code = str(p.get("materiaux_des_murs") or "0")[:1]
    bat.revetement = (
        CODES_MURS.get(code, "enduit") if style in ("maison", "collectif") else "enduit"
    )
    bat.variante = _hash(bat.cle, 4)
    bat.balcons = style == "collectif"
    if bat.z_toit_plat is not None and style in ("collectif", "activite"):
        bat.acrotere = cb["acrotere"]
    bat.infos.update({"style": style, "etages": bat.etages, "usage": usage})


def teinter_toit(bat: Batiment, ortho: Image.Image, terrain: Terrain) -> None:
    """Couleur moyenne du toit sur l'orthophoto et type de couverture déduit."""
    w, h = ortho.size
    bx0, by0, bx1, by1 = (
        bat.emprise.buffer(-0.5).bounds
        if not bat.emprise.buffer(-0.5).is_empty
        else bat.emprise.bounds
    )
    gx, gy = np.meshgrid(np.linspace(bx0, bx1, 24), np.linspace(by0, by1, 24))
    dedans = shapely.contains_xy(bat.emprise.buffer(-0.5), gx, gy)
    if dedans.sum() < 3:
        return
    uv = terrain.uv_ortho(np.column_stack([gx[dedans], gy[dedans]]))
    px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    tableau = np.asarray(ortho)
    rgb = np.median(tableau[py, px].astype(np.float64), axis=0) / 255
    r, g, b = rgb
    luminance = float(0.3 * r + 0.59 * g + 0.11 * b)
    # La photo donne la clarté et la famille de couverture ; la teinte est ramenée à une
    # couleur plausible (ombres et reflets bleutés faussent la teinte brute).
    if bat.z_toit_plat is not None:
        bat.type_toit = "terrasse"
        teinte = (0.62, 0.62, 0.6)
        clarte = luminance / 0.45
    elif r > g * 1.12 and r > b * 1.2:
        bat.type_toit = "tuiles"
        teinte = (0.72, 0.42, 0.31)
        clarte = luminance / 0.42
    elif luminance < 0.33:
        bat.type_toit = "ardoise"
        teinte = (0.38, 0.41, 0.45)
        clarte = luminance / 0.28
    else:
        bat.type_toit = "bac_acier"
        teinte = (0.66, 0.68, 0.7)
        clarte = luminance / 0.5
    k = float(np.clip(clarte, 0.75, 1.2))
    bat.teinte_toit = (
        float(np.clip(teinte[0] * k, 0, 1)),
        float(np.clip(teinte[1] * k, 0, 1)),
        float(np.clip(teinte[2] * k, 0, 1)),
    )


# --------------------------------------------------------------------------- façades


@dataclass
class Anneau:
    points: Tableau
    aretes: NDArray[np.int64]
    abscisses: Tableau
    longueurs: Tableau  # longueur de chaque arête d'origine
    origines: Tableau  # premier point de chaque arête d'origine


def anneaux_batiment(
    emprise: Polygon, pas: float, contacts: list[tuple[float, float]] | None = None
) -> list[Anneau]:
    resultat = []
    extras = np.asarray(contacts, dtype=np.float64) if contacts else None
    for ring in [emprise.exterior, *emprise.interiors]:
        coords = np.asarray(ring.coords)[:-1, :2]
        if len(coords) < 3:
            continue
        pts, aretes, abscisses = densifier(coords, pas, extras)
        suivants = np.roll(coords, -1, axis=0)
        longueurs = np.hypot(*(suivants - coords).T)
        resultat.append(Anneau(pts, aretes, abscisses, longueurs, coords))
    return resultat


def mitoyennete(
    emprise: Polygon, anneau: Anneau, voisins: shapely.STRtree, geometries: list[Polygon], soi: int
) -> NDArray[np.bool_]:
    """Arêtes collées à un autre bâtiment (mur mitoyen : aveugle)."""
    k = len(anneau.origines)
    suivants = np.roll(anneau.origines, -1, axis=0)
    d = suivants - anneau.origines
    n = np.column_stack([d[:, 1], -d[:, 0]]) / np.maximum(anneau.longueurs[:, None], 1e-9)
    resultat = np.zeros(k, dtype=bool)
    for i in range(k):
        for q in (0.25, 0.5, 0.75):
            pt = Point(*(anneau.origines[i] + d[i] * q + n[i] * 0.4))
            proches = voisins.query(pt)
            if any(j != soi and geometries[j].contains(pt) for j in proches):
                resultat[i] = True
                break
    return resultat


def plis(bat: Batiment) -> tuple[list[Tableau], list[tuple[float, float]]]:
    """Faîtages et arêtiers du toit paramétrique (lignes imposées au maillage du toit)."""
    if bat.modele is None:
        return [], []
    lignes, contacts = plis_dans(bat.modele, bat.emprise)
    return [np.asarray(ligne.coords)[:, :2] for ligne in lignes], contacts


def toit(
    bat: Batiment,
    anneaux_: list[Anneau],
    pas: float,
    lignes: list[Tableau] | None = None,
    debord: float = 0.0,
    epaisseur: float = 0.2,
    voisins: shapely.Geometry | None = None,
) -> tuple[Maillage, Maillage]:
    """Couverture (dessus) et, pour un toit en pente avec débord, sous-face et rives.

    Le débord prolonge les pans au-delà des murs (la pente continue : l'égout descend), sans
    empiéter sur les bâtiments voisins.
    """
    pente = bat.z_toit_plat is None
    if pente and debord > 0:
        etendu = bat.emprise.buffer(debord, join_style="mitre", mitre_limit=2.0)
        if voisins is not None and not voisins.is_empty:
            etendu = etendu.difference(voisins).union(bat.emprise)
        morceaux = [g for g in getattr(etendu, "geoms", [etendu]) if isinstance(g, Polygon)]
        poly = orient(max(morceaux, key=lambda g: g.area), sign=1.0).simplify(0.02)
        poly = orient(poly, sign=1.0) if isinstance(poly, Polygon) else bat.emprise
        contacts: list[tuple[float, float]] = []
        if bat.modele is not None:
            plis_l, contacts = plis_dans(bat.modele, poly)
            lignes = [np.asarray(pl.coords)[:, :2] for pl in plis_l]
        anneaux_t = anneaux_batiment(poly, pas, contacts)
    else:
        anneaux_t = anneaux_
    raffiner = pente and bat.modele is None
    xy, faces = trianguler_anneaux([a.points for a in anneaux_t], pas if raffiner else None, lignes)
    if not len(faces):
        return Maillage.vide(), Maillage.vide()
    faces = orienter_vers_le_haut(xy, faces)
    z = bat.hauteur_toit(xy)
    # Rangs de tuiles parallèles au faîtage (axe principal de l'emprise), tuile de 2 m.
    angle = _axe_principal(bat.emprise)
    c, s = np.cos(-angle), np.sin(-angle)
    uv = np.column_stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c]) / 2.0
    dessus = Maillage(np.column_stack([xy, z]), faces, uv)
    if not (pente and debord > 0):
        return dessus, Maillage.vide()
    morceaux_bord = [Maillage(np.column_stack([xy, z - epaisseur]), faces[:, [0, 2, 1]], xy / 2.0)]
    for a in anneaux_t:
        pts = a.points
        zt = bat.hauteur_toit(pts)
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            seg = float(np.hypot(*(pts[j] - pts[i])))
            if seg < 1e-3:
                continue
            quad = [(0.0, zt[i] - epaisseur), (seg, zt[j] - epaisseur), (seg, zt[j]), (0.0, zt[i])]
            morceaux_bord.append(quad_mur(pts[i], pts[j], quad, lambda s_, z_: (s_, z_)))
    return dessus, fusionner(morceaux_bord)


def _axe_principal(poly: Polygon) -> float:
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    cotes = np.diff(rect[:3], axis=0)
    long = cotes[np.argmax(np.hypot(*cotes.T))]
    return float(np.arctan2(long[1], long[0]))


def materiau_mur(bat: Batiment) -> str:
    """Nom du matériau de façade (voir `textures.facade`) ; la variante d'un enduit est sa
    teinte."""
    if bat.projet:
        return "facade_projet"
    if bat.style in ("activite", "annexe"):
        return f"facade_{bat.style}_{bat.variante % 2}"
    if bat.revetement != "enduit":
        return f"facade_{bat.revetement}"
    return f"facade_enduit_{bat.variante}"


# --------------------------------------------------------------------------- détail


@dataclass
class Ouverture:
    s0: float
    s1: float
    z0: float
    z1: float
    nature: str  # fenetre, porte, porte_activite, bandeau, porte_fenetre


def _ouvertures(
    bat: Batiment,
    longueur: float,
    z_bas: Callable[[float], float],
    z_haut: Callable[[float], float],
    porte: bool,
    cfg: dict[str, Any],
) -> list[Ouverture]:
    cd = cfg["detail"]
    cb = cfg["batiments"]
    ouvertures: list[Ouverture] = []
    if bat.style == "activite":
        travee = cb["largeur_travee"] * 2
        n = int(longueur // travee)
        haut_bandeau = bat.z_sol + bat.h_gouttiere - 0.6
        for k in range(n):
            c = (k + 0.5) * longueur / n
            ouvertures.append(
                Ouverture(c - 1.1, c + 1.1, haut_bandeau - 0.9, haut_bandeau, "bandeau")
            )
        if porte:
            largeur = min(cd["porte_activite_largeur"], longueur - 2)
            hauteur = min(cd["porte_activite_hauteur"], bat.h_gouttiere - 1.2)
            if largeur >= 2.5 and hauteur >= 2.4:
                c = longueur / 2
                ouvertures = [
                    o
                    for o in ouvertures
                    if o.s1 < c - largeur / 2 - 0.3
                    or o.s0 > c + largeur / 2 + 0.3
                    or o.z0 > bat.z_sol + hauteur + 0.3
                ]
                ouvertures.append(
                    Ouverture(
                        c - largeur / 2,
                        c + largeur / 2,
                        bat.z_sol,
                        bat.z_sol + hauteur,
                        "porte_activite",
                    )
                )
                s = c + largeur / 2 + 1.2
                if s + cd["porte_largeur"] < longueur - 0.5:
                    ouvertures.append(
                        Ouverture(
                            s,
                            s + cd["porte_largeur"],
                            bat.z_sol,
                            bat.z_sol + cd["porte_hauteur"],
                            "porte",
                        )
                    )
    elif bat.style == "annexe":
        if porte and longueur > 2:
            c = longueur / 2
            ouvertures.append(
                Ouverture(
                    c - cd["porte_largeur"] / 2,
                    c + cd["porte_largeur"] / 2,
                    bat.z_sol,
                    bat.z_sol + min(cd["porte_hauteur"], bat.h_gouttiere - 0.3),
                    "porte",
                )
            )
    else:
        n = round(longueur / cb["largeur_travee"]) if longueur >= 2.5 else 0
        projet = bat.projet
        largeur = 1.6 if projet else cd["fenetre_largeur"]
        for etage in range(bat.etages):
            plancher = bat.z_sol + etage * bat.h_etage
            for k in range(n):
                c = (k + 0.5) * longueur / n
                nature = "fenetre"
                z0, z1 = plancher + cd["allege"], plancher + cd["allege"] + cd["fenetre_hauteur"]
                if projet and (etage > 0 or k % 2 == 0):
                    nature, z0, z1 = "porte_fenetre", plancher + 0.05, plancher + 2.25
                if porte and etage == 0 and k == n // 2:
                    nature, z0, z1 = "porte", plancher, plancher + cd["porte_hauteur"]
                    largeur_k = cd["porte_largeur"]
                else:
                    largeur_k = largeur
                ouvertures.append(Ouverture(c - largeur_k / 2, c + largeur_k / 2, z0, z1, nature))
    # On ne garde que les ouvertures qui tiennent dans le mur (pignons, terrain en pente).
    gardees = []
    for o in ouvertures:
        marge_haut = min(z_haut(o.s0), z_haut(o.s1), z_haut((o.s0 + o.s1) / 2)) - 0.35
        marge_bas = max(z_bas(o.s0), z_bas(o.s1)) + cfg["batiments"]["enfouissement"]
        if (
            o.s0 > 0.4
            and o.s1 < longueur - 0.4
            and o.z1 <= marge_haut
            and (o.z0 >= marge_bas - 0.05 or o.nature.startswith("porte"))
        ):
            gardees.append(o)
    return gardees


def murs_detail(
    bat: Batiment,
    anneaux_: list[Anneau],
    mitoyens: list[NDArray[np.bool_]],
    terrain: Terrain,
    aretes_porte: set[tuple[int, int]],
    cfg: dict[str, Any],
) -> dict[str, list[Maillage]]:
    """Murs percés d'ouvertures en 3D : tableaux, vitrages, menuiseries, appuis, volets, portes."""
    cb = cfg["batiments"]
    cd = cfg["detail"]
    prof = cd["profondeur_tableau"]
    mat_mur = materiau_mur(bat)
    sortie: dict[str, list[Maillage]] = defaultdict(list)
    for ia, (anneau, mitoyen) in enumerate(zip(anneaux_, mitoyens, strict=True)):
        k = len(anneau.origines)
        for arete in range(k):
            a = anneau.origines[arete]
            b = anneau.origines[(arete + 1) % k]
            longueur = float(anneau.longueurs[arete])
            if longueur < 0.05:
                continue
            u2 = (b - a) / longueur
            ext2 = np.array([u2[1], -u2[0]])
            sel = anneau.aretes == arete
            s_pts = np.append(anneau.abscisses[sel], longueur)
            xy = np.vstack([anneau.points[sel], b])
            zb = (
                np.full(len(xy), bat.z_base)
                if bat.z_base is not None
                else terrain.hauteur(xy) - cb["enfouissement"]
            )
            zh = bat.hauteur_toit(xy) + bat.acrotere

            def z_bas(s: float, s_pts: Tableau = s_pts, zb: Tableau = zb) -> float:
                return float(np.interp(s, s_pts, zb))

            def z_haut(s: float, s_pts: Tableau = s_pts, zh: Tableau = zh) -> float:
                return float(np.interp(s, s_pts, zh))

            ouvertures = (
                []
                if mitoyen[arete]
                else _ouvertures(bat, longueur, z_bas, z_haut, (ia, arete) in aretes_porte, cfg)
            )
            contour = Polygon(
                list(zip(s_pts, zb, strict=True)) + list(zip(s_pts[::-1], zh[::-1], strict=True))
            )
            if not contour.is_valid:
                contour = contour.buffer(0)
            trous = (
                shapely.union_all([shapely.box(o.s0, o.z0, o.s1, o.z1) for o in ouvertures])
                if ouvertures
                else None
            )
            plein = contour.difference(trous) if trous is not None else contour

            def vers_3d(
                s: Tableau,
                z: Tableau,
                d: float | Tableau = 0.0,
                a: Tableau = a,
                u2: Tableau = u2,
                ext2: Tableau = ext2,
            ) -> Tableau:
                s = np.asarray(s, dtype=np.float64)
                d = np.broadcast_to(np.asarray(d, dtype=np.float64), s.shape)
                x = a[0] + u2[0] * s - ext2[0] * d
                y = a[1] + u2[1] * s - ext2[1] * d
                return np.column_stack([x, y, z])

            for morceau in getattr(plein, "geoms", [plein]):
                if not isinstance(morceau, Polygon) or morceau.area < 1e-4:
                    continue
                anneaux_sz = [np.asarray(morceau.exterior.coords)[:-1]] + [
                    np.asarray(i.coords)[:-1] for i in morceau.interiors
                ]
                sz, faces = trianguler_anneaux(anneaux_sz)
                if not len(faces):
                    continue
                faces = orienter_vers_le_haut(sz, faces)
                sortie[mat_mur].append(
                    Maillage(
                        vers_3d(sz[:, 0], sz[:, 1]),
                        faces,
                        np.column_stack([sz[:, 0] / 3, sz[:, 1] / 3]),
                    )
                )
            for o in ouvertures:
                _habiller(o, vers_3d, u2, ext2, a, bat, prof, mat_mur, sortie)
    return sortie


def _quad(p: Tableau, normale: Tableau, uv_echelle: float = 1.0) -> Maillage:
    """Quadrilatère (4 sommets) orienté selon `normale`."""
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    if np.dot(n, normale) < 0:
        faces = faces[:, [0, 2, 1]]
    d1 = float(np.linalg.norm(p[1] - p[0]))
    d2 = float(np.linalg.norm(p[3] - p[0]))
    uv = np.array([(0, 0), (d1, 0), (d1, d2), (0, d2)]) / uv_echelle
    return Maillage(np.asarray(p, dtype=np.float64), faces, uv)


def _habiller(
    o: Ouverture,
    vers_3d: Callable[..., Tableau],
    u2: Tableau,
    ext2: Tableau,
    a: Tableau,
    bat: Batiment,
    prof: float,
    mat_mur: str,
    sortie: dict[str, list[Maillage]],
) -> None:
    ext = np.array([ext2[0], ext2[1], 0.0])
    u = np.array([u2[0], u2[1], 0.0])
    zv = np.array([0.0, 0.0, 1.0])
    s = np.array([o.s0, o.s1, o.s1, o.s0])
    z = np.array([o.z0, o.z0, o.z1, o.z1])
    devant = vers_3d(s, z, 0.0)
    fond = vers_3d(s, z, prof)
    # Tableaux (côtés de l'ouverture), tournés vers l'intérieur de l'ouverture.
    tableaux = [((0, 1), zv), ((2, 3), -zv), ((1, 2), -u), ((3, 0), u)]
    for (i, j), normale in tableaux:
        if o.nature.startswith("porte") and i == 0:
            continue
        sortie[mat_mur].append(
            _quad(np.array([devant[i], devant[j], fond[j], fond[i]]), normale, 3.0)
        )
    largeur, hauteur = o.s1 - o.s0, o.z1 - o.z0
    if o.nature == "porte_activite":
        panneau = vers_3d(s, z, prof)
        sortie["porte_sectionnelle"].append(_quad(panneau, ext, 1.0))
        return
    if o.nature == "porte":
        sortie[f"porte_{bat.variante % 3}"].append(_quad(vers_3d(s, z, prof), ext, 1.0))
        return
    # Vitrage en retrait, menuiserie (cadre et meneau), appui saillant.
    sortie["verre"].append(_quad(vers_3d(s, z, prof), ext, 1.0))
    e = 0.06
    mat_menuiserie = "menuiserie_projet" if bat.projet else "menuiserie"
    cadre = [
        (o.s0, o.s1, o.z0, o.z0 + e),
        (o.s0, o.s1, o.z1 - e, o.z1),
        (o.s0, o.s0 + e, o.z0, o.z1),
        (o.s1 - e, o.s1, o.z0, o.z1),
    ]
    if largeur > 1.0 and o.nature != "bandeau":
        m = (o.s0 + o.s1) / 2
        cadre.append((m - e / 2, m + e / 2, o.z0, o.z1))
    if o.nature == "bandeau":
        for k in range(1, max(2, round(largeur / 1.1))):
            m = o.s0 + k * largeur / max(2, round(largeur / 1.1))
            cadre.append((m - e / 2, m + e / 2, o.z0, o.z1))
    for s0, s1, z0, z1 in cadre:
        centre = vers_3d(np.array([(s0 + s1) / 2]), np.array([z0]), prof - 0.04)[0]
        sortie[mat_menuiserie].append(boite(centre, u2, s1 - s0, 0.06, z1 - z0))
    if o.nature in ("fenetre", "bandeau"):
        # Appui : dépasse de 6 cm de la façade, recouvre le tableau bas.
        centre = vers_3d(np.array([(o.s0 + o.s1) / 2]), np.array([o.z0 - 0.04]), (prof - 0.06) / 2)[
            0
        ]
        sortie["appui"].append(boite(centre, u2, largeur + 0.1, prof + 0.06, 0.06))
    if o.nature == "fenetre" and bat.style == "maison" and not bat.projet:
        # Volets battants ouverts contre la façade.
        for cote, s_bord in ((-1, o.s0), (1, o.s1)):
            sc = s_bord + cote * largeur / 4
            centre = vers_3d(np.array([sc]), np.array([o.z0]), -0.03)[0]
            sortie[f"volet_{bat.variante}"].append(boite(centre, u2, largeur / 2, 0.035, hauteur))
    if o.nature == "porte_fenetre" and bat.balcons:
        centre_sol = vers_3d(np.array([(o.s0 + o.s1) / 2]), np.array([o.z0 - 0.2]), 0.0)[0]
        if o.z0 - bat.z_sol > 1.0:
            p_balcon = 1.3
            c = centre_sol.copy()
            c[:2] += ext2 * p_balcon / 2
            sortie["beton_clair"].append(boite(c, -u2, largeur + 1.0, p_balcon, 0.18))
            garde = c.copy()
            garde[:2] += ext2 * (p_balcon / 2 - 0.03)
            garde[2] += 0.18
            sortie["verre_garde_corps"].append(boite(garde, -u2, largeur + 1.0, 0.03, 1.0))
            for cote in (-1, 1):
                lat = c.copy()
                lat[:2] += u2 * cote * (largeur + 1.0) / 2
                lat[2] += 0.18
                sortie["verre_garde_corps"].append(boite(lat, ext2, p_balcon, 0.03, 1.0))
