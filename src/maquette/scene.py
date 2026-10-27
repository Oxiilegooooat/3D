"""Assemblage d'une maquette : données → géométrie → `scene.json` + `maillages.npz` + textures.

Le script Blender (`blender/construire.py`) lit ces fichiers pour produire le GLB et les rendus.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, Polygon, shape
from shapely.geometry.polygon import orient

from maquette import batiments as bt
from maquette import clotures, detections, mobilier, perimetre, sol, vegetation, voirie
from maquette.config import Site, Volume
from maquette.geoplateforme import (
    COUCHE_IRC,
    COUCHE_MNS,
    COUCHE_MNT,
    Client,
    Grille,
    emprise_alignee,
)
from maquette.maillage import Maillage, fusionner, polygones
from maquette.terrain import Terrain
from maquette.textures import catalogue
from maquette.toits import ModeleToit

PARCELLES = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"
BATIMENTS = "BDTOPO_V3:batiment"
ROUTES = "BDTOPO_V3:troncon_de_route"
EAU = "BDTOPO_V3:surface_hydrographique"

AVERTISSEMENTS = [
    "Périmètre : le site, ses parcelles contiguës et celles d'en face ; au-delà, rien n'est "
    "modélisé (perspectives à part).",
    "Hauteurs, toitures, relief, arbres, clôtures, petites constructions et véhicules mesurés "
    "(LiDAR HD de l'IGN, date du survol).",
    "Sol classé d'après la photo aérienne, l'infrarouge et le LiDAR, rendu en matériaux types "
    "(pelouse, gravier, enrobé, dalles…) : l'aspect exact des revêtements est indicatif.",
    "Grillages entre parcelles, grilles sur muret, portails et lampadaires sans source : "
    "indicatifs (le LiDAR ne voit pas un grillage).",
    "Fenêtres, portes et balcons : disposition indicative, générée d'après le nombre d'étages ; "
    "aucune source ouverte ne décrit les façades.",
    "Couleurs des façades indicatives (matériau des murs de la BD TOPO quand il est connu).",
    "Trottoirs déduits du domaine non cadastré ; marquages au sol simplifiés.",
]


@dataclass
class Objet:
    nom: str
    calque: str
    parts: dict[str, list[Maillage]] = field(default_factory=lambda: defaultdict(list))
    infos: dict[str, Any] = field(default_factory=dict)
    # Parties à sommets partagés (ombrage lisse) : surfaces continues, GLB bien plus léger.
    lisses: set[str] = field(
        default_factory=lambda: {"sous_toiture", "trottoir", "asphalte", "chemin", "eau"}
    )

    def ajouter(self, morceaux: dict[str, list[Maillage]]) -> None:
        for mat, liste in morceaux.items():
            self.parts[mat].extend(liste)


def _local(geom: shapely.Geometry, origine: tuple[float, float, float]) -> shapely.Geometry:
    g = shapely.force_2d(geom)
    return shapely.transform(g, lambda c: c - np.array([origine[0], origine[1]]))


def _grille_locale(g: Grille, origine: tuple[float, float, float]) -> Grille:
    x0, y0, z0 = origine
    e = g.emprise
    return Grille(
        (e[0] - x0, e[1] - y0, e[2] - x0, e[3] - y0),
        g.resolution,
        (g.valeurs - z0).astype(np.float32),
    )


def _batiment_projet(
    v: Volume, i: int, origine: tuple[float, float, float], terrain: Terrain, cfg: dict[str, Any]
) -> list[bt.Batiment]:
    emprise = orient(_local(Polygon(v.emprise), origine), sign=1.0)
    contour = np.asarray(shapely.segmentize(emprise.exterior, 1.0).coords)
    z_sol = float(terrain.hauteur(contour).min())
    volumes = []
    niveaux_bas = v.niveaux - (1 if v.attique and v.niveaux > 1 else 0)
    principal = bt.Batiment(f"projet_{i}", emprise, {}, detail=True, projet=True)
    principal.z_sol = z_sol
    principal.etages = niveaux_bas
    principal.h_etage = v.hauteur_etage
    principal.h_gouttiere = niveaux_bas * v.hauteur_etage
    principal.style = "collectif"
    principal.balcons = v.balcons
    principal.type_toit = "terrasse"
    principal.teinte_toit = (0.62, 0.62, 0.6)
    principal.infos = {
        "usage": v.nom or "Projet",
        "style": "projet",
        "forme_toit": "deux pans" if v.toiture == "deux_pans" else "terrasse",
    }
    if v.toiture == "deux_pans" and not v.attique:
        rect = emprise.minimum_rotated_rectangle
        coins = np.asarray(rect.exterior.coords)[:4]
        cotes = np.diff(np.vstack([coins, coins[:1]]), axis=0)
        longueurs = np.hypot(*cotes.T)
        k = int(np.argmax(longueurs[:2]))
        angle = float(np.arctan2(cotes[k, 1], cotes[k, 0]))
        demi_largeur = float(longueurs[1 - k]) / 2
        pente = float(np.tan(np.radians(35)))
        c = rect.centroid
        principal.modele = ModeleToit(
            "deux_pans",
            c.x,
            c.y,
            angle,
            (z_sol + principal.h_gouttiere + demi_largeur * pente, pente, pente, 0.0),
            0.0,
        )
        principal.type_toit = "tuiles"
        principal.teinte_toit = (0.62, 0.36, 0.27)
    else:
        principal.z_toit_plat = z_sol + principal.h_gouttiere
        principal.acrotere = cfg["batiments"]["acrotere"]
    volumes.append(principal)
    if v.attique and v.niveaux > 1:
        retrait = emprise.buffer(-2.0, join_style="mitre")
        for k, poly in enumerate(polygones(retrait)):
            att = bt.Batiment(
                f"projet_{i}_attique_{k}", orient(poly, sign=1.0), {}, detail=True, projet=True
            )
            att.z_sol = principal.z_toit_plat or z_sol
            att.z_base = att.z_sol
            att.etages, att.h_etage = 1, v.hauteur_etage
            att.h_gouttiere = v.hauteur_etage
            att.z_toit_plat = att.z_sol + v.hauteur_etage
            att.acrotere = cfg["batiments"]["acrotere"]
            att.style, att.balcons, att.type_toit = "collectif", False, "terrasse"
            att.teinte_toit = (0.62, 0.62, 0.6)
            volumes.append(att)
    return volumes


def _vers_local_wgs84(origine: tuple[float, float, float]) -> Any:
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

    def convertir(lon: float, lat: float) -> tuple[float, float]:
        x, y = t.transform(lon, lat)
        return x - origine[0], y - origine[1]

    return convertir


def _boite_wgs84(emprise: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    xs, ys = t.transform(
        [emprise[0], emprise[2], emprise[0], emprise[2]],
        [emprise[1], emprise[1], emprise[3], emprise[3]],
    )
    return min(xs), min(ys), max(xs), max(ys)


def construire(site: Site, cfg: dict[str, Any], dossier: Path, force: bool = False) -> Path:
    """Télécharge les données (en cache) et écrit la scène dans `dossier/scene`."""
    client = Client(dossier / "brut", force)
    cp, cc, cs, cb = cfg["perimetre"], cfg["contexte"], cfg["sources"], cfg["batiments"]

    # Périmètre ------------------------------------------------------------------------
    filtre = "idu IN (" + ",".join(f"'{i}'" for i in site.idus) + ")"
    parcelles_site = client.wfs(PARCELLES, "parcelles_site", filtre=filtre)
    manquantes = set(site.idus) - {f["properties"]["idu"] for f in parcelles_site}
    if manquantes:
        raise ValueError(f"parcelles introuvables au cadastre : {sorted(manquantes)}")
    site_l93 = shapely.union_all([shapely.force_2d(shape(f["geometry"])) for f in parcelles_site])
    centre = site_l93.centroid
    xy0 = (float(round(centre.x)), float(round(centre.y)), 0.0)
    m = cp["en_face"] + 40
    x0, y0, x1, y1 = site_l93.bounds
    proches = client.wfs(PARCELLES, "parcelles", emprise=(x0 - m, y0 - m, x1 + m, y1 + m))
    proprietes = {f["properties"]["idu"]: f["properties"] for f in proches}
    parcelles_l = {}
    for f in proches:
        g = polygones(_local(shape(f["geometry"]), xy0))
        if g:
            parcelles_l[f["properties"]["idu"]] = max(g, key=lambda p: p.area)
    site_local = _local(site_l93, xy0)
    perim = perimetre.calculer(site_local, parcelles_l, set(site.idus), cfg)
    zx0, zy0, zx1, zy1 = perim.zone.bounds
    md = cp["marge_donnees"]
    emprise = emprise_alignee(
        zx0 + xy0[0] - md, zy0 + xy0[1] - md, zx1 + xy0[0] + md, zy1 + xy0[1] + md
    )

    # Données de la zone ------------------------------------------------------------------
    batiments_bd = client.wfs(BATIMENTS, "batiments", emprise=emprise)
    routes = client.wfs(ROUTES, "routes", emprise=emprise)
    surfaces_eau = client.wfs(EAU, "eau", emprise=emprise)
    mnt93 = client.elevation(COUCHE_MNT, "mnt", emprise, cs["resolution_lidar"])
    mns93 = client.elevation(COUCHE_MNS, "mns", emprise, cs["resolution_lidar"])
    ortho = client.ortho(emprise, cs["resolution_ortho"])
    irc = client.ortho(emprise, cs["resolution_ortho"], COUCHE_IRC, "irc")
    elements_osm = client.osm(*_boite_wgs84(emprise))
    adresse = client.geocoder(site.adresse) if site.adresse else None

    cz = round(float(mnt93.echantillonner(np.array([centre.x]), np.array([centre.y]))[0]), 1)
    origine = (xy0[0], xy0[1], cz)
    mnt, mns = _grille_locale(mnt93, origine), _grille_locale(mns93, origine)
    mnh = Grille(mns.emprise, mns.resolution, (mns.valeurs - mnt.valeurs).astype(np.float32))
    terrain = Terrain.depuis_mnt(mnt93, origine, cc["pas_terrain"])
    e = mnt.emprise
    emprise_locale = (e[0], e[1], e[2], e[3])
    zone = perim.zone
    retenues = perim.parcelles
    domaine_public = zone.difference(shapely.union_all(list(parcelles_l.values())))

    sortie = dossier / "scene"
    textures = sortie / "textures"
    shutil.rmtree(textures, ignore_errors=True)
    textures.mkdir(parents=True)

    # Bâtiments (BD TOPO, puis petites constructions détectées) -------------------------
    liste: list[bt.Batiment] = []
    for f in batiments_bd:
        g = _local(shape(f["geometry"]), origine)
        if g.intersection(retenues).area < 0.5 * g.area:
            continue
        for k, poly in enumerate(polygones(g)):
            emprise_b = bt.preparer_emprise(poly, 0.15)
            if emprise_b is not None:
                p = f["properties"]
                liste.append(bt.Batiment(f"{p['cleabs']}_{k}", emprise_b, p, detail=True))
    connus = shapely.union_all([b.emprise for b in liste]) if liste else Polygon()
    lignes_routes = [(_local(shape(f["geometry"]), origine), f["properties"]) for f in routes]
    lignes_routes = [(g, p) for g, p in lignes_routes if isinstance(g, LineString)]
    rues, chaussee = voirie.voirie(
        lignes_routes,
        retenues.union(shapely.union_all(list(parcelles_l.values()))),
        connus,
        zone,
        terrain,
        cfg,
    )
    ndvi_lidar = sol.ndvi(
        sol.echantillonner_image(
            irc,
            emprise_locale,
            *np.meshgrid(
                mnh.emprise[0] + (np.arange(mnh.valeurs.shape[1]) + 0.5) * mnh.resolution,
                mnh.emprise[3] - (np.arange(mnh.valeurs.shape[0]) + 0.5) * mnh.resolution,
            ),
        )
    )
    seuil = sol.seuil_vegetal(
        ndvi_lidar[
            shapely.contains_xy(
                zone,
                *np.meshgrid(
                    mnh.emprise[0] + (np.arange(mnh.valeurs.shape[1]) + 0.5) * mnh.resolution,
                    mnh.emprise[3] - (np.arange(mnh.valeurs.shape[0]) + 0.5) * mnh.resolution,
                ),
            )
        ],
        cfg,
    )
    annexes, vehicules = detections.detecter(mnh, ndvi_lidar, zone, connus.buffer(0.6), seuil, cfg)
    for i, rect in enumerate(annexes):
        liste.append(
            bt.Batiment(
                f"LIDAR{i:03d}_0",
                rect,
                {"usage_1": "Annexe", "construction_legere": True, "source": "LiDAR"},
                detail=True,
            )
        )
    for b in liste:
        bt.mesurer(b, mns, terrain, cfg)
        bt.classer(b, cfg)
        bt.teinter_toit(b, ortho, terrain)
        if b.proprietes.get("source") == "LiDAR":
            b.infos["origine"] = "détectée par le LiDAR (absente du cadastre)"

    projet: list[bt.Batiment] = []
    if site.projet:
        for i, v in enumerate(site.projet.volumes):
            projet.extend(_batiment_projet(v, i, origine, terrain, cfg))

    geometries = [b.emprise for b in liste]
    arbre_bat = shapely.STRtree(geometries)
    reseau = shapely.union_all([g for g, _ in lignes_routes]) if lignes_routes else None

    objets: list[Objet] = []
    contexte = Objet("contexte", "contexte")
    toits: dict[str, tuple[str, tuple[float, float, float]]] = {}
    infos_site: list[dict[str, Any]] = []
    for idx, b in enumerate(liste + projet):
        dans_liste = idx < len(liste)
        lignes, contacts = bt.plis(b)
        anneaux = bt.anneaux_batiment(b.emprise, cb["pas_toit"], contacts)
        if dans_liste:
            mitoyens = [bt.mitoyennete(b.emprise, a, arbre_bat, geometries, idx) for a in anneaux]
            proches_b = [
                j for j in arbre_bat.query(b.emprise.buffer(cb["debord_toit"] + 0.1)) if j != idx
            ]
            voisins = shapely.union_all([geometries[j] for j in proches_b]) if proches_b else None
        else:
            mitoyens = [np.zeros(len(a.origines), dtype=bool) for a in anneaux]
            voisins = None
        toit, sous_toit = bt.toit(
            b, anneaux, cb["pas_toit"], lignes, cb["debord_toit"], cb["epaisseur_toit"], voisins
        )
        teinte = tuple(round(v * 10) / 10 for v in b.teinte_toit)
        nom_toit = f"toit_{b.type_toit}_{''.join(str(round(v * 10)) for v in teinte)}"
        toits[nom_toit] = (b.type_toit, (teinte[0], teinte[1], teinte[2]))
        portes = _aretes_porte(anneaux[0], mitoyens[0], reseau)
        morceaux = bt.murs_detail(b, anneaux, mitoyens, terrain, portes, cfg)
        sur_site = b.projet or b.emprise.intersection(site_local).area >= 0.5 * b.emprise.area
        if not sur_site:
            contexte.parts[nom_toit].append(toit)
            contexte.parts["sous_toiture"].append(sous_toit)
            contexte.ajouter(morceaux)
            continue
        calque = "projet" if b.projet else "existant"
        objet = Objet(f"{calque}_{len(infos_site)}", calque)
        objet.parts[nom_toit].append(toit)
        objet.parts["sous_toiture"].append(sous_toit)
        objet.ajouter(morceaux)
        centre_b = b.emprise.representative_point()
        infos = {
            "calque": calque,
            "cle": b.cle.rsplit("_", 1)[0],
            "surface_emprise": round(b.emprise.area),
            "etages": b.etages,
            "hauteur_gouttiere": round(b.h_gouttiere, 1),
            "toiture": b.type_toit,
            "centre": [
                round(centre_b.x, 1),
                round(centre_b.y, 1),
                round(b.z_sol + b.h_gouttiere, 1),
            ],
            **b.infos,
        }
        objet.infos = infos
        infos_site.append(infos)
        objets.append(objet)
    objets.append(contexte)
    # Bâti existant seulement : le sol reste continu sous le projet (masqué en mode existant)
    # et sous l'existant (démoli en mode projet).
    emprises = shapely.union_all(geometries) if geometries else Polygon()

    # Sol, voirie, eau -------------------------------------------------------------------
    osm, piscines_osm = mobilier.lire_osm(elements_osm, _vers_local_wgs84(origine), zone)
    region_sol = retenues.intersection(zone).difference(emprises.buffer(0.05))
    classes = sol.classer(region_sol, ortho, irc, emprise_locale, mnh, cfg)
    if piscines_osm:
        bassins = shapely.union_all(piscines_osm).intersection(region_sol)
        classes = {k: g.difference(bassins) for k, g in classes.items()}
        classes["eau"] = shapely.union_all([classes.get("eau", Polygon()), bassins])
    surfaces_sol, bassins_creuses = sol.maillages(classes, terrain, mnh, cfg)
    objet_sol = Objet(
        "sol",
        "terrain",
        lisses={"sol_base", "sol_pelouse", "sol_gravier", "sol_terre", "sol_enrobe", "sol_dalles"},
    )
    objet_sol.parts["sol_base"].append(sol.sol_de_base(zone, bassins_creuses, terrain))
    objet_sol.parts["socle"].append(sol.socle(zone, terrain, cc["epaisseur_socle"]))
    objet_sol.ajouter(surfaces_sol)
    objets.append(objet_sol)

    objet_voirie = Objet("voirie", "voirie")
    objet_voirie.ajouter(rues)
    eaux = [_local(shape(f["geometry"]), origine) for f in surfaces_eau]
    objet_voirie.parts["eau"].extend(voirie.eau(eaux, zone, terrain))

    # Clôtures, mobilier, véhicules ----------------------------------------------------------
    limites_retenues = shapely.union_all(
        [
            p.boundary
            for p in [*polygones(site_local), *perim.voisines.values(), *perim.en_face.values()]
        ]
    ).intersection(zone.buffer(0.2))
    bord_public = domaine_public.buffer(0.3)
    limites_rue = limites_retenues.intersection(bord_public)
    limites_internes = limites_retenues.difference(bord_public)
    interdit = shapely.union_all([emprises.buffer(0.4), chaussee])
    portails_osm = [(el.x, el.y) for el in osm if el.nature == "portail"]
    clos = clotures.detecter(
        limites_rue, limites_internes, mns, mnt, ortho, emprise_locale, interdit, portails_osm, cfg
    )
    objet_clotures = Objet("clotures", "contexte")
    objet_clotures.ajouter(clotures.maillages(clos, terrain, cfg))
    objets.append(objet_clotures)

    meubles, comptes = mobilier.construire(
        osm, vehicules, lignes_routes, chaussee, zone, emprises, terrain, cfg
    )
    objet_voirie.ajouter(meubles)
    objets.append(objet_voirie)

    exclusion = shapely.union_all([emprises.buffer(cfg["vegetation"]["marge_batiment"]), chaussee])
    arbres = [
        a
        for a in vegetation.detecter(mns, mnt, exclusion, terrain, cfg)
        if zone.contains(shapely.Point(a.x, a.y))
    ]
    objet_arbres = Objet("vegetation", "vegetation")
    c_site = site_local.centroid
    objet_arbres.ajouter(vegetation.maillages(arbres, (c_site.x, c_site.y, 1e6)))
    objets.append(objet_arbres)

    limites = Objet("limites", "limites")
    limites.ajouter(voirie.limite_site(site_local, terrain))
    objets.append(limites)

    # Écriture ----------------------------------------------------------------------
    noms_materiaux = {m for o in objets for m, liste_m in o.parts.items() if liste_m}
    materiaux = catalogue(noms_materiaux, textures, toits, "")
    tableaux: dict[str, np.ndarray] = {}
    description = []
    for i, o in enumerate(objets):
        parts = []
        for j, (mat, morceaux_m) in enumerate(sorted(o.parts.items())):
            mm = fusionner(morceaux_m)
            if not len(mm):
                continue
            cle = f"o{i}_p{j}"
            tableaux[f"{cle}_v"] = mm.sommets.astype(np.float32)
            tableaux[f"{cle}_f"] = mm.faces.astype(np.int32)
            tableaux[f"{cle}_uv"] = mm.uv.astype(np.float32)
            lisse = mat in o.lisses or materiaux[mat].lisse
            parts.append({"materiau": mat, "cle": cle, "triangles": len(mm), "lisse": lisse})
        if parts:
            description.append({"nom": o.nom, "calque": o.calque, "parts": parts, "infos": o.infos})
    np.savez_compressed(sortie / "maillages.npz", allow_pickle=False, **tableaux)

    modeles = []
    if site.projet and site.projet.modele:
        modeles.append(
            {
                "chemin": str(site.projet.modele),
                "calque": "projet",
                "decalage": [-origine[0], -origine[1], -origine[2]],
            }
        )
    scene = {
        "site": site.identifiant,
        "objets": description,
        "materiaux": {n: mt.en_dict() for n, mt in materiaux.items()},
        "modeles": modeles,
        "origine_lambert93": origine,
        "emprise_locale": list(zone.bounds),
    }
    (sortie / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))

    infos = _infos(
        site, parcelles_site, origine, terrain, site_local, adresse, infos_site, arbres, objets, cfg
    )
    infos["voisines"] = _etiquettes(perim.voisines, proprietes, terrain, "contiguë")
    infos["voisines"] += _etiquettes(perim.en_face, proprietes, terrain, "en face")
    infos["contour_zone"] = [
        [round(x, 2), round(y, 2)] for x, y in np.asarray(zone.exterior.coords)[:, :2]
    ]
    infos["rayon_zone"] = round(
        float(np.hypot(zone.bounds[2] - zone.bounds[0], zone.bounds[3] - zone.bounds[1])) / 2, 1
    )
    infos["clotures_m"] = clotures.infos(clos)
    infos["detections"] = {"petites_constructions": len(annexes), **comptes}
    infos["sol_m2"] = {k: round(g.area) for k, g in classes.items()}
    infos["vue_rue"] = _vue_rue(site_local, chaussee, emprises, infos["adresse_locale"], terrain)
    (sortie / "infos.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1))
    return sortie


def _etiquettes(
    parcelles: dict[str, Polygon],
    proprietes: dict[str, dict[str, Any]],
    terrain: Terrain,
    relation: str,
) -> list[dict[str, Any]]:
    resultat = []
    for idu, poly in sorted(parcelles.items()):
        p = proprietes.get(idu, {})
        c = poly.representative_point()
        z = float(terrain.hauteur(np.array([[c.x, c.y]]))[0])
        resultat.append(
            {
                "idu": idu,
                "libelle": f"{str(p.get('section', '')).lstrip('0')} {int(p.get('numero') or 0)}",
                "contenance": p.get("contenance"),
                "relation": relation,
                "centre": [round(c.x, 1), round(c.y, 1), round(z, 1)],
            }
        )
    return resultat


def _vue_rue(
    site_local: shapely.Geometry,
    chaussee: shapely.Geometry,
    emprises: shapely.Geometry,
    adresse: list[float] | None,
    terrain: Terrain,
) -> dict[str, list[float]] | None:
    """Point de vue d'un piéton sur le trottoir d'en face, légèrement de biais."""
    if chaussee.is_empty:
        return None
    depart = (
        shapely.Point(adresse[:2])
        if adresse
        else shapely.ops.nearest_points(site_local.boundary, chaussee)[0]
    )
    bord = shapely.ops.nearest_points(chaussee, depart)[0]
    d = np.array([bord.x - depart.x, bord.y - depart.y])
    if np.hypot(*d) < 0.1:
        c = site_local.centroid
        d = np.array([depart.x - c.x, depart.y - c.y])
    d /= max(float(np.hypot(*d)), 1e-9)
    # Traverser la chaussée jusqu'au trottoir opposé.
    sortie = np.array([bord.x, bord.y]) + d * 0.5
    for _ in range(80):
        if not chaussee.contains(shapely.Point(*sortie)):
            break
        sortie = sortie + d * 0.5
    sortie = sortie + d * 1.2
    lateral = np.array([-d[1], d[0]])
    interdit = emprises.buffer(0.8)
    for decalage in (9.0, -9.0, 5.0, -5.0, 0.0):
        p = sortie + lateral * decalage
        if not interdit.contains(shapely.Point(*p)):
            z = float(terrain.hauteur(p[None, :])[0])
            c = site_local.representative_point()
            zc = float(terrain.hauteur(np.array([[depart.x, depart.y]]))[0])
            return {
                "position": [round(p[0], 2), round(p[1], 2), round(z + 1.65, 2)],
                "cible": [
                    round(depart.x * 0.6 + c.x * 0.4, 2),
                    round(depart.y * 0.6 + c.y * 0.4, 2),
                    round(zc + 3.0, 2),
                ],
            }
    return None


def _aretes_porte(
    anneau: bt.Anneau, mitoyen: np.ndarray, reseau: shapely.Geometry | None
) -> set[tuple[int, int]]:
    """Arête du contour extérieur tournée vers la rue la plus proche (porte d'entrée)."""
    if reseau is None:
        return set()
    k = len(anneau.origines)
    meilleure, distance = None, np.inf
    for i in range(k):
        if mitoyen[i] or anneau.longueurs[i] < 2.5:
            continue
        a, b = anneau.origines[i], anneau.origines[(i + 1) % k]
        m = (a + b) / 2
        d = (b - a) / anneau.longueurs[i]
        ext = np.array([d[1], -d[0]])
        d0 = reseau.distance(shapely.Point(*m))
        d1 = reseau.distance(shapely.Point(*(m + ext * 2.0)))
        if d1 < d0 and d0 < distance:
            meilleure, distance = i, d0
    return set() if meilleure is None else {(0, meilleure)}


def _infos(
    site: Site,
    parcelles_site: list[dict[str, Any]],
    origine: tuple[float, float, float],
    terrain: Terrain,
    site_local: shapely.Geometry,
    adresse: tuple[float, float] | None,
    batiments_site: list[dict[str, Any]],
    arbres: list[Any],
    objets: list[Objet],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    parcelles = []
    for f in sorted(parcelles_site, key=lambda f: f["properties"]["idu"]):
        p = f["properties"]
        g = _local(shape(f["geometry"]), origine)
        c = g.representative_point()
        z = float(terrain.hauteur(np.array([[c.x, c.y]]))[0])
        parcelles.append(
            {
                "idu": p["idu"],
                "libelle": f"{p['section'].lstrip('0')} {int(p['numero'])}",
                "contenance": p.get("contenance"),
                "centre": [round(c.x, 1), round(c.y, 1), round(z, 1)],
            }
        )
    centre_site = site_local.representative_point()
    z_site = float(terrain.hauteur(np.array([[centre_site.x, centre_site.y]]))[0])
    adresse_locale = None
    if adresse:
        ax, ay = adresse[0] - origine[0], adresse[1] - origine[1]
        adresse_locale = [
            round(ax, 1),
            round(ay, 1),
            round(float(terrain.hauteur(np.array([[ax, ay]]))[0]), 1),
        ]
    triangles = sum(len(m) for o in objets for liste in o.parts.values() for m in liste)
    return {
        "site": site.identifiant,
        "nom": site.nom,
        "adresse": site.adresse,
        "adresse_locale": adresse_locale,
        "presentation": site.presentation,
        "projet": None
        if site.projet is None
        else {
            "nom": site.projet.nom,
            "indicatif": site.projet.indicatif,
            "demolition": site.projet.demolition,
        },
        "parcelles": parcelles,
        "surface_cadastrale": sum(p["contenance"] or 0 for p in parcelles),
        "centre_site": [round(centre_site.x, 1), round(centre_site.y, 1), round(z_site, 1)],
        "contour_site": [
            [round(x, 2), round(y, 2)]
            for x, y in np.asarray(polygones(site_local)[0].exterior.coords)[:, :2]
        ],
        "batiments_site": batiments_site,
        "arbres": len(arbres),
        "triangles": triangles,
        "emprise_locale": [terrain.xmin, terrain.ymin, terrain.xmax, terrain.ymax],
        "z_min": round(float(terrain.z.min()), 1),
        "z_max": round(float(terrain.z.max()), 1),
        "origine_lambert93": origine,
        "date": date.today().isoformat(),
        "sources": [
            "Cadastre : Parcellaire Express (IGN)",
            "Bâtiments, routes, eau : BD TOPO (IGN)",
            "Photographie aérienne : BD ORTHO (IGN), pour classer le sol et teinter les toits",
            "Relief, toitures, arbres, clôtures, objets : LiDAR HD, MNT et MNS (IGN)",
            "Infrarouge : BD ORTHO IRC (IGN)",
            "Adresse : Base adresse nationale",
            "Licence ouverte Etalab 2.0",
            "Mobilier et piscines : © les contributeurs d'OpenStreetMap (ODbL)",
        ],
        "avertissements": AVERTISSEMENTS,
        "couleurs": cfg["couleurs"],
    }
