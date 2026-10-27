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
from maquette import clotures, vegetation, voirie
from maquette.config import Site, Volume
from maquette.geoplateforme import COUCHE_MNS, COUCHE_MNT, Client, Grille, emprise_alignee
from maquette.maillage import Maillage, fusionner, polygones
from maquette.terrain import Terrain
from maquette.textures import catalogue
from maquette.toits import ModeleToit

PARCELLES = "CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle"
BATIMENTS = "BDTOPO_V3:batiment"
ROUTES = "BDTOPO_V3:troncon_de_route"
EAU = "BDTOPO_V3:surface_hydrographique"

AVERTISSEMENTS = [
    "Hauteurs, toitures, relief et arbres mesurés (LiDAR HD de l'IGN).",
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


def construire(site: Site, cfg: dict[str, Any], dossier: Path, force: bool = False) -> Path:
    """Télécharge les données (en cache) et écrit la scène dans `dossier/scene`."""
    client = Client(dossier / "brut", force)
    cc, cs = cfg["contexte"], cfg["sources"]

    filtre = "idu IN (" + ",".join(f"'{i}'" for i in site.idus) + ")"
    parcelles_site = client.wfs(PARCELLES, "parcelles_site", filtre=filtre)
    trouvees = {f["properties"]["idu"] for f in parcelles_site}
    manquantes = set(site.idus) - trouvees
    if manquantes:
        raise ValueError(f"parcelles introuvables au cadastre : {sorted(manquantes)}")
    site_l93 = shapely.union_all([shapely.force_2d(shape(f["geometry"])) for f in parcelles_site])
    x0, y0, x1, y1 = site_l93.bounds
    r = cc["rayon"]
    emprise = emprise_alignee(x0 - r, y0 - r, x1 + r, y1 + r)

    parcelles = client.wfs(PARCELLES, "parcelles", emprise=emprise)
    batiments_bd = client.wfs(BATIMENTS, "batiments", emprise=emprise)
    routes = client.wfs(ROUTES, "routes", emprise=emprise)
    surfaces_eau = client.wfs(EAU, "eau", emprise=emprise)
    mnt93 = client.elevation(COUCHE_MNT, "mnt", emprise, cs["resolution_lidar"])
    mns93 = client.elevation(COUCHE_MNS, "mns", emprise, cs["resolution_lidar"])
    ortho = client.ortho(emprise, cs["resolution_ortho"])
    adresse = client.geocoder(site.adresse) if site.adresse else None

    centre = site_l93.centroid
    cx, cy = round(centre.x), round(centre.y)
    cz = round(float(mnt93.echantillonner(np.array([centre.x]), np.array([centre.y]))[0]), 1)
    origine = (float(cx), float(cy), float(cz))
    mnt, mns = _grille_locale(mnt93, origine), _grille_locale(mns93, origine)
    terrain = Terrain.depuis_mnt(mnt93, origine, cc["pas_terrain"])
    fenetre = shapely.box(terrain.xmin, terrain.ymin, terrain.xmax, terrain.ymax)
    site_local = _local(site_l93, origine)

    sortie = dossier / "scene"
    textures = sortie / "textures"
    shutil.rmtree(textures, ignore_errors=True)
    textures.mkdir(parents=True)
    ortho.save(textures / "ortho.jpg", quality=86)

    objets: list[Objet] = []
    sol = Objet("terrain", "terrain")
    sol.parts["ortho"].append(terrain.maillage())
    sol.parts["socle"].append(terrain.socle(cc["epaisseur_socle"]))
    objets.append(sol)

    # Bâtiments ------------------------------------------------------------------------
    cb = cfg["batiments"]
    interieur = fenetre.buffer(-1.0)
    liste: list[bt.Batiment] = []
    for f in batiments_bd:
        g = _local(shape(f["geometry"]), origine)
        if not interieur.contains(g):
            continue
        part_site = g.intersection(site_local).area / max(g.area, 1e-9)
        detail = part_site >= 0.5
        for k, poly in enumerate(polygones(g)):
            emprise_b = bt.preparer_emprise(poly, 0.15 if detail else 0.3)
            if emprise_b is None:
                continue
            p = f["properties"]
            liste.append(bt.Batiment(f"{p['cleabs']}_{k}", emprise_b, p, detail=detail))
    for b in liste:
        bt.mesurer(b, mns, terrain, cfg)
        bt.classer(b, cfg)
        if b.detail:
            bt.teinter_toit(b, ortho, terrain)

    projet: list[bt.Batiment] = []
    if site.projet:
        for i, v in enumerate(site.projet.volumes):
            projet.extend(_batiment_projet(v, i, origine, terrain, cfg))

    geometries = [b.emprise for b in liste]
    arbre_bat = shapely.STRtree(geometries)
    lignes_routes = [(_local(shape(f["geometry"]), origine), f["properties"]) for f in routes]
    lignes_routes = [(g, p) for g, p in lignes_routes if isinstance(g, LineString)]
    reseau = shapely.union_all([g for g, _ in lignes_routes]) if lignes_routes else None

    contexte = Objet("contexte", "contexte")
    toits: dict[str, tuple[str, tuple[float, float, float]]] = {}
    infos_site: list[dict[str, Any]] = []
    for idx, b in enumerate(liste + projet):
        dans_liste = idx < len(liste)
        pas = cb["pas_toit_detail"] if b.detail else cb["pas_toit_contexte"]
        lignes, contacts = bt.plis(b)
        anneaux = bt.anneaux_batiment(b.emprise, pas, contacts)
        if dans_liste:
            mitoyens = [bt.mitoyennete(b.emprise, a, arbre_bat, geometries, idx) for a in anneaux]
        else:
            mitoyens = [np.zeros(len(a.origines), dtype=bool) for a in anneaux]
        voisins = None
        if dans_liste:
            proches = [
                j for j in arbre_bat.query(b.emprise.buffer(cb["debord_toit"] + 0.1)) if j != idx
            ]
            voisins = shapely.union_all([geometries[j] for j in proches]) if proches else None
        toit, sous_toit = bt.toit(
            b, anneaux, pas, terrain, lignes, cb["debord_toit"], cb["epaisseur_toit"], voisins
        )
        if not b.detail:
            contexte.parts["ortho"].append(toit)
            contexte.parts["sous_toiture"].append(sous_toit)
            contexte.ajouter(bt.murs_contexte(b, anneaux, mitoyens, terrain, cfg))
            contexte.ajouter(bt.balcons_contexte(b, anneaux, mitoyens, cfg))
            continue
        nom_toit = f"toit_{len(toits)}"
        toits[nom_toit] = (b.type_toit, b.teinte_toit)
        portes = _aretes_porte(anneaux[0], mitoyens[0], reseau)
        calque = "projet" if b.projet else "existant"
        objet = Objet(f"{calque}_{len(infos_site)}", calque)
        objet.parts[nom_toit].append(toit)
        objet.parts["sous_toiture"].append(sous_toit)
        objet.ajouter(bt.murs_detail(b, anneaux, mitoyens, terrain, portes, cfg))
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

    # Voirie, eau, végétation, limites ------------------------------------------------
    parcelles_union = shapely.union_all([_local(shape(f["geometry"]), origine) for f in parcelles])
    emprises = shapely.union_all(geometries + [b.emprise for b in projet])
    rues, chaussee = voirie.voirie(lignes_routes, parcelles_union, emprises, fenetre, terrain, cfg)
    objet_voirie = Objet("voirie", "voirie")
    objet_voirie.ajouter(rues)
    eaux = [_local(shape(f["geometry"]), origine) for f in surfaces_eau]
    objet_voirie.parts["eau"].extend(voirie.eau(eaux, fenetre, terrain))
    objets.append(objet_voirie)

    limites_parcelles = shapely.union_all(
        [_local(shape(f["geometry"]), origine).boundary for f in parcelles]
    ).intersection(fenetre.buffer(-1.0))
    murs = clotures.detecter(
        limites_parcelles,
        mns,
        mnt,
        ortho,
        terrain,
        shapely.union_all([emprises.buffer(0.4), chaussee]),
    )
    objet_voirie.ajouter(clotures.maillages(murs, terrain))

    exclusion = shapely.union_all([emprises.buffer(cfg["vegetation"]["marge_batiment"]), chaussee])
    arbres = vegetation.detecter(mns, mnt, exclusion, terrain, cfg)
    objet_arbres = Objet("vegetation", "vegetation")
    c_site = site_local.centroid
    objet_arbres.ajouter(vegetation.maillages(arbres, (c_site.x, c_site.y, 70.0)))
    objets.append(objet_arbres)

    limites = Objet("limites", "limites")
    limites.ajouter(voirie.limite_site(site_local, terrain))
    objets.append(limites)

    # Écriture ----------------------------------------------------------------------
    noms_materiaux = {m for o in objets for m, liste_m in o.parts.items() if liste_m}
    materiaux = catalogue(noms_materiaux, textures, toits, "ortho.jpg")
    tableaux: dict[str, np.ndarray] = {}
    description = []
    for i, o in enumerate(objets):
        parts = []
        for j, (mat, morceaux) in enumerate(sorted(o.parts.items())):
            m = fusionner(morceaux)
            if not len(m):
                continue
            cle = f"o{i}_p{j}"
            tableaux[f"{cle}_v"] = m.sommets.astype(np.float32)
            tableaux[f"{cle}_f"] = m.faces.astype(np.int32)
            tableaux[f"{cle}_uv"] = m.uv.astype(np.float32)
            parts.append({"materiau": mat, "cle": cle, "triangles": len(m)})
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
        "materiaux": {n: m.en_dict() for n, m in materiaux.items()},
        "modeles": modeles,
        "origine_lambert93": origine,
        "emprise_locale": [terrain.xmin, terrain.ymin, terrain.xmax, terrain.ymax],
    }
    (sortie / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))

    infos = _infos(
        site, parcelles_site, origine, terrain, site_local, adresse, infos_site, arbres, objets, cfg
    )
    infos["clotures_m"] = clotures.infos(murs)
    infos["vue_rue"] = _vue_rue(site_local, chaussee, emprises, infos["adresse_locale"], terrain)
    (sortie / "infos.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1))
    return sortie


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
            "Photographie aérienne : BD ORTHO (IGN)",
            "Relief, toitures, arbres : LiDAR HD, MNT et MNS (IGN)",
            "Adresse : Base adresse nationale",
            "Licence ouverte Etalab 2.0",
        ],
        "avertissements": AVERTISSEMENTS,
        "couleurs": cfg["couleurs"],
    }
