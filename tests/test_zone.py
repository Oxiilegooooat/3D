"""Périmètre, sol, détections, clôtures et mobilier sur données synthétiques (hors réseau)."""

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import LineString, Polygon, box

from maquette import clotures, detections, mobilier, perimetre, sol
from maquette.config import charger_config
from maquette.geoplateforme import Grille
from maquette.maillage import trianguler_robuste
from maquette.terrain import Terrain

CFG = charger_config()


def terrain_plat(demi: float = 80.0) -> Terrain:
    n = int(2 * demi) + 1
    return Terrain(-demi, -demi, 1.0, np.zeros((n, n)))


def grille(valeurs_fn, emprise=(-40.0, -40.0, 40.0, 40.0), res: float = 0.5) -> Grille:
    x0, y0, x1, y1 = emprise
    xs = x0 + (np.arange(int((x1 - x0) / res)) + 0.5) * res
    ys = y1 - (np.arange(int((y1 - y0) / res)) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    return Grille(emprise, res, valeurs_fn(gx, gy).astype(np.float32))


def test_perimetre_contigues_et_en_face() -> None:
    site = box(0, 0, 20, 30)
    parcelles = {
        "SITE": site,
        "CONTIGUE": box(20, 0, 40, 30),
        "DEUXIEME_RANG": box(40, 0, 60, 30),
        "EN_FACE": box(0, 40, 20, 70),  # de l'autre côté d'une voie de 10 m
        "LOIN": box(0, 110, 20, 130),
        "DERRIERE": box(0, -30, 20, 0),
    }
    p = perimetre.calculer(site, parcelles, {"SITE"}, CFG)
    assert set(p.voisines) == {"CONTIGUE", "DERRIERE"}
    assert set(p.en_face) == {"EN_FACE"}
    # La voie entre le site et la parcelle d'en face fait partie de la zone.
    assert p.zone.contains(shapely.Point(10, 35))
    assert not p.zone.contains(shapely.Point(50, 15))


def test_triangulation_robuste_trous_qui_se_touchent() -> None:
    zone = box(0, 0, 30, 30).difference(box(10, 10, 15, 15)).difference(box(15, 15, 20, 20))
    zone = zone.difference(box(0, 5, 5, 10))  # échancrure sur le bord
    xy, faces = trianguler_robuste(zone, 1.5)
    a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
    aire = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    assert abs(aire.sum() / 2 - zone.area) < 0.01


def test_classement_du_sol() -> None:
    emprise = (0.0, 0.0, 20.0, 20.0)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    irc = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:, :50] = (90, 130, 60)  # pelouse à l'ouest
    irc[:, :50] = (220, 60, 90)  # proche infrarouge fort
    rgb[:, 50:] = (196, 194, 188)  # dalles claires à l'est
    irc[:, 50:] = (150, 160, 150)
    rgb[10:30, 70:90] = (110, 200, 220)  # bassin (au nord-est)
    irc[10:30, 70:90] = (60, 110, 200)
    mnh = grille(lambda x, y: 0 * x, emprise, 0.5)
    classes = sol.classer(
        box(0, 0, 20, 20), Image.fromarray(rgb), Image.fromarray(irc), emprise, mnh, CFG
    )
    assert set(classes) >= {"pelouse", "dalles", "eau"}
    assert abs(classes["pelouse"].area - 200) < 15
    assert classes["eau"].contains(shapely.Point(16, 16))
    surfaces, bassins = sol.maillages(classes, terrain_plat(), mnh, CFG)
    assert surfaces["eau_piscine"] and surfaces["margelle"] and not bassins.is_empty


def test_vehicule_et_abri_detectes_mur_ignore() -> None:
    def hauteurs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.zeros_like(x)
        h[(np.abs(x - 10) < 2.25) & (np.abs(y - 10) < 0.9)] = 1.5  # voiture 4,5 × 1,8 m
        h[(np.abs(x + 10) < 2.0) & (np.abs(y + 10) < 1.5)] = 2.5  # abri 4 × 3 m
        h[(np.abs(x) < 5.0) & (np.abs(y + 25) < 0.12)] = 1.8  # mur mince
        return h

    mnh = grille(hauteurs)
    ndvi = np.full(mnh.valeurs.shape, -0.3)
    abris, vehicules = detections.detecter(mnh, ndvi, box(-40, -40, 40, 40), Polygon(), 0.0, CFG)
    assert len(vehicules) == 1 and abs(vehicules[0].x - 10) < 0.5
    assert 3.8 < vehicules[0].longueur <= 4.8
    assert len(abris) == 1 and abris[0].contains(shapely.Point(-10, -10))


def test_clotures_sur_rue_et_entre_parcelles() -> None:
    def hauteurs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.zeros_like(x)
        pres = np.abs(y) < 0.3
        h[pres & (x >= 0) & (x < 10)] = 1.8  # mur plein
        h[pres & (x >= 13.5) & (x < 30)] = 1.0  # muret surmonté d'une grille
        return h

    mns = grille(hauteurs)
    mnt = grille(lambda x, y: 0 * x)
    ortho = Image.new("RGB", (400, 400), (170, 165, 155))
    resultat = clotures.detecter(
        LineString([(0, 0), (30, 0)]),
        LineString([(0, 0), (0, -20)]),
        mns,
        mnt,
        ortho,
        (-40.0, -40.0, 40.0, 40.0),
        Polygon(),
        [],
        CFG,
    )
    natures = {c.nature for c in resultat}
    assert natures == {"mur", "muret_grille", "portail", "grillage"}
    portail = next(c for c in resultat if c.nature == "portail")
    assert 2.4 <= portail.ligne.length <= 6.0
    maillages = clotures.maillages(resultat, terrain_plat(), CFG)
    assert maillages["grille"] and maillages["chaperon"] and maillages["grillage"]


def test_passage_pieton_et_lecture_osm() -> None:
    elements = [
        {
            "type": "node",
            "id": 1,
            "lon": 0.0,
            "lat": 0.0,
            "tags": {"highway": "crossing", "crossing": "marked"},
        },
        {"type": "node", "id": 2, "lon": 5.0, "lat": 8.0, "tags": {"highway": "stop"}},
        {"type": "node", "id": 3, "lon": 9.0, "lat": 9.0, "tags": {"shop": "bakery"}},
    ]
    osm, piscines = mobilier.lire_osm(elements, lambda lon, lat: (lon, lat), box(-30, -30, 30, 30))
    assert [e.nature for e in osm] == ["passage_pieton", "stop"] and not piscines
    route = [
        (
            LineString([(-30, 0), (30, 0)]),
            {"nature": "Route à 1 chaussée", "largeur_de_chaussee": 6},
        )
    ]
    chaussee = box(-30, -3, 30, 3)
    meubles, comptes = mobilier.construire(
        osm, [], route, chaussee, box(-30, -30, 30, 30), Polygon(), terrain_plat(), CFG
    )
    assert len(meubles["marquage"]) >= 4
    assert meubles["panneau_stop"] and comptes["stop"] == 1
    assert comptes["lampadaire_indicatif"] >= 1
