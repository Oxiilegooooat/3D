"""Géométrie sur données synthétiques (aucun accès réseau)."""

from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient

from maquette import batiments as bt
from maquette import vegetation, voirie
from maquette.config import charger_config, charger_site
from maquette.geoplateforme import Grille
from maquette.maillage import (
    boite,
    decouper_bande,
    densifier,
    quad_mur,
    trianguler_anneaux,
)
from maquette.terrain import Terrain
from maquette.toits import ajuster, plis_dans

CFG = charger_config()


def normales(sommets: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = sommets[faces[:, 0]], sommets[faces[:, 1]], sommets[faces[:, 2]]
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def terrain_plat(z: float = 0.0, demi: float = 60.0) -> Terrain:
    n = int(2 * demi / 2) + 1
    return Terrain(-demi, -demi, 2.0, np.full((n, n), z))


def grille(fonction, demi: float = 60.0, res: float = 0.5) -> Grille:
    xs = -demi + (np.arange(int(2 * demi / res)) + 0.5) * res
    ys = demi - (np.arange(int(2 * demi / res)) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    return Grille((-demi, -demi, demi, demi), res, fonction(gx, gy).astype(np.float32))


def test_idu_des_parcelles(tmp_path: Path) -> None:
    fiche = tmp_path / "site.yaml"
    fiche.write_text("nom: Test\ninsee: '78643'\nparcelles: [AE0003, AO 406, b12]\n")
    assert charger_site(fiche).idus == ["78643000AE0003", "78643000AO0406", "786430000B0012"]


def test_presentation_projet_sans_projet_refusee(tmp_path: Path) -> None:
    fiche = tmp_path / "site.yaml"
    fiche.write_text("nom: Test\ninsee: '78643'\nparcelles: [AE3]\npresentation: projet\n")
    with pytest.raises(ValueError):
        charger_site(fiche)


def test_terrain_interpole_comme_son_maillage() -> None:
    rng = np.random.default_rng(1)
    t = Terrain(0.0, 0.0, 2.0, rng.random((6, 6)) * 5)
    m = t.maillage()
    np.testing.assert_allclose(t.hauteur(m.sommets[:, :2]), m.sommets[:, 2], atol=1e-4)
    # Au centre de chaque triangle, la hauteur est la moyenne de ses sommets (interpolation plane).
    centres = m.sommets[m.faces].mean(axis=1)
    np.testing.assert_allclose(t.hauteur(centres[:, :2]), centres[:, 2], atol=1e-4)
    assert (normales(m.sommets, m.faces)[:, 2] > 0).all()


def test_densifier_insere_les_contacts() -> None:
    carre = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    pts, aretes, abscisses = densifier(carre, 3.0, np.array([[4.2, 0.0]]))
    assert any(np.allclose(p, [4.2, 0]) for p in pts)
    assert aretes[0] == 0 and abscisses[0] == 0
    assert (np.diff(abscisses[aretes == 0]) > 0).all()


def test_triangulation_respecte_un_faitage() -> None:
    carre = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=float)
    pts, _, _ = densifier(carre, 20.0, np.array([[0.0, 3.0], [10.0, 3.0]]))
    xy, faces = trianguler_anneaux([pts], None, [np.array([[0.0, 3.0], [10.0, 3.0]])])
    aretes = {tuple(sorted(e)) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    i = int(np.argmin(np.hypot(*(xy - [0, 3]).T)))
    j = int(np.argmin(np.hypot(*(xy - [10, 3]).T)))
    assert tuple(sorted((i, j))) in aretes


def test_mur_regarde_a_droite_et_bande_coupee() -> None:
    p0, p1 = np.array([0.0, 0.0]), np.array([4.0, 0.0])
    poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 6.0), (0.0, 6.0)]
    m = quad_mur(p0, p1, poly, lambda s, z: (s, z))
    # Ouest → est : l'extérieur (à droite) est au sud.
    assert np.allclose(normales(m.sommets, m.faces), [0, -1, 0])
    bande = decouper_bande(poly, 2.0, 3.0)
    assert min(z for _, z in bande) == 2.0 and max(z for _, z in bande) == 3.0


def test_boite_fermee_vers_l_exterieur() -> None:
    m = boite(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0]), 2.0, 1.0, 1.0)
    centre = np.array([0, 0, 0.5])
    milieux = m.sommets[m.faces].mean(axis=1)
    assert (np.einsum("ij,ij->i", normales(m.sommets, m.faces), milieux - centre) > 0).all()


@pytest.mark.parametrize(
    ("nature", "fonction"),
    [
        ("plat", lambda u, v: 9.0 + 0 * u),
        ("deux_pans", lambda u, v: 10.0 - 0.7 * np.abs(v - 0.5)),
        ("croupe", lambda u, v: 10.0 - 0.8 * np.maximum(np.abs(v), np.abs(u) - 4.0)),
    ],
)
def test_ajustement_des_toits(nature: str, fonction) -> None:
    rng = np.random.default_rng(3)
    emprise = box(-8, -4, 8, 4)
    x, y = rng.uniform(-7.5, 7.5, 1500), rng.uniform(-3.5, 3.5, 1500)
    z = fonction(x, y) + rng.normal(0, 0.06, len(x))
    z[:40] += 4.0  # branches en surplomb
    modele = ajuster(emprise, x, y, z)
    assert modele is not None and modele.nature == nature
    assert abs(float(modele.z(np.array([[1.0, 1.0]]))[0]) - fonction(1.0, 1.0)) < 0.15


def test_plis_touchent_le_contour() -> None:
    rng = np.random.default_rng(4)
    x, y = rng.uniform(-7.5, 7.5, 800), rng.uniform(-3.5, 3.5, 800)
    modele = ajuster(box(-8, -4, 8, 4), x, y, 10.0 - 0.7 * np.abs(y))
    lignes, contacts = plis_dans(modele, box(-8, -4, 8, 4))
    assert len(lignes) == 1 and len(contacts) == 2
    assert all(abs(abs(cx) - 8) < 1e-6 for cx, _ in contacts)


def batiment_pignon(cle: str = "B1", detail: bool = True) -> tuple[bt.Batiment, Terrain]:
    emprise = orient(box(-8, -4, 8, 4), sign=1.0)
    mns = grille(
        lambda x, y: np.where((np.abs(x) < 8) & (np.abs(y) < 4), 9.0 - 0.75 * np.abs(y), 0.0)
    )
    terrain = terrain_plat()
    b = bt.Batiment(
        cle,
        emprise,
        {"usage_1": "Résidentiel", "nombre_d_etages": 2, "materiaux_des_murs": "10"},
        detail=detail,
    )
    bt.mesurer(b, mns, terrain, CFG)
    bt.classer(b, CFG)
    return b, terrain


def test_mesure_d_un_pignon() -> None:
    b, _ = batiment_pignon()
    assert b.modele is not None and b.modele.nature == "deux_pans"
    assert b.hauteur_source == "lidar"
    assert 5.5 < b.h_gouttiere < 7.0
    assert b.style == "maison" and b.revetement == "pierre" and b.etages == 2


def test_facade_detaillee_percee() -> None:
    b, terrain = batiment_pignon()
    lignes, contacts = bt.plis(b)
    anneaux = bt.anneaux_batiment(b.emprise, 0.5, contacts)
    mitoyens = [np.zeros(len(a.origines), dtype=bool) for a in anneaux]
    murs = bt.murs_detail(b, anneaux, mitoyens, terrain, {(0, 0)}, CFG)
    assert len(murs["verre"]) >= 8  # deux étages de fenêtres sur les longs pans
    assert any(k.startswith("porte_") for k in murs)
    assert any(k.startswith("volet_") for k in murs)
    dessus, dessous = bt.toit(b, anneaux, 0.5, terrain, lignes, 0.4, 0.2)
    assert len(dessus) and len(dessous)
    # Le débord descend sous l'égout.
    assert dessus.sommets[:, 2].min() < b.z_sol + b.h_gouttiere


def test_mur_mitoyen_aveugle() -> None:
    a = orient(box(0, 0, 8, 10), sign=1.0)
    voisin = orient(box(8, 0, 16, 10), sign=1.0)
    arbre = shapely.STRtree([a, voisin])
    anneau = bt.anneaux_batiment(a, 1.0)[0]
    mitoyen = bt.mitoyennete(a, anneau, arbre, [a, voisin], 0)
    assert mitoyen.sum() == 1


def test_voirie_reste_dans_le_domaine_public() -> None:
    fenetre = box(-50, -50, 50, 50)
    parcelles = shapely.union_all([box(-50, 6, 50, 50), box(-50, -50, 50, -6)])
    rue = [
        (
            LineString([(-60, 0), (60, 0)]),
            {"nature": "Route à 1 chaussée", "largeur_de_chaussee": 6, "nombre_de_voies": 2},
        )
    ]
    sortie, chaussee = voirie.voirie(rue, parcelles, Polygon(), fenetre, terrain_plat(), CFG)
    assert sortie["asphalte"] and sortie["trottoir"] and sortie["marquage"]
    trottoirs = np.concatenate([m.sommets for m in sortie["trottoir"]])
    assert (np.abs(trottoirs[:, 1]) <= 6.01).all()
    assert abs(chaussee.area - 6 * 99.9) < 15


def test_un_arbre_detecte() -> None:
    sol = grille(lambda x, y: 0 * x)
    canopee = grille(lambda x, y: 8.0 * np.exp(-((x - 10) ** 2 + (y - 5) ** 2) / 8.0))
    arbres = vegetation.detecter(canopee, sol, Polygon(), terrain_plat(), CFG)
    assert len(arbres) == 1
    assert abs(arbres[0].x - 10) < 0.6
    assert abs(arbres[0].y - 5) < 0.6
    assert 7.0 < arbres[0].hauteur <= 8.0
