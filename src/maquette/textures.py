"""Matériaux et textures procédurales (numpy) : enduits, pierre, meulière, brique, bois,
bardage, tuiles, ardoise, bac acier, fenêtres, volets, portes, asphalte, trottoirs.

Chaque matériau à texture produit trois images : couleur (sRGB), `orm` (vert = rugosité,
bleu = métal, convention glTF) et normales. Les textures sont raccordables (bords qui se
prolongent) : elles se répètent sans couture.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

Img = NDArray[np.float64]
TAILLE = 512

ENDUITS = [(232, 222, 200), (222, 205, 178), (238, 233, 223), (214, 194, 165)]
VOLETS = [(236, 236, 230), (146, 163, 148), (116, 138, 160), (112, 84, 62)]
PORTES = [(118, 84, 56), (52, 78, 62), (62, 64, 68)]
FEUILLAGES = [(64, 92, 44), (82, 110, 52), (58, 84, 48)]


@dataclass
class Materiau:
    nom: str
    couleur: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # facteur sRGB
    texture: str | None = None
    orm: str | None = None
    normale: str | None = None
    rugosite: float = 0.85
    metal: float = 0.0
    double_face: bool = False
    emission: float = 0.0
    lisse: bool = False

    def en_dict(self) -> dict[str, Any]:
        return asdict(self)


def srgb(rgb: tuple[int, int, int], alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)


def _rng(nom: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(nom.encode()).hexdigest()[:8], 16))


def bruit(rng: np.random.Generator, cellules: int, taille: int = TAILLE) -> Img:
    """Bruit lisse raccordable dans [0, 1]."""
    petit = rng.random((cellules, cellules))
    grand = ndimage.zoom(petit, taille / cellules, order=3, mode="grid-wrap")[:taille, :taille]
    grand -= grand.min()
    return np.asarray(grand / max(float(grand.max()), 1e-9))


def fbm(
    rng: np.random.Generator,
    octaves: tuple[tuple[int, float], ...] = ((4, 0.5), (16, 0.3), (64, 0.2)),
    taille: int = TAILLE,
) -> Img:
    return np.asarray(sum(p * bruit(rng, c, taille) for c, p in octaves))


def _couleur(rgb: tuple[int, int, int], variation: Img, ampleur: float) -> Img:
    base = np.array(rgb, dtype=np.float64)[None, None, :] / 255
    return np.clip(base * (1 + ampleur * (variation[..., None] - 0.5)), 0, 1)


def normales(hauteur: Img, force: float) -> Img:
    gy, gx = np.gradient(hauteur)
    n = np.dstack([-gx * force, gy * force, np.ones_like(hauteur)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n * 0.5 + 0.5


@dataclass
class Couches:
    couleur: Img
    hauteur: Img
    rugosite: Img
    metal: Img

    @classmethod
    def unie(cls, couleur: Img, rugosite: float) -> Couches:
        h, w = couleur.shape[:2]
        return cls(couleur, np.zeros((h, w)), np.full((h, w), rugosite), np.zeros((h, w)))

    def peindre(
        self,
        masque: NDArray[np.bool_],
        couleur: Img | tuple[int, int, int],
        hauteur: float | Img,
        rugosite: float,
        metal: float = 0.0,
    ) -> None:
        if isinstance(couleur, tuple):
            couleur = np.broadcast_to(np.array(couleur, dtype=np.float64) / 255, self.couleur.shape)
        self.couleur[masque] = couleur[masque]
        self.hauteur[masque] = hauteur[masque] if isinstance(hauteur, np.ndarray) else hauteur
        self.rugosite[masque] = rugosite
        self.metal[masque] = metal


# --------------------------------------------------------------------------- revêtements


def revetement(nom: str, rng: np.random.Generator, teinte: tuple[int, int, int]) -> Couches:
    t = TAILLE
    yy, xx = np.mgrid[0:t, 0:t] / t * 3.0  # mètres (une tuile = 3 m)
    grain = fbm(rng)
    if nom == "brique":
        rangs = (yy / 0.075).astype(int)
        decal = (rangs % 2) * 0.12
        colonnes = ((xx + decal) / 0.24).astype(int)
        joint = ((yy % 0.075) < 0.012) | (((xx + decal) % 0.24) < 0.012)
        ids = (rangs * 131 + colonnes * 17) % 97
        variation = (ids / 97.0) * 0.6 + grain * 0.4
        c = Couches.unie(_couleur(teinte, variation, 0.35), 0.85)
        c.hauteur = np.where(joint, -0.6, 0.0) + grain * 0.2
        c.peindre(joint, (196, 188, 172), -0.6, 0.95)
        return c
    if nom in ("pierre", "meuliere"):
        # Moellons irréguliers : cellules de Voronoï sur graines répétées (raccordable).
        n = 70 if nom == "pierre" else 140
        etirement = 1.6 if nom == "pierre" else 1.0
        graines = rng.random((n, 2)) * 3.0
        decalages = np.array([(ox, oy) for ox in (-3, 0, 3) for oy in (-3, 0, 3)])
        toutes = (graines[None, :, :] + decalages[:, None, :]).reshape(-1, 2)
        arbre = cKDTree(toutes * [1.0, etirement])
        dist, idx = arbre.query(np.column_stack([xx.ravel(), yy.ravel() * etirement]), k=2)
        d = dist[:, 0].reshape(t, t)
        d2 = dist[:, 1].reshape(t, t)
        ident = (idx[:, 0] % n).reshape(t, t)
        joint = (d2 - d) < (0.025 if nom == "pierre" else 0.035)
        variation = (ident % 13) / 13 * 0.5 + grain * 0.5
        c = Couches.unie(_couleur(teinte, variation, 0.3 if nom == "meuliere" else 0.22), 0.9)
        c.hauteur = np.clip((d2 - d) * 6, 0, 1) * 0.6 + grain * 0.3
        c.peindre(joint, (212, 204, 188), 0.0, 0.95)
        return c
    if nom == "bois":
        planches = (xx / 0.18).astype(int)
        joint = (xx % 0.18) < 0.012
        variation = (
            (planches * 37 % 11) / 11 * 0.5
            + bruit(rng, 8) * 0.2
            + ndimage.zoom(rng.random((t // 4, 4)), (4, t / 4), order=1)[:t, :t] * 0.3
        )
        c = Couches.unie(_couleur(teinte, variation, 0.4), 0.8)
        c.hauteur = np.where(joint, -0.8, 0.0)
        c.peindre(joint, (60, 45, 32), -0.8, 0.9)
        return c
    if nom == "bardage":
        onde = (xx % 0.25) / 0.25
        relief = np.clip(np.abs(onde - 0.5) * 4 - 0.6, 0, 1)
        c = Couches.unie(_couleur(teinte, grain * 0.5 + relief * 0.5, 0.18), 0.55)
        c.hauteur = relief
        c.metal[:] = 0.35
        return c
    if nom == "projet":
        c = Couches.unie(_couleur(teinte, bruit(rng, 6), 0.04), 0.75)
        joints = (yy % 3.0) < 0.012
        c.peindre(joints, (205, 203, 198), -0.3, 0.8)
        return c
    # Enduit gratté : grain fin et marbrures légères.
    fin = rng.random((t, t))
    c = Couches.unie(_couleur(teinte, grain * 0.7 + fin * 0.3, 0.12), 0.92)
    c.hauteur = ndimage.gaussian_filter(fin, 0.8) * 0.5 + grain * 0.2
    return c


def _fenetre(
    c: Couches,
    rng: np.random.Generator,
    style: str,
    volet: tuple[int, int, int] | None,
    largeur_m: float = 3.0,
    hauteur_m: float = 3.0,
) -> None:
    """Peint une fenêtre centrée (et ses volets ou son coffre) sur une tuile d'une travée."""
    t = TAILLE
    yy, xx = np.mgrid[0:t, 0:t]
    m_x = t / largeur_m
    m_y = t / hauteur_m
    if style == "activite":
        x0, x1 = t * 0.12, t * 0.88
        z0, z1 = hauteur_m - 1.6, hauteur_m - 0.6
    else:
        x0, x1 = t / 2 - 0.6 * m_x, t / 2 + 0.6 * m_x
        z0, z1 = 0.95, 2.4
    y0, y1 = t - z1 * m_y, t - z0 * m_y  # lignes (0 en haut)
    ouverture = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    e = 0.06 * m_x
    reflet = np.clip((yy - y0) / max(y1 - y0, 1), 0, 1)
    ciel = np.dstack(
        [0.28 + 0.25 * (1 - reflet), 0.34 + 0.25 * (1 - reflet), 0.42 + 0.22 * (1 - reflet)]
    )
    ciel *= (0.9 + 0.2 * bruit(rng, 4))[..., None]
    c.peindre(ouverture, ciel, -1.0, 0.08, 0.4)
    cadre = ouverture & ((xx < x0 + e) | (xx >= x1 - e) | (yy < y0 + e) | (yy >= y1 - e))
    meneaux = ouverture & (np.abs(xx - (x0 + x1) / 2) < e / 2)
    if style == "activite":
        for k in range(1, 4):
            meneaux |= ouverture & (np.abs(xx - (x0 + (x1 - x0) * k / 4)) < e / 2)
    couleur_cadre = (60, 62, 66) if style == "activite" else (238, 238, 234)
    c.peindre(cadre | meneaux, couleur_cadre, -0.4, 0.45)
    if style != "activite":
        appui = (
            (xx >= x0 - 0.05 * m_x) & (xx < x1 + 0.05 * m_x) & (yy >= y1) & (yy < y1 + 0.06 * m_y)
        )
        c.peindre(appui, (222, 216, 204), 0.6, 0.7)
        # Ombre portée de l'appui et du linteau.
        ombre = (xx >= x0) & (xx < x1) & (yy >= y1 + 0.06 * m_y) & (yy < y1 + 0.12 * m_y)
        c.couleur[ombre] *= 0.85
    if volet is not None:
        for cote in (-1, 1):
            vx0 = x0 - 0.62 * m_x if cote < 0 else x1 + 0.02 * m_x
            vx1 = vx0 + 0.6 * m_x
            battant = (xx >= vx0) & (xx < vx1) & (yy >= y0) & (yy < y1)
            lames = ((yy - y0) % (0.07 * m_y)) < 0.018 * m_y
            teinte = np.broadcast_to(
                np.array(volet, dtype=np.float64) / 255, c.couleur.shape
            ).copy()
            teinte[lames] *= 0.78
            c.peindre(battant, teinte, np.where(lames, 0.3, 0.55), 0.6)
    elif style == "collectif":
        coffre = (xx >= x0) & (xx < x1) & (yy >= y0 - 0.18 * m_y) & (yy < y0)
        c.peindre(coffre, (225, 225, 222), 0.4, 0.5)


def _texture_couches(
    c: Couches, dossier: Path, nom: str, force_normale: float = 3.0
) -> tuple[str, str, str]:
    """Couleur en 512 px ; rugosité/métal et normales en 256 px (JPEG : maquette légère)."""
    dossier.mkdir(parents=True, exist_ok=True)
    couleur = Image.fromarray((np.clip(c.couleur, 0, 1) * 255).astype(np.uint8))
    couleur.save(dossier / f"{nom}_couleur.jpg", quality=85)
    orm = np.dstack([np.ones_like(c.rugosite), c.rugosite, c.metal])
    image_orm = Image.fromarray((np.clip(orm, 0, 1) * 255).astype(np.uint8))
    image_orm.resize((TAILLE // 2, TAILLE // 2), Image.Resampling.BILINEAR).save(
        dossier / f"{nom}_orm.jpg", quality=90
    )
    n = normales(ndimage.gaussian_filter(c.hauteur, 1.0), force_normale)
    image_n = Image.fromarray((n * 255).astype(np.uint8))
    image_n.resize((TAILLE // 2, TAILLE // 2), Image.Resampling.BILINEAR).save(
        dossier / f"{nom}_normale.jpg", quality=90
    )
    return f"{nom}_couleur.jpg", f"{nom}_orm.jpg", f"{nom}_normale.jpg"


def _avec_textures(nom: str, c: Couches, dossier: Path, **kw: Any) -> Materiau:
    tex, orm, nor = _texture_couches(c, dossier, nom, kw.pop("force", 3.0))
    return Materiau(nom, texture=tex, orm=orm, normale=nor, **kw)


# --------------------------------------------------------------------------- catalogue


def facade(nom: str, dossier: Path) -> Materiau:
    """`facade_<style>_<revetement>[_<variante>]_<fenetres|pleine>` (variante : teinte de
    l'enduit, couleur des volets), `facade_activite_<v>_…`, `facade_annexe_<v>_pleine`,
    `facade_projet_pleine`."""
    morceaux = nom.split("_")
    bande = morceaux[-1]
    style = morceaux[1]
    rng = _rng(nom)
    if style == "activite":
        v = int(morceaux[2])
        c = revetement("bardage", rng, [(196, 198, 196), (205, 198, 182)][v])
        if bande == "fenetres":
            _fenetre(c, rng, "activite", None, 5.1, 5.0)
        return _avec_textures(nom, c, dossier, double_face=True)
    if style == "annexe":
        v = int(morceaux[2])
        return _avec_textures(
            nom,
            revetement("enduit", rng, [(206, 200, 188), (186, 176, 160)][v]),
            dossier,
            double_face=True,
        )
    if style == "projet":
        return _avec_textures(
            nom, revetement("projet", rng, (241, 239, 233)), dossier, double_face=True
        )
    rev = morceaux[2]
    v = int(morceaux[3]) if len(morceaux) == 5 else 0
    teinte = {
        "enduit": ENDUITS[v],
        "pierre": (208, 194, 168),
        "meuliere": (186, 152, 112),
        "brique": (158, 82, 58),
        "bois": (148, 108, 72),
    }[rev]
    c = revetement(rev, rng, teinte)
    if bande == "fenetres":
        _fenetre(c, rng, style, VOLETS[v] if style == "maison" else None)
    return _avec_textures(nom, c, dossier, double_face=True)


def toiture(type_toit: str, dossier: Path) -> tuple[str, str, str]:
    """Textures de couverture neutres (moyenne ≈ 0,8), teintées par la couleur de l'ortho.
    Une tuile de texture = 2 m ; u suit le faîtage, v la pente."""
    nom = f"couverture_{type_toit}"
    if (dossier / f"{nom}_couleur.jpg").exists():
        return f"{nom}_couleur.jpg", f"{nom}_orm.jpg", f"{nom}_normale.jpg"
    rng = _rng(nom)
    t = TAILLE
    yy, xx = np.mgrid[0:t, 0:t] / t * 2.0
    grain = fbm(rng)
    gris = (204, 204, 204)
    if type_toit == "tuiles":
        rangs = (yy / 0.25).astype(int)
        dans_rang = (yy % 0.25) / 0.25
        onde = 0.5 + 0.5 * np.cos(((xx + (rangs % 2) * 0.1) / 0.2) * 2 * np.pi)
        hauteur = dans_rang * 0.6 + onde * 0.4
        variation = ((rangs * 7 + (xx / 0.2).astype(int) * 13) % 17) / 17 * 0.4 + grain * 0.6
        c = Couches.unie(_couleur(gris, variation, 0.3), 0.8)
        c.couleur *= (0.75 + 0.25 * dans_rang)[..., None]
        c.hauteur = hauteur
    elif type_toit == "ardoise":
        rangs = (yy / 0.12).astype(int)
        decal = (rangs % 2) * 0.1
        joint = ((yy % 0.12) < 0.01) | (((xx + decal) % 0.2) < 0.008)
        variation = (
            (rangs * 11 + ((xx + decal) / 0.2).astype(int) * 5) % 13
        ) / 13 * 0.5 + grain * 0.5
        c = Couches.unie(_couleur(gris, variation, 0.25), 0.6)
        c.hauteur = (yy % 0.12) / 0.12 * 0.5
        c.peindre(joint, (120, 120, 120), -0.3, 0.7)
    elif type_toit == "bac_acier":
        onde = (xx % 0.333) / 0.333
        nervure = np.abs(onde - 0.5) < 0.08
        c = Couches.unie(_couleur(gris, grain, 0.15), 0.5)
        c.hauteur = np.where(nervure, 1.0, 0.0)
        c.metal[:] = 0.5
        c.couleur[nervure] *= 1.08
    else:  # terrasse : gravillons
        fin = rng.random((t, t))
        c = Couches.unie(_couleur(gris, fin * 0.6 + grain * 0.4, 0.35), 0.95)
        c.hauteur = ndimage.gaussian_filter(fin, 1.2)
    return _texture_couches(c, dossier, nom, 4.0)


def simple(nom: str, dossier: Path) -> Materiau:
    """Matériaux hors façades et toitures."""
    rng = _rng(nom)
    t = TAILLE
    if nom == "asphalte":
        fin = rng.random((t, t))
        c = Couches.unie(_couleur((74, 76, 80), fin * 0.5 + fbm(rng) * 0.5, 0.35), 0.9)
        granulats = fin > 0.985
        c.peindre(granulats, (130, 130, 128), 0.5, 0.7)
        c.hauteur = ndimage.gaussian_filter(fin, 0.7)
        return _avec_textures(nom, c, dossier, force=2.0)
    if nom == "trottoir":
        yy, xx = np.mgrid[0:t, 0:t] / t * 2.0
        c = Couches.unie(_couleur((178, 175, 168), fbm(rng), 0.18), 0.9)
        joint = ((xx % 1.0) < 0.012) | ((yy % 1.0) < 0.012)
        c.peindre(joint, (140, 138, 132), -0.5, 0.95)
        return _avec_textures(nom, c, dossier)
    if nom == "chemin":
        fin = rng.random((t, t))
        c = Couches.unie(_couleur((186, 170, 140), fin * 0.6 + fbm(rng) * 0.4, 0.3), 0.95)
        c.hauteur = ndimage.gaussian_filter(fin, 1.0)
        return _avec_textures(nom, c, dossier)
    if nom == "porte_sectionnelle":
        yy, xx = np.mgrid[0:t, 0:t] / t
        c = Couches.unie(_couleur((214, 216, 214), bruit(rng, 4), 0.06), 0.5)
        panneaux = (yy % 0.125) < 0.008
        rainures = (yy % 0.125 > 0.06) & (yy % 0.125 < 0.066)
        c.peindre(panneaux | rainures, (150, 152, 150), -0.5, 0.5)
        c.metal[:] = 0.3
        return _avec_textures(nom, c, dossier)
    if nom.startswith("porte_"):
        v = int(nom.split("_")[1])
        yy, xx = np.mgrid[0:t, 0:t] / t
        c = Couches.unie(_couleur(PORTES[v], bruit(rng, 6), 0.1), 0.6)
        moulures = (np.abs(xx - 0.5) > 0.35) | (np.abs(yy - 0.3) < 0.01) | (np.abs(yy - 0.7) < 0.01)
        c.hauteur = np.where(moulures, 0.4, 0.0)
        poignee = (np.abs(xx - 0.85) < 0.02) & (np.abs(yy - 0.5) < 0.04)
        c.peindre(poignee, (190, 180, 150), 1.0, 0.3, 0.9)
        return _avec_textures(nom, c, dossier)
    if nom == "mur_cloture":
        return _avec_textures(
            nom, revetement("pierre", rng, (198, 188, 168)), dossier, double_face=True
        )
    if nom.startswith("volet_"):
        return Materiau(nom, couleur=srgb(VOLETS[int(nom.split("_")[1])]), rugosite=0.6)
    if nom.startswith("feuillage_"):
        c = Couches.unie(
            _couleur(
                FEUILLAGES[int(nom.split("_")[1])], fbm(rng, ((8, 0.4), (32, 0.3), (128, 0.3))), 0.6
            ),
            0.9,
        )
        c.hauteur = fbm(rng, ((32, 0.5), (128, 0.5)))
        return _avec_textures(nom, c, dossier, lisse=True, force=6.0)
    fixes: dict[str, Callable[[], Materiau]] = {
        "socle": lambda: Materiau(nom, couleur=srgb((237, 230, 214)), rugosite=0.95),
        "bordure": lambda: Materiau(nom, couleur=srgb((152, 152, 150)), rugosite=0.8),
        "marquage": lambda: Materiau(nom, couleur=srgb((238, 238, 232)), rugosite=0.6),
        "eau": lambda: Materiau(nom, couleur=srgb((58, 88, 108)), rugosite=0.05, metal=0.3),
        "tronc": lambda: Materiau(nom, couleur=srgb((92, 72, 56)), rugosite=0.95, lisse=True),
        "limite_site": lambda: Materiau(
            nom, couleur=srgb((55, 125, 160)), rugosite=0.5, emission=0.6
        ),
        "voile_site": lambda: Materiau(
            nom, couleur=srgb((55, 125, 160), 0.16), rugosite=0.4, double_face=True
        ),
        "verre": lambda: Materiau(nom, couleur=srgb((44, 58, 72)), rugosite=0.05, metal=0.55),
        "verre_garde_corps": lambda: Materiau(
            nom, couleur=srgb((190, 210, 220), 0.35), rugosite=0.05, double_face=True
        ),
        "menuiserie": lambda: Materiau(nom, couleur=srgb((236, 236, 232)), rugosite=0.4),
        "menuiserie_projet": lambda: Materiau(nom, couleur=srgb((56, 59, 63)), rugosite=0.4),
        "appui": lambda: Materiau(nom, couleur=srgb((222, 216, 204)), rugosite=0.75),
        "beton_clair": lambda: Materiau(nom, couleur=srgb((204, 202, 196)), rugosite=0.85),
        "sous_toiture": lambda: Materiau(nom, couleur=srgb((226, 220, 208)), rugosite=0.85),
        "garde_corps": lambda: Materiau(nom, couleur=srgb((52, 54, 58)), rugosite=0.4, metal=0.6),
        "projet_volume": lambda: Materiau(nom, couleur=srgb((241, 239, 233)), rugosite=0.8),
    }
    if nom not in fixes:
        raise KeyError(f"matériau inconnu : {nom}")
    return fixes[nom]()


def catalogue(
    noms: set[str],
    dossier: Path,
    toits: dict[str, tuple[str, tuple[float, float, float]]],
    ortho: str,
) -> dict[str, Materiau]:
    """Définitions des matériaux utilisés ; `toits` : nom → (type de couverture, teinte)."""
    resultat: dict[str, Materiau] = {}
    for nom in sorted(noms):
        if nom == "ortho":
            resultat[nom] = Materiau(nom, texture=ortho, rugosite=0.9)
        elif nom.startswith("facade_"):
            resultat[nom] = facade(nom, dossier)
        elif nom in toits:
            type_toit, teinte = toits[nom]
            tex, orm, nor = toiture(type_toit, dossier)
            r, g, b = (float(np.clip(v / 0.8, 0, 1)) for v in teinte)
            resultat[nom] = Materiau(nom, couleur=(r, g, b, 1.0), texture=tex, orm=orm, normale=nor)
        else:
            resultat[nom] = simple(nom, dossier)
    return resultat
