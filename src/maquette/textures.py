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
VOLETS = [(236, 236, 230), (146, 163, 148), (116, 138, 160), (112, 84, 62)]  # teintes des volets
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
    echelle: float = 1.0  # répétitions de la texture par unité d'UV (KHR_texture_transform)

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
    """`facade_enduit_<v>` (v : teinte), `facade_<pierre|meuliere|brique|bois>`,
    `facade_activite_<v>` (bardage), `facade_annexe_<v>`, `facade_projet`. Tuile de 3 m."""
    morceaux = nom.split("_")
    famille = morceaux[1]
    v = int(morceaux[2]) if len(morceaux) > 2 else 0
    rng = _rng(nom)
    if famille == "activite":
        c = revetement("bardage", rng, [(196, 198, 196), (205, 198, 182)][v])
    elif famille == "annexe":
        c = revetement("enduit", rng, [(206, 200, 188), (186, 176, 160)][v])
    elif famille == "projet":
        c = revetement("projet", rng, (241, 239, 233))
    else:
        teinte = {
            "enduit": ENDUITS[v],
            "pierre": (208, 194, 168),
            "meuliere": (186, 152, 112),
            "brique": (158, 82, 58),
            "bois": (148, 108, 72),
        }[famille]
        c = revetement(famille, rng, teinte)
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
    if nom in ("asphalte", "sol_enrobe"):
        return (
            simple("asphalte", dossier)
            if nom == "asphalte"
            else _renommer(simple("asphalte", dossier), nom)
        )
    if nom == "sol_pelouse":
        fin = rng.random((t, t))
        brins = ndimage.gaussian_filter(fin, (2.5, 0.6), mode="wrap")
        variation = fbm(rng, ((3, 0.4), (12, 0.3))) * 0.6 + brins * 0.4
        c = Couches.unie(_couleur((88, 124, 56), variation, 0.55), 0.95)
        c.hauteur = brins
        return _avec_textures(nom, c, dossier, force=5.0)
    if nom in ("sol_gravier", "sol_base"):
        yy, xx = np.mgrid[0:t, 0:t] / t * 2.0
        n = 5000
        graines = rng.random((n, 2)) * 2.0
        decalages = np.array([(ox, oy) for ox in (-2, 0, 2) for oy in (-2, 0, 2)])
        arbre = cKDTree((graines[None] + decalages[:, None]).reshape(-1, 2))
        dist, idx = arbre.query(np.column_stack([xx.ravel(), yy.ravel()]), k=2)
        bord = (dist[:, 1] - dist[:, 0]).reshape(t, t)
        teinte_cailloux = ((idx[:, 0] % n) % 11 / 11).reshape(t, t)
        base = (188, 174, 148) if nom == "sol_gravier" else (178, 170, 152)
        c = Couches.unie(_couleur(base, teinte_cailloux * 0.7 + fbm(rng) * 0.3, 0.4), 0.95)
        c.hauteur = np.clip(bord * 40, 0, 1)
        c.couleur *= (0.82 + 0.18 * c.hauteur)[..., None]
        return _avec_textures(nom, c, dossier, force=4.0)
    if nom == "sol_terre":
        mottes = fbm(rng, ((6, 0.3), (24, 0.4), (96, 0.3)))
        c = Couches.unie(_couleur((132, 104, 76), mottes, 0.5), 0.97)
        c.hauteur = mottes
        return _avec_textures(nom, c, dossier, force=4.0)
    if nom == "sol_dalles":
        yy, xx = np.mgrid[0:t, 0:t] / t * 2.0
        ids = (xx // 0.5).astype(int) * 7 + (yy // 0.5).astype(int) * 13
        variation = (ids % 9) / 9 * 0.5 + fbm(rng) * 0.5
        c = Couches.unie(_couleur((184, 178, 166), variation, 0.15), 0.85)
        joint = ((xx % 0.5) < 0.008) | ((yy % 0.5) < 0.008)
        c.peindre(joint, (150, 146, 138), -0.6, 0.95)
        return _avec_textures(nom, c, dossier)
    if nom == "carrelage_piscine":
        yy, xx = np.mgrid[0:t, 0:t] / t * 0.5
        c = Couches.unie(_couleur((80, 160, 190), bruit(rng, 16), 0.12), 0.3)
        joint = ((xx % 0.05) < 0.004) | ((yy % 0.05) < 0.004)
        c.peindre(joint, (220, 228, 230), -0.5, 0.6)
        return _avec_textures(nom, c, dossier)
    if nom == "massif":
        c = Couches.unie(
            _couleur((66, 96, 46), fbm(rng, ((8, 0.4), (32, 0.3), (128, 0.3))), 0.6), 0.9
        )
        c.hauteur = fbm(rng, ((32, 0.5), (128, 0.5)))
        return _avec_textures(nom, c, dossier, force=6.0)
    if nom.startswith("panneau_") and nom != "panneau_dos":
        return Materiau(nom, texture=_panneau(nom.split("_")[1], dossier), rugosite=0.5)
    if nom.startswith("carrosserie_"):
        teintes = [
            (235, 235, 232),
            (30, 32, 36),
            (150, 152, 156),
            (190, 192, 195),
            (40, 60, 100),
            (150, 30, 30),
        ]
        return Materiau(nom, couleur=srgb(teintes[int(nom.split("_")[1])]), rugosite=0.3, metal=0.5)
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
        "margelle": lambda: Materiau(nom, couleur=srgb((224, 216, 200)), rugosite=0.8),
        "eau_piscine": lambda: Materiau(
            nom, couleur=srgb((70, 170, 196)), rugosite=0.04, metal=0.15
        ),
        "chaperon": lambda: Materiau(nom, couleur=srgb((214, 208, 196)), rugosite=0.8),
        "grille": lambda: Materiau(nom, couleur=srgb((44, 50, 48)), rugosite=0.45, metal=0.5),
        "grillage": lambda: Materiau(
            nom, couleur=srgb((56, 86, 58), 0.32), rugosite=0.6, double_face=True
        ),
        "poteau_grillage": lambda: Materiau(nom, couleur=srgb((56, 82, 58)), rugosite=0.5),
        "metal_mobilier": lambda: Materiau(
            nom, couleur=srgb((62, 64, 68)), rugosite=0.45, metal=0.5
        ),
        "lampe": lambda: Materiau(nom, couleur=srgb((255, 244, 222)), rugosite=0.3, emission=1.5),
        "panneau_dos": lambda: Materiau(
            nom, couleur=srgb((150, 152, 154)), rugosite=0.5, metal=0.5
        ),
        "feu_boitier": lambda: Materiau(nom, couleur=srgb((30, 32, 34)), rugosite=0.5),
        "feu_rouge": lambda: Materiau(nom, couleur=srgb((222, 40, 30)), rugosite=0.3, emission=1.5),
        "feu_orange": lambda: Materiau(nom, couleur=srgb((120, 80, 20)), rugosite=0.3),
        "feu_vert": lambda: Materiau(nom, couleur=srgb((20, 90, 50)), rugosite=0.3),
        "bois_banc": lambda: Materiau(nom, couleur=srgb((140, 100, 66)), rugosite=0.8),
        "vitre_auto": lambda: Materiau(nom, couleur=srgb((30, 38, 46)), rugosite=0.06, metal=0.5),
        "pneu": lambda: Materiau(nom, couleur=srgb((26, 26, 26)), rugosite=0.9),
    }
    if nom not in fixes:
        raise KeyError(f"matériau inconnu : {nom}")
    return fixes[nom]()


def _renommer(m: Materiau, nom: str) -> Materiau:
    m.nom = nom
    return m


def _panneau(nature: str, dossier: Path) -> str:
    """Face d'un panneau (stop, cédez-le-passage, bus), dans sa boîte englobante."""
    from PIL import ImageDraw, ImageFont

    fichier = f"panneau_{nature}.png"
    t = 256
    image = Image.new("RGB", (t, t), (150, 152, 154))
    d = ImageDraw.Draw(image)
    if nature == "stop":
        a = np.radians(22.5) + np.arange(8) * np.pi / 4
        pts = [
            (
                t / 2 + t / 2 * np.cos(x) / np.cos(np.pi / 8),
                t / 2 - t / 2 * np.sin(x) / np.cos(np.pi / 8),
            )
            for x in a
        ]
        d.polygon(pts, fill=(255, 255, 255))
        interieur = [(t / 2 + (px - t / 2) * 0.9, t / 2 + (py - t / 2) * 0.9) for px, py in pts]
        d.polygon(interieur, fill=(200, 30, 36))
        police = ImageFont.load_default(size=64)
        d.text((t / 2, t / 2), "STOP", fill=(255, 255, 255), font=police, anchor="mm")
    elif nature == "cedez":
        d.polygon([(0, 0), (t, 0), (t / 2, t)], fill=(200, 30, 36))
        d.polygon(
            [(t * 0.16, t * 0.08), (t * 0.84, t * 0.08), (t / 2, t * 0.8)], fill=(255, 255, 255)
        )
    else:
        d.rectangle((0, 0, t, t), fill=(30, 80, 160))
        police = ImageFont.load_default(size=72)
        d.text((t / 2, t / 2), "BUS", fill=(255, 255, 255), font=police, anchor="mm")
    image.save(dossier / fichier)
    return fichier


Teinte = tuple[float, float, float]


def _cle_photo(
    nom: str, toits: dict[str, tuple[str, Teinte]], cfg: dict[str, Any]
) -> tuple[str, Teinte | None]:
    """Entrée de `config/materiaux.yaml` d'un matériau et la teinte qu'il vise."""
    if nom in toits:
        return f"couverture_{toits[nom][0]}", toits[nom][1]
    variante = 0
    cle = nom
    if nom.startswith(("facade_enduit_", "facade_activite_", "facade_annexe_")):
        morceaux = nom.split("_")
        cle, variante = "_".join(morceaux[:2]), int(morceaux[2])
    entree = cfg["materiaux"].get(cle)
    if entree is None or not entree.get("teintes"):
        return cle, None
    r, g, b = entree["teintes"][variante % len(entree["teintes"])]
    return cle, (r / 255, g / 255, b / 255)


def _photographie(
    nom: str, dossier: Path, toits: dict[str, tuple[str, Teinte]], bases: dict[str, Teinte]
) -> Materiau | None:
    """Matériau photographié (Poly Haven, CC0) si `config/materiaux.yaml` en prévoit un.

    La texture est recolorée une fois vers la teinte la plus claire de son entrée (`bases`) ;
    chaque matériau l'assombrit ensuite vers sa propre teinte par le facteur de couleur.
    """
    from maquette import bibliotheque

    cfg = bibliotheque.configuration()
    cle, teinte = _cle_photo(nom, toits, cfg)
    entree = cfg["materiaux"].get(cle)
    if entree is None:
        return None
    base = bases.get(cle)
    ph = bibliotheque.photo(entree["id"], dossier, cfg, base, entree.get("taille"))
    if ph is None:
        return None
    couleur = (1.0, 1.0, 1.0, 1.0)
    if teinte is not None and base is not None:
        # Rapport à la teinte de base visée (pas à la moyenne mesurée, faussée par l'écrêtage) :
        # le grain et la teinte de l'image recolorée sont conservés.
        r, g, b = (
            float(np.clip(t / max(m, 1e-3), 0, 1)) for t, m in zip(teinte, base, strict=True)
        )
        couleur = (r, g, b, 1.0)
    return Materiau(
        nom,
        couleur=couleur,
        texture=ph.couleur,
        orm=ph.arm,
        normale=ph.normale,
        echelle=float(entree["uv"]) / ph.taille,
        double_face=nom.startswith(("facade_", "mur_cloture")),
        lisse=nom.startswith("feuillage_"),
    )


def catalogue(
    noms: set[str],
    dossier: Path,
    toits: dict[str, tuple[str, tuple[float, float, float]]],
    ortho: str,
) -> dict[str, Materiau]:
    """Définitions des matériaux utilisés ; `toits` : nom → (type de couverture, teinte)."""
    resultat: dict[str, Materiau] = {}
    from maquette import bibliotheque

    cfg_photos = bibliotheque.configuration()
    bases: dict[str, Teinte] = {}
    for nom in noms:
        cle, teinte = _cle_photo(nom, toits, cfg_photos)
        if teinte is not None:
            ancienne = bases.get(cle, (0.0, 0.0, 0.0))
            bases[cle] = (
                max(ancienne[0], teinte[0]),
                max(ancienne[1], teinte[1]),
                max(ancienne[2], teinte[2]),
            )
    for nom in sorted(noms):
        photographie = _photographie(nom, dossier, toits, bases)
        if photographie is not None:
            resultat[nom] = photographie
        elif nom == "ortho":
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
