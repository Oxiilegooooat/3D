"""Matériaux photographiés libres de droits (Poly Haven, licence CC0).

Pour chaque matériau de `config/materiaux.yaml` : couleur, normales et `arm` (occlusion,
rugosité, métal : convention glTF) en 1k, téléchargés une fois dans `outils/textures/`, réduits
pour la maquette et posés à leur taille réelle (`dimensions` de Poly Haven). Une teinte visée
recolore la texture en gardant son grain. Sans réseau, rien n'est renvoyé : la texture
procédurale prend le relais.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml
from PIL import Image

from maquette.config import RACINE

API = "https://api.polyhaven.com"
CACHE = RACINE / "outils" / "textures"
CARTES = {"couleur": "Diffuse", "arm": "arm", "normale": "nor_gl"}


@dataclass(frozen=True)
class Photo:
    couleur: str
    arm: str
    normale: str
    taille: float  # m, côté de la texture
    moyenne: tuple[float, float, float]  # couleur moyenne (sRGB, 0-1)


def configuration() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(
        (RACINE / "config" / "materiaux.yaml").read_text(encoding="utf-8")
    )
    return cfg


def _telecharger(http: httpx.Client, url: str, fichier: Path) -> Path:
    if not fichier.exists():
        fichier.parent.mkdir(parents=True, exist_ok=True)
        rep = http.get(url)
        rep.raise_for_status()
        fichier.write_bytes(rep.content)
    return fichier


def _source(http: httpx.Client, ident: str) -> tuple[dict[str, Path], float]:
    """Cartes 1k en cache et taille réelle (m) d'une texture Poly Haven."""
    dossier = CACHE / ident
    info_f = dossier / "info.json"
    if not info_f.exists():
        dossier.mkdir(parents=True, exist_ok=True)
        info = http.get(f"{API}/info/{ident}").raise_for_status().json()
        fichiers = http.get(f"{API}/files/{ident}").raise_for_status().json()
        info_f.write_text(json.dumps({"info": info, "fichiers": fichiers}))
    donnees = json.loads(info_f.read_text())
    taille = float(donnees["info"]["dimensions"][0]) / 1000
    chemins = {}
    for nom, carte in CARTES.items():
        url = donnees["fichiers"][carte]["1k"]["jpg"]["url"]
        chemins[nom] = _telecharger(http, url, dossier / f"{nom}.jpg")
    return chemins, taille


def coloriser(image: Image.Image, teinte: tuple[float, float, float]) -> Image.Image:
    """Impose une couleur moyenne en gardant le grain (luminance) de la texture.

    Un simple gain par canal ne corrige pas une teinte marquée (tôle rouge, feuilles mortes) :
    80 % de colorisation par la luminance, 20 % de la couleur d'origine ramenée à la teinte.
    """
    pixels = np.asarray(image, dtype=np.float64) / 255
    cible = np.asarray(teinte, dtype=np.float64)
    luminance = pixels @ np.array([0.299, 0.587, 0.114])
    relief = luminance / max(float(luminance.mean()), 1e-3)
    colorise = relief[..., None] * cible
    moyenne = pixels.reshape(-1, 3).mean(axis=0)
    origine = pixels * (cible / np.maximum(moyenne, 1e-3))
    resultat = np.clip(0.8 * colorise + 0.2 * origine, 0, 1)
    return Image.fromarray((resultat * 255).astype(np.uint8))


_PHOTOS: dict[tuple[str, str, tuple[float, ...] | None, int], Photo] = {}


def photo(
    ident: str,
    dossier: Path,
    cfg: dict[str, Any],
    teinte: tuple[float, float, float] | None = None,
    pixels_max: int | None = None,
) -> Photo | None:
    """Cartes réduites dans `dossier` (textures de la scène), ou None sans réseau.

    Avec `teinte` (sRGB 0-1), la couleur est recolorée pour que sa moyenne y arrive (le
    facteur de couleur glTF ne peut pas éclaircir : il est borné à 1). `pixels_max` limite la
    taille de l'image de couleur (petits éléments).
    """
    empreinte = None if teinte is None else tuple(round(t, 3) for t in teinte)
    cle = (ident, str(dossier), empreinte, pixels_max or 0)
    if cle in _PHOTOS:
        return _PHOTOS[cle]
    try:
        with httpx.Client(
            timeout=120, follow_redirects=True, headers={"User-Agent": "maquette-3d (textures CC0)"}
        ) as http:
            chemins, cote_m = _source(http, ident)
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    dossier.mkdir(parents=True, exist_ok=True)
    sorties = {}
    for nom, chemin in chemins.items():
        fichier = f"ph_{ident}_{nom}.jpg"
        if nom != "couleur" and (dossier / fichier).exists():
            sorties[nom] = fichier
            continue
        image = Image.open(chemin).convert("RGB")
        cote = int(cfg[f"taille_{nom}"])
        if nom == "couleur" and pixels_max:
            cote = min(cote, int(pixels_max))
            fichier = f"ph_{ident}_{cote}_{nom}.jpg"
        image = image.resize((cote, cote), Image.Resampling.LANCZOS)
        if nom == "couleur" and teinte is not None:
            image = coloriser(image, teinte)
            suffixe = "".join(f"{round(t * 255):02x}" for t in teinte)
            fichier = fichier.replace("_couleur", f"_{suffixe}_couleur")
        image.save(
            dossier / fichier, quality=int(cfg["qualite_couleur"]) if nom == "couleur" else 88
        )
        sorties[nom] = fichier
    couleur = np.asarray(Image.open(dossier / sorties["couleur"]), dtype=np.float64) / 255
    moyenne = couleur.reshape(-1, 3).mean(axis=0)
    resultat = Photo(
        sorties["couleur"],
        sorties["arm"],
        sorties["normale"],
        cote_m,
        (float(moyenne[0]), float(moyenne[1]), float(moyenne[2])),
    )
    _PHOTOS[cle] = resultat
    return resultat
