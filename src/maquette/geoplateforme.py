"""Accès aux services de la Géoplateforme de l'IGN (open data, licence Etalab 2.0).

- WFS : cadastre (Parcellaire Express), BD TOPO (bâtiments, routes, murs, eau) ;
- WMS raster : orthophoto (BD ORTHO) et modèles d'élévation LiDAR HD (MNT, MNS).

Chaque téléchargement est mis en cache avec un `.meta.json` (URL, date, taille, sha256) :
on ne retélécharge pas sauf avec `force`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray
from PIL import Image

WFS = "https://data.geopf.fr/wfs/ows"
WMS = "https://data.geopf.fr/wms-r"
GEOCODAGE = "https://data.geopf.fr/geocodage/search"

COUCHE_ORTHO = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
COUCHE_MNT = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
COUCHE_MNS = "IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"

# Taille maximale d'une image WMS (le service accepte 5010 px) : on tuile au-delà.
TUILE_MAX_PX = 4000
NODATA_MIN = -1000.0

Emprise = tuple[float, float, float, float]


@dataclass(frozen=True)
class Grille:
    """Raster aligné sur une emprise Lambert 93 : `valeurs[ligne, colonne]`, ligne 0 au nord."""

    emprise: Emprise
    resolution: float
    valeurs: NDArray[np.float32]

    def echantillonner(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolation bilinéaire aux points (x, y), bornée aux bords de la grille."""
        xmin, _, _, ymax = self.emprise
        c = (np.asarray(x) - xmin) / self.resolution - 0.5
        r = (ymax - np.asarray(y)) / self.resolution - 0.5
        h, w = self.valeurs.shape
        c = np.clip(c, 0, w - 1.000001)
        r = np.clip(r, 0, h - 1.000001)
        c0 = np.floor(c).astype(int)
        r0 = np.floor(r).astype(int)
        fc = c - c0
        fr = r - r0
        v = self.valeurs
        haut = v[r0, c0] * (1 - fc) + v[r0, c0 + 1] * fc
        bas = v[r0 + 1, c0] * (1 - fc) + v[r0 + 1, c0 + 1] * fc
        return np.asarray(haut * (1 - fr) + bas * fr, dtype=np.float64)


class Client:
    """Client HTTP avec cache disque et reprises sur les erreurs passagères du service."""

    def __init__(self, cache: Path, force: bool = False) -> None:
        self.cache = cache
        self.force = force
        self.http = httpx.Client(timeout=120, follow_redirects=True)

    def _telecharger(self, url: str, params: dict[str, Any], fichier: Path) -> bytes:
        if fichier.exists() and not self.force:
            return fichier.read_bytes()
        fichier.parent.mkdir(parents=True, exist_ok=True)
        for essai in range(5):
            try:
                rep = self.http.get(url, params=params)
                if rep.status_code in (400, 429, 500, 502, 503, 504) and essai < 4:
                    time.sleep(2**essai)
                    continue
                rep.raise_for_status()
                break
            except httpx.TransportError:
                if essai == 4:
                    raise
                time.sleep(2**essai)
        contenu = rep.content
        if "xml" in rep.headers.get("content-type", "") and b"Exception" in contenu[:2000]:
            raise RuntimeError(f"Erreur du service : {contenu[:500]!r}")
        fichier.write_bytes(contenu)
        meta = {
            "url": str(rep.url),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "taille": len(contenu),
            "sha256": hashlib.sha256(contenu).hexdigest(),
        }
        fichier.with_name(fichier.name + ".meta.json").write_text(json.dumps(meta, indent=2))
        return contenu

    def wfs(
        self, couche: str, nom: str, emprise: Emprise | None = None, filtre: str | None = None
    ) -> list[dict[str, Any]]:
        """Toutes les entités d'une couche WFS (Lambert 93), par pages de 1000."""
        entites: list[dict[str, Any]] = []
        page = 0
        while True:
            params: dict[str, Any] = {
                "SERVICE": "WFS",
                "VERSION": "2.0.0",
                "REQUEST": "GetFeature",
                "TYPENAMES": couche,
                "SRSNAME": "EPSG:2154",
                "OUTPUTFORMAT": "application/json",
                "COUNT": 1000,
                "STARTINDEX": page * 1000,
            }
            if emprise is not None:
                params["BBOX"] = ",".join(f"{v:.2f}" for v in emprise) + ",EPSG:2154"
            if filtre is not None:
                params["CQL_FILTER"] = filtre
            brut = self._telecharger(WFS, params, self.cache / "wfs" / f"{nom}_{page}.json")
            lot = json.loads(brut)["features"]
            entites.extend(lot)
            if len(lot) < 1000:
                return entites
            page += 1

    def _wms(
        self, couche: str, emprise: Emprise, largeur: int, hauteur: int, format_: str, fichier: Path
    ) -> bytes:
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": couche,
            "STYLES": "",
            "CRS": "EPSG:2154",
            "BBOX": ",".join(f"{v:.2f}" for v in emprise),
            "WIDTH": largeur,
            "HEIGHT": hauteur,
            "FORMAT": format_,
        }
        return self._telecharger(WMS, params, fichier)

    def _tuiles(
        self, emprise: Emprise, resolution: float
    ) -> tuple[int, int, list[tuple[int, int, int, int]]]:
        xmin, ymin, xmax, ymax = emprise
        w = round((xmax - xmin) / resolution)
        h = round((ymax - ymin) / resolution)
        tuiles = []
        for r0 in range(0, h, TUILE_MAX_PX):
            for c0 in range(0, w, TUILE_MAX_PX):
                tuiles.append((r0, c0, min(TUILE_MAX_PX, h - r0), min(TUILE_MAX_PX, w - c0)))
        return w, h, tuiles

    def ortho(self, emprise: Emprise, resolution: float) -> Image.Image:
        """Orthophoto RVB de l'emprise, assemblée par tuiles."""
        w, h, tuiles = self._tuiles(emprise, resolution)
        image = Image.new("RGB", (w, h))
        xmin, _, _, ymax = emprise
        for r0, c0, th, tw in tuiles:
            e = (
                xmin + c0 * resolution,
                ymax - (r0 + th) * resolution,
                xmin + (c0 + tw) * resolution,
                ymax - r0 * resolution,
            )
            brut = self._wms(
                COUCHE_ORTHO,
                e,
                tw,
                th,
                "image/jpeg",
                self.cache / "wms" / f"ortho_{resolution}_{r0}_{c0}.jpg",
            )
            image.paste(Image.open(io.BytesIO(brut)).convert("RGB"), (c0, r0))
        return image

    def elevation(self, couche: str, nom: str, emprise: Emprise, resolution: float) -> Grille:
        """Modèle d'élévation (float32) ; les trous sont bouchés par le plus proche voisin."""
        from scipy import ndimage

        w, h, tuiles = self._tuiles(emprise, resolution)
        valeurs = np.full((h, w), np.nan, dtype=np.float32)
        xmin, _, _, ymax = emprise
        for r0, c0, th, tw in tuiles:
            e = (
                xmin + c0 * resolution,
                ymax - (r0 + th) * resolution,
                xmin + (c0 + tw) * resolution,
                ymax - r0 * resolution,
            )
            brut = self._wms(
                couche,
                e,
                tw,
                th,
                "image/x-bil;bits=32",
                self.cache / "wms" / f"{nom}_{resolution}_{r0}_{c0}.bil",
            )
            if len(brut) != tw * th * 4:
                raise RuntimeError(f"{nom} : réponse inattendue ({len(brut)} octets)")
            valeurs[r0 : r0 + th, c0 : c0 + tw] = np.frombuffer(brut, dtype="<f4").reshape(th, tw)
        trous = ~np.isfinite(valeurs) | (valeurs < NODATA_MIN)
        if trous.all():
            raise RuntimeError(f"{nom} : aucune donnée sur l'emprise (hors couverture LiDAR HD ?)")
        if trous.any():
            _, (ri, ci) = ndimage.distance_transform_edt(trous, return_indices=True)
            valeurs = valeurs[ri, ci]
        return Grille(emprise, resolution, valeurs)

    def geocoder(self, adresse: str) -> tuple[float, float] | None:
        """Coordonnées Lambert 93 d'une adresse (Base adresse nationale)."""
        rep = self.http.get(GEOCODAGE, params={"q": adresse, "limit": 1})
        rep.raise_for_status()
        resultats = rep.json()["features"]
        if not resultats:
            return None
        p = resultats[0]["properties"]
        return float(p["x"]), float(p["y"])


def emprise_alignee(
    xmin: float, ymin: float, xmax: float, ymax: float, pas: float = 2.0
) -> Emprise:
    """Emprise élargie aux multiples de `pas` mètres (grilles et textures alignées)."""
    return (
        math.floor(xmin / pas) * pas,
        math.floor(ymin / pas) * pas,
        math.ceil(xmax / pas) * pas,
        math.ceil(ymax / pas) * pas,
    )
