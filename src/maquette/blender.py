"""Blender en mode sans écran : localisation, installation (version LTS officielle), exécution."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import httpx

from maquette.config import RACINE

VERSION = "4.5.15"
ARCHIVE = f"blender-{VERSION}-linux-x64"
URL = f"https://download.blender.org/release/Blender4.5/{ARCHIVE}.tar.xz"
SCRIPT = RACINE / "blender" / "construire.py"


def trouver() -> Path | None:
    """`MAQUETTE_BLENDER`, puis `outils/`, puis le PATH."""
    if os.environ.get("MAQUETTE_BLENDER"):
        return Path(os.environ["MAQUETTE_BLENDER"])
    local = RACINE / "outils" / ARCHIVE / "blender"
    if local.exists():
        return local
    systeme = shutil.which("blender")
    return Path(systeme) if systeme else None


def installer() -> Path:
    """Télécharge et extrait Blender LTS dans `outils/` (ignoré par git)."""
    outils = RACINE / "outils"
    outils.mkdir(exist_ok=True)
    archive = outils / f"{ARCHIVE}.tar.xz"
    if not archive.exists():
        with httpx.stream("GET", URL, follow_redirects=True, timeout=600) as rep:
            rep.raise_for_status()
            with archive.open("wb") as f:
                for morceau in rep.iter_bytes():
                    f.write(morceau)
    with tarfile.open(archive) as tar:
        tar.extractall(outils, filter="tar")
    archive.unlink()
    return outils / ARCHIVE / "blender"


def executer(scene: Path, sortie: Path, rendu: bool = True) -> None:
    executable = trouver() or installer()
    commande = [
        str(executable),
        "-b",
        "--factory-startup",
        "-P",
        str(SCRIPT),
        "--",
        str(scene),
        str(sortie),
    ]
    if not rendu:
        commande.append("--sans-rendu")
    resultat = subprocess.run(commande, capture_output=True, text=True)
    if resultat.returncode != 0 or "Traceback" in resultat.stdout + resultat.stderr:
        raise RuntimeError(
            f"Blender a échoué :\n{resultat.stdout[-3000:]}\n{resultat.stderr[-3000:]}"
        )
