"""Visionneuse web (three.js) : page autonome + GLB, prêtes à publier ou à intégrer au CRM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from maquette.config import RACINE

GABARIT = RACINE / "visionneuse" / "index.html"
GLTFPACK = "gltfpack@0.24.0"


def compresser(glb: Path) -> bool:
    """Compression meshopt (gltfpack) : noms, matériaux et métadonnées conservés.
    Sans Node.js, le GLB reste tel quel (la visionneuse lit les deux)."""
    if shutil.which("npx") is None:
        return False
    sortie = glb.with_suffix(".compresse.glb")
    commande = [
        "npx",
        "--yes",
        GLTFPACK,
        "-i",
        str(glb),
        "-o",
        str(sortie),
        "-cc",
        "-kn",
        "-km",
        "-ke",
        "-vp",
        "16",
        "-vt",
        "14",
    ]
    resultat = subprocess.run(commande, capture_output=True, text=True)
    if resultat.returncode != 0 or not sortie.exists():
        return False
    sortie.replace(glb)
    return True


def ecrire(scene: Path, sortie: Path) -> Path:
    infos = json.loads((scene / "infos.json").read_text(encoding="utf-8"))
    compresser(sortie / "maquette.glb")
    page = GABARIT.read_text(encoding="utf-8").replace(
        "/*INFOS*/null", json.dumps(infos, ensure_ascii=False).replace("</", "<\\/")
    )
    page = page.replace("<title>Maquette 3D</title>", f"<title>{infos['nom']}</title>")
    chemin = sortie / "index.html"
    chemin.write_text(page, encoding="utf-8")
    return chemin
