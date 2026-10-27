"""Visionneuse web (three.js) : page autonome + GLB, prêtes à publier ou à intégrer au CRM."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from maquette.config import RACINE

GABARIT = RACINE / "visionneuse" / "index.html"
GLTFPACK = "gltfpack@0.24.0"
TAILLE_MAX = 14_000_000  # octets : au-delà, compression meshopt (décodeur WebAssembly)


def compresser(glb: Path) -> str | None:
    """Quantification des sommets (KHR_mesh_quantization, lue sans décodeur) ; compression
    meshopt en plus si le fichier dépasse `TAILLE_MAX`. Noms, matériaux et métadonnées
    conservés. Sans Node.js, le GLB reste tel quel."""
    if shutil.which("npx") is None:
        return None
    sortie = glb.with_suffix(".compresse.glb")
    base = ["npx", "--yes", GLTFPACK, "-i", str(glb), "-o", str(sortie), "-kn", "-km", "-ke"]
    base += ["-vp", "16", "-vt", "14"]
    for mode, options in (("quantification", []), ("meshopt", ["-cc"])):
        resultat = subprocess.run(base + options, capture_output=True, text=True)
        if resultat.returncode != 0 or not sortie.exists():
            return None
        if mode == "meshopt" or sortie.stat().st_size <= TAILLE_MAX:
            sortie.replace(glb)
            return mode
    return None


def ecrire(scene: Path, sortie: Path) -> Path:
    infos = json.loads((scene / "infos.json").read_text(encoding="utf-8"))
    infos["compression"] = compresser(sortie / "maquette.glb")
    page = GABARIT.read_text(encoding="utf-8").replace(
        "/*INFOS*/null", json.dumps(infos, ensure_ascii=False).replace("</", "<\\/")
    )
    page = page.replace("<title>Maquette 3D</title>", f"<title>{infos['nom']}</title>")
    chemin = sortie / "index.html"
    chemin.write_text(page, encoding="utf-8")
    return chemin
