"""Ligne de commande : `maquette site sites/<site>.yaml`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from maquette import blender, visionneuse
from maquette.config import RACINE, charger_config, charger_site
from maquette.scene import construire

app = typer.Typer(
    help="Maquettes 3D des sites étudiés (open data IGN, Blender).", no_args_is_help=True
)


@app.command()
def site(
    fiche: Annotated[Path, typer.Argument(help="Fiche du site (sites/<site>.yaml)")],
    force: Annotated[bool, typer.Option(help="Retélécharger les données")] = False,
    rendu: Annotated[bool, typer.Option(help="Produire les rendus PNG (Cycles)")] = True,
) -> None:
    """Données → géométrie → Blender (GLB, rendus) → visionneuse web dans exports/<site>/."""
    debut = time.time()
    s = charger_site(fiche)
    cfg = charger_config()
    scene = construire(s, cfg, RACINE / "data" / s.identifiant, force)
    typer.echo(f"Scène préparée ({time.time() - debut:.0f} s)")
    sortie = RACINE / "exports" / s.identifiant
    blender.executer(scene, sortie, rendu)
    typer.echo(f"Blender terminé ({time.time() - debut:.0f} s)")
    page = visionneuse.ecrire(scene, sortie)
    typer.echo(f"Visionneuse : {page}")


@app.command("installer-blender")
def installer_blender() -> None:
    """Installe Blender LTS dans outils/ (sans écran, pour la génération)."""
    typer.echo(blender.trouver() or blender.installer())


if __name__ == "__main__":
    app()
