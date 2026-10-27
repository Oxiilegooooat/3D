"""Chargement de `config/maquette.yaml` et des fiches de site (`sites/*.yaml`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

RACINE = Path(__file__).resolve().parents[2]

Presentation = Literal["existant", "projet", "les-deux"]


@dataclass(frozen=True)
class Volume:
    """Volume de projet : emprise en Lambert 93, niveaux, toiture."""

    emprise: list[tuple[float, float]]
    niveaux: int
    hauteur_etage: float = 3.0
    toiture: Literal["terrasse", "deux_pans"] = "terrasse"
    attique: bool = False
    balcons: bool = True
    nom: str = ""


@dataclass(frozen=True)
class Projet:
    nom: str
    indicatif: bool
    volumes: list[Volume] = field(default_factory=list)
    modele: Path | None = None
    demolition: bool = True


@dataclass(frozen=True)
class Site:
    identifiant: str
    nom: str
    insee: str
    parcelles: list[str]
    adresse: str | None
    presentation: Presentation
    projet: Projet | None

    @property
    def idus(self) -> list[str]:
        """idu sur 14 caractères : insee + préfixe (000) + section + numéro."""
        idus = []
        for p in self.parcelles:
            p = p.replace(" ", "").upper()
            section = p.rstrip("0123456789").rjust(2, "0")
            numero = p[len(p.rstrip("0123456789")) :].rjust(4, "0")
            idus.append(f"{self.insee}000{section}{numero}")
        return idus


def charger_config(chemin: Path | None = None) -> dict[str, Any]:
    chemin = chemin or RACINE / "config" / "maquette.yaml"
    config: dict[str, Any] = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    return config


def charger_site(chemin: Path) -> Site:
    brut = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    projet = None
    if brut.get("projet"):
        p = brut["projet"]
        volumes = [
            Volume(
                emprise=[(float(x), float(y)) for x, y in v["emprise"]],
                niveaux=int(v["niveaux"]),
                hauteur_etage=float(v.get("hauteur_etage", 3.0)),
                toiture=v.get("toiture", "terrasse"),
                attique=bool(v.get("attique", False)),
                balcons=bool(v.get("balcons", True)),
                nom=v.get("nom", ""),
            )
            for v in p.get("volumes", [])
        ]
        modele = (chemin.parent / p["modele"]).resolve() if p.get("modele") else None
        projet = Projet(
            nom=p.get("nom", "Projet"),
            indicatif=bool(p.get("indicatif", True)),
            volumes=volumes,
            modele=modele,
            demolition=bool(p.get("demolition", True)),
        )
    presentation = brut.get("presentation", "existant")
    if presentation not in ("existant", "projet", "les-deux"):
        raise ValueError(f"presentation inconnue : {presentation}")
    if presentation != "existant" and projet is None:
        raise ValueError("presentation « projet » ou « les-deux » sans projet renseigné")
    return Site(
        identifiant=chemin.stem,
        nom=brut["nom"],
        insee=str(brut["insee"]),
        parcelles=[str(p) for p in brut["parcelles"]],
        adresse=brut.get("adresse"),
        presentation=presentation,
        projet=projet,
    )
