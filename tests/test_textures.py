"""Matériaux photographiés : colorisation, choix de l'entrée, repli procédural hors ligne."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from maquette import bibliotheque, textures


def test_colorisation_garde_le_grain_et_impose_la_teinte() -> None:
    rng = np.random.default_rng(0)
    rouge = np.zeros((64, 64, 3))
    rouge[..., 0] = 0.6 + 0.2 * rng.random((64, 64))
    rouge[..., 1:] = 0.1
    image = Image.fromarray((rouge * 255).astype(np.uint8))
    gris = (0.7, 0.7, 0.72)
    sortie = np.asarray(bibliotheque.coloriser(image, gris), dtype=np.float64) / 255
    moyenne = sortie.reshape(-1, 3).mean(axis=0)
    assert np.allclose(moyenne, gris, atol=0.06)
    assert sortie[..., 1].std() > 0.02  # le grain reste


def test_entrees_de_la_configuration() -> None:
    cfg = bibliotheque.configuration()
    cle, teinte = textures._cle_photo("facade_enduit_2", {}, cfg)
    assert cle == "facade_enduit" and teinte == pytest.approx((238 / 255, 233 / 255, 223 / 255))
    toits = {"toit_tuiles_754": ("tuiles", (0.7, 0.5, 0.4))}
    assert textures._cle_photo("toit_tuiles_754", toits, cfg) == (
        "couverture_tuiles",
        (0.7, 0.5, 0.4),
    )
    assert textures._cle_photo("facade_brique", {}, cfg) == ("facade_brique", None)


def test_repli_procedural_sans_reseau(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bibliotheque, "photo", lambda *a, **k: None)
    materiaux = textures.catalogue({"facade_brique", "sol_pelouse", "verre"}, tmp_path, {}, "")
    assert materiaux["facade_brique"].texture == "facade_brique_couleur.jpg"
    assert materiaux["sol_pelouse"].echelle == 1.0
    assert materiaux["verre"].texture is None
