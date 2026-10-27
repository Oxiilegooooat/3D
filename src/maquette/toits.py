"""Toits paramétriques ajustés sur le MNS LiDAR HD : plat, appentis, deux pans, croupe.

Le MNS est bruité (± 10 cm, cheminées, arbres en surplomb) : un toit maillé directement
dessus paraît bosselé. On ajuste donc des plans par moindres carrés robustes et on garde le
modèle le plus simple qui explique les mesures ; au-delà d'un écart de 0,6 m, le toit est
jugé complexe et reste maillé sur le MNS lissé.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon

Tableau = NDArray[np.float64]
ECART_COMPLEXE = 0.6
TOLERANCE_SIMPLICITE = 0.07


@dataclass(frozen=True)
class ModeleToit:
    nature: str  # plat, appentis, deux_pans, croupe
    cx: float
    cy: float
    angle: float  # axe u
    params: tuple[float, ...]
    ecart: float  # écart quadratique robuste (m)

    def _uv(self, xy: Tableau) -> tuple[Tableau, Tableau]:
        c, s = np.cos(self.angle), np.sin(self.angle)
        dx, dy = xy[:, 0] - self.cx, xy[:, 1] - self.cy
        return dx * c + dy * s, -dx * s + dy * c

    def _xy(self, u: float, v: float) -> tuple[float, float]:
        c, s = np.cos(self.angle), np.sin(self.angle)
        return self.cx + u * c - v * s, self.cy + u * s + v * c

    def z(self, xy: Tableau) -> Tableau:
        u, v = self._uv(np.atleast_2d(xy))
        p = self.params
        if self.nature == "plat":
            return np.full(len(u), p[0])
        if self.nature == "appentis":
            return p[0] + p[1] * u + p[2] * v
        if self.nature == "deux_pans":
            a, b1, b2, v0 = p
            return a - b1 * np.maximum(v0 - v, 0) - b2 * np.maximum(v - v0, 0)
        a, b, e = p  # croupe
        return a - b * np.maximum(np.abs(v), np.abs(u) - e)

    def plis(self, etendue: float) -> list[LineString]:
        """Arêtes du toit (faîtage, arêtiers), à découper par l'emprise."""
        p = self.params
        if self.nature == "deux_pans":
            v0 = p[3]
            return [LineString([self._xy(-etendue, v0), self._xy(etendue, v0)])]
        if self.nature == "croupe":
            e = p[2]
            lignes = [LineString([self._xy(-e, 0), self._xy(e, 0)])] if e > 0.05 else []
            for su in (-1, 1):
                for sv in (-1, 1):
                    lignes.append(
                        LineString(
                            [self._xy(su * e, 0), self._xy(su * (e + etendue), sv * etendue)]
                        )
                    )
            return lignes
        return []


def _robuste(a: Tableau, z: Tableau) -> tuple[Tableau, float]:
    """Moindres carrés, puis nouvel ajustement sur les 85 % de points les mieux expliqués."""
    coef = np.linalg.lstsq(a, z, rcond=None)[0]
    residus = np.abs(a @ coef - z)
    garde = residus <= np.quantile(residus, 0.85)
    if garde.sum() >= a.shape[1] + 2:
        coef = np.linalg.lstsq(a[garde], z[garde], rcond=None)[0]
    residus = np.abs(a @ coef - z)
    garde = residus <= np.quantile(residus, 0.85)
    return coef, float(np.sqrt(np.mean(residus[garde] ** 2)))


def ajuster(emprise: Polygon, x: Tableau, y: Tableau, z: Tableau) -> ModeleToit | None:
    """Meilleur modèle simple, ou None si le toit est complexe."""
    if len(z) < 8:
        return None
    rect = emprise.minimum_rotated_rectangle
    coins = np.asarray(rect.exterior.coords)[:4]
    cotes = np.diff(np.vstack([coins, coins[:1]]), axis=0)
    longueurs = np.hypot(*cotes.T)
    i = int(np.argmax(longueurs[:2]))
    angle = float(np.arctan2(cotes[i, 1], cotes[i, 0]))
    centre = rect.centroid
    cx, cy = centre.x, centre.y
    hl, hw = longueurs[i] / 2, longueurs[1 - i] / 2
    c, s = np.cos(angle), np.sin(angle)
    u = (x - cx) * c + (y - cy) * s
    v = -(x - cx) * s + (y - cy) * c
    un = np.ones_like(z)
    candidats: list[ModeleToit] = []

    coef, ecart = _robuste(un[:, None], z)
    candidats.append(ModeleToit("plat", cx, cy, angle, (float(coef[0]),), ecart))
    coef, ecart = _robuste(np.column_stack([un, u, v]), z)
    candidats.append(ModeleToit("appentis", cx, cy, angle, tuple(map(float, coef)), ecart))

    for axe in (0, 1):
        # Faîtage parallèle à u (axe 0) ou à v (axe 1) : on échange les rôles.
        vv, demi = (v, hw) if axe == 0 else (u, hl)
        meilleur: ModeleToit | None = None
        for v0 in np.linspace(-0.6 * demi, 0.6 * demi, 25):
            a = np.column_stack([un, -np.maximum(v0 - vv, 0), -np.maximum(vv - v0, 0)])
            coef, ecart = _robuste(a, z)
            if coef[1] < 0.08 or coef[2] < 0.08:
                continue
            if meilleur is None or ecart < meilleur.ecart:
                meilleur = ModeleToit(
                    "deux_pans",
                    cx,
                    cy,
                    angle - axe * np.pi / 2,
                    (float(coef[0]), float(coef[1]), float(coef[2]), float(v0)),
                    ecart,
                )
        if meilleur is not None:
            candidats.append(meilleur)

    if hl - hw > 0.3:
        meilleur = None
        for e in np.linspace(0, hl - hw, 12):
            a = np.column_stack([un, -np.maximum(np.abs(v), np.abs(u) - e)])
            coef, ecart = _robuste(a, z)
            if coef[1] < 0.08:
                continue
            if meilleur is None or ecart < meilleur.ecart:
                meilleur = ModeleToit(
                    "croupe", cx, cy, angle, (float(coef[0]), float(coef[1]), float(e)), ecart
                )
        if meilleur is not None:
            candidats.append(meilleur)

    plus_petit = min(m.ecart for m in candidats)
    if plus_petit > ECART_COMPLEXE:
        return None
    for m in candidats:  # dans l'ordre de simplicité
        if m.ecart <= plus_petit + TOLERANCE_SIMPLICITE:
            return m
    return None


def plis_dans(
    modele: ModeleToit, emprise: Polygon
) -> tuple[list[LineString], list[tuple[float, float]]]:
    """Plis découpés par l'emprise et leurs points de contact avec le contour."""
    etendue = float(np.hypot(*np.ptp(np.asarray(emprise.exterior.coords), axis=0))) + 2
    morceaux: list[LineString] = []
    contacts: list[tuple[float, float]] = []
    for pli in modele.plis(etendue):
        inter = pli.intersection(emprise)
        for g in getattr(inter, "geoms", [inter]):
            if isinstance(g, LineString) and g.length > 0.05:
                morceaux.append(g)
        bord = pli.intersection(emprise.boundary)
        for g in getattr(bord, "geoms", [bord]):
            if isinstance(g, shapely.Point):
                contacts.append((g.x, g.y))
    return morceaux, contacts
