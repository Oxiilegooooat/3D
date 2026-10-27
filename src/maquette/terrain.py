"""Terrain naturel (MNT LiDAR HD) maillé en grille régulière, avec un socle de maquette."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from maquette.geoplateforme import Grille
from maquette.maillage import Maillage, Tableau


@dataclass
class Terrain:
    """Grille d'altitudes locales `z[ligne, colonne]`, ligne 0 au sud, au pas `pas`.

    `hauteur` interpole exactement comme le maillage (deux triangles par maille, diagonale
    sud-ouest → nord-est) : routes et trottoirs drapés ne passent pas sous le sol.
    """

    xmin: float
    ymin: float
    pas: float
    z: NDArray[np.float64]

    @classmethod
    def depuis_mnt(cls, mnt: Grille, origine: tuple[float, float, float], pas: float) -> Terrain:
        xmin, ymin, xmax, ymax = mnt.emprise
        nx = round((xmax - xmin) / pas) + 1
        ny = round((ymax - ymin) / pas) + 1
        xs = xmin + np.arange(nx) * pas
        ys = ymin + np.arange(ny) * pas
        gx, gy = np.meshgrid(xs, ys)
        z = mnt.echantillonner(gx.ravel(), gy.ravel()).reshape(ny, nx) - origine[2]
        return cls(xmin - origine[0], ymin - origine[1], pas, z)

    @property
    def xmax(self) -> float:
        return float(self.xmin + (self.z.shape[1] - 1) * self.pas)

    @property
    def ymax(self) -> float:
        return float(self.ymin + (self.z.shape[0] - 1) * self.pas)

    def hauteur(self, xy: Tableau) -> Tableau:
        xy = np.atleast_2d(xy)
        ny, nx = self.z.shape
        fx = np.clip((xy[:, 0] - self.xmin) / self.pas, 0, nx - 1.000001)
        fy = np.clip((xy[:, 1] - self.ymin) / self.pas, 0, ny - 1.000001)
        i = np.floor(fx).astype(int)
        j = np.floor(fy).astype(int)
        u = fx - i
        v = fy - j
        z00 = self.z[j, i]
        z10 = self.z[j, i + 1]
        z01 = self.z[j + 1, i]
        z11 = self.z[j + 1, i + 1]
        bas = z00 + u * (z10 - z00) + v * (z11 - z10)
        haut = z00 + v * (z01 - z00) + u * (z11 - z01)
        return np.asarray(np.where(u >= v, bas, haut), dtype=np.float64)

    def uv_ortho(self, xy: Tableau) -> Tableau:
        return np.column_stack(
            [
                (xy[:, 0] - self.xmin) / (self.xmax - self.xmin),
                (xy[:, 1] - self.ymin) / (self.ymax - self.ymin),
            ]
        )

    def maillage(self) -> Maillage:
        ny, nx = self.z.shape
        gx, gy = np.meshgrid(
            self.xmin + np.arange(nx) * self.pas, self.ymin + np.arange(ny) * self.pas
        )
        sommets = np.column_stack([gx.ravel(), gy.ravel(), self.z.ravel()])
        idx = np.arange(nx * ny).reshape(ny, nx)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, 1:].ravel()
        d = idx[1:, :-1].ravel()
        faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        return Maillage(sommets, faces.astype(np.int64), self.uv_ortho(sommets[:, :2]))

    def socle(self, epaisseur: float) -> Maillage:
        """Flancs verticaux et dessous du socle, jusqu'à `epaisseur` sous le point le plus bas."""
        ny, nx = self.z.shape
        zbas = float(self.z.min()) - epaisseur
        xs = self.xmin + np.arange(nx) * self.pas
        ys = self.ymin + np.arange(ny) * self.pas
        # Contour parcouru dans le sens trigonométrique (vu de dessus) : sud, est, nord, ouest.
        bord = (
            [(x, self.ymin, self.z[0, i]) for i, x in enumerate(xs)]
            + [(self.xmax, y, self.z[j, -1]) for j, y in enumerate(ys)][1:]
            + [(x, self.ymax, self.z[-1, i]) for i, x in reversed(list(enumerate(xs)))][1:]
            + [(self.xmin, y, self.z[j, 0]) for j, y in reversed(list(enumerate(ys)))][1:]
        )
        sommets: list[tuple[float, float, float]] = []
        faces: list[list[int]] = []
        uv: list[tuple[float, float]] = []
        s = 0.0
        for k in range(len(bord) - 1):
            p, q = bord[k], bord[k + 1]
            n = len(sommets)
            sommets += [(p[0], p[1], zbas), (q[0], q[1], zbas), q, p]
            ds = float(np.hypot(q[0] - p[0], q[1] - p[1]))
            uv += [(s, 0), (s + ds, 0), (s + ds, q[2] - zbas), (s, p[2] - zbas)]
            s += ds
            faces += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
        n = len(sommets)
        sommets += [
            (self.xmin, self.ymin, zbas),
            (self.xmin, self.ymax, zbas),
            (self.xmax, self.ymax, zbas),
            (self.xmax, self.ymin, zbas),
        ]
        uv += [(0, 0), (0, 1), (1, 1), (1, 0)]
        faces += [[n, n + 1, n + 2], [n, n + 2, n + 3]]
        return Maillage(np.array(sommets), np.array(faces, dtype=np.int64), np.array(uv) / 4.0)
