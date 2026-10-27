"""Maillages triangulés (sommets, faces, UV) et primitives géométriques.

Repère local : X à l'est, Y au nord, Z vers le haut, en mètres, origine au centre du site.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import shapely
import triangle
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

Tableau = NDArray[np.float64]


@dataclass
class Maillage:
    sommets: Tableau  # (n, 3)
    faces: NDArray[np.int64]  # (m, 3)
    uv: Tableau  # (n, 2)

    @staticmethod
    def vide() -> Maillage:
        return Maillage(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2)))

    def __len__(self) -> int:
        return len(self.faces)


def fusionner(maillages: Iterable[Maillage]) -> Maillage:
    liste = [m for m in maillages if len(m)]
    if not liste:
        return Maillage.vide()
    decalages = np.cumsum([0] + [len(m.sommets) for m in liste[:-1]])
    return Maillage(
        np.concatenate([m.sommets for m in liste]),
        np.concatenate([m.faces + d for m, d in zip(liste, decalages, strict=True)]),
        np.concatenate([m.uv for m in liste]),
    )


def polygones(geometrie: shapely.Geometry) -> list[Polygon]:
    """Polygones non vides d'une géométrie quelconque (collections comprises)."""
    if geometrie is None or geometrie.is_empty:
        return []
    if isinstance(geometrie, Polygon):
        return [geometrie]
    if isinstance(geometrie, MultiPolygon):
        return [g for g in geometrie.geoms if not g.is_empty]
    if hasattr(geometrie, "geoms"):
        return [p for g in geometrie.geoms for p in polygones(g)]
    return []


def _anneau(coords: Iterable[tuple[float, ...]]) -> Tableau:
    pts = np.asarray([c[:2] for c in coords], dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    # Points consécutifs confondus : `triangle` les refuse.
    garde = np.ones(len(pts), dtype=bool)
    garde[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3
    return pts[garde]


def anneaux(poly: Polygon, pas: float | None = None) -> list[Tableau]:
    """Anneau extérieur puis trous, densifiés au pas donné, sans point de fermeture."""
    if pas:
        poly = shapely.segmentize(poly, pas)
    return [_anneau(poly.exterior.coords)] + [_anneau(i.coords) for i in poly.interiors]


def densifier(
    anneau: Tableau, pas: float, extras: Tableau | None = None
) -> tuple[Tableau, NDArray[np.int64], Tableau]:
    """Anneau densifié : points, indice de l'arête d'origine, abscisse sur cette arête.

    `extras` : points posés sur le contour (contact d'un faîtage) insérés à leur place.
    """
    pts, aretes, abscisses = [], [], []
    k = len(anneau)
    for i in range(k):
        a, b = anneau[i], anneau[(i + 1) % k]
        d = b - a
        longueur = float(np.hypot(*d))
        n = max(1, int(np.ceil(longueur / pas - 1e-9)))
        ts = [longueur * j / n for j in range(n)]
        if extras is not None and len(extras) and longueur > 1e-9:
            t = (extras - a) @ d / longueur
            proj = a + np.outer(t / longueur, d)
            sur = (np.hypot(*(extras - proj).T) < 1e-3) & (t > 0.02) & (t < longueur - 0.02)
            ajouts = sorted(float(v) for v in t[sur])
            ts = [x for x in ts if x == 0 or all(abs(x - y) > 0.05 for y in ajouts)] + ajouts
            ts.sort()
        for t in ts:
            pts.append(a + d * t / max(longueur, 1e-12))
            aretes.append(i)
            abscisses.append(t)
    return np.array(pts), np.array(aretes, dtype=np.int64), np.array(abscisses)


def trianguler(poly: Polygon, pas: float | None = None) -> tuple[Tableau, NDArray[np.int64]]:
    """Triangulation contrainte d'un polygone à trous (2D).

    Avec `pas`, le contour est densifié et l'intérieur raffiné (triangles d'aire ≤ pas²/2) :
    le maillage peut alors épouser un relief. Le contour n'est jamais redécoupé (option Y),
    pour rester conforme aux murs.
    """
    if not poly.is_valid:
        valides = polygones(shapely.make_valid(poly))
        if not valides:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
        poly = max(valides, key=lambda p: p.area)
    return trianguler_anneaux(anneaux(poly, pas), pas)


def trianguler_anneaux(
    liste: list[Tableau], pas: float | None = None, lignes: list[Tableau] | None = None
) -> tuple[Tableau, NDArray[np.int64]]:
    """Triangulation d'anneaux déjà densifiés : le premier est l'extérieur, les suivants des trous.

    `lignes` : polylignes intérieures imposées (faîtages, arêtiers), dont les extrémités sur le
    contour coïncident avec des points des anneaux.
    """
    sommets: list[Tableau] = []
    segments: list[NDArray[np.int64]] = []
    trous: list[tuple[float, float]] = []
    n = 0
    for i, anneau in enumerate(liste):
        if len(anneau) < 3:
            continue
        k = len(anneau)
        sommets.append(anneau)
        segments.append(np.stack([np.arange(k) + n, (np.arange(k) + 1) % k + n], axis=1))
        n += k
        if i > 0:
            p = Polygon(anneau).representative_point()
            trous.append((p.x, p.y))
    if not sommets:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    tous = list(np.concatenate(sommets))
    for ligne in lignes or []:
        indices = []
        for q in ligne:
            ecarts = np.hypot(*(np.asarray(tous) - q).T)
            j = int(np.argmin(ecarts))
            if ecarts[j] < 0.02:
                indices.append(j)
            else:
                tous.append(np.asarray(q, dtype=np.float64))
                indices.append(len(tous) - 1)
        for a, b in pairwise(indices):
            if a != b:
                segments.append(np.array([[a, b]]))
    entree: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "vertices": np.asarray(tous),
        "segments": np.concatenate(segments),
    }
    if trous:
        entree["holes"] = np.asarray(trous)
    options = "pY" + ("q20" if pas else "") + (f"a{pas * pas / 2:.4f}" if pas else "")
    try:
        sortie = triangle.triangulate(entree, options)
    except Exception:
        sortie = triangle.triangulate(entree, "p")
    if "triangles" not in sortie:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    return np.asarray(sortie["vertices"]), np.asarray(sortie["triangles"], dtype=np.int64)


def surface(
    poly: Polygon,
    hauteur: Callable[[Tableau], Tableau],
    pas: float | None,
    uv: Callable[[Tableau], Tableau],
) -> Maillage:
    """Surface triangulée d'un polygone, d'altitude `hauteur(xy)` (orientée vers le haut).

    Avec `pas`, le polygone est découpé par une grille avant d'être triangulé : la surface
    suit le relief à ce pas. Triangulation contrainte de GEOS, robuste aux polygones à trous
    qui se touchent (sol percé par les bâtiments).
    """
    xy, faces = trianguler_robuste(poly, pas)
    if not len(faces):
        return Maillage.vide()
    z = hauteur(xy)
    faces = orienter_vers_le_haut(xy, faces)
    return Maillage(np.column_stack([xy, z]), faces, uv(xy))


def trianguler_robuste(
    geometrie: shapely.Geometry, pas: float | None = None
) -> tuple[Tableau, NDArray[np.int64]]:
    """Triangles d'une géométrie surfacique quelconque (sommets partagés)."""
    geometrie = shapely.make_valid(geometrie)
    if geometrie.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    if pas:
        x0, y0, x1, y1 = geometrie.bounds
        xs = np.arange(np.floor(x0 / pas) * pas, x1 + pas, pas)
        ys = np.arange(np.floor(y0 / pas) * pas, y1 + pas, pas)
        gx, gy = np.meshgrid(xs[:-1], ys[:-1])
        cellules = shapely.box(gx.ravel(), gy.ravel(), gx.ravel() + pas, gy.ravel() + pas)
        shapely.prepare(geometrie)
        cellules = cellules[shapely.intersects(geometrie, cellules)]
        morceaux = shapely.intersection(cellules, geometrie)
    else:
        morceaux = np.array([geometrie])
    morceaux = morceaux[~shapely.is_empty(morceaux)]
    triangles = shapely.get_parts(
        shapely.get_parts(shapely.constrained_delaunay_triangles(morceaux))
    )
    triangles = triangles[shapely.get_type_id(triangles) == 3]
    triangles = triangles[shapely.area(triangles) > 1e-8]
    if not len(triangles):
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    coins = shapely.get_coordinates(shapely.get_exterior_ring(triangles)).reshape(-1, 4, 2)[:, :3]
    cles = np.round(coins.reshape(-1, 2), 4)
    xy, inverse = np.unique(cles, axis=0, return_inverse=True)
    return xy, inverse.reshape(-1, 3).astype(np.int64)


def orienter_vers_le_haut(xy: Tableau, faces: NDArray[np.int64]) -> NDArray[np.int64]:
    a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
    aire = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    faces = faces.copy()
    faces[aire < 0] = faces[aire < 0][:, [0, 2, 1]]
    return faces


def decouper_bande(
    polygone_sz: list[tuple[float, float]], zmin: float, zmax: float
) -> list[tuple[float, float]]:
    """Coupe un polygone convexe du plan (s, z) entre deux altitudes (Sutherland-Hodgman)."""

    def couper(
        pts: list[tuple[float, float]], limite: float, dessus: bool
    ) -> list[tuple[float, float]]:
        sortie: list[tuple[float, float]] = []
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            p_in = p[1] >= limite if dessus else p[1] <= limite
            q_in = q[1] >= limite if dessus else q[1] <= limite
            if p_in:
                sortie.append(p)
            if p_in != q_in and q[1] != p[1]:
                t = (limite - p[1]) / (q[1] - p[1])
                sortie.append((p[0] + t * (q[0] - p[0]), limite))
        return sortie

    pts = couper(polygone_sz, zmin, True)
    if len(pts) >= 3:
        pts = couper(pts, zmax, False)
    return pts if len(pts) >= 3 else []


def quad_mur(
    p0: Tableau,
    p1: Tableau,
    polygone_sz: list[tuple[float, float]],
    uv: Callable[[float, float], tuple[float, float]],
) -> Maillage:
    """Polygone convexe du plan vertical (p0 → p1), s en mètres depuis p0, triangulé en éventail.

    Donné dans le sens trigonométrique du plan (s, z), il regarde à droite du sens p0 → p1 :
    l'extérieur d'un anneau extérieur trigonométrique (ou d'un trou horaire).
    """
    if len(polygone_sz) < 3:
        return Maillage.vide()
    d = p1 - p0
    longueur = float(np.hypot(*d))
    if longueur < 1e-6:
        return Maillage.vide()
    direction = d / longueur
    sommets = np.array(
        [[p0[0] + s * direction[0], p0[1] + s * direction[1], z] for s, z in polygone_sz]
    )
    uvs = np.array([uv(s, z) for s, z in polygone_sz])
    k = len(polygone_sz)
    faces = np.array([[0, i, i + 1] for i in range(1, k - 1)], dtype=np.int64)
    return Maillage(sommets, faces, uvs)


def boite(
    centre: Tableau,
    axe_u: Tableau,
    largeur: float,
    profondeur: float,
    hauteur: float,
    echelle_uv: float = 1.0,
) -> Maillage:
    """Pavé posé sur `centre` (base), largeur selon `axe_u` (horizontal), profondeur selon la
    normale horizontale à gauche de `axe_u`."""
    u = np.array([axe_u[0], axe_u[1], 0.0])
    u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    du, dn, dz = u * largeur / 2, n * profondeur / 2, z * hauteur
    c = np.asarray(centre, dtype=np.float64)
    coins = [c - du - dn, c + du - dn, c + du + dn, c - du + dn]
    bas = [p for p in coins]
    haut = [p + dz for p in coins]
    faces_q = [
        (bas[0], bas[1], haut[1], haut[0], largeur, hauteur),
        (bas[1], bas[2], haut[2], haut[1], profondeur, hauteur),
        (bas[2], bas[3], haut[3], haut[2], largeur, hauteur),
        (bas[3], bas[0], haut[0], haut[3], profondeur, hauteur),
        (haut[0], haut[1], haut[2], haut[3], largeur, profondeur),
        (bas[3], bas[2], bas[1], bas[0], largeur, profondeur),
    ]
    sommets, uvs, faces = [], [], []
    for i, (a, b, cc, d, lu, lv) in enumerate(faces_q):
        sommets += [a, b, cc, d]
        uvs += [
            (0, 0),
            (lu / echelle_uv, 0),
            (lu / echelle_uv, lv / echelle_uv),
            (0, lv / echelle_uv),
        ]
        faces += [[4 * i, 4 * i + 1, 4 * i + 2], [4 * i, 4 * i + 2, 4 * i + 3]]
    return Maillage(
        np.array(sommets), np.array(faces, dtype=np.int64), np.array(uvs, dtype=np.float64)
    )


def rectangle_vertical(
    origine: Tableau,
    axe_u: Tableau,
    largeur: float,
    hauteur: float,
    decalage_normal: float = 0.0,
    uv_echelle: tuple[float, float] = (1.0, 1.0),
) -> Maillage:
    """Rectangle vertical à partir de `origine` (coin bas gauche), le long de `axe_u`, avancé de
    `decalage_normal` vers l'extérieur (à droite de `axe_u`). Face vers l'extérieur."""
    u = np.array([axe_u[0], axe_u[1], 0.0])
    u /= np.linalg.norm(u)
    exterieur = np.array([u[1], -u[0], 0.0])
    o = np.asarray(origine, dtype=np.float64) + exterieur * decalage_normal
    z = np.array([0.0, 0.0, 1.0])
    sommets = np.array([o, o + u * largeur, o + u * largeur + z * hauteur, o + z * hauteur])
    uv = np.array(
        [
            (0, 0),
            (largeur / uv_echelle[0], 0),
            (largeur / uv_echelle[0], hauteur / uv_echelle[1]),
            (0, hauteur / uv_echelle[1]),
        ]
    )
    return Maillage(sommets, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64), uv)


def icosphere(subdivisions: int = 1) -> Maillage:
    """Sphère unité (icosaèdre subdivisé)."""
    t = (1 + 5**0.5) / 2
    v = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    sommets = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    faces = list(f)
    for _ in range(subdivisions):
        milieux: dict[tuple[int, int], int] = {}

        def milieu(a: int, b: int, milieux: dict[tuple[int, int], int] = milieux) -> int:
            cle = (min(a, b), max(a, b))
            if cle not in milieux:
                p = sommets[a] + sommets[b]
                sommets.append(p / np.linalg.norm(p))
                milieux[cle] = len(sommets) - 1
            return milieux[cle]

        nouvelles = []
        for a, b, c in faces:
            ab, bc, ca = milieu(a, b), milieu(b, c), milieu(c, a)
            nouvelles += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nouvelles
    s = np.array(sommets)
    uv = np.column_stack([0.5 + np.arctan2(s[:, 1], s[:, 0]) / (2 * np.pi), 0.5 + s[:, 2] / 2])
    return Maillage(s, np.array(faces, dtype=np.int64), uv)


def cylindre(cotes: int = 6) -> Maillage:
    """Cylindre unité (rayon 1, de z = 0 à z = 1), sans fonds."""
    a = np.linspace(0, 2 * np.pi, cotes, endpoint=False)
    bas = np.column_stack([np.cos(a), np.sin(a), np.zeros(cotes)])
    haut = bas + np.array([0.0, 0.0, 1.0])
    sommets = np.concatenate([bas, haut])
    faces = []
    for i in range(cotes):
        j = (i + 1) % cotes
        faces += [[i, j, cotes + j], [i, cotes + j, cotes + i]]
    uv = np.column_stack([np.concatenate([a, a]) / (2 * np.pi), np.repeat([0.0, 1.0], cotes)])
    return Maillage(sommets, np.array(faces, dtype=np.int64), uv)


def transformer(m: Maillage, echelle: Tableau, rotation_z: float, translation: Tableau) -> Maillage:
    c, s = np.cos(rotation_z), np.sin(rotation_z)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Maillage((m.sommets * echelle) @ r.T + translation, m.faces.copy(), m.uv.copy())


def retourner(m: Maillage) -> Maillage:
    """Inverse l'orientation des faces."""
    return Maillage(m.sommets, m.faces[:, [0, 2, 1]], m.uv)


def repere(m: Maillage, ex: Tableau, ey: Tableau, ez: Tableau, origine: Tableau) -> Maillage:
    """Place un maillage modélisé dans son repère propre (x, y, z) sur les axes donnés."""
    base = np.column_stack([ex, ey, ez])
    return Maillage(m.sommets @ base.T + origine, m.faces.copy(), m.uv.copy())


def plaque(contour: Tableau) -> Maillage:
    """Polygone plan (x, y) à z = 0, UV dans sa boîte englobante (0-1), face vers +z."""
    xy, faces = trianguler_anneaux([np.asarray(contour, dtype=np.float64)])
    faces = orienter_vers_le_haut(xy, faces)
    mini, maxi = xy.min(axis=0), xy.max(axis=0)
    uv = (xy - mini) / np.maximum(maxi - mini, 1e-9)
    return Maillage(np.column_stack([xy, np.zeros(len(xy))]), faces, uv)
