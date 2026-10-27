"""Règle de périmètre : on modélise le site, ses parcelles contiguës et celles d'en face.

- **contiguë** : à moins de `contigu` mètres du site (les contours du cadastre ne se touchent
  pas toujours exactement) ;
- **en face** : de l'autre côté d'une voie, à moins de `en_face` mètres, et le plus court
  segment qui la relie au site passe surtout dans le domaine non cadastré (la voie) ;
- **voirie** : le domaine non cadastré compris dans l'enveloppe de ces parcelles, élargie de
  `marge_voirie`.

Au-delà, rien n'est modélisé : le reste relève des perspectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import shapely
import shapely.ops
from shapely.geometry import LineString, Polygon

from maquette.maillage import polygones


@dataclass
class Perimetre:
    site: shapely.Geometry
    voisines: dict[str, Polygon] = field(default_factory=dict)  # idu → parcelle
    en_face: dict[str, Polygon] = field(default_factory=dict)
    voirie: shapely.Geometry = field(default_factory=Polygon)
    zone: Polygon = field(default_factory=Polygon)

    @property
    def parcelles(self) -> shapely.Geometry:
        """Parcelles modélisées (site compris)."""
        return shapely.union_all([self.site, *self.voisines.values(), *self.en_face.values()])


def calculer(
    site: shapely.Geometry, parcelles: dict[str, Polygon], idus_site: set[str], cfg: dict[str, Any]
) -> Perimetre:
    cp = cfg["perimetre"]
    toutes = shapely.union_all(list(parcelles.values()))
    enveloppe = toutes.envelope.buffer(cp["en_face"] + 5)
    domaine_public = enveloppe.difference(toutes)
    # Intérieur strict : un segment qui longe une limite de parcelle ne traverse pas la voie.
    voie = domaine_public.buffer(-0.05)
    perimetre = Perimetre(site)
    for idu, p in parcelles.items():
        if idu in idus_site:
            continue
        d = p.distance(site)
        if d <= cp["contigu"]:
            perimetre.voisines[idu] = p
        elif d <= cp["en_face"]:
            a, b = shapely.ops.nearest_points(site, p)
            segment = LineString([a, b])
            if segment.length > 0 and (
                segment.intersection(voie).length >= cp["part_voie"] * segment.length
            ):
                perimetre.en_face[idu] = p
    retenues = perimetre.parcelles
    perimetre.voirie = domaine_public.intersection(
        retenues.convex_hull.buffer(cp["marge_voirie"])
    ).intersection(retenues.buffer(cp["en_face"]))
    zone = shapely.union_all([retenues, perimetre.voirie]).buffer(0.05).buffer(-0.05)
    morceaux = polygones(zone)
    perimetre.zone = max(morceaux, key=lambda g: g.area).simplify(0.05)
    return perimetre
