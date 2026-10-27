# Sources

Open data uniquement, sans clé ni compte. Toutes les données IGN passent par la
Géoplateforme (`data.geopf.fr`), sous Licence ouverte Etalab 2.0.

| Donnée | Service | Format | Couverture | Usage dans la maquette |
|---|---|---|---|---|
| Parcelles | WFS `CADASTRALPARCELS.PARCELLAIRE_EXPRESS:parcelle` | GeoJSON (Lambert 93) | France | Site (par `idu`), limites, domaine non cadastré (trottoirs) |
| Bâtiments | WFS `BDTOPO_V3:batiment` | GeoJSON | France | Emprises, usage, étages, logements, matériau des murs |
| Routes | WFS `BDTOPO_V3:troncon_de_route` | GeoJSON | France | Chaussées (largeur, nature, sens), marquage axial |
| Eau | WFS `BDTOPO_V3:surface_hydrographique` | GeoJSON | France | Surfaces en eau |
| Orthophoto | WMS raster `HR.ORTHOIMAGERY.ORTHOPHOTOS` | JPEG, 10 cm/px demandés (20 cm natifs) | France | Texture du sol et des toits du contexte, teinte des toits du site |
| MNT LiDAR HD | WMS raster `IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93` | BIL float32, 50 cm | Zones survolées (quasi-totalité de la métropole en 2026) | Relief, pied des murs |
| MNS LiDAR HD | WMS raster `IGNF_LIDAR-HD_MNS_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93` | BIL float32, 50 cm | Idem | Hauteurs et formes des toits, arbres, clôtures |
| Adresse | `data.geopf.fr/geocodage/search` (Base adresse nationale) | JSON | France | Étiquette et point de vue depuis la rue |

Points d'attention :

- Le service raster accepte 5 010 px par côté : au-delà, les images sont tuilées.
- Hors couverture LiDAR HD, le MNT et le MNS manquent : la génération s'arrête avec un message.
- La hauteur de la BD TOPO est souvent interpolée (« Interpolation bâti BDTopo ») : le LiDAR
  prime (exemple : hangar de Vernouillet, 5,9 m en BD TOPO, 7,6 m à l'égout et 10,5 m au faîtage
  mesurés).
- Panoramax (photos de rue, licence ouverte) n'a aucune image autour du site de Vernouillet
  (octobre 2026) : les façades restent indicatives.

Logiciels : Blender 4.5 LTS (GPL, téléchargé dans `outils/`), three.js 0.170 (MIT, CDN jsDelivr),
gltfpack 0.24 (MIT, via `npx`).
