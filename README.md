# Maquettes 3D des sites étudiés

Maquette 3D d'un site foncier et de son quartier, à partir de l'open data de l'IGN, générée
avec Blender et consultable dans le navigateur : on tourne autour du site, on s'y déplace en
caméra libre ou à hauteur de piéton, on règle le soleil, on mesure.

## Règle de périmètre

On modélise **le site, ses parcelles contiguës et celles d'en face** (de l'autre côté de la
voie), avec les rues entre elles. Rien au-delà : le grand paysage relève des perspectives.
La maquette est posée sur un socle découpé selon cette zone. Les seuils sont dans
`config/maquette.yaml` (`perimetre`).

## Ce que contient une maquette

| Élément | Source | Mesuré ou indicatif |
|---|---|---|
| Relief | MNT LiDAR HD (50 cm) | mesuré |
| Bâtiments : emprise | BD TOPO | mesuré |
| Bâtiments : hauteurs, toits (plat, appentis, deux pans, croupe), débords | MNS LiDAR HD | mesuré |
| Couverture (tuiles, ardoise, bac acier, terrasse) et sa clarté | orthophoto | déduit |
| Façades : matériau | BD TOPO (`materiaux_des_murs`) | déduit quand il est connu |
| Fenêtres, portes, volets, balcons (en 3D) | nombre d'étages | **indicatif** |
| Sol : pelouse, massifs, gravier, terre, enrobé, dalles, piscines | orthophoto + infrarouge + LiDAR | classé ; aspect des matériaux **indicatif** |
| Arbres, massifs et haies (hauteur) | MNS − MNT | mesuré, forme stylisée |
| Petites constructions hors cadastre (abris, vérandas) | LiDAR + infrarouge | mesuré à la date du survol |
| Véhicules | LiDAR | mesurés à la date du survol, modèle stylisé |
| Murs pleins, murets, haies sur les limites | LiDAR + orthophoto | mesuré |
| Grilles sur muret, portails, grillages entre parcelles | ouvertures et limites | **indicatif** |
| Chaussées, trottoirs, bordures, marquage axial | BD TOPO, domaine non cadastré | largeur BD TOPO, trottoirs déduits |
| Passages piétons, stops, arrêts de bus, feux, bancs | OpenStreetMap | selon OSM |
| Lampadaires | OpenStreetMap, sinon tous les 25 m | **indicatif** sans OSM |
| Projet | fiche du site (volumes) ou modèle de l'architecte | selon la fiche |

Les murs mitoyens sont aveugles. L'orthophoto n'est pas plaquée sur le sol (floue de près) :
elle sert à classer le sol, rendu en matériaux nets.

## Utilisation

```bash
uv sync
uv run maquette installer-blender                     # une fois : Blender 4.5 LTS dans outils/
uv run maquette site sites/vernouillet-aristide-briand.yaml   # ~30 s, + ~5 min de rendus
uv run maquette site sites/<site>.yaml --no-rendu     # sans les images Cycles
```

Résultat dans `exports/<site>/` : `index.html` (visionneuse), `maquette.glb`, `vue_aerienne.png`,
`vue_rue.png`. La page se sert de la maquette à côté d'elle : il faut la publier par HTTP
(un `file://` ne charge pas le GLB). `?qualite=basse` retire les ombres (téléphones modestes).

### Fiche d'un site

```yaml
nom: Vernouillet — 54 rue Aristide Briand
adresse: 54 rue Aristide Briand, 78540 Vernouillet
insee: "78643"
parcelles: [AE0003, AO0404, AO0406]
presentation: existant        # existant | projet | les-deux (calque affiché à l'ouverture)
projet:                       # facultatif
  nom: Programme R+2+attique
  indicatif: false            # true : volume d'étude, signalé comme tel dans la visionneuse
  demolition: true            # le bâti existant du site disparaît en mode « projet »
  volumes:
    - emprise: [[x, y], ...]  # Lambert 93
      niveaux: 4
      attique: true           # dernier niveau en retrait de 2 m
      toiture: terrasse       # terrasse | deux_pans
      balcons: true
  modele: projet.glb          # ou un modèle d'architecte (GLB, OBJ, FBX) en Lambert 93
```

## Chaîne

1. `scene.py` télécharge les données (cache `data/<site>/brut/`, `--force` pour refaire),
   arrête la zone (`perimetre.py`) et calcule la géométrie : `terrain.py`, `batiments.py` +
   `toits.py`, `sol.py`, `voirie.py`, `clotures.py`, `detections.py`, `mobilier.py`,
   `vegetation.py`, textures procédurales `textures.py`.
2. `blender/construire.py` assemble la scène dans Blender (matériaux PBR), exporte le GLB et
   rend deux vues avec Cycles.
3. `visionneuse.py` compresse le GLB (gltfpack, meshopt) et écrit la page three.js.

Contrôles : `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`.
