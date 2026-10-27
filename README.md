# Maquettes 3D des sites étudiés

Maquette 3D d'un site foncier et de son quartier, à partir de l'open data de l'IGN, générée
avec Blender et consultable dans le navigateur : on tourne autour du site, on s'y déplace en
caméra libre ou à hauteur de piéton, on règle le soleil, on mesure.

## Ce que contient une maquette

| Élément | Source | Mesuré ou indicatif |
|---|---|---|
| Relief | MNT LiDAR HD (50 cm) | mesuré |
| Bâtiments : emprise | BD TOPO | mesuré |
| Bâtiments : hauteurs, forme des toits (plat, appentis, deux pans, croupe) | MNS LiDAR HD | mesuré |
| Toits du quartier | orthophoto 20 cm | photographie réelle |
| Toits du site : couverture (tuiles, ardoise, bac acier, terrasse) | couleur de l'orthophoto | déduit |
| Façades : matériau | BD TOPO (`materiaux_des_murs`) | déduit quand il est connu |
| Fenêtres, portes, volets, balcons | nombre d'étages | **indicatif** |
| Arbres : position, hauteur, houppier | MNS − MNT | mesuré, forme stylisée |
| Murs de clôture et haies | MNS − MNT le long des limites + orthophoto | mesuré |
| Chaussées, marquage | BD TOPO | largeur BD TOPO, marquage simplifié |
| Trottoirs | domaine non cadastré | déduit |
| Projet | fiche du site (volumes) ou modèle de l'architecte | selon la fiche |

Les bâtiments du site ont des ouvertures en 3D (tableaux, vitrages, menuiseries, appuis, volets,
portes) ; ceux du quartier ont des façades texturées. Les murs mitoyens sont aveugles.

## Utilisation

```bash
uv sync
uv run maquette installer-blender                     # une fois : Blender 4.5 LTS dans outils/
uv run maquette site sites/vernouillet-aristide-briand.yaml   # ~30 s, + ~4 min de rendus
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

1. `scene.py` télécharge les données (cache `data/<site>/brut/`, `--force` pour refaire) et
   calcule la géométrie : `terrain.py`, `batiments.py` + `toits.py`, `voirie.py`,
   `clotures.py`, `vegetation.py`, textures procédurales `textures.py`.
2. `blender/construire.py` assemble la scène dans Blender (matériaux PBR), exporte le GLB et
   rend deux vues avec Cycles.
3. `visionneuse.py` compresse le GLB (gltfpack, meshopt) et écrit la page three.js.

Contrôles : `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`.
