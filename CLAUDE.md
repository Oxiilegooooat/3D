# CLAUDE.md — maquettes 3D

Maquettes 3D des sites étudiés par le développeur foncier (prospection TraqueSoc, CRM foncier).
Voir `README.md` pour l'usage et `docs/sources.md` pour les sources.

## Règles

1. **Open data uniquement**, sans clé : Géoplateforme IGN, BAN, OpenStreetMap (API principale,
   une requête par site). Toute nouvelle source est documentée dans `docs/sources.md` avant
   d'écrire son accès.
2. **Dépôt indépendant** : aucun import de TraqueSoc ni du CRM. L'échange passe par des
   fichiers (GLB, PNG, page HTML).
3. **Mesuré ou indicatif, et on le dit** : ce qui ne vient pas d'une mesure (fenêtres, balcons,
   couleurs de façade, volume de projet d'étude) est signalé dans `AVERTISSEMENTS`
   (`scene.py`) et dans la visionneuse. On n'invente pas une hauteur : LiDAR, sinon BD TOPO,
   sinon étages × 3 m, sinon défaut, avec `hauteur_source`.
4. **Paramètres dans `config/maquette.yaml`**, jamais en dur (hauteur d'étage 3 m comme
   TraqueSoc, travées, débords, seuils de détection).
5. **Périmètre** : le site, ses parcelles contiguës et celles d'en face, avec les voies entre
   elles (`perimetre.py`, `config/maquette.yaml`). Rien n'est modélisé au-delà : le reste se
   traite en perspectives.
6. **Aucune donnée brute dans git** : `data/`, `exports/`, `outils/` sont ignorés. Les tests
   sont synthétiques et hors réseau. Pas de donnée personnelle (ni propriétaire, ni dirigeant).

## Conventions

- Langue : français (code métier, docstrings, messages, commits `maquette: …` à l'impératif).
- Repère local : X est, Y nord, Z haut, en mètres, origine au centre des parcelles du site
  (`origine_lambert93` dans `scene.json`). three.js : `V(x, y, z) = (x, z, -y)`.
- Anneaux extérieurs dans le sens trigonométrique, trous dans le sens horaire : un mur décrit
  dans le sens (s, z) trigonométrique regarde à droite de son arête (`quad_mur`).
- Toits : modèle paramétrique (`toits.ajuster`) dont les plis (faîtage, arêtiers) sont imposés
  au maillage et insérés dans le contour des murs ; toit composé : meilleur modèle simple,
  sinon toit plat. Teinte de couverture ramenée à une couleur plausible par famille.
- Matériaux : nommés par famille (`facade_enduit_<v>`, `facade_<revetement>`,
  `toit_<type>_<teinte>`, `sol_<classe>`, `porte_<n>`, `volet_<n>`…), définis dans
  `textures.catalogue`.
- Calques (extras glTF `calque`) : terrain, voirie, contexte, existant, projet, vegetation,
  limites. Le sol reste continu sous tous les bâtiments (un calque masqué ne laisse pas de trou).
- Surfaces drapées (sol, voirie) : `maillage.trianguler_robuste` (grille + triangulation de
  GEOS) ; `triangle` est réservé aux géométries maîtrisées (toits, murs percés), il plante sur
  des polygones à trous qui se touchent.
- Un nœud par bâtiment du site, avec `infos` (JSON) pour la visionneuse ; voisins regroupés.
- L'infrarouge n'est pas étalonné : le seuil végétal / minéral est calculé par zone (Otsu).
- Textures photographiées (`bibliotheque.py`, `config/materiaux.yaml`) : une image par texture,
  recolorée une fois vers la teinte la plus claire de son entrée ; les variantes l'assombrissent
  par le facteur de couleur (glTF le borne à 1). Échelle réelle par `KHR_texture_transform`.
  Budget : GLB < 14 Mo, sinon gltfpack passe en meshopt (décodeur WebAssembly).
- Typage strict (`mypy`), `ruff` (100 colonnes), un test synthétique par transformation.
