"""Construit la maquette dans Blender, l'exporte en GLB et produit des rendus.

blender -b --factory-startup -P blender/construire.py -- <scene> <sortie> [--sans-rendu]

Entrées (écrites par `maquette.scene`) : scene.json, maillages.npz, textures/, infos.json.
Sorties : maquette.glb, vue_aerienne.png, vue_rue.png.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
SCENE, SORTIE = Path(args[0]), Path(args[1])
RENDU = "--sans-rendu" not in args
SORTIE.mkdir(parents=True, exist_ok=True)

scene_json = json.loads((SCENE / "scene.json").read_text(encoding="utf-8"))
infos = json.loads((SCENE / "infos.json").read_text(encoding="utf-8"))
donnees = np.load(SCENE / "maillages.npz")

bpy.ops.wm.read_factory_settings(use_empty=True)


def lineaire(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def image(nom, non_couleur=False):
    img = bpy.data.images.load(str(SCENE / "textures" / nom), check_existing=True)
    if non_couleur:
        img.colorspace_settings.name = "Non-Color"
    return img


def materiau(nom, d):
    m = bpy.data.materials.new(nom)
    m.use_nodes = True
    nodes, liens = m.node_tree.nodes, m.node_tree.links
    bsdf = nodes["Principled BSDF"]
    r, g, b, a = d["couleur"]
    facteur = (lineaire(r), lineaire(g), lineaire(b), 1.0)
    if d["texture"]:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image(d["texture"])
        if facteur[:3] != (1.0, 1.0, 1.0):
            mix = nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            liens.new(tex.outputs["Color"], mix.inputs[6])
            mix.inputs[7].default_value = facteur
            liens.new(mix.outputs[2], bsdf.inputs["Base Color"])
        else:
            liens.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = facteur
    if d["orm"]:
        orm = nodes.new("ShaderNodeTexImage")
        orm.image = image(d["orm"], True)
        sep = nodes.new("ShaderNodeSeparateColor")
        liens.new(orm.outputs["Color"], sep.inputs["Color"])
        liens.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        liens.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Roughness"].default_value = d["rugosite"]
        bsdf.inputs["Metallic"].default_value = d["metal"]
    if d["normale"]:
        nor = nodes.new("ShaderNodeTexImage")
        nor.image = image(d["normale"], True)
        carte = nodes.new("ShaderNodeNormalMap")
        liens.new(nor.outputs["Color"], carte.inputs["Color"])
        liens.new(carte.outputs["Normal"], bsdf.inputs["Normal"])
    if a < 1.0:
        bsdf.inputs["Alpha"].default_value = a
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
        if hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    if d["emission"]:
        bsdf.inputs["Emission Color"].default_value = facteur
        bsdf.inputs["Emission Strength"].default_value = d["emission"]
    m.use_backface_culling = not d["double_face"]
    return m


materiaux = {n: materiau(n, d) for n, d in scene_json["materiaux"].items()}
lisses = {n for n, d in scene_json["materiaux"].items() if d["lisse"]}

collections = {}


def collection(calque):
    if calque not in collections:
        c = bpy.data.collections.new(calque)
        bpy.context.scene.collection.children.link(c)
        collections[calque] = c
    return collections[calque]


for o in scene_json["objets"]:
    sommets, faces, uvs, mats, lisse = [], [], [], [], []
    decalage = 0
    noms_mat = []
    for i, part in enumerate(o["parts"]):
        v = donnees[part["cle"] + "_v"]
        f = donnees[part["cle"] + "_f"]
        sommets.append(v)
        faces.append(f + decalage)
        uvs.append(donnees[part["cle"] + "_uv"])
        mats.append(np.full(len(f), i, dtype=np.int32))
        lisse.append(np.full(len(f), part["materiau"] in lisses))
        noms_mat.append(part["materiau"])
        decalage += len(v)
    v = np.concatenate(sommets)
    f = np.concatenate(faces)
    uv = np.concatenate(uvs)
    me = bpy.data.meshes.new(o["nom"])
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.astype(np.float32).ravel())
    me.loops.add(f.size)
    me.loops.foreach_set("vertex_index", f.astype(np.int32).ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(f), 3, dtype=np.int32))
    me.polygons.foreach_set("material_index", np.concatenate(mats))
    me.polygons.foreach_set("use_smooth", np.concatenate(lisse))
    couche = me.uv_layers.new(name="UVMap")
    couche.data.foreach_set("uv", uv[f.ravel()].astype(np.float32).ravel())
    for n in noms_mat:
        me.materials.append(materiaux[n])
    me.update()
    me.validate(clean_customdata=False)
    ob = bpy.data.objects.new(o["nom"], me)
    ob["calque"] = o["calque"]
    if o["infos"]:
        ob["infos"] = json.dumps(o["infos"], ensure_ascii=False)
    collection(o["calque"]).objects.link(ob)

# Modèles externes (architecte) en Lambert 93 : recalés sur l'origine locale.
for modele in scene_json["modeles"]:
    avant = set(bpy.data.objects)
    chemin = modele["chemin"]
    if chemin.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=chemin)
    elif chemin.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=chemin)
    elif chemin.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=chemin)
    for ob in set(bpy.data.objects) - avant:
        if ob.parent is None:
            ob.location += Vector(modele["decalage"])
        ob["calque"] = modele["calque"]
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        collection(modele["calque"]).objects.link(ob)

bpy.ops.export_scene.gltf(
    filepath=str(SORTIE / "maquette.glb"),
    export_format="GLB",
    export_extras=True,
    export_yup=True,
    export_cameras=False,
    export_lights=False,
    export_image_format="AUTO",
    export_apply=False,
)
print(f"GLB : {(SORTIE / 'maquette.glb').stat().st_size / 1e6:.1f} Mo")

if RENDU:
    scene = bpy.context.scene
    monde = bpy.data.worlds.new("ciel")
    monde.use_nodes = True
    ciel = monde.node_tree.nodes.new("ShaderNodeTexSky")
    ciel.sky_type = "NISHITA"
    ciel.sun_disc = False
    ciel.sun_elevation = math.radians(38)
    ciel.sun_rotation = math.radians(200)
    monde.node_tree.links.new(
        ciel.outputs["Color"], monde.node_tree.nodes["Background"].inputs["Color"]
    )
    monde.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.22
    scene.world = monde
    soleil = bpy.data.objects.new("soleil", bpy.data.lights.new("soleil", "SUN"))
    soleil.data.energy = 3.2
    soleil.data.angle = math.radians(1.5)
    soleil.rotation_euler = (math.radians(52), 0, math.radians(200 - 180 + 90))
    scene.collection.objects.link(soleil)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.6

    cx, cy, cz = infos["centre_site"]
    cible = bpy.data.objects.new("cible", None)
    cible.location = (cx, cy, cz + 3)
    scene.collection.objects.link(cible)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    contrainte = camera.constraints.new("TRACK_TO")
    contrainte.target = cible
    scene.camera = camera

    def rendre(nom, position, focale):
        camera.location = position
        camera.data.lens = focale
        scene.render.filepath = str(SORTIE / f"{nom}.png")
        bpy.ops.render.render(write_still=True)

    if "projet" in collections and infos["presentation"] == "existant":
        collections["projet"].hide_render = True
    if "existant" in collections and infos["presentation"] == "projet":
        collections["existant"].hide_render = True
    rendre("vue_aerienne", (cx - 85, cy - 110, cz + 95), 40)
    if infos.get("vue_rue"):
        for ob in collections.get("limites", bpy.data.collections.new("vide")).objects:
            if any(m and m.name.startswith("voile") for m in ob.data.materials):
                ob.hide_render = True
        cible.location = infos["vue_rue"]["cible"]
        rendre("vue_rue", infos["vue_rue"]["position"], 20)
