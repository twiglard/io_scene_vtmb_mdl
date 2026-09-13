"""Blender addon: import Vampire: The Masquerade - Bloodlines models (MDL v2531).

Run selfcheck.py under plain Python for the format check; it needs no Blender.
"""

bl_info = {
    "name": "VTMB Model (MDL v2531)",
    "author": "Claude Opus 5 xhigh / Twiglard",
    "blender": (3, 0, 0),
    "location": "File > Import > VTMB Model (.mdl), File > Export > VTMB Model (.mdl), "
                "File > Export > VTMB Model, no donor (.mdl), Add > VTMB > Bone set",
    # bl_info has no "website"; doc_url is the key that becomes a button.
    "doc_url": "https://rpgcodex.net/",
    "description": "Import Bloodlines skeletons, meshes, UVs, weights, textures and "
                   "animations including the chained models. Export rebuilds a .mdl "
                   "from its own decoded records: animations from Blender's poses, "
                   "and optionally vertex positions, normals, UVs and weights. A "
                   "second exporter authors a .mdl and .dx80.vtx from the scene with "
                   "no donor file at all",
    "version": (0, 2, 49),
    "category": "Import-Export",
}

from . import (checksum, mdl, mdl_build, mdl_rebuild, mdl_write, mesh_write,
               normal_table, paths,
               relocs, sections, tth, vmt_write, vpk, vtx, vtx_rebuild, vtx_write,
               bone_templates,
               blender_import, blender_export, blender_scratch, blender_templates,
               blender_panel)

import importlib
import os
import traceback

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper

# In the operator title, because the File menu entries pass their own text and a stale
# addon is otherwise indistinguishable from a bug.
VERSION = ".".join(str(x) for x in bl_info["version"])

ACTIVE, ALL, NONE = "~active", "~all", "~none"
SPLIT = 0.34
ROOT_MOTION_LABELS = {"keep": "the file's own blocks",
                      "extract": "the engine carries it",
                      "none": "the skeleton moves",
                      "in_place": "nothing moves"}


def _pair(box, key, value, icon="NONE"):
    row = box.split(factor=SPLIT)
    row.label(text=key)
    row.label(text=value, icon=icon)


def _mode_tally(actions):
    """How the actions of a whole-scene write split across the four modes."""
    got = {}
    for act in actions:
        mode = blender_export.root_motion_mode(
            act, blender_export.PER_ACTION, donor=False, default="extract")
        got[mode] = got.get(mode, 0) + 1
    return ", ".join("%d %s" % (got[k], ROOT_MOTION_LABELS[k])
                     for k in blender_export.ROOT_MOTION_MODES if k in got) or "no actions"


def _write_vmts(context, arm_obj, dest, names, enabled):
    """The `.vmt` for each material the export just added, and what to say about it.

    The directory is the first the armature carries, because a name resolves against the
    whole list as a cross product and the first is the only one a writer can pick without
    guessing. With no directory at all there is nowhere to put the file and the operator
    says so instead of inventing `models/`.
    """
    dirs = blender_scratch.scene_cdtextures(arm_obj)
    where = os.path.join(paths.content_root(dest) or os.path.dirname(dest),
                         paths.MATERIALS_DIR)
    rel = [os.path.join(dirs[0] if dirs else "", n + ".vmt") for n in names]
    if not dirs:
        return "; no material directory, so nothing says where their .vmt go"
    if not enabled:
        return "; write %s to finish them" % ", ".join(
            (paths.MATERIALS_DIR + "/" + r).replace("\\", "/") for r in rel)
    wrote, kept = [], []
    for name, r in zip(names, rel):
        path = os.path.join(where, r)
        text = vmt_write.vmt_text(vmt_write.texture_path(dirs[0], name))
        (wrote if vmt_write.write_vmt(path, text) else kept).append(r)
    out = ""
    if wrote:
        out += "; wrote %d .vmt under %s" % (len(wrote), where)
    if kept:
        out += "; %d .vmt already existed and were left alone" % len(kept)
    return out


def _base_image(mat):
    """The image a `$basetexture` means: the one feeding Base Color, else the only one.

    Two image textures with neither wired to Base Color are not ranked -- a `.vmt` names
    one basetexture and nothing in the material says which of them it is.
    """
    tree = getattr(mat, "node_tree", None)
    if tree is None:
        return None
    have = [n for n in tree.nodes if n.type == "TEX_IMAGE" and n.image]
    if not have:
        return None
    for node in tree.nodes:
        sock = node.inputs.get("Base Color") if hasattr(node, "inputs") else None
        for link in (sock.links if sock is not None else ()):
            if link.from_node in have:
                return link.from_node.image
    return have[0].image if len(have) == 1 else None


def _image_rgba(img):
    """(w, h, top-down RGBA bytes) for an image with pixels, else None.

    `Image.pixels` is bottom-up, and on a byte buffer it is the file's own bytes over 255
    with no colour transform -- 0 of 48 channels move over a round trip through it. A
    float buffer holds scene-linear instead, so it takes the sRGB transfer function that
    a .tth's bytes are encoded with.
    """
    w, h = int(img.size[0]), int(img.size[1])
    if not (img.has_data and w > 0 and h > 0):
        return None
    ch = int(img.channels)
    if ch not in (1, 3, 4):
        return None
    try:
        import numpy as np

        a = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(h, w, ch)[::-1]
        out = np.ones((h, w, 4), dtype=np.float32)
        if ch == 1:
            out[:, :, 0] = out[:, :, 1] = out[:, :, 2] = a[:, :, 0]
        else:
            out[:, :, :3] = a[:, :, :3]
            if ch == 4:
                out[:, :, 3] = a[:, :, 3]
        if img.is_float and not img.colorspace_settings.is_data:
            rgb = out[:, :, :3]
            out[:, :, :3] = np.where(
                rgb <= 0.0031308, rgb * 12.92,
                1.055 * np.power(np.maximum(rgb, 0.0031308), 1.0 / 2.4) - 0.055)
        return w, h, (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tobytes()
    except (ImportError, RuntimeError, ValueError, TypeError):
        return None


def _write_tths(context, arm_obj, dest, names, enabled, fmt="auto"):
    """The `.tth`/`.ttz` pair for each material the export just added.

    Same directory and same stem as the `.vmt` beside it, because that file's
    `$basetexture` names exactly this path and nothing else in the `.mdl` says where an
    image is. The pixels come from the material's own Image Texture node.
    """
    dirs = blender_scratch.scene_cdtextures(arm_obj)
    if not dirs:
        return ""
    where = os.path.join(paths.content_root(dest) or os.path.dirname(dest),
                         paths.MATERIALS_DIR)
    got, blank = [], []
    for name in names:
        mat = bpy.data.materials.get(name)
        img = _base_image(mat) if mat is not None else None
        px = _image_rgba(img) if img is not None else None
        if px is None:
            blank.append(name)
        else:
            got.append((name, px))
    if not enabled:
        return ("; %d of them carr%s an image, which \"Write .tth/.ttz\" writes beside "
                "the .vmt" % (len(got), "ies" if len(got) == 1 else "y")) if got else ""
    wrote, kept = [], []
    for name, (w, h, rgba) in got:
        stem = os.path.join(where, dirs[0], name)
        want = {"auto": None, "dxt1": tth.DXT1, "dxt5": tth.DXT5,
                "bgra": tth.BGRA8888}[fmt]
        (wrote if tth.write_pair(stem, w, h, rgba, fmt=want) else kept).append(name)
    out = ""
    if wrote:
        out += "; wrote %d .tth/.ttz under %s" % (len(wrote), where)
    if kept:
        out += "; %d pair%s already existed and %s left alone" % (
            len(kept), "" if len(kept) == 1 else "s",
            "was" if len(kept) == 1 else "were")
    if blank:
        out += ("; %d material%s named no image this export could read pixels from: %s"
                % (len(blank), "" if len(blank) == 1 else "s", ", ".join(blank[:3])))
    return out


def _report_unkeepable(op, names):
    """Actions appended under "the file's own blocks", which an appended one has none of.

    Forced, not stamped: `per_action` turns a stored "keep" into "extract" wherever there
    is no donor animation behind the action, so this is only reachable by naming `keep` in
    the dialog, and the travel the source file carried is then written nowhere.
    """
    if not names:
        return
    op.report({"WARNING"},
              "%d action%s appended as %s new animation%s under \"the file's own "
              "blocks\", and a file has none for an animation it does not have, so the "
              "root motion stayed in the keys: %s. \"Each action's own\" or \"the "
              "engine carries it\" fits one from the keys instead"
              % (len(names), "" if len(names) == 1 else "s",
                 "a" if len(names) == 1 else "", "" if len(names) == 1 else "s",
                 ", ".join(names[:3])
                 + ("" if len(names) <= 3 else " and %d more" % (len(names) - 3))))


def _report_unfitted(op, names):
    """Actions asking the engine to carry them whose keys do not travel.

    An import with Root motion off never puts the travel in the scene, so nothing can fit
    a block from it and the animation loses the one it came with.
    """
    if not names:
        return
    op.report({"WARNING"},
              "%d action%s ask the engine to carry the character, and their keys do not "
              "travel, so no movement block was written: %s. That is what an import with "
              "Root motion off leaves behind -- the travel never reached the scene"
              % (len(names), "" if len(names) == 1 else "s", ", ".join(names[:3])
                 + ("" if len(names) <= 3 else " and %d more" % (len(names) - 3))))


def _section(lay, idname, title, icon="NONE"):
    """A real collapsible header where Blender offers one, else a captioned box.

    `UILayout.panel` is 4.1 and later and returns a None body when collapsed, so callers
    have to skip their contents rather than draw into it.
    """
    try:
        header, body = lay.panel(idname, default_closed=False)
        header.label(text=title, icon=icon)
        return body
    except (AttributeError, TypeError):
        box = lay.box()
        box.label(text=title, icon=icon)
        return box


# Blender frees enum item strings it does not own, so the list a dynamic callback returns
# has to stay referenced here or the dropdown shows garbage.
_ANIM_ITEMS = []
_DROP_ITEMS = []
_ANIM_CACHE = {}
_MDL_CACHE = {}


def _cached_mdl(path):
    """One parsed Mdl, so the panel can count meshes on every redraw."""
    key = (path, os.path.getmtime(path))
    if key not in _MDL_CACHE:
        _MDL_CACHE.clear()
        _MDL_CACHE[key] = mdl.Mdl(path)
    return _MDL_CACHE[key]


def _base_path(op, context):
    """The .mdl the active action belongs to, which is the one being rewritten.

    Never the save path: importing with the chain on gives actions from about 29 files at
    once, so the file is a property of the action and not of where the result is put.
    """
    if getattr(op, "source", ""):
        return bpy.path.abspath(op.source)
    obj = context.active_object
    ad = obj.animation_data if obj else None
    return (ad.action.get("vtmb_source") if ad and ad.action else None) or ""


def _restamp_sequences(obj, written, src, added, dropped_seqs):
    """Write `vtmb_sequences` again when the export moved how many sequences the file has.

    `apply_sequences` matches the stash to the file by position, so an append leaves the
    new sequence in the file and in no panel -- its events, pose parameters and record
    tail all resolve through the stash -- while a delete that took sequences leaves a
    stash claiming more than are there, which every later export refuses outright.
    `sequence_stash` states the rule and this is the whole of it.

    Only when the file just written is the one the stash describes: the stash belongs to
    the armature's own model, and an export to another path leaves that model alone.
    Returns how many sequences moved, and 0 when nothing was written.
    """
    moved = len(dropped_seqs) + len(added)
    if not moved or not blender_export.same_file(written, src):
        return 0
    obj["vtmb_sequences"] = blender_import.sequence_stash(mdl.Mdl(written))
    return moved


def _base_anims(path):
    if not path or not os.path.exists(path):
        return []
    key = (path, os.path.getmtime(path))
    if key not in _ANIM_CACHE:
        _ANIM_CACHE.clear()
        try:
            _ANIM_CACHE[key] = [(a.name, a.numframes) for a in mdl.Mdl(path).anims]
        except Exception:
            _ANIM_CACHE[key] = []
    return _ANIM_CACHE[key]


def _mesh_fields(op):
    return tuple(f for f, p in (("positions", "write_positions"),
                                ("normals", "write_normals"),
                                ("uvs", "write_uvs"),
                                ("weights", "write_weights"))
                 if getattr(op, p))


def _mesh_status(op, context, base, fields):
    """(models the scene supplies, models the file has, fields it cannot store)."""
    try:
        m = _cached_mdl(base)
    except Exception:
        return 0, 0, []
    no = set()
    for _bi, _mi, _bp, mo in mesh_write.models_of(m):
        no |= set(mesh_write.supported(mo.filetype, fields)[1])
    return (len(blender_export.mesh_objects(m, base)),
            len(mesh_write.models_of(m)), sorted(no))


def _drop_takes(op, context, base):
    """Sequence labels a delete of `op.drop` would take with it, or None if the file has
    no such animation. The refusal itself is `mdl_build.remove_animation`'s; this only
    says what the user is about to lose."""
    try:
        m = _cached_mdl(base)
    except Exception:
        return None
    names = [a.name for a in m.anims]
    if op.drop not in names:
        return None
    i = names.index(op.drop)
    return [s.label for s in m.seqs
            if i in [x for col in s.blends for x in col]]


def _surplus_bones(op, context, base):
    """Bones the armature has and the file does not, for the dialog. [] on any error:
    the draw runs on every redraw and a bad path is the Model section's to report."""
    obj = context.active_object
    if not base or obj is None or obj.type != "ARMATURE":
        return []
    try:
        return blender_export.surplus_bones(_cached_mdl(base), obj)
    except Exception:
        return []


def _mesh_verts(base):
    """Vertices the file's models declare. 0 for the animation libraries and null.mdl."""
    try:
        return sum(mo.numvertices
                   for _bi, _mi, _bp, mo in mesh_write.models_of(_cached_mdl(base)))
    except Exception:
        return 0


def _donor_cdtextures(base):
    """The directory list the file being rewritten already carries."""
    try:
        return list(_cached_mdl(base).material_paths)
    except Exception:
        return []


def _auto_name(op, context):
    """Which animation the active-action entry lands on, through the same resolver the
    write uses, so the dialog cannot name one slot while the write takes another."""
    obj = context.active_object
    ad = obj.animation_data if obj else None
    if not (ad and ad.action):
        return None
    anims = _base_anims(_base_path(op, context))
    i = blender_export.name_index([n for n, _ in anims], ad.action)
    return None if i is None else (i, anims[i][0], anims[i][1])


def _matches(op, context):
    """(animation, action) pairs the every-matching-action mode would write, and the
    (animation, action, kept) triples it would leave out."""
    path = _base_path(op, context)
    anims = _base_anims(path)
    hits, unwritten, _add = blender_export.match_indices([n for n, _ in anims], path,
                                                         context.active_object)
    return [(anims[i][0], a.name) for i, a in sorted(hits.items())], unwritten


def _drop_items(self, context):
    """The file's animations, plus a first entry that removes none.

    Blender takes a dynamic enum's first item as its default, so the do-nothing entry has
    to be first or opening the dialog would arm a delete.
    """
    global _DROP_ITEMS
    items = [(NONE, "Nothing", "Remove no animation. The default")]
    for i, (n, nf) in enumerate(_base_anims(_base_path(self, context))):
        items.append((n, "[%d] %s" % (i, n),
                      "Remove %s, %d frames long, and every sequence left with nothing "
                      "to play. A sequence that blends it alongside others is refused "
                      "instead, since one corner of a blend grid cannot be left empty"
                      % (n, nf)))
    _DROP_ITEMS = items
    return _DROP_ITEMS


def _target_items(self, context):
    """Blender takes a dynamic enum's first item as its default, and falls back to it
    without saying so when a remembered value is gone from the rebuilt list."""
    global _ANIM_ITEMS
    auto = _auto_name(self, context)
    # Kept short: the collapsed widget is about 30 characters wide and truncates in the
    # middle, and the line under it already names the slot.
    items = [
        (ACTIVE, "The active action", "Replace the one animation the active action came "
         "from%s. The usual case: import a model, edit one animation, export it back"
         % (", %s" % auto[1] if auto else "")),
        (ALL, "Every matching action",
         "Replace each animation that has an action of the same name. Use this after "
         "editing several animations of one file in the same Blender session"),
        (NONE, "Nothing",
         "Leave every animation alone. The skeleton, the material names and the "
         "sequence table still go from the scene into the file, as they do on every "
         "export, and Mesh below adds the vertex fields it names"),
    ]
    anims = _base_anims(_base_path(self, context))
    if anims:
        items.append(None)
        for i, (n, nf) in enumerate(anims):
            cur = bool(auto) and auto[0] == i
            items.append((n, "[%d] %s%s" % (i, n, " (*)" if cur else ""),
                          "Replace %s, %d frames long, with the active action. %s"
                          % (n, nf,
                             "(*) is where the active action came from, so this writes "
                             "back to its own slot" if cur else
                             "Use this to put a pose authored elsewhere into a slot it "
                             "did not come from")))
    _ANIM_ITEMS = items
    return _ANIM_ITEMS


class VTMB_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    game_root: bpy.props.StringProperty(
        name="Game root", subtype="DIR_PATH", default="",
        description="The dir holding Vampire\\ and the mod dirs beside it")
    mods: bpy.props.StringProperty(
        name="Mod dirs", default="Unofficial_Patch;Vampire",
        description="Searched in this order, as the engine's -game argument would. "
                    "No file on disk records the order, so it has to be stated")
    extract_root: bpy.props.StringProperty(
        name="Extracted content", subtype="DIR_PATH", default="",
        description="Extra tree holding models/ and materials/, searched last. Optional: "
                    "a plain install already gives geometry, animation and textures, "
                    "since the packs are read directly and .tth/.ttz is decoded. Point "
                    "it at your own converted art, or at an extraction like VpkContent "
                    "whose loose .tga then win over the packed originals")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "game_root")
        col.prop(self, "mods")
        col.prop(self, "extract_root")


def _prefs(context):
    addon = context.preferences.addons.get(__package__)
    p = getattr(addon, "preferences", None)
    if p is None:
        return "", (), ()
    return p.game_root, paths.split_list(p.mods), paths.split_list(p.extract_root)


class IMPORT_OT_vtmb_mdl(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.vtmb_mdl"
    bl_label = "Import VTMB Model " + VERSION
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".mdl"
    filter_glob: bpy.props.StringProperty(default="*.mdl", options={"HIDDEN"})

    use_packs: bpy.props.BoolProperty(
        name="Read game VPKs", default=True,
        description="Search the pack0NN.vpk archives under Game root as well as the loose "
                    "trees. Off restricts every lookup -- materials, textures and the "
                    "include-model chain -- to loose files, which on a stock install finds "
                    "almost nothing: a shipped model then imports untextured and a "
                    "character's animations do not resolve. Only useful against an "
                    "unpacked tree, or to prove a loose file is the one being read")
    with_mesh: bpy.props.BoolProperty(
        name="Mesh", default=True,
        description="Read the geometry: vertices, UVs, vertex groups and materials. "
                    "Faces come from the .vtx file beside the .mdl. Off leaves the "
                    "armature and its animations alone")
    with_flexes: bpy.props.BoolProperty(
        name="Flexes as shape keys", default=True,
        description="Build one shape key per flex the model names -- the facial morphs "
                    "195 of the shipped models carry. Position deltas only: a shape key "
                    "holds coordinates, so the normal delta each flex record also "
                    "carries stays in the file and out of the scene. Off leaves the "
                    "mesh with no shape keys at all")
    with_anims: bpy.props.BoolProperty(
        name="Animations", default=True,
        description="Import animations as actions")
    with_chained: bpy.props.BoolProperty(
        name="Chained models", default=True,
        description="Follow the include-model chain, which is where a character "
                    "model's animations actually live. Good for browsing what a "
                    "character can play, bad for editing one: a PC model chains to "
                    "about 1650 animations and Blender stutters for seconds at a time. "
                    "Turn it off, or pair it with a name filter, when authoring. Needs "
                    "Game root set in the addon preferences unless the chain sits under "
                    "this file's own tree")
    root_motion: bpy.props.BoolProperty(
        name="Root motion", default=True,
        description="Put back the motion studiomdl took out of the animation and left in "
                    "mstudiomovement_t, so a walk cycle crosses the scene instead of "
                    "marching in place. Off gives the bone data exactly as stored")
    anim_filter: bpy.props.StringProperty(
        name="Name filter", default="",
        description="Only import animations whose name contains this")
    max_anims: bpy.props.IntProperty(
        name="Max animations", default=0, min=0,
        description="0 imports every match; the chain of a player model holds "
                    "thousands, so pair 0 with a filter")

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(self, "use_packs")
        lay.prop(self, "with_mesh")
        sub = lay.column()
        sub.enabled = self.with_mesh
        sub.prop(self, "with_flexes")
        lay.prop(self, "with_anims")
        col = lay.column()
        col.enabled = self.with_anims
        for p in ("with_chained", "root_motion", "anim_filter", "max_anims"):
            col.prop(self, p)
        if not self.with_anims:
            lay.label(text="Animations off: the four above only pick animations",
                      icon="INFO")

    def execute(self, context):
        game_root, mods, extract = _prefs(context)
        try:
            r = blender_import.import_mdl(
                context, self.filepath, anim_filter=self.anim_filter,
                max_anims=self.max_anims,
                with_mesh=self.with_mesh, with_flexes=self.with_flexes,
                with_anims=self.with_anims, with_chained=self.with_chained,
                game_root=game_root, mods=mods, extra_roots=extract,
                use_packs=self.use_packs, root_motion=self.root_motion)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, "%s: %s -- traceback on the console"
                        % (type(exc).__name__, exc))
            return {"CANCELLED"}
        if r["warning"]:
            self.report({"WARNING"}, r["warning"])
        self.report({"INFO"}, "vtmb mdl: %d bones, %d verts, %d faces, %d animations "
                              "(%d chained), %d sequences, %d/%d materials textured"
                    % (r["bones"], r["verts"], r["faces"], r["anims"],
                       r["chained"], r["seqs"], r["textured"], r["materials"]))
        return {"FINISHED"}


class EXPORT_OT_vtmb_mdl(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.vtmb_mdl"
    bl_label = "Export VTMB Model " + VERSION
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".mdl"
    filter_glob: bpy.props.StringProperty(default="*.mdl", options={"HIDDEN"})

    source: bpy.props.StringProperty(
        name="Rewrite", subtype="FILE_PATH", default="", options={"HIDDEN"},
        description="Scripting only: rewrite this .mdl instead of the one the active "
                    "action was imported from. The armature must carry every bone it has")
    # Drawn disabled purely to hang a tooltip off the filename; a label cannot have one.
    reading: bpy.props.StringProperty(
        name="File", default="",
        description="The .mdl the active action was imported from. It is the template for "
                    "the output: its hitboxes, attachments, cloth, flex, spring bones and "
                    "every animation you are not replacing are copied across byte for "
                    "byte, and so are its mesh, skeleton, materials and sequences unless "
                    "you tick them below. Everything written comes from this scene's "
                    "armature and meshes; nothing else in the scene reaches the file")
    target: bpy.props.EnumProperty(
        name="Replace", items=_target_items,
        description="Which of that file's animations get poses from Blender. The rest of "
                    "the file changes only where Mesh below says so")
    add_new: bpy.props.BoolProperty(
        name="Add the rest as new", default=False,
        description="Append every action that matches no animation of the file, "
                    "instead of listing it under Not written and dropping it. Each "
                    "gets a sequence of the same name, because the engine reaches an "
                    "animation only through one. Nothing is renumbered -- the "
                    "appended animation goes on the end and every index already "
                    "stored points below it -- and the animations already in the file "
                    "keep their poses, since one posscale/rotscale set serves the "
                    "whole file and they are re-encoded when a new pose widens it")
    drop: bpy.props.EnumProperty(
        name="Delete", items=_drop_items,
        description="One animation to remove from the file, taken after everything "
                    "else so the slots above still mean what they said. The sequences "
                    "it leaves with nothing to play go with it -- the engine reaches an "
                    "animation only through a sequence -- and a sequence that blends it "
                    "alongside others is refused rather than left with a hole. The action "
                    "stays in the blend; delete that too if you do not want it appended "
                    "back")
    write_vmt: bpy.props.BoolProperty(
        name="Write .vmt for new materials", default=False,
        description="A material slot the file has no texture record for gets one, and "
                    "with this also a materials/<dir>/<name>.vmt beside the model, so it "
                    "resolves to something rather than the purple checkerboard. The .mdl "
                    "carries only a name and a directory list, so without the .vmt "
                    "nothing says which image the material means. An existing .vmt is "
                    "never overwritten")
    write_tth: bpy.props.BoolProperty(
        name="Write .tth/.ttz for new materials", default=False,
        description="The image on that material's Image Texture node, written as the "
                    ".tth/.ttz pair the .vmt's $basetexture names, so the material draws "
                    "that image and not the purple checkerboard. A full mip chain, in "
                    "whichever format the box below names. A material with no Image "
                    "Texture, or with several and "
                    "none of them feeding Base Color, is named in the report and skipped. "
                    "Neither half of an existing pair is overwritten. Nothing looks for "
                    "the image at all unless a .vmt is there too")
    write_2bone_vtx: bpy.props.BoolProperty(
        name="Also rewrite .dx7_2bone.vtx", default=False,
        description="Rewrite the second strip file beside the model as well as the "
                    ".dx80.vtx. The engine asks for it only under -dxlevel 70, and "
                    "leaving it alone leaves it describing the old geometry, which is "
                    "what the report warns about. It caps bones at two per triangle "
                    "where the .dx80.vtx allows nine, so it is revised from its own copy "
                    "and not derived from the other one. Nothing happens when no "
                    ".dx7_2bone.vtx sits beside the source")
    write_cloth: bpy.props.BoolProperty(
        name="Cloth from the scene", default=True,
        description="Refit each cloth object to the scene before writing: the stiffness, "
                    "slack and gravity on the VTMB cloth panel, and the rest length of "
                    "every spring against where its vertex now sits. Off leaves the "
                    "file's own cloth object alone, which after a geometry edit simulates "
                    "the garment the model used to be; either way the report says what "
                    "the scene asked for. Nothing is written where the scene and the file "
                    "already agree, so an unedited model still comes back byte for byte")
    cut_lods: bpy.props.BoolProperty(
        name="Turn LODs off", default=False,
        description="Write 1 into every numLODs field of every .vtx beside the model, so "
                    "the engine draws LOD 0 at any distance and never swaps to a coarse "
                    "mesh. Two dwords per file over the finished bytes and nothing else, "
                    "which is the edit the Unofficial Patch makes to its own four cut "
                    "models -- the LOD 1..n payload stays on disk with nothing pointing "
                    "at it, and the file keeps its length. Every flavour present is cut, "
                    "not only the .dx80.vtx, because the engine picks by -dxlevel and one "
                    "left alone still swaps. A model already at one LOD is untouched")
    write_flexes: bpy.props.BoolProperty(
        name="Shape keys as flexes", default=False,
        description="Rewrite each model's flexes from its shape keys -- the way to get a "
                    "facial morph authored in Blender into the file. Off carries the "
                    "file's own flex records through untouched, which is what makes an "
                    "unedited face model come back byte for byte. On, the scene's morph "
                    "is written entire and not as a patch: a shape key holds coordinates, "
                    "so the normal delta each flex record also carries is recomputed from "
                    "the morphed geometry and whatever the file said is gone. A model "
                    "whose vertex numbering the scene cannot reproduce is named in the "
                    "report and left alone, since the vertanim key is an index into the "
                    "mesh and nothing bounds it")
    tth_format: bpy.props.EnumProperty(
        name="Texture format", default="auto",
        items=[("auto", "Automatic",
                "DXT5 where the image uses alpha at all, DXT1 where it does not"),
               ("dxt1", "DXT1", "0.5 bytes a texel and no alpha"),
               ("dxt5", "DXT5", "1 byte a texel, alpha to about 1/16 of a step"),
               ("bgra", "BGRA8888 (uncompressed)",
                "4 bytes a texel and no compression loss -- 16 of the 83 shipped normal "
                "maps carry it")],
        description="What format the .tth/.ttz pair stores. The corpus is DXT5 x51, "
                    "BGRA8888 x16, DXT1 x13 and BGR888 x3 over the 83 normal maps that "
                    "decode, so compressed is the shipped norm and uncompressed is 4x to "
                    "8x the size")
    fit_hull: bpy.props.BoolProperty(
        name="Refit the bounding boxes", default=False,
        description="Recompute the movement hull from the mesh in the scene, and re-sweep "
                    "every sequence's cull box over the animations being exported. Off "
                    "keeps what the file shipped, which after a geometry change describes "
                    "where the vertices used to be. The model's own bounds start from "
                    "the header hull and are only widened by the playing sequence's box, "
                    "but what gets drawn is bounded by that sequence box alone, so a zero "
                    "one leaves the model visible only while its origin is on screen -- "
                    "and costs the entity the collision radius taken off the same six "
                    "floats. A sequence whose animations you are not exporting cannot be "
                    "swept and is left alone")
    model_name: bpy.props.StringProperty(
        name="Name in the file", default="",
        description="What studiohdr_t.name says this model is. Blank takes it from where "
                    "you are saving -- from this dialog only, so a script that does not "
                    "set it keeps the donor's. That is the convention 4383 of the 4445 "
                    "models follow: the path below models/, forward slashes, no models/ "
                    "prefix. A copy that keeps its donor's name lies in the one message "
                    "that matters when a .mdl and its .vtx stop pairing, since that "
                    "error prints this field and not the path actually loaded")
    cdtexture: bpy.props.StringProperty(
        name="Material dirs",
        description="Where the engine looks for the .vmt files, under materials/. Each "
                    "material name is tried in each of these in turn, and nothing binds "
                    "a directory to a particular material. Several are separated by ;. "
                    "The dialog starts this at what the armature was imported with, so "
                    "editing it in the Object Data panel is what changes it; empty "
                    "keeps the list the file already carries")
    root_motion_in_keys: bpy.props.BoolProperty(
        name="Imported with root motion", default=True,
        description="This action's keys still carry the motion an import put into them, "
                    "so it has to come back out before writing. Untick only for an "
                    "action imported with Root motion off. Leaving it ticked wrongly "
                    "writes the motion a second time on top of what the file already "
                    "stores and the animation covers twice the ground. Not the same as "
                    "the mode below: this is about the keys in the scene, that is about "
                    "the movement blocks in the file")
    root_motion: bpy.props.EnumProperty(
        name="Root motion",
        items=[("per_action", "Each action's own",
                "Every action decides for itself, from the Root motion field in the "
                "Action editor's sidebar. An import stamps that field to match the "
                "animation it came from, so this reproduces the file; an action that "
                "was never imported and was never set keeps the file's blocks. This is "
                "what a character wants -- a walk cycle is carried by the engine while a "
                "land or a charge travels on its own"),
               ("keep", "Engine carries the character, as the file has it",
                "Reuse the movement blocks the file already has, unchanged. Right "
                "whenever you did not move the animation, only changed how it looks. "
                "This is about the file's blocks; whether the keys in the scene still "
                "carry root motion is the checkbox above, and the two are set "
                "separately"),
               ("extract", "Engine carries the character",
                "Fit new movement blocks to the root bone's path in the scene, "
                "replacing whatever the file had. Use this when you changed where the "
                "animation goes, not just how it looks"),
               ("none", "Skeleton moves, engine does not",
                "Write no movement blocks and leave the motion in the keys, so the "
                "skeleton itself walks away from the origin and snaps back when the "
                "sequence loops. This is a real way to store an animation and 1344 of "
                "the game's own use it -- lands, charges and swarms, where the travel "
                "belongs to the animation rather than to the character"),
               ("in_place", "Nothing moves",
                "Write no movement blocks and take the net ground distance back out of "
                "the keys, so neither the engine nor the skeleton advances. The bob and "
                "sway stay, so a run cycle still runs -- on the spot")],
        default="per_action")
    use_range: bpy.props.BoolProperty(
        name="Scene range", default=False,
        description="Write the scene's Start and End frames rather than the action's "
                    "own first and last keyframe. Use this to export a slice of a "
                    "longer action")
    write_positions: bpy.props.BoolProperty(
        name="Positions", default=False,
        description="Write each vertex's location. Adding and removing vertices is "
                    "allowed: a mesh whose count moved is rebuilt whole and its "
                    ".dx80.vtx rewritten beside it")
    write_normals: bpy.props.BoolProperty(
        name="Normals", default=False,
        description="Write each vertex's normal, which is what the file shades with. "
                    "The format stores one per vertex and spells a hard edge by "
                    "duplicating the vertex, which the rebuild does for you")
    write_uvs: bpy.props.BoolProperty(
        name="UVs", default=False,
        description="Write the texture coordinates. The file stores one UV per vertex "
                    "and spells a seam by duplicating the vertex, so a seam cut in "
                    "Blender rebuilds the mesh rather than being refused")
    write_weights: bpy.props.BoolProperty(
        name="Weights", default=False,
        description="Write which bones move each vertex and how much, from the vertex "
                    "groups named after the bones. Four bones per vertex at most and the "
                    "largest four win when Blender has more; three weights are stored to "
                    "1/255 and the fourth is whatever they leave over")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == "ARMATURE"
                and obj.animation_data is not None
                and obj.animation_data.action is not None)

    def invoke(self, context, event):
        # A scripted run leaves `source` set, and operator properties persist, so without
        # this the next dialog silently rewrites whatever that script pinned.
        self.source = ""
        # From the armature and not from the file: the import stamps the file's own list
        # there, so they agree until the panel is used, and the panel is the edit this
        # carries through. Empty with no stamp, which keeps the donor's.
        self.cdtexture = ";".join(
            blender_scratch.scene_cdtextures(context.active_object))
        base = _base_path(self, context)
        self.reading = os.path.basename(base) or "(none)"
        self.filepath = base or self.filepath
        return super().invoke(context, event)

    def draw(self, context):
        lay = self.layout
        base = _base_path(self, context)
        act = context.active_object.animation_data.action

        box = _section(lay, "vtmb_model", "Model", icon="FILE_3D")
        if box is not None:
            if not base:
                box.label(text="the active action came from no .mdl", icon="ERROR")
            else:
                row = box.row()
                row.enabled = False
                row.prop(self, "reading", text="")
                if not self.filepath:
                    pass
                elif blender_export.same_file(base, self.filepath):
                    box.label(text="overwritten in place", icon="INFO")
                else:
                    box.label(text="saved as %s" % os.path.basename(self.filepath),
                              icon="INFO")
                box.prop(self, "write_flexes")
                box.prop(self, "write_cloth")
                box.prop(self, "cut_lods")
                box.prop(self, "write_2bone_vtx")
                box.prop(self, "fit_hull")
                row = box.split(factor=SPLIT)
                row.label(text="name in the file")
                row.prop(self, "model_name",
                         text="", placeholder=blender_scratch.embedded_name(self.filepath)
                         if self.filepath else "")
                was = _donor_cdtextures(base)
                box.prop(self, "cdtexture", placeholder=";".join(was))
                box.prop(self, "write_vmt")
                box.prop(self, "write_tth")
                row = box.column()
                row.enabled = self.write_tth
                row.prop(self, "tth_format")
                dirs = paths.cdtexture_list(self.cdtexture) if self.cdtexture else []
                if dirs and paths.engine_paths(dirs) != paths.engine_paths(was):
                    _pair(box, "the file says",
                          ";".join(was[:2]) + (" and %d more" % (len(was) - 2)
                                               if len(was) > 2 else "")
                          if was else "nothing", icon="ERROR")
                extra = _surplus_bones(self, context, base)
                if extra:
                    _pair(box, "bones not in the file",
                          "%d, appended on export" % len(extra), icon="BONE_DATA")
                    box.label(text="    " + ", ".join(extra[:3])
                              + ("" if len(extra) <= 3 else " and %d more"
                                 % (len(extra) - 3)))

        box = _section(lay, "vtmb_anims", "Animations", icon="ACTION")
        if box is not None:
            row = box.split(factor=SPLIT)
            row.label(text="Replace")
            row.prop(self, "target", text="")
            row = box.split(factor=SPLIT)
            row.label(text="Unmatched")
            row.prop(self, "add_new", text="Add the rest as new")
            row = box.split(factor=SPLIT)
            row.label(text="Delete")
            row.prop(self, "drop", text="")
            if self.drop != NONE:
                takes = _drop_takes(self, context, base)
                if takes is None:
                    box.label(text="    %s is not an animation of this file" % self.drop,
                              icon="ERROR")
                elif takes:
                    box.label(text="    and the %d sequence%s that play%s only it: %s"
                              % (len(takes), "" if len(takes) == 1 else "s",
                                 "s" if len(takes) == 1 else "", ", ".join(takes[:3]))
                              + ("" if len(takes) <= 3 else " and %d more"
                                 % (len(takes) - 3)), icon="TRASH")
                else:
                    box.label(text="    no sequence plays it, so none goes with it",
                              icon="TRASH")
            if self.target == ALL:
                if self.use_range:
                    box.label(text="    Scene range is ignored here: it applies to one "
                                   "animation at a time", icon="INFO")
                hits, unwritten = _matches(self, context)
                _pair(box, "matched", "%d of %d" % (len(hits), len(_base_anims(base))),
                      icon="NONE" if hits else "ERROR")
                for name, act_name in hits[:4]:
                    box.label(text="    " + (name if name == act_name
                                             else "%s ← %s" % (name, act_name)))
                if len(hits) > 4:
                    box.label(text="    and %d more" % (len(hits) - 4))
                adds, rest = blender_export.split_unwritten(unwritten, self.add_new)
                if adds:
                    _pair(box, "added", "%d animation%s" % (
                        len(adds), "" if len(adds) == 1 else "s"), icon="ADD")
                    for u in adds[:3]:
                        box.label(text="    %s → new animation and sequence" % u[1])
                    if len(adds) > 3:
                        box.label(text="    and %d more" % (len(adds) - 3))
                if rest:
                    _pair(box, "not written", "%d action%s" % (
                        len(rest), "" if len(rest) == 1 else "s"), icon="ERROR")
                    for u in rest[:3]:
                        box.label(text="    " + blender_export.unwritten_line(*u))
                    if len(rest) > 3:
                        box.label(text="    and %d more" % (len(rest) - 3))
            elif self.target != NONE:
                auto = _auto_name(self, context)
                if self.target == ACTIVE and not auto:
                    box.label(text="that action came from no animation", icon="ERROR")
                else:
                    lo, hi = act.frame_range
                    a, b = (context.scene.frame_start, context.scene.frame_end) \
                        if self.use_range else (int(round(lo)), int(round(hi)))
                    into = auto[1] if self.target == ACTIVE else self.target
                    _pair(box, "into", into if into == act.name
                          else "%s ← %s" % (into, act.name))
                    _pair(box, "frames", "%d-%d (%d)" % (a, b, b - a + 1))
                    box.prop(self, "use_range")

        if self.target != NONE:
            box = _section(lay, "vtmb_rootmotion", "Root motion",
                           icon="ORIENTATION_GIMBAL")
            if box is not None:
                # The mode is the action's own, edited in Object Data > VTMB animations.
                # `root_motion` survives as a property a scripted caller can still force.
                arm = context.active_object
                _pair(box, "this action's keys",
                      "carry the travel" if act.get("vtmb_root_motion")
                      else "do not carry it")
                nmv = arm.get("vtmb_movement_anims")
                if nmv is not None:
                    _pair(box, "the file's animations",
                          "%d carry a movement block" % nmv)
                box.prop(self, "root_motion_in_keys")
                try:
                    _pair(box, "this action writes", ROOT_MOTION_LABELS[
                        blender_export.root_motion_mode(act, self.root_motion)])
                except (ValueError, KeyError) as exc:
                    box.label(text=str(exc), icon="ERROR")
                if self.root_motion != blender_export.PER_ACTION:
                    box.label(text="forced by the caller, not by the action",
                              icon="INFO")
                else:
                    box.label(text="set it in Object Data > VTMB animations",
                              icon="ACTION")

        nverts = _mesh_verts(base)
        fields = _mesh_fields(self)
        box = _section(lay, "vtmb_mesh", "Mesh", icon="MESH_DATA")
        if box is not None:
            col = box.column(align=True)
            col.enabled = bool(nverts)
            for p in ("write_positions", "write_normals", "write_uvs", "write_weights"):
                col.prop(self, p)
            if not nverts:
                _pair(box, "meshes", "none: the file declares no vertices", icon="INFO")
            elif fields:
                got, total, no = _mesh_status(self, context, base, fields)
                _pair(box, "meshes", "%d of %d" % (got, total),
                      icon="NONE" if got == total else "ERROR")
                if no:
                    _pair(box, "cannot store", ", ".join(no), icon="INFO")
            else:
                _pair(box, "meshes", "left alone", icon="INFO")

        lay.label(text="Rebuilt from scratch: every offset recomputed, unreferenced "
                       "spans dropped.", icon="FILE_REFRESH")
        lay.label(text="Skeleton, materials and sequences follow the scene where it "
                       "differs from the file.", icon="OUTLINER")
        lay.label(text="Everything else is carried across unchanged.", icon="LOCKED")

    def execute(self, context):
        obj = context.active_object
        src = _base_path(self, context)
        if not src:
            self.report({"ERROR"}, "nothing to read from: this action was not imported "
                                   "from a .mdl, and the file being written does not "
                                   "exist yet, so there is no model to put it into")
            return {"CANCELLED"}
        one = self.target not in (ALL, NONE)
        frame = context.scene.frame_current
        try:
            m = mdl.Mdl(src)
            matched, unwritten, addable = blender_export.match_actions(m, src, obj)
            adds = list(addable) if self.add_new else []
            if self.target == NONE:
                actions = {}
            elif self.target == ALL:
                actions = matched
                if not actions and not adds:
                    raise ValueError("no action shares a name with an animation of %s"
                                     % os.path.basename(src))
                _taken, rest = blender_export.split_unwritten(unwritten, self.add_new)
                if rest:
                    self.report({"WARNING"},
                                blender_export.describe_unwritten(rest))
            else:
                act = obj.animation_data.action
                actions = {blender_export.resolve_target(
                    m, act, "" if self.target == ACTIVE else self.target): act}
                adds = [a for a in adds if a is not act]
            # Writing an animation and removing it in the same pass is contradictory, and
            # appending its action back is worse: the delete runs last and would leave the
            # file holding a fresh copy of what was asked to go.
            gone = "" if self.drop == NONE else self.drop
            if gone:
                names = [a.name for a in m.anims]
                if gone in names:
                    actions.pop(names.index(gone), None)
                adds = [a for a in adds if a.name != gone]
            # `vtmb_scale` is 1.0 on every imported armature -- the importer has no scale
            # option -- but the skeleton template takes one and stamps it, and that armature
            # reaches this path as soon as an action of it carries a `vtmb_source`. Without
            # this the bone positions, the poses, the accessories and the face all go in
            # unscaled while `fit_hull` alone divides.
            arm_scale = float(obj.get("vtmb_scale", 1.0) or 1.0)
            r = blender_export.export_actions(
                context, obj, src, self.filepath, actions, scale=arm_scale,
                root_motion_in_keys=self.root_motion_in_keys,
                root_motion=self.root_motion,
                mesh_fields=_mesh_fields(self), add=adds, drop=gone,
                write_flexes=self.write_flexes,
                write_cloth=self.write_cloth,
                cut_lods=self.cut_lods,
                vtx_flavours=(("dx80", "dx7_2bone") if self.write_2bone_vtx
                              else ("dx80",)),
                # Derived only for a dialog, where the field is on screen with the
                # derived value in it and the user can see and change it. A scripted call
                # runs `execute` straight and gets `""`, which keeps the donor's name --
                # otherwise every export to a new path would silently rewrite @12 and an
                # otherwise-unedited file would stop coming back byte for byte.
                model_name=(self.model_name or
                            (blender_scratch.embedded_name(self.filepath)
                             if self.options.is_invoke else "")),
                cdtexture=self.cdtexture or None,
                hull=(blender_scratch.fit_hull(
                    list(blender_export.mesh_objects(m, src).values()), arm_scale)
                    if self.fit_hull else None),
                frame_start=context.scene.frame_start if self.use_range and one else None,
                frame_end=context.scene.frame_end if self.use_range and one else None)
        except mdl_build.Refused as exc:
            self.report({"ERROR"}, "refused: %s" % exc)
            return {"CANCELLED"}
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, "%s: %s -- traceback on the console"
                        % (type(exc).__name__, exc))
            return {"CANCELLED"}
        finally:
            context.scene.frame_set(frame)
        _report_unfitted(self, r.get("unfitted"))
        _report_unkeepable(self, r.get("unkeepable"))
        mesh = r["mesh"]
        if mesh["missing"] and mesh["fields"]:
            self.report({"WARNING"}, "%d model%s of %s had no mesh in the scene and kept "
                                     "the file's own: %s"
                        % (len(mesh["missing"]), "" if len(mesh["missing"]) == 1 else "s",
                           os.path.basename(src), ", ".join(mesh["missing"][:4])))
        if mesh["normals"]:
            self.report({"WARNING"}, "%d normals came from Blender rather than the file: "
                                     "up to 0.9 degrees on a filetype-0 model, and 2.0 or "
                                     "10.5 on a quantised one, whose normal is an index "
                                     "into a table the renderer holds"
                        % mesh["normals"])
        if mesh["unsupported"]:
            self.report({"WARNING"}, "%s cannot be stored by every model of this file and "
                                     "was skipped there" % ", ".join(mesh["unsupported"]))
        for name, was, now in mesh["rebuilt"]:
            self.report({"INFO"}, "%s was rebuilt: %d vertices -> %d, and the "
                                  ".dx80.vtx rewritten with it" % (name, was, now))
        if mesh["unskinned"]:
            self.report({"WARNING"}, "%d vertices are driven by no bone -- in no vertex "
                                     "group, in none above a zero weight, or in none that "
                                     "names a bone -- and were bound to bone 0, which "
                                     "drags them wherever it goes"
                        % mesh["unskinned"])
        for name, odd in mesh.get("stray_groups") or ():
            self.report({"WARNING"}, "%s: %s name%s no bone, so %s bind%s nothing -- a "
                                     "vertex whose only group it is goes to bone 0. A "
                                     "bone name spelt wrong lands here"
                        % (name, ", ".join(repr(g) for g in odd),
                           "s" if len(odd) == 1 else "",
                           "it" if len(odd) == 1 else "they",
                           "s" if len(odd) == 1 else ""))
        if mesh.get("rebuilt_uvs"):
            self.report({"INFO"}, "%d UV%s came from Blender rather than the file, on the "
                                  "file's own vertex"
                        % (mesh["rebuilt_uvs"],
                           "" if mesh["rebuilt_uvs"] == 1 else "s"))
        if mesh.get("rebuilt_added"):
            self.report({"INFO"}, "a rebuild added %d %s, which is what the format costs "
                                  "for a seam or a hard edge"
                        % (mesh["rebuilt_added"],
                           "vertex" if mesh["rebuilt_added"] == 1 else "vertices"))
        if mesh.get("tangents"):
            self.report({"INFO"}, "%d tangent%s recomputed, on the vertices that moved "
                                  "and their triangle neighbours. Everything else keeps "
                                  "the vector the file shipped"
                        % (mesh["tangents"], "" if mesh["tangents"] == 1 else "s"))
        for name, used, spare, lost in mesh.get("uv_spare") or ():
            if spare:
                self.report({"INFO"}, "%s: the file's UVs were written from %r and %s "
                                      "ignored. The format stores one UV per vertex, so "
                                      "a second layer cannot reach the file"
                            % (name, used,
                               "%r was" % spare[0] if len(spare) == 1
                               else "%d other layers were" % len(spare)))
            if lost:
                self.report({"WARNING"},
                            "%s: the import put the file's UVs in %r, which is no longer "
                            "the mesh's first UV layer -- it was renamed or deleted. %r "
                            "was written instead" % (name, lost, used))
        if mesh["crowded"]:
            total = sum(k for _n, k in mesh["crowded"])
            self.report({"WARNING"}, "%d vertex%s carr%s a fifth vertex group, which no "
                                     "record can hold: %s. The four heaviest are written "
                                     "and the fourth bone takes what the rest held"
                        % (total, "" if total == 1 else "es",
                           "ies" if total == 1 else "y",
                           ", ".join("%s x%d" % (n, k) for n, k in mesh["crowded"][:3])
                           + ("" if len(mesh["crowded"]) <= 3
                              else " and %d more mesh%s" % (len(mesh["crowded"]) - 3,
                                   "" if len(mesh["crowded"]) == 4 else "es"))))
        if mesh["renumbered"]:
            self.report({"WARNING"}, "%d rebuilt mesh%s could not keep the file's "
                                     "own vertex numbering, so nothing it keyed by "
                                     "vertex survives"
                        % (mesh["renumbered"],
                           "" if mesh["renumbered"] == 1 else "es"))
        if mesh["rebuilt_deleted"]:
            self.report({"INFO"},
                        "%d of the file's vertices are claimed by no vertex of the scene "
                        "and were dropped. Every survivor keeps the file's own order and "
                        "the flex payloads follow the new numbering"
                        % mesh["rebuilt_deleted"])
        if mesh["flex_dropped"] or mesh["flex_emptied"]:
            self.report({"WARNING"},
                        "%d morph-target delta%s named a deleted vertex and went with "
                        "it%s" % (mesh["flex_dropped"],
                                  "" if mesh["flex_dropped"] == 1 else "s",
                                  "" if not mesh["flex_emptied"] else
                                  ", leaving %d flex%s holding none"
                                  % (mesh["flex_emptied"],
                                     "" if mesh["flex_emptied"] == 1 else "es")))
        if mesh["flexes"] or mesh["flex_records"]:
            self.report({"INFO"},
                        "%d morph target%s rebuilt from shape keys, %d vertex delta%s"
                        % (mesh["flexes"], "" if mesh["flexes"] == 1 else "s",
                           mesh["flex_records"],
                           "" if mesh["flex_records"] == 1 else "s"))
        if mesh["flex_skipped"]:
            self.report({"WARNING"},
                        "%d shape-key delta%s sat on a vertex the file has not got, so "
                        "there is no mesh-local index to write %s at"
                        % (mesh["flex_skipped"],
                           "" if mesh["flex_skipped"] == 1 else "s",
                           "it" if mesh["flex_skipped"] == 1 else "them"))
        for name, why in mesh["flex_refused"]:
            self.report({"WARNING"},
                        "%s: no shape key could be written -- %s" % (name, why))
        vtx = r.get("vtx")
        if vtx is not None:
            self.report({"INFO"}, "%s rewritten, %d bytes over %d strip groups"
                        % (os.path.basename(vtx["path"]), vtx["bytes"], vtx["groups"]))
        for x in r.get("vtx_more") or ():
            self.report({"INFO"}, "%s rewritten too, %d bytes over %d strip groups"
                        % (os.path.basename(x["path"]), x["bytes"], x["groups"]))
        for name in r["stale"]:
            self.report({"WARNING"}, "%s beside the model still describes the old "
                                     "geometry and was not rewritten" % name)
        phy = r.get("phy")
        if phy is not None:
            if phy["missing"]:
                self.report({"WARNING"},
                            "no %s beside the file just written, and the donor had one. "
                            "A model with no .phy is not hit by traces at all, which is "
                            "worse than being hit at the wrong shape -- copy it across"
                            % phy["file"])
            # A rename or a delete is the only failure that is not merely geometric: the
            # solid is looked up by name and an unmatched one is not created at all.
            for a, b in phy["renamed"]:
                self.report({"ERROR"},
                            "%s names bone %r, which this export renames to %r, so its "
                            "solid will not be found and that part of the ragdoll will "
                            "not be created. Rename it there too" % (phy["file"], a, b))
            for a in phy["removed"]:
                self.report({"ERROR"},
                            "%s names bone %r, which this export removes, so its solid "
                            "will not be found and that part of the ragdoll will not be "
                            "created" % (phy["file"], a))
            if phy["geometry"] and not phy["renamed"] and not phy["removed"]:
                self.report({"WARNING"},
                            "%s still describes the old geometry. Nothing here writes it "
                            "and the engine never compares the two, so collision stays at "
                            "the shape the donor had" % phy["file"])
        for gone in r.get("removed") or []:
            self.report({"WARNING"},
                        "%r is not in the armature and was removed from the file: "
                        "%d child%s reparented onto its parent, %d vertex slot%s "
                        "repointed, %d vertex%s left following nothing%s"
                        % (gone["name"],
                           len(gone["children"]),
                           "" if len(gone["children"]) == 1 else "ren",
                           gone["slots"], "" if gone["slots"] == 1 else "s",
                           gone["rigid"], "" if gone["rigid"] == 1 else "es",
                           ("" if not gone["rebound"] else
                            ", and %d record%s that named it now name%s its parent: %s"
                            % (len(gone["rebound"]),
                               "" if len(gone["rebound"]) == 1 else "s",
                               "s" if len(gone["rebound"]) == 1 else "",
                               ", ".join(gone["rebound"][:3])))))
        for got in r.get("added_bones") or []:
            self.report({"INFO"},
                        "%r is in the armature and was not in the file, and is now bone "
                        "%d%s. Every animation grew an entry for it holding the bind "
                        "pose, and the chain joins by name, so an including model that "
                        "has no bone of that name leaves it unmatched"
                        % (got["name"], got["index"],
                           "" if got["parent"] is None else
                           " under %r" % got["parent"]))
        blind = r["mesh"].get("blind_normals") or 0
        if blind:
            self.report({"WARNING"},
                        "%d vertex normal%s moved by more than the round trip's own error "
                        "and by less than the export can tell from it, and %s written "
                        "from the file. Move one further than 1.5e-2 per component for the "
                        "edit to reach the file"
                        % (blind, "" if blind == 1 else "s",
                           "was" if blind == 1 else "were"))
        stale = r["mesh"].get("stale_stash") or []
        if stale:
            self.report({"WARNING"},
                        "%d filetype-2 mesh%s -- %s -- came from an import older than the "
                        "packed-position fix of 2026-08-20, or from one that stamped no "
                        "version. Those scenes hold the model 255x oversized and the "
                        "stash the export compares against is wrong by the same factor, "
                        "so nothing here can measure it. Re-import to be sure"
                        % (len(stale), "" if len(stale) == 1 else "es",
                           ", ".join("%s (%s)" % (n, "unstamped" if v is None
                                                  else ".".join(str(x) for x in v))
                                     for n, _ft, v in stale[:3])
                           + ("" if len(stale) <= 3 else " and %d more" % (len(stale) - 3))))
        stamps = r["mesh"].get("stale_stamps") or []
        if stamps:
            for name, miss in stamps[:3]:
                self.report({"WARNING"},
                            "%s came from an import that stamped no %s, so this export "
                            "cannot see %s. Re-import to edit that"
                            % (name, " and no ".join(k for k, _w in miss),
                               "; nor ".join(w for _k, w in miss)))
            if len(stamps) > 3:
                self.report({"WARNING"},
                            "%d more object%s came from an import older than one of those "
                            "stamps" % (len(stamps) - 3, "" if len(stamps) == 4 else "s"))
        gaps = r["scene"].get("blind_bones") or []
        if gaps:
            self.report({"WARNING"},
                        "%d bone%s carr%s no imported rest pose -- %s -- so the only "
                        "baseline is the file's own record and a move under 1e-3 units, or "
                        "2e-3 on a quaternion entry, is not written. Re-import the model "
                        "to get the exact comparison"
                        % (len(gaps), "" if len(gaps) == 1 else "s",
                           "ies" if len(gaps) == 1 else "y",
                           ", ".join(gaps[:3])
                           + ("" if len(gaps) <= 3 else " and %d more" % (len(gaps) - 3))))
        forced = r.get("lods_forced")
        for row in r.get("lods") or []:
            if row["why"] is not None:
                self.report({"WARNING"}, "%s could not be cut to one LOD: %s"
                            % (row["file"], row["why"]))
            elif row["was"] in (None, 1):
                if not forced:
                    self.report({"INFO"}, "%s was already at one LOD" % row["file"])
            elif forced:
                self.report({"WARNING"},
                            "%s cut from %d LODs to 1: the file's own vertex numbering "
                            "was rebuilt, so every lower LOD named vertices that are no "
                            "longer there. A lower LOD is an authored decimation and "
                            "cannot be rebuilt from LOD 0. It used to swap at %s"
                            % (row["file"], row["was"],
                               ", ".join("%g" % x for x in row["dropped"])))
            else:
                self.report({"INFO"},
                            "%s cut from %d LODs to 1, %d byte%s changed. It used to swap "
                            "at %s, and that geometry is still in the file with nothing "
                            "pointing at it"
                            % (row["file"], row["was"], row["bytes"],
                               "" if row["bytes"] == 1 else "s",
                               ", ".join("%g" % x for x in row["dropped"])))
        cl = r.get("cloth")
        for e in r.get("cloth_edits") or []:
            moved = [w for w in ("stiffness" if e["sigma"] else None,
                                 "slack" if e["slack"] else None,
                                 "gravity" if e["scale"] else None) if w]
            if e["sigma_edges"]:
                moved.append("%d spring stiffness(es) off the mesh's own edges"
                             % e["sigma_edges"])
            if e["moved"]:
                moved.append("%d of the cloth's particles moved" % e["moved"])
            if e["why"] is not None:
                self.report({"WARNING"},
                            "%s: %s, so %s"
                            % (e["object"] or e["model"], e["why"],
                               ("%s stayed as the file had it" % " and ".join(moved))
                               if moved else "the file keeps the cloth object it was "
                               "compiled with, which describes the geometry as it was "
                               "then"))
                continue
            if not moved:
                continue
            if cl is None:
                self.report({"WARNING"},
                            "%s: %s and Cloth from the scene is off, so the file keeps the "
                            "cloth object it shipped with"
                            % (e["object"], " and ".join(moved)))
            else:
                self.report({"INFO"}, "%s: cloth refitted -- %s, over %d springs"
                            % (e["object"], " and ".join(moved), e["springs"]))
        for name in (cl or {}).get("flattened") or []:
            self.report({"WARNING"},
                        "%s: this cloth object's stiffness varied spring by spring and the "
                        "panel holds one number, so every spring now carries it" % name)
        if r.get("boxes") is not None:
            done, total = r["boxes"]
            self.report({"INFO"} if done == total else {"WARNING"},
                        "the hull was refitted to the scene and %d of %d sequence cull "
                        "boxes re-swept%s" % (done, total, "" if done == total else
                                              "; the rest keep the file's own, which "
                                              "describe the geometry before this edit"))
        for name, (bi, mi) in r.get("remodelled") or []:
            self.report({"INFO"}, "model %d.%d is now named %r in the file"
                        % (bi, mi, name))
        if r.get("model_name"):
            self.report({"INFO"}, "the file now calls itself %r" % r["model_name"])
        if r.get("cdtexture"):
            old, now = r["cdtexture"]
            self.report({"INFO"}, "the material directories are now %s, where the file "
                                  "had %s" % (", ".join(now), ", ".join(old) or "none"))
        if r["scene"]["accessories"]:
            self.report({"INFO"},
                        "%d attachment or hitbox record%s written from the scene's "
                        "empties%s"
                        % (r["scene"]["accessories"],
                           "" if r["scene"]["accessories"] == 1 else "s",
                           "" if r["scene"]["hitboxsets"]
                           else "; every box empty is gone, so the file now holds "
                                "numhitboxsets 0"))
        if r.get("includes"):
            self.report({"INFO"},
                        "this model chains its animations to %s"
                        % ", ".join(r["includes"]))
        renamed = r.get("renamed") or []
        if renamed:
            self.report({"INFO"},
                        "%d bone%s renamed: %s"
                        % (len(renamed), "" if len(renamed) == 1 else "s",
                           ", ".join("%r is now %r" % (a, b) for a, b in renamed[:4])))
            # The chain join is by name and nothing else -- an included file's bone that
            # stops matching is left at its -1 sentinel and simply stops being driven.
            if r.get("includes"):
                self.report({"WARNING"},
                            "an animation chain is joined by bone name, so a bone renamed "
                            "here stops matching in %s unless the same rename is made "
                            "there" % ", ".join(r["includes"][:3]))
        moved_up = r["scene"].get("reparented") or []
        if moved_up:
            self.report({"INFO"}, "%s reparented onto %s"
                        % (", ".join(a for a, _b in moved_up[:4])
                           + ("" if len(moved_up) <= 4
                              else " and %d more" % (len(moved_up) - 4)),
                           ", ".join(b for _a, b in moved_up[:4])))
        n = r["scene"].get("rebased") or 0
        if n:
            self.report({"INFO"}, "%d bone%s turned, so %d animation%s you did not export "
                                  "%s re-encoded onto the new bind"
                        % (r["scene"]["bones"], "" if r["scene"]["bones"] == 1 else "s",
                           n, "" if n == 1 else "s", "was" if n == 1 else "were"))
        wide = r["scene"].get("requantised") or []
        if wide:
            # One scale set serves every animation in the file, and fit_scales never
            # narrows, so widening it for the new bind requantises the bone everywhere.
            self.report({"WARNING"}, "%s needed a wider rotation scale, which requantises "
                                     "%s in every animation in the file"
                        % (", ".join(wide[:4])
                           + ("" if len(wide) <= 4 else " and %d more" % (len(wide) - 4)),
                           "it" if len(wide) == 1 else "them"))
        root = r["scene"].get("root_turned") or []
        if root:
            # A movement block is an offset on the entity transform, above the skeleton, so
            # nothing here can derive a rotation for it.
            self.report({"WARNING"}, "%s has no parent and its bind turned, so the skeleton "
                                     "now faces across the travel direction the movement "
                                     "blocks still state" % ", ".join(root))
        gone = r["dropped"]
        moved = len(r["added"]) + len(gone["seqs"])
        stamped = _restamp_sequences(obj, self.filepath, src, r["added"],
                                     gone["seqs"])
        if moved and not stamped:
            # The stash describes the armature's own model, which this export left alone,
            # so what it moved is reachable only by importing what it wrote.
            self.report({"WARNING"},
                        "%d sequence%s %s %s, and this armature's sequence list describes "
                        "%s, which is what the export read. Import the written file to "
                        "edit the sequences it holds"
                        % (moved, "" if moved == 1 else "s",
                           "appended to" if not gone["seqs"] else
                           "removed from" if not r["added"] else "moved in",
                           os.path.basename(self.filepath), os.path.basename(src)))
        if gone["anim"]:
            self.report({"WARNING"}, "removed animation %s%s. Its action is still in the "
                                     "blend and will be offered as an append"
                        % (gone["anim"],
                           "" if not gone["seqs"] else
                           " and the %d sequence%s that played only it"
                           % (len(gone["seqs"]),
                              "" if len(gone["seqs"]) == 1 else "s")))
        # The stamp is what makes the next export replace this animation rather than
        # append it a second time. `vtmb_source` is left alone when the action already
        # has one: the name match comes first anyway, and repointing it would move the
        # donor out from under an action exported to a scratch path.
        for i, name, _nf, _mv in r["added"]:
            act = bpy.data.actions.get(name)
            if act is None:
                continue
            act["vtmb_anim_index"] = i
            if not act.get("vtmb_source"):
                act["vtmb_source"] = self.filepath
        what = ", ".join("%s from %r (%d frames)" % (n, a, f)
                         for _, n, a, f, _ in r["wrote"]) or "nothing"
        if r["added"]:
            what += "; added %s" % ", ".join("%s (%d frames)" % (n, f)
                                             for _i, n, f, _mv in r["added"])
        extra = ("; %s of %d vertices over %d meshes"
                 % ("+".join(mesh["fields"]), mesh["verts"], mesh["models"])
                 if mesh["verts"] else "")
        scene = r["scene"]
        if any(scene[k] for k in ("bones", "materials", "sequences")):
            extra += ("; the scene moved %d bones, renamed %d materials and changed %d "
                      "sequence fields"
                      % (scene["bones"], scene["materials"], scene["sequences"]))
        if scene["springs"]:
            extra += "; retuned %d spring bone fields" % scene["springs"]
        if scene.get("spring_ends"):
            self.report({"INFO"}, "%d spring bone chain%s given a named end bone. 0 of the "
                                  "600 shipped records name one -- they all run to the "
                                  "leaf -- so this takes a path no shipped file takes"
                        % (scene["spring_ends"],
                           "" if scene["spring_ends"] == 1 else "s"))
        if scene.get("spring_switched"):
            self.report({"WARNING"},
                        "%d spring bone chain%s switched on or off. A chain that starts "
                        "switched off carries its start bone negated, and the game looks a "
                        "chain up by the raw field, so nothing in it can switch that chain "
                        "back on. 0 of the 600 shipped records are written that way"
                        % (scene["spring_switched"],
                           "" if scene["spring_switched"] == 1 else "s"))
        sb = scene.get("blends_out_of_range") or []
        if sb:
            self.report({"WARNING"},
                        "%d blend cell%s name%s an animation the file has not got -- %s -- "
                        "so each keeps the index the file already carries. The model "
                        "shipped that way and nothing in the scene can name what is not "
                        "there"
                        % (len(sb), "" if len(sb) == 1 else "s",
                           "s" if len(sb) == 1 else "",
                           ", ".join("%r blend %d,%d at animation %d" % tuple(x)
                                     for x in sb[:3])
                           + ("" if len(sb) <= 3 else " and %d more" % (len(sb) - 3))))
        for name, was in scene.get("dup_bones") or ():
            self.report({"WARNING"}, "%r says it was copied from %r, which another bone "
                                     "still is, so it is appended as a new bone and %r "
                                     "keeps the record" % (name, was, was))
        for name, model in scene.get("dup_models") or ():
            self.report({"WARNING"}, "%r claims model %r, which another object owns, so "
                                     "nothing of it reached the file -- the donor path "
                                     "writes one mesh per model and cannot add one"
                        % (name, model))
        if scene.get("face"):
            extra += ("; rewrote %d eyeball or mouth record%s"
                      % (scene["face"], "" if scene["face"] == 1 else "s"))
        if scene.get("flex") and scene["flex"][0] + scene["flex"][1]:
            nc, nr, added = scene["flex"]
            extra += ("; wrote %d flex controller%s and %d rule%s"
                      % (nc, "" if nc == 1 else "s", nr, "" if nr == 1 else "s"))
            if added:
                extra += " (%d new flexdesc%s)" % (added, "" if added == 1 else "es")
        if scene["added_materials"]:
            extra += ("; added %d material%s: %s"
                      % (len(scene["added_materials"]),
                         "" if len(scene["added_materials"]) == 1 else "s",
                         ", ".join(scene["added_materials"])))
            extra += _write_vmts(context, obj, self.filepath,
                                 scene["added_materials"], self.write_vmt)
            extra += _write_tths(context, obj, self.filepath,
                                 scene["added_materials"], self.write_tth,
                                 self.tth_format)
        if scene.get("slots_moved"):
            extra += ("; followed %d material slot%s that had moved since the import"
                      % (scene["slots_moved"],
                         "" if scene["slots_moved"] == 1 else "s"))
        for name in scene.get("slots_gone") or []:
            self.report({"WARNING"},
                        "the material slot holding %r is no longer in the scene. Every "
                        "mesh names a texture record and the format cannot spell "
                        "\"no material\", so that record keeps the name the file gave "
                        "it. A rename made in the same edit cannot be told from the "
                        "delete and was not taken either" % name)
        self.report({"INFO"}, "wrote %d of %d animations over %d bones, %d -> %d bytes: "
                              "%s%s" % (len(r["wrote"]), r["anims"], r["bones"], r["was"],
                                        r["bytes"], what, extra))
        return {"FINISHED"}


# `build` emits each of these with a count of zero, which nobody reads out of a header.
# The chain is the worst on a character: its animations live in the included files.
SCRATCH_DROPS = (
    "collision and ragdoll -- both live in the sibling .phy, not written",
    "flex descs, controllers, rules and every vertanim",
    "eyeballs, mouths and pose parameters",
    "spring bones, procedural bones, IK chains and bone controllers",
    "sequence events and autolayers",
)


def _scratch_meshes(context, arm_obj):
    """The meshes the armature owns, else every mesh in the scene.

    Not filtered on `vtmb_model`: that marker comes from the importer, and a hand-built
    scene -- the case this operator exists for -- is exactly the one with none.
    """
    own = [o for o in context.scene.objects if o.type == "MESH"
           and (o.parent is arm_obj
                or any(getattr(mo, "object", None) is arm_obj for mo in o.modifiers))]
    return own or [o for o in context.scene.objects if o.type == "MESH"]


def _scratch_cdtexture(arm_obj):
    """What the dialog starts the Material dirs field at.

    Reachable without a window on purpose: `invoke` cannot be driven headless, so the seed
    would otherwise be the one part of the dialog no check can see.
    """
    return ";".join(blender_scratch.scene_cdtextures(arm_obj)) or "models/"


def _unresolved(context, arm_obj, filepath, includes, cdtextures):
    """{kind: [name, ...]} a write would point at nothing, or {} with no content root.

    Anchored on the model the armature came from rather than on where it is being saved:
    an export usually goes somewhere outside the install, and rooting on that would find
    no content and then have nothing to say. Silence with no root at all would read as
    "everything resolves", so only a lookup that really ran is reported.
    """
    game_root, mods, extract = _prefs(context)
    roots = paths.roots(arm_obj.get("vtmb_source") or filepath, game_root, mods, extract)
    if not roots:
        return {}
    bad = blender_import.missing_paths(
        blender_import.Content(roots), includes, cdtextures,
        blender_panel.scene_material_names(context, arm_obj))
    return {k: v for k, v in bad.items() if v}


def _scratch_actions(arm_obj=None):
    """Every action keying a bone this armature has, so nothing needs marking up first.

    Matched on the bone names rather than on `pose.bones` alone: a second rig's actions
    live in the same `bpy.data.actions`, and `sample_action` would assign one to this
    armature and read a pose of nothing moving. `blender_scratch._ordered` then sorts by
    `vtmb_anim_index` where the importer left one and keeps collection order otherwise.
    """
    have = None if arm_obj is None else {b.name for b in arm_obj.data.bones}
    out = []
    for a in bpy.data.actions:
        for fc in blender_import.action_fcurves(a):
            path = fc.data_path
            if not path.startswith('pose.bones["'):
                continue
            end = path.find('"]', 12)
            # A malformed path counts as a match: dropping the action would be a silent
            # narrowing, and exporting one too many is visible in the count.
            if have is None or end < 0 or path[12:end] in have:
                out.append(a)
                break
    return out


class EXPORT_OT_vtmb_mdl_scratch(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.vtmb_mdl_scratch"
    bl_label = "Export VTMB Model, no donor " + VERSION
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".mdl"
    filter_glob: bpy.props.StringProperty(default="*.mdl", options={"HIDDEN"})

    scale: bpy.props.FloatProperty(
        name="Scale", default=1.0, min=1e-4, soft_max=100.0,
        description="Blender units per file unit; every bone and vertex position is "
                    "divided by it on the way out. An import stamps what it used on the "
                    "armature and this starts from that")
    surfaceprop: bpy.props.StringProperty(
        name="Surface", default="flesh",
        description="What the model sounds like and behaves like when hit: flesh, metal, "
                    "wood, concrete. Free text -- the engine looks it up in "
                    "scripts/surfaceproperties.txt and falls back without a word when "
                    "the name is not there")
    cdtexture: bpy.props.StringProperty(
        name="Material dirs", default="models/",
        description="Where the engine looks for the .vmt files, under materials/. Each "
                    "Blender material slot becomes a name tried in each of these in turn, "
                    "so a slot named tor_femamor0head resolves as "
                    "materials/<first that has it>/tor_femamor0head.vmt. Several are "
                    "separated by ;, because nothing binds a directory to a particular "
                    "material and 615 of the 4445 shipped models need more than one. An "
                    "import fills this in from the file it read")
    activity: bpy.props.StringProperty(
        name="Activity", default="ACT_IDLE",
        description="The activity every sequence claims. An action carrying its own "
                    "vtmb_activity overrides this one")
    use_range: bpy.props.BoolProperty(
        name="Scene range", default=False,
        description="Sample the scene's Start and End frames rather than each action's "
                    "own first and last keyframe")
    root_motion: bpy.props.EnumProperty(
        name="Root motion",
        items=[("per_action", "Each action's own",
                "Every action decides for itself, from the Root motion field in the "
                "Action editor's sidebar. There is no donor file here, so an action set "
                "to keep the file's blocks has them fitted from its keys instead -- the "
                "only way to say the engine carries this one when there is nothing to "
                "keep. An action that was never imported and was never set is extracted"),
               ("extract", "Engine carries the character",
                "Put the ground distance the root covers into a movement block, so the "
                "engine carries the character while the animation plays on the spot. "
                "The bob and sway stay in the keys. This is what a walk cycle wants, "
                "and what an import with Root motion on needs putting back"),
               ("none", "Skeleton moves, engine does not",
                "Write no movement blocks and leave the motion in the keys, so the "
                "skeleton itself moves and snaps back when the sequence loops. Right "
                "where the travel belongs to the animation rather than to the character"),
               ("in_place", "Nothing moves",
                "Write no movement blocks and take the net ground distance back out of "
                "the keys, so neither the engine nor the skeleton advances. The bob and "
                "sway stay, so a run cycle still runs -- on the spot")],
        default="per_action")
    chain: bpy.props.BoolProperty(
        name="Keep the animation chain", default=True,
        description="Write the included models the armature carries, so the engine reads "
                    "their animations as if they were this file's. This is where a "
                    "character's walk cycles actually live -- a bone-set template puts "
                    "the chain there, and an import stamps whatever the donor had. The "
                    "join is by bone name and is case-insensitive, so a renamed bone "
                    "silently loses every animation that moved it")
    fit_hull: bpy.props.BoolProperty(
        name="Fit hull to the mesh", default=True,
        description="Set the movement hull to the bounding box of the geometry. Off "
                    "leaves the humanoid default of (-16,-16,0)..(16,16,72), which is "
                    "wrong for anything that is not a person. Neither is the truth: this "
                    "field is the hull the .qc's $hbox sets, not the mesh's extent")
    fit_hitboxes: bpy.props.BoolProperty(
        name="Fit hitboxes to the skin", default=True,
        description="Give every bone that owns geometry a hitbox enclosing it, so the "
                    "model can be shot. Off writes no hitbox set at all and the engine "
                    "falls back to the movement hull, which is one box for the whole "
                    "body. The hit group each box reports is a guess off the bone name "
                    "unless the pose bone carries a vtmb_hitgroup")
    checksum: bpy.props.IntProperty(
        name="Checksum", default=0x5A534E31, subtype="UNSIGNED",
        description="Scripting only. Written into both files; any value is legal as long "
                    "as the two agree, which they do by construction. The engine draws "
                    "nothing at all when they disagree")

    @classmethod
    def poll(cls, context):
        # Not the donor operator's four conditions: those want an assigned action, and
        # allsequences.mdl ships 59 bones with no animations and no sequences.
        obj = context.active_object
        return obj is not None and obj.type == "ARMATURE" and bool(obj.data.bones)

    def invoke(self, context, event):
        obj = context.active_object
        self.scale = float(obj.get("vtmb_scale", 1.0) or 1.0)
        self.cdtexture = _scratch_cdtexture(obj)
        return super().invoke(context, event)

    def draw(self, context):
        lay = self.layout
        obj = context.active_object
        meshes = _scratch_meshes(context, obj)
        actions = _scratch_actions(obj)

        box = _section(lay, "vtmb_s_model", "Model", icon="FILE_3D")
        if box is not None:
            _pair(box, "name in the file", blender_scratch.embedded_name(self.filepath))
            box.prop(self, "scale")
            box.prop(self, "surfaceprop")
            box.prop(self, "cdtexture")
            dirs = paths.cdtexture_list(self.cdtexture)
            stamped = blender_scratch.scene_cdtextures(obj)
            if len(dirs) > 1:
                for p in dirs[:4]:
                    box.label(text="    materials/" + p)
                if len(dirs) > 4:
                    box.label(text="    and %d more" % (len(dirs) - 4))
            if stamped and dirs != stamped:
                _pair(box, "the imported model said", ";".join(stamped[:2])
                      + (" and %d more" % (len(stamped) - 2) if len(stamped) > 2 else ""),
                      icon="ERROR")

        box = _section(lay, "vtmb_s_geom", "Geometry", icon="MESH_DATA")
        if box is not None:
            _pair(box, "bones", str(len(obj.data.bones)))
            _pair(box, "meshes", "%d object%s" % (len(meshes),
                                                  "" if len(meshes) == 1 else "s"),
                  icon="NONE" if meshes else "ERROR")
            if not meshes:
                box.label(text="a skeleton with no mesh is written and draws nothing",
                          icon="INFO")
            no_uv = [o.name for o in meshes
                     if blender_export.uv_layer_of(o)[0] is None]
            if no_uv:
                _pair(box, "no UV layer", ", ".join(no_uv[:3]), icon="ERROR")
            spare = sorted({n for o in meshes
                            for n in blender_export.uv_layer_of(o)[1]})
            if spare:
                _pair(box, "UV layers not written", ", ".join(spare[:3]), icon="INFO")
            lost = sorted({blender_export.uv_layer_of(o)[2] for o in meshes} - {None})
            if lost:
                _pair(box, "UV layer gone", ", ".join(lost[:3]), icon="ERROR")

        box = _section(lay, "vtmb_s_anims", "Animations", icon="ACTION")
        if box is not None:
            _pair(box, "actions", str(len(actions)),
                  icon="NONE" if actions else "INFO")
            for a in actions[:4]:
                box.label(text="    " + a.name)
            if len(actions) > 4:
                box.label(text="    and %d more" % (len(actions) - 4))
            box.prop(self, "use_range")
            box.prop(self, "activity")
            # This operator writes every action, so one action's answer says nothing --
            # what is reported is the split across all of them.
            try:
                _pair(box, "root motion", _mode_tally(actions)
                      if self.root_motion == blender_export.PER_ACTION
                      else "%s, forced by the caller"
                           % ROOT_MOTION_LABELS[self.root_motion])
            except (ValueError, KeyError) as exc:
                box.label(text=str(exc), icon="ERROR")
            if self.root_motion == blender_export.PER_ACTION:
                box.label(text="set it in Object Data > VTMB animations", icon="ACTION")
            box.prop(self, "chain")
            chain = blender_scratch.scene_includes(obj)
            if not chain:
                _pair(box, "chained models", "none on this armature", icon="INFO")
            elif self.chain:
                for p in chain[:3]:
                    box.label(text="    " + p)
                if len(chain) > 3:
                    box.label(text="    and %d more" % (len(chain) - 3))
            else:
                _pair(box, "chained models", "%d dropped" % len(chain), icon="ERROR")

        box = _section(lay, "vtmb_s_fit", "Fit", icon="SHADING_BBOX")
        if box is not None:
            box.prop(self, "fit_hull")
            box.prop(self, "fit_hitboxes")
            hull = blender_scratch.fit_hull(meshes, self.scale) if self.fit_hull else None
            if hull is not None:
                _pair(box, "hull", "%.0f %.0f %.0f .. %.0f %.0f %.0f"
                      % (hull[0] + hull[1]))
            elif self.fit_hull:
                _pair(box, "hull", "no vertices, so the default is kept", icon="INFO")

        box = _section(lay, "vtmb_s_drops", "Not written", icon="LOCKED")
        if box is not None:
            for line in SCRATCH_DROPS:
                box.label(text=line)

    def execute(self, context):
        obj = context.active_object
        meshes = _scratch_meshes(context, obj)
        hull = blender_scratch.fit_hull(meshes, self.scale) if self.fit_hull else None
        try:
            r = blender_scratch.export_scene(
                context, obj, meshes, _scratch_actions(obj), self.filepath, self.checksum,
                scale=self.scale, surfaceprop=self.surfaceprop,
                cdtexture=self.cdtexture, hull=hull, use_range=self.use_range,
                activity=self.activity, hitboxes=self.fit_hitboxes,
                root_motion=self.root_motion,
                includes=(blender_scratch.scene_includes(obj) if self.chain else ()))
        except (blender_scratch.Refused, mdl_build.Refused) as exc:
            self.report({"ERROR"}, "refused: %s" % exc)
            return {"CANCELLED"}
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, "%s: %s -- traceback on the console"
                        % (type(exc).__name__, exc))
            return {"CANCELLED"}
        _report_unfitted(self, r.get("unfitted"))
        _report_unkeepable(self, r.get("unkeepable"))
        if r["unskinned"]:
            self.report({"WARNING"}, "%d vertices belong to no bone and were pinned to "
                                     "bone 0, which drags them wherever it goes"
                        % r["unskinned"])
        for name, odd in r.get("stray_groups") or ():
            self.report({"WARNING"}, "%s: %s name%s no bone, so %s bind%s nothing -- a "
                                     "vertex whose only group it is goes to bone 0. A "
                                     "bone name spelt wrong lands here"
                        % (name, ", ".join(repr(g) for g in odd),
                           "s" if len(odd) == 1 else "",
                           "it" if len(odd) == 1 else "they",
                           "s" if len(odd) == 1 else ""))
        if r["crowded"]:
            total = sum(k for _n, k in r["crowded"])
            self.report({"WARNING"}, "%d vertex%s carr%s a fifth vertex group, which no "
                                     "record can hold: %s. The four heaviest are written "
                                     "and the fourth bone takes what the rest held"
                        % (total, "" if total == 1 else "es",
                           "ies" if total == 1 else "y",
                           ", ".join("%s x%d" % (n, k) for n, k in r["crowded"][:3])
                           + ("" if len(r["crowded"]) <= 3
                              else " and %d more mesh%s" % (len(r["crowded"]) - 3,
                                   "" if len(r["crowded"]) == 4 else "es"))))
        for name, npin, nvert in r.get("cloths") or ():
            self.report({"INFO"}, "%s: cloth over %d particles, %d of them pinned"
                        % (name, nvert, npin))
        if r["dropped"]:
            self.report({"WARNING"}, "dropped %s"
                        % ", ".join("%s x%d" % (k, v)
                                    for k, v in sorted(r["dropped"].items())))
        unresolved = _unresolved(
            context, obj, self.filepath,
            blender_scratch.scene_includes(obj) if self.chain else (),
            paths.cdtexture_list(self.cdtexture))
        for kind, bad in sorted(unresolved.items()):
            self.report({"WARNING"}, "%d %s resolve against nothing and will not draw: %s"
                        % (len(bad), kind, ", ".join(bad[:4])))
        self.report({"INFO"}, "wrote %s and its .dx80.vtx: %d bones, %d bodyparts, "
                              "%d materials, %d animations (%d with root motion), "
                              "%d sequences, %d chained, %d hitboxes, %d faces, "
                              "%d verts, %d + %d bytes"
                    % (os.path.basename(self.filepath), r["bones"], r["bodyparts"],
                       r["materials"], r["anims"], r["with_root_motion"], r["seqs"],
                       r["includes"], r["hitboxes"], r["faces"], r["verts"], r["bytes"],
                       r["vtx_bytes"]))
        return {"FINISHED"}


CLASSES = ([VTMB_AddonPreferences, IMPORT_OT_vtmb_mdl, EXPORT_OT_vtmb_mdl,
            EXPORT_OT_vtmb_mdl_scratch] + list(blender_templates.CLASSES)
           + list(blender_panel.CLASSES))

if hasattr(bpy.types, "FileHandler"):
    class IO_FH_vtmb_mdl(bpy.types.FileHandler):
        bl_idname = "IO_FH_vtmb_mdl"
        bl_label = "VTMB Model"
        bl_import_operator = "import_scene.vtmb_mdl"
        bl_file_extensions = ".mdl"

        @classmethod
        def poll_drop(cls, context):
            return context.area and context.area.type == "VIEW_3D"

    CLASSES.append(IO_FH_vtmb_mdl)


def _menu(self, context):
    self.layout.operator(IMPORT_OT_vtmb_mdl.bl_idname, text="VTMB Model (.mdl)")


def _menu_export(self, context):
    self.layout.operator(EXPORT_OT_vtmb_mdl.bl_idname, text="VTMB Model (.mdl)")


def _menu_export_scratch(self, context):
    self.layout.operator(EXPORT_OT_vtmb_mdl_scratch.bl_idname,
                         text="VTMB Model, no donor (.mdl)")


def unregister():
    for menu, fn in ((bpy.types.TOPBAR_MT_file_import, _menu),
                     (bpy.types.TOPBAR_MT_file_export, _menu_export),
                     (bpy.types.TOPBAR_MT_file_export, _menu_export_scratch),
                     (bpy.types.VIEW3D_MT_add, blender_templates.menu)):
        try:
            menu.remove(fn)
        except Exception:
            pass
    blender_panel.unregister_props()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


def register():
    # addon_utils compares only __init__.py's mtime, so without this an edit to any
    # other module survives a disable/enable cycle as stale code.
    # Dependency order: a module has to be reloaded before anything that imports it, or the
    # importer keeps the old object and the reload buys nothing.
    for m in (checksum, sections, relocs, normal_table, mdl, mdl_write, mdl_build,
              mdl_rebuild, mesh_write, paths, tth, vpk, vtx, vtx_write, vtx_rebuild,
              bone_templates,
              blender_import, blender_export, blender_scratch, blender_templates,
              blender_panel):
        importlib.reload(m)
    # Tolerate a half-registered state left by an edit-and-re-enable cycle: a stale
    # class of the same bl_idname otherwise makes register_class raise.
    unregister()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    blender_panel.register_props()
    bpy.types.TOPBAR_MT_file_import.append(_menu)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export_scratch)
    bpy.types.VIEW3D_MT_add.append(blender_templates.menu)
