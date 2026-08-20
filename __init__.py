"""Blender addon: import Vampire: The Masquerade - Bloodlines models (MDL v2531).

Run selfcheck.py under plain Python for the format check; it needs no Blender.
"""

bl_info = {
    "name": "VTMB Model (MDL v2531)",
    "author": "Claude Opus 5 xhigh / Twiglard",
    "blender": (3, 0, 0),
    "location": "File > Import > VTMB Model (.mdl), File > Export > VTMB Model (.mdl), "
                "File > Export > VTMB Model, no donor (.mdl)",
    # bl_info has no "website"; doc_url is the key that becomes a button.
    "doc_url": "https://rpgcodex.net/",
    "description": "Import Bloodlines skeletons, meshes, UVs, weights, textures and "
                   "animations including the chained models. Export rebuilds a .mdl "
                   "from its own decoded records: animations from Blender's poses, "
                   "and optionally vertex positions, normals, UVs and weights. A "
                   "second exporter authors a .mdl and .dx80.vtx from the scene with "
                   "no donor file at all",
    "version": (0, 2, 0),
    "category": "Import-Export",
}

from . import (checksum, mdl, mdl_build, mdl_rebuild, mdl_write, mesh_write, paths,
               relocs, sections, tth, vpk, vtx, vtx_rebuild, vtx_write,
               blender_import, blender_export, blender_scratch)

import importlib
import os

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper

# In the operator title, because the File menu entries pass their own text and a stale
# addon is otherwise indistinguishable from a bug.
VERSION = ".".join(str(x) for x in bl_info["version"])

ACTIVE, ALL, NONE = "~active", "~all", "~none"
SPLIT = 0.34


def _pair(box, key, value, icon="NONE"):
    row = box.split(factor=SPLIT)
    row.label(text=key)
    row.label(text=value, icon=icon)


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


def _mesh_verts(base):
    """Vertices the file's models declare. 0 for the animation libraries and null.mdl."""
    try:
        return sum(mo.numvertices
                   for _bi, _mi, _bp, mo in mesh_write.models_of(_cached_mdl(base)))
    except Exception:
        return 0


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
    """(animation, action) pairs the every-matching-action mode would write, and how
    many further actions aim at a slot already taken."""
    path = _base_path(op, context)
    anims = _base_anims(path)
    hits, ignored = blender_export.match_indices([n for n, _ in anims], path)
    return [(anims[i][0], a.name) for i, a in sorted(hits.items())], len(ignored)


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

    with_mesh: bpy.props.BoolProperty(
        name="Mesh", default=True,
        description="Read the geometry: vertices, UVs, vertex groups and materials. "
                    "Faces come from the .vtx file beside the .mdl. Off leaves the "
                    "armature and its animations alone")
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
        description="Put back the travel studiomdl took out of the animation and left in "
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
        lay.prop(self, "with_mesh")
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
                with_mesh=self.with_mesh,
                with_anims=self.with_anims, with_chained=self.with_chained,
                game_root=game_root, mods=mods, extra_roots=extract,
                root_motion=self.root_motion)
        except Exception as exc:
            self.report({"ERROR"}, "%s: %s" % (type(exc).__name__, exc))
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
    keep_travel: bpy.props.BoolProperty(
        name="Was applied", default=True,
        description="Take the travel back out of the keys before writing. Tick this "
                    "whenever the model was imported with Root motion on, or the travel "
                    "is written a second time on top of what the file already stores and "
                    "the animation covers twice the ground")
    travel: bpy.props.EnumProperty(
        name="Travel",
        items=[("keep", "Leave the file's alone",
                "Keep the movement blocks the file already has. Right whenever the "
                "animation still travels the way it did"),
               ("extract", "Rebuild from the root bone",
                "Fit new movement blocks to the root bone's path. Use this when you "
                "changed where the animation goes, not just how it looks"),
               ("none", "Strip it",
                "Write no movement blocks. The animation plays on the spot and the "
                "engine does not move the character")],
        default="keep")
    use_range: bpy.props.BoolProperty(
        name="Scene range", default=False,
        description="Write the scene's Start and End frames rather than the action's "
                    "own first and last keyframe. Use this to export a slice of a "
                    "longer action")
    write_positions: bpy.props.BoolProperty(
        name="Positions", default=False,
        description="Write each vertex's location. Moving vertices is allowed; adding or "
                    "removing them is not, because the triangle lists live in the .vtx "
                    "files and every section below the mesh would have to move")
    write_normals: bpy.props.BoolProperty(
        name="Normals", default=False,
        description="Write each vertex's normal, which is what the file shades with. "
                    "Every face meeting at a vertex must agree on it, since the format "
                    "stores one normal per vertex and spells a hard edge by duplicating "
                    "the vertex")
    write_uvs: bpy.props.BoolProperty(
        name="UVs", default=False,
        description="Write the texture coordinates without touching the geometry they "
                    "sit on. Every face meeting at a vertex must agree on the "
                    "coordinate: the file stores one UV per vertex and spells a seam by "
                    "duplicating the vertex, so a seam newly cut in Blender has nowhere "
                    "to go")
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

        box = _section(lay, "vtmb_anims", "Animations", icon="ACTION")
        if box is not None:
            row = box.split(factor=SPLIT)
            row.label(text="Replace")
            row.prop(self, "target", text="")
            if self.target == ALL:
                hits, ignored = _matches(self, context)
                _pair(box, "matched", "%d of %d" % (len(hits), len(_base_anims(base))),
                      icon="NONE" if hits else "ERROR")
                for name, act_name in hits[:4]:
                    box.label(text="    " + (name if name == act_name
                                             else "%s ← %s" % (name, act_name)))
                if len(hits) > 4:
                    box.label(text="    and %d more" % (len(hits) - 4))
                if ignored:
                    _pair(box, "ignored", "%d duplicate" % ignored, icon="INFO")
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
            box = _section(lay, "vtmb_travel", "Root motion", icon="ORIENTATION_GIMBAL")
            if box is not None:
                box.prop(self, "keep_travel")
                row = box.split(factor=SPLIT)
                row.label(text="Travel")
                row.prop(self, "travel", text="")

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
            if self.target == NONE:
                actions = {}
            elif self.target == ALL:
                actions, ignored = blender_export.match_actions(m, src)
                if not actions:
                    raise ValueError("no action shares a name with an animation of %s"
                                     % os.path.basename(src))
                if ignored:
                    self.report({"WARNING"}, blender_export.describe_ignored(ignored))
            else:
                act = obj.animation_data.action
                actions = {blender_export.resolve_target(
                    m, act, "" if self.target == ACTIVE else self.target): act}
            r = blender_export.export_actions(
                context, obj, src, self.filepath, actions,
                keep_travel=self.keep_travel, travel=self.travel,
                mesh_fields=_mesh_fields(self),
                frame_start=context.scene.frame_start if self.use_range and one else None,
                frame_end=context.scene.frame_end if self.use_range and one else None)
        except Exception as exc:
            self.report({"ERROR"}, "%s: %s" % (type(exc).__name__, exc))
            return {"CANCELLED"}
        finally:
            context.scene.frame_set(frame)
        mesh = r["mesh"]
        if mesh["missing"] and mesh["fields"]:
            self.report({"WARNING"}, "%d model%s of %s had no mesh in the scene and kept "
                                     "the file's own: %s"
                        % (len(mesh["missing"]), "" if len(mesh["missing"]) == 1 else "s",
                           os.path.basename(src), ", ".join(mesh["missing"][:4])))
        if mesh["normals"]:
            self.report({"WARNING"}, "%d normals came from Blender rather than the file, "
                                     "which costs up to 0.9 degrees each"
                        % mesh["normals"])
        if mesh["unsupported"]:
            self.report({"WARNING"}, "%s cannot be stored by every model of this file and "
                                     "was skipped there" % ", ".join(mesh["unsupported"]))
        if r["scene"]["bones"] and r["scene"]["stale"]:
            self.report({"WARNING"}, "%d bone%s moved; the %d animation%s you did not export "
                                     "still key the old skeleton and may not follow it. "
                                     "Export them together to re-encode."
                        % (r["scene"]["bones"], "" if r["scene"]["bones"] == 1 else "s",
                           r["scene"]["stale"], "" if r["scene"]["stale"] == 1 else "s"))
        what = ", ".join("%s from %r (%d frames)" % (n, a, f)
                         for _, n, a, f, _ in r["wrote"]) or "nothing"
        extra = ("; %s of %d vertices over %d meshes"
                 % ("+".join(mesh["fields"]), mesh["verts"], mesh["models"])
                 if mesh["verts"] else "")
        scene = r["scene"]
        if any(scene.values()):
            extra += ("; the scene moved %d bones, renamed %d materials and changed %d "
                      "sequence fields"
                      % (scene["bones"], scene["materials"], scene["sequences"]))
        self.report({"INFO"}, "wrote %d of %d animations over %d bones, %d -> %d bytes: "
                              "%s%s" % (len(r["wrote"]), r["anims"], r["bones"], r["was"],
                                        r["bytes"], what, extra))
        return {"FINISHED"}


# `build` emits each of these with a count of zero, which nobody reads out of a header.
# The chain is the worst on a character: its animations live in the included files.
SCRATCH_DROPS = (
    "the include chain, so only this scene's animations exist",
    "collision and ragdoll -- both live in the sibling .phy, not written",
    "cloth, flex descs, controllers, rules and every vertanim",
    "eyeballs, mouths and pose parameters",
    "spring bones, procedural bones, IK chains and bone controllers",
    "attachments, sequence events and autolayers",
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
        name="Material dir", default="models/",
        description="Where the engine looks for the .vmt files, under materials/. Each "
                    "Blender material slot becomes a name in this directory, so a slot "
                    "named tor_femamor0head resolves as "
                    "materials/<this>/tor_femamor0head.vmt")
    activity: bpy.props.StringProperty(
        name="Activity", default="ACT_IDLE",
        description="The activity every sequence claims. An action carrying its own "
                    "vtmb_activity overrides this one")
    use_range: bpy.props.BoolProperty(
        name="Scene range", default=False,
        description="Sample the scene's Start and End frames rather than each action's "
                    "own first and last keyframe")
    travel: bpy.props.EnumProperty(
        name="Travel",
        items=[("extract", "Move the character",
                "Put the ground distance the root covers into a movement block, so the "
                "engine carries the character while the animation plays on the spot. "
                "The bob and sway stay in the keys. This is what a walk cycle wants, "
                "and what an import with Root motion on needs putting back"),
               ("none", "Leave it in the keys",
                "Write no movement blocks. The skeleton itself travels and snaps back "
                "when the sequence loops, which is right only for something that really "
                "does move in place")],
        default="extract")
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
                    "unless the bone carries a vtmb_hitgroup")
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
        self.scale = float(context.active_object.get("vtmb_scale", 1.0) or 1.0)
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

        box = _section(lay, "vtmb_s_geom", "Geometry", icon="MESH_DATA")
        if box is not None:
            _pair(box, "bones", str(len(obj.data.bones)))
            _pair(box, "meshes", "%d object%s" % (len(meshes),
                                                  "" if len(meshes) == 1 else "s"),
                  icon="NONE" if meshes else "ERROR")
            if not meshes:
                box.label(text="a skeleton with no mesh is written and draws nothing",
                          icon="INFO")
            no_uv = [o.name for o in meshes if not o.data.uv_layers.active]
            if no_uv:
                _pair(box, "no UV layer", ", ".join(no_uv[:3]), icon="ERROR")

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
            row = box.row()
            row.label(text="Travel")
            row.prop(self, "travel", text="")

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
                activity=self.activity, hitboxes=self.fit_hitboxes, travel=self.travel)
        except blender_scratch.Refused as exc:
            self.report({"ERROR"}, "refused: %s" % exc)
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, "%s: %s" % (type(exc).__name__, exc))
            return {"CANCELLED"}
        if r["unskinned"]:
            self.report({"WARNING"}, "%d vertices belong to no bone and were pinned to "
                                     "bone 0, which drags them wherever it goes"
                        % r["unskinned"])
        if r["dropped"]:
            self.report({"WARNING"}, "dropped %s"
                        % ", ".join("%s x%d" % (k, v)
                                    for k, v in sorted(r["dropped"].items())))
        self.report({"INFO"}, "wrote %s and its .dx80.vtx: %d bones, %d bodyparts, "
                              "%d materials, %d animations (%d travelling), "
                              "%d sequences, %d hitboxes, %d faces, %d verts, "
                              "%d + %d bytes"
                    % (os.path.basename(self.filepath), r["bones"], r["bodyparts"],
                       r["materials"], r["anims"], r["travelling"], r["seqs"],
                       r["hitboxes"], r["faces"], r["verts"], r["bytes"],
                       r["vtx_bytes"]))
        return {"FINISHED"}


CLASSES = [VTMB_AddonPreferences, IMPORT_OT_vtmb_mdl, EXPORT_OT_vtmb_mdl,
           EXPORT_OT_vtmb_mdl_scratch]

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
                     (bpy.types.TOPBAR_MT_file_export, _menu_export_scratch)):
        try:
            menu.remove(fn)
        except Exception:
            pass
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
    for m in (checksum, sections, relocs, mdl, mdl_write, mdl_build, mdl_rebuild,
              mesh_write, paths, tth, vpk, vtx, vtx_write, vtx_rebuild,
              blender_import, blender_export, blender_scratch):
        importlib.reload(m)
    # Tolerate a half-registered state left by an edit-and-re-enable cycle: a stale
    # class of the same bl_idname otherwise makes register_class raise.
    unregister()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_menu)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export_scratch)
