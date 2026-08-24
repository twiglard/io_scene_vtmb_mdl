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
    "version": (0, 2, 0),
    "category": "Import-Export",
}

from . import (checksum, mdl, mdl_build, mdl_rebuild, mdl_write, mesh_write,
               normal_table, paths,
               relocs, sections, tth, vpk, vtx, vtx_rebuild, vtx_write, bone_templates,
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
    keep_travel: bpy.props.BoolProperty(
        name="Imported with root motion", default=True,
        description="This action's keys still carry the travel an import put into them, "
                    "so it has to come back out before writing. Untick only for an "
                    "action imported with Root motion off. Leaving it ticked wrongly "
                    "writes the travel a second time on top of what the file already "
                    "stores and the animation covers twice the ground. Not the same as "
                    "the mode below: this is about the keys in the scene, that is about "
                    "the movement blocks in the file")
    travel: bpy.props.EnumProperty(
        name="Root motion",
        items=[("keep", "Engine carries the character, as the file has it",
                "Reuse the movement blocks the file already has, unchanged. Right "
                "whenever you did not move the animation, only changed how it looks. "
                "This is about the file's blocks; whether the keys in the scene still "
                "carry travel is the checkbox above, and the two are set separately"),
               ("extract", "Engine carries the character",
                "Fit new movement blocks to the root bone's path in the scene, "
                "replacing whatever the file had. Use this when you changed where the "
                "animation goes, not just how it looks"),
               ("none", "Skeleton moves, engine does not",
                "Write no movement blocks. The travel stays in the keys, so the skeleton "
                "itself walks away from the origin and snaps back when the sequence "
                "loops, and the engine never advances the character")],
        default="keep")
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
                extra = _surplus_bones(self, context, base)
                if extra:
                    _pair(box, "bones not in the file", "%d, not written" % len(extra),
                          icon="ERROR")
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
            box = _section(lay, "vtmb_travel", "Root motion", icon="ORIENTATION_GIMBAL")
            if box is not None:
                box.prop(self, "keep_travel")
                box.prop(self, "travel", text="")

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
            r = blender_export.export_actions(
                context, obj, src, self.filepath, actions,
                keep_travel=self.keep_travel, travel=self.travel,
                mesh_fields=_mesh_fields(self), add=adds, drop=gone,
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
            self.report({"WARNING"}, "%d vertices of a rebuilt mesh are in no vertex "
                                     "group and were bound to bone 0"
                        % mesh["unskinned"])
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
        for name in r["stale"]:
            self.report({"WARNING"}, "%s beside the model still describes the old "
                                     "geometry; only .dx80.vtx is written" % name)
        surplus = r["scene"]["surplus"]
        if surplus:
            self.report({"WARNING"}, "%d bone%s of this armature %s not in %s and %s not "
                                     "written: %s%s. The file's own bone list is what an "
                                     "export walks, so a bone added in Blender is left out"
                        % (len(surplus), "" if len(surplus) == 1 else "s",
                           "is" if len(surplus) == 1 else "are",
                           os.path.basename(src),
                           "was" if len(surplus) == 1 else "were",
                           ", ".join(surplus[:4]),
                           "" if len(surplus) <= 4 else " and %d more" % (len(surplus) - 4)))
        if r["scene"]["bones"] and r["scene"]["stale"]:
            self.report({"WARNING"}, "%d bone%s moved; the %d animation%s you did not export "
                                     "still key the old skeleton and may not follow it. "
                                     "Export them together to re-encode."
                        % (r["scene"]["bones"], "" if r["scene"]["bones"] == 1 else "s",
                           r["scene"]["stale"], "" if r["scene"]["stale"] == 1 else "s"))
        gone = r["dropped"]
        if gone["anim"]:
            # apply_sequences matches the stash to the file by position and refuses one
            # claiming more sequences than are there, so a delete that took sequences
            # makes every later export fail until this is written again.
            if gone["seqs"]:
                obj["vtmb_sequences"] = blender_import.sequence_stash(
                    mdl.Mdl(self.filepath))
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
        self.report({"INFO"}, "wrote %d of %d animations over %d bones, %d -> %d bytes: "
                              "%s%s" % (len(r["wrote"]), r["anims"], r["bones"], r["was"],
                                        r["bytes"], what, extra))
        return {"FINISHED"}


# `build` emits each of these with a count of zero, which nobody reads out of a header.
# The chain is the worst on a character: its animations live in the included files.
SCRATCH_DROPS = (
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
    travel: bpy.props.EnumProperty(
        name="Root motion",
        items=[("extract", "Engine carries the character",
                "Put the ground distance the root covers into a movement block, so the "
                "engine carries the character while the animation plays on the spot. "
                "The bob and sway stay in the keys. This is what a walk cycle wants, "
                "and what an import with Root motion on needs putting back"),
               ("none", "Skeleton moves, engine does not",
                "Write no movement blocks. The skeleton itself travels and snaps back "
                "when the sequence loops, which is right only for something that really "
                "does move in place")],
        default="extract")
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
            row.label(text="Root motion")
            row.prop(self, "travel", text="")
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
                activity=self.activity, hitboxes=self.fit_hitboxes, travel=self.travel,
                includes=(blender_scratch.scene_includes(obj) if self.chain else ()))
        except (blender_scratch.Refused, mdl_build.Refused) as exc:
            self.report({"ERROR"}, "refused: %s" % exc)
            return {"CANCELLED"}
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, "%s: %s -- traceback on the console"
                        % (type(exc).__name__, exc))
            return {"CANCELLED"}
        if r["unskinned"]:
            self.report({"WARNING"}, "%d vertices belong to no bone and were pinned to "
                                     "bone 0, which drags them wherever it goes"
                        % r["unskinned"])
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
                              "%d materials, %d animations (%d travelling), "
                              "%d sequences, %d chained, %d hitboxes, %d faces, "
                              "%d verts, %d + %d bytes"
                    % (os.path.basename(self.filepath), r["bones"], r["bodyparts"],
                       r["materials"], r["anims"], r["travelling"], r["seqs"],
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
