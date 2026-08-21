"""The armature's two path lists, in Properties > Object Data.

They are the plain custom properties the rest of the addon already reads -- an import
stamps both, a bone-set template writes `vtmb_includes` -- so this edits those keys rather
than shadowing them in a PropertyGroup. Neither is reachable from a mesh or a material:
the `.mdl` binds them to the model, not to anything the scene has one of per material.
"""

import os

import bpy

from . import blender_import
from . import blender_scratch
from . import paths as paths_mod

INCLUDES = "vtmb_includes"
CDTEXTURE = "vtmb_cdtexture"


def _text(obj, key):
    return ";".join(paths_mod.engine_paths(obj.get(key)))


def _store(obj, key, value):
    out = paths_mod.engine_paths(str(value).split(";"))
    if out:
        obj[key] = out
    elif obj.get(key) is not None:
        del obj[key]


def _append(obj, key, path):
    out = paths_mod.engine_paths(list(obj.get(key) or ()) + [path])
    obj[key] = out
    return out


def _prefs(context):
    # Deferred: the package module imports this one, so at import time it is half-built.
    from . import _prefs as f
    return f(context)


def _content(context, obj):
    game_root, mods, extra = _prefs(context)
    roots = paths_mod.roots(obj.get("vtmb_source") or "", game_root, mods, extra)
    return blender_import.Content(roots) if roots else None


def scene_material_names(context, arm_obj):
    """Every material slot name on the meshes this armature owns, first-seen order."""
    out = []
    for o in context.scene.objects:
        if o.type != "MESH":
            continue
        if o.parent is not arm_obj and not any(
                getattr(mo, "object", None) is arm_obj for mo in o.modifiers):
            continue
        for m in o.data.materials:
            if m is not None and m.name not in out:
                out.append(m.name)
    return out


class VTMB_OT_add_cdtexture(bpy.types.Operator):
    bl_idname = "object.vtmb_add_cdtexture"
    bl_label = "Add material directory"
    bl_description = ("Pick a .vmt, or the directory holding them, and add where it sits "
                      "under materials/")
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH", options={"HIDDEN"})
    directory: bpy.props.StringProperty(subtype="DIR_PATH", options={"HIDDEN"})
    filter_glob: bpy.props.StringProperty(default="*.vmt", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def invoke(self, context, event):
        game_root, mods, _extra = _prefs(context)
        for m in mods:
            start = os.path.join(game_root, m, paths_mod.MATERIALS_DIR)
            if game_root and os.path.isdir(start):
                self.directory = start + os.sep
                break
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        pick = self.filepath or self.directory
        if not pick:
            return {"CANCELLED"}
        name = None
        if os.path.isfile(pick):
            name = os.path.splitext(os.path.basename(pick))[0]
            pick = os.path.dirname(pick)
        rel = paths_mod.content_relative(pick, paths_mod.MATERIALS_DIR)
        if rel is None:
            self.report({"ERROR"}, "%s is not under a materials/ directory, so there is no "
                                   "engine path for it" % pick)
            return {"CANCELLED"}
        out = _append(context.object, CDTEXTURE, rel + "/")
        if name:
            self.report({"INFO"}, "added %s/ -- a material slot named %r resolves through "
                                  "it now (%d director%s)"
                        % (rel, name, len(out), "y" if len(out) == 1 else "ies"))
        else:
            self.report({"INFO"}, "added %s/ (%d director%s)"
                        % (rel, len(out), "y" if len(out) == 1 else "ies"))
        return {"FINISHED"}


class VTMB_OT_add_include(bpy.types.Operator):
    bl_idname = "object.vtmb_add_include"
    bl_label = "Add chained model"
    bl_description = ("Pick a .mdl whose animations this one should read, and add where it "
                      "sits under models/")
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH", options={"HIDDEN"})
    filter_glob: bpy.props.StringProperty(default="*.mdl", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def invoke(self, context, event):
        game_root, mods, _extra = _prefs(context)
        for m in mods:
            start = os.path.join(game_root, m, paths_mod.MODELS_DIR)
            if game_root and os.path.isdir(start):
                self.filepath = start + os.sep
                break
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        rel = paths_mod.content_relative(self.filepath, paths_mod.MODELS_DIR)
        if rel is None:
            self.report({"ERROR"}, "%s is not under a models/ directory, so there is no "
                                   "engine path for it" % self.filepath)
            return {"CANCELLED"}
        out = _append(context.object, INCLUDES, paths_mod.MODELS_DIR + "/" + rel)
        self.report({"INFO"}, "added %s/%s (%d chained)"
                    % (paths_mod.MODELS_DIR, rel, len(out)))
        return {"FINISHED"}


class VTMB_OT_check_paths(bpy.types.Operator):
    bl_idname = "object.vtmb_check_paths"
    bl_label = "Check paths"
    bl_description = ("Resolve every chained model, and every material slot name against "
                      "the directories above, using Game root")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def execute(self, context):
        obj = context.object
        content = _content(context, obj)
        if content is None:
            self.report({"ERROR"}, "no content root: set Game root in the add-on "
                                   "preferences, or open a model from an install")
            return {"CANCELLED"}
        names = scene_material_names(context, obj)
        bad = blender_import.missing_paths(
            content, blender_scratch.scene_includes(obj),
            blender_scratch.scene_cdtextures(obj), names)
        for kind in ("includes", "materials"):
            if bad[kind]:
                self.report({"WARNING"}, "%d %s resolve against nothing: %s"
                            % (len(bad[kind]), kind, ", ".join(bad[kind][:4])))
        if not bad["includes"] and not bad["materials"]:
            self.report({"INFO"}, "all %d chained and all %d material slots resolve"
                        % (len(blender_scratch.scene_includes(obj)), len(names)))
        return {"FINISHED"}


class VTMB_PT_armature(bpy.types.Panel):
    bl_label = "VTMB"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        obj = context.object
        lay = self.layout
        lay.use_property_split = False

        col = lay.column(align=True)
        col.label(text="Material directories, under materials/", icon="MATERIAL")
        row = col.row(align=True)
        row.prop(obj, "vtmb_cdtexture_text", text="")
        row.operator(VTMB_OT_add_cdtexture.bl_idname, text="", icon="FILEBROWSER")
        for p in blender_scratch.scene_cdtextures(obj)[:6]:
            col.label(text="    materials/" + p)

        col = lay.column(align=True)
        col.label(text="Animation chain, joined by bone name", icon="ACTION")
        row = col.row(align=True)
        row.prop(obj, "vtmb_includes_text", text="")
        row.operator(VTMB_OT_add_include.bl_idname, text="", icon="FILEBROWSER")
        for p in blender_scratch.scene_includes(obj)[:6]:
            col.label(text="    " + p)

        lay.operator(VTMB_OT_check_paths.bl_idname, icon="VIEWZOOM")


CLASSES = [VTMB_OT_add_cdtexture, VTMB_OT_add_include, VTMB_OT_check_paths,
           VTMB_PT_armature]

_PROPS = (
    ("vtmb_cdtexture_text", CDTEXTURE, "Material dirs",
     "The directories under materials/ a material name is tried against, ; separated and "
     "in order. A slot named Andrei with models/character/monster/andrei/ here resolves "
     "as materials/models/character/monster/andrei/Andrei.vmt. Nothing binds a directory "
     "to a particular material, so this is the whole of what the .mdl says about where "
     "its images live"),
    ("vtmb_includes_text", INCLUDES, "Animation chain",
     "The models this one reads its animations through, ; separated and in engine order, "
     "e.g. models/character/shared/male/npc_allsequences.mdl. The engine joins them by "
     "bone name and is case-insensitive, so a renamed bone silently stops animating"),
)


def register_props():
    for attr, key, name, desc in _PROPS:
        setattr(bpy.types.Object, attr, bpy.props.StringProperty(
            name=name, description=desc,
            get=(lambda k: lambda self: _text(self, k))(key),
            set=(lambda k: lambda self, v: _store(self, k, v))(key)))


def unregister_props():
    for attr, _key, _name, _desc in _PROPS:
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
