"""The armature's two path lists, in Properties > Object Data, and the per-action fields
in the Action editor's sidebar.

They are the plain custom properties the rest of the addon already reads -- an import
stamps them, a bone-set template writes `vtmb_includes` -- so this edits those keys rather
than shadowing them in a PropertyGroup. The path lists are not reachable from a mesh or a
material: the `.mdl` binds them to the model, not to anything the scene has one of per
material.
"""

import os

import bpy

from . import blender_export
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


MODE_ITEMS = (
    ("unset", "Not set",
     "Leave it to the export dialog, which falls back to the animation this action came "
     "from: the file's own blocks where it had one, nothing where it did not"),
    ("keep", "Engine carries the character, as the file has it",
     "Reuse the movement blocks the file already has, unchanged. A from-scratch export "
     "has no file to keep them from and fits them from the keys instead"),
    ("extract", "Engine carries the character",
     "Fit new movement blocks to the root bone's path in the scene, replacing whatever "
     "the file had"),
    ("none", "Skeleton moves, engine does not",
     "Write no movement blocks and leave the motion in the keys, so the skeleton itself "
     "walks away from the origin. 1344 of the game's own animations do this"),
    ("in_place", "Nothing moves",
     "Write no movement blocks and take the net ground distance back out of the keys. "
     "The bob and sway stay, so a run cycle still runs -- on the spot"),
)


def _mode_get(action):
    v = action.get(blender_export.MODE_ATTR)
    for i, item in enumerate(MODE_ITEMS):
        if item[0] == v:
            return i
    return 0


def _mode_set(action, index):
    if index <= 0:
        if action.get(blender_export.MODE_ATTR) is not None:
            del action[blender_export.MODE_ATTR]
    else:
        action[blender_export.MODE_ATTR] = MODE_ITEMS[index][0]


def _loops_get(action):
    return bool(int(action.get("vtmb_seq_flags") or 0) & 1)


def _loops_set(action, value):
    flags = int(action.get("vtmb_seq_flags") or 0)
    action["vtmb_seq_flags"] = (flags | 1) if value else (flags & ~1)


def _activity_get(action):
    return str(action.get("vtmb_activity") or "")


def _activity_set(action, value):
    value = str(value).strip()
    if value:
        action["vtmb_activity"] = value
    elif action.get("vtmb_activity") is not None:
        del action["vtmb_activity"]


def panel_action(context):
    """The action a sidebar panel is about: the Action editor's, else the object's."""
    act = getattr(context.space_data, "action", None)
    if act is not None:
        return act
    ad = context.object.animation_data if context.object is not None else None
    return ad.action if ad is not None else None


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


class VTMB_PT_action(bpy.types.Panel):
    bl_label = "VTMB"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "VTMB"

    @classmethod
    def poll(cls, context):
        return panel_action(context) is not None

    def draw(self, context):
        act = panel_action(context)
        lay = self.layout
        lay.use_property_split = False

        col = lay.column(align=True)
        col.label(text=act.name, icon="ACTION")
        col.prop(act, "vtmb_root_motion_choice", text="")
        if act.get(blender_export.MODE_ATTR) is None:
            col.label(text="    -> " + _fallback_line(act))

        col = lay.column(align=True)
        col.prop(act, "vtmb_loops")
        col.prop(act, "vtmb_activity_text")

        nmv = act.get("vtmb_movements")
        if nmv is not None:
            col = lay.column(align=True)
            col.label(text="came from an animation with %d movement block%s"
                           % (nmv, "" if nmv == 1 else "s"), icon="INFO")
            col.label(text="keys %s the travel"
                           % ("carry" if act.get("vtmb_root_motion") else "do not carry"))


SPRING_FIELDS = (
    ("vtmb_spring_gravity", "Gravity", "3 on 202 of the corpus's 600 records, 0 on 187, "
     "1 on 168. Subtracted as gravity*dt from each node's z every substep"),
    ("vtmb_spring_damping", "Damping", "0.9 on 351 of 600. Velocity kept from one Verlet "
     "step to the next, so 1.0 never settles and 0.0 is dead weight"),
    ("vtmb_spring_exp", "Spring", "0.3 on 326 of 600. The solver takes pow(10.0, -this), "
     "so 0.0 is the stiffest setting and larger is slacker"),
    ("vtmb_spring_maxangle", "Max angle", "Degrees, 30 on 259 of 600. How far a node may "
     "leave the pose the animation put it in"),
)


class VTMB_PT_bone(bpy.types.Panel):
    bl_label = "VTMB spring bone"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        pb = _spring_bone(context)
        return pb is not None and pb.get("vtmb_spring_index") is not None

    def draw(self, context):
        pb = _spring_bone(context)
        lay = self.layout
        lay.use_property_split = True

        col = lay.column(align=True)
        col.label(text="chain %d" % int(pb["vtmb_spring_index"]), icon="PHYSICS")
        end = pb.get("vtmb_spring_end")
        col.label(text="ends at %s" % (end if end else "the first-child chain"))
        if pb.get("vtmb_spring_disabled"):
            col.label(text="starts switched off, and cannot be named back on",
                      icon="ERROR")

        col = lay.column(align=True)
        for key, name, _desc in SPRING_FIELDS:
            if pb.get(key) is not None:
                col.prop(pb, '["%s"]' % key, text=name)

        col = lay.column(align=True)
        col.label(text="bc_override 1 retunes these live, without a re-export", icon="INFO")


def _spring_bone(context):
    """The pose bone behind the Bone tab's active bone, or None."""
    obj, bone = context.object, context.bone
    if obj is None or bone is None or obj.type != "ARMATURE":
        return None
    return obj.pose.bones.get(bone.name)


def _fallback_line(act):
    """What Not set resolves to, so the field is never silently a guess."""
    try:
        mode = blender_export.root_motion_mode(act, blender_export.PER_ACTION)
    except ValueError as exc:
        return str(exc)
    for item in MODE_ITEMS:
        if item[0] == mode:
            return item[1]
    return mode


CLASSES = [VTMB_OT_add_cdtexture, VTMB_OT_add_include, VTMB_OT_check_paths,
           VTMB_PT_armature, VTMB_PT_action, VTMB_PT_bone]

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
    bpy.types.Action.vtmb_root_motion_choice = bpy.props.EnumProperty(
        name="Root motion", items=MODE_ITEMS, get=_mode_get, set=_mode_set,
        description="What a re-export does with this action's travel. Not set leaves it "
                    "to the export dialog, which reads the animation this action came "
                    "from -- a block in the file means the engine carries it")
    bpy.types.Action.vtmb_loops = bpy.props.BoolProperty(
        name="Loops", get=_loops_get, set=_loops_set,
        description="Bit 0 of the sequence flags, which is what the engine reads to play "
                    "this animation round again rather than holding its last frame. "
                    "mstudioanimdesc_t has its own flags field and the engine does not "
                    "read looping from it")
    bpy.types.Action.vtmb_activity_text = bpy.props.StringProperty(
        name="Activity", get=_activity_get, set=_activity_set,
        description="The activity the sequence written for this action claims, e.g. "
                    "ACT_IDLE. Empty takes whichever the export dialog offers")


def unregister_props():
    for attr, _key, _name, _desc in _PROPS:
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    for attr in ("vtmb_root_motion_choice", "vtmb_loops", "vtmb_activity_text"):
        if hasattr(bpy.types.Action, attr):
            delattr(bpy.types.Action, attr)
