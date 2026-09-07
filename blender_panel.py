"""The armature's fields in Properties > Object Data -- the two path lists, the pose
parameter array and the skin family picker -- and the per-action fields in the Action
editor's sidebar.

They are the plain custom properties the rest of the addon already reads -- an import
stamps them, a bone-set template writes `vtmb_includes` -- so this edits those keys rather
than shadowing them in a PropertyGroup. The path lists are not reachable from a mesh or a
material: the `.mdl` binds them to the model, not to anything the scene has one of per
material.
"""

import os

import bpy
import mathutils

from . import blender_export
from . import blender_import
from . import blender_scratch
from . import mdl as mdl_mod
from . import paths as paths_mod

INCLUDES = "vtmb_includes"
CHAIN = "vtmb_chain"
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


def arm_meshes(context, arm_obj):
    """The mesh objects this armature owns, parented or bound by an Armature modifier."""
    out = []
    for o in context.scene.objects:
        if o.type != "MESH":
            continue
        if o.parent is arm_obj or any(getattr(mo, "object", None) is arm_obj
                                      for mo in o.modifiers):
            out.append(o)
    return out


def scene_material_names(context, arm_obj):
    """Every material slot name on the meshes this armature owns, first-seen order."""
    out = []
    for o in arm_meshes(context, arm_obj):
        for m in o.data.materials:
            if m is not None and m.name not in out:
                out.append(m.name)
    return out


def skin_rows(arm_obj):
    """The skin table as rows, back out of the flattened `vtmb_skin_table` stamp."""
    flat = list(arm_obj.get("vtmb_skin_table") or ())
    nref = int(arm_obj.get("vtmb_skin_refs") or 0)
    if not flat or nref <= 0:
        return []
    return [[int(x) for x in flat[f:f + nref]] for f in range(0, len(flat), nref)]


def skin_family_names(arm_obj, family):
    """The material name each skinref selects under `family`, in skinref order."""
    rows = skin_rows(arm_obj)
    mats = list(arm_obj.get("vtmb_skin_materials") or ())
    if not 0 <= family < len(rows):
        return []
    return [str(mats[r]) if 0 <= r < len(mats) else "" for r in rows[family]]


class VTMB_OT_set_skin_family(bpy.types.Operator):
    bl_idname = "vtmb.set_skin_family"
    bl_label = "Show skin family"
    bl_description = ("Point every mesh's material slots at one row of the model's skin "
                      "table. The engine picks the same row from CBaseAnimating.m_nSkin, "
                      "and 488 entities across 51 of the 108 shipped maps set a non-zero "
                      "one. The export writes texture names back through whichever row is "
                      "showing, so this is what the file records, not only what is drawn")
    bl_options = {"REGISTER", "UNDO"}

    family: bpy.props.IntProperty(name="Skin family", default=0, min=0)

    def execute(self, context):
        arm_obj = context.object
        if arm_obj is None or arm_obj.type != "ARMATURE":
            self.report({"ERROR"}, "select the model's armature")
            return {"CANCELLED"}
        rows = skin_rows(arm_obj)
        if not 0 <= self.family < len(rows):
            self.report({"ERROR"}, "this model carries %d skin famil%s, so there is no "
                                   "family %d" % (len(rows),
                                                  "y" if len(rows) == 1 else "ies",
                                                  self.family))
            return {"CANCELLED"}
        mats = list(arm_obj.get("vtmb_skin_materials") or ())
        moved, missing = 0, []
        # Every mesh in one pass: the export inverts one row for the whole model, so a
        # half-switched scene names one texture record two ways and is refused outright.
        for o in arm_meshes(context, arm_obj):
            refs = [int(x) for x in (o.get("vtmb_skinrefs") or ())]
            for slot, r in enumerate(refs):
                if slot >= len(o.data.materials):
                    continue
                ref = rows[self.family][r] if 0 <= r < len(rows[self.family]) else r
                name = str(mats[ref]) if 0 <= ref < len(mats) else ""
                mat = bpy.data.materials.get(name) if name else None
                if mat is None:
                    missing.append(name or "record %d" % ref)
                    continue
                if o.data.materials[slot] is not mat:
                    o.data.materials[slot] = mat
                    moved += 1
        arm_obj["vtmb_skin_family"] = self.family
        if missing:
            self.report({"WARNING"}, "family %d: %d slot(s) moved, and no material in "
                                     "the blend is named %s"
                        % (self.family, moved, ", ".join(sorted(set(missing))[:3])))
        else:
            self.report({"INFO"}, "skin family %d, %d slot(s) moved"
                                  % (self.family, moved))
        return {"FINISHED"}


def draw_skin_families(lay, arm_obj):
    """One row per skin family, the one on screen marked.

    Drawn only above one family, which is 201 of the 4445 shipped models.
    """
    rows = skin_rows(arm_obj)
    if len(rows) < 2:
        lay.label(text="one skin family", icon="INFO")
        return
    active = int(arm_obj.get("vtmb_skin_family") or 0)
    box = lay.box()
    for f in range(len(rows)):
        row = box.row(align=True)
        op = row.operator("vtmb.set_skin_family", text="",
                          icon="RADIOBUT_ON" if f == active else "RADIOBUT_OFF")
        op.family = f
        row.label(text="skin %d" % f)
        row.label(text=", ".join(n for n in skin_family_names(arm_obj, f) if n)[:64])


class VTMB_PT_skin_families(bpy.types.Panel):
    bl_label = "VTMB skin families"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        self.layout.use_property_split = False
        draw_skin_families(self.layout, context.object)


CLOTH = "vtmb_cloth"
CLOTH_NUMBERS = (
    ("sigma", "Sigma", 0.0, 1000.0,
     "The stiffness the solver divides by. It is the one number the corpus gives no rule "
     "for -- 118654 of 118654 springs fix the mass split and both spring groups follow "
     "from the faces, so a preset supplies this and nothing derives it"),
    ("slack", "Slack", 0.0, 4.0,
     "The rest length as a fraction of the measured distance. Below 1 the sheet is pulled "
     "taut, above 1 it hangs"),
    ("scale", "Scale", 0.0, 1000.0,
     "The object's own +0x00 float, which the solver multiplies its step by"),
)
CLOTH_CAP = 8192                # counting group members is O(verts) and draw() runs per redraw


def _cloth_get(obj, key):
    return bool(obj.get(key))


def _cloth_set(obj, key, value):
    if value:
        obj[key] = True
    elif key in obj:
        del obj[key]


_PRESET_ITEMS = []              # Blender frees a dynamic items list it does not own


def _preset_names():
    return [""] + sorted(blender_scratch.cloth_mod.PRESETS)


def _cloth_preset_items(self, context):
    del _PRESET_ITEMS[:]
    _PRESET_ITEMS.append(("", "None", "The three numbers come from the overrides below, "
                                      "or from cloth.resolve's own fallbacks"))
    _PRESET_ITEMS.extend((k, k, "") for k in sorted(blender_scratch.cloth_mod.PRESETS))
    return _PRESET_ITEMS


def _preset_get(obj):
    names = _preset_names()
    v = str(obj.get("vtmb_cloth_preset") or "")
    return names.index(v) if v in names else 0


def _preset_set(obj, value):
    names = _preset_names()
    v = names[value] if 0 <= value < len(names) else ""
    if v:
        obj["vtmb_cloth_preset"] = v
    elif "vtmb_cloth_preset" in obj:
        del obj["vtmb_cloth_preset"]


def _cloth_text(obj, key, value):
    value = value.strip()
    if value and value != blender_scratch.PIN_GROUP:
        obj[key] = value
    elif key in obj:
        del obj[key]


def _cloth_num_on(obj, k, value):
    key = "vtmb_cloth_%s" % k
    if not value:
        if key in obj:
            del obj[key]
        return
    if key in obj:
        return
    # Turning an override on seeds it with what the preset was already resolving to, so
    # the number never jumps when the checkbox is ticked.
    numbers = cloth_numbers(obj)[0]
    at = [n[0] for n in CLOTH_NUMBERS].index(k)
    obj[key] = float(numbers[at]) if numbers else 0.0


def cloth_numbers(obj):
    """(sigma, slack, scale) as the export will resolve them, and which of the three are
    overridden on this object."""
    over = tuple("vtmb_cloth_%s" % k in obj for k, _l, _lo, _hi, _d in CLOTH_NUMBERS)
    preset = str(obj.get("vtmb_cloth_preset") or "") or None
    if preset is not None and preset not in blender_scratch.cloth_mod.PRESETS:
        return None, over
    got = [obj.get("vtmb_cloth_%s" % k) for k, _l, _lo, _hi, _d in CLOTH_NUMBERS]
    return blender_scratch.cloth_mod.resolve(preset, *got), over


def cloth_pins(obj):
    """(pinned, total) for the object's pin group, or None where it cannot be counted."""
    me = obj.data
    vg = obj.vertex_groups.get(str(obj.get("vtmb_cloth_pin_group") or
                                   blender_scratch.PIN_GROUP))
    if vg is None or len(me.vertices) > CLOTH_CAP:
        return None
    n = sum(1 for v in me.vertices if any(g.group == vg.index for g in v.groups))
    return n, len(me.vertices)


def draw_cloth(lay, obj):
    lay.prop(obj, "vtmb_cloth_on")
    if not obj.get(CLOTH):
        return
    lay.prop(obj, "vtmb_cloth_preset_name")
    lay.prop(obj, "vtmb_cloth_pin_text")
    lay.prop(obj, "vtmb_cloth_flip_on")
    numbers, over = cloth_numbers(obj)
    if numbers is None:
        lay.label(text="preset %r is not one of the %d"
                       % (str(obj.get("vtmb_cloth_preset")),
                          len(blender_scratch.cloth_mod.PRESETS)), icon="ERROR")
        return
    for i, (k, label, _lo, _hi, _desc) in enumerate(CLOTH_NUMBERS):
        row = lay.row(align=True)
        row.prop(obj, "vtmb_cloth_%s_on" % k, text="")
        if over[i]:
            row.prop(obj, "vtmb_cloth_%s_num" % k, text=label)
        else:
            sub = row.row()
            sub.enabled = False
            sub.label(text="%s  %g" % (label, numbers[i]))
    group = str(obj.get("vtmb_cloth_pin_group") or blender_scratch.PIN_GROUP)
    pins = cloth_pins(obj)
    if pins is None:
        lay.label(text="no vertex group %r" % group, icon="ERROR")
    elif pins[0] == 0:
        lay.label(text="%r is empty -- every particle would be free" % group, icon="ERROR")
    elif pins[0] == pins[1]:
        lay.label(text="%r pins all %d -- nothing would move" % (group, pins[1]),
                  icon="ERROR")
    else:
        lay.label(text="%d pinned, %d free" % (pins[0], pins[1] - pins[0]))


class VTMB_PT_cloth(bpy.types.Panel):
    bl_label = "VTMB cloth"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        self.layout.use_property_split = False
        draw_cloth(self.layout, context.object)


def accessories_of(arm_obj):
    """(attachment empties, {set ordinal: [box empties]}, {set ordinal: name}) for drawing.

    The export's own grouping, so what the panel counts is what the file will carry.
    """
    return blender_export.accessory_objects(arm_obj)


class VTMB_OT_add_attachment(bpy.types.Operator):
    bl_idname = "vtmb.add_attachment"
    bl_label = "Add attachment"
    bl_description = ("Mount point on the active bone. 495 shipped characters carry theirs "
                      "as attachment '0' on Bip01 R Hand")
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(name="Name", default="0")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and obj.type == "ARMATURE"
                and context.active_bone is not None)

    def execute(self, context):
        arm_obj, bone = context.object, context.active_bone
        obj = bpy.data.objects.new("%s.%s" % (arm_obj.name, self.name), None)
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = 1.0
        _link_beside(arm_obj, obj)
        obj.parent = arm_obj
        obj.parent_type = "BONE"
        obj.parent_bone = bone.name
        obj.matrix_parent_inverse = mathutils.Matrix.Translation((0.0, -bone.length, 0.0))
        obj.matrix_basis = mathutils.Matrix.Identity(4)
        obj["vtmb_attachment"] = self.name
        obj["vtmb_attachment_type"] = 0
        return {"FINISHED"}


class VTMB_OT_add_hitbox(bpy.types.Operator):
    bl_idname = "vtmb.add_hitbox"
    bl_label = "Add hitbox"
    bl_description = ("Axis-aligned box on the active bone, in the named set. Group is the "
                      "Source hit group and nothing in the file derives it")
    bl_options = {"REGISTER", "UNDO"}

    group: bpy.props.IntProperty(name="Group", default=0, min=0)
    set_index: bpy.props.IntProperty(name="Set", default=0, min=0)

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and obj.type == "ARMATURE"
                and context.active_bone is not None)

    def execute(self, context):
        arm_obj, bone = context.object, context.active_bone
        _attach, boxes, names = accessories_of(arm_obj)
        if self.set_index not in names:
            root = bpy.data.objects.new("%s.default" % arm_obj.name, None)
            root.empty_display_type = "PLAIN_AXES"
            _link_beside(arm_obj, root)
            root.parent = arm_obj
            root["vtmb_hitboxset"] = "default"
            root["vtmb_hitboxset_index"] = self.set_index
        obj = bpy.data.objects.new("%s.box" % arm_obj.name, None)
        obj.empty_display_type = "CUBE"
        obj.empty_display_size = 1.0
        _link_beside(arm_obj, obj)
        obj.parent = arm_obj
        obj.parent_type = "BONE"
        obj.parent_bone = bone.name
        obj.matrix_parent_inverse = mathutils.Matrix.Translation((0.0, -bone.length, 0.0))
        # Half the bone's own length, so a new box is visible without being the whole model.
        h = max(bone.length, 1e-4) * 0.5
        obj.matrix_basis = mathutils.Matrix.Diagonal((h, h, h, 1.0))
        obj["vtmb_hitbox_group"] = self.group
        obj["vtmb_hitboxset_index"] = self.set_index
        return {"FINISHED"}


def _link_beside(arm_obj, obj):
    for c in bpy.data.collections:
        if arm_obj.name in c.objects:
            c.objects.link(obj)
            return
    bpy.context.scene.collection.objects.link(obj)


def draw_accessories(lay, context, arm_obj):
    attach, boxes, names = accessories_of(arm_obj)
    row = lay.row(align=True)
    row.operator("vtmb.add_attachment", icon="EMPTY_ARROWS")
    row.operator("vtmb.add_hitbox", icon="MESH_CUBE")
    if not attach and not boxes:
        lay.label(text="no attachment and no hitbox", icon="INFO")
        return
    for obj in attach:
        lay.label(text="%s on %s" % (obj.get("vtmb_attachment") or obj.name,
                                     obj.parent_bone or "no bone"), icon="EMPTY_ARROWS")
    for k in sorted(boxes):
        lay.label(text="%s: %d box%s" % (names.get(k, "default"), len(boxes[k]),
                                         "" if len(boxes[k]) == 1 else "es"),
                  icon="MESH_CUBE")


class VTMB_PT_accessories(bpy.types.Panel):
    bl_label = "VTMB attachments and hitboxes"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        self.layout.use_property_split = False
        draw_accessories(self.layout, context, context.object)


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


def draw_action_fields(lay, act):
    """The three per-action fields, wherever they are hosted.

    Two panels draw them -- the Action editor's sidebar, which is about the action on
    screen, and the Object Data tab's list, which is about the model's whole set -- and
    `operator-check.py` reads the `.prop(act, ...)` calls out of this one function, so a
    field added here is a field both hosts get and the check counts.
    """
    col = lay.column(align=True)
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


# `keep` and `extract` differ only after the comma in MODE_ITEMS, so a truncation of
# those labels cannot tell them apart in one list column.
SHORT_MODES = {
    "keep": "engine carries, as the file has it",
    "extract": "engine carries, fitted to the keys",
    "none": "skeleton moves",
    "in_place": "nothing moves",
}


def _mode_label(act):
    """The short name of the mode this action writes, stamped or fallen back to."""
    try:
        mode = blender_export.root_motion_mode(act, blender_export.PER_ACTION)
    except ValueError:
        return "bad value"
    return SHORT_MODES.get(mode, mode)


def listed_action(obj):
    """The action `VTMB_PT_actions` is editing, or None.

    The index is into all of `bpy.data.actions` rather than into the filtered view, which
    is what `template_list` stores, so deleting an action leaves it past the end. It is
    an RNA property and not a key, so `obj.get` does not see it.
    """
    if obj is None:
        return None
    acts = bpy.data.actions
    i = int(getattr(obj, "vtmb_action_index", 0))
    return acts[i] if 0 <= i < len(acts) else None


def chain_paths(obj):
    """The set of `.mdl` an import read into this armature, ready to compare against an
    action's `vtmb_source`.

    Compared as normalised strings and not with `os.path.samefile`: a model served out of
    a `.vpk` has no file to stat, so `samefile` raises there and would cost a syscall per
    action per redraw where it does not. Both sides of the compare are written by one
    import from one loader, so the strings are identical by construction.
    """
    got = obj.get(CHAIN) if obj is not None else None
    if not got:
        return None
    return {os.path.normcase(os.path.normpath(p)) for p in got if p}


class VTMB_UL_actions(bpy.types.UIList):
    """Every action in the blend that could be this model's, and the mode each writes.

    An action stamped for a model this import never read is dropped; one with no stamp is
    kept, because that is what an action authored in the scene looks like and dropping it
    would hide exactly the ones a user made. The comparison is against every file the
    import opened -- `vtmb_chain` -- and not against the armature's own `vtmb_source`,
    because on a default import almost every action comes from an include and matches the
    armature's source nowhere. An armature carrying no `vtmb_chain` filters nothing: that
    is a blend written before the key existed, and an unfiltered list loses none of it.
    """

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="ACTION")
        sub = row.row()
        sub.alignment = "RIGHT"
        # Dimmed where the mode is the import's fallback and not a stamp, so the list
        # never shows a derived value looking like a decision someone made.
        sub.active = item.get(blender_export.MODE_ATTR) is not None
        sub.label(text=_mode_label(item))

    def filter_items(self, context, data, propname):
        acts = getattr(data, propname)
        mine = chain_paths(context.object)
        bit = self.bitflag_filter_item
        flt = [bit] * len(acts)
        if mine:
            for i, a in enumerate(acts):
                own = a.get("vtmb_source")
                if own and os.path.normcase(os.path.normpath(own)) not in mine:
                    flt[i] = 0
        helper = bpy.types.UI_UL_list
        if self.filter_name:
            named = helper.filter_items_by_name(self.filter_name, bit, acts, "name")
            flt = [f & n for f, n in zip(flt, named)]
        order = []
        if self.use_filter_sort_alpha:
            order = helper.sort_items_by_name(acts, "name")
        return flt, order


class VTMB_PT_actions(bpy.types.Panel):
    bl_label = "VTMB animations"
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
        lay.template_list("VTMB_UL_actions", "", bpy.data, "actions",
                          obj, "vtmb_action_index", rows=6)
        act = listed_action(obj)
        if act is None:
            lay.label(text="the blend holds no action", icon="INFO")
            return
        draw_action_fields(lay, act)
        ad = obj.animation_data
        if ad is None or ad.action is not act:
            lay.label(text="not the assigned action -- an export with target Active "
                           "writes %s" % (ad.action.name if ad and ad.action
                                          else "nothing"), icon="INFO")


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
        lay.label(text=act.name, icon="ACTION")
        draw_action_fields(lay, act)


# Only the ids whose handler carries its own name: HandleAnimEvent's third arm, 4005,
# reaches entity+0x6f0, and the other 33 the corpus ships have no located handler at all.
EVENT_HANDLERS = {
    2070: "TurnOffPhysicsChain",
    2071: "TurnOnPhysicsChain",
}


def _action_seq(arm_obj, act):
    """(stash index, the sequence dict) whose first blend names this action, else (None, None).

    Chained actions belong to another file and no sequence here names them, which is why
    the match is on the action's own name rather than on `vtmb_anim_index`.
    """
    stash = arm_obj.get("vtmb_sequences") if arm_obj is not None else None
    for k, seq in enumerate(stash or ()):
        blends = seq.get("blends") or []
        if blends and blends[0] and blends[0][0] == act.name:
            return k, seq
    return None, None


def _action_armature(context, act):
    obj = context.object
    if obj is not None and obj.type == "ARMATURE" and _action_seq(obj, act)[0] is not None:
        return obj
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and _action_seq(o, act)[0] is not None:
            return o
    return None


def _write_events(arm_obj, index, events):
    """Replace one sequence's event list in the stash.

    An ID-property list is not mutable in place, so the whole stash is rebuilt -- which is
    also what keeps the sequence order the export matches by position.
    """
    stash = [dict(x) for x in arm_obj.get("vtmb_sequences") or ()]
    stash[index]["events"] = events
    arm_obj["vtmb_sequences"] = stash


def _sync_markers(act, events):
    """Redraw the action's event markers from the stash, which is the record.

    Markers Blender's retiming moved are discarded rather than read back: a marker holds
    a name and a frame and cannot say which record it came from.
    """
    for mk in [m for m in act.pose_markers if m.name.startswith("event ")]:
        act.pose_markers.remove(mk)
    span = max(1, int(act.get("vtmb_numframes") or 1) - 1)
    for e in events:
        mk = act.pose_markers.new("event %d" % int(e.get("event") or 0))
        mk.frame = int(round(float(e.get("cycle") or 0.0) * span))


class VTMB_OT_add_event(bpy.types.Operator):
    bl_idname = "vtmb.add_event"
    bl_label = "Add animation event"
    bl_description = ("Append an mstudioevent_t to the sequence this action drives, at "
                      "the current frame. The id is the number the game switches on")
    bl_options = {"REGISTER", "UNDO"}

    event_id: bpy.props.IntProperty(name="Event id", default=2050, min=0)
    options: bpy.props.StringProperty(
        name="Options", default="",
        description="The event's options[64] string. Read by atoi for most ids and as a "
                    "spring bone chain name for 2070 and 2071. 63 bytes at most")

    def execute(self, context):
        act = panel_action(context)
        arm_obj = _action_armature(context, act) if act else None
        if act is None or arm_obj is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        if len(self.options.encode("latin1", "replace")) > 63:
            self.report({"ERROR"}, "options is over 63 bytes and the field holds 63 "
                                   "plus a terminator")
            return {"CANCELLED"}
        k, seq = _action_seq(arm_obj, act)
        span = max(1, int(act.get("vtmb_numframes") or 1) - 1)
        cycle = min(1.0, max(0.0, context.scene.frame_current / float(span)))
        events = [dict(x) for x in seq.get("events") or ()]
        events.append({"cycle": cycle, "event": self.event_id, "type": 0,
                       "options": self.options})
        _write_events(arm_obj, k, events)
        _sync_markers(act, events)
        self.report({"INFO"}, "event %d at cycle %.4f, %d on this sequence"
                              % (self.event_id, cycle, len(events)))
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class VTMB_OT_remove_event(bpy.types.Operator):
    bl_idname = "vtmb.remove_event"
    bl_label = "Remove animation event"
    bl_description = "Drop one mstudioevent_t from the sequence this action drives"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(name="Index", default=0, min=0)

    def execute(self, context):
        act = panel_action(context)
        arm_obj = _action_armature(context, act) if act else None
        if act is None or arm_obj is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        k, seq = _action_seq(arm_obj, act)
        events = [dict(x) for x in seq.get("events") or ()]
        if not 0 <= self.index < len(events):
            self.report({"ERROR"}, "no event %d on this sequence" % self.index)
            return {"CANCELLED"}
        gone = events.pop(self.index)
        _write_events(arm_obj, k, events)
        _sync_markers(act, events)
        self.report({"INFO"}, "dropped event %d, %d left"
                              % (int(gone.get("event") or 0), len(events)))
        return {"FINISHED"}


def draw_action_events(lay, context, act):
    """The sequence's mstudioevent_t array, one row each.

    Drawn from `vtmb_sequences` and never from the markers: a marker carries a name and a
    frame, so the stash is the only thing that can say what an event's options string was.
    """
    arm_obj = _action_armature(context, act)
    if arm_obj is None:
        lay.label(text="no imported sequence names this action", icon="INFO")
        return
    _k, seq = _action_seq(arm_obj, act)
    events = list(seq.get("events") or ())
    span = max(1, int(act.get("vtmb_numframes") or 1) - 1)
    if not events:
        lay.label(text="no events", icon="INFO")
    box = lay.box() if events else None
    for i, e in enumerate(events):
        eid = int(e.get("event") or 0)
        cycle = float(e.get("cycle") or 0.0)
        row = box.row(align=True)
        named = EVENT_HANDLERS.get(eid)
        row.label(text="%d%s" % (eid, "  " + named if named else ""))
        row.label(text="frame %d" % int(round(cycle * span)))
        opt = str(e.get("options") or "")
        row.label(text=opt if opt else "-")
        row.operator("vtmb.remove_event", text="", icon="X").index = i
    lay.operator("vtmb.add_event", icon="ADD")


class VTMB_PT_action_events(bpy.types.Panel):
    bl_label = "VTMB events"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "VTMB"

    @classmethod
    def poll(cls, context):
        return panel_action(context) is not None

    def draw(self, context):
        self.layout.use_property_split = False
        draw_action_events(self.layout, context, panel_action(context))


def _param_names(arm_obj):
    return [str(p.get("name") or "") for p in arm_obj.get("vtmb_poseparams") or ()]


def _param_rec(arm_obj, name):
    for p in arm_obj.get("vtmb_poseparams") or ():
        if str(p.get("name") or "") == name:
            return dict(p)
    return None


def _write_params(arm_obj, index, params):
    stash = [dict(x) for x in arm_obj.get("vtmb_sequences") or ()]
    stash[index]["params"] = params
    arm_obj["vtmb_sequences"] = stash


class VTMB_OT_set_param(bpy.types.Operator):
    bl_idname = "vtmb.set_param"
    bl_label = "Set blend axis pose parameter"
    bl_description = ("Name the pose parameter that drives one blend axis of the sequence "
                      "this action belongs to. An empty name writes -1, which is what "
                      "13724 of the 14012 shipped sequences carry")
    bl_options = {"REGISTER", "UNDO"}

    axis: bpy.props.IntProperty(name="Axis", default=0, min=0, max=1)
    name: bpy.props.StringProperty(name="Pose parameter", default="")

    def execute(self, context):
        act = panel_action(context)
        arm_obj = _action_armature(context, act) if act else None
        if act is None or arm_obj is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        names = _param_names(arm_obj)
        if self.name and self.name not in names:
            self.report({"ERROR"}, "no pose parameter %r on this model -- it has %s"
                        % (self.name, ", ".join(n for n in names if n) or "none"))
            return {"CANCELLED"}
        k, seq = _action_seq(arm_obj, act)
        params = [dict(x) for x in seq.get("params") or ()]
        while len(params) < 2:
            params.append({"name": "", "start": 0.0, "end": 0.0})
        rec = _param_rec(arm_obj, self.name) if self.name else None
        # The sequence's range equals the parameter's own on all 288 shipped sequences
        # that name one, so that is the default rather than zero.
        params[self.axis] = {"name": self.name,
                             "start": float(rec.get("start") or 0.0) if rec else 0.0,
                             "end": float(rec.get("end") or 0.0) if rec else 0.0}
        _write_params(arm_obj, k, params)
        self.report({"INFO"}, "blend axis %d is driven by %s"
                              % (self.axis, self.name or "nothing"))
        return {"FINISHED"}


def draw_action_params(lay, context, act):
    """Which pose parameter drives each blend axis of this action's sequence.

    Only an axis with more than one blend is drawn: a 1-wide axis has nothing to blend
    between, and 13715 of the 14012 shipped sequences are (1, 1).
    """
    arm_obj = _action_armature(context, act)
    if arm_obj is None:
        lay.label(text="no imported sequence names this action", icon="INFO")
        return
    _k, seq = _action_seq(arm_obj, act)
    names = [n for n in _param_names(arm_obj) if n]
    if not names:
        lay.label(text="this model carries no pose parameter", icon="INFO")
        return
    gs = [int(x) for x in seq.get("groupsize") or (1, 1)]
    params = list(seq.get("params") or ())
    drawn = 0
    for axis in range(2):
        if axis >= len(gs) or gs[axis] <= 1:
            continue
        drawn += 1
        cur = dict(params[axis]) if axis < len(params) else {}
        cur_name = str(cur.get("name") or "")
        box = lay.box()
        box.label(text="axis %d, %d blends" % (axis, gs[axis]))
        row = box.row(align=True)
        for name in [""] + names:
            op = row.operator("vtmb.set_param", text=name or "none",
                              depress=(name == cur_name))
            op.axis = axis
            op.name = name
        if cur_name:
            box.label(text="%s over %.1f to %.1f"
                           % (cur_name, float(cur.get("start") or 0.0),
                              float(cur.get("end") or 0.0)))
    if not drawn:
        lay.label(text="every blend axis is 1 wide, so no pose parameter drives this",
                  icon="INFO")


class VTMB_PT_action_params(bpy.types.Panel):
    bl_label = "VTMB pose parameters"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "VTMB"

    @classmethod
    def poll(cls, context):
        return panel_action(context) is not None

    def draw(self, context):
        self.layout.use_property_split = False
        draw_action_params(self.layout, context, panel_action(context))


def draw_armature_params(lay, arm_obj):
    """The model's mstudioposeparamdesc_t array, which no export authors.

    What reads `move_yaw` and `hit_yaw` at runtime is not located, so a name invented here
    would drive nothing -- the array is displayed and carried, never written.
    """
    pps = list(arm_obj.get("vtmb_poseparams") or ())
    if not pps:
        lay.label(text="no pose parameters", icon="INFO")
        return
    box = lay.box()
    for pp in pps:
        row = box.row(align=True)
        row.label(text=str(pp.get("name") or ""))
        row.label(text="%.1f to %.1f" % (float(pp.get("start") or 0.0),
                                         float(pp.get("end") or 0.0)))
        row.label(text="loop %g" % float(pp.get("loop") or 0.0))


class VTMB_PT_poseparams(bpy.types.Panel):
    bl_label = "VTMB pose parameters"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        self.layout.use_property_split = False
        draw_armature_params(self.layout, context.object)


FLAGS_ATTR = "vtmb_bone_flags"

# Not Valve's BONE_ALWAYS_PROCEDURAL, which is 0x4; here 0x1 equals proctype != 0
# exactly. ref/2531/studio-verified.h:814-842.
BONE_ALWAYS_PROCEDURAL = 0x1
USED_BY_MASK = 0xFFFC


def _bone_flags(pb):
    f = pb.get(FLAGS_ATTR) if pb is not None else None
    return None if f is None else int(f)


def _from_root_get(pb):
    f = _bone_flags(pb)
    return bool(f is not None and f & mdl_mod.BONE_ROTATION_FROM_ROOT)


def _from_root_set(pb, value):
    f = _bone_flags(pb)
    if f is None:
        return
    bit = mdl_mod.BONE_ROTATION_FROM_ROOT
    pb[FLAGS_ATTR] = (f | bit) if value else (f & ~bit)


def _root_bones(arm_obj):
    """Every parentless bone. The export fits movement blocks from bone 0's path and
    nothing checks that bone 0 is the only root, so a second one is worth naming."""
    return [b.name for b in arm_obj.data.bones if b.parent is None]


class VTMB_PT_bone_flags(bpy.types.Panel):
    bl_label = "VTMB bone"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        pb = _spring_bone(context)
        return pb is not None and (pb.get(FLAGS_ATTR) is not None
                                   or pb.get("vtmb_bone_name") is not None)

    def draw(self, context):
        pb = _spring_bone(context)
        obj = context.object
        lay = self.layout
        lay.use_property_split = False

        name = pb.get("vtmb_bone_name")
        if name and name != pb.name:
            _bone_pair(lay, "named in the file", str(name))

        roots = _root_bones(obj)
        if pb.name in roots:
            col = lay.column(align=True)
            col.label(text="no parent -- root motion is fitted from this bone's path",
                      icon="ORIENTATION_GIMBAL")
            if len(roots) > 1:
                col.label(text="%d bones have no parent, and only the first is read: %s"
                               % (len(roots), ", ".join(roots[:3])), icon="ERROR")

        f = _bone_flags(pb)
        col = lay.column(align=True)
        if f is None:
            col.label(text="no flags stashed -- an export keeps the file's own",
                      icon="INFO")
            return
        col.prop(pb, '["%s"]' % FLAGS_ATTR, text="flags")
        col.prop(pb, "vtmb_from_root")
        if not f & USED_BY_MASK:
            col.label(text="no bit of the 0xfffc used-by mask: this bone gets no matrix "
                           "and anything skinned to it draws nothing", icon="ERROR")
        if f & BONE_ALWAYS_PROCEDURAL:
            col.label(text="always procedural, so the file carries a proctype record "
                           "this addon neither reads nor writes", icon="INFO")

        g = pb.get("vtmb_hitgroup")
        col = lay.column(align=True)
        if g is None:
            col.label(text="hit group by name, from the bone's own name", icon="INFO")
        else:
            col.prop(pb, '["vtmb_hitgroup"]', text="Hit group")
            col.label(text="stashed on the pose bone, where a scratch export reads it. "
                           "Nothing else does -- a re-export carries the file's own",
                      icon="INFO")


def _bone_pair(lay, key, value):
    row = lay.row()
    row.label(text=key)
    row.label(text=value)


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
    """The pose bone behind the Bone tab's active bone, or None.

    Edit mode leaves `context.bone` None and puts the active bone in `context.edit_bone`,
    which is why Blender's own bone panels poll `context.bone or context.edit_bone`
    (`bl_ui/properties_data_bone.py:19`). A bone added in Edit mode has no pose bone until
    the mode is left, so that case still returns None and the panel stays away from a bone
    with nothing stashed.
    """
    obj = getattr(context, "object", None)
    bone = getattr(context, "bone", None) or getattr(context, "edit_bone", None)
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
           VTMB_OT_add_event, VTMB_OT_remove_event, VTMB_OT_set_param,
           VTMB_UL_actions,
           VTMB_OT_set_skin_family, VTMB_PT_skin_families,
           VTMB_PT_cloth,
           VTMB_OT_add_attachment, VTMB_OT_add_hitbox, VTMB_PT_accessories,
           VTMB_PT_armature, VTMB_PT_poseparams, VTMB_PT_actions, VTMB_PT_action,
           VTMB_PT_action_events, VTMB_PT_action_params,
           VTMB_PT_bone_flags, VTMB_PT_bone]

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
    bpy.types.Object.vtmb_cloth_on = bpy.props.BoolProperty(
        name="Cloth", description="Export this mesh as a cloth object -- the model's "
                                  "whole particle array, so it must use one material",
        get=lambda self: _cloth_get(self, CLOTH),
        set=lambda self, v: _cloth_set(self, CLOTH, v))
    bpy.types.Object.vtmb_cloth_flip_on = bpy.props.BoolProperty(
        name="Flip normals", description="Bit 15 of every particle index, which flips the "
                                         "cloth normal the solver hands the draw path",
        get=lambda self: _cloth_get(self, "vtmb_cloth_flip"),
        set=lambda self, v: _cloth_set(self, "vtmb_cloth_flip", v))
    bpy.types.Object.vtmb_cloth_preset_name = bpy.props.EnumProperty(
        name="Preset", items=_cloth_preset_items,
        description="One of the 17 (scale, s) pairs the shipped objects use. It supplies "
                    "whichever of the three numbers is not overridden below",
        get=lambda self: _preset_get(self),
        set=lambda self, v: _preset_set(self, v))
    bpy.types.Object.vtmb_cloth_pin_text = bpy.props.StringProperty(
        name="Pin group", description="The vertex group naming the pinned particles. The "
                                      "format takes the pin set as a COUNT and the array "
                                      "is ordered pinned-first, so the export reorders "
                                      "the mesh's vertices and nothing else marks a pin",
        get=lambda self: str(self.get("vtmb_cloth_pin_group") or blender_scratch.PIN_GROUP),
        set=lambda self, v: _cloth_text(self, "vtmb_cloth_pin_group", v))
    for _k, _label, _lo, _hi, _desc in CLOTH_NUMBERS:
        setattr(bpy.types.Object, "vtmb_cloth_%s_on" % _k, bpy.props.BoolProperty(
            name=_label, description="Override the preset's %s" % _k,
            get=(lambda k: lambda self: "vtmb_cloth_%s" % k in self)(_k),
            set=(lambda k: lambda self, v: _cloth_num_on(self, k, v))(_k)))
        setattr(bpy.types.Object, "vtmb_cloth_%s_num" % _k, bpy.props.FloatProperty(
            name=_label, description=_desc, min=_lo, max=_hi,
            get=(lambda k: lambda self: float(self.get("vtmb_cloth_%s" % k) or 0.0))(_k),
            set=(lambda k: lambda self, v: self.__setitem__("vtmb_cloth_%s" % k,
                                                            float(v)))(_k)))
    bpy.types.Object.vtmb_action_index = bpy.props.IntProperty(
        name="Action", default=0, min=0,
        description="Which of the blend's actions the VTMB animations list is on. It "
                    "indexes bpy.data.actions and not the filtered view, which is what "
                    "template_list stores")
    bpy.types.Action.vtmb_root_motion_choice = bpy.props.EnumProperty(
        name="Root motion", items=MODE_ITEMS, get=_mode_get, set=_mode_set,
        description="What a re-export does with this action's travel. Not set leaves it "
                    "to the export dialog, which reads the animation this action came "
                    "from -- a block in the file means the engine carries it")
    bpy.types.PoseBone.vtmb_from_root = bpy.props.BoolProperty(
        name="Rotation from root", get=_from_root_get, set=_from_root_set,
        description="Bit 0x2 of the bone flags. The engine composes this bone's world "
                    "rotation from the root's rather than from its parent's, so its "
                    "translation still follows the chain while its orientation does "
                    "not. 525 bones over 4268 shipped models carry it and every one is "
                    "named <bip> Spine1 -- but 251 of 776 Spine1s do not, so this is "
                    "the flag and never the name. Off where no flags are stashed")
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
    if hasattr(bpy.types.Object, "vtmb_action_index"):
        del bpy.types.Object.vtmb_action_index
    names = ["vtmb_cloth_on", "vtmb_cloth_flip_on", "vtmb_cloth_preset_name",
             "vtmb_cloth_pin_text"]
    for k, _l, _lo, _hi, _d in CLOTH_NUMBERS:
        names += ["vtmb_cloth_%s_on" % k, "vtmb_cloth_%s_num" % k]
    for attr in names:
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    for attr in ("vtmb_root_motion_choice", "vtmb_loops", "vtmb_activity_text"):
        if hasattr(bpy.types.Action, attr):
            delattr(bpy.types.Action, attr)
    if hasattr(bpy.types.PoseBone, "vtmb_from_root"):
        del bpy.types.PoseBone.vtmb_from_root
