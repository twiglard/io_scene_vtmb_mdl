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
import re

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
    # Emptying the box stores the empty string rather than dropping the key: absent means
    # the scene never stated one and the sequence keeps whatever the file says, so
    # deleting it here would make clearing a donor's activity impossible.
    action["vtmb_activity"] = str(value).strip()


def _activity_search(self, context, edit_text):
    """The activities this model's own sequences already claim.

    Offered rather than a table of the game's, so no list of names read out of Troika's
    files ships in the addon at all. `SUGGESTION` is what keeps the field a free string,
    which it has to stay: the format stores a name and the engine resolves it, so an
    activity no shipped model uses is still legal.
    """
    arm = _action_armature(context, self)
    if arm is None:
        obj = getattr(context, "object", None)
        arm = obj if obj is not None and obj.type == "ARMATURE" else None
    got = set()
    for seq in (arm.get("vtmb_sequences") if arm is not None else None) or ():
        name = str(seq.get("activity") or "")
        if name and (not edit_text or edit_text.lower() in name.lower()):
            got.add(name)
    return sorted(got)


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
            # Through the import's stamp, since a reordered slot holds a different
            # material and writing by the stamped index would put this family's picture
            # on the wrong mesh.
            moves = blender_export.slot_moves(o)[0]
            for slot, r in enumerate(refs):
                at = slot if moves is None else moves.get(slot)
                if at is None or at >= len(o.data.materials):
                    continue
                ref = rows[self.family][r] if 0 <= r < len(rows[self.family]) else r
                name = str(mats[ref]) if 0 <= ref < len(mats) else ""
                mat = bpy.data.materials.get(name) if name else None
                if mat is None:
                    missing.append(name or "record %d" % ref)
                    continue
                if o.data.materials[at] is not mat:
                    o.data.materials[at] = mat
                    moved += 1
            # Re-stamped because this is the one thing that replaces a slot's material on
            # purpose: leaving the old stamp would make the next export read every slot
            # as renamed.
            o["vtmb_slot_mats"] = [mm.name if mm else "" for mm in o.data.materials]
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


def cloth_flips(obj):
    """(vertices the flip attribute sets, total), or None where the object's flag covers
    the whole mesh. Bit 15 of `+0x34` negates that vertex's normal and 63 of the 84
    shipped meshes set it on some vertices and not others, so the import stamps it per
    vertex; Blender's own attribute editing is what edits one."""
    me = obj.data
    att = me.attributes.get("vtmb_cloth_flip")
    if att is None or len(att.data) != len(me.vertices):
        return None
    buf = [0] * len(me.vertices)
    att.data.foreach_get("value", buf)
    return sum(1 for v in buf if v), len(me.vertices)


def cloth_sigmas(obj):
    """(lowest, highest, edges) the per-edge sigma attribute holds, or None where the
    object has none. 28 of the 59 shipped row-0 objects vary sigma spring by spring, and
    group 0 is the face list's edge set, so the import stamps one float per edge; the
    number below is the whole-object override and writing it flattens them."""
    me = obj.data
    att = me.attributes.get("vtmb_cloth_sigma")
    if (att is None or att.domain != "EDGE" or len(att.data) != len(me.edges)
            or not len(me.edges)):
        return None
    buf = [0.0] * len(me.edges)
    att.data.foreach_get("value", buf)
    return min(buf), max(buf), len(buf)


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
    flips = cloth_flips(obj)
    if flips is None:
        lay.prop(obj, "vtmb_cloth_flip_on")
    else:
        lay.label(text="Flip normals: %d of %d vertices" % flips)
        lay.label(text="edited per vertex, in the vtmb_cloth_flip attribute")
    numbers, over = cloth_numbers(obj)
    if numbers is None:
        lay.label(text="preset %r is not one of the %d"
                       % (str(obj.get("vtmb_cloth_preset")),
                          len(blender_scratch.cloth_mod.PRESETS)), icon="ERROR")
        return
    sigmas = cloth_sigmas(obj)
    if sigmas is not None:
        lay.label(text="Sigma: %g to %g over %d edges" % sigmas)
        lay.label(text="edited per edge, in the vtmb_cloth_sigma attribute; the number "
                       "below overrides all of them")
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

    # An enum over the eight groups both reference trees name and the corpus uses. 8 and 9
    # ship on `mingxiao.mdl` alone and have no name anywhere, so they are reachable by
    # typing the number into the box's own panel and not from here.
    group: bpy.props.EnumProperty(
        name="Group", default="0",
        items=[(str(g), "%d -- %s" % (g, mdl_mod.hitgroup_name(g)), "") for g in range(8)])
    set_index: bpy.props.IntProperty(name="Set", default=0, min=0)
    set_name: bpy.props.StringProperty(
        name="Set name", default="default",
        description="Only used where the set does not exist yet. 4444 of the 4445 shipped "
                    "models call theirs 'default'; mingxiao.mdl is the one with seven")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and obj.type == "ARMATURE"
                and context.active_bone is not None)

    def execute(self, context):
        arm_obj, bone = context.object, context.active_bone
        _attach, boxes, names = accessories_of(arm_obj)
        if self.set_index not in names:
            root = bpy.data.objects.new("%s.%s" % (arm_obj.name, self.set_name), None)
            root.empty_display_type = "PLAIN_AXES"
            _link_beside(arm_obj, root)
            root.parent = arm_obj
            root["vtmb_hitboxset"] = self.set_name
            root["vtmb_hitboxset_index"] = self.set_index
        group = int(self.group)
        obj = bpy.data.objects.new(
            "%s.%s" % (arm_obj.name, mdl_mod.hitgroup_name(group)), None)
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
        obj["vtmb_hitbox_group"] = group
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
    for k in sorted(set(boxes) | set(names)):
        kids = boxes.get(k, ())
        lay.label(text="%s: %d box%s" % (names.get(k, "default"), len(kids),
                                         "" if len(kids) == 1 else "es"),
                  icon="MESH_CUBE")
        for obj in kids:
            lay.label(text="    %s on %s"
                      % (mdl_mod.hitgroup_name(obj.get("vtmb_hitbox_group") or 0),
                         obj.parent_bone or "no bone"))


def draw_face(lay, context, arm_obj):
    """The one mouth record as properties, and the eyeballs named rather than indexed.

    The mouth's bone and flex live in the `vtmb_mouth` stash as NAMES and never as indices:
    over the 200 shipped carriers the bone index is 6 on 187, 12 on 8, 7 on 3 and 14 on 2
    and the flexdesc index is 16 on 196 and 0 on 4, while the names are `Bip01 Head` and
    `mouth` on 200 of 200 and forward is (0, -1, 0) on 200 of 200.

    That unanimity is Troika's rig and not the format, which is why all three are drawn:
    the 200 carriers span 113 distinct bone sets but every one holds the whole Biped core,
    504 models carry a `Bip01 Head` and only 200 carry a mouth, and
    CStudioRender::R_MouthSetupVertexShader reads all three -- the bone as an index into
    m_BoneToWorld, forward rotated by that matrix into the material's $forward, the flex as
    an index into m_FlexWeights -- so a rig that is not a Biped needs every one of them.

    The eyeball's own numbers are on the eyeball, which VTMB_PT_eyeball draws.
    """
    mouth = arm_obj.get("vtmb_mouth")
    if not mouth:
        lay.label(text="no mouth", icon="INFO")
    else:
        box = lay.box()
        box.label(text="mouth", icon="USER")
        # A nested id-property path is a real RNA path in 5.2: `prop` and `prop_search`
        # both take `["vtmb_mouth"]["bone"]` and a write through one reaches the stash.
        # `prop` on an absent key still raises inside draw(), and the import is the only
        # thing that stamps these, so a key missing is an older .blend and gets a label.
        if mouth.get("bone") is None:
            box.label(text="bone: not stashed", icon="INFO")
        else:
            box.prop_search(arm_obj, '["vtmb_mouth"]["bone"]', *bone_search(arm_obj),
                            text="Bone")
        for k, label in (("forward", "Forward"), ("flex", "Jaw flex")):
            if mouth.get(k) is None:
                box.label(text="%s: not stashed" % label, icon="INFO")
            else:
                box.prop(arm_obj, '["vtmb_mouth"]["%s"]' % k, text=label)
        box.label(text="a flex name this file has not got is added to it, and the export "
                       "says so", icon="INFO")
    eyes = blender_export.eyeball_objects(arm_obj)
    n = sum(len(v) for v in eyes.values())
    if not n:
        lay.label(text="no eyeball", icon="INFO")
        return
    for model in sorted(eyes):
        for obj in eyes[model]:
            lids = list(obj.get("vtmb_eyeball_lidflexes") or ())
            side = ("left" if any("_left" in str(x) for x in lids)
                    else "right" if any("_right" in str(x) for x in lids)
                    else "no lid flexes")
            lay.label(text="eye %d on %s: %s, iris %s"
                           % (int(obj.get("vtmb_eyeball") or 0),
                              obj.parent_bone or "no bone", side,
                              str(obj.get("vtmb_eyeball_iris") or "?")),
                      icon="HIDE_OFF")


class VTMB_PT_face(bpy.types.Panel):
    bl_label = "VTMB eyeballs and mouth"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        self.layout.use_property_split = False
        draw_face(self.layout, context, context.object)


EYE_LID_LABELS = ("upper lowerer", "upper neutral", "upper raiser",
                  "lower lowerer", "lower neutral", "lower raiser",
                  "upper lid", "lower lid")


def eyeball_scale(obj):
    """The model scale the eyeball's own armature carries, 1.0 where it carries none."""
    arm = obj.parent
    return float((arm.get("vtmb_scale", 1.0) if arm is not None else 1.0) or 1.0)


def draw_eyeball(lay, obj):
    """One eyeball record, every field the format varies drawn as a property.

    The radius is the empty's own display size and never `vtmb_eyeball_radius`: the import
    draws the record as a SPHERE of exactly that size, so resizing the sphere is the edit a
    person makes, and a key that won would make the viewport lie about the number written.
    The key is what the file held and is shown beside it where the two disagree.

    zoffset, the texture slot and the two pitch/yaw pairs get one label and no widget,
    each being 0 on all 602 shipped records with the export writing zero -- the reason
    `mstudioattachment_t.type` got none either.
    """
    lay.label(text="eyeball %d of model %r"
                   % (int(obj.get("vtmb_eyeball") or 0),
                      str(obj.get("vtmb_eyeball_model") or "")), icon="HIDE_OFF")
    lay.label(text="bone: %s" % (obj.parent_bone or "not parented to a bone"))
    scale = eyeball_scale(obj)
    lay.prop(obj, "empty_display_size", text="Radius")
    key = obj.get("vtmb_eyeball_radius")
    if key is not None and abs(float(key) - obj.empty_display_size / scale) > 1e-6:
        lay.label(text="the file held %.4f; the sphere is what gets written"
                       % float(key), icon="INFO")
    # `lay.prop` on an absent ID key raises inside draw(), and every one of these is
    # stamped by the import alone -- nothing here creates an eyeball -- so a key missing
    # is an older .blend and gets the value as a label instead. The two aim points are
    # 3-float ID-property arrays and `["key"]` is the right path for one: it is what
    # Blender's own Custom Properties panel uses, rna_prop_ui.py:228, and its MAX_DISPLAY_ROWS
    # cutout at 8 elements is well above three.
    for k, label in (("vtmb_eyeball_iris", "Iris material"),
                     ("vtmb_eyeball_iris_scale", "Iris scale"),
                     ("vtmb_eyeball_glint", "Glint material"),
                     ("vtmb_eyeball_uppertarget", "Upper aim"),
                     ("vtmb_eyeball_lowertarget", "Lower aim")):
        if obj.get(k) is None:
            lay.label(text="%s: not stashed" % label, icon="INFO")
        else:
            lay.prop(obj, '["%s"]' % k, text=label)
    lay.label(text="a material name this file has not got is added to it, and the export "
                   "says so", icon="INFO")
    lay.prop(obj, "vtmb_eyeball_lids", text="Lid flexes")
    lids = [str(x) for x in (obj.get("vtmb_eyeball_lidflexes") or ())]
    if not lids:
        lay.label(text="no lid flexes, which 210 of 602 shipped records also have",
                  icon="INFO")
        return
    box = lay.box()
    box.label(text="lid flexes")
    for label, name in zip(EYE_LID_LABELS, lids):
        box.label(text="%s: %s" % (label, name or "-"))


def _lid_side(obj):
    """"none", "right", "left", or "other" for a set matching no shipped one."""
    lids = tuple(str(x) for x in (obj.get("vtmb_eyeball_lidflexes") or ()))
    if not any(lids):
        return "none"
    for side, names in blender_export.EYE_LID_FLEXES.items():
        if lids == names:
            return side
    return "other"


def _lids_get(self):
    return ("none", "right", "left", "other").index(_lid_side(self))


def _lids_set(self, v):
    # "other" writes nothing: it is what an unrecognised set reads back as, and a scene
    # holding one has said something this panel cannot restate.
    side = ("none", "right", "left", "other")[int(v)]
    if side == "other":
        return
    self["vtmb_eyeball_lidflexes"] = list(
        blender_export.EYE_LID_FLEXES.get(side) or ())


def draw_hitbox(lay, obj):
    """One hitbox empty's own two numbers.

    `vtmb_hitbox_group` is what `accessory_objects` classifies an empty as a box by, so the
    poll below and the exporter agree by construction and the key must not be removed.
    The group's meaning is Source's, `mdl.HITGROUPS` off both reference trees, and the
    name is drawn beside the number so the field says which body part it is.  Nothing in
    the format caps it: `mingxiao.mdl` ships 8 and 9, which neither tree names, so those
    read back as their number.
    """
    lay.label(text="bone: %s" % (obj.parent_bone or "no bone"), icon="BONE_DATA")
    row = lay.row(align=True)
    row.prop(obj, '["vtmb_hitbox_group"]', text="Hit group")
    row.label(text=mdl_mod.hitgroup_name(obj.get("vtmb_hitbox_group") or 0))
    # Both creation paths stamp the ordinal -- the importer at blender_import.py:674 and the
    # add operator at :528 -- but `lay.prop` on an absent key raises inside draw(), and
    # `accessory_objects` reads it with a `or 0` default, so a box without one is legal.
    if obj.get("vtmb_hitboxset_index") is None:
        lay.label(text="set 0, no ordinal stamped", icon="MESH_CUBE")
        return
    lay.prop(obj, '["vtmb_hitboxset_index"]', text="Set")
    k = int(obj.get("vtmb_hitboxset_index") or 0)
    arm_obj = obj.parent
    name = None
    if arm_obj is not None:
        for sib in arm_obj.children_recursive:
            if (sib.get("vtmb_hitboxset") is not None
                    and int(sib.get("vtmb_hitboxset_index") or 0) == k):
                name = str(sib["vtmb_hitboxset"])
                break
    lay.label(text="set %d: %s" % (k, name or "default, unnamed"), icon="MESH_CUBE")
    lay.label(text="%s -- Source's own HITGROUP names, which the corpus agrees with; "
                   "nothing in this format says so" % ", ".join(
                       "%d %s" % (g, mdl_mod.HITGROUPS[g]) for g in range(8)),
              icon="INFO")


class VTMB_PT_hitbox(bpy.types.Panel):
    bl_label = "VTMB hitbox"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return (context.object is not None
                and context.object.get("vtmb_hitbox_group") is not None)

    def draw(self, context):
        self.layout.use_property_split = False
        draw_hitbox(self.layout, context.object)


class VTMB_PT_eyeball(bpy.types.Panel):
    bl_label = "VTMB eyeball"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return (context.object is not None
                and context.object.get("vtmb_eyeball") is not None)

    def draw(self, context):
        self.layout.use_property_split = False
        draw_eyeball(self.layout, context.object)


FLEX_KEYS = {"CONTROLLER": "vtmb_flexcontrollers", "RULE": "vtmb_flexrules"}
FLEX_KIND_ITEMS = [
    ("CONTROLLER", "Flex controller", "`<type> [range <min> <max>] <name>`"),
    ("RULE", "Flex rule", "`<flexdesc> = <expression>`"),
]


def _flex_lines(arm_obj, kind):
    return [str(x) for x in (arm_obj.get(FLEX_KEYS[kind]) or ())]


def _write_flex(arm_obj, kind, lines):
    """Reassign the whole list, which is the only write a string ID property takes.

    Reading one back hands over a plain Python copy, so `arm[key][0] = x` writes into the
    copy and is discarded -- Blender's own Custom Properties list cannot edit one either.
    """
    arm_obj[FLEX_KEYS[kind]] = list(lines)


def _flex_type_search(self, context, edit_text):
    return blender_export.FLEX_TYPES


def _flex_compose(kind, ctype, name, expr, rng=(0.0, 1.0)):
    """The line these fields spell, or ValueError saying what is wrong with them."""
    if kind == "CONTROLLER":
        line = blender_export.flex_controller_line(ctype.strip(), rng[0], rng[1],
                                                   name.strip())
        blender_export.flex_controller_fields(line)
        return line
    if not name.strip():
        raise ValueError("a flex rule names the flex it drives, and this one names none")
    if "=" in name:
        raise ValueError("a flex name cannot carry `=`")
    if not expr.strip():
        raise ValueError("flex rule %r has no expression" % name.strip())
    return "%s = %s" % (name.strip(), expr.strip())


def _flex_cited(arm_obj, name):
    """The first flex rule naming this controller, or None."""
    pat = re.compile(r"\b%s\b" % re.escape(name))
    return next((r for r in _flex_lines(arm_obj, "RULE") if pat.search(r)), None)


def _flex_refuse(arm_obj, index, line):
    """Why this controller line cannot stand at `index`, or None.

    A rule resolves a controller by NAME and by nothing else -- the engine patches `link`
    at load from a process-global name table -- so a duplicate drives the first silently
    and a rename leaves every rule that cited the old name driving nothing.
    """
    lines = _flex_lines(arm_obj, "CONTROLLER")
    try:
        name = blender_export.flex_controller_fields(line)[3]
    except ValueError as e:
        return str(e)
    for k, other in enumerate(lines):
        if k == index:
            continue
        try:
            if blender_export.flex_controller_fields(other)[3] == name:
                return ("another flex controller is already named %r, and a rule names a "
                        "controller by its name alone" % name)
        except ValueError:
            pass
    if 0 <= index < len(lines):
        was = blender_export.flex_controller_fields(lines[index])[3]
        cited = _flex_cited(arm_obj, was) if was != name else None
        if cited is not None:
            return ("flex rule %r names %r, so renaming it to %r would leave that rule "
                    "driving nothing" % (cited, was, name))
    return None


def _flex_armature(context):
    obj = getattr(context, "object", None)
    return obj if obj is not None and obj.type == "ARMATURE" else None


class VTMB_OT_edit_flex_line(bpy.types.Operator):
    bl_idname = "vtmb.edit_flex_line"
    bl_label = "Edit flex line"
    bl_description = ("Rewrite one flex controller or flex rule. A controller keeps the "
                      "range it had: it is 0..1 on all 8649 shipped controllers")
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Array", items=FLEX_KIND_ITEMS,
                                 default="CONTROLLER")
    index: bpy.props.IntProperty(name="Line", default=0, min=0)
    ctype: bpy.props.StringProperty(
        name="Type", default="mouth", search=_flex_type_search,
        search_options={"SORT", "SUGGESTION"},
        description="One of the seven words the corpus carries, or any other -- the "
                    "field is a string in the file and not an enum")
    name: bpy.props.StringProperty(
        name="Name", default="",
        description="A controller's name, or the flex a rule drives. That name is the "
                    "only identity either has")
    expr: bpy.props.StringProperty(name="Expression", default="")

    @classmethod
    def poll(cls, context):
        return _flex_armature(context) is not None

    def invoke(self, context, event):
        lines = _flex_lines(_flex_armature(context), self.kind)
        if 0 <= self.index < len(lines):
            line = lines[self.index]
            if self.kind == "CONTROLLER":
                try:
                    fields = blender_export.flex_controller_fields(line)
                    self.ctype, self.name = fields[0], fields[3]
                except ValueError:
                    self.name = line
            elif "=" in line:
                self.name, self.expr = (x.strip() for x in line.split("=", 1))
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        col = self.layout.column()
        if self.kind == "CONTROLLER":
            col.prop(self, "ctype")
            col.prop(self, "name")
        else:
            col.prop(self, "name", text="Flex")
            col.prop(self, "expr")

    def execute(self, context):
        arm_obj = _flex_armature(context)
        lines = _flex_lines(arm_obj, self.kind)
        if not 0 <= self.index < len(lines):
            self.report({"ERROR"}, "there is no %s line %d"
                        % (self.kind.lower(), self.index))
            return {"CANCELLED"}
        rng = (0.0, 1.0)
        if self.kind == "CONTROLLER":
            try:
                fields = blender_export.flex_controller_fields(lines[self.index])
                rng = (fields[1], fields[2])
            except ValueError:
                pass
        try:
            line = _flex_compose(self.kind, self.ctype, self.name, self.expr, rng)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if self.kind == "CONTROLLER":
            why = _flex_refuse(arm_obj, self.index, line)
            if why:
                self.report({"ERROR"}, why)
                return {"CANCELLED"}
        lines[self.index] = line
        _write_flex(arm_obj, self.kind, lines)
        self.report({"INFO"}, "line %d is now %r" % (self.index, line))
        return {"FINISHED"}


class VTMB_OT_add_flex_line(bpy.types.Operator):
    bl_idname = "vtmb.add_flex_line"
    bl_label = "Add flex line"
    bl_description = "Append a flex controller or a flex rule"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Array", items=FLEX_KIND_ITEMS,
                                 default="CONTROLLER")
    ctype: bpy.props.StringProperty(
        name="Type", default="mouth", search=_flex_type_search,
        search_options={"SORT", "SUGGESTION"})
    name: bpy.props.StringProperty(name="Name", default="")
    expr: bpy.props.StringProperty(name="Expression", default="")

    @classmethod
    def poll(cls, context):
        return _flex_armature(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        col = self.layout.column()
        if self.kind == "CONTROLLER":
            col.prop(self, "ctype")
            col.prop(self, "name")
        else:
            col.prop(self, "name", text="Flex")
            col.prop(self, "expr")

    def execute(self, context):
        arm_obj = _flex_armature(context)
        lines = _flex_lines(arm_obj, self.kind)
        try:
            line = _flex_compose(self.kind, self.ctype, self.name, self.expr)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if self.kind == "CONTROLLER":
            why = _flex_refuse(arm_obj, len(lines), line)
            if why:
                self.report({"ERROR"}, why)
                return {"CANCELLED"}
        _write_flex(arm_obj, self.kind, lines + [line])
        self.report({"INFO"}, "appended %r" % line)
        return {"FINISHED"}


class VTMB_OT_remove_flex_line(bpy.types.Operator):
    bl_idname = "vtmb.remove_flex_line"
    bl_label = "Remove flex line"
    bl_description = "Drop one flex controller or flex rule"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(name="Array", items=FLEX_KIND_ITEMS,
                                 default="CONTROLLER")
    index: bpy.props.IntProperty(name="Line", default=0, min=0)

    @classmethod
    def poll(cls, context):
        return _flex_armature(context) is not None

    def execute(self, context):
        arm_obj = _flex_armature(context)
        lines = _flex_lines(arm_obj, self.kind)
        if not 0 <= self.index < len(lines):
            self.report({"ERROR"}, "there is no %s line %d"
                        % (self.kind.lower(), self.index))
            return {"CANCELLED"}
        if self.kind == "CONTROLLER":
            try:
                was = blender_export.flex_controller_fields(lines[self.index])[3]
            except ValueError:
                was = None
            cited = _flex_cited(arm_obj, was) if was else None
            if cited is not None:
                self.report({"ERROR"}, "flex rule %r names %r, so dropping that controller "
                                       "would leave the rule driving nothing" % (cited, was))
                return {"CANCELLED"}
        gone = lines.pop(self.index)
        _write_flex(arm_obj, self.kind, lines)
        self.report({"INFO"}, "dropped %r" % gone)
        return {"FINISHED"}


def draw_flex(lay, arm_obj):
    """The flex controllers and the flex rules, as the QC lines they are.

    Both are stored as text because studiomdl's own syntax is the only syntax either has
    -- Option_Flexcontroller and Option_Flexrule, studiomdl.cpp:2875 and :2914. A string
    list in an ID property cannot be edited in place, so every write here reassigns the
    whole list through `_write_flex` and the three operators are what reach it.
    """
    ctls = _flex_lines(arm_obj, "CONTROLLER")
    rules = _flex_lines(arm_obj, "RULE")
    box = lay.box()
    box.label(text="%d flex controller(s)" % len(ctls), icon="DRIVER")
    by_type = {}
    for line in ctls:
        by_type.setdefault(line.split(None, 1)[0] if line else "?", []).append(line)
    if by_type:
        box.label(text=", ".join("%s %d" % (t, len(by_type[t])) for t in sorted(by_type)))
    for k, line in enumerate(ctls):
        row = box.row(align=True)
        row.label(text=line)
        op = row.operator("vtmb.edit_flex_line", text="", icon="GREASEPENCIL")
        op.kind, op.index = "CONTROLLER", k
        op = row.operator("vtmb.remove_flex_line", text="", icon="X")
        op.kind, op.index = "CONTROLLER", k
    box.operator("vtmb.add_flex_line", text="Add controller",
                 icon="ADD").kind = "CONTROLLER"
    # 0.0..1.0 on all 8649 shipped controllers, so the range is stated and not offered.
    box.label(text="every shipped controller ranges 0..1, and an edit keeps the range "
                   "the line already has", icon="INFO")

    box = lay.box()
    box.label(text="%d flex rule(s)" % len(rules), icon="SHAPEKEY_DATA")
    for k, line in enumerate(rules):
        row = box.row(align=True)
        row.label(text=line)
        op = row.operator("vtmb.edit_flex_line", text="", icon="GREASEPENCIL")
        op.kind, op.index = "RULE", k
        op = row.operator("vtmb.remove_flex_line", text="", icon="X")
        op.kind, op.index = "RULE", k
    box.operator("vtmb.add_flex_line", text="Add rule", icon="ADD").kind = "RULE"


class VTMB_PT_flex(bpy.types.Panel):
    bl_label = "VTMB flex controllers"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE"

    def draw(self, context):
        self.layout.use_property_split = False
        draw_flex(self.layout, context.object)


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


def draw_action_fields(lay, act, arm_obj=None):
    """The three per-action fields, wherever they are hosted.

    Two panels draw them -- the Action editor's sidebar, which is about the action on
    screen, and the Object Data tab's list, which is about the model's whole set -- and
    `operator-check.py` reads the `.prop(act, ...)` calls out of this one function, so a
    field added here is a field both hosts get and the check counts.

    `arm_obj` is only what `_activity_line` resolves the sequence stash through. The Action
    editor has no armature in its context and passes what `_action_armature` finds, which is
    None where no armature's stash names the action.
    """
    col = lay.column(align=True)
    col.prop(act, "vtmb_root_motion_choice", text="")
    if act.get(blender_export.MODE_ATTR) is None:
        col.label(text="    → " + _fallback_line(act))

    col = lay.column(align=True)
    col.prop(act, "vtmb_loops")
    col.prop(act, "vtmb_activity_text")
    if act.get("vtmb_activity") is None:
        col.label(text="    → " + _activity_line(arm_obj, act))

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
        draw_action_fields(lay, act, obj)
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
        draw_action_fields(lay, act, _action_armature(context, act))


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


FLT_MAX = 3.4028234663852886e+38


def _melee_text(lo, hi):
    """`(FLT_MIN, FLT_MAX)` is the unbounded default, on 13431 of 14012 sequences."""
    if hi >= FLT_MAX:
        return "unbounded"
    return "%g .. %g" % (lo, hi)


def draw_action_seqtail(lay, context, act):
    """The sequence record past the blend grid: what drags along, what melee reads, and
    the four activity/sequence names the load-time fixup resolves.

    Read-only. Every field here is an authored number with no default a scene can supply,
    so the panel says what the file holds and the export writes it back unchanged.
    """
    arm_obj = _action_armature(context, act)
    if arm_obj is None:
        lay.label(text="no imported sequence names this action", icon="INFO")
        return
    _k, seq = _action_seq(arm_obj, act)

    al = [str(x) for x in seq.get("autolayers") or ()]
    box = lay.box()
    box.label(text="auto-layers: %d" % len(al))
    for name in al:
        box.label(text=name, icon="ACTION")

    node = [int(x) for x in seq.get("node") or (0, 0, 0)]
    phase = [float(x) for x in seq.get("phase") or (0.0, 0.0)]
    box = lay.box()
    box.label(text="transition node %d → %d, flags %d" % tuple(node))
    box.label(text="phase %g .. %g" % tuple(phase))

    box = lay.box()
    mr = [float(x) for x in seq.get("meleerange") or (0.0, 0.0)]
    box.label(text="melee range: %s" % _melee_text(*mr))
    mask = int(seq.get("seqselectmask", -1))
    box.label(text="select mask: %s" % ("never" if mask == -1 else "0x%x" % mask))
    st = int(seq.get("statrequired", -1))
    box.label(text="stat required: %s" % ("none" if st < 0 else str(st)))
    cw = [float(x) for x in seq.get("cyclewindow") or (0.0, 1.0, 1.0)]
    box.label(text="cycle window %g .. %g, threshold %g" % tuple(cw))

    names = [(k, str(seq.get(k) or "")) for k in ("dodge", "block", "name2e8", "name2ec")]
    if any(v for _k2, v in names):
        box = lay.box()
        for key, v in names:
            if v:
                box.label(text="%s: %s" % (key, v))

    kbs = list(seq.get("knockbacks") or ())
    hvs = list(seq.get("hitvolumes") or ())
    box = lay.box()
    box.label(text="%d knockback(s), %d hit volume(s)" % (len(kbs), len(hvs)))
    for i, k in enumerate(kbs):
        acts = [str(x) for row in (k.get("activities") or ()) for x in row if x]
        box.label(text="%d  %s  end %g  %s"
                       % (i, str(k.get("bone") or "?"), float(k.get("cycleend") or 0.0),
                          ", ".join(acts) if acts else "-"))


class VTMB_PT_action_seqtail(bpy.types.Panel):
    bl_label = "VTMB sequence tail"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "VTMB"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return panel_action(context) is not None

    def draw(self, context):
        self.layout.use_property_split = False
        draw_action_seqtail(self.layout, context, panel_action(context))


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


def bone_search(arm_obj):
    """The collection a bone picker on this armature searches, as (owner, property name).

    In Edit mode the live bones are `edit_bones`; `armature.bones` is still populated but
    frozen at the last Object-mode state, so a picker aimed at it offers a name a rename has
    already taken away and omits the one it gave. Outside Edit mode `edit_bones` is empty.
    The bone panels draw in Edit mode the way Blender's own do, so the mode decides.
    """
    arm = arm_obj.data
    return arm, ("edit_bones" if arm_obj.mode == "EDIT" else "bones")


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


def spring_unk08_line(pb):
    """What the panel says about record `+0x08`, or None where the bone carries no chain.

    Drawn and not edited, and the reason is a measurement rather than caution: no code in
    any of the four modules reads it. `ref/2531/studio-verified.h` carries the sweep --
    24 341 candidate `[reg+0x8]` sites narrowed to ten by requiring a `+0x18c` operand and
    a 0x1c stride nearby, all ten refuted by hand, and a disp32 scan over every other PE in
    the install finding no walker either. The corpus writes 60.0 on 539 of its 600 records,
    9.0 on 41 and 30.0 on 20.

    The word is *carried* and never *refused*: `apply_springbones` retunes all five of
    `blender_export.SPRING_KEYS`, so a value set by hand or by a script does reach the file.
    """
    val = pb.get("vtmb_spring_unk08")
    if val is None:
        return None
    return ("unk08 %g, carried unchanged -- nothing in the game reads record +0x08"
            % float(val))


# Every key a chain puts on its pose bone. `vtmb_spring_index` is the ordinal the import
# stamped and `vtmb_spring_new` is what this panel writes for a chain the scene authored;
# `vtmb_spring_removed` is the tombstone that says drop the record -- deleting the keys
# cannot, an unclaimed record being carried verbatim.
SPRING_ALL = ("vtmb_spring_index", "vtmb_spring_new", "vtmb_spring_removed",
              "vtmb_spring_end",
              "vtmb_spring_disabled") + tuple(k for _a, k, _f in blender_export.SPRING_KEYS)


def _has_chain(pb):
    return (pb.get("vtmb_spring_index") is not None
            or pb.get("vtmb_spring_gravity") is not None)


class VTMB_OT_add_spring_bone(bpy.types.Operator):
    bl_idname = "vtmb.add_spring_bone"
    bl_label = "Add spring bone chain"
    bl_description = ("Author a spring bone chain starting at this bone. It runs "
                      "first-child to the leaf, and the five fields start at the most "
                      "common value the shipped models carry for each. The export appends "
                      "the record; the no-donor path writes it too")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        pb = _spring_bone(context)
        return pb is not None and not _has_chain(pb)

    def execute(self, context):
        pb = _spring_bone(context)
        for key, value in blender_export.spring_defaults().items():
            pb[key] = value
        pb["vtmb_spring_end"] = ""
        pb["vtmb_spring_disabled"] = False
        pb["vtmb_spring_new"] = True
        self.report({"INFO"}, "spring bone chain authored on %r, and appended to the file "
                              "by the next export" % pb.name)
        return {"FINISHED"}


class VTMB_OT_remove_spring_bone(bpy.types.Operator):
    bl_idname = "vtmb.remove_spring_bone"
    bl_label = "Remove spring bone chain"
    bl_description = ("Drop this chain. One the file carries is marked here and removed by "
                      "the next export, which renumbers every later chain; one this scene "
                      "authored and has not written yet is taken off the bone outright")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        pb = _spring_bone(context)
        return (pb is not None and _has_chain(pb)
                and pb.get("vtmb_spring_removed") is None)

    def execute(self, context):
        pb = _spring_bone(context)
        if pb.get("vtmb_spring_index") is None:
            # Nothing in the file answers to this one, so there is no record to tombstone.
            for key in SPRING_ALL:
                if key in pb:
                    del pb[key]
            self.report({"INFO"}, "the chain authored on %r is gone. Nothing was written, "
                                  "so the file has nothing to drop" % pb.name)
            return {"FINISHED"}
        pb["vtmb_spring_removed"] = True
        self.report({"INFO"}, "chain %d is marked for removal. Untick the box to keep it"
                    % int(pb["vtmb_spring_index"]))
        return {"FINISHED"}


class VTMB_PT_bone(bpy.types.Panel):
    bl_label = "VTMB spring bone"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "bone"

    @classmethod
    def poll(cls, context):
        pb = _spring_bone(context)
        return pb is not None and (_has_chain(pb) or pb.get(FLAGS_ATTR) is not None
                                   or pb.get("vtmb_bone_name") is not None)

    def draw(self, context):
        pb = _spring_bone(context)
        lay = self.layout
        lay.use_property_split = True

        if not _has_chain(pb):
            col = lay.column(align=True)
            col.operator(VTMB_OT_add_spring_bone.bl_idname, icon="ADD")
            col.label(text="no chain starts at this bone", icon="INFO")
            return

        col = lay.column(align=True)
        k = pb.get("vtmb_spring_index")
        if k is not None:
            col.label(text="chain %d" % int(k), icon="PHYSICS")
        elif pb.get("vtmb_spring_new"):
            col.label(text="a new chain -- the export appends it", icon="PHYSICS")
        else:
            # The fields are here and no record answers to them, which is what deleting the
            # ordinal leaves. The export carries that record as the file has it.
            col.label(text="no record claims this bone -- these fields reach no file",
                      icon="ERROR")
        rm = pb.get("vtmb_spring_removed")
        if rm is None:
            col.operator(VTMB_OT_remove_spring_bone.bl_idname, icon="X")
        else:
            col.prop(pb, '["vtmb_spring_removed"]', text="Remove on export")
            if rm:
                col.label(text="dropped on export, and every later chain renumbers, so "
                               "the ordinals here move with it", icon="ERROR")
        if pb.get("vtmb_spring_end") is None:
            # Absent is a third state and prop_search cannot draw a key that is not there:
            # the export carries the file's own end bone rather than repointing the chain,
            # so saying so beats an empty box that reads as "no end bone".
            col.label(text="ends where the file says -- this scene has never named a bone")
        else:
            col.prop_search(pb, '["vtmb_spring_end"]',
                            *bone_search(context.object), text="Ends at")
            col.label(text="empty runs the chain first-child to the leaf")
        off = pb.get("vtmb_spring_disabled")
        if off is None:
            # The third state again, and no widget can draw it: the export carries the
            # file's own switch rather than deciding one.
            col.label(text="switched on or off where the file says -- this scene has "
                           "never said")
        else:
            col.prop(pb, '["vtmb_spring_disabled"]', text="Starts switched off")
            if off:
                col.label(text="and nothing in the game switches it back on: the chain "
                               "is looked up by the raw start bone, which is negative "
                               "here", icon="ERROR")

        col = lay.column(align=True)
        for key, name, _desc in SPRING_FIELDS:
            if pb.get(key) is not None:
                col.prop(pb, '["%s"]' % key, text=name)
        line = spring_unk08_line(pb)
        if line is not None:
            col.label(text=line)

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


def _activity_line(arm_obj, act):
    """What an Activity box the scene has never filled in resolves to.

    Absent and empty are different states -- `_activity_set` stores `""` for "no activity"
    and only an action the panel has never written has no key at all -- and the box draws
    both as empty. So the absent one says what the file holds instead.
    """
    if arm_obj is None:
        return "no armature here names this action"
    seq = _action_seq(arm_obj, act)[1]
    if seq is None:
        return "no sequence of this file names this action"
    return str(seq.get("activity") or "") or "no activity"


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


def _write_stash(arm_obj, stash):
    """Replace the whole sequence stash. An ID-property list is not mutable in place."""
    arm_obj["vtmb_sequences"] = [dict(x) for x in stash]


def _file_anims(arm_obj):
    """The animation names the file carries, as a blend cell has to spell them.

    `vtmb_anim_name` and not the action's own name: Blender deduplicates, so two animations
    called the same thing are `walk` and `walk.001` in the scene, and a cell written under
    the suffixed name would name an animation the file has not got. The stamp is what the
    record was called and `_restamp_renamed_anims` carries it forward over a rename.
    """
    src = arm_obj.get("vtmb_source")
    out = []
    for act in bpy.data.actions:
        if act.get("vtmb_source") != src or act.get("vtmb_anim_index") is None:
            continue
        out.append(str(act.get("vtmb_anim_name") or act.name))
    return sorted(set(out))


def _anim_items(self, context):
    """The enum behind a blend cell: every animation of the armature's own file."""
    act = panel_action(context)
    arm_obj = _action_armature(context, act) if act is not None else None
    names = _file_anims(arm_obj) if arm_obj is not None else []
    # `~none` and not "": an enum identifier has to be non-empty, and the same spelling is
    # what the export dialog's own dropdowns use for "nothing".
    return [(n, n, "") for n in names] or [("~none", "no animation of this file", "")]


def _seq_here(context):
    """(armature, stash index, the stash) for the panel's action, else (None, None, None)."""
    act = panel_action(context)
    if act is None:
        return None, None, None
    arm_obj = _action_armature(context, act)
    if arm_obj is None:
        return None, None, None
    k, _seq = _action_seq(arm_obj, act)
    if k is None:
        return None, None, None
    return arm_obj, k, [dict(x) for x in arm_obj.get("vtmb_sequences") or ()]


class VTMB_OT_rename_sequence(bpy.types.Operator):
    bl_idname = "vtmb.rename_sequence"
    bl_label = "Rename sequence"
    bl_description = ("Give this action's sequence another label. The name is what the "
                      "engine looks a sequence up by and what another sequence's "
                      "auto-layer, dodge and block fields name, so those follow it")
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(name="Label")

    def invoke(self, context, event):
        _arm, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        self.name = str(stash[k].get("label") or "")
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        name = self.name.strip()
        if not name:
            self.report({"ERROR"}, "a sequence label cannot be empty")
            return {"CANCELLED"}
        for j, x in enumerate(stash):
            if j != k and str(x.get("label") or "") == name:
                self.report({"ERROR"}, "sequence %d is already called %r, and the engine "
                                       "looks one up by name" % (j, name))
                return {"CANCELLED"}
        was = str(stash[k].get("label") or "")
        stash[k]["label"] = name
        for x in stash:
            x["autolayers"] = [name if str(v) == was else str(v)
                               for v in x.get("autolayers") or ()]
            for key in ("dodge", "block", "name2e8", "name2ec"):
                if str(x.get(key) or "") == was:
                    x[key] = name
        _write_stash(arm_obj, stash)
        self.report({"INFO"}, "sequence %d: %s -> %s" % (k, was, name))
        return {"FINISHED"}


class VTMB_OT_add_sequence(bpy.types.Operator):
    bl_idname = "vtmb.add_sequence"
    bl_label = "Add sequence"
    bl_description = ("Append a second sequence playing this action's animation. 0 of the "
                      "4445 shipped models carry an animation two sequences cite, and the "
                      "panels here reach the earlier of the two")
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(name="Label")

    def invoke(self, context, event):
        _arm, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        self.name = "%s_2" % str(stash[k].get("label") or "sequence")
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        name = self.name.strip()
        if not name or any(str(x.get("label") or "") == name for x in stash):
            self.report({"ERROR"}, "%r is empty or already a sequence of this file"
                        % self.name)
            return {"CANCELLED"}
        act = panel_action(context)
        anim = str(act.get("vtmb_anim_name") or act.name)
        # `new` is the mark `apply_sequences` appends on, and an added sequence has to be
        # last in the stash because `add_sequence` appends the record.
        # A tail field left out is one `_apply_seq_tail` skips on a None, so
        # `add_sequence`'s own template stands.
        stash.append({"new": 1, "label": name, "activity": "",
                      "flags": int(act.get("vtmb_seq_flags") or 0),
                      "groupsize": [1, 1], "blends": [[anim]],
                      "events": [], "params": [], "autolayers": []})
        _write_stash(arm_obj, stash)
        self.report({"INFO"}, "sequence %d appended, %r, blending %s"
                    % (len(stash) - 1, name, anim))
        return {"FINISHED"}


class VTMB_OT_remove_sequence(bpy.types.Operator):
    bl_idname = "vtmb.remove_sequence"
    bl_label = "Remove sequence"
    bl_description = ("Drop this action's sequence from the file. Its animation stays and "
                      "nothing will play it -- 0 of the 4445 shipped models carry one no "
                      "sequence cites")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        gone = stash.pop(k)
        label = str(gone.get("label") or "")
        if gone.get("new"):
            # Never written, so there is nothing for the export to remove.
            _write_stash(arm_obj, stash)
            self.report({"INFO"}, "dropped %r, which this scene had added" % label)
            return {"FINISHED"}
        tomb = [str(x) for x in arm_obj.get("vtmb_seqs_removed") or ()]
        arm_obj["vtmb_seqs_removed"] = tomb + [label]
        for x in stash:
            x["autolayers"] = [str(v) for v in x.get("autolayers") or ()
                               if str(v) != label]
        _write_stash(arm_obj, stash)
        self.report({"INFO"}, "sequence %d, %r, marked for removal on the next export"
                    % (k, label))
        return {"FINISHED"}


class VTMB_OT_move_sequence(bpy.types.Operator):
    bl_idname = "vtmb.move_sequence"
    bl_label = "Move sequence"
    bl_description = ("Swap this action's sequence with its neighbour. Order is what an "
                      "auto-layer resolves to an index, and those follow the move")
    bl_options = {"REGISTER", "UNDO"}

    up: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        j = k - 1 if self.up else k + 1
        if not 0 <= j < len(stash):
            self.report({"ERROR"}, "sequence %d is already %s of %d"
                        % (k, "first" if self.up else "last", len(stash)))
            return {"CANCELLED"}
        if stash[k].get("new") or stash[j].get("new"):
            self.report({"ERROR"}, "a sequence the scene added is appended by the export, "
                                   "so it cannot be moved before one the file has")
            return {"CANCELLED"}
        stash[k], stash[j] = stash[j], stash[k]
        _write_stash(arm_obj, stash)
        self.report({"INFO"}, "sequence %r is now %d of %d"
                    % (str(stash[j].get("label") or ""), j, len(stash)))
        return {"FINISHED"}


class VTMB_OT_set_blend(bpy.types.Operator):
    bl_idname = "vtmb.set_blend"
    bl_label = "Set blend cell"
    bl_description = ("Name the animation one cell of this sequence's blend grid plays. "
                      "The file holds an index and the stash a name, because an index "
                      "means nothing once the file is re-emitted")
    bl_options = {"REGISTER", "UNDO"}

    x: bpy.props.IntProperty(default=0)
    y: bpy.props.IntProperty(default=0)
    anim: bpy.props.EnumProperty(name="Animation", items=_anim_items)

    def invoke(self, context, event):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        cells = [list(c) for c in stash[k].get("blends") or ()]
        if self.x < len(cells) and self.y < len(cells[self.x]):
            was = str(cells[self.x][self.y])
            if was in _file_anims(arm_obj):
                self.anim = was
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        cells = [[str(v) for v in col] for col in stash[k].get("blends") or ()]
        if not (0 <= self.x < len(cells) and 0 <= self.y < len(cells[self.x])):
            self.report({"ERROR"}, "this sequence has no blend cell %d, %d"
                        % (self.x, self.y))
            return {"CANCELLED"}
        if not self.anim or self.anim == "~none":
            self.report({"ERROR"}, "no animation of this file to name")
            return {"CANCELLED"}
        was, cells[self.x][self.y] = cells[self.x][self.y], self.anim
        stash[k]["blends"] = cells
        _write_stash(arm_obj, stash)
        self.report({"INFO"}, "blend %d, %d: %s -> %s"
                    % (self.x, self.y, was or "(kept)", self.anim))
        return {"FINISHED"}


class VTMB_OT_resize_blends(bpy.types.Operator):
    bl_idname = "vtmb.resize_blends"
    bl_label = "Resize blend grid"
    bl_description = ("How many blends this sequence has along each axis. 13715 of the "
                      "14012 shipped sequences are 1x1; the grid sits inside the 764-byte "
                      "record at a 0x20 stride, so 16 by 16 is the most it holds")
    bl_options = {"REGISTER", "UNDO"}

    gx: bpy.props.IntProperty(name="Blends along X", default=1, min=1, max=16)
    gy: bpy.props.IntProperty(name="Blends along Y", default=1, min=1, max=16)

    def invoke(self, context, event):
        _arm, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        cells = [list(c) for c in stash[k].get("blends") or ()]
        self.gx = max(1, len(cells))
        self.gy = max(1, max((len(c) for c in cells), default=1))
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm_obj, k, stash = _seq_here(context)
        if k is None:
            self.report({"ERROR"}, "no imported sequence names this action")
            return {"CANCELLED"}
        cells = [[str(v) for v in col] for col in stash[k].get("blends") or ()]
        fill = cells[0][0] if cells and cells[0] else ""
        if not fill:
            self.report({"ERROR"}, "this sequence's first blend names no animation, so a "
                                   "new cell has nothing to copy")
            return {"CANCELLED"}
        was = sum(min(len(c), self.gy) for c in cells[:self.gx])
        out = []
        for x in range(self.gx):
            col = list(cells[x]) if x < len(cells) else []
            col = (col + [fill] * self.gy)[:self.gy]
            out.append([c or fill for c in col])
        stash[k]["blends"] = out
        stash[k]["groupsize"] = [self.gx, self.gy]
        _write_stash(arm_obj, stash)
        self.report({"INFO"}, "sequence %d blends %d x %d, %d cell(s) filled with %s"
                    % (k, self.gx, self.gy, self.gx * self.gy - was, fill))
        return {"FINISHED"}


def draw_action_sequence(lay, context, act):
    """The sequence list itself: which record this action drives, its label, its blend grid
    and where it sits among the others.

    Every cross-reference in the stash is a NAME -- a blend cell an animation's, an
    auto-layer and the four tail fields a sequence's -- so a rename, a removal and a move
    cost the stash nothing but their own entry. The export resolves them and renumbers the
    autolayer array, which is the one index the record itself keeps.
    """
    arm_obj = _action_armature(context, act)
    if arm_obj is None:
        lay.label(text="no imported sequence names this action", icon="INFO")
        return
    k, seq = _action_seq(arm_obj, act)
    stash = list(arm_obj.get("vtmb_sequences") or ())
    anim = str(act.get("vtmb_anim_name") or act.name)
    also = [j for j, x in enumerate(stash)
            if j != k and any(str(v) == anim for col in x.get("blends") or ()
                              for v in col)]

    row = lay.row(align=True)
    row.label(text="%s  (%d of %d)" % (str(seq.get("label") or ""), k, len(stash)),
              icon="ACTION")
    row.operator("vtmb.rename_sequence", text="", icon="GREASEPENCIL")
    row = lay.row(align=True)
    row.operator("vtmb.move_sequence", text="Up", icon="TRIA_UP").up = True
    row.operator("vtmb.move_sequence", text="Down", icon="TRIA_DOWN").up = False
    row.operator("vtmb.remove_sequence", text="", icon="X")
    lay.operator("vtmb.add_sequence", icon="ADD")
    if seq.get("new"):
        lay.label(text="added here, appended on the next export", icon="INFO")
    tomb = [str(x) for x in arm_obj.get("vtmb_seqs_removed") or ()]
    if tomb:
        lay.label(text="%d removal%s pending: %s"
                       % (len(tomb), "" if len(tomb) == 1 else "s", ", ".join(tomb[:3])),
                  icon="TRASH")
    if also:
        # `_action_seq` takes the first entry whose first blend names this action, so the
        # others are reachable from no panel.
        lay.label(text="%d other sequence%s play%s %s, and this panel does not reach %s"
                       % (len(also), "" if len(also) == 1 else "s",
                          "s" if len(also) == 1 else "", anim,
                          "it" if len(also) == 1 else "them"),
                  icon="ERROR")

    cells = [[str(v) for v in col] for col in seq.get("blends") or ()]
    gy = max((len(c) for c in cells), default=0)
    box = lay.box()
    row = box.row(align=True)
    row.label(text="blends %d x %d" % (len(cells), gy))
    row.operator("vtmb.resize_blends", text="", icon="MOD_ARRAY")
    for x, col in enumerate(cells):
        for y, name in enumerate(col):
            r = box.row(align=True)
            r.label(text="%d, %d" % (x, y))
            op = r.operator("vtmb.set_blend", text=name or "(kept)")
            op.x, op.y = x, y


class VTMB_PT_action_sequence(bpy.types.Panel):
    bl_label = "VTMB sequence"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "VTMB"

    @classmethod
    def poll(cls, context):
        return panel_action(context) is not None

    def draw(self, context):
        self.layout.use_property_split = False
        draw_action_sequence(self.layout, context, panel_action(context))


CLASSES = [VTMB_OT_add_cdtexture, VTMB_OT_add_include, VTMB_OT_check_paths,
           VTMB_OT_add_event, VTMB_OT_remove_event, VTMB_OT_set_param,
           VTMB_UL_actions,
           VTMB_OT_set_skin_family, VTMB_PT_skin_families,
           VTMB_PT_cloth,
           VTMB_OT_add_attachment, VTMB_OT_add_hitbox, VTMB_PT_accessories,
           VTMB_OT_add_spring_bone, VTMB_OT_remove_spring_bone,
           VTMB_OT_edit_flex_line, VTMB_OT_add_flex_line,
           VTMB_OT_remove_flex_line,
           VTMB_PT_face, VTMB_PT_eyeball, VTMB_PT_hitbox, VTMB_PT_flex,
           VTMB_PT_armature, VTMB_PT_poseparams, VTMB_PT_actions, VTMB_PT_action,
           VTMB_OT_rename_sequence, VTMB_OT_add_sequence, VTMB_OT_remove_sequence,
           VTMB_OT_move_sequence, VTMB_OT_set_blend, VTMB_OT_resize_blends,
           VTMB_PT_action_sequence,
           VTMB_PT_action_events, VTMB_PT_action_params, VTMB_PT_action_seqtail,
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
    bpy.types.Object.vtmb_eyeball_lids = bpy.props.EnumProperty(
        name="Lid flexes", get=_lids_get, set=_lids_set,
        items=(("none", "None", "All eight flexdesc indices zero, which is how 210 of "
                                "the 602 shipped records say the eye has no lids"),
               ("right", "Right eye", "upper_right_lowerer / _neutral / _raiser, the "
                                      "three lower_right ones, then upper_right and "
                                      "lower_right"),
               ("left", "Left eye", "The same eight names with _left, which 196 shipped "
                                    "records carry"),
               ("other", "Other", "A set matching neither shipped one. Selecting it "
                                  "writes nothing")),
        description="The eight lid flexes as one choice. They are written as a matched "
                    "set because no shipped record is partly written -- all eight are "
                    "names on 392 records and all eight are absent on 210 -- and because "
                    "an index the file has not got is appended rather than refused")
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
        search=_activity_search, search_options={"SORT", "SUGGESTION"},
        description="The activity the sequence written for this action claims, e.g. "
                    "ACT_IDLE. Emptying it writes a sequence with no activity; an action "
                    "that has never carried one instead keeps whatever the donor says, "
                    "and on the no-donor export takes the dialog's. The picker offers the "
                    "activities this model's own sequences claim and still accepts any "
                    "other name, the engine resolving the string rather than an index")


def unregister_props():
    for attr, _key, _name, _desc in _PROPS:
        if hasattr(bpy.types.Object, attr):
            delattr(bpy.types.Object, attr)
    if hasattr(bpy.types.Object, "vtmb_action_index"):
        del bpy.types.Object.vtmb_action_index
    names = ["vtmb_cloth_on", "vtmb_cloth_flip_on", "vtmb_cloth_preset_name",
             "vtmb_cloth_pin_text", "vtmb_eyeball_lids"]
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
