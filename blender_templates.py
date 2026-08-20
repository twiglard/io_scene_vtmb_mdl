"""Put a bone-set template into the scene, or merge one onto the armature already there.

`bone_templates` does the arithmetic and knows no bpy; this turns its output into edit
bones and the `vtmb_*` properties the scratch exporter reads back.
"""

import bpy

from . import bone_templates as tpl_mod

# Blender keeps no reference to the strings an EnumProperty callback returns, so a
# freshly-built list is garbage-collected out from under the UI. Cache it.
_ITEMS = []
_CACHE = {}


def templates(reload=False):
    if reload or not _CACHE:
        _CACHE.clear()
        _CACHE.update(tpl_mod.load_all())
    return _CACHE


def _icon(kind):
    return {"skeleton": "ARMATURE_DATA", "animations": "ACTION",
            "material": "MATERIAL"}.get(kind, "DOT")


def template_items(self, context):
    del _ITEMS[:]
    try:
        found = templates()
    except tpl_mod.Refused as exc:
        _ITEMS.append(("", "template error: %s" % exc, "", "ERROR", 0))
        return _ITEMS
    order = {"skeleton": 0, "animations": 1, "material": 2}
    for k, (tid, t) in enumerate(sorted(found.items(),
                                        key=lambda x: (order.get(x[1]["kind"], 9), x[0]))):
        pts = tpl_mod.attach_points(t)
        label = t.get("label") or tid
        if pts:
            label += "  (onto %s)" % ", ".join(pts)
        _ITEMS.append((tid, label, t.get("description") or "", _icon(t["kind"]), k))
    return _ITEMS


def _armature_of(context):
    obj = context.active_object
    if obj is not None and obj.type == "ARMATURE":
        return obj
    return None


def anchors_from(arm_obj, names, scale):
    """(head, tail) in FILE units for bones the scene already has."""
    out = {}
    for n in names:
        db = arm_obj.data.bones.get(n)
        if db is None:
            continue
        out[n] = (tuple(c / scale for c in db.head_local),
                  tuple(c / scale for c in db.tail_local))
    return out


def apply_skeleton(context, t, params, scale=1.0, arm_obj=None):
    """Add the template's bones, to `arm_obj` or to a new armature. Returns (object, n)."""
    pts = tpl_mod.attach_points(t)
    if pts and arm_obj is None:
        raise tpl_mod.Refused("%s attaches to %s, so select the armature carrying them "
                              "first" % (t["id"], ", ".join(pts)))
    anchors = anchors_from(arm_obj, pts, scale) if arm_obj is not None else {}
    bones = tpl_mod.resolve(t, anchors=anchors, params=params)

    created = arm_obj is None
    if created:
        data = bpy.data.armatures.new(t["id"])
        arm_obj = bpy.data.objects.new(t["id"], data)
        context.collection.objects.link(arm_obj)
    context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    added = []
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        ebs = arm_obj.data.edit_bones
        for b in bones:
            # Applying twice must not duplicate: the join is by name, so a bone already
            # present is the same bone and is left exactly as the user has it.
            if b.name in ebs:
                continue
            eb = ebs.new(b.name)
            eb.head = tuple(c * scale for c in b.head)
            eb.tail = tuple(c * scale for c in b.tail)
            eb.roll = 0.0
            if b.parent:
                eb.parent = ebs.get(b.parent)
            added.append(b)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    dbs = arm_obj.data.bones
    for b in added:
        pb = arm_obj.pose.bones.get(b.name)
        if pb is None:
            continue
        pb.rotation_mode = "QUATERNION"
        pb["vtmb_bone_flags"] = b.flags
        if b.hitgroup is not None:
            pb["vtmb_hitgroup"] = int(b.hitgroup)
        local = dbs[b.name].matrix_local
        if b.parent and b.parent in dbs:
            local = dbs[b.parent].matrix_local.inverted() @ local
        pb["vtmb_rest_local"] = [f for row in local for f in row]
    if created:
        arm_obj["vtmb_scale"] = scale
    arm_obj["vtmb_templates"] = list(arm_obj.get("vtmb_templates") or []) + [t["id"]]
    return arm_obj, len(added)


def apply_animations(arm_obj, t):
    """Merge one animations template's chain into `vtmb_includes`. Returns how many are new."""
    have = list(arm_obj.get("vtmb_includes") or [])
    new = [p for p in t["includes"] if p not in have]
    arm_obj["vtmb_includes"] = have + new
    arm_obj["vtmb_templates"] = list(arm_obj.get("vtmb_templates") or []) + [t["id"]]
    return len(new)


class OBJECT_OT_vtmb_add_skeleton(bpy.types.Operator):
    bl_idname = "object.vtmb_add_skeleton"
    bl_label = "Add VTMB bone set"
    bl_options = {"REGISTER", "UNDO"}

    template: bpy.props.EnumProperty(name="Template", items=template_items)
    scale: bpy.props.FloatProperty(
        name="Scale", default=1.0, min=1e-4, soft_max=100.0,
        description="Blender units per file unit, matching the export dialog's own")
    height: bpy.props.FloatProperty(
        name="Height", default=72.0, min=1.0, soft_max=400.0,
        description="Every offset is a fraction of this, so the whole rig scales with it. "
                    "72 is the humanoid hull the format defaults to; for a quadruped it "
                    "is withers height, not nose to tail")
    width: bpy.props.FloatProperty(
        name="Width", default=1.0, min=0.05, soft_max=3.0,
        description="Side-to-side multiplier: shoulders, hips and stance")
    depth: bpy.props.FloatProperty(
        name="Depth", default=1.0, min=0.05, soft_max=3.0,
        description="Front-to-back multiplier")
    reload: bpy.props.BoolProperty(
        name="Re-read the template files", default=False,
        description="Pick up a .json edited since Blender started")

    def invoke(self, context, event):
        arm = _armature_of(context)
        if arm is not None:
            self.scale = float(arm.get("vtmb_scale", 1.0) or 1.0)
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        lay = self.layout
        lay.prop(self, "template")
        try:
            t = templates()[self.template]
        except Exception:
            lay.label(text="no template selected", icon="ERROR")
            return
        for line in (t.get("description") or "").split(". "):
            if line.strip():
                lay.label(text=line.strip().rstrip(".") + ".")

        arm = _armature_of(context)
        pts = tpl_mod.attach_points(t)
        box = lay.box()
        if t["kind"] == "animations":
            if arm is None:
                box.label(text="select an armature: this adds no bones, only the chain",
                          icon="ERROR")
            else:
                box.label(text="adds %d chained model(s) to %s"
                               % (len(t["includes"]), arm.name), icon="ACTION")
                for p in t["includes"]:
                    box.label(text="    " + p)
            return
        if pts:
            missing = [] if arm is None else [
                p for p in pts if arm.data.bones.get(p) is None]
            if arm is None:
                box.label(text="select the armature to merge onto first", icon="ERROR")
            elif missing:
                box.label(text="%s has no %s" % (arm.name, ", ".join(missing)),
                          icon="ERROR")
            else:
                box.label(text="merges %d bones onto %s" % (len(t["bones"]), arm.name),
                          icon="ARMATURE_DATA")
        elif arm is not None:
            box.label(text="adds %d bones to %s" % (len(t["bones"]), arm.name),
                      icon="ARMATURE_DATA")
        else:
            box.label(text="creates a new armature of %d bones" % len(t["bones"]),
                      icon="ARMATURE_DATA")

        col = lay.column()
        col.prop(self, "scale")
        gen = (t.get("proportions") or {}).get("generator")
        if gen in ("biped", "quadruped", "single"):
            col.prop(self, "height")
        if gen in ("biped", "quadruped"):
            col.prop(self, "width")
            col.prop(self, "depth")
        lay.prop(self, "reload")

    def execute(self, context):
        try:
            found = templates(reload=self.reload)
            t = found[self.template]
        except tpl_mod.Refused as exc:
            self.report({"ERROR"}, "refused: %s" % exc)
            return {"CANCELLED"}
        except KeyError:
            self.report({"ERROR"}, "no template %r on disk" % self.template)
            return {"CANCELLED"}

        arm = _armature_of(context)
        try:
            if t["kind"] == "animations":
                if arm is None:
                    raise tpl_mod.Refused("an animations template carries no bones, so "
                                          "select the armature to put the chain on")
                n = apply_animations(arm, t)
                self.report({"INFO"}, "%s: %d chained model(s) added, %d already there"
                            % (t["id"], n, len(t["includes"]) - n))
                return {"FINISHED"}
            if t["kind"] != "skeleton":
                raise tpl_mod.Refused("%s is a %s template, which nothing applies yet"
                                      % (t["id"], t["kind"]))
            params = {"height": self.height, "width": self.width, "depth": self.depth}
            arm, n = apply_skeleton(context, t, params, self.scale, arm)
        except tpl_mod.Refused as exc:
            self.report({"ERROR"}, "refused: %s" % exc)
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, "%s: %s" % (type(exc).__name__, exc))
            return {"CANCELLED"}
        skipped = len(t["bones"]) - n
        self.report({"INFO"}, "%s: %d bones added to %s%s"
                    % (t["id"], n, arm.name,
                       ", %d already present" % skipped if skipped else ""))
        return {"FINISHED"}


class VIEW3D_MT_vtmb_add(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_vtmb_add"
    bl_label = "VTMB"

    def draw(self, context):
        self.layout.operator(OBJECT_OT_vtmb_add_skeleton.bl_idname,
                             text="Bone set...", icon="ARMATURE_DATA")


CLASSES = (OBJECT_OT_vtmb_add_skeleton, VIEW3D_MT_vtmb_add)


def menu(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_vtmb_add.bl_idname, icon="ARMATURE_DATA")
