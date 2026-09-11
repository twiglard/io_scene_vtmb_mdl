#!/usr/bin/env python3
"""Blender side of the VTMB MDL importer: armature, meshes, materials, actions."""

import array
import math
import os
import shutil
import struct
import sys
import tempfile
import zlib

import bpy
import mathutils

from . import cloth as cloth_mod
from . import mdl as mdl_mod
from . import paths as paths_mod
from . import tth as tth_mod
from . import vpk as vpk_mod
from . import vtx as vtx_mod

_PACK_CACHE = {}


def _packs(root):
    if not root:
        return None
    try:
        if root not in _PACK_CACHE:
            _PACK_CACHE[root] = vpk_mod.Vpks(root) if vpk_mod.mount_order(root) else None
    except (OSError, ValueError):
        _PACK_CACHE[root] = None
    return _PACK_CACHE[root]


class Content:
    """Resolution in the engine's own order: each root's loose files, then that root's
    mounted packs. FileSystem_Stdio's AddSearchPath appends the directory before its
    packs and lookup takes the first hit -- recon 26.6."""

    def __init__(self, roots, use_packs=True):
        self.roots = list(roots)
        self.packs = [_packs(r) if use_packs else None for r in self.roots]

    def find(self, rel):
        """(path, bytes or None). The bytes are set only for a pack hit, where path is
        'pack00N.vpk!models/...' and names nothing on disk."""
        rel = rel.replace("\\", "/").lstrip("/")
        key = rel.lower()
        for r, v in zip(self.roots, self.packs):
            p = os.path.join(r, rel.replace("/", os.sep))
            if os.path.isfile(p):
                return p, None
            hit = v.index.get(key) if v is not None else None
            if hit is not None:
                return "%s!%s" % (hit[0], key), v.read(key)
        return None, None

    def read(self, rel):
        path, blob = self.find(rel)
        if path is None:
            return None
        if blob is not None:
            return blob
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def mdl(self, rel):
        path, blob = self.find(rel)
        if path is None:
            return None
        try:
            m = mdl_mod.Mdl(path, blob)
        except (OSError, ValueError):
            return None
        m.rel = rel
        return m

    def companion(self, m, suffixes):
        """A .vtx for a .mdl: beside the file, then the same relative path on every root
        and in its packs -- the Unofficial Patch ships .mdl without .vtx."""
        for p in paths_mod.companions(m.path, self.roots, suffixes):
            if os.path.isfile(p):
                return p, None
        rel = getattr(m, "rel", None) or paths_mod.relative(m.path, self.roots)
        if rel:
            stem = rel[:-4] if rel.lower().endswith(".mdl") else rel
            for s in suffixes:
                path, blob = self.find(stem + s)
                if path is not None:
                    return path, blob
        return None, None


def _addon_version():
    mod = sys.modules.get(__package__)
    return list(getattr(mod, "bl_info", {}).get("version", ()))


def _scaled(m, s):
    # Scaling before the Matrix rounds the translation to float32 once instead of twice.
    # Bit-identical at scale 1.0; one ulp apart from the old order at any other scale.
    a, b, c = m
    return mathutils.Matrix(((a[0], a[1], a[2], a[3] * s),
                             (b[0], b[1], b[2], b[3] * s),
                             (c[0], c[1], c[2], c[3] * s),
                             (0.0, 0.0, 0.0, 1.0)))


def _display_lengths(m, rest, scale):
    """Draw each bone out to its first child, and a childless one to half its parent.

    Nothing here may be a constant: every length has to carry `scale`, or a non-1.0 import
    leaves the leaf bones dwarfing the real ones.  Parents precede children, so one pass
    suffices.

    The `1e-4` floor costs accuracy rather than only looks: matrix_local is derived from
    `tail - head` in float32, so a bone at the floor with its head 27 units out comes back
    1.4e-3 per quaternion entry away from what was assigned, against 1.3e-5 at length 1.
    """
    first = {}
    for b in m.bones:
        if b.parent >= 0:
            first.setdefault(b.parent, b.index)
    out = []
    for b, w in zip(m.bones, rest):
        kid = first.get(b.index)
        if kid is not None:
            d = math.dist([w[i][3] for i in range(3)],
                          [rest[kid][i][3] for i in range(3)]) * scale
        elif b.parent >= 0:
            d = out[b.parent] * 0.5
        else:
            d = scale
        out.append(max(d, 1e-4))
    return out


def build_armature(context, m, name, scale, root_motion=True):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    context.collection.objects.link(arm_obj)
    context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    rest = m.rest_world_matrices()
    lengths = _display_lengths(m, rest, scale)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = []
    for b, w, ln in zip(m.bones, rest, lengths):
        eb = arm_data.edit_bones.new(b.name)
        # A zero-length bone is silently dropped, so give it a tail before the matrix
        # assignment, which then overwrites head, tail direction and roll together.
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, ln, 0.0)
        eb.matrix = _scaled(w, scale)
        eb.length = ln
        edit_bones.append(eb)
    for b, eb in zip(m.bones, edit_bones):
        if b.parent >= 0:
            eb.parent = edit_bones[b.parent]
    bpy.ops.object.mode_set(mode="OBJECT")

    dbs = arm_obj.data.bones
    # Keyed on the record ordinal, not on the bone: two records may name one start bone,
    # and the disable mask is `1 << recordIndex`, so the ordinal is what identifies a chain.
    springs = {}
    for s in m.springbones:
        springs.setdefault(s.bone, s)
    for b in m.bones:
        pb = arm_obj.pose.bones[b.name]
        pb.rotation_mode = "QUATERNION"
        # rotscale bounds what can ever be written back to this bone.
        # Bit 1 is BONE_ROTATION_FROM_ROOT, which decides what a pose's local rotation even
        # means, so a scene-authored rewrite has to know it before it samples a frame.
        # Blender keeps no record of a rename, and the export resolves every file bone by
        # name, so without this a renamed bone reads as one deleted and one added.
        pb["vtmb_bone_name"] = b.name
        pb["vtmb_bone_flags"] = b.flags
        pb["vtmb_rotscale"] = list(b.rotscale)
        pb["vtmb_posscale"] = list(b.posscale)
        pb["vtmb_rot_limit"] = [32767.0 * s for s in b.rotscale]
        # The export path's did-this-move baseline. The file's own record cannot serve:
        # head/tail/roll storage loses up to 1.4e-3 per quaternion entry on a short bone.
        local = dbs[b.name].matrix_local
        if b.parent >= 0:
            local = dbs[m.bones[b.parent].name].matrix_local.inverted() @ local
        pb["vtmb_rest_local"] = [f for row in local for f in row]
        sb = springs.get(b.index)
        if sb is not None:
            # The five floats are what the eight bc_* ConVars override at runtime; writing
            # them here is the same retune made permanent. unk08 is read by nothing in the
            # whole install and is carried so an untouched export stays byte-identical.
            pb["vtmb_spring_index"] = sb.index
            pb["vtmb_spring_end"] = (m.bones[sb.endbone].name
                                     if 0 <= sb.endbone < len(m.bones) else "")
            pb["vtmb_spring_disabled"] = bool(sb.disabled)
            pb["vtmb_spring_unk08"] = sb.unk08
            pb["vtmb_spring_gravity"] = sb.gravity
            pb["vtmb_spring_damping"] = sb.damping
            pb["vtmb_spring_exp"] = sb.springexp
            pb["vtmb_spring_maxangle"] = sb.maxangledeg

    arm_obj["vtmb_scale"] = scale
    arm_obj["vtmb_checksum"] = m.checksum
    arm_obj["vtmb_source"] = m.path
    arm_obj["vtmb_includes"] = list(m.includes)
    # Per-model and not per-material, so nothing else in the scene can carry it back out.
    arm_obj["vtmb_cdtexture"] = list(m.material_paths)
    arm_obj["vtmb_poseparams"] = [
        {"name": pp.name, "flags": pp.flags, "start": pp.start, "end": pp.end,
         "loop": pp.loop} for pp in m.poseparams]
    arm_obj["vtmb_sequences"] = sequence_stash(m)
    # The user's answer to "was the travel applied to the keys". The count that goes with
    # it is stamped in import_mdl, once the imported actions are known.
    arm_obj["vtmb_root_motion"] = bool(root_motion)
    # nummouths is 0 or 1 over the whole corpus.  The bone goes in by NAME and the flex by
    # name too: the index is 16 on 196 of the 200 shipped records and 0 on 4 that simply
    # list `mouth` first, so an index would say the wrong thing about either.
    if m.mouth is not None and 0 <= m.mouth.bone < len(m.bones):
        arm_obj["vtmb_mouth"] = {
            "bone": m.bones[m.mouth.bone].name,
            "forward": list(m.mouth.forward),
            "flex": m.mouth.name or "",
        }
    # Both lists are studiomdl's own QC lines, which is the only syntax there is for
    # either -- Option_Flexcontroller and Option_Flexrule, studiomdl.cpp:2875 and :2914.
    # A controller is identified across models by its NAME alone: client.dll 100c4351
    # looks each one up in a process-global name table and writes the row it lands in
    # back into the record's `link`, which every shipped file leaves at -1.
    arm_obj["vtmb_flexcontrollers"] = [_flexctrl_line(c) for c in m.flexcontrollers]
    # A rule drives a FLEXDESC, not a shape key: one flexdesc is a flex record on every
    # mesh that morphs, so the rule belongs to the model and is keyed by name here.
    arm_obj["vtmb_flexrules"] = ["%s = %s" % (r.name, r.expr)
                                 for r in m.flexrules if r.name and r.expr]
    return arm_obj


def _flexctrl_line(c):
    """`$flexcontroller` without the command word: `<type> [range <min> <max>] <name>`."""
    rng = "" if (c.min, c.max) == (0.0, 1.0) else "range %g %g " % (c.min, c.max)
    return "%s %s%s" % (c.type, rng, c.name)


def sequence_stash(m):
    """`arm["vtmb_sequences"]` for a file: label, activity, group size, the blend grid,
    the events, which pose parameter drives each blend axis, and the record's tail.

    Blends, pose parameters, autolayers and a knockback's bone are stored as names because
    an index means nothing once the file is re-emitted. Matched back by position, so
    anything that changes how many sequences the file has has to write this again --
    `apply_sequences` refuses a stash claiming more sequences than are there.
    """
    labels = [x.label for x in m.seqs]
    return [{"label": s.label, "activity": s.activity, "flags": s.flags,
             "groupsize": list(s.groupsize),
             "blends": [[m.anims[i].name if 0 <= i < len(m.anims) else ""
                         for i in col] for col in s.blends],
             "events": [{"cycle": e.cycle, "event": e.event, "type": e.type,
                         "options": e.options} for e in s.events],
             "params": [{"name": (m.poseparams[i].name
                                  if 0 <= i < len(m.poseparams) else ""),
                         "start": st, "end": en}
                        for i, st, en in zip(s.paramindex, s.paramstart,
                                             s.paramend)],
             "autolayers": [labels[i] if 0 <= i < len(labels) else "" for i in s.autolayers],
             "hitvolumes": [{"bbmin": list(h.bbmin), "bbmax": list(h.bbmax)}
                            for h in s.hitvolumes],
             "knockbacks": [{"bone": (m.bones[k.bone].name
                                      if 0 <= k.bone < len(m.bones) else ""),
                             "cycleend": k.cycleend,
                             "activities": [[x or "" for x in row] for row in k.activities]}
                            for k in s.knockbacks],
             "node": [s.entrynode, s.exitnode, s.nodeflags],
             "phase": [s.entryphase, s.exitphase],
             "dodge": s.dodge or "", "block": s.block or "",
             "name2e8": s.name2e8 or "", "name2ec": s.name2ec or "",
             "statrequired": s.statrequired,
             "seqselectmask": s.seqselectmask,
             "meleerange": list(s.meleerange),
             "cyclewindow": list(s.cyclewindow)}
            for s in m.seqs]


def _event_markers(m, made):
    """One pose marker per mstudioevent_t, on the action of its sequence's first blend.

    A marker is the only per-frame datum an Action already carries and it survives the
    NLA, so it is what makes an event visible on the timeline. It is not the record --
    a marker holds a name and a frame and nothing else, and Blender's own retiming
    renumbers it -- so `vtmb_sequences` stays the truth and the export reads that.
    """
    by_index = {}
    for act in made:
        if act.get("vtmb_source") == m.path:
            by_index.setdefault(act.get("vtmb_anim_index"), act)
    stamped = 0
    for s in m.seqs:
        if not s.events or not s.blends or not s.blends[0]:
            continue
        act = by_index.get(s.blends[0][0])
        if act is None:
            continue
        span = max(1, int(act.get("vtmb_numframes") or 1) - 1)
        for e in s.events:
            mk = act.pose_markers.new("event %d" % e.event)
            mk.frame = int(round(e.cycle * span))
            stamped += 1
    return stamped


def missing_paths(content, includes=(), cdtextures=(), material_names=()):
    """{"includes": [...], "materials": [...]} -- what a write would point at nothing.

    A material is missing only when no cdtexture directory has it, which is the same
    cross product the engine walks, so one prefix serving nothing is not itself an error:
    `toreador_female_armor_0` carries 11 for 10 materials.
    """
    gone = [p for p in includes if content.find(p)[0] is None]
    bad = []
    for name in material_names:
        stems = paths_mod.material_stems(name, cdtextures)
        if all(content.find(s + ".vmt")[0] is None for s in stems):
            bad.append(name)
    return {"includes": gone, "materials": bad}


def _vmt(path, blob, *keys):
    """`paths.vmt_value` over a .vmt that may still be on disk rather than in a pack."""
    if blob is None:
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except OSError:
            return None
    return paths_mod.vmt_value(blob, *keys)


def _load_file(path):
    """None rather than an exception: a texture this build of Blender cannot read must
    cost one white material, not the whole import."""
    try:
        img = bpy.data.images.load(path, check_existing=True)
    except (RuntimeError, OSError):
        return None
    if img.size[0] and img.size[1]:
        return img
    bpy.data.images.remove(img)
    return None


def _packed_image(name, blob, ext, source):
    """bpy.data.images.load takes a path and nothing else, so a texture that lives in a
    pack goes through a temp file and is packed into the .blend before that file dies."""
    source = source.replace("\\", "/").lower()
    for img in bpy.data.images:
        if img.get("vtmb_source") == source:
            return img
    d = tempfile.mkdtemp(prefix="vtmb_tex_")
    try:
        p = os.path.join(d, name + ext)
        with open(p, "wb") as f:
            f.write(blob)
        img = _load_file(p)
        if img is None:
            return None
        img.name = name
        try:
            img.pack()
        except RuntimeError:
            bpy.data.images.remove(img)
            return None
    except OSError:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)
    img["vtmb_source"] = source
    return img


def _tth_image(content, rel, name):
    head = content.read(rel + ".tth")
    if head is None:
        return None
    try:
        t = tth_mod.Tth(head)
        _level, w, h, mip = t.top_mip(content.read(rel + ".ttz"))
    except (ValueError, zlib.error, struct.error):
        return None
    source = content.find(rel + ".tth")[0]
    try:
        rgba = tth_mod.decode(w, h, t.format, mip)
    except ValueError:
        return None
    return _packed_image(name, tth_mod.to_tga(w, h, rgba), ".tga", source)


def _image(content, rel):
    """An image datablock for an extensionless materials/ path, or None."""
    name = rel.rsplit("/", 1)[-1]
    path, blob = content.find(rel + ".tga")
    if path is not None:
        img = _load_file(path) if blob is None else _packed_image(name, blob, ".tga", path)
        if img is not None:
            return img
    return _tth_image(content, rel, name)


# A VertexLitGeneric base texture keeps its specular/envmap mask in the alpha channel, so
# the channel is opacity only where one of these says it is. Jeanette's body averages 0.031
# there, which Workbench draws as 97% see-through in Solid + Texture.
ALPHA_KEYS = ("$translucent", "$alphatest", "$additive")


def _alpha_is_opacity(path, blob):
    for key in ALPHA_KEYS:
        v = _vmt(path, blob, key)
        if v is not None and v.strip().strip('"') not in ("", "0"):
            return True
    return False


def _find_image(content, search_paths, name):
    """(image, whether its alpha channel is opacity), or (None, False).

    VMT first, since $basetexture may point outside the model's own directory.
    """
    for stem in paths_mod.material_stems(name, search_paths):
        vmt, blob = content.find(stem + ".vmt")
        if vmt:
            # $dudvmap is what a Refract names instead: five shipped models carry a
            # material whose .vmt has no $basetexture and no texture of its own name.
            base = _vmt(vmt, blob, "$basetexture", "$dudvmap")
            if base:
                img = _image(content, paths_mod.MATERIALS_DIR + "/"
                             + base.replace("\\", "/").strip("/"))
                if img is not None:
                    return img, _alpha_is_opacity(vmt, blob)
        img = _image(content, stem)
        if img is not None:
            return img, bool(vmt) and _alpha_is_opacity(vmt, blob)
    return None, False


def _set_alpha_mode(img, opacity):
    """One image datablock serves several materials -- Jeanette's hair map serves six --
    so a material that needs the channel claims it and an opaque one never takes it back."""
    if opacity:
        img["vtmb_alpha_opacity"] = True
        img.alpha_mode = "STRAIGHT"
    elif not img.get("vtmb_alpha_opacity"):
        img.alpha_mode = "NONE"


def _average_color(img, stride=16):
    """Solid viewport shading paints mat.diffuse_color and ignores the node tree, so a
    correctly textured import still reads as flat white there unless this is set."""
    cached = img.get("vtmb_average")
    if cached is not None:
        return tuple(cached)
    out = (0.8, 0.8, 0.8, 1.0)
    try:
        import numpy as np

        a = np.empty(img.size[0] * img.size[1] * img.channels, dtype=np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(-1, img.channels)[::stride]
        rgb = a[:, :3].mean(axis=0)
        out = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    except (ImportError, RuntimeError, ValueError):
        pass
    img["vtmb_average"] = out
    return out


def _has_image(mat):
    tree = mat.node_tree
    return bool(tree and any(n.type == "TEX_IMAGE" and n.image for n in tree.nodes))


def _material(m, index, content, family=0):
    ref = index
    row = m.skins[family] if 0 <= family < len(m.skins) else None
    if row and 0 <= index < len(row):
        ref = row[index]
    name = m.materials[ref] if 0 <= ref < len(m.materials) else "material_%d" % index
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    elif _has_image(mat):
        return mat
    # Reuse by name, but top up a material left over from an import that found no
    # texture -- returning it as-is kept it white however good the new lookup got.
    mat.use_nodes = True
    if m.material_paths:
        mat["vtmb_search_paths"] = list(m.material_paths)

    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        # Principled's 0.5 specular over a flat diffuse map reads as wet chrome; 1842 of
        # the 1921 character VMTs carry $basetexture and nothing else.
        bsdf.inputs["Roughness"].default_value = 1.0
        for sock in ("Specular IOR Level", "Specular"):
            if sock in bsdf.inputs:
                bsdf.inputs[sock].default_value = 0.0
                break

    img, opacity = (_find_image(content, m.material_paths, name)
                    if bsdf is not None else (None, False))
    if img is None:
        return mat
    _set_alpha_mode(img, opacity)
    mat["vtmb_texture"] = img.get("vtmb_source") or img.filepath or img.name
    mat.diffuse_color = _average_color(img)
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (bsdf.location.x - 400, bsdf.location.y)
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex.outputs["Color"])
    return mat


def build_shape_keys(obj, model, scale):
    """One shape key per flexdesc the model's meshes name. (keys, records, dropped).

    `mstudioflex_t` is per mesh and the vertanim key is mesh-local, so a morph that
    crosses two meshes of one model is two flex records naming one flexdesc, and both
    land in the same key here -- the Blender vertex is `mesh.vertexoffset + index`,
    which is exactly what `R_StudioFlexVerts` indexes.

    Position deltas only. A shape key holds coordinates and nothing else, so the normal
    delta each record also carries has nowhere to go; the count is stashed on the object
    and the caller reports it rather than dropping it in silence.
    """
    flexes = [(mesh, f) for mesh in model.meshes for f in mesh.flexes]
    if not flexes:
        return 0, 0, 0
    order, targets = [], {}
    for _mesh, f in flexes:
        if f.flexdesc not in targets:
            order.append(f.flexdesc)
            targets[f.flexdesc] = f.target
    nverts = len(obj.data.vertices)
    # Blender 5.2's shape_key_add hands back a key at value 1.0, so without this every
    # flex arrives fully applied and the face is the sum of all of them at once.
    obj.shape_key_add(name="Basis", from_mix=False).value = 0.0
    nrec = dropped = 0
    for fd in order:
        nm = None
        for _mesh, f in flexes:
            if f.flexdesc == fd:
                nm = f.name
                break
        kb = obj.shape_key_add(name=nm or "flex_%d" % fd, from_mix=False)
        kb.value = 0.0
        kb.slider_min, kb.slider_max = 0.0, 1.0
        data = kb.data
        for mesh, f in flexes:
            if f.flexdesc != fd:
                continue
            for v in f.verts:
                nrec += 1
                d = mdl_mod.flex_delta(v, f.vertanimtype)
                vi = mesh.vertexoffset + v.index
                if d is None or not 0 <= vi < nverts:
                    dropped += 1
                    continue
                co = data[vi].co
                co[0] += d[0][0] * scale
                co[1] += d[0][1] * scale
                co[2] += d[0][2] * scale
    # -11.0 .. 11.0 over the corpus, read by nothing this addon writes, and there is no
    # Blender field for it -- kept so a rebuild can put back what the file said.
    obj["vtmb_flex_targets"] = [[float(c) for c in targets[fd]] for fd in order]
    obj["vtmb_flex_names"] = [obj.data.shape_keys.key_blocks[i + 1].name
                              for i in range(len(order))]
    return len(order), nrec, dropped


def _accessory_collection(context, arm_obj, name):
    """A sub-collection so the export's mesh walk and the user's own selections miss these.

    Linked under whatever collection the armature landed in, not under the scene root.
    """
    coll = bpy.data.collections.new("%s accessories" % name)
    for c in bpy.data.collections:
        if arm_obj.name in c.objects:
            c.children.link(coll)
            break
    else:
        context.scene.collection.children.link(coll)
    return coll


def _bone_child(obj, arm_obj, bone):
    """Parent to the bone, with the tail offset taken back out.

    Measured on Blender 5.2: with both matrices identity a bone child lands at
    `bone.matrix_local @ Translation((0, bone.length, 0))`, so this inverse is what makes
    `matrix_basis` mean bone space and nothing else.
    """
    obj.parent = arm_obj
    obj.parent_type = "BONE"
    obj.parent_bone = bone.name
    obj.matrix_parent_inverse = mathutils.Matrix.Translation((0.0, -bone.length, 0.0))


def build_accessories(context, m, arm_obj, name, scale):
    """Attachments as ARROWS empties and hitboxes as CUBE empties, both in bone space.

    A CUBE empty draws -size to +size in its local space, so an arbitrary AABB is the
    scale and the centre and comes back exactly.  `matrix_basis` holds the file's own
    bone-space matrix and no bone matrix at all -- Blender's bone parenting supplies that
    -- so the export reads back what it will write with nothing to invert.
    """
    dbs = arm_obj.data.bones
    made = []
    eyes = any(mo.eyeballs for bp in m.bodyparts for mo in bp.models)
    if not m.attachments and not m.hitboxsets and not eyes:
        return made
    coll = _accessory_collection(context, arm_obj, name)
    for a in m.attachments:
        if not 0 <= a.bone < len(m.bones):
            continue
        bone = dbs.get(m.bones[a.bone].name)
        if bone is None:
            continue
        obj = bpy.data.objects.new("%s.%s" % (name, a.name or str(a.index)), None)
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = max(scale, 1e-4)
        coll.objects.link(obj)
        _bone_child(obj, arm_obj, bone)
        obj.matrix_basis = _scaled(a.local, scale)
        obj["vtmb_attachment"] = a.name
        obj["vtmb_attachment_type"] = a.type
        obj["vtmb_attachment_index"] = a.index
        made.append(obj)
    for bp in m.bodyparts:
        for model in bp.models:
            for e in model.eyeballs:
                obj = _eyeball_empty(m, e, model, arm_obj, name, scale, dbs)
                if obj is not None:
                    coll.objects.link(obj)
                    _bone_child(obj, arm_obj, dbs[m.bones[e.bone].name])
                    made.append(obj)
    for hs in m.hitboxsets:
        root = bpy.data.objects.new("%s.%s" % (name, hs.name or "default"), None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = max(scale, 1e-4)
        coll.objects.link(root)
        root.parent = arm_obj
        root["vtmb_hitboxset"] = hs.name
        root["vtmb_hitboxset_index"] = hs.index
        made.append(root)
        for x in hs.boxes:
            if not 0 <= x.bone < len(m.bones):
                continue
            bone = dbs.get(m.bones[x.bone].name)
            if bone is None:
                continue
            obj = bpy.data.objects.new("%s.box%d" % (root.name, x.index), None)
            obj.empty_display_type = "CUBE"
            obj.empty_display_size = 1.0
            coll.objects.link(obj)
            _bone_child(obj, arm_obj, bone)
            half = [(hi - lo) * 0.5 * scale for lo, hi in zip(x.bbmin, x.bbmax)]
            mid = [(hi + lo) * 0.5 * scale for lo, hi in zip(x.bbmin, x.bbmax)]
            obj.matrix_basis = (mathutils.Matrix.Translation(mid)
                                @ mathutils.Matrix.Diagonal(half + [1.0]))
            obj["vtmb_hitbox_group"] = x.group
            obj["vtmb_hitboxset_index"] = hs.index
            obj["vtmb_hitbox_index"] = x.index
            made.append(obj)
    return made


def _eyeball_empty(m, e, model, arm_obj, name, scale, dbs):
    """One mstudioeyeball_t as a SPHERE empty in its bone's space.

    `up` and `forward` are orthonormal on 602 of 602 shipped records -- worst
    |up . forward| 5.1e-08, worst ||v|| - 1 1.2e-07 -- so a basis (up x forward, up,
    forward) is a rotation and the export reads the two vectors straight back out of the
    matrix.  Everything else the record holds is a custom property: the two material names,
    because a writer resolves them by name and an index would go stale the moment a slot
    moves, and the eight lid flexes by name for the same reason.

    zoffset is 0 on 602 of 602 and pitch/yaw on 602 of 602, so neither is drawn; the
    export writes both as zero.
    """
    if not 0 <= e.bone < len(m.bones):
        return None
    bone = dbs.get(m.bones[e.bone].name)
    if bone is None:
        return None
    obj = bpy.data.objects.new("%s.eye%d" % (name, e.index), None)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = max(e.radius * scale, 1e-4)
    # NOT normalised: the export reads these two columns straight back, so normalising
    # here would rewrite every record with values a hair off the donor's. They are
    # orthonormal on 602 of 602 shipped records anyway, worst ||v|| - 1 = 1.16e-07.
    up = mathutils.Vector(e.up)
    fw = mathutils.Vector(e.forward)
    right = up.cross(fw)
    rot = mathutils.Matrix((right, up, fw)).transposed().to_4x4()
    obj.matrix_basis = (mathutils.Matrix.Translation(
        [c * scale for c in e.org]) @ rot)
    obj["vtmb_eyeball"] = e.index
    obj["vtmb_eyeball_model"] = model.name
    obj["vtmb_eyeball_radius"] = e.radius
    obj["vtmb_eyeball_iris_scale"] = e.iris_scale
    obj["vtmb_eyeball_iris"] = e.iris_name or ""
    obj["vtmb_eyeball_glint"] = e.glint_name or ""
    obj["vtmb_eyeball_uppertarget"] = list(e.uppertarget)
    obj["vtmb_eyeball_lowertarget"] = list(e.lowertarget)
    # All eight zero is the only shipped way of saying "no lid flexes"; the right eye's
    # own upperlidflexdesc is a genuine 0, so a lone 0 is not absence.
    obj["vtmb_eyeball_lidflexes"] = ["" if x is None else x
                                     for x in (e.lidflexes or ())]
    return obj


def build_meshes(context, m, arm_obj, name, scale, content, with_flexes=True):
    path, blob = content.companion(m, vtx_mod.SUFFIXES)
    if path is None:
        return [], "no .vtx for this .mdl on any content root, so no faces to import"
    try:
        v = vtx_mod.Vtx(path, blob)
    except Exception as exc:
        # The bones and the animations are in the .mdl and are readable, so a broken
        # companion costs the geometry and not the import.
        return [], "%s does not parse: %s: %s" % (path, type(exc).__name__, exc)
    if v.checksum != m.checksum:
        return [], ("checksum mismatch: .vtx %#x vs .mdl %#x -- they are not a pair"
                    % (v.checksum & 0xffffffff, m.checksum & 0xffffffff))

    models = [(bp, mo) for bp in m.bodyparts for mo in bp.models]
    faces_by_model = {}
    for g in v.groups:
        if g.lod != 0 or g.model >= len(models):
            continue
        _bp, model = models[g.model]
        if g.mesh >= len(model.meshes):
            continue
        mesh = model.meshes[g.mesh]
        ids = v.orig_vert_ids(g)
        acc = faces_by_model.setdefault(g.model, [])
        # Reversed, because the format winds a triangle against its own outward normal and
        # Blender winds it with: carried through unchanged, every imported mesh faces inward
        # and backface culling shows the far wall through the near one. The exporter reverses
        # back, so the file's own order is what a round trip writes.
        for tri in v.triangles(g):
            acc.append((tuple(mesh.vertexoffset + ids[i] for i in reversed(tri)),
                        mesh.material))

    objs = []
    skin_mats = {}
    cloth_notes = []
    nkeys = nflexrec = nflexdrop = 0
    for gi, (bp, model) in enumerate(models):
        faces = faces_by_model.get(gi)
        if not faces:
            continue
        verts = m.vertices(model)
        # The datablock takes `mstudiomodel_t.name` verbatim and the object keeps the
        # disambiguating form, so the outliner stops showing one string twice and the
        # inner row is the file's own value. Blender caps a datablock name at 63 bytes
        # and suffixes a duplicate, and the corpus has 3 names over 63 (longest 69) and
        # 14 records repeating a name inside one file -- so what it actually assigned is
        # stashed beside what the file said, and only a difference between the two counts
        # as a rename.
        me = bpy.data.meshes.new(model.name or "%s_%d" % (name, gi))
        me["vtmb_model_name"] = model.name
        me["vtmb_model_label"] = me.name
        me.from_pydata([[c * scale for c in v_.pos] for v_ in verts], [],
                       [list(f[0]) for f in faces])
        me.update()

        # Seeded from the meshes rather than the faces so a mesh whose triangles all sit
        # in a lower LOD still gets a slot, which is what lets the exporter place it.
        slots = {}
        for mesh in model.meshes:
            if mesh.material not in slots:
                slots[mesh.material] = len(slots)
                me.materials.append(_material(m, mesh.material, content))
        for _tri, matidx in faces:
            if matidx not in slots:
                slots[matidx] = len(slots)
                me.materials.append(_material(m, matidx, content))
        for poly, (_tri, matidx) in zip(me.polygons, faces):
            poly.material_index = slots[matidx]

        uv = me.uv_layers.new(name="UVMap")
        for loop in me.loops:
            u, w = verts[loop.vertex_index].uv
            uv.data[loop.index].uv = (u, 1.0 - w)

        # A vertex added later copies this rather than getting a sentinel, so a tag stops
        # being unique; obj["vtmb_orig_co"] below is what breaks the tie.
        att = me.attributes.new("vtmb_orig", "INT", "POINT")
        att.data.foreach_set("value", list(range(len(verts))))
        # A loose vertex has no loop, so the UV layer cannot give it one back.
        att = me.attributes.new("vtmb_uv", "FLOAT2", "POINT")
        att.data.foreach_set("vector",
                             [c for v_ in verts for c in (v_.uv[0], 1.0 - v_.uv[1])])

        # Every filetype carries a normal: 0 stores the vector, 1 and 2 an index into one
        # of StudioRender's two tables, which `normal_table` resolves.
        me.normals_split_custom_set_from_vertices([v_.normal for v_ in verts])
        # Custom split normals are stored as two 16-bit angles and come back off by
        # up to 9.2e-03, so the exporter needs the exact ones to write back.
        att = me.attributes.new("vtmb_normal", "FLOAT_VECTOR", "POINT")
        att.data.foreach_set("vector", [c for v_ in verts for c in v_.normal])

        if model.filetype == 0:
            # Slot order is neither by weight nor by bone -- it is whatever studiomdl read
            # out of the SMD -- so vertex groups alone cannot reproduce it.
            att = me.attributes.new("vtmb_skin", "FLOAT_COLOR", "POINT")
            att.data.foreach_set("color", [float(b) for v_ in verts
                                           for b in v_.bones])
            att = me.attributes.new("vtmb_weight", "FLOAT_COLOR", "POINT")
            att.data.foreach_set("color", [w for v_ in verts for w in v_.weights])
            att = me.attributes.new("vtmb_numbones", "INT", "POINT")
            att.data.foreach_set("value", [v_.numbones for v_ in verts])

        obj = bpy.data.objects.new(
            "%s_%s" % (name, (model.name or str(gi)).rsplit(".", 1)[0]), me)
        obj["vtmb_bodypart"] = bp.name
        obj["vtmb_model"] = model.name
        obj["vtmb_filetype"] = model.filetype
        # The mesh partition of the vertex array, which Blender's per-face material slot
        # cannot express: two meshes may share a material, and a loose vertex has no face.
        obj["vtmb_meshes"] = [[slots[x.material], x.numvertices] for x in model.meshes]
        # The skinref each slot was built from, in slot order. vtmb.set_skin_family needs
        # the inverse of `slots` and cannot rebuild it: that would take the .mdl back.
        obj["vtmb_skinrefs"] = [r for r, _ in sorted(slots.items(), key=lambda kv: kv[1])]
        # What sits in each slot, so the export can find a material the user moved. The
        # slot index alone is stale the moment a slot is reordered or deleted, and 1686 of
        # the corpus's 4567 models carry two or more.
        obj["vtmb_slot_mats"] = [mm.name if mm else "" for mm in me.materials]
        # The + button in the UV Maps panel makes the layer it adds active, so reading the
        # active layer on export wrote whichever one was selected.
        obj["vtmb_uv_layer"] = uv.name
        for fam in range(len(m.skins) or 1):
            row = m.skins[fam] if fam < len(m.skins) else None
            for r, j in slots.items():
                ref = row[r] if row and 0 <= r < len(row) else r
                skin_mats[ref] = (me.materials[j].name if fam == 0
                                  else _material(m, r, content, fam).name)
        # Keyed by file vertex, not by Blender vertex: an edit rewrites the second and
        # interpolates any float attribute, so only an object property survives one.
        obj["vtmb_orig_co"] = [c * scale for v_ in verts for c in v_.pos]
        # Which import wrote the stash, so a later decode fix can be told from a live edit.
        # `vtmb_import` on the armature carries the same number and is not enough: one scene
        # holds objects from several imports.
        obj["vtmb_addon_version"] = _addon_version()
        # Names are not unique: move_and_ranged has four bodyparts whose model is called
        # sharedbones.smd, so only the position in the file tells them apart.
        obj["vtmb_index"] = gi
        context.collection.objects.link(obj)

        if model.filetype == 0:
            for b in m.bones:
                obj.vertex_groups.new(name=b.name)
            groups = obj.vertex_groups
            for vi, v_ in enumerate(verts):
                # Sliced to numbones: `bones` and `weights` are always four long and the
                # fourth weight is the shortfall 255 - sum(stored), so a vertex with none
                # live reads as bone 0 at weight 1.0 and follows the root instead of
                # nothing. 5716 shipped vertices over 13 models have numbones 0.
                n_ = v_.numbones
                for bone, weight in zip(v_.bones[:n_], v_.weights[:n_]):
                    if weight > 0.0 and 0 <= bone < len(m.bones):
                        groups[m.bones[bone].name].add([vi], weight, "REPLACE")
        cloth_notes += _stamp_cloth(obj, model, verts)
        if with_flexes:
            k, r, dropped = build_shape_keys(obj, model, scale)
            nkeys += k
            nflexrec += r
            nflexdrop += dropped

        obj.parent = arm_obj
        obj.modifiers.new(name="Armature", type="ARMATURE").object = arm_obj
        objs.append(obj)
    # Every family's material datablock, indexed by mstudiotexture_t record, so the picker
    # re-points a slot without needing a Content or the .mdl back.
    arm_obj["vtmb_skin_materials"] = [skin_mats.get(i, "")
                                      for i in range(len(m.materials))]
    note = None
    if nflexrec:
        # A shape key is coordinates only, so every record's normal delta stays in the
        # file and out of the scene. Said once per import rather than not at all.
        note = ("%d shape key(s) from %d flex vertex delta(s); the normal delta each "
                "record also carries has no Blender field and is not in the scene"
                % (nkeys, nflexrec))
        if nflexdrop:
            note += ", and %d record(s) named a vertex or a direction this file does "                     "not hold" % nflexdrop
    if cloth_notes:
        note = "; ".join(([note] if note else []) + cloth_notes)
    return objs, note


PIN_GROUP = "vtmb_pinned"


def _stamp_cloth(obj, model, verts):
    """The model's row-0 cloth object put on the mesh object, and what would not fit.

    Row 0 is what the importer draws, and 59 of the 60 shipped cloth models carry exactly
    one object there. `scale` is exact, `slack` is the median group-1 ratio and reproduces a
    preset's `s` to 3.8e-07 over 142 objects, and `(scale, slack)` names one of the 17
    presets or none, that pair being unique over them.

    Three things cannot be represented and are returned rather than dropped in silence: a
    pinned particle no vertex names (2297 over the corpus against 2614 that are named), a
    per-vertex flip bit where the writer has one flag per mesh (mixed within 62 of the 84
    shipped meshes), and a sigma that varies across the object (39 of 150).
    """
    row0 = [c for c in model.cloths if c.row == 0]
    if not row0:
        return []
    out = []
    if len(row0) > 1:
        out.append("%s carries %d cloth objects in row 0 and the exporter writes one, so "
                   "only the first is in the scene" % (obj.name, len(row0)))
    c = row0[0]
    obj["vtmb_cloth"] = 1
    pos = [verts[v].pos if v < len(verts) else (0.0, 0.0, 0.0) for v in c.pv]
    r = cloth_mod.recover(c.scale, c.springs, c.ns0, pos if c.pv else None)
    if r.preset is not None:
        obj["vtmb_cloth_preset"] = r.preset
    else:
        obj["vtmb_cloth_scale"] = c.scale
        if r.slack is not None:
            obj["vtmb_cloth_slack"] = r.slack
    if r.sigma is not None and not r.uniform:
        out.append("%s: sigma runs %.4f to %.4f across the object and one number is what "
                   "the scene holds, so a re-export writes %.4f everywhere"
                   % (obj.name, r.sigma_min, r.sigma, r.sigma))
    if r.sigma is not None and (r.preset is None or not r.uniform):
        obj["vtmb_cloth_sigma"] = r.sigma
    # The mesh binding is what says which vertices are particles at all; c.pv is the
    # inverse and names one vertex per particle where a seam has several.
    pinned, flips = set(), set()
    for e in model.meshes:
        if not e.clothbind:
            continue
        for v, (col, part, flip) in e.clothbind[0].items():
            if col != c.col:
                continue
            flips.add(flip)
            if part < c.numfixed:
                pinned.add(e.vertexoffset + v)
    obj["vtmb_cloth_pin_group"] = PIN_GROUP
    vg = obj.vertex_groups.new(name=PIN_GROUP)
    if pinned:
        vg.add(sorted(pinned), 1.0, "REPLACE")
    named = len(set(c.pv[p] for p in range(c.numfixed)) & pinned)
    if named < c.numfixed:
        out.append("%s: %d of %d pinned particles are named by no vertex and a vertex "
                   "group cannot hold them, so a re-export pins %d"
                   % (obj.name, c.numfixed - named, c.numfixed, named))
    if flips == {True}:
        obj["vtmb_cloth_flip"] = 1
    elif len(flips) > 1:
        out.append("%s: the +0x34 flip bit is set on some vertices and clear on others, "
                   "and the writer has one flag per mesh" % obj.name)
    return out


def _bind_slot(arm_obj, action):
    ad = arm_obj.animation_data
    if not hasattr(ad, "action_slot"):
        return
    slots = getattr(action, "slots", None)
    if not slots:
        return
    if ad.action_slot in list(slots):
        return
    for slot in slots:
        if getattr(slot, "target_id_type", "OBJECT") == "OBJECT":
            ad.action_slot = slot
            return
    ad.action_slot = slots[0]


_CURVES = (("location", 3), ("rotation_quaternion", 4), ("scale", 3))
_NCURVE = sum(n for _, n in _CURVES)


def _curve_factory(action, arm_obj):
    """Since Blender 4.4 an action's curves hang off the slot, and one made on
    action.fcurves is invisible to a slotted assignment. The keyword differs too."""
    ad = arm_obj.animation_data
    if not hasattr(action, "layers"):
        fcurves = action.fcurves
        return lambda path, i, group: fcurves.new(path, index=i, action_group=group)
    slot = ad.action_slot
    if slot is None:
        slot = action.slots.new("OBJECT", arm_obj.name)
        ad.action_slot = slot
    layer = action.layers[0] if len(action.layers) else action.layers.new("Layer")
    strip = layer.strips[0] if len(layer.strips) else layer.strips.new(type="KEYFRAME")
    fcurves = strip.channelbag(slot, ensure=True).fcurves
    return lambda path, i, group: fcurves.new(path, index=i, group_name=group)


def action_fcurves(action):
    """Every curve of an action, over both APIs. `action.fcurves` is gone rather than
    empty since 4.4, so this is the read side of `_curve_factory`."""
    if not hasattr(action, "layers"):
        return list(action.fcurves)
    return [fc for layer in action.layers for strip in layer.strips
            for bag in getattr(strip, "channelbags", ()) for fc in bag.fcurves]


def _rest_local_inv(m, arm_obj):
    """The parent-relative rest matrix Blender composes the basis against, inverted.

    Not the file's: an edit bone is stored as head, tail and roll, so `eb.matrix` comes
    back 1.3e-4 per entry away from what went in, and against the file's matrix that
    difference lands on every posed bone times its distance from its own head. It must
    also be parent-relative, which for a BONE_ROTATION_FROM_ROOT bone is not the file's
    own local -- hence the round trip through world matrices below.
    """
    out = []
    for b in m.bones:
        rest = arm_obj.data.bones[b.name].matrix_local
        if b.parent >= 0:
            rest = arm_obj.data.bones[m.bones[b.parent].name].matrix_local.inverted() @ rest
        out.append(rest.inverted())
    return out


def _first_seq(src, _cache={}):
    """animation index -> the first sequence citing it.

    The scratch exporter writes one sequence per action, so first-citer is the pairing
    a re-export reproduces. It is also the pairing `apply_sequences` reads back and the
    one `_event_markers` puts the markers on, so an action edited in the panel reaches
    the sequence it was stamped from. An animation no sequence names gets nothing.
    """
    got = _cache.get(src.path)
    if got is None:
        got = {}
        for s in src.seqs:
            for row in s.blends:
                for i in row:
                    got.setdefault(i, s)
        _cache[src.path] = got
    return got


def build_actions(m, arm_obj, wanted, scale, root_motion=True):
    """wanted is (source model, animation); source is m for the file's own animations
    and a chained include for the rest, which decode against their own bones."""
    arm_obj.animation_data_create()
    rest_local_inv = _rest_local_inv(m, arm_obj)
    # Indexing pose.bones here rather than per key keeps a mangled bone name an immediate
    # KeyError instead of a curve on a data path that resolves to nothing.
    paths = [(b.name, arm_obj.pose.bones[b.name].path_from_id()) for b in m.bones]
    made = []
    for src, a in wanted:
        action = bpy.data.actions.new(action_name(m, src, a))
        action.use_fake_user = True
        arm_obj.animation_data.action = action
        _bind_slot(arm_obj, action)
        new_curve = _curve_factory(action, arm_obj)
        remap = None if src is m else m.bone_remap(src)
        in_keys = root_motion and bool(a.movements)
        nframes = max(1, a.numframes)
        # keyframe_points.co is (frame, value) interleaved and the frame column is the
        # same for every curve, so it is stamped once and the rest are slice copies.
        stamp = array.array("f", bytes(8 * nframes))
        for f in range(nframes):
            stamp[2 * f] = f
        keys = [[stamp[:] for _ in range(_NCURVE)] for _ in m.bones]
        for frame in range(nframes):
            local = src.local_pose(a, frame) if remap is None \
                else m.retarget_pose(src, a, frame, remap)
            world = m.world_matrices(local)
            if in_keys:
                off = mdl_mod.root_motion_matrix(a, frame)
                world = [mdl_mod.mat_mul(off, w) for w in world]
            world = [_scaled(w, scale) for w in world]
            winv = [w.inverted() for w in world]
            at = 2 * frame + 1
            for b in m.bones:
                rel = world[b.index] if b.parent < 0 \
                    else winv[b.parent] @ world[b.index]
                # Bit-identical to assigning matrix_basis and reading the three properties
                # back, measured over 3000 random matrices, and it never enters RNA.
                loc, quat, sca = (rest_local_inv[b.index] @ rel).decompose()
                k = keys[b.index]
                k[0][at], k[1][at], k[2][at] = loc[0], loc[1], loc[2]
                k[3][at], k[4][at] = quat[0], quat[1]
                k[5][at], k[6][at] = quat[2], quat[3]
                # File quaternions are unit only to ~3e-5, so the decomposition leaves a
                # residual in scale; unkeyed it would hold the last frame's value.
                k[7][at], k[8][at], k[9][at] = sca[0], sca[1], sca[2]
        for b in m.bones:
            name, path = paths[b.index]
            k, c = keys[b.index], 0
            for prop, ncomp in _CURVES:
                for i in range(ncomp):
                    fcu = new_curve("%s.%s" % (path, prop), i, name)
                    fcu.keyframe_points.add(nframes)
                    fcu.keyframe_points.foreach_set("co", k[c])
                    # Handles are left at the origin by the bulk fill, which makes every
                    # value between two keys garbage until they are recomputed.
                    fcu.update()
                    c += 1
        action["vtmb_fps"] = a.fps
        action["vtmb_flags"] = a.flags
        seq = _first_seq(src).get(a.index)
        action["vtmb_seq_flags"] = seq.flags if seq is not None else 0
        # Stamped even when it is empty: absent means "the scene never said", and
        # the export needs that apart from "cleared", or a donor's own activity
        # can never be taken off.
        action["vtmb_activity"] = (seq.activity or "") if seq is not None else ""
        action["vtmb_numframes"] = a.numframes
        action["vtmb_source"] = src.path
        action["vtmb_anim_index"] = a.index
        # An exporter has to take the root motion back out of the keys, so record whether
        # it was ever put in -- a.movements alone does not say.
        action["vtmb_root_motion"] = in_keys
        action["vtmb_movements"] = len(a.movements)
        # Per action rather than one choice for the whole export. On an animation with no
        # block "keep" and "none" are the same write, so this reproduces the file.
        action["vtmb_root_motion_mode"] = "keep" if a.movements else "none"
        made.append(action)
    # Leave the first action assigned and slotted. Blender 4.4+ actions hold channels
    # per slot, and an unslotted assignment shows an empty Action Editor.
    if made:
        arm_obj.animation_data.action = made[0]
        _bind_slot(arm_obj, made[0])
    return made


def action_name(m, src, a):
    # No armature prefix: since Blender 4.4 the slot carries which object the action
    # belongs to, so the name only has to say which include the animation came from.
    if src is m:
        return a.name
    stem = os.path.splitext(os.path.basename(src.path))[0]
    return "%s|%s" % (stem, a.name)


def chain(m, content, seen=None):
    """Included models depth first, loaded lazily so a filled quota stops the walk.
    Yields (relative path, model or None); None means it is on no content root."""
    if seen is None:
        seen = set()
        own = getattr(m, "rel", None) or paths_mod.relative(m.path, content.roots)
        if own:
            seen.add(own.lower())
    for rel in m.includes:
        key = rel.lower()
        if key in seen:
            continue
        seen.add(key)
        sub = content.mdl(rel)
        if sub is None:
            yield rel, None
            continue
        yield rel, sub
        for item in chain(sub, content, seen):
            yield item


def pick_animations(m, content, anim_filter, max_anims, with_chained):
    """(source, animation) pairs, the file's own first, then the include chain.

    Third return is every model the walk opened, m's own path first -- the same set
    `chain` keeps to stop a cycle, kept rather than discarded so the scene records which
    files an action may have come from. A quota that stops the walk early leaves it short
    of the full closure, but never short of a file an imported action names.
    """
    want = anim_filter.lower()
    out = [(m, a) for a in m.anims if want in a.name.lower()]
    missing, opened = [], [m.path]
    if with_chained and not (max_anims and len(out) >= max_anims):
        for rel, sub in chain(m, content):
            if sub is None:
                missing.append(rel)
                continue
            opened.append(sub.path)
            out += [(sub, a) for a in sub.anims if want in a.name.lower()]
            if max_anims and len(out) >= max_anims:
                break
    return (out[:max_anims] if max_anims else out), missing, opened


def import_mdl(context, path, anim_filter="", max_anims=0,
               scale=1.0, with_mesh=True, with_anims=True, with_chained=True,
               game_root="", mods=(), extra_roots=(), use_packs=True,
               root_motion=True, with_flexes=True):
    m = mdl_mod.Mdl(path)
    roots = paths_mod.roots(path, game_root, mods, extra_roots)
    content = Content(roots, use_packs)
    name = os.path.splitext(os.path.basename(path))[0]
    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    arm_obj = build_armature(context, m, name, scale, root_motion)
    # The exporter has to undo whatever the import did, and a .blend reopened months later
    # is the only record of which options produced it.
    arm_obj["vtmb_import"] = {
        "addon_version": _addon_version(), "scale": scale,
        "with_mesh": with_mesh, "with_anims": with_anims, "with_chained": with_chained,
        "with_flexes": with_flexes,
        "use_packs": use_packs, "root_motion": root_motion, "anim_filter": anim_filter,
        "max_anims": max_anims, "game_root": game_root, "mods": list(mods),
        "extra_roots": list(extra_roots),
    }
    objs, warning = ([], None) if not with_mesh else \
        build_meshes(context, m, arm_obj, name, scale, content, with_flexes)
    build_accessories(context, m, arm_obj, name, scale)

    wanted, missing, opened = [], [], [m.path]
    if with_anims:
        wanted, missing, opened = pick_animations(m, content, anim_filter, max_anims,
                                                  with_chained)
        made = build_actions(m, arm_obj, wanted, scale, root_motion)
        _event_markers(m, made)
    # Every .mdl the import read, so the animations panel can tell an action this import
    # made from one another model left in the blend. `vtmb_includes` cannot: it is this
    # file's direct includes, and the chain is transitive -- toreador_female_armor_0
    # reaches move_and_ranged.mdl through pcidles_allsequences.mdl and names neither.
    arm_obj["vtmb_chain"] = opened
    # Over what was imported, not over m.anims: a chained import's actions come mostly from
    # included files, and the export dialogs seed Root motion from this count.
    arm_obj["vtmb_movement_anims"] = sum(1 for _, a in wanted if a.movements)
    if missing:
        note = "%d chained model(s) not found on any content root: %s" \
            % (len(missing), ", ".join(missing[:3]))
        warning = "%s; %s" % (warning, note) if warning else note

    # Flattened because a Blender ID property takes no nested list. Slots built from row N
    # while `read_materials` inverts through row 0 renames every `mstudiotexture_t`.
    arm_obj["vtmb_skin_families"] = len(m.skins)
    arm_obj["vtmb_skin_refs"] = len(m.skins[0]) if m.skins else 0
    arm_obj["vtmb_skin_table"] = [r for row in m.skins for r in row]
    arm_obj["vtmb_skin_family"] = 0
    alt = sum(1 for f in m.skins[1:] if f != m.skins[0]) if m.skins else 0
    arm_obj["vtmb_skin_alt"] = alt
    if alt:
        note = ("%d of the file's %d skin families differ from the first; the slots show "
                "family 0 and VTMB skin families in the armature's Object Data tab "
                "switches them" % (alt, len(m.skins)))
        warning = "%s; %s" % (warning, note) if warning else note

    packed = sum(len(v.index) for v in content.packs if v is not None)
    clash = [x for v in content.packs if v is not None for x in v.collisions]
    if clash:
        # Two names differing only in case are one file on disk, so which of them the
        # engine serves is not what this reader picked. Silence here would show up as a
        # texture that resolves to the wrong image and nothing to read it off.
        seen = ", ".join("%s %s" % (c[0], c[1]) for c in clash[:3])
        note = ("%d name(s) in one archive differ only in case, so one hides the "
                "other: %s" % (len(clash), seen))
        warning = "%s; %s" % (warning, note) if warning else note
    mats = {mm for o in objs for mm in o.data.materials if mm}
    lit = sum(1 for mm in mats if mm.get("vtmb_texture"))
    if mats and not lit:
        # An all-white model looks like a texture bug but is almost always an unset
        # Game root, since geometry still resolves from the .mdl's own loose tree.
        if not game_root:
            why = "Game root is not set in the addon preferences"
        elif not use_packs and packed == 0:
            why = "Read game VPKs is off and no loose materials/ tree was found"
        elif packed == 0:
            why = "no pack0NN.vpk under any of: %s" % (", ".join(roots) or "no root")
        else:
            why = "searched %d root(s) and %d packed files" % (len(roots), packed)
        note = "%d material(s), none textured: %s" % (len(mats), why)
        warning = "%s; %s" % (warning, note) if warning else note
    elif mats and lit < len(mats):
        # A model that is mostly textured has no all-white tell, so the one material the
        # reader could not resolve leaves nothing behind to notice.
        dark = sorted(mm.name for mm in mats if not mm.get("vtmb_texture"))
        note = ("%d of %d material(s) untextured: %s"
                % (len(dark), len(mats), ", ".join(dark[:3])))
        warning = "%s; %s" % (warning, note) if warning else note

    return {
        "mdl": m, "armature": arm_obj, "objects": objs, "roots": roots,
        "packs": packed, "materials": len(mats), "textured": lit,
        "bones": len(m.bones), "anims": len(wanted), "seqs": len(m.seqs),
        "chained": sum(1 for src, _ in wanted if src is not m),
        "faces": sum(len(o.data.polygons) for o in objs),
        "verts": sum(len(o.data.vertices) for o in objs),
        "warning": warning,
    }
