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

    Length is cosmetic -- it moves the tail along the bone's own Y and leaves head, roll
    and matrix_local alone -- but nothing here may be a constant: every length has to
    carry `scale`, or a non-1.0 import leaves the leaf bones dwarfing the real ones.
    Parents precede children, so one pass suffices.
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


def build_armature(context, m, name, scale):
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

    for b in m.bones:
        pb = arm_obj.pose.bones[b.name]
        pb.rotation_mode = "QUATERNION"
        # rotscale bounds what can ever be written back to this bone.
        pb["vtmb_rotscale"] = list(b.rotscale)
        pb["vtmb_posscale"] = list(b.posscale)
        pb["vtmb_rot_limit"] = [32767.0 * s for s in b.rotscale]

    arm_obj["vtmb_scale"] = scale
    arm_obj["vtmb_checksum"] = m.checksum
    arm_obj["vtmb_source"] = m.path
    arm_obj["vtmb_includes"] = list(m.includes)
    arm_obj["vtmb_sequences"] = [
        {"label": s.label, "activity": s.activity,
         "groupsize": list(s.groupsize),
         "blends": [[m.anims[i].name if 0 <= i < len(m.anims) else ""
                     for i in col] for col in s.blends]}
        for s in m.seqs]
    return arm_obj


def _vmt_basetexture(path, blob=None):
    if blob is None:
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except OSError:
            return None
    # Odd-indexed pieces of a split on '"' are the quoted tokens, so the value is the
    # token after the key -- not text[key:].split('"')[1], which is the gap between.
    tokens = blob.decode("latin1").split('"')[1::2]
    for i, tok in enumerate(tokens[:-1]):
        if tok.strip().lower() == "$basetexture":
            return tokens[i + 1].strip()
    return None


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
        img = _packed_image(name, tth_mod.to_dds(w, h, t.format, mip), ".dds", source)
    except ValueError:
        img = None
    if img is not None:
        return img
    # A Blender built without DDS lands here too, not just the formats DDS cannot
    # carry, so the slow Python decoder is the fallback for both.
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


def _find_image(content, search_paths, name):
    """VMT first, since $basetexture may point outside the model's own directory."""
    for stem in paths_mod.material_stems(name, search_paths):
        vmt, blob = content.find(stem + ".vmt")
        if vmt:
            base = _vmt_basetexture(vmt, blob)
            if base:
                img = _image(content, paths_mod.MATERIALS_DIR + "/"
                             + base.replace("\\", "/").strip("/"))
                if img is not None:
                    return img
        img = _image(content, stem)
        if img is not None:
            return img
    return None


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


def _material(m, index, content):
    ref = index
    if m.skins and 0 <= index < len(m.skins[0]):
        ref = m.skins[0][index]
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

    img = _find_image(content, m.material_paths, name) if bsdf is not None else None
    if img is None:
        return mat
    mat["vtmb_texture"] = img.get("vtmb_source") or img.filepath or img.name
    mat.diffuse_color = _average_color(img)
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (bsdf.location.x - 400, bsdf.location.y)
    mat.node_tree.links.new(bsdf.inputs["Base Color"], tex.outputs["Color"])
    return mat


def build_meshes(context, m, arm_obj, name, scale, content):
    path, blob = content.companion(m, vtx_mod.SUFFIXES)
    if path is None:
        return [], "no .vtx for this .mdl on any content root, so no faces to import"
    v = vtx_mod.Vtx(path, blob)
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
        for tri in v.triangles(g):
            acc.append((tuple(mesh.vertexoffset + ids[i] for i in tri),
                        mesh.material))

    objs = []
    for gi, (bp, model) in enumerate(models):
        faces = faces_by_model.get(gi)
        if not faces:
            continue
        verts = m.vertices(model)
        me = bpy.data.meshes.new("%s_%s" % (name, model.name.rsplit(".", 1)[0]))
        me.from_pydata([[c * scale for c in v_.pos] for v_ in verts], [],
                       [list(f[0]) for f in faces])
        me.update()

        slots = {}
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

        if model.filetype == 0:
            me.normals_split_custom_set_from_vertices(
                [v_.normal for v_ in verts])
            # Custom split normals are stored as two 16-bit angles and come back off by
            # up to 9.2e-03, so the exporter needs the exact ones to write back.
            att = me.attributes.new("vtmb_normal", "FLOAT_VECTOR", "POINT")
            att.data.foreach_set("vector",
                                 [c for v_ in verts for c in v_.normal])
            # Slot order is neither by weight nor by bone -- it is whatever studiomdl read
            # out of the SMD -- so vertex groups alone cannot reproduce it.
            att = me.attributes.new("vtmb_skin", "FLOAT_COLOR", "POINT")
            att.data.foreach_set("color", [float(b) for v_ in verts
                                           for b in v_.bones])
            att = me.attributes.new("vtmb_weight", "FLOAT_COLOR", "POINT")
            att.data.foreach_set("color", [w for v_ in verts for w in v_.weights])
            att = me.attributes.new("vtmb_numbones", "INT", "POINT")
            att.data.foreach_set("value", [v_.numbones for v_ in verts])

        obj = bpy.data.objects.new(me.name, me)
        obj["vtmb_bodypart"] = bp.name
        obj["vtmb_model"] = model.name
        obj["vtmb_filetype"] = model.filetype
        # Names are not unique: move_and_ranged has four bodyparts whose model is called
        # sharedbones.smd, so only the position in the file tells them apart.
        obj["vtmb_index"] = gi
        context.collection.objects.link(obj)

        if model.filetype == 0:
            for b in m.bones:
                obj.vertex_groups.new(name=b.name)
            groups = obj.vertex_groups
            for vi, v_ in enumerate(verts):
                for bone, weight in zip(v_.bones, v_.weights):
                    if weight > 0.0 and 0 <= bone < len(m.bones):
                        groups[m.bones[bone].name].add([vi], weight, "REPLACE")
        obj.parent = arm_obj
        obj.modifiers.new(name="Armature", type="ARMATURE").object = arm_obj
        objs.append(obj)
    return objs, None


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
        travel = root_motion and bool(a.movements)
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
            if travel:
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
        action["vtmb_numframes"] = a.numframes
        action["vtmb_source"] = src.path
        action["vtmb_anim_index"] = a.index
        # An exporter has to take the travel back out of the keys, so record whether it
        # was ever put in -- a.movements alone does not say.
        action["vtmb_root_motion"] = travel
        action["vtmb_movements"] = len(a.movements)
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
    """(source, animation) pairs, the file's own first, then the include chain."""
    want = anim_filter.lower()
    out = [(m, a) for a in m.anims if want in a.name.lower()]
    missing = []
    if with_chained and not (max_anims and len(out) >= max_anims):
        for rel, sub in chain(m, content):
            if sub is None:
                missing.append(rel)
                continue
            out += [(sub, a) for a in sub.anims if want in a.name.lower()]
            if max_anims and len(out) >= max_anims:
                break
    return (out[:max_anims] if max_anims else out), missing


def import_mdl(context, path, anim_filter="", max_anims=0,
               scale=1.0, with_mesh=True, with_anims=True, with_chained=True,
               game_root="", mods=(), extra_roots=(), use_packs=True,
               root_motion=True):
    m = mdl_mod.Mdl(path)
    roots = paths_mod.roots(path, game_root, mods, extra_roots)
    content = Content(roots, use_packs)
    name = os.path.splitext(os.path.basename(path))[0]
    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    arm_obj = build_armature(context, m, name, scale)
    # The exporter has to undo whatever the import did, and a .blend reopened months later
    # is the only record of which options produced it.
    arm_obj["vtmb_import"] = {
        "addon_version": _addon_version(), "scale": scale,
        "with_mesh": with_mesh, "with_anims": with_anims, "with_chained": with_chained,
        "use_packs": use_packs, "root_motion": root_motion, "anim_filter": anim_filter,
        "max_anims": max_anims, "game_root": game_root, "mods": list(mods),
        "extra_roots": list(extra_roots),
    }
    objs, warning = ([], None) if not with_mesh else \
        build_meshes(context, m, arm_obj, name, scale, content)

    wanted, missing = [], []
    if with_anims:
        wanted, missing = pick_animations(m, content, anim_filter, max_anims,
                                          with_chained)
        build_actions(m, arm_obj, wanted, scale, root_motion)
    if missing:
        note = "%d chained model(s) not found on any content root: %s" \
            % (len(missing), ", ".join(missing[:3]))
        warning = "%s; %s" % (warning, note) if warning else note

    packed = sum(len(v.index) for v in content.packs if v is not None)
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

    return {
        "mdl": m, "armature": arm_obj, "objects": objs, "roots": roots,
        "packs": packed, "materials": len(mats), "textured": lit,
        "bones": len(m.bones), "anims": len(wanted), "seqs": len(m.seqs),
        "chained": sum(1 for src, _ in wanted if src is not m),
        "faces": sum(len(o.data.polygons) for o in objs),
        "verts": sum(len(o.data.vertices) for o in objs),
        "warning": warning,
    }
