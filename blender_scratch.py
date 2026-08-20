"""Author a .mdl and its .dx80.vtx from a Blender scene, with no donor file.

`blender_export` rewrites a file that already exists: it opens with `Mdl(source)` and
carries every record the scene does not supply. This does not open anything. The armature
becomes `mstudiobone_t[]`, the mesh objects become bodyparts, the actions become animations
and one sequence each, and everything the format holds that no scene expresses -- cloth,
flex, eyeballs, spring bones, procedural bones, IK -- is emitted with a count of zero.

The rest pose is re-derived from the armature rather than carried. That costs `2.198e-04`
units of bone translation and `6.386e-04` units of vertex displacement at r=8 worst over
2140 bone comparisons (`plans/rest-fidelity.py`), against a `posscale` step of 1/256, and
it converges rather than accumulating.
"""

import os
import struct

import bpy  # noqa: F401

# Package-only, like blender_export, which it reuses: both are bpy-side and neither is
# imported flat by a check script.
from . import blender_export as export_mod
from . import mdl as mdl_mod
from . import mdl_build as build_mod
from . import mdl_write as write_mod
from . import vtx_rebuild as vtxr_mod

# A bone needs at least one bit of the 0xFFFC used-by mask or it gets no bone matrix and
# everything weighted to it renders nowhere.
BONE_FLAGS = build_mod.BONE_USED

# Splitting key resolution. Two loops of one vertex that agree to this are one file vertex.
NORMAL_Q = 1e-5
UV_Q = 1e-6
# An original that has not been moved sits exactly on its imported position -- float32 both
# sides -- so this is slack, not licence to have moved.
HOME_TOL = 1e-3


class Refused(Exception):
    pass


def bone_order(arm_obj):
    """Armature bones with every parent ahead of its children.

    `add_bone` refuses a forward parent reference, and the engine walks the array once in
    order, so a child emitted first reads a world matrix that has not been computed.
    """
    out, seen = [], set()

    def visit(b):
        if b.name in seen:
            return
        if b.parent is not None:
            visit(b.parent)
        seen.add(b.name)
        out.append(b)

    for b in arm_obj.data.bones:
        visit(b)
    return out


_to_file = export_mod.to_file


def _rest_local(db, parent_db):
    if parent_db is None:
        return db.matrix_local.copy()
    return parent_db.matrix_local.inverted() @ db.matrix_local


def _flags_of(arm_obj, name):
    """`vtmb_bone_flags` if the importer stashed it, and never a zero used-by mask: a bone
    with no bit of 0xFFFC gets no bone matrix and draws nothing weighted to it."""
    pb = arm_obj.pose.bones.get(name)
    f = int(pb.get("vtmb_bone_flags", 0)) if pb is not None else 0
    return f if f & 0xFFFC else f | BONE_FLAGS


def add_bones(d, arm_obj, scale=1.0, surfaceprop="flesh"):
    """Every armature bone as an `mstudiobone_t`. Returns {name: index}."""
    index = {}
    for db in bone_order(arm_obj):
        parent = -1 if db.parent is None else index[db.parent.name]
        pos, quat = _to_file(_rest_local(db, db.parent), scale)
        index[db.name] = build_mod.add_bone(d, db.name, parent, pos, quat,
                                            flags=_flags_of(arm_obj, db.name),
                                            surfaceprop=surfaceprop)
    return index


def _skin(v, groups, bone_index):
    """The four heaviest bone bindings of one Blender vertex, as [(bone, weight), ...]."""
    pairs = []
    for g in v.groups:
        name = groups[g.group].name
        if name in bone_index and g.weight > 0.0:
            pairs.append((g.weight, bone_index[name]))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [(b, w) for w, b in pairs[:4]]


def _point_vectors(me, name, width):
    """A POINT attribute as one tuple per vertex, or None if the mesh has no such layer."""
    att = me.attributes.get(name)
    if att is None or len(att.data) != len(me.vertices):
        return None
    field = "color" if width == 4 else "vector"
    buf = [0.0] * (width * len(me.vertices))
    att.data.foreach_get(field, buf)
    return [tuple(buf[i * width:(i + 1) * width]) for i in range(len(me.vertices))]


def _point_ints(me, name):
    att = me.attributes.get(name)
    if att is None or len(att.data) != len(me.vertices):
        return None
    buf = [0] * len(me.vertices)
    att.data.foreach_get("value", buf)
    return buf


def identity_map(obj, me):
    """{original file vertex -> Blender vertex} for the tags that survived editing.

    A tag only one vertex claims is trusted however far it has been moved. A tag several
    claim resolves only to a claimant still standing where the file put that original,
    read from `obj["vtmb_orig_co"]`, which is keyed by file vertex so no edit rewrites it.

    The ceiling is what stops a subdivided vertex taking the slot of an original dragged
    further from home than the new vertex sits: distance cannot tell those two apart, and
    the wrong one would then be written as the original. Unresolved is the safe answer.
    """
    orig = _point_ints(me, "vtmb_orig")
    home = obj.get("vtmb_orig_co")
    if orig is None or home is None:
        return None
    claims = {}
    for vi, o in enumerate(orig):
        claims.setdefault(o, []).append(vi)
    out = {}
    for o, vs in claims.items():
        if len(vs) == 1:
            out[o] = vs[0]
            continue
        if (o + 1) * 3 > len(home):
            continue
        h = home[o * 3:o * 3 + 3]
        near = sorted((sum((me.vertices[vi].co[c] - h[c]) ** 2
                           for c in range(3)), vi) for vi in vs)
        if near[0][0] < near[1][0] and near[0][0] <= HOME_TOL * HOME_TOL:
            out[o] = near[0][1]
    return out


def original_runs(obj, me):
    """The file's mesh partition as [(material slot, [Blender vertex, ...]), ...].

    None unless every original vertex is still present: one that has gone cannot be put
    back in its slot, and renumbering around the hole is what the map exists to avoid.
    A slot naming two runs is also refused, since a vertex the edit added is placed by
    slot and there would be no saying which of them it belongs to.
    """
    spec = obj.get("vtmb_meshes")
    ident = identity_map(obj, me)
    if not spec or ident is None:
        return None
    runs, at, slots = [], 0, set()
    for slot, n in spec:
        members = [ident.get(at + k) for k in range(int(n))]
        if any(x is None for x in members) or int(slot) in slots:
            return None
        slots.add(int(slot))
        runs.append((int(slot), members))
        at += int(n)
    return runs


def split_mesh(obj, bone_index, scale=1.0):
    """([(material slot, verts, faces), ...], unskinned, kept) per distinct corner.

    The format stores one normal, one UV and one skin per vertex, so a seam or a hard edge
    is spelled by duplicating the vertex. `kept` counts the originals written back into
    their own mesh in file order; it is 0 when the partition had to be rebuilt from the
    material slots instead, which puts every vertex in triangle-visit order.
    """
    me = obj.data
    if not me.polygons:
        raise Refused("%s has no faces" % obj.name)
    uv_layer = me.uv_layers.active
    if uv_layer is None:
        raise Refused("%s has no UV layer" % obj.name)
    me.calc_loop_triangles()
    normals = me.corner_normals
    # Custom split normals round-trip through two 16-bit angles, come back up to 9.2e-03
    # out and differ between loops of one vertex, which would split every vertex there is.
    stash = _point_vectors(me, "vtmb_normal", 3)
    uvs = _point_vectors(me, "vtmb_uv", 2)
    skins = [_skin(v, obj.vertex_groups, bone_index) for v in me.vertices]
    unskinned = sum(1 for s in skins if not s)

    def rec(vi, n, u, w):
        co = me.vertices[vi].co
        return ((co.x / scale, co.y / scale, co.z / scale), tuple(n),
                (u, 1.0 - w), skins[vi] or [(0, 1.0)])

    def key_of(vi, n, u, w):
        return (vi, round(n[0] / NORMAL_Q), round(n[1] / NORMAL_Q),
                round(n[2] / NORMAL_Q), round(u / UV_Q), round(w / UV_Q))

    if stash is not None and uvs is not None:
        runs = original_runs(obj, me)
        if runs is not None:
            out = _split_preserved(me, runs, stash, uvs, normals, uv_layer,
                                   rec, key_of)
            if out is not None:
                return out[0], unskinned, out[1]

    per_slot = {}
    for tri in me.loop_triangles:
        verts, faces, seen = per_slot.setdefault(tri.material_index, ([], [], {}))
        corners = []
        for li in tri.loops:
            vi = me.loops[li].vertex_index
            n = stash[vi] if stash else normals[li].vector
            u, w = uv_layer.data[li].uv
            key = key_of(vi, n, u, w)
            at = seen.get(key)
            if at is None:
                at = seen[key] = len(verts)
                verts.append(rec(vi, n, u, w))
            corners.append(at)
        faces.append(tuple(corners))
    return ([(k, per_slot[k][0], per_slot[k][1]) for k in sorted(per_slot)],
            unskinned, 0)


def _split_preserved(me, runs, stash, uvs, normals, uv_layer, rec, key_of):
    """Every original vertex back in its own slot, then whatever an edit added after it.

    A vertex with no original of its own is placed in the run its triangle's material slot
    names, and takes Blender's own normal: `stash` and `uvs` are indexed by Blender vertex
    and an edit copies or interpolates them, so they belong to nobody but the original.
    None when a triangle straddles two runs, which cannot be written without renumbering.
    """
    used = set()
    for tri in me.loop_triangles:
        for li in tri.loops:
            used.add(me.loops[li].vertex_index)

    where, run_of_slot, out, seen, kept = {}, {}, [], [], 0
    for ri, (slot, members) in enumerate(runs):
        run_of_slot[slot] = ri
        verts, keys = [], {}
        for vi in members:
            where[vi] = ri
            # LODs are stripped, so a vertex no face reaches is written by nobody.
            if vi not in used:
                continue
            n, (u, w) = stash[vi], uvs[vi]
            keys[key_of(vi, n, u, w)] = len(verts)
            verts.append(rec(vi, n, u, w))
            kept += 1
        out.append((verts, []))
        seen.append(keys)

    added = 0
    for tri in me.loop_triangles:
        corners = []
        for li in tri.loops:
            vi = me.loops[li].vertex_index
            ri = where.get(vi)
            own = ri is not None
            if not own:
                ri = run_of_slot.get(tri.material_index)
                if ri is None:
                    return None
            n = stash[vi] if own else tuple(normals[li].vector)
            u, w = uv_layer.data[li].uv
            key = key_of(vi, n, u, w)
            at = seen[ri].get(key)
            if at is None:
                at = seen[ri][key] = len(out[ri][0])
                out[ri][0].append(rec(vi, n, u, w))
                if not own:
                    added += 1
            corners.append((ri, at))
        if corners[0][0] != corners[1][0] or corners[1][0] != corners[2][0]:
            return None
        out[corners[0][0]][1].append(tuple(c[1] for c in corners))
    return [(runs[ri][0], out[ri][0], out[ri][1])
            for ri in range(len(runs))], kept, added


def add_meshes(d, mesh_objs, bone_index, scale=1.0, cdtexture="models/"):
    """One bodypart per object, holding one model whose meshes are its material slots.

    A bodypart is a variant selector -- the engine draws one of its models, chosen by
    `m_nBody` -- so a bodypart per material would declare that many bodygroups. 4381 of
    the 4464 shipped files carry exactly one bodypart; the material unit is the mesh.
    """
    slot_of = {}
    total_unskinned = total_kept = 0
    for obj in mesh_objs:
        runs, unskinned, kept = split_mesh(obj, bone_index, scale)
        total_unskinned += unskinned
        total_kept += kept
        mats = obj.data.materials
        meshes = []
        for slot, verts, faces in runs:
            name = mats[slot].name if slot < len(mats) and mats[slot] else obj.name
            if name not in slot_of:
                slot_of[name] = build_mod.add_material(d, name, cdtexture)
            meshes.append((slot_of[name], verts, faces))
        if meshes:
            build_mod.add_model(d, meshes,
                                bodypart=obj.get("vtmb_bodypart") or obj.name,
                                model=obj.get("vtmb_model") or obj.name + ".smd")
    return list(d.faces), total_unskinned, total_kept


_ARM = ("upperarm", "forearm", "hand", "finger", "thumb")
_LEG = ("thigh", "calf", "foot", "toe")


def _hitgroup_of(name):
    """The hit group a bone falls in: 1 head, 2 chest, 3 stomach, 4/5 arms, 6/7 legs, 0
    generic. Not derivable from a file, so this is the corpus convention over 14333
    shipped boxes -- pelvis 0 on 372 of 372, bare Spine 3 on 512 of 514, Spine1 2 on 501
    of 508, neck and head 1, the limbs 4/5 and 6/7 on 98.9%. A `vtmb_hitgroup` on the
    Blender bone wins over it."""
    n = " %s " % name.lower().replace("_", " ")
    if "pelvis" in n:
        return 0
    if "head" in n or "neck" in n or "jaw" in n:
        return 1
    right = " r " in n or "right" in n
    if any(k in n for k in _ARM):
        return 5 if right else 4
    if any(k in n for k in _LEG):
        return 7 if right else 6
    if "spine" in n:
        return 2 if n.rstrip()[-1].isdigit() else 3
    if "clavicle" in n:
        return 2
    return 0


def fit_hitboxes(d, mesh_objs, bone_index, arm_obj, scale=1.0, floor=0.05,
                 name="default"):
    """One box per bone that owns geometry, in that bone's own space. Returns the count.

    A vertex counts for every bone binding it at `floor` or more, so a joint's box covers
    the flesh on both sides of it. At 0.50 only 3 of `security_guard`'s 20 shipped boxes
    come out enclosed and at 0.05 it is 18, the two others being a degenerate pair Troika
    shipped on `Bip01 Spine2` (item 41). The bindings are the ones being written, not the
    raw vertex groups, so the box cannot enclose flesh the file does not give that bone.
    """
    inv = [[list(f[0:4]), list(f[4:8]), list(f[8:12])]
           for f in (struct.unpack_from("<12f", r.raw, 0x58) for r in d.bones)]
    lo, hi = {}, {}
    for obj in mesh_objs:
        for v in obj.data.vertices:
            p = (v.co[0] / scale, v.co[1] / scale, v.co[2] / scale)
            for bi, w in _skin(v, obj.vertex_groups, bone_index):
                if w < floor:
                    continue
                m = inv[bi]
                q = [m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]
                     for r in range(3)]
                if bi not in lo:
                    lo[bi], hi[bi] = list(q), list(q)
                    continue
                for c in range(3):
                    lo[bi][c] = min(lo[bi][c], q[c])
                    hi[bi][c] = max(hi[bi][c], q[c])
    boxes = []
    for bi in sorted(lo):
        bname = d.bones[bi].name or ""
        db = arm_obj.data.bones.get(bname)
        g = db.get("vtmb_hitgroup") if db else None
        boxes.append((bi, int(_hitgroup_of(bname) if g is None else g),
                      tuple(lo[bi]), tuple(hi[bi])))
    if boxes:
        build_mod.add_hitbox(d, boxes, name)
    return len(boxes)


def sample_action(context, arm_obj, action, d, scale=1.0, use_range=False):
    """`poses[frame][bone]` of local (pos, quat), in the bone order already authored.

    Goes through `blender_export.read_poses` rather than composing parent-relative
    matrices here: a BONE_ROTATION_FROM_ROOT bone's local rotation is its *world* one, so
    the naive chain is wrong for it, and that inverse is already written and checked.
    """
    ad = arm_obj.animation_data or arm_obj.animation_data_create()
    was, at = ad.action, context.scene.frame_current
    ad.action = action
    if use_range:
        first, last = context.scene.frame_start, context.scene.frame_end
    else:
        lo, hi = action.frame_range
        first, last = int(round(lo)), int(round(hi))
    try:
        return export_mod.read_poses(context, arm_obj, build_mod._skeleton(d),
                                     range(first, last + 1), scale)
    finally:
        ad.action = was
        context.scene.frame_set(at)


def _ordered(actions):
    """The file's animation order, not Blender's.

    `bpy.data.actions` is sorted by name, while every sequence's blend table cites an
    animation by index, so authoring in collection order renumbers all of them.
    """
    idx = [a.get("vtmb_anim_index") for a in actions]
    if any(i is None for i in idx):
        return list(actions)
    return [a for _i, a in sorted(zip(idx, actions), key=lambda p: p[0])]


def add_actions(context, arm_obj, d, actions, scale=1.0, use_range=False,
                activity="ACT_IDLE", travel="extract"):
    """`travel` is what becomes of the root bone's path. "extract" takes its net ground
    travel into one `mstudiomovement_t` and leaves the rest -- the cycle's own sway and
    rise -- in the keys, which is how a walk cycle is stored: the engine carries the
    entity, the skeleton stays put and still bobs. "none" leaves the whole path in the
    poses, which is right only for an animation that really does translate in model
    space."""
    moved = 0
    for act in _ordered(actions):
        poses = sample_action(context, arm_obj, act, d, scale, use_range)
        if not poses:
            raise Refused("action %r has no frames" % act.name)
        movements = ()
        if travel == "extract":
            movements, poses = write_mod.extract_travel(build_mod._skeleton(d), poses)
            moved += bool(movements)
        fps = float(act.get("vtmb_fps", context.scene.render.fps))
        flags = int(act.get("vtmb_flags", 0))
        a = build_mod.add_animation(d, act.name, poses, fps, flags, movements)
        build_mod.add_sequence(d, act.name, a, act.get("vtmb_activity", activity))
    return len(actions), moved


def scene_includes(arm_obj):
    """The chain an import stamped, or a template wrote, as a list of engine paths.

    An import stores it verbatim; a bone-set template writes the same key, which is where
    a generated skeleton gets its animations from. Duplicates are dropped in first-seen
    order -- the engine walks the array and a repeat costs a whole second bone map.
    """
    out = []
    for p in (arm_obj.get("vtmb_includes") or ()):
        p = str(p).replace("\\", "/").strip()
        if p and p not in out:
            out.append(p)
    return out


def build(context, arm_obj, mesh_objs, actions, name, scale=1.0, surfaceprop="flesh",
          cdtexture="models/", hull=None, use_range=False, activity="ACT_IDLE",
          hitboxes=False, travel="extract", includes=()):
    """(Desc, faces) for the scene. `hull` is (min, max) and stays the caller's:
    @180/@192 are the movement hull the .qc's $hbox sets, not the mesh's bounds."""
    d = build_mod.new(name, surfaceprop)
    if hull is not None:
        lo, hi = hull
        _set_hull(d, lo, hi)
    for p in includes:
        build_mod.add_include(d, p)
    bone_index = add_bones(d, arm_obj, scale, surfaceprop)
    if not bone_index:
        raise Refused("the armature has no bones")
    faces, unskinned, kept = add_meshes(d, mesh_objs, bone_index, scale, cdtexture)
    if hitboxes:
        fit_hitboxes(d, mesh_objs, bone_index, arm_obj, scale)
    _n, moved = add_actions(context, arm_obj, d, actions, scale, use_range, activity,
                            travel)
    return d, faces, unskinned, kept, moved


def _set_hull(d, lo, hi):
    struct.pack_into("<3f", d.hdr, 180, *lo)
    struct.pack_into("<3f", d.hdr, 192, *hi)
    # Only reached by a model with no sequence at all: mdl_build.stamp_sequence_boxes
    # rewrites @168 as the centre of sequence 0's box, which is studiomdl's own rule.
    struct.pack_into("<3f", d.hdr, 168, *[(a + b) * 0.5 for a, b in zip(lo, hi)])


def fit_hull(mesh_objs, scale=1.0):
    """(min, max) over every vertex handed in, in file units, or None when there are none.

    A starting point and not a truth: @180/@192 is the movement hull the .qc's `$hbox`
    sets, which is why `build` takes it from the caller rather than deriving it. Without
    one `mdl_build.new` leaves (-16,-16,0)..(16,16,72), so a bat claims a humanoid volume.
    Mesh-local like `split_mesh`, so an object transform is ignored by both alike.
    """
    lo = hi = None
    for obj in mesh_objs:
        for v in obj.data.vertices:
            p = [v.co[c] / scale for c in range(3)]
            if lo is None:
                lo, hi = list(p), list(p)
                continue
            for c in range(3):
                lo[c] = min(lo[c], p[c])
                hi[c] = max(hi[c], p[c])
    return None if lo is None else (tuple(lo), tuple(hi))


def embedded_name(path):
    """What goes in `studiohdr_t.name[128]`: the path from `models/` down."""
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    slash = stem.replace("\\", "/")
    cut = slash.lower().rfind("/models/")
    base = slash[cut + 1:] if cut >= 0 else "models/" + os.path.basename(stem)
    return base + ".mdl"


def write(d, faces, path, checksum):
    """Emit both files. Returns (mdl bytes, vtx bytes, vtx stats)."""
    data = build_mod.emit(d, checksum)
    m = mdl_mod.Mdl("<scene>", data=data)
    vtx, st = vtxr_mod.scratch(m, faces, checksum)
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    with open(stem + ".mdl", "wb") as f:
        f.write(data)
    with open(stem + ".dx80.vtx", "wb") as f:
        f.write(vtx)
    return data, vtx, st


def export_scene(context, arm_obj, mesh_objs, actions, path, checksum, **kw):
    d, faces, unskinned, kept, moved = build(context, arm_obj, mesh_objs, actions,
                                             embedded_name(path), **kw)
    data, vtx, st = write(d, faces, path, checksum)
    return {"bytes": len(data), "vtx_bytes": len(vtx), "bones": len(d.bones),
            "travelling": moved,
            "bodyparts": len(d.bodyparts), "materials": len(d.textures),
            "anims": len(d.anims), "seqs": len(d.seqs),
            "includes": len(d.includes),
            "hitboxes": sum(len(r.kids) for r in d.hitboxsets),
            "faces": st["tris_out"], "verts": st["verts_out"],
            "model_verts": sum(len(x.extra.get("tangents") or b"") // 16
                               for bp in d.bodyparts for x in bp.kids),
            "kept": kept,
            "unskinned": unskinned, "dropped": dict(d.dropped)}
