#!/usr/bin/env python3
"""Blender side of the VTMB MDL exporter: a scene back into an existing .mdl.

The file is rebuilt from its own decoded records with the scene applied on top: the
animations named, the skeleton, the material names, the sequence table, and whichever of
positions, normals, UVs and weights the caller asked for. Everything else keeps the donor's
bytes, so an unedited model comes back byte for byte.

A mesh whose vertex count moved, or whose UV corners disagree -- a seam, which the format
spells by duplicating the vertex -- is rebuilt whole and its `.dx80.vtx` rewritten beside
the model. A mesh that did not move takes the index-for-index patch instead.
"""

import math
import os
import struct

import bpy
import mathutils

from . import mdl as mdl_mod
from . import mdl_build as build_mod
from . import mdl_rebuild as rebuild_mod
from . import mdl_write as write_mod
from . import mesh_write as mesh_mod
from . import paths as paths_mod
from . import vtx_rebuild as vtxr_mod


UV_TOL = 1e-6
# Custom split normals are stored as two 16-bit angles, so an untouched mesh's corners
# disagree by up to 9.2e-03; under this a normal is taken as unedited.
NORMAL_EPS = 1.5e-2
NORMAL_ATTR = "vtmb_normal"
# Against the import stash an untouched bone compares exactly; this only absorbs a
# recomposition of the same float32s.
REST_EPS = 1e-6
# Without a stash the file's record is the baseline, and Blender loses 3.2e-05 of a bone's
# offset to chain composition and 1.4e-03 per quaternion entry to head/tail/roll storage.
REST_POS_EPS = 1e-3
REST_POS_REL = 1e-4
REST_QUAT_EPS = 2e-3
# A v2531 bone is a position and a quaternion; there is no scale field anywhere in
# `mstudiobone_t`. So a scaled pose bone cannot be written, and a collapsed one is worse
# than unwritable: `local_from_world` inverts a parent's matrix to take its children back
# into its frame, and a singular matrix divides by zero in `mat_inverse` -- three frames
# below where a user could see what caused it.
#
# Two tests, because the two need different bounds. Neither can be tight: file quaternions
# are unit only to ~4.4e-5, the importer hands Blender the matrix rather than the channels,
# and Blender back-solves a scale out of it. Measured over andrei, toreador, vv and
# werewolf on an untouched import -- 48 000 samples over 40 frames each -- `pb.scale`
# reaches 1.62e-04 off unit and the object-space `pb.matrix` reaches 4.0e-04, both worst on
# `Bip01 R Thigh`. So this is set 60x above the drift and still far under any scale a user
# would type.
POSE_SCALE_EPS = 1e-2
# A rotation has determinant 1, so this only has to separate collapsed from not.
SINGULAR_DET = 1e-9
SKIN_ATTR = "vtmb_skin"
WEIGHT_ATTR = "vtmb_weight"
COUNT_ATTR = "vtmb_numbones"


def _rows(mat, scale):
    """A mathutils.Matrix as the format's 3x4, with the import scale taken back out."""
    return [[mat[i][0], mat[i][1], mat[i][2], mat[i][3] / scale] for i in range(3)]


def bone_map(m, arm_obj):
    """{file bone name: armature bone name} for the file bones the armature still has.

    Blender records nothing about a rename, and every reader here resolves a file bone by
    name, so without a stash a renamed bone is indistinguishable from one deleted and one
    added -- and `drop_missing_bones` would take it out of the file. The importer stamps
    `vtmb_bone_name` on each pose bone, so the bone says which record it came from whatever
    it is called now.

    The stash is taken first and a Blender bone is claimed once, so a name freed by a rename
    cannot be picked up by a second file bone. A bone with no stash -- one the user built,
    or a scene older than the stamp -- resolves by its own name, which is what every export
    before this did.
    """
    dbs = arm_obj.data.bones
    stash = {}
    for pb in arm_obj.pose.bones:
        was = pb.get("vtmb_bone_name")
        if was:
            stash.setdefault(str(was), pb.name)
    out, taken = {}, set()
    for b in m.bones:
        got = stash.get(b.name)
        if got is not None and got in dbs and got not in taken:
            out[b.name] = got
            taken.add(got)
    for b in m.bones:
        if b.name not in out and b.name in dbs and b.name not in taken:
            out[b.name] = b.name
            taken.add(b.name)
    return out


def renamed_bones(m, arm_obj, bmap=None):
    """{file bone index: the name Blender now gives it}, for the ones that differ.

    A `.mdl` bone name is a string-table entry `emit` authors from `Rec.name`, so writing
    one costs nothing. What it costs elsewhere is the include chain, whose only join is by
    name: `Studio_BuildChainedModelBoneMaps` pairs two files' bones with `__strcmpi` and
    leaves an unmatched entry at its -1 sentinel, so a bone renamed on one side of a chain
    stops being driven by the other. The operator reports that where the file has includes.
    """
    bmap = bone_map(m, arm_obj) if bmap is None else bmap
    return {k: bmap[b.name] for k, b in enumerate(m.bones)
            if b.name in bmap and bmap[b.name] != b.name}


def _det3(w):
    """Determinant of a 3x4's rotation part."""
    return (w[0][0] * (w[1][1] * w[2][2] - w[1][2] * w[2][1])
            - w[0][1] * (w[1][0] * w[2][2] - w[1][2] * w[2][0])
            + w[0][2] * (w[1][0] * w[2][1] - w[1][1] * w[2][0]))


def unwritable_scale(pbs, world, frame):
    """Why this frame cannot be written, naming the bone, or None if it can.

    Parents precede their children in the file, so a scale that came down the chain names
    the bone it started on rather than the first descendant to notice it.
    """
    for pb, w in zip(pbs, world):
        sc = tuple(pb.scale)
        if any(abs(x - 1.0) > POSE_SCALE_EPS for x in sc):
            return ("%r is scaled (%.4g, %.4g, %.4g) at frame %d. A v2531 bone holds a "
                    "position and a rotation and has no scale field, so there is nowhere "
                    "to write this: key the scale back to 1 and build the size into the "
                    "geometry instead" % (pb.name, sc[0], sc[1], sc[2], frame))
        if abs(_det3(w)) < SINGULAR_DET:
            return ("%r has no volume at frame %d, so its pose matrix cannot be inverted "
                    "and no child of it can be taken back into its frame. A constraint or "
                    "a driver is the usual cause where the bone's own scale reads 1"
                    % (pb.name, frame))
    return None


def read_poses(context, arm_obj, m, frames, scale, anim=None, root_motion_in_keys=False):
    """Local (pos, quat) per bone per frame, inverting everything the importer applied.

    `pose.bones[].matrix` is object space, and the importer's rest_local_inv and
    matrix_local factors cancel down the chain, so it is exactly the world matrix the
    importer built -- no reconstruction from matrix_basis needed.
    """
    bmap = bone_map(m, arm_obj)
    missing = [b.name for b in m.bones if b.name not in bmap]
    if missing:
        raise ValueError("armature has no bone %s (and %d more)"
                         % (missing[0], len(missing) - 1))
    pbs = [arm_obj.pose.bones[bmap[b.name]] for b in m.bones]
    scene = context.scene
    out = []
    for f in frames:
        scene.frame_set(f)
        world = [_rows(pb.matrix, scale) for pb in pbs]
        bad = unwritable_scale(pbs, world, f)
        if bad:
            # `Refused` and not ValueError: both operators turn it into one clean line and
            # no traceback, which is what a limit of the format deserves.
            raise build_mod.Refused(bad)
        if root_motion_in_keys and anim is not None and anim.movements:
            off = mdl_mod.mat_inverse(mdl_mod.root_motion_matrix(anim, f))
            world = [mdl_mod.mat_mul(off, w) for w in world]
        out.append(write_mod.local_from_world(m, world))
    return write_mod.unwind_signs(out)


def to_file(local, scale):
    """(pos, quat) in the file's convention out of a parent-relative Blender matrix.

    The importer scales translation only, so the rotation comes back untouched and only the
    offset divides out. Blender orders a quaternion w first and the file x first.
    """
    t, q = local.to_translation(), local.to_quaternion()
    return (t.x / scale, t.y / scale, t.z / scale), (q.x, q.y, q.z, q.w)


def bone_flags(arm_obj, name):
    """`vtmb_bone_flags` as the importer stashed it, or None to keep the file's own.

    Verbatim, with none of `blender_scratch`'s zero-used-by-mask guard: there is a file
    value here and it is already what the engine accepts -- manbat's Dummy01 ships 0x0 --
    so forcing a bit in would edit a bone nobody asked to edit.
    """
    pb = arm_obj.pose.bones.get(name)
    f = pb.get("vtmb_bone_flags") if pb is not None else None
    return None if f is None else int(f)


def rest_baseline(arm_obj, name):
    """`vtmb_rest_local` as a Matrix, or None where the importer stashed none."""
    pb = arm_obj.pose.bones.get(name)
    v = pb.get("vtmb_rest_local") if pb is not None else None
    if v is None or len(v) != 16:
        return None
    return mathutils.Matrix([tuple(v[i * 4:i * 4 + 4]) for i in range(4)])


def moved_from_file(m, b, pos, quat):
    """Did this bone leave the file's own record, allowing for what Blender cannot hold.

    Only for a bone the importer stashed no baseline for. The bounds are far coarser than
    an edit, so the answer is one-sided: a real edit under them is missed, silently.
    """
    tol = REST_POS_EPS + REST_POS_REL * max(abs(x) for x in b.pos)
    return not (all(abs(x - y) <= tol for x, y in zip(pos, b.pos))
                and all(abs(x - y) <= REST_QUAT_EPS for x, y in zip(quat, b.quat)))


def read_bones(m, arm_obj, scale):
    """{index: (pos, quat, flags)} for the bones of `m` the armature actually moved.

    A bone still sitting on its imported pose is left out, so an unedited export does not
    touch the bone array and comes back byte for byte -- the same reason the mesh puts an
    unmoved vertex back with its file bytes rather than Blender's.

    The file's parent chain is what the local matrix is taken against, not Blender's. The
    two agree on an imported armature, and where they disagree it is the file the output
    has to stay consistent with.
    """
    dbs = arm_obj.data.bones
    bmap = bone_map(m, arm_obj)
    missing = [b.name for b in m.bones if b.name not in bmap]
    if missing:
        raise ValueError("armature has no bone %s (and %d more)"
                         % (missing[0], len(missing) - 1))
    out = {}
    for k, b in enumerate(m.bones):
        local = dbs[bmap[b.name]].matrix_local
        if b.parent >= 0:
            local = dbs[bmap[m.bones[b.parent].name]].matrix_local.inverted() @ local
        pos, quat = to_file(local, scale)
        flags = bone_flags(arm_obj, bmap[b.name])
        # A quaternion and its negation are one rotation, so the nearer sign wins or a bone
        # the importer flipped reads as moved.
        if sum(x * y for x, y in zip(quat, b.quat)) < 0:
            quat = tuple(-x for x in quat)
        base = rest_baseline(arm_obj, bmap[b.name])
        if base is None:
            moved = moved_from_file(m, b, pos, quat)
        else:
            moved = any(abs(local[i][j] - base[i][j]) > REST_EPS
                        for i in range(4) for j in range(4))
        if not moved and (flags is None or flags == b.flags):
            continue
        out[k] = (pos, quat, flags)
    return out


def surplus_bones(m, arm_obj):
    """Bones the armature has that the file does not, in armature order.

    `add_surplus_bones` writes these, so on the export path this comes back empty and a
    name left in it is a bone that could not be placed. The export dialog calls it before
    anything is written, where it is the list of bones the export is about to add.
    """
    have = set(bone_map(m, arm_obj).values())
    return [b.name for b in arm_obj.data.bones if b.name not in have]


class _EditedBones(object):
    """`m` with the bone list an add or a removal left behind.

    Either changes the file's bone list and nothing else a reader takes off `m`: geometry,
    materials and animations are all still the donor's. So the bone list is substituted and
    every other attribute delegates. `mdl_build._skeleton` is already what hands that same
    list to `mdl_write`, and `blender_scratch` already passes it where an `Mdl` is expected.
    """

    def __init__(self, m, d):
        self._m = m
        self.bones = build_mod._skeleton(d).bones

    def __getattr__(self, name):
        return getattr(self._m, name)


def drop_missing_bones(d, m, arm_obj):
    """Take out of `d` every bone the armature no longer has, highest index first.

    Blender performed the deletion and this is the file catching up, so nothing here decides
    what a removal means -- `read_bones` takes each survivor's rest matrix against the file's
    reparented chain afterwards, which is Blender's own geometry.

    Descending, because an index is stated against the file as it was and `remove_bone`
    renumbers everything above the bone it takes and nothing below. `Refused` is left to
    reach the operator, which reports it.
    """
    gone = [k for k, b in enumerate(m.bones) if b.name not in bone_map(m, arm_obj)]
    out = []
    for k in reversed(gone):
        name, kids, slots, rigid, rebound = build_mod.remove_bone(d, k)
        out.append({"name": name, "children": kids, "slots": slots, "rigid": rigid,
                    "rebound": rebound})
    out.reverse()
    return out


def add_surplus_bones(d, m, arm_obj, scale):
    """Append every armature bone the file does not have, parents before children.

    The mirror of `drop_missing_bones`: Blender put the bone there and this is the file
    catching up. `add_bone` appends, so nothing already in the file renumbers -- what it
    must not do is emit a child ahead of its parent, because the engine walks the array
    once in order and a child read first composes against a world matrix that has not been
    computed. Armature order is neither hierarchy order nor stable, so a bone waits until
    its parent has an index.

    The rest transform is taken the way `read_bones` takes it, parent-local off Blender's
    own `matrix_local`, so a bone whose parent is also new is placed against the parent
    Blender shows rather than against anything the file holds.

    A new bone has no `vtmb_bone_flags` to read, so it takes `BONE_USED` -- 62439 of 62702
    corpus bones carry that bit, and a bone carrying no bit of the 0xfffc mask gets no bone
    matrix, so anything skinned to it draws nothing.
    """
    dbs = arm_obj.data.bones
    have = bone_map(m, arm_obj)
    index = {}
    for k, b in enumerate(m.bones):
        arm_name = have.get(b.name)
        if arm_name is not None:
            index[arm_name] = k
    todo, out = [b for b in dbs if b.name not in index], []
    while todo:
        left = []
        for db in todo:
            par = db.parent
            if par is not None and par.name not in index:
                left.append(db)
                continue
            local = db.matrix_local
            if par is not None:
                local = par.matrix_local.inverted() @ local
            pos, quat = to_file(local, scale)
            flags = bone_flags(arm_obj, db.name)
            i = build_mod.add_bone(d, db.name,
                                   index[par.name] if par is not None else -1,
                                   pos, quat,
                                   build_mod.BONE_USED if flags is None else flags)
            index[db.name] = i
            out.append({"name": db.name, "index": i,
                        "parent": par.name if par is not None else None})
        if len(left) == len(todo):
            # Blender cannot hold a parent cycle, so this is a reader that stopped
            # agreeing with `data.bones` rather than a scene the user can build.
            raise ValueError("cannot place %s: no bone of it has a parent that resolves"
                             % ", ".join(sorted(b.name for b in left)[:4]))
        todo = left
    return out


def mesh_objects(m, source):
    """{(bodypart index, model index): object} for the scene meshes belonging to `m`.

    The importer stamps `vtmb_bodypart`/`vtmb_model` on each object and `vtmb_source` on
    the armature it parents them to, so the file is identified through the parent rather
    than by object name, which the user is free to change.
    """
    flat = mesh_mod.models_of(m)
    # Model names repeat across bodyparts, so the name pair is only a fallback for objects
    # an older version imported without `vtmb_index`.
    want = {(bp.name, mo.name): (bi, mi) for bi, mi, bp, mo in flat}
    out = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        par = obj.parent
        if par is not None and not same_file(par.get("vtmb_source"), source):
            continue
        gi = obj.get("vtmb_index")
        if gi is not None and 0 <= gi < len(flat):
            key = flat[gi][0], flat[gi][1]
        else:
            key = want.get((obj.get("vtmb_bodypart"), obj.get("vtmb_model")))
        if key is not None:
            out.setdefault(key, obj)
    return out


def renamed_models(m, source):
    """{(bodypart, model): the name Blender now gives it} for the mesh datablocks renamed.

    The import puts `mstudiomodel_t.name` on the datablock and stashes both what the file
    said and what Blender actually assigned, because Blender caps a datablock name at 63
    bytes and suffixes a duplicate: 3 of the corpus's 4567 model names are over 63 and 14
    repeat inside one file. Comparing against what was assigned is therefore the only way
    to tell a rename from Blender's own bookkeeping, and a datablock with no stash -- an
    older scene, or a mesh the user built -- is left alone rather than guessed at.
    """
    out = {}
    for key, obj in mesh_objects(m, source).items():
        me = obj.data
        was = me.get("vtmb_model_label")
        if was is not None and me.name != was:
            out[key] = me.name
    return out


def _per_vertex(me, count, get):
    """Every distinct per-loop value each vertex carries, as a list per vertex."""
    out = [[] for _ in range(count)]
    for loop in me.loops:
        out[loop.vertex_index].append(get(loop))
    return out


def _one_uv(me, count, uv_layer):
    """One UV per vertex, refusing a vertex whose loops disagree.

    The refusal is unreachable: `_must_rebuild` tests the same vertices against the same
    UV_TOL under the same `"uvs" in fields` gate and routes a disagreement to the split
    path before this runs. It stands as the precondition of the in-place path rather than
    as a limit -- the format spells a seam by duplicating the vertex, and the rebuild does.
    """
    per = _per_vertex(me, count, lambda l: tuple(uv_layer.data[l.index].uv))
    split = [i for i, vs in enumerate(per)
             if any(max(abs(a - b) for a, b in zip(vs[0], v)) > UV_TOL for v in vs)]
    if split:
        raise ValueError("%s: %d vertices carry more than one UV (first is vertex %d), "
                         "which the format can only store by splitting the vertex and "
                         "that would change the vertex count"
                         % (me.name, len(split), split[0]))
    return [vs[0] if vs else None for vs in per]


def _one_normal(me, count, stash):
    """One normal per vertex, and how many were taken from Blender rather than `stash`.

    Averaging the corners is only ever a fallback: the round trip through custom split
    normals is lossy, so a vertex still within NORMAL_EPS of the normal the importer
    stashed is written back exactly instead of being degraded by re-export.
    """
    per = _per_vertex(me, count, lambda l: tuple(me.corner_normals[l.index].vector))
    out, edited = [], 0
    for i, vs in enumerate(per):
        if not vs:
            out.append(stash[i] if stash else None)
            continue
        avg = [sum(v[c] for v in vs) / len(vs) for c in range(3)]
        n = math.sqrt(sum(x * x for x in avg)) or 1.0
        avg = tuple(x / n for x in avg)
        if stash and max(abs(a - b) for a, b in zip(avg, stash[i])) <= NORMAL_EPS:
            out.append(stash[i])
        else:
            out.append(avg)
            edited += 1
    return out, edited


def _stashed(me, name, count, width=4, field="color"):
    """An attribute the importer wrote, as one tuple per vertex."""
    att = me.attributes.get(name)
    if att is None or len(att.data) != count:
        return None
    flat = [0.0] * (count * width)
    att.data.foreach_get(field, flat)
    return [tuple(flat[i * width:i * width + width]) for i in range(count)]


def _one_skin(v, groups, bone_index, stash):
    """(weights, bones, count) for one vertex, keeping the file's slot order where it
    still describes the same skinning -- the order is not derivable, so losing it would
    move bytes on a rewrite that changed nothing."""
    pairs = sorted(((g.weight, -bone_index[groups[g.group].name])
                    for g in v.groups if groups[g.group].name in bone_index),
                   reverse=True)[:4]
    w = [x for x, _ in pairs] + [0.0] * (4 - len(pairs))
    b = [-x for _, x in pairs] + [0] * (4 - len(pairs))
    n = sum(1 for x in w if x > 0.0)
    if stash is not None:
        sw, sb, sn = stash
        if sorted(zip(mesh_mod.u8_weights(sw), (int(x) for x in sb))) == \
           sorted(zip(mesh_mod.u8_weights(w), b)):
            return list(sw), [int(x) for x in sb], sn
    return w, b, n


def crowded_vertices(obj, bone_index):
    """Vertices carrying a fifth bone group, which no record can hold.

    Counted off the object and not off what `_one_skin` returns, which has already taken
    the four heaviest. A group naming no bone is not a binding and does not count -- that
    is a group which never skinned, not skinning the format lost. What the fifth held is
    not simply gone: the three stored bytes come from the kept weights and the engine
    derives the fourth as their shortfall, so the fourth bone absorbs it.
    """
    groups = obj.vertex_groups
    n = 0
    for v in obj.data.vertices:
        ws = [g.weight for g in v.groups if groups[g.group].name in bone_index]
        if len(ws) > 4 and mesh_mod.dropped_weights(ws):
            n += 1
    return n


def read_mesh(obj, model, bone_index, fields):
    """One `mdl.Vertex` per file vertex, in file order, inverting what the importer did.

    Only the requested fields are read, because the per-loop agreement check refuses a
    mesh a write of some other field would have been fine with.
    """
    me = obj.data
    if len(me.vertices) != model.numvertices:
        raise ValueError("%s: %d vertices in the scene, %d in %s -- a mesh write cannot "
                         "change the count"
                         % (obj.name, len(me.vertices), model.numvertices, model.name))
    n = model.numvertices
    uvs = normals = [None] * n
    edited = 0
    if "uvs" in fields:
        uv_layer = me.uv_layers.active
        if uv_layer is None:
            raise ValueError("%s has no UV layer" % obj.name)
        uvs = _one_uv(me, n, uv_layer)
    if "normals" in fields:
        normals, edited = _one_normal(
            me, n, _stashed(me, NORMAL_ATTR, n, width=3, field="vector"))
    skin = None
    if "weights" in fields:
        sb, sw = _stashed(me, SKIN_ATTR, n), _stashed(me, WEIGHT_ATTR, n)
        sn = _stashed(me, COUNT_ATTR, n, width=1, field="value")
        skin = list(zip(sw, sb, [int(x[0]) for x in sn])) if sb and sw and sn else None
    out = []
    for i, v in enumerate(me.vertices):
        x = mdl_mod.Vertex()
        x.pos = tuple(v.co)
        # None where Blender holds nothing: a vertex no triangle references has no loops,
        # so it has neither UV nor corner normal, and the file's own bytes must stand.
        x.normal = normals[i]
        x.uv = (uvs[i][0], 1.0 - uvs[i][1]) if uvs[i] else None
        x.weights, x.bones, x.numbones = _one_skin(
            v, obj.vertex_groups, bone_index, skin[i] if skin else None)
        out.append(x)
    return out, edited


def _must_rebuild(obj, model, fields):
    """Whether this object has outgrown the in-place path.

    In place patches fields index for index and is byte-identical wherever nothing moved,
    so it stays the default; the split path rewrites the whole model and is what a changed
    count or a UV seam needs. A seam is a vertex whose corners disagree, which the format
    spells by duplicating the vertex -- so it is a count change wearing another hat.
    """
    me = obj.data
    if len(me.vertices) != model.numvertices:
        return True
    if "uvs" not in fields:
        return False
    uv = me.uv_layers.active
    if uv is None:
        return False
    per = _per_vertex(me, model.numvertices, lambda l: tuple(uv.data[l.index].uv))
    return any(vs and any(max(abs(a - b) for a, b in zip(vs[0], v)) > UV_TOL for v in vs)
               for vs in per)


def rebuild_cell(d, obj, bi, mi, bone_index):
    """One model's geometry replaced from the scene, and its per-mesh triangles.

    Refuses a renumbering the file cannot absorb rather than dropping what it would break:
    `split_mesh` reports `kept` 0 when the original partition could not be recovered, and
    names which of its eight conditions forced that. A flex payload or a cloth binding
    keyed to the old numbering would be carried onto the wrong vertices; with neither of
    those present renumbering costs nothing.
    """
    # Deferred: blender_scratch imports this module, so a top-level import is a cycle.
    from . import blender_scratch as scratch_mod
    runs, unskinned, kept, why = scratch_mod.split_mesh(obj, bone_index, 1.0)
    mr = d.bodyparts[bi].kids[mi]
    if not kept:
        carries = []
        if any(x.kids for x in mr.kids):
            carries.append("morph targets, which are keyed by vertex")
        if mr.extra.get("cloth"):
            carries.append("a cloth binding, which is one entry per vertex per row")
        if carries:
            raise ValueError(
                "%s: the file's own vertex numbering could not be recovered -- %s -- and "
                "this model carries %s" % (obj.name, why, " and ".join(carries)))
    faces = build_mod.replace_model(d, bi, mi, [(None, v, f) for _slot, v, f in runs])
    return faces, unskinned, kept


def read_meshes(m, source, fields):
    """{(bodypart, model): vertices} for every model of `m` the scene supplies, plus the
    models it does not, the fields the file cannot carry, and the objects holding a vertex
    whose fifth bone group no record can hold."""
    found = mesh_objects(m, source)
    bone_index = {b.name: b.index for b in m.bones}
    edits, rebuild, missing, unsupported, renormals = {}, {}, [], set(), 0
    crowded = []
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        ok, no = mesh_mod.supported(mo.filetype, fields)
        obj = found.get((bi, mi))
        if obj is None:
            missing.append(mo.name)
            unsupported |= set(no)
            continue
        over = crowded_vertices(obj, bone_index)
        if over:
            crowded.append((obj.name, over))
        if _must_rebuild(obj, mo, ok):
            # The split writes 44-byte records, so a quantised model gains the weights and
            # normals its own record has no field for; nothing is unsupported there.
            rebuild[(bi, mi)] = obj
            continue
        unsupported |= set(no)
        if ok:
            edits[(bi, mi)], n = read_mesh(obj, mo, bone_index, ok)
            renormals += n
    return edits, rebuild, missing, sorted(unsupported), renormals, crowded


def named_index(anim_names, action):
    """The slot whose animation bears this action's exact name, or None."""
    for i, n in enumerate(anim_names):
        if n == action.name:
            return i
    return None


def stamped_index(anim_names, action):
    """The slot `vtmb_anim_index` points at, or None when it is absent or out of range."""
    i = action.get("vtmb_anim_index")
    return int(i) if i is not None and 0 <= int(i) < len(anim_names) else None


def name_index(anim_names, action):
    """The slot one named action lands on, or None.

    Its own name first, `vtmb_anim_index` second: duplicating an action copies the
    property, so it can point at a slot the action is no longer called after. The dialog
    and the write must both come through here, or one displays a slot and the other
    writes a different one. Only for the paths that carry a single action -- with several
    in play the stamp is not enough on its own, which is what `match_indices` decides.
    """
    i = named_index(anim_names, action)
    return stamped_index(anim_names, action) if i is None else i


def resolve_target(m, action, name):
    """Which animation of the donor this action replaces."""
    names = [a.name for a in m.anims]
    if name:
        if name not in names:
            raise ValueError("%s has no animation named %r" % (m.path, name))
        return names.index(name)
    i = name_index(names, action)
    if i is None:
        raise ValueError("action %r is not named after any animation of %s and carries "
                         "no usable index, so name the target animation explicitly"
                         % (action.name, os.path.basename(m.path)))
    return i


def same_file(a, b):
    try:
        return bool(a) and bool(b) and os.path.samefile(a, b)
    except OSError:
        return bool(a) and bool(b) and \
            os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def armature_actions(arm_obj):
    """Every action the armature currently plays -- its own, and its NLA strips'."""
    ad = getattr(arm_obj, "animation_data", None)
    if ad is None:
        return []
    out = [ad.action] if ad.action else []
    for tr in ad.nla_tracks:
        out += [s.action for s in tr.strips if s.action]
    return out


def match_indices(anim_names, source, arm_obj=None):
    """Which action replaces which animation, as ({index: action}, unwritten, addable).

    The name decides; `vtmb_anim_index` answers only for an action nobody else contests.
    `action.copy()` carries custom properties over, so `run_0.001`, `run_0.002` and
    `run_0.003` all read index 0 and all claim animation 0 -- picking one is picking by
    `bpy.data.actions` order, which the author cannot see, so none is written and the tie
    is reported. `unwritten` is (animation, action, kept) for every candidate left out,
    `kept` being the action that took the slot or None when the animation is left alone.

    `addable` is the same actions the `anim is None` rows name, as objects rather than
    names: an action of this file's that matches no animation of it, and one the armature
    plays that claims nothing at all. Those are exactly the ones an append can take, and
    they are collected here so the rule lives in one place rather than in the dialog.

    Takes names rather than an Mdl so the dialog can call it every redraw off its cached
    list.
    """
    named, stamped, loose = {}, {}, []
    for act in bpy.data.actions:
        i = named_index(anim_names, act)
        if i is not None:
            named[i] = act
            continue
        if not same_file(act.get("vtmb_source"), source):
            continue
        j = stamped_index(anim_names, act)
        if j is None:
            loose.append(act)
        else:
            stamped.setdefault(j, []).append(act)

    found, unwritten = dict(named), []
    for j, acts in sorted(stamped.items()):
        if j in found:
            unwritten += [(anim_names[j], a.name, found[j].name) for a in acts]
        elif len(acts) == 1:
            found[j] = acts[0]
        else:
            unwritten += [(anim_names[j], a.name, None) for a in acts]
    # Came from this file and now points at no animation of it -- a rename, or a stamp
    # that outlived the animation it named.
    addable = list(loose)
    unwritten += [(None, a.name, None) for a in loose]
    # Keying into a fresh action leaves neither a name nor a stamp, so it never became a
    # candidate above and would otherwise be dropped without appearing anywhere at all.
    seen = set(found.values()) | {a for a in loose}
    named_out = {u[1] for u in unwritten}
    for act in armature_actions(arm_obj):
        if act not in seen and act.name not in named_out:
            unwritten.append((None, act.name, None))
            named_out.add(act.name)
            addable.append(act)
    return found, unwritten, addable


def match_actions(m, source, arm_obj=None):
    return match_indices([a.name for a in m.anims], source, arm_obj)


def split_unwritten(unwritten, adding):
    """(rows an append would take, rows nothing takes).

    A row with no animation named it is an action matching nothing, which is the append
    candidate; every other row is a contested slot, which adding cannot help.
    """
    if not adding:
        return [], list(unwritten)
    return ([u for u in unwritten if u[0] is None],
            [u for u in unwritten if u[0] is not None])


def unwritten_line(anim, act, kept):
    """One row of `match_indices`' second return, in the dialog's own arrow idiom."""
    if kept is not None:
        return "%s ← %s (not %s)" % (anim, kept, act)
    if anim is None:
        return "%s → nothing" % act
    return "%s ← nothing (%s and others claim it)" % (anim, act)


def describe_unwritten(unwritten, limit=3):
    head = "; ".join(unwritten_line(*u) for u in unwritten[:limit])
    more = "" if len(unwritten) <= limit else " and %d more" % (len(unwritten) - limit)
    return "%d action%s not written: %s%s" % (
        len(unwritten), "" if len(unwritten) == 1 else "s", head, more)


def read_materials(m, source):
    """{file texture index: name} from the scene's material slots.

    `obj["vtmb_meshes"]` records which slot each file mesh landed in, and it is the only
    thing that maps a material back: two meshes may share one, and Blender's per-face slot
    cannot speak for a mesh whose triangles all sit in a lower LOD.
    """
    out = {}
    for (bi, mi), obj in sorted(mesh_objects(m, source).items()):
        stash = obj.get("vtmb_meshes")
        if not stash:
            continue
        mats = obj.data.materials
        for j, mesh in enumerate(m.bodyparts[bi].models[mi].meshes):
            if j >= len(stash):
                break
            slot = int(stash[j][0])
            if not 0 <= slot < len(mats) or mats[slot] is None:
                continue
            ref = mesh.material
            if m.skins and 0 <= ref < len(m.skins[0]):
                ref = m.skins[0][ref]
            name = mats[slot].name
            if out.get(ref, name) != name:
                raise ValueError("texture %d is named both %r and %r by the scene's "
                                 "materials" % (ref, out[ref], name))
            out[ref] = name
    return out


def apply_sequences(d, arm_obj, anim_names):
    """Label, activity, group size and the blend grid out of `arm_obj["vtmb_sequences"]`.

    Matched by position: the stash is written in file order. A file carrying more than the
    stash does is one an append has just grown, and the sequences past its end are ones the
    scene says nothing about, so they keep what `add_sequence` wrote; a stash longer than
    the file claims sequences that are not there and is refused. Blends are stored as
    animation names because an index means nothing once the file is re-emitted, so they
    resolve back through `anim_names`.
    """
    stash = arm_obj.get("vtmb_sequences")
    if not stash:
        return 0
    if len(stash) > len(d.seqs):
        raise ValueError("the armature carries %d sequences and the file has %d"
                         % (len(stash), len(d.seqs)))
    index = {n: k for k, n in enumerate(anim_names)}
    n = 0
    for rec, s in zip(d.seqs, stash):
        label, activity = s.get("label"), s.get("activity")
        if label and rec.name != label:
            rec.name, n = label, n + 1
        if activity and rec.extra.get("activity") != activity:
            rec.extra["activity"], n = activity, n + 1
        blends = [list(col) for col in s.get("blends") or []]
        gx, gy = len(blends), max((len(c) for c in blends), default=0)
        # The grid sits inside the 764-byte record at a 0x20 stride, so one that would run
        # past it is left alone rather than written over the fields behind it.
        if not gx or mdl_mod.SEQ_ANIM + (gx - 1) * 0x20 + gy * 2 > len(rec.raw):
            continue
        was = struct.unpack_from("<ii", rec.raw, mdl_mod.SEQ_GROUPSIZE)
        struct.pack_into("<ii", rec.raw, mdl_mod.SEQ_GROUPSIZE, gx, gy)
        if was != (gx, gy):
            n += 1
        for x, col in enumerate(blends):
            for y, name in enumerate(col):
                at = mdl_mod.SEQ_ANIM + x * 0x20 + y * 2
                a = index.get(name, -1)
                if a < 0:
                    raise ValueError("sequence %r blends animation %r, which the file does "
                                     "not have" % (rec.name, name))
                if struct.unpack_from("<h", rec.raw, at)[0] != a:
                    n += 1
                struct.pack_into("<h", rec.raw, at, a)
    return n


def verify_writer(src):
    """The writer must reproduce this donor before anything from the scene is fed in.

    Grades the relayout alone, which is what makes the differences that follow attributable
    to the scene: an edit is an intended difference and would drown the check, so the
    comparison is `emit(from_bytes(x))` against `x` and never against the edited output.
    """
    plain = build_mod.emit(build_mod.from_bytes(src))
    bad, _n = rebuild_mod.verify(src, plain)
    return bad


def vtx_path(path, flavour="dx80"):
    """The .vtx beside a .mdl. Only .dx80 is written, which is the flavour
    `Mod_LoadVtxFile_vtmb` asks for first."""
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    return "%s.%s.vtx" % (stem, flavour)


def stale_flavours(dest):
    """Other .vtx flavours beside the file just written, which now disagree with it.

    Only `.dx80.vtx` is emitted, so a `.dx7_2bone.vtx` next to a model whose geometry moved
    still describes the old one. The engine tests only the flavour it loaded and takes
    `.dx80.vtx` first, so this is a warning and not a refusal.
    """
    return [os.path.basename(p) for p in
            (vtx_path(dest, "dx7_2bone"), vtx_path(dest, "dx90"), vtx_path(dest, "sw"))
            if os.path.exists(p)]


def revise_vtx(source, dest, data, revised):
    """Rewrite the .vtx for a model whose geometry moved, and return what it cost.

    The donor's own file supplies every strip group the edit did not touch, so a change to
    one mesh leaves the others byte for byte as the compiler emitted them.
    """
    src = vtx_path(source)
    if not os.path.exists(src):
        raise ValueError("%s has no .dx80.vtx beside it, and a changed vertex or face "
                         "count needs one rewritten -- the engine draws nothing when the "
                         "pair disagrees" % os.path.basename(source))
    blob, st = vtxr_mod.revise(mdl_mod.Mdl("<written>", data=data), src, revised)
    out = vtx_path(dest)
    with open(out, "wb") as f:
        f.write(blob)
    st["path"] = out
    st["bytes"] = len(blob)
    return st


def export_actions(context, arm_obj, source, dest, actions, scale=1.0,
                   frame_start=None, frame_end=None, fps=None, root_motion_in_keys=True,
                   root_motion="keep", mesh_fields=(), verify=True, add=(), drop="",
                   model_name="", hull=None, cdtexture=None):
    """Author `source` again with `actions`, an {animation index: action} map, applied.

    `add` is actions appended as new animations rather than replacing one, each with a
    sequence of the same name -- the engine reaches an animation only through a sequence,
    so one with none is dead weight. Appending renumbers nothing, every stored index
    pointing below the insertion point, and `emit` requantises the animations already in
    the file when the new pose widens the file-wide `mstudiobone_t` scales.

    `drop` is one animation of the file, by name, to remove. It is taken last, so every
    index above is one the caller stated against the file as it was; removing it takes the
    sequences it leaves with nothing to play, which is why the caller has to restamp
    `arm_obj["vtmb_sequences"]` from what came back.

    The file is always rebuilt from its own decoded records -- every count from a `len()`,
    every offset from where its target landed -- so an empty map is meaningful and re-emits
    the model unchanged rather than copying it.

    The skeleton, the material names and the sequence table come from the scene too, and
    unconditionally: each is compared against what the file already says and written only
    where it differs, so an unedited model still comes back byte for byte.
    """
    m = mdl_mod.Mdl(source)
    if verify:
        bad = verify_writer(bytes(m.d))
        if bad:
            raise ValueError("the writer does not reproduce %s, so nothing was written: %s"
                             % (os.path.basename(source), "; ".join(bad)))
    # Before the first reader: `read_poses` and `read_bones` both walk the FILE's bone list
    # and refuse by name on one the armature does not have, so a bone deleted in Blender has
    # to leave the description here or it never reaches the writer.
    d = build_mod.from_bytes(bytes(m.d))
    removed = drop_missing_bones(d, m, arm_obj)
    if removed:
        m = _EditedBones(m, d)
    # After the removal and before the rename: an index is stated against the bone list
    # that will be written, and a bone added here has no stash, so its file name is its
    # Blender name and `renamed_bones` has nothing to say about it.
    added_bones = add_surplus_bones(d, m, arm_obj, scale)
    if added_bones:
        m = _EditedBones(m, d)
    # After the removal, so an index is stated against the bone list that will be written.
    # `m.bones` keeps the file's own names, which is what every reader here resolves
    # through `bone_map`, so renaming the record does not move the scene out from under it.
    renamed = renamed_bones(m, arm_obj)
    for k, name in renamed.items():
        d.bones[k].name = name
    renamed = [(m.bones[k].name, name) for k, name in sorted(renamed.items())]

    # Both names are `char name[128]` inline, so writing either moves nothing.
    remodelled = renamed_models(m, source)
    for (bi, mi), name in sorted(remodelled.items()):
        build_mod.set_model_name(d, bi, mi, name)
    if model_name and model_name != d.name:
        d.name = model_name
    else:
        model_name = ""
    # @180/@192 is the .qc's $bbox, and the per-sequence boxes at +0x1c/+0x28 are what the
    # engine actually culls against -- a zero one draws and then vanishes as the camera
    # turns. Neither follows an edit on its own: a donor keeps the boxes it shipped, which
    # after a geometry change describe where the vertices used to be.
    if hull is not None:
        struct.pack_into("<3f", d.hdr, 180, *hull[0])
        struct.pack_into("<3f", d.hdr, 192, *hull[1])
        d.refit_boxes = True

    # Compared normalised and written only where the lists differ, because normalising is
    # not free: separators and repeats fold on 106 of 4445 models, which resolve the same
    # either way but would stop coming back byte for byte if this rewrote them anyway.
    cdtex = None
    if cdtexture is not None:
        want = paths_mod.cdtexture_list(cdtexture)
        if paths_mod.engine_paths(want) != paths_mod.engine_paths(d.cdtextures):
            cdtex = (list(d.cdtextures), build_mod.set_cdtextures(d, want))

    ad = arm_obj.animation_data
    if actions and ad is None:
        raise ValueError("%s has no animation data" % arm_obj.name)
    restore = ad.action if ad else None

    edits, wrote, pending = {}, [], []
    try:
        for index, action in sorted(actions.items()):
            if ad.action is not action:
                ad.action = action
            anim = m.anims[index]
            lo, hi = action.frame_range
            a = int(round(lo)) if frame_start is None else frame_start
            b = int(round(hi)) if frame_end is None else frame_end
            frames = list(range(a, b + 1))
            if not frames:
                raise ValueError("%s: empty frame range" % action.name)

            # "extract" wants the motion still in the poses, so it is only taken back out
            # when the donor's own blocks are the ones being kept.
            poses = read_poses(context, arm_obj, m, frames, scale, anim,
                               root_motion_in_keys=(root_motion == "keep"
                                                    and root_motion_in_keys
                                                    and bool(action.get("vtmb_root_motion"))))
            movements = None
            if root_motion == "extract":
                movements, poses = write_mod.extract_root_motion(m, poses)
            elif root_motion == "in_place":
                # No blocks and no net travel: the ramp `fit_movements` fits IS the net
                # ground displacement, so subtracting it and throwing the block away is
                # "extract" with nothing for the engine to carry. What the ramp does not
                # account for stays in the keys -- the rise and sway of a run cycle, which
                # an animation played in place should still have.
                _mv, poses = write_mod.extract_root_motion(m, poses)
                movements = []
            elif root_motion == "none":
                # The travel stays on the skeleton, which is what 1344 of 10205 shipped
                # animations do and the largest of the two travel-carrying groups.
                movements = []
            edits[index] = {"poses": poses, "movements": movements,
                            "fps": fps if fps else action.get("vtmb_fps"),
                            "flags": action.get("vtmb_flags")}
            wrote.append((index, anim.name, action.name, len(frames),
                          len(anim.movements if movements is None else movements)))

        for action in add:
            if ad.action is not action:
                ad.action = action
            lo, hi = action.frame_range
            a = int(round(lo)) if frame_start is None else frame_start
            b = int(round(hi)) if frame_end is None else frame_end
            frames = list(range(a, b + 1))
            if not frames:
                raise ValueError("%s: empty frame range" % action.name)
            # No donor animation behind this one, so there is no root motion to take back
            # out and nothing for "keep" to keep: an appended animation either has the
            # blocks fitted here or has none.
            poses = read_poses(context, arm_obj, m, frames, scale)
            movements = ()
            if root_motion == "extract":
                movements, poses = write_mod.extract_root_motion(m, poses)
            pending.append((action, poses, tuple(movements), len(frames)))
    finally:
        if ad is not None and ad.action is not restore:
            ad.action = restore

    d = build_mod.apply_anims(d, edits, source)

    scene = {"bones": 0, "materials": 0, "sequences": 0, "stale": 0,
             "surplus": surplus_bones(m, arm_obj)}
    poses = read_bones(m, arm_obj, scale)
    if poses:
        build_mod.set_bone_poses(d, poses)
        scene["bones"] = len(poses)
        # A rotation channel is `int16 * rotscale` with no bind base, so an animation left
        # un-re-encoded keeps a pose the moved bind disagrees with -- 1.0e-01 against 5.4e-04.
        scene["stale"] = len(d.anims) - len(edits)
    for ref, name in sorted(read_materials(m, source).items()):
        if 0 <= ref < len(d.textures) and d.textures[ref].name != name:
            d.textures[ref].name = name
            scene["materials"] += 1
    # Before the append, not after: apply_sequences refuses outright when the armature's
    # stash and the file disagree on how many sequences there are.
    scene["sequences"] = apply_sequences(d, arm_obj, [r.name for r in d.anims])

    added = []
    for action, poses, movements, nframes in pending:
        rate = float(fps or action.get("vtmb_fps") or context.scene.render.fps)
        i = build_mod.add_animation(d, action.name, poses, rate,
                                    int(action.get("vtmb_flags") or 0), movements)
        build_mod.add_sequence(d, action.name, i, action.get("vtmb_activity"),
                               int(action.get("vtmb_seq_flags") or 0))
        added.append((i, action.name, nframes, len(movements)))

    dropped = {"anim": "", "seqs": []}
    if drop:
        names = [r.name for r in d.anims]
        if drop not in names:
            raise ValueError("%s has no animation %r to remove"
                             % (os.path.basename(source), drop))
        seqs, _left = build_mod.remove_animation(d, names.index(drop))
        dropped = {"anim": drop, "seqs": seqs}

    mesh = {"fields": tuple(mesh_fields), "verts": 0, "models": 0,
            "missing": [], "unsupported": [], "normals": 0, "rebuilt": [],
            "unskinned": 0, "renumbered": 0, "crowded": []}
    revised = {}
    if mesh_fields:
        cells, rebuild, mesh["missing"], mesh["unsupported"], mesh["normals"], \
            mesh["crowded"] = read_meshes(m, source, mesh_fields)
        if not cells and not rebuild and not mesh["missing"]:
            raise ValueError("no scene mesh belongs to %s" % os.path.basename(source))
        for (bi, mi), verts in sorted(cells.items()):
            rec = d.bodyparts[bi].kids[mi]
            rec.extra["verts"], n = mesh_mod.pack_model(
                m.bodyparts[bi].models[mi], verts, mesh_fields, rec.extra["verts"])
            mesh["verts"] += n
        mesh["models"] = len(cells)
        bone_index = {b.name: b.index for b in m.bones}
        for (bi, mi), obj in sorted(rebuild.items()):
            was = m.bodyparts[bi].models[mi].numvertices
            faces, unskinned, kept = rebuild_cell(d, obj, bi, mi, bone_index)
            for k, tris in enumerate(faces):
                revised[(bi, mi, k)] = tris
            now = sum(struct.unpack_from("<i", x.raw, 0x08)[0]
                      for x in d.bodyparts[bi].kids[mi].kids)
            mesh["rebuilt"].append((obj.name, was, now))
            mesh["unskinned"] += unskinned
            mesh["renumbered"] += 0 if kept else 1

    # The donor checksum is kept whether or not the .vtx is rewritten: the pair only
    # has to agree with each other, and the engine draws nothing when it does not.
    data = build_mod.emit(d, checksum=d.checksum)
    # A bone removal or an append moves no vertex and no triangle, so `revised` is empty --
    # and the .vtx still has to be rewritten, because every strip group's bone data is bound
    # to the .mdl's numbering, and a group without flag 0x02 carries the bone id per vertex.
    # `vtx_rebuild.revise` rebinds all of them off the model as written whether or not a cell
    # was named, so an empty face map is the whole of what either needs.
    vtx = (revise_vtx(source, dest, data, revised)
           if (revised or removed or added_bones) else None)
    with open(dest, "wb") as f:
        f.write(data)
    return {"wrote": wrote, "added": added, "dropped": dropped, "bones": len(m.bones),
            "mesh": mesh, "scene": scene, "bytes": len(data), "was": len(m.d),
            "anims": len(m.anims), "sequences": len(d.seqs),
            "vtx": vtx, "removed": removed, "added_bones": added_bones,
            "renamed": renamed,
            "model_name": model_name, "hull": hull, "cdtexture": cdtex,
            "boxes": (d.refit_count, len(d.seqs)) if hull is not None else None,
            "remodelled": sorted((v, k) for k, v in remodelled.items()),
            "includes": [r.name for r in d.includes],
            "stale": stale_flavours(dest) if (revised or removed or added_bones) else []}


def export_action(context, arm_obj, source, dest, anim_name="", **kw):
    """One active action into one animation. Kept because it is what the round-trip and
    self-check scripts drive."""
    ad = arm_obj.animation_data
    action = ad.action if ad else None
    if action is None:
        raise ValueError("%s has no action assigned" % arm_obj.name)
    index = resolve_target(mdl_mod.Mdl(source), action, anim_name)
    r = export_actions(context, arm_obj, source, dest, {index: action}, **kw)
    i, name, _, frames, movements = r["wrote"][0]
    r.update(anim=name, index=i, frames=frames, movements=movements)
    return r
