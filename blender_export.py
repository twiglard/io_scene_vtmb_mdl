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
import re
import struct

import bpy
import mathutils

from . import cloth as cloth_mod
from . import mdl as mdl_mod
from . import mdl_build as build_mod
from . import mdl_rebuild as rebuild_mod
from . import mdl_write as write_mod
from . import mesh_write as mesh_mod
from . import paths as paths_mod
from . import vtx as vtx_mod
from . import vtx_rebuild as vtxr_mod
from . import vtx_write as vtxw_mod


UV_TOL = 1e-6
# Custom split normals are stored as two 16-bit angles, so an untouched mesh's corners
# disagree by up to 9.2e-03; under this a normal is taken as unedited.
NORMAL_EPS = 1.5e-2
# The round trip's own worst error, so between this and NORMAL_EPS a vertex may have been
# edited and cannot be told from storage noise. `_one_normal` counts those and the export
# reports them rather than deciding in silence.
NORMAL_NOISE = 9.2e-3
NORMAL_ATTR = "vtmb_normal"
# Against the import stash an untouched bone compares exactly -- until an edit happens.
# Leaving Edit Mode recomposes every bone from head/tail/roll, and that reaches bones the
# edit never named: over andrei (171 bones) and toreador_male_armor_3 (73), moving a bone
# 3 units and turning one 0.35 rad, the worst deviation on an unreached bone is 1.368e-05
# and the smallest on a reached one is 3.403e-01. It does not track how far the bone sits
# from its parent -- the worst lands at |pos| 1.0 to 9.2 where bones reach 44 -- so an
# absolute bound is the right form. This is 7.3x over the noise and 3400x under the signal.
REST_EPS = 1e-4
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
ROOT_MOTION_MODES = ("keep", "extract", "none", "in_place")
PER_ACTION = "per_action"
MODE_ATTR = "vtmb_root_motion_mode"


def root_motion_mode(action, chosen, donor=True, default="keep"):
    """Which of the four writes this action's travel takes.

    `chosen` is the export dialog's setting and forces every action unless it is
    `per_action`, which reads the action instead. An action with no stamp falls back to
    the count the import stamps beside it -- `keep` where the animation it came from
    carried a movement block, `none` where it did not, which reproduces the file -- and an
    action that was never imported falls back to `default`.

    A stamped `keep` means the movement blocks the file already has, so where there is no
    donor animation behind the action -- the scratch export, and an append -- it becomes
    `extract`: fitting blocks from the keys is the only way to say the engine carries this
    one when there is nothing to keep.
    """
    # A forced mode is taken literally, `keep` included: only a stamp is reinterpreted,
    # so a caller naming one of the four still gets exactly it.
    if chosen != PER_ACTION:
        return chosen
    mode = action.get(MODE_ATTR)
    if mode is None:
        nmv = action.get("vtmb_movements")
        mode = default if nmv is None else ("keep" if nmv else "none")
    mode = str(mode)
    if mode not in ROOT_MOTION_MODES:
        raise ValueError("action %r carries %s %r, which is none of %s"
                         % (action.name, MODE_ATTR, mode, ", ".join(ROOT_MOTION_MODES)))
    return "extract" if (mode == "keep" and not donor) else mode


def lost_travel(action, movements):
    """True where an action asked the engine to carry it and the keys did not travel.

    `fit_movements` returns nothing for a path that does not move, so an action imported
    with Root motion off -- the travel never reached the scene -- writes no block at all.
    Worth a report rather than silence, since the animation it came from had one.
    """
    return not movements and bool(action.get("vtmb_movements"))


def _rows(mat, scale):
    """A mathutils.Matrix as the format's 3x4, with the import scale taken back out."""
    return [[mat[i][0], mat[i][1], mat[i][2], mat[i][3] / scale] for i in range(3)]


def _stash_claims(arm_obj):
    """{the file bone a pose bone says it came from: [pose bone names]}."""
    out = {}
    for pb in arm_obj.pose.bones:
        was = pb.get("vtmb_bone_name")
        if was:
            out.setdefault(str(was), []).append(pb.name)
    return out


def _one_claimant(was, names):
    """The pose bone that owns file bone `was`, or None where nothing in the scene decides.

    Shift+D copies the stash, so two bones can both say they came from one record. The copy
    is the one Blender renamed, which leaves the original still called after the record --
    that is the whole tiebreak, and `_one_model` takes the same one through
    `vtmb_model_label`. Two claimants and neither named after it is a rename or a delete on
    top of a duplicate, where nothing says which of them is the record's own.
    """
    if len(names) == 1:
        return names[0]
    own = [n for n in names if n == was]
    return own[0] if len(own) == 1 else None


def duplicate_bones(m, arm_obj):
    """[(pose bone, the file bone it says it came from)] for the copies that are not it.

    `add_surplus_bones` appends each of them, which is what a Shift+D asks for. Reported
    because nothing else says so and a copy sits on top of the original until it is moved.
    """
    claims = _stash_claims(arm_obj)
    out = []
    for b in m.bones:
        names = claims.get(b.name) or []
        if len(names) < 2:
            continue
        got = _one_claimant(b.name, names)
        out += [(n, b.name) for n in sorted(names) if n != got]
    return out


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

    A record more than one pose bone claims goes to `_one_claimant` rather than to whichever
    came first in `pose.bones`, whose order is the armature's and not the user's.
    """
    dbs = arm_obj.data.bones
    claims = _stash_claims(arm_obj)
    stash = {}
    for b in m.bones:
        names = claims.get(b.name)
        if not names:
            continue
        got = _one_claimant(b.name, names)
        if got is None:
            raise build_mod.Refused(
                "%d bones say they came from %r -- %s -- and none of them is still called "
                "that, so nothing in the scene says which one is the record. Rename one "
                "back to %r, or clear vtmb_bone_name on the copies to have them appended "
                "as new bones instead"
                % (len(names), b.name, ", ".join(repr(n) for n in sorted(names)), b.name))
        stash[b.name] = got
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


def knockback_bones(d, m):
    """{the name the file gave a bone: its index now, or None where the scene deleted it}.

    A knockback record is the only stash entry that names a bone, and it names it by the
    name the file carried at import. `d.bones` already carries the name the export is about
    to write, so resolving through that refuses every renamed bone -- and the knockback bone
    is a label in the panel, so the rename is the only way a scene reaches these records at
    all. `m.bones` is index-aligned with `d.bones` through both `_EditedBones` wraps and
    keeps the file's own names, which is what every other reader here resolves through.

    A deleted bone answers None rather than being absent: `remove_bone` has already rebound
    the record to that bone's parent and reported it, so the file's own index stands.
    """
    if len(m.bones) != len(d.bones):
        raise ValueError(
            "the bone list being written has %d bones and the one carrying the file's own "
            "names has %d, so a stashed bone name resolves to the wrong index"
            % (len(d.bones), len(m.bones)))
    out = {b.name: k for k, b in enumerate(m.bones)}
    for name in getattr(m, "gone", {}):
        out.setdefault(name, None)
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


def reparented(m, k, db, file_of):
    """The file bone index Blender's parent names, or None where it agrees with the file.

    Raises where the edit cannot be written. The engine walks the bone array in order and
    builds each world matrix off its parent's, so a parent has to sit earlier than its
    child -- 0 of 61 925 bones over the 4445 shipped models break that -- and nothing here
    renumbers the array.

    A deletion is a reparent too, and reaches this in agreement rather than as an edit:
    `drop_missing_bones` rebinds the children in `d` and `_EditedBones` puts that chain
    back in `m` before the first reader runs.
    """
    b = m.bones[k]
    name = db.parent.name if db.parent is not None else None
    now = -1 if name is None else file_of.get(name)
    if now is None:
        raise ValueError(
            "bone %r is parented to %r, which the file has no bone for -- parent it to "
            "one the file carries, or let the export add %r first"
            % (b.name, name, name))
    if now == b.parent:
        return None
    if now >= k:
        was = m.bones[b.parent].name if b.parent >= 0 else "nothing"
        raise ValueError(
            "bone %r was reparented from %s to %r, which sits at index %d against its own "
            "%d -- the engine builds each bone's world matrix off its parent's while "
            "walking the array in order, so a parent must come first, and nothing here "
            "renumbers the bone array" % (b.name, was, m.bones[now].name, now, k))
    return now


def read_bones(m, arm_obj, scale, blind=None):
    """{index: (pos, quat, flags, parent)} for the bones of `m` the armature actually moved.

    A bone still sitting on its imported pose is left out, so an unedited export does not
    touch the bone array and comes back byte for byte -- the same reason the mesh puts an
    unmoved vertex back with its file bytes rather than Blender's.

    `parent` is None on all but a reparented bone. The local matrix is taken against the
    chain the OUTPUT will have -- the file's where Blender agrees with it, Blender's where
    it does not -- so the record and the parent it names stay consistent with each other.

    `blind` is filled with the bones whose only baseline is the file's own record, where
    the comparison is one-sided -- see `moved_from_file`.
    """
    dbs = arm_obj.data.bones
    bmap = bone_map(m, arm_obj)
    missing = [b.name for b in m.bones if b.name not in bmap]
    if missing:
        raise ValueError("armature has no bone %s (and %d more)"
                         % (missing[0], len(missing) - 1))
    file_of = {bmap[b.name]: k for k, b in enumerate(m.bones)}
    out = {}
    for k, b in enumerate(m.bones):
        parent = reparented(m, k, dbs[bmap[b.name]], file_of)
        pk = b.parent if parent is None else parent
        local = dbs[bmap[b.name]].matrix_local
        if pk >= 0:
            local = dbs[bmap[m.bones[pk].name]].matrix_local.inverted() @ local
        pos, quat = to_file(local, scale)
        flags = bone_flags(arm_obj, bmap[b.name])
        # A quaternion and its negation are one rotation, so the nearer sign wins or a bone
        # the importer flipped reads as moved.
        if sum(x * y for x, y in zip(quat, b.quat)) < 0:
            quat = tuple(-x for x in quat)
        base = rest_baseline(arm_obj, bmap[b.name])
        if base is None:
            moved = moved_from_file(m, b, pos, quat)
            if (blind is not None and not moved
                    and (pos != tuple(b.pos) or quat != tuple(b.quat))):
                blind.append(b.name)
        else:
            moved = any(abs(local[i][j] - base[i][j]) > REST_EPS
                        for i in range(4) for j in range(4))
        if not moved and parent is None and (flags is None or flags == b.flags):
            continue
        if not moved:
            # A flags-only record. Blender's own pos and quat are the file's to within the
            # bound above, and writing them anyway hands `rebase_carried` a bone that reads
            # as moved under its exact `!=`, which re-quantises every animation in the file
            # for storage noise. A reparent keeps Blender's, since the two parents give
            # genuinely different parent-relative binds.
            pos, quat = (pos, quat) if parent is not None else (b.pos, b.quat)
        out[k] = (pos, quat, flags, parent)
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
        # {removed bone: its parent}, both file names, carried across a second wrap.
        have = {b.name for b in self.bones}
        self.gone = dict(getattr(m, "gone", {}))
        for b in m.bones:
            if b.name not in have:
                self.gone[b.name] = m.bones[b.parent].name if b.parent >= 0 else None

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


def _model_claims(m, source):
    """{(bodypart index, model index): [objects]} -- every scene mesh saying it is that one.

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
            out.setdefault(key, []).append(obj)
    return out


def _one_model(objs):
    """The object that owns the model, or None where nothing in the scene decides.

    `_one_claimant` for meshes: Shift+D copies the stamps and renames the mesh datablock,
    so the original is the one whose datablock is still called what `vtmb_model_label`
    says -- the comparison `renamed_models` already makes. Alt+D shares the datablock, so
    both objects match it and neither wins.
    """
    if len(objs) == 1:
        return objs[0]
    own = [o for o in objs if o.data.get("vtmb_model_label") == o.data.name]
    return own[0] if len(own) == 1 else None


def duplicate_models(m, source):
    """(copies not written, models no one object owns), out of the objects claiming one.

    A copy is not written at all: the format holds one mesh per model and the donor path
    has no way to add one, so the file keeps what the object it was copied from says.
    """
    dups, clash = [], []
    for key, objs in sorted(_model_claims(m, source).items()):
        if len(objs) < 2:
            continue
        label = m.bodyparts[key[0]].models[key[1]].name
        got = _one_model(objs)
        names = sorted(o.name for o in objs)
        if got is None:
            clash.append((label, names))
        else:
            dups += [(n, label) for n in names if n != got.name]
    return dups, clash


def mesh_objects(m, source):
    """{(bodypart index, model index): object} for the scene meshes belonging to `m`.

    A model more than one object claims is left out rather than settled by `bpy.data`
    order: `duplicate_models` is what names it and `export_actions` refuses on it. This
    does not raise, the export dialog drawing through it (`__init__._mesh_status`).
    """
    out = {}
    for key, objs in _model_claims(m, source).items():
        obj = _one_model(objs)
        if obj is not None:
            out[key] = obj
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


def _per_vertex(me, count, get, skip=None):
    """Every distinct per-loop value each vertex carries, as a list per vertex."""
    out = [[] for _ in range(count)]
    for loop in me.loops:
        if skip and loop.index in skip:
            continue
        out[loop.vertex_index].append(get(loop))
    return out


def _dead_loops(me):
    """Loops on a face with no area, whose corner normal is not a reading of anything.

    Blender builds each corner's normal space from the face normal, and a face whose
    corners coincide has none -- it falls back to (0, 0, 1), and the custom normal decoded
    against that space comes back with every component perpendicular to it gone and its
    length short of 1. Measured on `heather_3.mdl`: a face with all three corners at
    (23.32, 1.721, 45.727) turns (0.07721, 0.10293, -0.99169) into (0, 0, -0.99168).
    """
    out = set()
    for p in me.polygons:
        if p.area == 0.0:
            out.update(p.loop_indices)
    return out


def uv_layer_of(obj):
    """(the layer holding the file's UVs, the layers ignored, the import's layer if it is
    no longer the first).

    The format stores one UV per vertex, so a second layer is a Blender-side working set
    and cannot reach the file. The first layer is the file's: the import made it on a fresh
    mesh and Blender has no operator that reorders UV layers, only ones that append and
    delete. The active layer is not it -- `mesh.uv_texture_add`, which the + button in the
    UV Maps panel runs, makes the layer it adds active.

    `vtmb_uv_layer` is not resolved through, only compared: the same operator names its
    layer `UVMap` too, so once the import's layer has been renamed the stamp matches the
    new one instead. It answers whether the import's layer is still the first, and a
    mismatch is a rename or a delete either way.
    """
    me = obj.data
    layers = list(me.uv_layers)
    if not layers:
        return None, [], None
    use = layers[0]
    want = obj.get("vtmb_uv_layer")
    lost = None if want is None or str(want) == use.name else str(want)
    return use, [x.name for x in layers[1:]], lost


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
    """One normal per vertex, how many came from Blender rather than `stash`, and how many
    sat in the band where the two cannot be told apart.

    Averaging the corners is only ever a fallback: the round trip through custom split
    normals is lossy, so a vertex still within NORMAL_EPS of the normal the importer
    stashed is written back exactly instead of being degraded by re-export.

    An exactly-zero stash is kept unconditionally.  3021 vertices over 71 corpus files
    carry one, Blender hands back (0, 0, 1) for them, and no average can be within
    NORMAL_EPS of zero, so the test above would replace every one.

    Corners on a face with no area are left out of the average, and a vertex that has no
    other corner keeps its stash.  `_dead_loops` is what they are; five of `heather_3`'s
    8388 vertices sit on nothing else, and averaging what Blender hands back for them
    writes a normal 7.392 deg from the file's.
    """
    def corner(l):
        return tuple(me.corner_normals[l.index].vector)

    dead = _dead_loops(me)
    per = _per_vertex(me, count, corner, skip=dead)
    every = _per_vertex(me, count, corner) if dead else per
    out, edited, blind = [], 0, 0
    for i, vs in enumerate(per):
        if not vs and stash:
            out.append(stash[i])
            continue
        # With no stash there is nothing better to hand back than what Blender says, bad
        # normal space and all: a scratch model has no file normal to keep.
        vs = vs or every[i]
        if not vs:
            out.append(None)
            continue
        avg = [sum(v[c] for v in vs) / len(vs) for c in range(3)]
        n = math.sqrt(sum(x * x for x in avg)) or 1.0
        avg = tuple(x / n for x in avg)
        dev = max(abs(a - b) for a, b in zip(avg, stash[i])) if stash else None
        if stash and (stash[i] == (0.0, 0.0, 0.0) or dev <= NORMAL_EPS):
            out.append(stash[i])
            if stash[i] != (0.0, 0.0, 0.0) and dev > NORMAL_NOISE:
                blind += 1
        else:
            out.append(avg)
            edited += 1
    return out, edited, blind


def _stashed(me, name, count, width=4, field="color"):
    """An attribute the importer wrote, as one tuple per vertex."""
    att = me.attributes.get(name)
    if att is None or len(att.data) != count:
        return None
    flat = [0.0] * (count * width)
    att.data.foreach_get(field, flat)
    return [tuple(flat[i * width:i * width + width]) for i in range(count)]


def _skinning(weights, bones):
    """What a record would skin by: its four stored weights against their bones, as a
    multiset, with the unweighted slots left out.

    Four, not the three `u8_weights` returns -- the fourth is the shortfall and pairing
    three bytes against four bones drops slot 3 from the comparison entirely. Unweighted
    slots are left out because their bone index is whatever studiomdl left there and no
    reconstruction can produce it.
    """
    q = mesh_mod.u8_weights(weights)
    q.append(255 - sum(q))
    return sorted((x, bone) for x, bone in zip(q, bones) if x)


def _one_skin(v, groups, bone_index, stash):
    """(weights, bones, count, whether a binding was lost) for one vertex, keeping the
    file's slot order where it still describes the same skinning -- the order is not
    derivable, so losing it would move bytes on a rewrite that changed nothing.

    A vertex the file binds to nothing keeps its stash outright. 5716 shipped vertices
    over 13 models have numbones 0 and weight bytes (255, 0, 0); the importer makes no
    vertex group for one, so an empty reconstruction is what an untouched vertex looks
    like and there is no edit to honour.

    A vertex no bone drives on a file that bound it to one is an edit, and it goes to bone
    0 at full weight: the format has no model space -- the skinning block reads bone[0]
    and transforms by it before it reads the count at all, StudioRender 0x2c0172b9 -- so
    numbones 0 does not mean "follows nothing", and `split_mesh` gives such a vertex
    `[(0, 1.0)]` already. Writing that here is what makes one scene state produce one
    record whichever path takes it, and the count is reported rather than left to be
    found in game.
    """
    pairs = sorted(((g.weight, -bone_index[groups[g.group].name])
                    for g in v.groups if groups[g.group].name in bone_index),
                   reverse=True)[:4]
    w = [x for x, _ in pairs] + [0.0] * (4 - len(pairs))
    b = [-x for _, x in pairs] + [0] * (4 - len(pairs))
    n = sum(1 for x in w if x > 0.0)
    if stash is not None:
        sw, sb, sn = stash
        sb = [int(x) for x in sb]
        if (not n and not sn) or _skinning(sw, sb) == _skinning(w, b):
            return list(sw), sb, sn, False
    if not n:
        return [1.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], 1, True
    return w, b, n, False


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


def claimed_groups(obj):
    """The vertex groups something on the object records a use of its own for.

    A vertex group is Blender's general per-vertex weight and skinning is one use of it:
    the import makes one for the cloth pin set, a modifier names one to limit itself to,
    a shape key names one to scale its range by. None of those is a binding, so naming
    them would make `stray_groups` noise instead of a finding.
    """
    from . import blender_scratch as scratch_mod
    out = {str(obj.get("vtmb_cloth_pin_group") or scratch_mod.PIN_GROUP)}
    for mod in obj.modifiers:
        out.add(getattr(mod, "vertex_group", "") or "")
    keys = getattr(obj.data, "shape_keys", None)
    for key in (keys.key_blocks if keys else ()):
        out.add(key.vertex_group or "")
    out.discard("")
    return out


def stray_groups(obj, bone_index):
    """The object's vertex groups that hold a vertex at a weight above zero and name no
    bone of the file.

    Everything that reads a skin filters on `name in bone_index` -- `_one_skin`,
    `crowded_vertices` and `split_mesh` alike -- so a bone name typed wrong, or a group
    left behind by a bone the scene deleted, takes every vertex in it out of the skinning
    and says nothing. It does not come out as an obviously broken file either: a vertex
    left with no binding goes to bone 0 at full weight, so the geometry ships drawn by the
    wrong bone.

    Named and not refused, because nothing here tells a typo from a group somebody keeps
    on purpose; `claimed_groups` covers the purposes the scene itself records. A group
    with no weight above zero drives nothing and is left out -- that is an empty group and
    it costs the file nothing.
    """
    groups = obj.vertex_groups
    if not groups:
        return []
    claimed = claimed_groups(obj)
    live = set()
    for v in obj.data.vertices:
        for g in v.groups:
            if g.weight > 0.0:
                live.add(g.group)
    return sorted(groups[i].name for i in live
                  if groups[i].name not in bone_index
                  and groups[i].name not in claimed)


def read_mesh(obj, model, bone_index, fields):
    """One `mdl.Vertex` per file vertex, in file order, inverting what the importer did,
    plus the normals taken from Blender, the ones in the band where the two cannot be told
    apart, and the vertices the scene has taken out of every group.

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
    edited = blind = unskinned = 0
    if "uvs" in fields:
        uv_layer = uv_layer_of(obj)[0]
        if uv_layer is None:
            raise ValueError("%s has no UV layer" % obj.name)
        uvs = _one_uv(me, n, uv_layer)
    if "normals" in fields:
        normals, edited, blind = _one_normal(
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
        x.weights, x.bones, x.numbones, lost = _one_skin(
            v, obj.vertex_groups, bone_index, skin[i] if skin else None)
        unskinned += lost
        out.append(x)
    return out, edited, blind, unskinned


def _canon_faces(tris):
    """Triangles counted as rotations starting at the lowest corner.

    Rotation-blind so the two spellings of one face compare equal, winding-sensitive so a
    reversed one does not, and a Counter rather than a set because 11 of 4463 shipped
    models carry a repeated triangle -- 25 extra copies, `chinabldgroof` 4 and `tunabp2b`
    5 -- and Blender keeps every one of them, so multiplicity is part of what the donor
    said.
    """
    import collections
    return collections.Counter(vtxr_mod._canon(t) for t in tris)


def _donor_faces(source, m):
    """{(bodypart, model): the donor's own LOD-0 triangles}, in the file's winding.

    Triangles live in the .vtx alone -- `mstudiomesh_t` carries no count -- so a face
    added, deleted or reversed reaches the file by no other route and the .mdl offers
    nothing to compare on. Corners come out model-local, which is what a Blender object's
    vertices are indexed by, by the same mapping `blender_import` reads them in through.

    Empty where no .vtx sits beside the source, which leaves the caller comparing what it
    compared before rather than rebuilding every model on a missing file.
    """
    path = vtx_path(source)
    if not source or not os.path.exists(path):
        return {}
    reader = vtx_mod.Vtx(path)
    models = mesh_mod.models_of(m)
    out, per = {}, {}
    for sg in reader.groups:
        if sg.lod != 0 or sg.model >= len(models):
            continue
        mo = models[sg.model][3]
        if sg.mesh >= len(mo.meshes):
            continue
        base = mo.meshes[sg.mesh].vertexoffset
        ids = reader.orig_vert_ids(sg)
        per.setdefault(sg.model, []).extend(
            tuple(ids[x] + base for x in t) for t in reader.triangles(sg))
    for gi, tris in per.items():
        bi, mi = models[gi][0], models[gi][1]
        out[(bi, mi)] = _canon_faces(tris)
    return out


def _scene_faces(me):
    """The scene's triangles in the file's own winding, model-local.

    `wound` reverses in both directions -- the format winds a triangle against its outward
    normal and Blender winds it with -- so the reverse here is what puts a Blender face
    back in the donor's terms.
    """
    me.calc_loop_triangles()
    return _canon_faces(tuple(tri.vertices)[::-1] for tri in me.loop_triangles)


def _must_rebuild(obj, model, fields, donor_faces=None):
    """Whether this object has outgrown the in-place path.

    In place patches fields index for index; the split path rewrites the whole model and
    is what a changed vertex count, a UV seam or a changed face needs. A seam is a vertex
    whose corners disagree, which the format spells by duplicating the vertex -- so it is
    a count change wearing another hat.

    `donor_faces` is what the .vtx said, and None where no .vtx was found. Comparing it is
    what stops a face deleted, added between existing vertices, or flipped from writing a
    file identical to the one it was edited from: none of the three moves the vertex count
    or a UV, so every other test here passes on an edit the file never took.
    """
    me = obj.data
    if len(me.vertices) != model.numvertices:
        return True
    if donor_faces is not None and _scene_faces(me) != donor_faces:
        return True
    if "uvs" not in fields:
        return False
    uv = uv_layer_of(obj)[0]
    if uv is None:
        return False
    per = _per_vertex(me, model.numvertices, lambda l: tuple(uv.data[l.index].uv))
    return any(vs and any(max(abs(a - b) for a, b in zip(vs[0], v)) > UV_TOL for v in vs)
               for vs in per)


def rebuild_cell(d, obj, bi, mi, bone_index, fields=("uvs", "normals"),
                 donor_tris=None):
    """One model's geometry replaced from the scene, its per-mesh triangles and its edits.

    Refuses a renumbering the file cannot absorb rather than dropping what it would break:
    `split_mesh` reports `kept` 0 when the original partition could not be recovered, and
    names which of its ten conditions forced that. A flex payload or a cloth binding keyed
    to the old numbering would be carried onto the wrong vertices; with neither of those
    present renumbering costs nothing.

    A delete is not that case. The survivors keep the file's own order, so `split_mesh`
    reports the donor vertex each written one came from and the flex payloads are
    renumbered through it -- a record naming the deleted vertex is dropped and the rest
    keep the vertex they named. A cloth binding still refuses: `mstudiocloth_t.vertindex`
    names a model vertex that has to exist, so a particle whose anchor has gone has no
    entry to emit.

    `fields` is what the checkboxes asked for. Without it a face added or a winding
    reversed changes Blender's corner normal on every vertex it touches, and the split
    would take that as an edit and write it -- on an export that never asked for normals.
    """
    # Deferred: blender_scratch imports this module, so a top-level import is a cycle.
    from . import blender_scratch as scratch_mod
    carry = []
    runs, unskinned, kept, why, edits, _origins = scratch_mod.split_mesh(
        obj, bone_index, 1.0, fields, carry)
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
    cl = mr.extra.get("cloth") if edits["deleted"] else None
    bound = [] if not cl else [
        k for k, cmap in enumerate(carry)
        if k in cl["meshes"] and cmap is not None
        and len(cmap) - sum(1 for o in cmap if o is None)
        < struct.unpack_from("<i", mr.kids[k].raw, 0x08)[0]]
    if bound:
        raise ValueError(
            "%s: mesh %s lost a vertex and binds this model's cloth. "
            "mstudiocloth_t.vertindex names a model vertex that has to exist, so a "
            "particle whose anchor has gone has no entry to emit, and dropping the "
            "particle renumbers the spring array"
            % (obj.name, ", ".join(str(k) for k in bound)))
    faces, edits["tangents"], flex = build_mod.replace_model(
        d, bi, mi, [(None, v, f) for _slot, v, f in runs], donor_tris=donor_tris,
        carry=carry if kept else None)
    edits["flex_dropped"], edits["flex_emptied"] = flex
    return faces, unskinned, kept, edits


def _vertex_normals(me, coords):
    """Per-vertex normals off `coords`, area-weighted over the mesh's own triangles.

    The file's own normals are not the base here: what a flex record adds is the CHANGE
    a morph makes, so both sides have to be computed the same way for the difference to
    mean anything.
    """
    acc = [[0.0, 0.0, 0.0] for _ in range(len(me.vertices))]
    for tri in me.loop_triangles:
        a, b, c = (coords[i] for i in tri.vertices)
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        # Not normalised: the cross product's length is twice the triangle's area, which
        # is the weight studiomdl and Blender both use.
        n = (u[1] * v[2] - u[2] * v[1],
             u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        for i in tri.vertices:
            acc[i][0] += n[0]
            acc[i][1] += n[1]
            acc[i][2] += n[2]
    out = []
    for n in acc:
        L = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
        out.append((n[0] / L, n[1] / L, n[2] / L) if L else (0.0, 0.0, 0.0))
    return out


# A record whose two magnitude bytes both round to zero says nothing and still costs 8
# bytes, so the step each byte spells is what decides whether a vertex is in the flex at
# all: 8.0/255 = 0.0314 units of position, 2.0/255 = 0.00784 of normal.
FLEX_POS_STEP = build_mod.M.FLEX_POSITION_DELTA_SCALE / 255.0
FLEX_NRM_STEP = build_mod.M.FLEX_NORMAL_DELTA_SCALE / 255.0


def shape_key_flexes(d, obj, bi, mi, scale=1.0):
    """Rebuild one model's flexes from its shape keys. (keys, records, skipped, why).

    The vertanim key is mesh-local and the reader adds it to the mesh's own first vertex,
    so the file's partition has to be recoverable before a key can be written at all --
    `original_runs` is what recovers it, and its members are in file order, member k of
    run r being mesh r's vertex k. A vertex the edit added is in no run and its delta is
    counted as skipped rather than guessed at.

    The file's own flexes go first. A shape key holds coordinates only, so what is
    written is the scene's morph entire and not a patch on what the file said -- the
    normal delta is recomputed from the morphed geometry, and the file's own is gone.
    """
    from . import blender_scratch as scratch_mod
    me = obj.data
    keys = me.shape_keys
    if keys is None or len(keys.key_blocks) < 2:
        return 0, 0, 0, None
    runs = scratch_mod.original_runs(obj, me)
    if runs is None:
        why = []
        scratch_mod.original_runs(obj, me, why)
        return 0, 0, 0, (why[0] if why else "the file's partition was not recoverable")
    me.calc_loop_triangles()
    local = {}
    for k, (_slot, members) in enumerate(runs):
        for j, vi in enumerate(members):
            local[vi] = (k, j)

    blocks = list(keys.key_blocks)
    base = [tuple(v.co) for v in blocks[0].data]
    base_n = _vertex_normals(me, base)
    build_mod.clear_flexes(d, bi, mi)
    targets = obj.get("vtmb_flex_targets") or []
    names = list(obj.get("vtmb_flex_names") or [])
    nkeys = nrec = skipped = 0
    for bk, kb in enumerate(blocks[1:]):
        co = [tuple(v.co) for v in kb.data]
        moved = [i for i in range(len(co))
                 if max(abs(co[i][c] - base[i][c]) for c in range(3)) > 1e-6]
        if not moved:
            continue
        morph_n = _vertex_normals(me, co)
        per_mesh = {}
        for vi in moved:
            delta = tuple((co[vi][c] - base[vi][c]) / scale for c in range(3))
            ndelta = tuple(morph_n[vi][c] - base_n[vi][c] for c in range(3))
            if (sum(c * c for c in delta) ** 0.5 < FLEX_POS_STEP / 2.0
                    and sum(c * c for c in ndelta) ** 0.5 < FLEX_NRM_STEP / 2.0):
                continue
            at = local.get(vi)
            if at is None:
                skipped += 1
                continue
            k, j = at
            per_mesh.setdefault(k, []).append((j, delta, ndelta))
        if not per_mesh:
            continue
        # The target the file carried, matched by the name the import assigned rather
        # than by position: a key renamed or reordered in Blender must not take another
        # flex's numbers.
        t = (0.0, 0.0, 0.0, 0.0)
        if kb.name in names:
            at = names.index(kb.name)
            if at < len(targets):
                t = tuple(float(c) for c in targets[at])
        nkeys += 1
        for k in sorted(per_mesh):
            recs = sorted(per_mesh[k])
            build_mod.add_flex(d, bi, mi, k, kb.name, recs, t)
            nrec += len(recs)
    return nkeys, nrec, skipped, None


# PR #2 corrected the packed position on 2026-08-20. A `.blend` saved by an import older
# than that holds every filetype-2 model 255x oversized, and `vtmb_orig_co` is wrong the same
# way, so the export's own before/after comparison cannot see it. Filetypes 0 and 1 decode
# the same today as they did then.
STALE_DECODE = {2: (0, 2, 0)}


def stale_stash(m, source):
    """[(object, filetype, stamped version or None)] for scene objects an old import made.

    An unstamped object counts as stale: every `.blend` that exists today is unstamped, and
    the versions that wrote no stamp are exactly the ones the decode fix post-dates.
    """
    found = mesh_objects(m, source)
    out = []
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        want = STALE_DECODE.get(mo.filetype)
        obj = found.get((bi, mi))
        if want is None or obj is None:
            continue
        got = obj.get("vtmb_addon_version")
        got = tuple(int(x) for x in got) if got is not None else None
        if got is None or got < want:
            out.append((obj.name, mo.filetype, got))
    return out


# Four stamps landed after 0.2.49 was tagged and `bl_info` has not moved since, so
# `vtmb_addon_version` cannot tell a `.blend` written before one from a scene imported today:
# both stamp 0.2.49. What a stamp's ABSENCE does say is that the import predates it, which is
# the test here, and it needs no version. The cloth pin group is the one of the four absence
# cannot reach -- `vtmb_pinned` existed before its meaning changed on 2026-09-12 -- and
# separating those two needs the version bump.
STALE_STAMPS = (
    ("vtmb_slot_mats", "object",
     "which material sat in each slot, so a slot reordered or deleted since the import is "
     "resolved by index and a rename can reach the wrong texture record"),
    ("vtmb_uv_layer", "object",
     "which UV layer is the file's, so a renamed first layer cannot be told from a deleted "
     "one and neither is reported"),
    ("vtmb_model_label", "mesh",
     "which object owns the model, so two objects claiming one are refused rather than "
     "resolved to the one still named after it"),
)


def stale_stamps(m, source):
    """[(object, [(stamp, what it costs)])] for scene objects an import older than a stamp made.

    Reported and never refused: every one of the three falls back to the behaviour that stood
    before its stamp, so the export is correct for a scene that made no such edit and wrong
    only for one that did, which is a thing to say rather than a thing to stop.
    """
    out, seen = [], set()
    for obj in mesh_objects(m, source).values():
        if obj.name in seen:
            continue
        seen.add(obj.name)
        miss = [(k, why) for k, dom, why in STALE_STAMPS
                if (obj if dom == "object" else obj.data).get(k) is None]
        if miss:
            out.append((obj.name, miss))
    return out


def read_meshes(m, source, fields):
    """{(bodypart, model): vertices} for every model of `m` the scene supplies, plus the
    models it does not, the fields the file cannot carry, the objects holding a vertex whose
    fifth bone group no record can hold, the vertices whose normal moved by more than the
    round trip's own error and less than `NORMAL_EPS`, which are written from the stash and
    so lose the edit, the UV layers no model wrote, the donor's own triangles, the
    vertices no bone drives any more, which go to bone 0 at full weight, and the vertex
    groups that skin something and name no bone."""
    found = mesh_objects(m, source)
    bone_index = {b.name: b.index for b in m.bones}
    donor_faces = _donor_faces(source, m)
    edits, rebuild, missing, unsupported, renormals = {}, {}, [], set(), 0
    blind_normals = unskinned = 0
    crowded, uv_spare, stray = [], [], []
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        ok, no = mesh_mod.supported(mo.filetype, fields)
        obj = found.get((bi, mi))
        if obj is None:
            missing.append(mo.name)
            unsupported |= set(no)
            continue
        # Before the rebuild branch below, which would report nothing for a model that
        # took the split path.
        used, spare, lost = uv_layer_of(obj)
        if spare or lost:
            uv_spare.append((obj.name, used.name, spare, lost))
        over = crowded_vertices(obj, bone_index)
        if over:
            crowded.append((obj.name, over))
        odd = stray_groups(obj, bone_index)
        if odd:
            stray.append((obj.name, odd))
        if _must_rebuild(obj, mo, ok, donor_faces.get((bi, mi))):
            # The split writes 44-byte records, so a quantised model gains the weights and
            # normals its own record has no field for; nothing is unsupported there.
            rebuild[(bi, mi)] = obj
            continue
        unsupported |= set(no)
        if ok:
            edits[(bi, mi)], n, nb, nu = read_mesh(obj, mo, bone_index, ok)
            renormals += n
            blind_normals += nb
            unskinned += nu
    return (edits, rebuild, missing, sorted(unsupported), renormals, crowded,
            blind_normals, uv_spare, donor_faces, unskinned, stray)


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


def renamed_anims(anim_names, source, arm_obj=None):
    """({index: the name Blender now gives it}, the candidates no stamp can confirm).

    The index comes off `match_indices`, which is already what decides which action writes
    which animation, so this adds no second rule -- it reads that map and reports the slots
    whose action is no longer called after the record.  An action that took its slot by
    name renames nothing by construction.

    `vtmb_anim_name` is what makes a rename sayable, and the file's own name at the slot is
    not enough: Blender deduplicates, so the two animations called `walk` that 3 of the
    4445 shipped models carry import as `walk` and `walk.001`, and the second differs from
    the record's name on a plain re-export that renamed nothing.  The stamp also has to
    still agree with the file, or it describes an animation some earlier export already
    renamed and the scene is not the authority any more.

    An action from an import older than the stamp reaches the second return instead, so a
    scene that cannot express a rename says so rather than taking one silently either way.
    """
    found = match_indices(anim_names, source, arm_obj)[0]
    out, unstamped = {}, []
    for i, act in sorted(found.items()):
        if act.name == anim_names[i]:
            continue
        was = act.get("vtmb_anim_name")
        if was is None:
            unstamped.append((anim_names[i], act.name))
        elif str(was) == anim_names[i]:
            out[i] = act.name
    return out, unstamped


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


def skin_family(m, arm_obj):
    """Which row of the skin table the scene's material slots are showing.

    Out of range is refused, not clamped: the engine clamps to row 0 silently, and a
    rename read back through the wrong row renames every mstudiotexture_t in the file.
    """
    fam = int(arm_obj.get("vtmb_skin_family") or 0)
    if m.skins and not 0 <= fam < len(m.skins):
        raise ValueError("the armature's skin family is %d, but %s carries %d"
                         % (fam, os.path.basename(m.path), len(m.skins)))
    return fam


def slot_moves(obj):
    """({slot the import stamped: the slot it is in now}, [names no slot holds any more]).

    A material slot index is the addon's only handle on which material speaks for which
    file mesh, and Blender renumbers slots on a reorder or a delete, so the stamped index
    names a different material afterwards and every texture record downstream is renamed
    to whatever moved into its place. 1686 of the corpus's 4567 models carry two or more
    slots, so that is 38% of it. `obj["vtmb_slot_mats"]` is what the import put in each
    slot, and finding a stamped name at another index is what says the slots moved.

    A name no slot holds is read two ways, and the slot count is what separates them.
    Blender appends an added slot, so the count only falls when a slot was deleted: while
    it holds or grows, an absent name is a rename and the slots no stamped name claimed
    are what it was renamed to, its own index first and the rest in index order -- one
    rename alone is exact, and with no reorder every rename is. Once the count has fallen,
    a deletion and a rename are the same evidence and neither is guessed at, so the
    unmatched names are reported and a rename made in the same edit is lost with them.

    None rather than a map where the object carries no stamp -- a `.blend` saved before
    2026-09-09 -- since the index is then all there is and reading every slot as deleted
    would be worse than the defect.
    """
    stamp = [str(x) for x in (obj.get("vtmb_slot_mats") or ())]
    if not stamp:
        return None, []
    mats = obj.data.materials
    here = {}
    for j, mm in enumerate(mats):
        if mm is not None:
            here.setdefault(mm.name, j)
    out = {slot: here.get(name) for slot, name in enumerate(stamp)}
    left = [slot for slot in sorted(out) if out[slot] is None]
    idle = [j for j in range(len(mats))
            if mats[j] is not None and j not in set(out.values())]
    gone = []
    if len(mats) < len(stamp):
        gone = [stamp[slot] for slot in left]
    else:
        for slot in left:
            if slot in idle:
                idle.remove(slot)
                out[slot] = slot
        for slot in left:
            if out[slot] is not None:
                continue
            if idle:
                out[slot] = idle.pop(0)
            else:
                gone.append(stamp[slot])
    return out, gone


def slot_of(stash, moves, j):
    """The slot file mesh `j` speaks through now, or None if there is no longer one."""
    if j >= len(stash):
        return None
    slot = int(stash[j][0])
    if moves is None:
        return slot
    return moves.get(slot)


def read_materials(m, source, family=0):
    """({file texture index: name}, slots reordered, [slots deleted]).

    `obj["vtmb_meshes"]` records which slot each file mesh landed in, and it is the only
    thing that maps a material back: Blender's per-face slot cannot speak for a mesh whose
    triangles all sit in a lower LOD, and nothing stops two meshes sharing one material --
    though 0 of the corpus's 4567 models do. That index is resolved through `slot_moves`
    rather than used directly.

    A deleted slot leaves the file's own name standing, because every mesh names an
    `mstudiotexture_t` and there is no way to spell "no material" in the format.
    """
    out, moved, gone = {}, 0, []
    for (bi, mi), obj in sorted(mesh_objects(m, source).items()):
        stash = obj.get("vtmb_meshes")
        if not stash:
            continue
        mats = obj.data.materials
        moves, missing = slot_moves(obj)
        gone.extend(missing)
        for j, mesh in enumerate(m.bodyparts[bi].models[mi].meshes):
            if j >= len(stash):
                break
            slot = slot_of(stash, moves, j)
            if slot is None or not 0 <= slot < len(mats) or mats[slot] is None:
                continue
            if slot != int(stash[j][0]):
                moved += 1
            ref = mesh.material
            row = m.skins[family] if 0 <= family < len(m.skins) else None
            if row and 0 <= ref < len(row):
                ref = row[ref]
            name = mats[slot].name
            if out.get(ref, name) != name:
                raise ValueError("texture %d is named both %r and %r by the scene's "
                                 "materials" % (ref, out[ref], name))
            out[ref] = name
    return out, moved, gone


def new_materials(m, source):
    """Blender material slot names on this file's meshes that no file mesh maps to.

    `obj["vtmb_meshes"]` records the slot each file mesh landed in, so a slot outside that
    set is one the scene added. Whether anything then references the record is a separate
    question and not one this has to answer: 648 shipped models carry an
    `mstudiotexture_t` no strip group reaches, so an unused record is legal.
    """
    out, seen = [], set()
    for (_bi, _mi), obj in sorted(mesh_objects(m, source).items()):
        stash = obj.get("vtmb_meshes")
        if not stash:
            continue
        # Resolved the same way `read_materials` resolves it, or a reorder makes every
        # slot look like one the scene added and the export appends the whole set again.
        moves = slot_moves(obj)[0]
        used = {slot_of(stash, moves, j) for j in range(len(stash))}
        for slot, mat in enumerate(obj.data.materials):
            if mat is None or slot in used or mat.name in seen:
                continue
            seen.add(mat.name)
            out.append(mat.name)
    return out


# The 32-chain limit and why it is hardware rather than a declared bound are on
# mdl_build.SPRING_MASK_BITS, which is the one definition of it.
SPRING_MASK_BITS = build_mod.SPRING_MASK_BITS

# (record offset, the pose bone key, what mdl_build.SPRING_DEFAULTS calls it). The retune
# loop and the panel's Add button both read this, so the five live in one place.
SPRING_KEYS = ((0x08, "vtmb_spring_unk08", "unk08"),
               (0x0c, "vtmb_spring_gravity", "gravity"),
               (0x10, "vtmb_spring_damping", "damping"),
               (0x14, "vtmb_spring_exp", "springexp"),
               (0x18, "vtmb_spring_maxangle", "maxangledeg"))


def spring_defaults():
    """{the pose bone key: the value a new chain starts at}, the corpus's own per field."""
    was = dict(build_mod.SPRING_DEFAULTS)
    return dict((key, was[field]) for _at, key, field in SPRING_KEYS)


def spring_start(k, have, off):
    """The `mstudiospringbone_t.bone` a chain takes when the scene switches it off or on.

    A negative field means three things at once and only the first is an encoding. The start
    bone is `-1 - bone`, read off SpringBoneChain_Construct (`client.dll 0x100ac115`);
    `CBaseAnimating::SetModel` switches the chain off through the mask above; and
    `CBaseAnimating::LookupPhysicsChain` (`vampire.dll 0x10097980`) compares the RAW field at
    `0x10097a5f` against a `LookupBone` result, which is never negative. That lookup is the one
    route from a bone name to the ordinal `TurnOnPhysicsChain` takes -- it is what animation
    events 2070 and 2071 resolve -- so a record written in the negative form can never be found
    and nothing in the game turns the chain back on. 0 of 600 shipped records use it.
    """
    start = have if have >= 0 else -1 - have
    if not off:
        return start
    if k >= SPRING_MASK_BITS:
        raise build_mod.Refused(
            "spring bone chain %d cannot start switched off: the engine sets bit "
            "%d of a 32-bit mask for it, which belongs to chain %d"
            % (k, k % SPRING_MASK_BITS, k % SPRING_MASK_BITS))
    return -1 - start


def spring_endbone(k, start, end, m):
    """The `mstudiospringbone_t.endbone` a named end bone resolves to, or a refusal.

    -1 is the sentinel and the only value 600 of 600 shipped records carry: the chain
    descends first-child from its start bone, `client.dll 0x100ac243`. A name takes the
    other branch, `0x100ac1dd`, which scans candidate bone indices from `numbones - 1`
    downward and appends one whenever it equals the current tail's parent, ending at
    `candidate < startbone` with no test that the start bone was ever reached. So the walk
    succeeds exactly when the start bone is the end bone or an ancestor of it, and on
    anything else the engine builds a chain the start bone is not in -- silently. That is
    refused here rather than written.
    """
    for b in m.bones:
        if b.name == end:
            j = b.index
            break
    else:
        raise build_mod.Refused(
            "spring bone chain %d ends at %r, which the file has no bone called. Empty "
            "means the chain runs to the leaf; delete the property and the file's own "
            "end bone is kept" % (k, end))
    up, steps = j, 0
    while up >= 0 and up != start and steps <= len(m.bones):
        up = m.bones[up].parent
        steps += 1
    if up != start:
        raise build_mod.Refused(
            "spring bone chain %d starts at %r and ends at %r, which is not below it. The "
            "engine walks up from the end bone and stops at the start bone's index, so "
            "the chain it builds would not contain %r"
            % (k, m.bones[start].name, end, m.bones[start].name))
    return j


def spring_chains(arm_obj):
    """(claimed {ordinal: pose bone}, [pose bones to append], {ordinal: pose bone to drop}).

    A pose bone carrying `vtmb_spring_index` claims that record; `vtmb_spring_new`, which
    the Add operator writes and the restamp clears, is a chain the scene authored and is
    appended.

    Both markers are positive and neither is the absence of the other, because absence is
    already spoken for twice over. A deleted stamp is exactly what a record no pose bone can
    claim looks like -- 4 of the 4445 shipped files carry one, two records on one start bone,
    which the import keys its lookup by -- and that record is carried verbatim, so a bone
    left holding the fields without a stamp is not an addition. A removal is a tombstone,
    `vtmb_spring_removed`, for the same reason.
    """
    claimed, add, drop = {}, [], {}
    for pb in arm_obj.pose.bones:
        k = pb.get("vtmb_spring_index")
        if k is None:
            if pb.get("vtmb_spring_new"):
                add.append(pb)
            continue
        k = int(k)
        if k in claimed or k in drop:
            continue
        if pb.get("vtmb_spring_removed"):
            drop[k] = pb
        else:
            claimed[k] = pb
    return claimed, add, drop


def apply_springbones(d, m, arm_obj):
    """What the scene does to the spring bone chains, as a dict.

    Matched on the record ordinal the import stamped rather than on the bone, because a
    chain is identified by its ordinal everywhere the engine touches it -- the disable
    mask is `1 << recordIndex` -- and two records may name one start bone.

    A record whose ordinal no pose bone claims is carried verbatim and named in
    `unclaimed` as `(ordinal, start bone name)`. The route a shipped file takes is exactly
    that shared start bone: the import keys its lookup by the bone, so the first record on
    one is stamped and any further record on it is not. 4 of the 4445 files carry it --
    ghost.mdl chains 2 and 9, the three tremere_female_armor_* chains 2 and 4, every one on
    a bone called Bone05 -- and the two records differ in all five floats.

    Dropped first, appended second, retuned last, so every ordinal a mask limit or a report
    names is the one this export is writing and not the one it read. `stamps` is what each
    surviving chain's pose bone should carry afterwards, which the operator writes back when
    the file it wrote is the one the scene reads: a removal shifts every later ordinal, so a
    second export off the stamps this one read would drop the wrong record.

    Comparison is against the packed float32, not the Python float, so a value the user
    never touched cannot rewrite the bytes it came from.
    """
    claimed, add, drop = spring_chains(arm_obj)
    out = {"changed": 0, "ends": 0, "switched": 0, "unclaimed": [],
           "added": [], "removed": [], "stamps": {}, "gone": [], "over_mask": 0}

    def name_of(i):
        return m.bones[i].name if 0 <= i < len(m.bones) else "bone %d" % i

    def start_of(r):
        v = struct.unpack_from("<i", r.raw, 0x00)[0]
        return v if v >= 0 else -1 - v

    had = len(d.springbones)
    for k in sorted(drop, reverse=True):
        r = build_mod.remove_springbone(d, k)
        out["removed"].append((k, name_of(start_of(r))))
        out["gone"].append(drop[k].name)
    out["removed"].reverse()
    out["gone"].reverse()
    # A claimed ordinal at or past the count the file had names a record this file has not
    # got, which is a stale stamp, and it is dropped here exactly as it was ignored before.
    renum = dict((k, k - sum(1 for g in drop if g < k))
                 for k in range(had) if k not in drop)
    claimed = dict((renum[k], pb) for k, pb in claimed.items() if k in renum)

    for pb in add:
        for b in m.bones:
            if b.name == pb.name:
                bi = b.index
                break
        else:
            raise build_mod.Refused(
                "a spring bone chain is authored on %r, which the file has no bone called"
                % pb.name)
        k = build_mod.add_springbone(d, bi,
                                     disabled=bool(pb.get("vtmb_spring_disabled")))
        out["added"].append((k, pb.name))
        claimed[k] = pb
        if k >= build_mod.SPRING_MASK_BITS:
            out["over_mask"] += 1

    for k, r in enumerate(d.springbones):
        pb = claimed.get(k)
        if pb is None:
            out["unclaimed"].append((k, name_of(start_of(r))))
            continue
        out["stamps"][pb.name] = k
        have0 = struct.unpack_from("<i", r.raw, 0x00)[0]
        off = pb.get("vtmb_spring_disabled")
        # Absent is the scene never having said, the rule +0x04 and the five floats follow.
        want0 = have0 if off is None else spring_start(k, have0, off)
        if have0 != want0:
            struct.pack_into("<i", r.raw, 0x00, want0)
            out["changed"] += 1
            out["switched"] += 1
        have = struct.unpack_from("<i", r.raw, 0x04)[0]
        end = pb.get("vtmb_spring_end")
        if end is None:
            # The scene never said, which is what a .blend predating the property and a
            # deleted key both look like. Carry the file's own, the rule the five floats
            # below already follow -- writing -1 here would repoint the chain at the leaf.
            want = have
        elif end:
            # +0x00 carries the start bone as -1-bone when the chain starts disabled, and
            # want0 is the form this export is writing rather than the one it read.
            want = spring_endbone(k, want0 if want0 >= 0 else -1 - want0, end, m)
            out["ends"] += 1
        else:
            want = -1
        if have != want:
            struct.pack_into("<i", r.raw, 0x04, want)
            out["changed"] += 1
        for at, key, _field in SPRING_KEYS:
            v = pb.get(key)
            if v is None:
                continue
            packed = struct.pack("<f", float(v))
            if bytes(r.raw[at:at + 4]) != packed:
                r.raw[at:at + 4] = packed
                out["changed"] += 1
    return out


# An accessory matrix survives Blender as the object's loc/rot/scale, so reading one back
# is a decomposition and not a copy. Measured over both check donors, the worst deviation
# is 1.325e-08 on an attachment matrix and 1.666e-06 on a box half-extent, which is a
# sqrt of float32 sums -- so the bound is relative, at 18x the worst seen.
ACC_EPS = 1e-6


def _same(a, b):
    return abs(a - b) <= ACC_EPS * max(1.0, abs(a), abs(b))


def accessory_objects(arm_obj):
    """(attachment empties, {set ordinal: [box empties]}, {set ordinal: set name}).

    Grouped by the stamped ordinal and not by the hierarchy: an empty carries one parent
    and that parent is the bone, so a set cannot also be an object parent.  A record the
    scene added carries no ordinal and sorts last, which is where `emit` appends it.
    """
    attach, boxes, names = [], {}, {}
    for obj in arm_obj.children_recursive:
        if obj.get("vtmb_attachment") is not None:
            attach.append(obj)
        elif obj.get("vtmb_hitbox_group") is not None:
            boxes.setdefault(int(obj.get("vtmb_hitboxset_index") or 0), []).append(obj)
        elif obj.get("vtmb_hitboxset") is not None:
            names[int(obj.get("vtmb_hitboxset_index") or 0)] = str(obj["vtmb_hitboxset"])
    big = 1 << 30
    attach.sort(key=lambda o: (int(o.get("vtmb_attachment_index", big)), o.name))
    for k in boxes:
        boxes[k].sort(key=lambda o: (int(o.get("vtmb_hitbox_index", big)), o.name))
    return attach, boxes, names


def eyeball_objects(arm_obj):
    """The eyeball empties under one armature, in the order the file will carry them.

    Keyed on `vtmb_eyeball_model` because an eyeball belongs to one mstudiomodel_t and the
    scene has no other way to say which; an empty the scene added carries no model name and
    lands on the first model the export writes.
    """
    out = {}
    for obj in arm_obj.children_recursive:
        if obj.get("vtmb_eyeball") is None:
            continue
        out.setdefault(str(obj.get("vtmb_eyeball_model") or ""), []).append(obj)
    big = 1 << 30
    for k in out:
        out[k].sort(key=lambda o: (int(o.get("vtmb_eyeball", big)), o.name))
    return out


def _accessory_bone(obj, m, arm_obj, bmap, what):
    """The file bone index an accessory empty is mounted on.

    An empty whose bone Blender deleted passes to the first surviving ancestor, which is
    where `remove_bone` already put the file's own record.
    """
    want = obj.parent_bone if obj.parent_type == "BONE" else ""
    if not want:
        raise ValueError("%s %r is not parented to a bone" % (what, obj.name))
    for b in m.bones:
        if bmap.get(b.name) == want:
            return b.index
    gone = getattr(m, "gone", {})
    name = want
    while name in gone:
        name = gone[name]
        if name in bmap:
            return next(b.index for b in m.bones if b.name == name)
    raise ValueError("%s %r is on bone %r, which is not one this model carries"
                     % (what, obj.name, want))


def read_attachments(m, arm_obj, scale):
    """[(name, type, bone, 12 floats)] out of the scene, in bone space.

    `matrix_basis` is already the file's own bone-space matrix -- Blender's bone parenting
    supplies `bone.matrix_local` and `matrix_parent_inverse` takes the tail offset back
    out -- so nothing here inverts anything and there is no error to cancel.
    """
    bmap = bone_map(m, arm_obj)
    out = []
    for obj in accessory_objects(arm_obj)[0]:
        bi = _accessory_bone(obj, m, arm_obj, bmap, "attachment")
        local = obj.matrix_basis
        out.append((str(obj.get("vtmb_attachment") or obj.name),
                    int(obj.get("vtmb_attachment_type") or 0), bi,
                    tuple(local[r][c] / (scale if c == 3 else 1.0)
                          for r in range(3) for c in range(4))))
    return out


def hitbox_extent(obj, scale):
    """(bbmin, bbmax) in bone space off one box empty's own matrix.

    A CUBE empty draws -size to +size, so the box is the empty's own scale and translation
    and comes back without a bounding-box walk.  0 of the 14151 shipped boxes has a zero
    extent on any axis, so nothing here has to survive a degenerate one.  Shared by the
    donor path and `blender_scratch.scene_hitboxes`, so what a box empty means has one
    definition and the turn is refused on both.
    """
    local = obj.matrix_basis
    mid = [local[r][3] / scale for r in range(3)]
    half = [local.col[c].to_3d().length / scale for c in range(3)]
    # mstudiobbox_t is an AABB in bone space and carries no rotation, so a turned empty
    # would be written as its unturned self. 1e-3 is three orders above the 1e-6 the
    # identity round trip costs and far below any deliberate turn.
    turn = max(abs(local.col[c].to_3d()[r] / (half[c] * scale or 1.0)
                   - (1.0 if r == c else 0.0))
               for c in range(3) for r in range(3))
    if turn > 1e-3:
        raise ValueError("hitbox %r is turned %.4f out of its bone's axes, and "
                         "mstudiobbox_t carries no rotation" % (obj.name, turn))
    return (tuple(a - b for a, b in zip(mid, half)),
            tuple(a + b for a, b in zip(mid, half)))


def read_hitboxsets(m, arm_obj, scale):
    """[(set name, [(bone, group, bbmin+bbmax)])] out of the scene, in bone space."""
    bmap = bone_map(m, arm_obj)
    _attach, boxes, names = accessory_objects(arm_obj)
    out = []
    # A set with no boxes is still a set: 358 of the 4445 shipped models carry exactly one
    # and it is empty, so keying off `boxes` alone drops it and numhitboxsets goes to 0.
    for k in sorted(set(boxes) | set(names)):
        recs = []
        for obj in boxes.get(k, ()):
            bi = _accessory_bone(obj, m, arm_obj, bmap, "hitbox")
            lo, hi = hitbox_extent(obj, scale)
            recs.append((bi, int(obj.get("vtmb_hitbox_group") or 0), lo + hi))
        out.append((names.get(k, "default"), recs))
    return out


# Assigning `matrix_basis` decomposes to loc/quaternion/scale in float32 and recomposes on
# read, which moves a direction cosine by up to 4.3e-06 on jeanette -- enough to flip the
# sign of a component that is itself 2.2e-06. That is far below anything an eyeball's aim
# can mean and far above ACC_EPS, so the two unit vectors get their own absolute bound;
# without it every re-export rewrites every eyeball record it did not touch.
DIR_EPS = 1e-5


def _same_dir(a, b):
    return abs(a - b) <= DIR_EPS


def _material_index(d, name, added=None):
    """The mstudiotexture_t slot named `name`, appended if the file carries none.

    An append is collected rather than silent: it is the one route by which an eyeball
    puts a material in a file, it runs after the mesh pass that fills
    `scene["added_materials"]`, and a name reaching here by a typo would otherwise cost a
    texture record and no `.vmt` with nothing said.
    """
    for i, r in enumerate(d.textures):
        if r.name == name:
            return i
    build_mod.add_material(d, name, None)
    if added is not None:
        added.append(name)
    return len(d.textures) - 1


def _flexdesc_index(d, name, added=None):
    """The mstudioflexdesc_t slot named `name`, appended if the file carries none.

    Collected the way `_material_index` collects a material: a flex name reaching here by
    a typo costs a descriptor the engine will never weight and nothing would be said.
    """
    was = len(d.flexdescs)
    i = build_mod.flexdesc_index(d, name)
    if added is not None and len(d.flexdescs) > was:
        added.append(name)
    return i


# The three lid-flex name sets the corpus carries, and there is no fourth: over the 602
# shipped records the eight names are these sixteen or all absent -- right on 196 records,
# left on 196, none on 210 -- and NO record is partly written. So a scene picks a set and
# never eight strings, which is what makes the half-written record unreachable.
EYE_LID_FLEXES = {
    "right": ("upper_right_lowerer", "upper_right_neutral", "upper_right_raiser",
              "lower_right_lowerer", "lower_right_neutral", "lower_right_raiser",
              "upper_right", "lower_right"),
    "left": ("upper_left_lowerer", "upper_left_neutral", "upper_left_raiser",
             "lower_left_lowerer", "lower_left_neutral", "lower_left_raiser",
             "upper_left", "lower_left"),
}


def eyeball_radius(obj, scale):
    """The record's radius in model units, which is the empty's own display size.

    The import draws each eyeball as a SPHERE empty of exactly that size, so resizing the
    sphere is the edit a person makes; `vtmb_eyeball_radius` records what the file held
    and is compared against this rather than read, because a key that wins makes the
    viewport lie about the number being written.  Shipped radii are 0.5 on 568 records,
    0.577 on 20, 0.57 on 10 and 0.6 on 4, so the import's `max(..., 1e-4)` clamp is
    reached by 0 of 602 and the round trip through the display size is exact.
    """
    return obj.empty_display_size / (scale or 1.0)


_FLEXCTRL_RE = re.compile(
    r"^\s*(\S+)\s+(?:range\s+(\S+)\s+(\S+)\s+)?(\S+)\s*$")

# Commonest first over 8649 controllers on 198 models: 2940, 2159, 1568, 1176, 784, 20, 2.
# The field is a string and not an enum, so the panel offers these and takes an eighth.
FLEX_TYPES = ("phoneme", "mouth", "eyelid", "brow", "nose", "morph", "wholeface")


def flex_controller_fields(line):
    """(type, min, max, name) off a `$flexcontroller` line, or ValueError naming it.

    The one grammar: the panel stores through it and `apply_flex` reads through it, so a
    line the panel wrote is one the export can parse.
    """
    mo = _FLEXCTRL_RE.match(line)
    if not mo:
        raise ValueError("flex controller %r is not `<type> [range <min> <max>] <name>`"
                         % line)
    t, lo, hi, name = mo.groups()
    try:
        lo = 0.0 if lo is None else float(lo)
        hi = 1.0 if hi is None else float(hi)
    except ValueError:
        raise ValueError("flex controller %r has a range that is not two numbers" % line)
    return t, lo, hi, name


def flex_controller_line(ctype, lo, hi, name):
    """The inverse of `flex_controller_fields`, in the import's own spelling."""
    rng = "" if (lo, hi) == (0.0, 1.0) else "range %g %g " % (lo, hi)
    return "%s %s%s" % (ctype, rng, name)


# Springs store `w0 = 2*sigma*k0` and `w1 = -2*sigma*(1-k0)` and group 1 stores
# `rest2 = slack * d2`, so one authored number moving is visible in every spring. Under
# 1e-4 is the preset table's own rounding -- `cloth.recover` matches a preset to exactly
# that -- and writing it would move a file nobody edited.
CLOTH_TOL = 1e-4
# Below this a rest separation carries no direction and its ratio is not recoverable.
CLOTH_MIN_D2 = 1e-9
# Cleared by cloth-donor.py's `noratio` control, which refits to the new separation itself
# and so loses the group-1 slack and whatever else the spring's own ratio carried.
CLOTH_KEEP_RATIO = True


def _cloth_slot(cl):
    """(slot, offset into cl["data"]) of the row-0 object the importer drew, or None.

    Row 0 is what `blender_import._stamp_cloth` puts on the mesh object, so it is the only
    one the scene can speak for.
    """
    cols = cl.get("cols") or 0
    for k, at in cl.get("slots") or ():
        if cols > 0 and k // cols == 0:
            return k, at
    return None


def _cloth_header(data, at):
    """The mstudiocloth_t fields a rewrite needs, off one object inside the carried blob."""
    scale = struct.unpack_from("<f", data, at)[0]
    npart, nfixed, _nfree, pvoff = struct.unpack_from("<4i", data, at + 0x04)
    _ns, ns0, ns1, spoff = struct.unpack_from("<4i", data, at + 0x14)
    pv = list(struct.unpack_from("<%dH" % npart, data, at + pvoff)) \
        if pvoff and npart else []
    springs = [struct.unpack_from("<2H3f", data, at + spoff + q * 16)
               for q in range(ns0 + ns1)] if spoff else []
    return scale, npart, nfixed, ns0, spoff, pv, springs


def _d2(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def cloth_sigma_edges(obj, pv, springs, ns0, donor):
    """{group-0 spring: the sigma the mesh edge carrying it holds}, empty where none does.

    `cloth.edge_keys` is the resolution and the import stamps the attribute through the same
    call, so a spring reaches the edge it was written to. 289 of the corpus's 30 623 group-0
    springs reach none and are absent here, which leaves them the value the file holds.
    """
    me = obj.data
    att = me.attributes.get("vtmb_cloth_sigma")
    if att is None or att.domain != "EDGE" or len(att.data) != len(me.edges):
        return {}
    buf = [0.0] * len(me.edges)
    att.data.foreach_get("value", buf)
    at = {cloth_mod.pair(*tuple(e.vertices)): e.index for e in me.edges}
    keys, _missed = cloth_mod.edge_keys(pv, springs, ns0, [v.pos for v in donor], set(at))
    return {q: buf[at[k]] for q, k in keys.items()}


def cloth_edits(m, source, d):
    """What each cloth-bound model's scene says that its donor bytes do not.

    One entry per model carrying cloth: which of the three authored numbers moved, how many
    particles sit somewhere else than the file compiled them at, and the reason the object
    cannot be spoken for at all. Split from the write so the same comparison reports the
    narrowing when the write is off -- a cloth object left describing the geometry it was
    compiled against simulates the old garment, and nothing on screen says so.
    """
    found = mesh_objects(m, source)
    out = []
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        rec = d.bodyparts[bi].kids[mi]
        cl = rec.extra.get("cloth")
        if not cl:
            continue
        obj = found.get((bi, mi))
        e = {"key": (bi, mi), "model": mo.name, "object": obj.name if obj else None,
             "why": None, "scale": None, "sigma": None, "slack": None, "moved": 0,
             "springs": 0, "flattens": False, "sigma_edges": 0}
        out.append(e)
        if obj is None:
            e["why"] = "no mesh object in the scene belongs to this model"
            continue
        slot = _cloth_slot(cl)
        if slot is None:
            e["why"] = "the file's cloth table names no object in row 0"
            continue
        if not obj.get("vtmb_cloth"):
            e["why"] = ("the Cloth box is off and the file's cloth object stays -- "
                        "removing one is not a write this exporter has")
            continue
        _k, at = slot
        scale_f, _npart, _nfixed, ns0, _spoff, pv, springs = _cloth_header(cl["data"], at)
        if not springs or not pv:
            e["why"] = "the file's cloth object carries no springs or no particle map"
            continue
        me = obj.data
        if len(me.vertices) != mo.numvertices:
            e["why"] = ("the scene holds %d vertices where the file's model has %d, so a "
                        "particle's rest position cannot be read"
                        % (len(me.vertices), mo.numvertices))
            continue
        donor = m.vertices(mo)
        was = [donor[v].pos if v < len(donor) else (0.0, 0.0, 0.0) for v in pv]
        r = cloth_mod.recover(scale_f, springs, ns0, was)
        preset = str(obj.get("vtmb_cloth_preset") or "") or None
        if preset is not None and preset not in cloth_mod.PRESETS:
            e["why"] = ("the object names cloth preset %r, which is not one of the 17"
                        % preset)
            continue
        p = cloth_mod.PRESETS[preset] if preset else None
        # The panel's own number first, then the preset it names, then what the file
        # already holds. A preset's `sigma` is a mean over a whole garment's objects and
        # is not this object's number, so it is never a source here.
        want_scale = obj.get("vtmb_cloth_scale")
        if want_scale is None:
            want_scale = p.scale if p else scale_f
        want_slack = obj.get("vtmb_cloth_slack")
        if want_slack is None:
            want_slack = p.s if p else r.slack
        want_sigma = obj.get("vtmb_cloth_sigma")
        if want_scale is not None and abs(float(want_scale) - scale_f) > CLOTH_TOL:
            e["scale"] = (scale_f, float(want_scale))
        if want_slack is not None and r.slack is not None \
                and abs(float(want_slack) - r.slack) > CLOTH_TOL:
            e["slack"] = (r.slack, float(want_slack))
        if want_sigma is not None and r.sigma is not None \
                and abs(float(want_sigma) - r.sigma) > CLOTH_TOL:
            e["sigma"] = (r.sigma, float(want_sigma))
            # 39 of the corpus's 150 objects carry a sigma that varies spring by spring and
            # the panel holds one number, so writing it replaces the variation.
            e["flattens"] = not r.uniform
        # The mesh's own per-edge sigma, which is what stops a re-export flattening the 28
        # of 59 shipped row-0 objects that vary it. A whole-object number set in the panel
        # wins over it and is what `flattens` reports.
        if want_sigma is None:
            e["sigma_edges"] = sum(
                1 for q, v in cloth_sigma_edges(obj, pv, springs, ns0, donor).items()
                if abs(v - (springs[q][2] - springs[q][3]) / 2.0) > CLOTH_TOL)
        e["moved"] = sum(1 for v, w in zip(pv, was)
                         if v < len(me.vertices)
                         and _d2(tuple(me.vertices[v].co), w) > CLOTH_MIN_D2)
        e["springs"] = len(springs)
    return out


def apply_cloth(d, m, source, edits=None):
    """Each model's row-0 cloth object rewritten in place from the scene.

    `scale` at +0x00, each spring's `w0`/`w1` from the panel's sigma with the file's own
    mass split kept, and `rest2` refitted to where the particle's vertex now sits. Nothing
    outside the object's own bytes moves, so the region keeps its length and the second
    indirection stays as it was -- the table stores no count and every payload offset is
    object-relative (anomalies section B10).

    A spring's own `rest2 / d2` carries the group-1 slack and whatever else the compiler
    left in it, so that ratio is what a geometry refit multiplies. The panel's slack
    replaces it only where the user moved that number, which is what keeps an unedited
    re-export byte for byte.
    """
    if edits is None:
        edits = cloth_edits(m, source, d)
    found = mesh_objects(m, source)
    out = {"models": 0, "springs": 0, "scale": 0, "sigma": 0, "slack": 0, "refitted": 0,
           "sigma_edges": 0, "flattened": []}
    for e in edits:
        if e["why"] is not None:
            continue
        if not (e["scale"] or e["sigma"] or e["slack"] or e["moved"]
                or e["sigma_edges"]):
            continue
        bi, mi = e["key"]
        cl = d.bodyparts[bi].kids[mi].extra["cloth"]
        _k, at = _cloth_slot(cl)
        data = bytearray(cl["data"])
        _scale_f, _npart, _nfixed, ns0, spoff, pv, springs = _cloth_header(data, at)
        mo = m.bodyparts[bi].models[mi]
        me, donor = found[(bi, mi)].data, m.vertices(mo)
        now = [tuple(me.vertices[v].co) if v < len(me.vertices) else (0.0, 0.0, 0.0)
               for v in pv]
        was = [donor[v].pos if v < len(donor) else (0.0, 0.0, 0.0) for v in pv]
        if e["scale"]:
            struct.pack_into("<f", data, at, e["scale"][1])
            out["scale"] += 1
        sigma = e["sigma"][1] if e["sigma"] else None
        slack = e["slack"][1] if e["slack"] else None
        per_edge = ({} if sigma is not None
                    else cloth_sigma_edges(found[(bi, mi)], pv, springs, ns0, donor))
        for q, (a, b, w0, w1, rest2) in enumerate(springs):
            # The panel's one number first, then the edge the spring sits on, then the
            # file's own. Rewriting a spring whose sigma did not move would re-round w0
            # through k0, so an untouched re-export is left alone rather than recomputed.
            s = sigma
            if (s is None and q in per_edge
                    and abs(per_edge[q] - (w0 - w1) / 2.0) > CLOTH_TOL):
                s = per_edge[q]
                out["sigma_edges"] += 1
            if s is not None and (w0 - w1) != 0.0:
                # k0 is w0 / (2*sigma), and 2*sigma is w0 - w1 whatever the mass split.
                k0 = w0 / (w0 - w1)
                w0, w1 = 2.0 * s * k0, -2.0 * s * (1.0 - k0)
            if a < len(now) and b < len(now):
                d2_was, d2_now = _d2(was[a], was[b]), _d2(now[a], now[b])
                if q >= ns0 and slack is not None:
                    rest2 = slack * d2_now
                elif d2_was > CLOTH_MIN_D2 and abs(d2_now - d2_was) > CLOTH_MIN_D2:
                    rest2 = (rest2 / d2_was) * d2_now if CLOTH_KEEP_RATIO else d2_now
                    out["refitted"] += 1
            struct.pack_into("<2H3f", data, at + spoff + q * 16, a, b, w0, w1, rest2)
            out["springs"] += 1
        cl["data"] = bytes(data)
        out["models"] += 1
        out["sigma"] += 1 if sigma is not None else 0
        out["slack"] += 1 if slack is not None else 0
        if e["flattens"]:
            out["flattened"].append(e["object"])
    return out


def apply_flex(d, m, arm_obj):
    """The flex controller and flex rule arrays, rebuilt from the armature's QC lines.

    (controllers, rules, added flexdescs). Raises Refused naming the line that would not
    parse, rather than dropping it: a rule that silently vanishes takes a facial
    expression with it and nothing downstream can tell.

    `link` is written -1 the way studiomdl/write.cpp:943 does. The engine patches it in
    place on first use (client.dll 100c4351) from a process-global name table, so the
    value in the file is never read and the NAME is what identifies the controller.
    """
    lines = [str(x) for x in (arm_obj.get("vtmb_flexcontrollers") or ())]
    rules = [str(x) for x in (arm_obj.get("vtmb_flexrules") or ())]
    if not lines and not rules:
        return 0, 0, 0

    ctls = []
    for line in lines:
        try:
            t, lo, hi, name = flex_controller_fields(line)
        except ValueError as e:
            raise build_mod.Refused(str(e))
        raw = bytearray(20)
        struct.pack_into("<iff", raw, 8, -1, lo, hi)
        ctls.append(build_mod.Rec(raw, name, None, {"type": t}))

    names = [c.name for c in ctls]
    # A rule resolves a controller by name, so a duplicate makes every rule naming it drive
    # the first silently. 0 of the 198 shipped models carry one.
    dup = next((n for k, n in enumerate(names) if n in names[:k]), None)
    if dup is not None:
        raise build_mod.Refused("two flex controllers are named %r, and a rule names a "
                                "controller by its name alone" % dup)
    added = 0
    recs = []
    # A rule the scene did not touch keeps the donor's own op bytes, so an unedited export
    # reproduces the uninitialised `d` the file carries instead of writing 0 over it.
    donor = {}
    for r in d.flexrules:
        donor.setdefault((struct.unpack_from("<i", r.raw, 0)[0],
                          _op_key(r.extra["ops"])), []).append(r.extra["ops"])
    for line in rules:
        if "=" not in line:
            raise build_mod.Refused("flex rule %r is not `<flexdesc> = <expression>`" % line)
        target, expr = line.split("=", 1)
        target = target.strip()
        if not target:
            raise build_mod.Refused("flex rule %r names no flex" % line)
        before = len(d.flexdescs)
        flex = build_mod.flexdesc_index(d, target)
        added += len(d.flexdescs) - before
        try:
            ops = mdl_mod.parse_flex_expr(
                expr, [_Named(n) for n in names],
                [r.name or "" for r in d.flexdescs])
        except ValueError as e:
            raise build_mod.Refused("flex rule %r: %s" % (line, e))
        raw = bytearray(12)
        struct.pack_into("<i", raw, 0, flex)
        blob = bytearray()
        for op, val in ops:
            blob += struct.pack("<2i", op, val)
        keep = donor.get((flex, _op_key(blob)))
        recs.append(build_mod.Rec(raw, None, None,
                                  {"ops": keep.pop(0) if keep else bytes(blob)}))

    d.flexcontrollers = ctls
    d.flexrules = recs
    return len(ctls), len(recs), added


def _op_key(blob):
    """The op stream with `d` dropped on the four binary opcodes.

    That field is uninitialised in every shipped file -- non-zero on 40 685 of 41 362 --
    and `client.dll 100c3de6/df9/e0c/e1c` read `+0x00` only, so two blocks agreeing
    everywhere else are the same rule.  Anomalies A10.
    """
    return tuple((op, 0 if 4 <= op <= 7 else val)
                 for op, val in struct.iter_unpack("<2i", bytes(blob)))


class _Named(object):
    """What `mdl.parse_flex_expr` reads off a controller, without an Mdl to read it from."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


def apply_face(d, m, arm_obj, scale):
    """The eyeball records and the one mouth record, written only where they differ.

    Answers `{changed, materials, radius_keys, lid_targets, flexdescs}` -- the record
    count, the material names an eyeball put in the file, the eyeballs whose
    `vtmb_eyeball_radius` says something neither the sphere nor the file says, the
    eyeballs holding lid flexes without aim points or the other way round, and the flex
    names a lid or the mouth appended to the file's mstudioflexdesc_t array.

    Every index a scene holds is a NAME here and is resolved against this file: the two
    materials against mstudiotexture_t, the eight lid flexes and the mouth's flex against
    mstudioflexdesc_t, the mouth's bone against the bone array. The corpus is why -- the
    mouth's flexdesc is 16 on 196 of 200 records and 0 on 4 that list `mouth` first, and
    the eyeball's own upperlidflexdesc is a genuine 0 on the right eye, so a stored index
    would say the wrong thing about either the moment a slot moves.

    zoffset (+0x14) and pitch/yaw (+0x7c, +0x84) are left at whatever the donor holds:
    they are 0 on 602 of 602 shipped records and the QC `eyeball` command has no token
    that sets them, so the scene draws neither.
    """
    out = {"changed": 0, "materials": [], "radius_keys": [], "lid_targets": [],
           "flexdescs": []}
    eyes = eyeball_objects(arm_obj)
    # A model record carries no name of its own in the description, so the name comes off
    # the parsed file walked beside it -- same file, same order.
    for bp, dbp in zip(m.bodyparts, d.bodyparts):
        for model, mr in zip(bp.models, dbp.kids):
            recs = (mr.extra or {}).get("eyes") or []
            want = eyes.get(model.name) or eyes.get("")
            if not recs or not want:
                continue
            for rec, obj in zip(recs, want):
                out["changed"] += _apply_eyeball(d, rec, obj, scale, out)
    out["changed"] += _apply_mouth(d, m, arm_obj, out)
    return out


def _apply_mouth(d, m, arm_obj, out):
    """The one mstudiomouth_t, written only where the scene says something else.

    All three fields are the scene's to set.  Over the 200 shipped carriers the names are
    unanimous -- `Bip01 Head`, (0, -1, 0) and `mouth` on 200 of 200 -- but that is Troika's
    rig and not the format: those 200 span 113 distinct bone sets which are all one rig
    family, every one carrying the whole Biped core, and 504 models carry a `Bip01 Head`
    where only 200 carry a mouth.  R_MouthSetupVertexShader reads every one of the three.

    The bone and the flex are NAMES in the stash: the bone index is 6 on 187 carriers, 12
    on 8, 7 on 3 and 14 on 2, and the flexdesc index is 16 on 196 and 0 on 4, so a stored
    index would say the wrong thing the moment a slot moves.
    """
    mouth = arm_obj.get("vtmb_mouth")
    if not mouth or not d.mouths:
        return 0
    raw = d.mouths[0].raw
    have = struct.unpack_from("<i3fi", raw, 0)
    # A key absent means the scene never said, so the file's own value stands -- the rule
    # a spring bone's end bone and an out-of-range blend cell already follow.
    bone = str(mouth.get("bone") or "")
    bi = have[0]
    if bone:
        bi = next((b.index for b in m.bones if b.name == bone), None)
        if bi is None:
            raise ValueError("the mouth is on bone %r, which is not one this model "
                             "carries" % bone)
    fwd = (tuple(float(c) for c in mouth["forward"])
           if mouth.get("forward") is not None else have[1:4])
    flex = str(mouth.get("flex") or "")
    fi = _flexdesc_index(d, flex, out["flexdescs"]) if flex else have[4]
    if (have[0] != bi or have[4] != fi
            or not all(_same(a, b) for a, b in zip(have[1:4], fwd))):
        struct.pack_into("<i3fi", raw, 0, bi, fwd[0], fwd[1], fwd[2], fi)
        return 1
    return 0


def _apply_eyeball(d, rec, obj, scale, out):
    raw = rec.raw
    local = obj.matrix_basis
    org = tuple(local[r][3] / scale for r in range(3))
    up = tuple(local.col[1][:3])
    fw = tuple(local.col[2][:3])
    radius = eyeball_radius(obj, scale)
    iris = _material_index(d, str(obj.get("vtmb_eyeball_iris") or ""), out["materials"])
    glint = _material_index(d, str(obj.get("vtmb_eyeball_glint") or ""), out["materials"])
    iscale = float(obj.get("vtmb_eyeball_iris_scale") or 0.0)
    upt = [float(c) for c in (obj.get("vtmb_eyeball_uppertarget") or (0.0, 0.0, 0.0))]
    lot = [float(c) for c in (obj.get("vtmb_eyeball_lowertarget") or (0.0, 0.0, 0.0))]
    lids = [str(x) for x in (obj.get("vtmb_eyeball_lidflexes") or ())]
    # A shipped record says "no lid flexes" by holding zero in all eight, never -1, so an
    # empty scene list writes eight zeros and not a sentinel.
    ids = ([_flexdesc_index(d, x, out["flexdescs"]) if x else 0 for x in lids]
           if len(lids) == 8 else [0] * 8)
    file_radius = struct.unpack_from("<f", raw, 0x18)[0]
    key = obj.get("vtmb_eyeball_radius")
    if key is not None:
        key = float(key)
        # The key is no longer the radius the export writes, so one that agrees with
        # neither the sphere nor the file was hand-edited and is being ignored. Said once
        # and then corrected, since a scene re-exported twice would otherwise hear it
        # about a radius it wrote itself.
        if not _same(key, radius) and not _same(key, file_radius):
            out["radius_keys"].append((obj.name, key, radius))
        if not _same(key, radius):
            obj["vtmb_eyeball_radius"] = radius
    # Lid flexes and the two aim points move together on all 602 shipped records -- 392
    # carry both and 210 carry neither -- so either alone is a state the format has never
    # held, and it is named rather than refused.
    if (len(lids) == 8 and any(lids)) == all(c == 0.0 for c in list(upt) + list(lot)):
        out["lid_targets"].append((obj.name, bool(len(lids) == 8 and any(lids))))
    have = (struct.unpack_from("<3f", raw, 0x08) + struct.unpack_from("<f", raw, 0x18)
            + struct.unpack_from("<3f", raw, 0x1c) + struct.unpack_from("<3f", raw, 0x28)
            + struct.unpack_from("<i", raw, 0x38) + struct.unpack_from("<f", raw, 0x3c)
            + struct.unpack_from("<i", raw, 0x40) + struct.unpack_from("<6i", raw, 0x44)
            + struct.unpack_from("<3f", raw, 0x5c) + struct.unpack_from("<3f", raw, 0x68)
            + struct.unpack_from("<2i", raw, 0x74))
    want = (org + (radius,) + up + fw + (iris, iscale, glint)
            + tuple(ids[:6]) + tuple(upt) + tuple(lot) + tuple(ids[6:]))
    # 0-2 org, 3 radius, 4-6 up, 7-9 forward, 10 iris, 11 iris_scale, 12 glint,
    # 13-18 the six lid flexes, 19-21 uppertarget, 22-24 lowertarget, 25-26 the two lids.
    ints = (10, 12, 13, 14, 15, 16, 17, 18, 25, 26)
    if all((a == b) if i in ints else
           (_same_dir(a, b) if 4 <= i <= 9 else _same(a, b))
           for i, (a, b) in enumerate(zip(have, want))):
        return 0
    struct.pack_into("<3f", raw, 0x08, *org)
    struct.pack_into("<f", raw, 0x18, radius)
    struct.pack_into("<3f", raw, 0x1c, *up)
    struct.pack_into("<3f", raw, 0x28, *fw)
    struct.pack_into("<ifi", raw, 0x38, iris, iscale, glint)
    struct.pack_into("<6i", raw, 0x44, *ids[:6])
    struct.pack_into("<3f", raw, 0x5c, *upt)
    struct.pack_into("<3f", raw, 0x68, *lot)
    struct.pack_into("<2i", raw, 0x74, *ids[6:])
    return 1


def apply_accessories(d, m, arm_obj, scale):
    """Both record arrays back into the description, written only where they differ.

    Comparison is against the packed float32 and not the Python float, the way
    `apply_springbones` does it, so a record the user never touched cannot rewrite the
    bytes it came from.
    """
    changed = 0
    want = read_attachments(m, arm_obj, scale)
    if len(want) != len(d.attachments):
        d.attachments = d.attachments[:0]
        for name, kind, bone, local in want:
            build_mod.add_attachment(d, name, bone,
                                     (local[0:4], local[4:8], local[8:12]), kind)
        changed += len(want)
    else:
        for rec, (name, kind, bone, local) in zip(d.attachments, want):
            if rec.name != name:
                rec.name = name
                changed += 1
            have = struct.unpack_from("<ii", rec.raw, 4) + struct.unpack_from("<12f", rec.raw, 12)
            if (have[0], have[1]) != (kind, bone) or not all(
                    _same(a, b) for a, b in zip(have[2:], local)):
                struct.pack_into("<ii12f", rec.raw, 4, kind, bone, *local)
                changed += 1

    want = read_hitboxsets(m, arm_obj, scale)
    have = [(r.name, [(struct.unpack_from("<ii", x.raw, 0)
                       + struct.unpack_from("<6f", x.raw, 8)) for x in r.kids])
            for r in d.hitboxsets]
    same = len(want) == len(have) and all(
        wn == hn and len(wb) == len(hb)
        and all((b, g) == (h[0], h[1]) and all(_same(x, y) for x, y in zip(v, h[2:]))
                for (b, g, v), h in zip(wb, hb))
        for (wn, wb), (hn, hb) in zip(want, have))
    if not same:
        d.hitboxsets = d.hitboxsets[:0]
        for name, recs in want:
            build_mod.add_hitbox(d, [(b, g, v[0:3], v[3:6]) for b, g, v in recs], name)
        changed += sum(len(r) for _n, r in want) or 1
    return changed


def _apply_params(rec, params, names, where):
    """One sequence's paramindex/paramstart/paramend, resolved back from names.

    The resolve and its refusal live in `mdl_build` beside the rest of the record
    layout, so a check can reach them without Blender.
    """
    if params is None:
        return 0
    idx, start, end = build_mod.encode_params(params, names, "sequence %r: " % where)
    was = (struct.unpack_from("<2i", rec.raw, mdl_mod.SEQ_PARAMINDEX),
           struct.unpack_from("<2f", rec.raw, mdl_mod.SEQ_PARAMSTART),
           struct.unpack_from("<2f", rec.raw, mdl_mod.SEQ_PARAMEND))
    struct.pack_into("<2i", rec.raw, mdl_mod.SEQ_PARAMINDEX, *idx)
    struct.pack_into("<2f", rec.raw, mdl_mod.SEQ_PARAMSTART, *start)
    struct.pack_into("<2f", rec.raw, mdl_mod.SEQ_PARAMEND, *end)
    return 1 if was != (tuple(idx), tuple(start), tuple(end)) else 0


def sequence_actions(anim_names, source, arm_obj, actions=None):
    """{animation index: action} for reading Loops and Activity off, the export's map last.

    `match_indices` is what decides which action speaks for which animation everywhere
    else, ties dropped and all, so it decides it here too rather than a second rule
    picking one of three copies by `bpy.data.actions` order. The wider scan is what makes
    the panel's two fields reach the file on an export that replaces no animation at all;
    an action the export is writing wins over one merely stamped from the same file.
    """
    out = dict(match_indices(anim_names, source, arm_obj)[0])
    out.update({int(k): v for k, v in (actions or {}).items()})
    return out


# Cleared by blend-index-check.py's `refuse` control, which puts the raise back on a cell
# the stash holds as "" -- the form that failed every export of the four models that ship
# one.
BLEND_KEEP_INDEX = True


def apply_sequences(d, m, arm_obj, anim_names, actions=None):
    """Label, activity, group size and the blend grid out of `arm_obj["vtmb_sequences"]`.

    Matched by position: the stash is written in file order. A file carrying more than the
    stash does is one an append has just grown, and the sequences past its end are ones the
    scene says nothing about, so they keep what `add_sequence` wrote; a stash longer than
    the file claims sequences that are not there and is refused. Blends are stored as
    animation names because an index means nothing once the file is re-emitted, so they
    resolve back through `anim_names`.

    `actions` is {animation index: action}. Two fields are the action's rather than the
    armature's -- the flags word at +0x08, whose bit 0 is Loops, and the activity -- since
    those are the two the panel puts on an Action and the stash is written once at import
    and never again. An action reaches its sequence through the animation its first blend
    names, which is the pairing the import stamps both properties through.

    A blend cell the stash holds as `""` is one whose index named no animation of the file
    -- 8 of the 14012 shipped sequences, over 4 models -- so there is no name to resolve
    and the word the file already carries stands, the rule a knockback record on a deleted
    bone follows.

    The list itself is the scene's as well. `vtmb_seqs_removed` names the labels a removal
    tombstoned, `src` on an entry is the file ordinal it came from and is what a reorder is
    resolved through, and a trailing entry marked `new` is one the scene authored, which
    `add_sequence` appends. All three are positive marks rather than absences, the
    spring-chain rule: a stash with no `src` on it is an older `.blend` and is matched by
    position exactly as it was, and one shorter than the file with nothing tombstoned still
    leaves the trailing records alone.

    Returns (what moved, those cells, what the list edits did).
    """
    stash = list(arm_obj.get("vtmb_sequences") or ())
    moves = {"added": [], "removed": [], "reordered": 0, "orphans": [],
             "autolayers_lost": [], "dangling": [], "missing": [], "twice": []}
    index = {n: k for k, n in enumerate(anim_names)}
    n = 0

    # Resolved to the record OBJECT before anything is dropped: a removal renumbers every
    # later ordinal, and a `src` names the file as the import read it.
    byord = list(d.seqs)
    for label in [str(x) for x in (arm_obj.get("vtmb_seqs_removed") or ())]:
        at = [k for k, r in enumerate(d.seqs) if (r.name or "") == label]
        if not at:
            # The donor no longer has it, so the tombstone has nothing to answer. Reported
            # rather than refused: an export to another path leaves the mark standing, and
            # the next export to that path removes the sequence off the donor again.
            moves["missing"].append(label)
            continue
        was, lost, orphans, naming = build_mod.remove_sequence(d, at[0])
        moves["removed"].append(was)
        moves["autolayers_lost"] += lost
        moves["dangling"] += naming
        moves["orphans"] = orphans
        n += 1

    fresh = 0
    while fresh < len(stash) and stash[len(stash) - 1 - fresh].get("new"):
        fresh += 1
    head = stash[:len(stash) - fresh]
    for k, x in enumerate(head):
        if x.get("new"):
            raise ValueError("the armature's sequence %d of %d is one the scene added, and "
                             "an added sequence is appended, so it has to be the last"
                             % (k, len(stash)))
    if len(head) > len(d.seqs):
        raise ValueError("the armature carries %d sequence%s the file is supposed to already "
                         "have, %d of them marked as added, and the file has %d"
                         % (len(head), "" if len(head) == 1 else "s", fresh, len(d.seqs)))

    live = {id(r): k for k, r in enumerate(d.seqs)}
    want, keyed = [], bool(head)
    for x in head:
        src = x.get("src")
        at = (live.get(id(byord[int(src)]))
              if src is not None and 0 <= int(src) < len(byord) else None)
        if at is None:
            keyed = False
            break
        want.append(at)
    if keyed and len(set(want)) == len(want):
        order = want + [k for k in range(len(d.seqs)) if k not in set(want)]
        if order != list(range(len(d.seqs))):
            moves["reordered"] = build_mod.reorder_sequences(d, order)
            n += moves["reordered"]

    for x in stash[len(head):]:
        cells = [list(col) for col in x.get("blends") or ()]
        first = str(cells[0][0]) if cells and cells[0] else ""
        if first not in index:
            raise ValueError("the sequence the scene added blends animation %r, which the "
                             "file does not have" % first)
        build_mod.add_sequence(d, str(x.get("label") or first), index[first],
                               x.get("activity"), int(x.get("flags") or 0))
        moves["added"].append(str(x.get("label") or first))
        n += 1

    if not stash:
        # Nothing to match, and the two-value early return this used to take made every
        # caller unpack an int -- BUGS 136.
        return n, [], moves
    pp = [r.name or "" for r in getattr(d, "poseparams", [])]
    # Taken from the STASH, not from the file: a scene that renamed a sequence has to be
    # able to name it in another sequence's autolayer list under the new name.
    labels = {}
    for k, s in enumerate(stash):
        labels.setdefault(s.get("label") or (d.seqs[k].name or ""), k)
    for k, rec in enumerate(d.seqs):
        labels.setdefault(rec.name or "", k)
    bones = knockback_bones(d, m)
    out_of_range = []
    for rec, s in zip(d.seqs, stash):
        label = s.get("label")
        if label and rec.name != label:
            rec.name, n = label, n + 1
        activity = s.get("activity")
        if activity is not None:
            # `""` is written as an empty string and not as a zero word: 2433 of the 14012
            # shipped sequences carry a pointer to one and none carries a zero, so that is
            # what "this sequence claims no activity" looks like in the file.
            activity = str(activity)
            if rec.extra.get("activity") != activity:
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
                if not name and BLEND_KEEP_INDEX:
                    # `sequence_stash` writes "" where the file's own index named no
                    # animation, so the word the file carries stands and is reported.
                    out_of_range.append((rec.name or "", x, y,
                                         struct.unpack_from("<h", rec.raw, at)[0]))
                    continue
                a = index.get(name, -1)
                if a < 0:
                    raise ValueError("sequence %r blends animation %r, which the file does "
                                     "not have" % (rec.name, name))
                if struct.unpack_from("<h", rec.raw, at)[0] != a:
                    n += 1
                struct.pack_into("<h", rec.raw, at, a)
        n += _apply_events(rec, s.get("events"))
        n += _apply_params(rec, s.get("params"), pp, rec.name)
        n += _apply_seq_tail(rec, s, labels, bones)
    # After the grid, not inside the loop: which animation a sequence names is what picks
    # the action, and the stash has just rewritten it.
    for k, act in _first_citers(d, actions).items():
        n += _apply_seq_action(d.seqs[k], act)
    # Last, off the grid the stash has just written. 0 of the 4445 shipped models carry an
    # animation two sequences cite, so an added sequence naming one another already plays
    # makes the first of them, and `_action_seq` reaches only the earlier of the two.
    cited = {}
    for rec in d.seqs:
        for a in set(build_mod._seq_anims(rec.raw)):
            cited[a] = cited.get(a, 0) + 1
    moves["twice"] = [anim_names[a] if 0 <= a < len(anim_names) else "animation %d" % a
                      for a, c in sorted(cited.items()) if c > 1]
    return n, out_of_range, moves


def _first_citers(d, actions):
    """{sequence index: action}, one sequence per animation and the first that names it.

    The import stamps `vtmb_seq_flags` and `vtmb_activity` from the first sequence citing
    each animation, so that is the one sequence an action can be read back into. Handing
    the value to every citer instead would let one sequence's flags overwrite another's on
    a plain re-export, which is the one thing this must not cost.
    """
    if not actions:
        return {}
    out, taken = {}, set()
    for k, rec in enumerate(d.seqs):
        gx, gy = struct.unpack_from("<ii", rec.raw, mdl_mod.SEQ_GROUPSIZE)
        # The whole grid and in the stash's own order, which is what `_first_seq` walks
        # on the way in -- a grid's second row citing an animation first is what decides
        # which sequence that animation's action was stamped from.
        for x in range(max(0, min(gx, 16))):
            for y in range(max(0, min(gy, 16))):
                at = mdl_mod.SEQ_ANIM + x * 0x20 + y * 2
                if at + 2 > len(rec.raw):
                    continue
                a = struct.unpack_from("<h", rec.raw, at)[0]
                if a in taken:
                    continue
                taken.add(a)
                if a in actions:
                    out.setdefault(k, actions[a])
    return out


def _apply_seq_action(rec, act):
    """The two sequence fields the panel puts on an Action, after the stash has had its
    say. Returns what moved.

    An action carrying neither key says nothing and the stash stands; `vtmb_activity`
    present and empty is a sequence with no activity, which is the only way a donor's own
    comes off.
    """
    if act is None:
        return 0
    n = 0
    flags = act.get("vtmb_seq_flags")
    if flags is not None:
        flags = int(flags)
        if struct.unpack_from("<i", rec.raw, mdl_mod.SEQ_FLAGS)[0] != flags:
            n += 1
        struct.pack_into("<i", rec.raw, mdl_mod.SEQ_FLAGS, flags)
    activity = act.get("vtmb_activity")
    if activity is not None:
        activity = str(activity)
        if rec.extra.get("activity") != activity:
            rec.extra["activity"], n = activity, n + 1
    return n


_SEQ_SCALARS = (("flags", "<i", mdl_mod.SEQ_FLAGS),
                ("statrequired", "<i", mdl_mod.SEQ_STATREQUIRED),
                ("seqselectmask", "<i", mdl_mod.SEQ_SEQSELECTMASK),
                ("node", "<3i", mdl_mod.SEQ_ENTRYNODE),
                ("phase", "<2f", mdl_mod.SEQ_ENTRYPHASE),
                ("meleerange", "<2f", mdl_mod.SEQ_MELEERANGE),
                ("cyclewindow", "<3f", mdl_mod.SEQ_CYCLEWINDOW))


def _apply_seq_tail(rec, s, labels, bones):
    """The sequence record past the blend grid, out of the stash. Returns what moved.

    The four string fields and the three arrays go through `rec.extra`, which is where
    `mdl_build` re-emits them from; the scalars are packed straight into the 764-byte
    record, whose pointer words `mdl_build` zeroes and re-patches on its own.
    """
    n = 0
    for key, fmt, at in _SEQ_SCALARS:
        v = s.get(key)
        if v is None:
            continue
        # a Blender ID property array is neither a list nor a tuple
        v = list(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else [v]
        if len(v) != struct.calcsize(fmt) // 4:
            raise build_mod.Refused("sequence %r: %s wants %d number(s), the scene has %d"
                                    % (rec.name, key, struct.calcsize(fmt) // 4, len(v)))
        v = [int(x) if fmt.endswith("i") else float(x) for x in v]
        if list(struct.unpack_from(fmt, rec.raw, at)) != v:
            n += 1
        struct.pack_into(fmt, rec.raw, at, *v)

    for key, extra in (("dodge", "dodge"), ("block", "block"),
                       ("name2e8", "seq2e8"), ("name2ec", "seq2ec")):
        v = s.get(key)
        if v is None:
            continue
        v = str(v) or None
        if rec.extra.get(extra) != v:
            rec.extra[extra], n = v, n + 1

    al = s.get("autolayers")
    if al is not None:
        out = []
        for name in al:
            if str(name) not in labels:
                raise build_mod.Refused(
                    "sequence %r auto-layers %r, which the file has no sequence called"
                    % (rec.name, str(name)))
            out.append(labels[str(name)])
        if list(rec.extra.get("autolayers") or []) != out:
            rec.extra["autolayers"], n = out, n + 1

    hv = s.get("hitvolumes")
    if hv is not None:
        blob = bytearray()
        for h in hv:
            blob += struct.pack("<6f", *(list(h["bbmin"]) + list(h["bbmax"])))
        if bytes(rec.extra.get("hitvolumes") or b"") != bytes(blob):
            rec.extra["hitvolumes"], n = bytes(blob), n + 1

    n += _apply_knockbacks(rec, s.get("knockbacks"), bones)
    return n


def _apply_knockbacks(rec, kbs, bones):
    """Bone, cycle end and the 4x4 activity grid, into the donor's own 188-byte records.

    The other 170 bytes have no scene representation, so a stash whose count differs from
    the file's is refused rather than half-written -- a new record would go out as zeroes
    and the engine walks it unconditionally.

    `bones` is `knockback_bones`: keyed by the name the file gave each bone, since that is
    what the stash holds, and None where the scene deleted one.
    """
    if kbs is None:
        return 0
    have = rec.extra.get("knockbacks") or []
    if len(kbs) != len(have):
        raise build_mod.Refused(
            "sequence %r carries %d knockback record(s) in the scene and %d in the file; "
            "the rest of each record is not in the scene, so the count cannot change"
            % (rec.name, len(kbs), len(have)))
    n = 0
    for k, (src, dst) in enumerate(zip(kbs, have)):
        name = str(src.get("bone") or "")
        if name not in bones:
            raise build_mod.Refused(
                "knockback %d of sequence %r drives bone %r, which the file has not got and "
                "the scene did not delete. The stash names a bone by the name the file gave "
                "it, so a stash written against another model reaches this"
                % (k, rec.name, name))
        at = bones[name]
        # None is a bone the scene deleted, and the record has already been rebound to that
        # bone's parent by `remove_bone` and reported, so the file's own index stands. Cycle
        # end and the activity grid below are still written: those are the scene's whatever
        # happened to the bone.
        if at is not None:
            if struct.unpack_from("<i", dst.raw, 0x08)[0] != at:
                n += 1
            struct.pack_into("<i", dst.raw, 0x08, at)
        ce = float(src.get("cycleend", 0.0))
        if struct.unpack_from("<f", dst.raw, 0x04)[0] != ce:
            n += 1
        struct.pack_into("<f", dst.raw, 0x04, ce)
        counts = struct.unpack_from("<4i", dst.raw, 0x28)
        flat = list(dst.extra.get("names") or [None] * 16)
        for g, row in enumerate(src.get("activities") or []):
            for t, txt in enumerate(row):
                at = g * 4 + t
                allowed = t < min(abs(counts[g]), 4)
                if bool(txt) and not allowed:
                    raise build_mod.Refused(
                        "knockback %d of sequence %r names activity %r in slot %d of group "
                        "%d, which its own count of %d does not reach"
                        % (k, rec.name, str(txt), t, g, counts[g]))
                v = str(txt) or None if allowed else None
                if flat[at] != v:
                    n += 1
                flat[at] = v
        dst.extra["names"] = flat
    return n


def _apply_events(rec, events):
    """Rewrite one sequence's mstudioevent_t array from the stash. Returns what moved.

    The encoding lives in `mdl_build` beside the rest of the record layout, so a check
    can reach it without Blender.
    """
    if events is None:
        return 0
    out = build_mod.encode_events(events, "sequence %r: " % rec.name)
    if out == list(rec.extra.get("events") or []):
        return 0
    rec.extra["events"] = out
    return 1


def verify_writer(src):
    """The writer must reproduce this donor before anything from the scene is fed in.

    Grades the relayout alone, which is what makes the differences that follow attributable
    to the scene: an edit is an intended difference and would drown the check, so the
    comparison is `emit(from_bytes(x))` against `x` and never against the edited output.
    """
    plain = build_mod.emit(build_mod.from_bytes(src))
    bad, _n = rebuild_mod.verify(src, plain)
    return bad


# The two flavour strings Mod_LoadVtxFile_vtmb builds a path from, .dx80 first. The
# engine asks for .dx7_2bone only under -dxlevel 70.
VTX_FLAVOURS = ("dx80", "dx7_2bone")


def vtx_path(path, flavour="dx80"):
    """The .vtx beside a .mdl."""
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    return "%s.%s.vtx" % (stem, flavour)


def stale_flavours(dest, written=("dx80",)):
    """Other .vtx flavours beside the file just written, which now disagree with it.

    A flavour nobody rewrote still describes the old geometry. The engine tests only the
    flavour it loaded and takes `.dx80.vtx` first, so this is a warning, not an error.
    """
    return [os.path.basename(p) for p in
            (vtx_path(dest, f)
             for f in ("dx7_2bone", "dx90", "sw") if f not in written)
            if os.path.exists(p)]


def cut_vtx_lods(source, dest, written=(), flavours=None):
    """Every `.vtx` beside the written model turned down to one LOD. One row per file.

    Every flavour that exists is edited, not only the one the geometry pass rewrote: the
    engine picks by `-dxlevel` and a flavour left at seven LODs still swaps to a coarse
    mesh. A flavour the geometry pass did not write is copied across from the donor first,
    since the cut is the only reason that file would appear beside a new `dest`.

    `flavours` narrows that to the ones named, which is what a cut forced by a renumbering
    rebuild wants: a donor flavour nobody rewrote describes the old numbering whole, so
    copying a cut version of it across replaces one wrong file with another.
    """
    out = []
    for flavour in (VTX_FLAVOURS if flavours is None else flavours):
        src_p, dst_p = vtx_path(source, flavour), vtx_path(dest, flavour)
        at = dst_p if (flavour in written and os.path.exists(dst_p)) else src_p
        if not os.path.exists(at):
            continue
        with open(at, "rb") as f:
            data = f.read()
        try:
            cut, was, dropped = vtxw_mod.cut_lods(data)
        except (ValueError, struct.error) as exc:
            out.append({"flavour": flavour, "file": os.path.basename(dst_p), "was": None,
                        "dropped": [], "why": str(exc)})
            continue
        if cut != data or at != dst_p:
            with open(dst_p, "wb") as f:
                f.write(cut)
        out.append({"flavour": flavour, "file": os.path.basename(dst_p), "was": was,
                    "dropped": dropped, "why": None,
                    "bytes": sum(1 for a, b in zip(data, cut) if a != b)})
    return out


def phy_path(p):
    return (p[:-4] if p[-4:].lower() == ".mdl" else p) + ".phy"


def phy_bone_names(path, names):
    """Which of `names` the sibling `.phy` cites, or None when there is no `.phy`.

    Collision and ragdoll live in that file and nothing here writes it. It references the
    model by bone name and by nothing else -- over 2929 shipped files no key names a
    vertex, a face, a mesh or a bone index -- so a moved vertex leaves collision merely
    wrong, while a renamed or deleted bone leaves `CRagdollProp::CreateObjects` unable to
    look the solid up at all. Anomalies I.9.

    Matching every quoted value rather than a key list means a key some other tool added
    cannot hide a reference from this.
    """
    ph = phy_path(path)
    if not os.path.exists(ph):
        return None
    try:
        with open(ph, "rb") as f:
            b = f.read()
        size, _, nsolid, _ = struct.unpack_from("<4i", b, 0)
        o = size
        for _ in range(nsolid):
            o += 4 + struct.unpack_from("<i", b, o)[0]
        text = b[o:].decode("ascii", "replace")
    except (OSError, struct.error, IndexError):
        # A file this cannot read is one to warn about rather than to fail the export on.
        return set(names)
    quoted = set(text.split('"')[1::2])
    return set(n for n in names if n in quoted)


def phy_report(source, dest, removed, renamed, geometry_moved):
    """What the sibling `.phy` no longer describes, or None when it has nothing to say."""
    old_names = [g["name"] for g in removed or ()] + [a for a, _ in renamed or ()]
    cited = phy_bone_names(dest, old_names)
    if cited is None:
        # Nothing beside the file just written. Deleting a `.phy` is worse than leaving one
        # stale: `CBaseAnimating::TestCollision` returns false with no solid, so the model
        # stops being hit by traces at all rather than being hit at the wrong shape.
        if os.path.exists(phy_path(source)) and os.path.abspath(source) != os.path.abspath(dest):
            return {"file": os.path.basename(phy_path(dest)), "missing": True,
                    "removed": [], "renamed": [], "geometry": geometry_moved}
        return None
    gone = sorted(g["name"] for g in removed or () if g["name"] in cited)
    moved = sorted((a, b) for a, b in renamed or () if a in cited)
    if not gone and not moved and not geometry_moved:
        return None
    return {"file": os.path.basename(phy_path(dest)), "missing": False,
            "removed": gone, "renamed": moved, "geometry": geometry_moved}


def revise_vtx(source, dest, data, revised, flavours=("dx80",)):
    """Rewrite the .vtx for a model whose geometry moved, and return what each cost.

    The donor's own file supplies every strip group the edit did not touch, so a change to
    one mesh leaves the others byte for byte as the compiler emitted them. Each flavour is
    revised from its own donor: the two files cap bones per triangle at 9 and at 2, so
    they do not share a strip-group partition and neither can be derived from the other.

    A flavour asked for but absent beside the source is skipped, except `.dx80.vtx`, which
    the engine takes first and without which it draws nothing.
    """
    written = mdl_mod.Mdl("<written>", data=data)
    out = []
    for flavour in flavours:
        src = vtx_path(source, flavour)
        if not os.path.exists(src):
            if flavour == "dx80":
                raise ValueError(
                    "%s has no .dx80.vtx beside it, and a changed vertex or face count "
                    "needs one rewritten -- the engine draws nothing when the pair "
                    "disagrees" % os.path.basename(source))
            continue
        blob, st = vtxr_mod.revise(written, src, revised)
        path = vtx_path(dest, flavour)
        with open(path, "wb") as f:
            f.write(blob)
        st["path"] = path
        st["bytes"] = len(blob)
        st["flavour"] = flavour
        out.append(st)
    return out


def export_actions(context, arm_obj, source, dest, actions, scale=1.0,
                   frame_start=None, frame_end=None, fps=None, root_motion_in_keys=True,
                   root_motion="keep", mesh_fields=(), verify=True, add=(), drop="",
                   model_name="", hull=None, cdtexture=None, write_flexes=False,
                   write_cloth=True, cut_lods=False, vtx_flavours=("dx80",)):
    """Author `source` again with `actions`, an {animation index: action} map, applied.

    `add` is actions appended as new animations rather than replacing one, each with a
    sequence of the same name -- the engine reaches an animation only through a sequence,
    so one with none is dead weight. Appending renumbers nothing, every stored index
    pointing below the insertion point, and `emit` requantises the animations already in
    the file when the new pose widens the file-wide `mstudiobone_t` scales.

    `root_motion` is one of the four writes forced on every action, or `per_action`, which
    reads each action's own stamp -- see `root_motion_mode`.

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
    # Before anything is read off the scene: Shift+D copies the stamps, so two objects can
    # claim one model and the file has one mesh per model to put them in. A copy the export
    # can tell apart is dropped and reported; one it cannot is refused, since taking either
    # discards the other's edits whole.
    dup_models, dup_clash = duplicate_models(m, source)
    if dup_clash:
        label, names = dup_clash[0]
        raise build_mod.Refused(
            "%s claim model %r and none of them is the one the import made -- its mesh "
            "datablock is still called what vtmb_model_label says. The file holds one mesh "
            "per model, so unparent all but one from the armature, or clear vtmb_index and "
            "the bodypart stamps on the copies"
            % (", ".join(repr(n) for n in names), label))
    # Before `bone_map` raises on an unresolvable one, and before the appends make each copy
    # a bone of the file in its own right.
    dup_bones = duplicate_bones(m, arm_obj)
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
    # @180/@192 is the .qc's $bbox, which Mod_GetBounds starts from and +0x1c/+0x28 only
    # widen. GetRenderBounds reads that sequence box alone, so a zero one draws only while
    # the entity origin is on screen, and costs the collision radius vampire.dll takes off
    # the same floats. Neither follows an edit: a donor keeps the boxes it shipped, which
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

    edits, wrote, pending, unfitted, unkeepable = {}, [], [], [], []
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

            mode = root_motion_mode(action, root_motion)
            # "extract" wants the motion still in the poses, so it is only taken back out
            # when the donor's own blocks are the ones being kept.
            poses = read_poses(context, arm_obj, m, frames, scale, anim,
                               root_motion_in_keys=(mode == "keep"
                                                    and root_motion_in_keys
                                                    and bool(action.get("vtmb_root_motion"))))
            movements = None
            if mode == "extract":
                movements, poses = write_mod.extract_root_motion(m, poses)
                if lost_travel(action, movements):
                    unfitted.append(action.name)
            elif mode == "in_place":
                # No blocks and no net travel: the ramp `fit_movements` fits IS the net
                # ground displacement, so subtracting it and throwing the block away is
                # "extract" with nothing for the engine to carry. What the ramp does not
                # account for stays in the keys -- the rise and sway of a run cycle, which
                # an animation played in place should still have.
                _mv, poses = write_mod.extract_root_motion(m, poses)
                movements = []
            elif mode == "none":
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
            mode = root_motion_mode(action, root_motion, donor=False, default="extract")
            poses = read_poses(context, arm_obj, m, frames, scale)
            movements = ()
            if mode == "extract":
                movements, poses = write_mod.extract_root_motion(m, poses)
                if lost_travel(action, movements):
                    unfitted.append(action.name)
            elif mode == "in_place":
                _mv, poses = write_mod.extract_root_motion(m, poses)
            elif mode == "keep" and lost_travel(action, movements):
                # Only a forced "keep" reaches this: a stamped one became "extract" above.
                unkeepable.append(action.name)
            pending.append((action, poses, tuple(movements), len(frames)))
    finally:
        if ad is not None and ad.action is not restore:
            ad.action = restore

    d = build_mod.apply_anims(d, edits, source)

    scene = {"bones": 0, "materials": 0, "sequences": 0, "springs": 0,
             "spring_ends": 0, "spring_switched": 0, "spring_unclaimed": [],
             "spring_added": [], "spring_removed": [], "spring_stamps": {},
             "spring_gone": [], "spring_over_mask": 0,
             "stale": 0,
             "accessories": 0, "hitboxsets": 0,
             "rebased": 0, "requantised": [], "root_turned": [], "reparented": [],
             "added_materials": [], "surplus": surplus_bones(m, arm_obj),
             "slots_moved": 0, "slots_gone": [],
             "renamed_anims": [], "unstamped_renames": [],
             "seqlist": {},
             "dup_models": dup_models, "dup_bones": dup_bones,
             "blind_bones": []}
    poses = read_bones(m, arm_obj, scale, scene["blind_bones"])
    if poses:
        # Translation decodes additively over the record's own `pos`, so a bind that only
        # moved carries every animation with it for nothing. A bind that TURNED does not:
        # a rotation channel is `int16 * rotscale` with no bind base at all, so an
        # animation nobody re-exported would keep pointing where the old bind put it.
        # Those are re-encoded here and not left for a later export -- one file answering
        # the same edit two ways, the re-exported animations following the bone and the
        # carried ones not, is the defect and an option would be a second way to produce it.
        scene["reparented"] = [(m.bones[k].name,
                                m.bones[v[3]].name if v[3] >= 0 else "nothing")
                               for k, v in sorted(poses.items()) if v[3] is not None]
        rebase = build_mod.set_bone_poses(d, poses)
        scene["bones"] = len(poses)
        scene["rebased"] = rebase["anims"]
        scene["requantised"] = rebase["widened"]
        scene["root_turned"] = [d.bones[k].name for k in rebase["root_turned"]]
    named, scene["slots_moved"], scene["slots_gone"] = read_materials(
        m, source, skin_family(m, arm_obj))
    for ref, name in sorted(named.items()):
        if 0 <= ref < len(d.textures) and d.textures[ref].name != name:
            d.textures[ref].name = name
            scene["materials"] += 1
    # After the rename loop, so a slot renamed onto an existing texture is a rename and
    # not an addition. `cdtexture` has already been applied to d.cdtextures above, so None
    # here leaves the list the scene asked for alone.
    have = {t.name for t in d.textures}
    for name in new_materials(m, source):
        if name in have:
            continue
        build_mod.add_material(d, name, None)
        have.add(name)
        scene["added_materials"].append(name)
    sp = apply_springbones(d, m, arm_obj)
    scene["springs"] = sp["changed"]
    for key in ("ends", "switched", "unclaimed", "added", "removed", "stamps",
                "gone", "over_mask"):
        scene["spring_" + key] = sp[key]
    scene["accessories"] = apply_accessories(d, m, arm_obj, scale)
    # Deleting every box empty writes numhitboxsets 0, which nothing else would say.
    scene["hitboxsets"] = len(d.hitboxsets)
    face = apply_face(d, m, arm_obj, scale)
    scene["face"] = face["changed"]
    scene["eye_radius_keys"] = face["radius_keys"]
    scene["eye_lid_targets"] = face["lid_targets"]
    scene["face_flexdescs"] = face["flexdescs"]
    # After the mesh pass above rather than beside it: an eyeball is the one thing that
    # names a material the meshes do not, and the `.vmt` and `.tth` writers read this list.
    for name in face["materials"]:
        if name not in scene["added_materials"]:
            scene["added_materials"].append(name)
    scene["flex"] = apply_flex(d, m, arm_obj)
    # Before the append, not after: apply_sequences refuses outright when the armature's
    # stash and the file disagree on how many sequences there are.
    anim_names = [r.name for r in d.anims]
    # Collected here and applied at the end: `apply_sequences` resolves every blend cell of
    # the stash by animation NAME, so renaming first makes it refuse the file it is about
    # to write. The records are held by object rather than by index because a drop below
    # renumbers the list.
    renames, scene["unstamped_renames"] = renamed_anims(anim_names, source, arm_obj)
    pending_renames = [(d.anims[i], anim_names[i], n) for i, n in sorted(renames.items())]
    scene["sequences"], scene["blends_out_of_range"], scene["seqlist"] = apply_sequences(
        d, m, arm_obj, anim_names,
        sequence_actions(anim_names, source, arm_obj, actions))

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

    # Last, so `drop` resolves against the name the file still carried and every blend cell
    # has already been written as an index. A renamed animation the same export dropped is
    # simply gone.
    live = {id(r) for r in d.anims}
    for rec, was, now in pending_renames:
        if id(rec) not in live:
            continue
        build_mod.rename_animation(d, d.anims.index(rec), now)
        scene["renamed_anims"].append((was, now))

    mesh = {"fields": tuple(mesh_fields), "verts": 0, "models": 0,
            "missing": [], "unsupported": [], "normals": 0, "rebuilt": [],
            "unskinned": 0, "renumbered": 0, "crowded": [], "blind_normals": 0,
            "rebuilt_uvs": 0, "rebuilt_added": 0, "rebuilt_deleted": 0,
            "flex_dropped": 0, "flex_emptied": 0, "uv_spare": [], "tangents": 0,
            "flexes": 0, "flex_records": 0, "flex_skipped": 0, "flex_refused": [],
            "stray_groups": [], "stale_stash": stale_stash(m, source),
            "stale_stamps": stale_stamps(m, source)}
    revised = {}
    if mesh_fields:
        cells, rebuild, mesh["missing"], mesh["unsupported"], mesh["normals"], \
            mesh["crowded"], mesh["blind_normals"], mesh["uv_spare"], donor_faces, \
            mesh["unskinned"], mesh["stray_groups"] = read_meshes(
                m, source, mesh_fields)
        if not cells and not rebuild and not mesh["missing"]:
            raise ValueError("no scene mesh belongs to %s" % os.path.basename(source))
        for (bi, mi), verts in sorted(cells.items()):
            rec = d.bodyparts[bi].kids[mi]
            mo = m.bodyparts[bi].models[mi]
            was_vb = rec.extra["verts"]
            rec.extra["verts"], n = mesh_mod.pack_model(mo, verts, mesh_fields, was_vb)
            # _must_rebuild sends every changed triangle set to the rebuild, so on this
            # path the donor's triangles are still the written vertices' own.
            tri = donor_faces.get((bi, mi))
            rec.extra["tangents"], t = build_mod.retangent_block(
                mo.filetype, was_vb, rec.extra["verts"], rec.extra.get("tangents") or b"",
                tri.elements() if tri else (), mo.quant_offset, mo.quant_scale)
            mesh["verts"] += n
            mesh["tangents"] += t
        mesh["models"] = len(cells)
        bone_index = {b.name: b.index for b in m.bones}
        for (bi, mi), obj in sorted(rebuild.items()):
            was = m.bodyparts[bi].models[mi].numvertices
            tri = donor_faces.get((bi, mi))
            faces, unskinned, kept, edits = rebuild_cell(
                d, obj, bi, mi, bone_index, mesh_fields,
                list(tri.elements()) if tri else None)
            for k, tris in enumerate(faces):
                revised[(bi, mi, k)] = tris
            now = sum(struct.unpack_from("<i", x.raw, 0x08)[0]
                      for x in d.bodyparts[bi].kids[mi].kids)
            mesh["rebuilt"].append((obj.name, was, now))
            mesh["unskinned"] += unskinned
            mesh["renumbered"] += 0 if kept else 1
            # Same meaning as the in-place path's, so they share the report line.
            mesh["normals"] += edits["normals"]
            mesh["blind_normals"] += edits["blind"]
            mesh["rebuilt_uvs"] += edits["uvs"]
            mesh["rebuilt_added"] += edits["added"]
            mesh["rebuilt_deleted"] += edits["deleted"]
            mesh["flex_dropped"] += edits.get("flex_dropped", 0)
            mesh["flex_emptied"] += edits.get("flex_emptied", 0)
            mesh["tangents"] += edits["tangents"]

    if write_flexes:
        # After the geometry, because add_flex bounds every key against the mesh's
        # numvertices as written and a rebuild has just changed it.
        found = mesh_objects(m, source)
        for (bi, mi, _bp, _mo) in mesh_mod.models_of(m):
            obj = found.get((bi, mi))
            if obj is None:
                continue
            k, r, skipped, why = shape_key_flexes(d, obj, bi, mi, scale)
            mesh["flexes"] += k
            mesh["flex_records"] += r
            mesh["flex_skipped"] += skipped
            if why:
                mesh["flex_refused"].append((obj.name, why))

    # After the geometry: a rest length is refitted against where the vertex now sits, and
    # a rebuilt model has just moved them.
    cloth_was = cloth_edits(m, source, d)
    cloth = apply_cloth(d, m, source, cloth_was) if write_cloth else None

    # The donor checksum is kept whether or not the .vtx is rewritten: the pair only
    # has to agree with each other, and the engine draws nothing when it does not.
    data = build_mod.emit(d, checksum=d.checksum)
    # A bone removal or an append moves no vertex and no triangle, so `revised` is empty --
    # and the .vtx still has to be rewritten, because every strip group's bone data is bound
    # to the .mdl's numbering, and a group without flag 0x02 carries the bone id per vertex.
    # `vtx_rebuild.revise` rebinds all of them off the model as written whether or not a cell
    # was named, so an empty face map is the whole of what either needs.
    touched = (revise_vtx(source, dest, data, revised, vtx_flavours)
               if (revised or removed or added_bones) else [])
    vtx = next((x for x in touched if x["flavour"] == "dx80"), None)
    # After the geometry pass, which writes whole .vtx files: the cut is two dwords over
    # the finished bytes and would otherwise be laid back over.
    # `revise` rewrites LOD 0 and leaves the lower ones the donor's own triangles, which
    # index by original vertex id -- valid while the numbering holds and meaningless once a
    # rebuild renumbers or a delete shifts every survivor after the hole. A lower LOD's ids
    # name the vertices a delete took, so they cannot be remapped either. The decimation is
    # authored and cannot be rebuilt from LOD 0, so the LODs go. The cut is the whole
    # file's and not the one model's: `numLODs` agrees between the file header and every
    # model header on all 8887 shipped files, and the Unofficial Patch's own cut models set
    # both.
    forced = bool(mesh["renumbered"] or mesh["rebuilt_deleted"]) and not cut_lods
    lods = (cut_vtx_lods(source, dest, [x["flavour"] for x in touched],
                         None if cut_lods else [x["flavour"] for x in touched])
            if (cut_lods or forced) else [])
    with open(dest, "wb") as f:
        f.write(data)
    return {"wrote": wrote, "added": added, "dropped": dropped, "unfitted": unfitted,
            "unkeepable": unkeepable, "bones": len(m.bones),
            "mesh": mesh, "scene": scene, "bytes": len(data), "was": len(m.d),
            "anims": len(m.anims), "sequences": len(d.seqs),
            "vtx": vtx, "removed": removed, "added_bones": added_bones,
            "vtx_more": [x for x in touched if x["flavour"] != "dx80"],
            "renamed": renamed,
            "model_name": model_name, "hull": hull, "cdtexture": cdtex,
            "boxes": (d.refit_count, len(d.seqs)) if hull is not None else None,
            "remodelled": sorted((v, k) for k, v in remodelled.items()),
            "includes": [r.name for r in d.includes],
            "stale": (stale_flavours(dest, [x["flavour"] for x in touched])
                      if (revised or removed or added_bones) else []),
            "cloth": cloth, "cloth_edits": cloth_was, "lods": lods,
            "lods_forced": forced,
            "phy": phy_report(source, dest, removed, renamed, bool(revised))}


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
