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

import array
import collections
import contextlib
import copy
import math
import os
import re
import struct
import types
import zlib

import bpy
import mathutils

from . import cloth as cloth_mod
from . import mdl as mdl_mod
from . import mdl_build as build_mod
from . import mdl_rebuild as rebuild_mod
from . import mdl_write as write_mod
from . import mesh_write as mesh_mod
from . import paths as paths_mod
from . import phy as phy_mod
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


class SceneRefused(build_mod.Refused, ValueError):
    """A refusal of what the scene holds: `Refused`, so both operators report it as one
    with no traceback, and ValueError, so a caller that catches one still does."""


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
        raise SceneRefused("action %r carries %s %r, which is none of %s"
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
        raise SceneRefused("armature has no bone %s (and %d more)"
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
        raise SceneRefused(
            "bone %r is parented to %r, which the file has no bone for -- parent it to "
            "one the file carries, or let the export add %r first"
            % (b.name, name, name))
    if now == b.parent:
        return None
    if now >= k:
        was = m.bones[b.parent].name if b.parent >= 0 else "nothing"
        raise SceneRefused(
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
        raise SceneRefused("armature has no bone %s (and %d more)"
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

    The export dialog's prediction, called before anything is written, where it is the list
    of bones the export is about to add. The export path itself has no use for it:
    `add_surplus_bones` places every one of them or raises naming the ones it could not, so
    asked afterwards this answers [] whatever happened. What the export reports is
    `add_surplus_bones`' own list of what it placed.
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
            raise SceneRefused("cannot place %s: no bone of it has a parent that resolves"
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
    out = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        par = obj.parent
        if par is not None and not same_file(par.get("vtmb_source"), source):
            continue
        gi = obj.get("vtmb_index")
        if gi is not None and 0 <= gi < len(flat):
            out.setdefault((flat[gi][0], flat[gi][1]), []).append(obj)
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
    """(copies, models no one object owns), out of the objects claiming one.

    A copy never writes the model it claims. `new_model_objects` appends one parented to
    the armature as a model of its own; any other reaches the file nowhere.
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


PLACED_TOL = 1e-6


@contextlib.contextmanager
def placed(objs, arm_obj):
    """Each mesh object's transform relative to the armature, baked into its vertices and
    shape keys for one export as Ctrl+A would, then the saved arrays written back.

    Every reader takes `co` in mesh space, and the file holds the mesh where the viewport
    draws it. Custom normals are stored relative to the faces, so a rotation turns them and
    a non-uniform scale bends them as Ctrl+A does. The importer's stashes are mesh space
    too and move with it: `NORMAL_ATTR` is what `_one_normal` writes for a vertex on no
    live face and for one within `NORMAL_EPS` of it, and `vtmb_orig_co` is what
    `identity_map` measures a duplicated vertex against. Yields one dict per object moved.
    """
    inv = arm_obj.matrix_world.inverted()
    todo = []
    for obj in objs:
        rel = inv @ obj.matrix_world
        if all(abs(rel[r][c] - (r == c)) <= PLACED_TOL for r in range(4) for c in range(4)):
            continue
        if rel.determinant() < 0:
            raise build_mod.Refused(
                "%s has a negative scale in Object mode, which would turn its faces inside "
                "out in the file -- clear it before exporting" % obj.name)
        users = [o.name for o in bpy.data.objects if o.data is obj.data and o is not obj]
        if users:
            raise build_mod.Refused(
                "%s is moved in Object mode and shares its mesh with %s, so writing it where "
                "it stands would move the other as well" % (obj.name, ", ".join(users)))
        lin = rel.to_3x3()
        s = rel.to_scale()
        todo.append((obj, rel, {
            "object": obj.name,
            "matrix": tuple(tuple(rel[r][c] for c in range(4)) for r in range(4)),
            "offset": tuple(rel.translation),
            "turned": any(abs(lin[r][c] - (r == c)) > PLACED_TOL
                          for r in range(3) for c in range(3)),
            "stretched": max(s) - min(s) > PLACED_TOL * max(s)}))
    saved = []
    try:
        for obj, rel, _row in todo:
            me = obj.data
            co = array.array("f", [0.0]) * (len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            keys = []
            for kb in me.shape_keys.key_blocks if me.shape_keys else ():
                a = array.array("f", [0.0]) * (len(kb.data) * 3)
                kb.data.foreach_get("co", a)
                keys.append(a)
            att = me.attributes.get(NORMAL_ATTR)
            nor = None
            if att is not None and att.data_type == "FLOAT_VECTOR":
                nor = array.array("f", [0.0]) * (len(att.data) * 3)
                att.data.foreach_get("vector", nor)
            home = obj.get("vtmb_orig_co")
            home = list(home) if home is not None else None
            saved.append((obj, me, co, keys, nor, home))
            me.transform(rel, shape_keys=True)
            if nor is not None:
                att.data.foreach_set("vector", _turned_normals(rel, nor))
            if home is not None:
                obj["vtmb_orig_co"] = [c for i in range(0, len(home) - 2, 3)
                                       for c in rel @ mathutils.Vector(home[i:i + 3])]
            me.update()
        yield [row for _obj, _rel, row in todo]
    finally:
        for obj, me, co, keys, nor, home in saved:
            me.vertices.foreach_set("co", co)
            for kb, a in zip(me.shape_keys.key_blocks if me.shape_keys else (), keys):
                kb.data.foreach_set("co", a)
            if nor is not None:
                me.attributes[NORMAL_ATTR].data.foreach_set("vector", nor)
            if home is not None:
                obj["vtmb_orig_co"] = home
            me.update()


def _turned_normals(rel, flat):
    """`flat` normals through the inverse transpose of `rel`'s 3x3, each kept at its own
    length: a zero stash stays zero and a unit one unit, whatever the scale."""
    nm = rel.to_3x3().inverted_safe().transposed()
    out = array.array("f", flat)
    for i in range(0, len(out) - 2, 3):
        n = mathutils.Vector(flat[i:i + 3])
        v = nm @ n
        if v.length:
            v *= n.length / v.length
        out[i:i + 3] = array.array("f", v)
    return out


def new_model_objects(m, source, arm_obj):
    """The armature's mesh children that claim no model of `source`, or are a Shift+D copy
    of one that does, by name."""
    if arm_obj is None or not same_file(arm_obj.get("vtmb_source"), source):
        return []
    claimed, copies = set(), set()
    for objs in _model_claims(m, source).values():
        claimed |= set(id(o) for o in objs)
        got = _one_model(objs) if len(objs) > 1 else None
        if got is not None:
            copies |= set(id(o) for o in objs if o is not got)
    return sorted((o for o in arm_obj.children
                   if o.type == "MESH" and not o.get("vtmb_phy_hull")
                   and (id(o) not in claimed or id(o) in copies)),
                  key=lambda o: o.name)


def _skin_ref(d, tex, family):
    """The skinref that draws texture `tex` in skin family `family`, appended if none does.

    One that draws it in every family is preferred, so another family's skin does not
    repaint the new mesh."""
    if not d.skin:
        return tex
    row = d.skin[family] if 0 <= family < len(d.skin) else d.skin[0]
    refs = [r for r, t in enumerate(row) if t == tex]
    same = [r for r in refs if all(r < len(x) and x[r] == tex for x in d.skin)]
    if same or refs:
        return (same or refs)[0]
    for x in d.skin:
        x.append(tex)
    return len(row) - 1


def append_models(d, objs, bone_index, family, added, scale=1.0):
    """Each object appended as a bodypart of one model, read as the no-donor path reads it.

    ([row], {(bodypart, 0, mesh): triangles}, [(object name, why not appended)])."""
    from . import blender_scratch as scratch_mod
    rows, faces, skipped = [], {}, []
    for obj in objs:
        try:
            runs, unskinned = scratch_mod.split_mesh(obj, bone_index, scale)[:2]
            cloth = scratch_mod.cloth_params(obj) is not None
        except scratch_mod.Refused as e:
            skipped.append((obj.name, str(e)))
            continue
        mats = obj.data.materials
        meshes, slots = [], []
        for slot, verts, tris in runs:
            name = mats[slot].name if slot < len(mats) and mats[slot] else obj.name
            ref = _skin_ref(d, _material_index(d, name, added), family)
            meshes.append((ref, verts, tris))
            slots.append((slot, ref, len(verts)))
        bi = build_mod.add_model(d, meshes, bodypart=obj.name, model=obj.name + ".smd")
        d.faces.pop()
        for k, (_r, _v, tris) in enumerate(meshes):
            faces[(bi, 0, k)] = list(tris)
        unwritten = []
        keys = obj.data.shape_keys
        if keys is not None and len(keys.key_blocks) > 1:
            n = len(keys.key_blocks) - 1
            unwritten.append("%d shape key%s" % (n, "" if n == 1 else "s"))
        if cloth:
            unwritten.append("its cloth")
        rows.append({"object": obj, "name": obj.name, "bodypart": bi, "slots": slots,
                     "verts": sum(len(v) for _r, v, _t in meshes),
                     "tris": sum(len(t) for _r, _v, t in meshes),
                     "unskinned": unskinned, "unwritten": unwritten, "stamped": False})
    return rows, faces, skipped


def stamp_appended(d, rows, arm_obj=None):
    """The import's own stamps on each appended object, so the next export rewrites its
    model rather than appending another."""
    eyes = eyeball_objects(arm_obj) if arm_obj is not None else {}
    for row in rows:
        obj, me = row["object"], row["object"].data
        obj["vtmb_index"] = sum(len(bp.kids) for bp in d.bodyparts[:row["bodypart"]])
        obj["vtmb_bodypart"] = obj.name
        obj["vtmb_model"] = obj.name + ".smd"
        obj["vtmb_filetype"] = 0
        obj["vtmb_appended"] = True
        # A copy keeps the import's, which numbers the vertices its `vtmb_orig` tags name.
        if obj.get("vtmb_meshes") is None:
            obj["vtmb_meshes"] = [[slot, n] for slot, _r, n in row["slots"]]
        stamp_slots(obj)
        # The model was written with no eyeball record, so an eye stamp no empty on it
        # claims is a copy's, naming its original's eyeball, which this model has not got.
        mine = {int(o["vtmb_eyeball"]) for o in eyes.get(obj.name + ".smd", ())}
        eye = [[int(a), int(b)] for a, b in obj.get("vtmb_eye_slots") or () if int(b) in mine]
        if eye:
            obj["vtmb_eye_slots"] = eye
        else:
            obj.pop("vtmb_eye_slots", None)
        refs = dict((slot, ref) for slot, ref, _n in row["slots"])
        # The skin-family switch reads one per slot, and an empty slot has none.
        if all(s in refs for s in range(len(me.materials))):
            obj["vtmb_skinrefs"] = [refs[s] for s in range(len(me.materials))]
        obj["vtmb_uv_layer"] = uv_layer_of(obj)[0].name
        me["vtmb_model_name"] = obj.name + ".smd"
        me["vtmb_model_label"] = me.name
        row["stamped"] = True
    if arm_obj is not None and rows:
        widen_skin_table(d, arm_obj)


def widen_skin_table(d, arm_obj):
    """The columns an append added to the file's skin table, added to the armature's.

    A slot stamped with a position past the armature's table was otherwise read by the
    family switch as a record index, which put another record's material on the new mesh
    and the export then renamed its record to match. `_skin_ref` appends each column
    drawing one record in every family, so row 0 names it.
    """
    table, names, _nfile = skin_table(arm_obj)
    if not table or not d.skin or len(d.skin[0]) <= len(table[0]):
        return 0
    if "vtmb_skin_textures" not in arm_obj:
        arm_obj["vtmb_skin_textures"] = len(names)
    w = len(table[0])
    for c in range(w, len(d.skin[0])):
        name = d.textures[d.skin[0][c]].name
        if name not in names:
            names.append(name)
        for row in table:
            row.append(names.index(name))
    arm_obj["vtmb_skin_table"] = [r for row in table for r in row]
    arm_obj["vtmb_skin_refs"] = len(table[0])
    arm_obj["vtmb_skin_materials"] = names
    return len(d.skin[0]) - w


def renamed_models(m, source):
    """{(bodypart, model): the name Blender now gives it} for the mesh datablocks renamed.

    The import puts `mstudiomodel_t.name` on the datablock and stashes both what the file
    said and what Blender actually assigned, because Blender caps a datablock name at 63
    bytes and suffixes a duplicate: 3 of the corpus's 4567 model names are over 63 and 14
    repeat inside one file. Comparing against what was assigned is therefore the only way
    to tell a rename from Blender's own bookkeeping, and a datablock with no stash, which
    is a mesh the user put on the object, is left alone rather than guessed at.
    """
    out = {}
    for key, obj in mesh_objects(m, source).items():
        me = obj.data
        was = me.get("vtmb_model_label")
        if was is not None and me.name != was:
            out[key] = me.name
    return out


def renamed_bodyparts(m, source):
    """{bodypart index: the name the scene now gives it} for the bodyparts renamed.

    The import stamps `mstudiobodyparts_t`'s name as `vtmb_bodypart` on each object of the
    bodypart, and the panel renames them together. Two objects of one bodypart giving it two
    new names cannot both be written.
    """
    got = {}
    for (bi, _mi), obj in mesh_objects(m, source).items():
        name = obj.get("vtmb_bodypart")
        if name is None:
            continue
        if str(name) != m.bodyparts[bi].name:
            got.setdefault(bi, {}).setdefault(str(name), []).append(obj.name)
    out = {}
    for bi, names in sorted(got.items()):
        if len(names) > 1:
            raise build_mod.Refused(
                "bodypart %r is renamed %s; a bodypart has one name, and the Bodypart "
                "field sets it on every object of the bodypart at once"
                % (m.bodyparts[bi].name,
                   " and ".join("%r by %s" % (n, ", ".join(sorted(objs)))
                                for n, objs in sorted(names.items()))))
        out[bi] = next(iter(names))
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
        raise SceneRefused("%s: %d vertices carry more than one UV (first is vertex %d), "
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


def skin_counted(model):
    """Whether a vertex no bone drives on this model is reported.

    Not on a quantised one: its record has no weight field and the engine binds each of its
    vertices to bone 0 at full weight (StudioRender 0x2c018720), which is what the split
    writes for a vertex in no group. A scene imported before the importer put them in bone
    0's group holds none in any."""
    return model.filetype not in mdl_mod.QUANT_MAX


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
        raise SceneRefused("%s: %d vertices in the scene, %d in %s -- a mesh write cannot "
                           "change the count"
                           % (obj.name, len(me.vertices), model.numvertices, model.name))
    n = model.numvertices
    uvs = normals = [None] * n
    edited = blind = unskinned = 0
    if "uvs" in fields:
        uv_layer = uv_layer_of(obj)[0]
        if uv_layer is None:
            raise SceneRefused("%s has no UV layer" % obj.name)
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


def _donor_faces(source, m, meshes=None):
    """{(bodypart, model): the donor's own LOD-0 triangles}, in the file's winding.

    Triangles live in the .vtx alone -- `mstudiomesh_t` carries no count -- so a face
    added, deleted or reversed reaches the file by no other route and the .mdl offers
    nothing to compare on. Corners come out model-local, which is what a Blender object's
    vertices are indexed by, by the same mapping `blender_import` reads them in through.

    Empty where no .vtx sits beside the source, which leaves the caller comparing what it
    compared before rather than rebuilding every model on a missing file.

    `meshes`, if given, is filled with {(bodypart, model): {triangle: {file mesh index}}},
    which is what says a face has moved to another material slot.
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
        tris = [tuple(ids[x] + base for x in t) for t in reader.triangles(sg)]
        per.setdefault(sg.model, []).extend(tris)
        if meshes is not None:
            at = meshes.setdefault((models[sg.model][0], models[sg.model][1]), {})
            for t in tris:
                at.setdefault(vtxr_mod._canon(t), set()).add(sg.mesh)
    for gi, tris in per.items():
        bi, mi = models[gi][0], models[gi][1]
        out[(bi, mi)] = _canon_faces(tris)
    return out


def moved_faces(obj, donor_meshes):
    """How many triangles sit on a material slot other than the one their file mesh speaks
    through now, by `vtmb_meshes` and `slot_moves`. A triangle the donor carries on two
    meshes, or on a mesh whose slot is gone, decides nothing."""
    stash = obj.get("vtmb_meshes")
    if not stash or not donor_meshes:
        return 0
    moves = slot_moves(obj)[0]
    me = obj.data
    me.calc_loop_triangles()
    n = 0
    for tri in me.loop_triangles:
        js = donor_meshes.get(vtxr_mod._canon(tuple(tri.vertices)[::-1]))
        if not js or len(js) != 1:
            continue
        slot = slot_of(stash, moves, next(iter(js)))
        if slot is not None and slot != tri.material_index:
            n += 1
    return n


def _scene_faces(me):
    """The scene's triangles in the file's own winding, model-local.

    `wound` reverses in both directions -- the format winds a triangle against its outward
    normal and Blender winds it with -- so the reverse here is what puts a Blender face
    back in the donor's terms.
    """
    me.calc_loop_triangles()
    return _canon_faces(tuple(tri.vertices)[::-1] for tri in me.loop_triangles)


def _must_rebuild(obj, model, fields, donor_faces=None, box=None, donor_meshes=None):
    """Whether this object has outgrown the in-place path.

    In place patches fields index for index; the split path rewrites the whole model and
    is what a changed vertex count, a UV seam or a changed face needs. A seam is a vertex
    whose corners disagree, which the format spells by duplicating the vertex -- so it is
    a count change wearing another hat.

    `donor_faces` is what the .vtx said, and None where no .vtx was found. Comparing it is
    what stops a face deleted, added between existing vertices, or flipped from writing a
    file identical to the one it was edited from: none of the three moves the vertex count
    or a UV, so every other test here passes on an edit the file never took.

    `box` is the quantisation box `rebased_box` moved with the object, if it did.
    """
    me = obj.data
    # An appended model's vertices are in `split_mesh` order, so patching it index for
    # index would write each Blender vertex onto another file vertex.
    if obj.get("vtmb_appended"):
        return True
    if len(me.vertices) != model.numvertices:
        return True
    if donor_faces is not None and _scene_faces(me) != donor_faces:
        return True
    # In place keeps each triangle in the strip group it came from, so a face moved to
    # another slot would keep its old material.
    if moved_faces(obj, donor_meshes):
        return True
    if "positions" in fields and outside_box(me, model, box):
        return True
    if "uvs" not in fields:
        return False
    uv = uv_layer_of(obj)[0]
    if uv is None:
        return False
    per = _per_vertex(me, model.numvertices, lambda l: tuple(uv.data[l.index].uv))
    return any(vs and any(max(abs(a - b) for a, b in zip(vs[0], v)) > UV_TOL for v in vs)
               for vs in per)


def outside_box(me, model, box=None):
    """How many vertices stand where a quantised model's positions cannot reach.

    `pack_model` clamps each coordinate into the box `quant_offset`/`quant_scale` span, and
    every shipped quantised model has a vertex on each wall (4324 of 4324 model-axes), so
    any outward move would land on the wall. The split writes 44-byte float records.
    `box` is (offset, scale) in place of the model's own, as `rebased_box` gives.
    """
    if model.filetype not in mdl_mod.QUANT_MAX:
        return 0
    lim = mdl_mod.QUANT_MAX[model.filetype]
    norm = mdl_mod.QUANT_NORM[model.filetype]
    off, sc = box or (model.quant_offset, model.quant_scale)
    co = [0.0] * (len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return sum(1 for i in range(len(me.vertices))
               if any(not 0 <= mesh_mod.quantise(co[i * 3 + c], off[c], sc[c], norm) <= lim
                      for c in range(3)))


def off_bone0(obj, model, bone_index):
    """How many vertices of a quantised model are weighted to a bone other than bone 0.

    A filetype 1 or 2 record has no bone or weight field and StudioRender `0x2c018720`
    binds every such vertex to bone 0 at 255, so the import puts each one in bone 0's group
    at 1.0 and any other weight past `WEIGHT_FLOOR` is an edit only filetype 0 can hold. A
    group naming no bone is `stray_groups`' to report and counts nothing here.
    """
    if model.filetype not in mdl_mod.QUANT_MAX:
        return 0
    other = {g.index for g in obj.vertex_groups if bone_index.get(g.name) not in (None, 0)}
    if not other:
        return 0
    return sum(1 for v in obj.data.vertices
               if any(g.group in other and g.weight > mesh_mod.WEIGHT_FLOOR
                      for g in v.groups))


HDR_VECTORS = (("vtmb_eyeposition", mdl_mod.HDR_EYEPOSITION),
               ("vtmb_hull_min", mdl_mod.HDR_HULL_MIN),
               ("vtmb_hull_max", mdl_mod.HDR_HULL_MAX))


_SOURCE_VECTORS = {}


def source_vectors(path):
    """{header offset: (x, y, z)} off the model at `path`, in file units, or {} when nothing
    readable as a v2531 model is there. Cached on the path, its mtime and its size."""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    got = _SOURCE_VECTORS.get(path)
    if got is not None and got[0] == stamp:
        return got[1]
    end = mdl_mod.HDR_HULL_MAX + 12
    try:
        with open(path, "rb") as f:
            head = f.read(end)
    except OSError:
        return {}
    out = {}
    if (len(head) == end and head[:4] == mdl_mod.MAGIC
            and struct.unpack_from("<i", head, 4)[0] == mdl_mod.VERSION):
        out = {off: struct.unpack_from("<3f", head, off) for _key, off in HDR_VECTORS}
    _SOURCE_VECTORS[path] = (stamp, out)
    return out


def header_vectors(arm_obj, scale=1.0):
    """{header offset: (x, y, z)} in file units, for each of the three the armature carries.

    A key is absent on a .blend saved before the import stamped them. It is then read off
    the armature's `vtmb_source`, which is what the panel shows for it, so editing one
    component of a vector keeps the file's other two. With no readable source either -- an
    armature no import made -- it is left out, and the donor path keeps the file's and the
    no-donor path `new`'s.
    """
    out, src = {}, None
    for key, off in HDR_VECTORS:
        v = arm_obj.get(key)
        if v is None:
            if src is None:
                src = source_vectors(arm_obj.get("vtmb_source"))
            if off in src:
                out[off] = src[off]
            continue
        v = [float(c) for c in v]
        if len(v) != 3:
            raise build_mod.Refused("%s holds %d numbers where a position takes 3"
                                    % (key, len(v)))
        out[off] = tuple(c / scale for c in v)
    return out


def inverted_axes(d):
    """The axes on which the written hull's min is above its max. 0 of 4445 shipped."""
    lo = struct.unpack_from("<3f", d.hdr, mdl_mod.HDR_HULL_MIN)
    hi = struct.unpack_from("<3f", d.hdr, mdl_mod.HDR_HULL_MAX)
    return "".join("xyz"[c] for c in range(3) if lo[c] > hi[c])


def rebased_box(model, place):
    """(offset, scale) of the quantisation box a quantised model moved by `placed` takes, or
    None where the move has no such box.

    A vertex is `offset + q * norm * scale`, so moved by a uniform scale `s` and a
    translation `T` it is the same `q` in the box (s * offset + T, s * scale): the vertex
    records, the normal indices and the tangents stand, and only the box moves. A rotation
    or a stretch turns the normals, and an index into the renderer's table cannot be turned
    without moving it up to 10.5 degrees, so that model is rebuilt as filetype 0. Rounded
    to f32, which is what the record holds and what the quantise divides by.
    """
    if place is None or model.filetype not in mdl_mod.QUANT_MAX:
        return None
    s = place[0][0]
    if s <= 0.0 or any(abs(place[r][c] - (s if r == c else 0.0)) > PLACED_TOL * s
                       for r in range(3) for c in range(3)):
        return None
    f32 = struct.Struct("<f")
    return (tuple(f32.unpack(f32.pack(s * model.quant_offset[c] + place[c][3]))[0]
                  for c in range(3)),
            tuple(f32.unpack(f32.pack(s * model.quant_scale[c]))[0] for c in range(3)))


def rebuild_cell(d, obj, bi, mi, bone_index, fields=("uvs", "normals"),
                 donor_tris=None, place=None, empty=()):
    """One model's geometry replaced from the scene, its per-mesh triangles and its edits.

    Refuses a renumbering the file cannot absorb rather than dropping what it would break:
    `split_mesh` reports `kept` 0 when the original partition could not be recovered, and
    names which of its ten conditions forced that. A flex payload or a cloth binding keyed
    to the old numbering would be carried onto the wrong vertices; with neither of those
    present renumbering costs nothing.

    A delete is not that case. The survivors keep the file's own order, so `split_mesh`
    reports the donor vertex each written one came from and the flex payloads are
    renumbered through it -- a record naming the deleted vertex is dropped and the rest
    keep the vertex they named. Where no deleted vertex was a particle's anchor or bound in
    row 0, the row-0 cloth objects are carried as they stand and come back as
    {"carried": True, ...}. Otherwise the binding is regenerated for the particles whose
    anchor vertex survived, `mstudiocloth_t.vertindex` being one ushort per particle naming
    a model vertex that has to exist; what that drops comes back in `edits["cloth"]`. An
    addition alone keeps every particle and renumbers `vertindex` past it, which comes back
    as {"renumbered": n}, unless an added vertex shares triangles with one cloth object's
    particles: then the object is regenerated with it as a particle, "promoted" counting
    them. Added vertices touching cloth and left rigid come back as "rigid" and "why".

    A model carrying cloth also gets `edits["cloth_map"]`, the written numbering
    `cloth_scene` reads the scene through.

    `fields` is what the checkboxes asked for. Without it a face added or a winding
    reversed changes Blender's corner normal on every vertex it touches, and the split
    would take that as an edit and write it -- on an export that never asked for normals.

    `empty` names file meshes written with no vertex, for the caller to drop.
    """
    # Deferred: blender_scratch imports this module, so a top-level import is a cycle.
    from . import blender_scratch as scratch_mod
    carry = []
    runs, unskinned, kept, why, edits, origins = scratch_mod.split_mesh(
        obj, bone_index, 1.0, fields, carry, empty)
    mr = d.bodyparts[bi].kids[mi]
    if not kept:
        carries = []
        if any(x.kids for x in mr.kids):
            carries.append("morph targets, which are keyed by vertex")
        if mr.extra.get("cloth"):
            carries.append("a cloth binding, which is one entry per vertex per row")
        if carries:
            raise build_mod.Refused(
                "%s: the file's own vertex numbering could not be recovered -- %s -- and "
                "this model carries %s" % (obj.name, why, " and ".join(carries)))
    # `origins` is one list per run and the written model-local numbering is the runs
    # concatenated in that order, which is the order `replace_model` is handed.
    vertex = [vi for run in origins for vi in run]
    faces, edits["tangents"], flex, edits["cloth"] = build_mod.replace_model(
        d, bi, mi, [(None, v, f) for _slot, v, f in runs], donor_tris=donor_tris,
        carry=carry if kept else None, flips=cloth_flip_bits(obj, vertex), place=place)
    edits["flex_dropped"], edits["flex_emptied"] = flex
    if d.bodyparts[bi].kids[mi].extra.get("cloth"):
        # The cloth object now names the WRITTEN numbering, which is Blender's only where
        # nothing was added or deleted, so `cloth_edits` reads the scene through this.
        outcome = edits["cloth"]
        edits["cloth_map"] = {
            "vertex": vertex, "runs": [len(v) for _s, v, _f in runs],
            "written": [v[0] for _s, verts, _f in runs for v in verts],
            "carry": carry if kept else None,
            "regenerated": isinstance(outcome, dict) and "renumbered" not in outcome}
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


def key_flex(name, names, descs, targets):
    """(flexdesc, target or None) for the shape key called `name`, off the import's stamps.

    Matched by the name the import assigned rather than by position: a key renamed or
    reordered in Blender must not take another flex's numbers. A key the file did not carry
    gets studiomdl's default target and a flexdesc of its own name.
    """
    fd, t = name, None
    if name in names:
        at = names.index(name)
        if at < len(targets):
            t = tuple(float(c) for c in targets[at])
        if descs is not None and at < len(descs):
            fd = str(descs[at])
    return fd, t


def moved_vertices(co, base):
    return [i for i in range(len(co))
            if max(abs(co[i][c] - base[i][c]) for c in range(3)) > 1e-6]


def key_deltas(me, co, base, base_n, moved, scale=1.0):
    """[(Blender vertex, position delta, normal delta)] over `moved`, in file units.

    A vertex under half a step of both bytes is left out: its record would say nothing.
    """
    morph_n = _vertex_normals(me, co)
    out = []
    for vi in moved:
        delta = tuple((co[vi][c] - base[vi][c]) / scale for c in range(3))
        ndelta = tuple(morph_n[vi][c] - base_n[vi][c] for c in range(3))
        if (sum(c * c for c in delta) ** 0.5 < FLEX_POS_STEP / 2.0
                and sum(c * c for c in ndelta) ** 0.5 < FLEX_NRM_STEP / 2.0):
            continue
        out.append((vi, delta, ndelta))
    return out


def shape_key_flexes(d, obj, bi, mi, scale=1.0):
    """Rebuild one model's flexes from its shape keys. (keys, records, skipped, why, two).

    The vertanim key is mesh-local and the reader adds it to the mesh's own first vertex,
    so the file's partition has to be recoverable before a key can be written at all --
    `original_runs` is what recovers it, and its members are in file order, member k of
    run r being mesh r's vertex k. A vertex the edit added is in no run and its delta is
    counted as skipped rather than guessed at.

    The file's own flexes go first. A key that still moves every vertex by what its
    records say is unedited, and those records go back byte for byte, normal deltas and
    all. Any other key is written as the scene's morph entire and not a patch on what the
    file said: a shape key holds coordinates only, so the normal delta is recomputed from
    the morphed geometry and the file's own is gone.

    `two` names the keys carried whole. `gone` names each of the file's (flexdesc,
    target) records no key claims any more, which go unwritten, and `unkeyed` is how many
    records a model with no shape key at all keeps as the file had them. Keys an import
    before 2026-10-15 made raise Refused (`prestamp_flexes`).
    """
    from . import blender_scratch as scratch_mod
    me = obj.data
    keys = me.shape_keys
    two = {"carried": [], "gone": [], "unkeyed": 0}
    if keys is None or len(keys.key_blocks) < 2:
        # No key is read as no edit, so deleting every key cannot remove a flex.
        two["unkeyed"] = sum(len(x.kids) for x in d.bodyparts[bi].kids[mi].kids)
        return 0, 0, 0, None, two
    if prestamp_flexes(obj):
        raise build_mod.Refused(_summed_keys(obj.name))
    runs = scratch_mod.original_runs(obj, me)
    if runs is None:
        why = []
        scratch_mod.original_runs(obj, me, why)
        return 0, 0, 0, (why[0] if why else "the file's partition was not recoverable"), two
    me.calc_loop_triangles()
    local = {}
    for k, (_slot, members) in enumerate(runs):
        for j, vi in enumerate(members):
            local[vi] = (k, j)

    blocks = list(keys.key_blocks)
    base = [tuple(v.co) for v in blocks[0].data]
    base_n = _vertex_normals(me, base)
    targets = obj.get("vtmb_flex_targets") or []
    names = list(obj.get("vtmb_flex_names") or [])
    descs = obj.get("vtmb_flex_descs")
    # Read before the clear: the file's records are already on the written numbering.
    held = build_mod.flex_groups(d, bi, mi)
    meshes = d.bodyparts[bi].kids[mi].kids
    # Each record's place in its mesh, so a key the file carried goes back where the file
    # had it. Key order differs from the file's on 71 of the 195 flexed models.
    place = {id(rec): j for mesh in meshes for j, rec in enumerate(mesh.kids)}
    rank = {}
    claimed = set()
    build_mod.clear_flexes(d, bi, mi)
    nkeys = nrec = skipped = nnew = 0
    for kb in blocks[1:]:
        co = [tuple(v.co) for v in kb.data]
        moved = moved_vertices(co, base)
        fd, t = key_flex(kb.name, names, descs, targets)
        # A key with no stamp claims no file record: it is one the user made.
        group = (held.get((fd, build_mod.flex_target_key(t)))
                 if descs is not None and t is not None else None)
        # Before the no-move test: the two eyelid records of a pair can cancel exactly, and
        # a key holding their sum then moves nothing.
        slot = {}
        if group is not None:
            claimed.add((fd, build_mod.flex_target_key(t)))
            recs, total = group
            for k, rec in recs:
                slot.setdefault(k, place[id(rec)])
            if _is_sum(co, base, local, total, scale):
                for k, rec in recs:
                    meshes[k].kids.append(rec)
                    rank[id(rec)] = (0, place[id(rec)])
                nkeys += 1
                nrec += sum(struct.unpack_from("<i", rec.raw, 0x14)[0] & 0xFFFF
                            for _k, rec in recs)
                two["carried"].append(kb.name)
                continue
        if not moved:
            continue
        per_mesh = {}
        for vi, delta, ndelta in key_deltas(me, co, base, base_n, moved, scale):
            at = local.get(vi)
            if at is None:
                skipped += 1
                continue
            k, j = at
            per_mesh.setdefault(k, []).append((j, delta, ndelta))
        if not per_mesh:
            continue
        nkeys += 1
        nnew += not slot
        for k in sorted(per_mesh):
            recs = sorted(per_mesh[k])
            build_mod.add_flex(d, bi, mi, k, fd, recs, t)
            rank[id(meshes[k].kids[-1])] = (0, slot[k]) if k in slot else (1, nnew)
            nrec += len(recs)
    _file_order(meshes, rank)
    two["gone"] = _unclaimed(held, claimed, names, descs, targets)
    return nkeys, nrec, skipped, None, two


def prestamp_flexes(obj):
    """Whether an import before `vtmb_flex_descs` (2026-10-15) made this object's keys. Every
    other import writes both, `vtmb_flex_names` having arrived with the flex import (007bc19)."""
    return obj.get("vtmb_flex_names") is not None and obj.get("vtmb_flex_descs") is None


def _summed_keys(name):
    return ("%s was imported before 2026-10-15, when a morph the file draws in steps, each "
            "over its own flex weight range, was summed into one shape key, and nothing in "
            "the scene says which keys those are. Written as one record the game would draw "
            "each in one step. Re-import the model, which makes one key per step" % name)


def _unclaimed(held, claimed, names, descs, targets):
    """Each file record group no key claimed, as the key name the import gave it, its
    flexdesc, and which of that flexdesc's steps it is: [(key, flexdesc, k, n)].

    A deleted key and a renamed one both land here: the stamps pair a key by its name.
    """
    steps = {}
    for fd, t in held:
        steps.setdefault(fd, []).append(t)
    out = []
    for fd, t in held:
        if (fd, t) in claimed:
            continue
        key = fd
        for at, name in enumerate(names if descs is not None else ()):
            if (at < len(descs) and str(descs[at]) == fd and at < len(targets)
                    and build_mod.flex_target_key(
                        tuple(float(c) for c in targets[at])) == t):
                key = name
                break
        out.append((key, fd, steps[fd].index(t) + 1, len(steps[fd])))
    return out


def _file_order(meshes, rank):
    """Each mesh's flexes back in the file's order, a key the file did not carry last."""
    for mesh in meshes:
        mesh.kids.sort(key=lambda rec: rank[id(rec)])


def _is_sum(co, base, local, total, scale, tol=1e-3):
    """Whether a shape key still moves every vertex by `total`, the summed records the
    import built it from. A vertex the edit added has no entry and must not move."""
    for vi in range(len(co)):
        want = total.get(local.get(vi), (0.0, 0.0, 0.0))
        for c in range(3):
            if abs((co[vi][c] - base[vi][c]) / scale - want[c]) > tol:
                return False
    return True


# PR #2 corrected the packed position on 2026-08-20. A `.blend` saved by an import older
# than that holds every filetype-2 model 255x oversized, and `vtmb_orig_co` is wrong the same
# way, so the export's own before/after comparison cannot see it. Filetypes 0 and 1 decode
# the same today as they did then.
STALE_DECODE = {2: (0, 2, 0)}


# Both landed while `bl_info` read 0.2.49, so only their absence dates an import. A datablock
# with no `vtmb_model_label` under an object carrying both is one the user put there.
OLD_STAMPS = (
    ("vtmb_slot_mats", "which material sat in each slot, so a slot reordered or deleted "
                       "since cannot be told from one that was not"),
    ("vtmb_uv_layer", "which UV layer is the file's, so a renamed first layer cannot be told "
                      "from a deleted one"),
)


def _version(obj):
    got = obj.get("vtmb_addon_version")
    return tuple(int(x) for x in got) if got is not None else None


def old_imports(m, source, arm_obj):
    """[(name, what it lacks)] for each part of the scene an import older than a stamp the
    export reads made. Nothing tells an edit there from what that import left, so
    `export_actions` raises Refused on any of it."""
    flat = mesh_mod.models_of(m)
    want = {(bp.name, mo.name) for _bi, _mi, bp, mo in flat}
    out = []
    for obj in bpy.data.objects:
        par = obj.parent
        if (obj.type == "MESH" and obj.get("vtmb_index") is None
                and (par is None or same_file(par.get("vtmb_source"), source))
                and (obj.get("vtmb_bodypart"), obj.get("vtmb_model")) in want):
            out.append((obj.name, "carries vtmb_bodypart and vtmb_model and no vtmb_index"))
    models = {(bi, mi): mo for bi, mi, _bp, mo in flat}
    seen = set()
    for key, obj in sorted(mesh_objects(m, source).items()):
        if obj.name in seen:
            continue
        seen.add(obj.name)
        mo, got = models[key], _version(obj)
        floor = STALE_DECODE.get(mo.filetype)
        if floor is not None and (got is None or got < floor):
            out.append((obj.name, "holds a filetype-%d model and was imported before the "
                                  "packed-position fix of 2026-08-20, which left it and its "
                                  "stash 255x oversized" % mo.filetype))
        for k, why in OLD_STAMPS:
            if obj.get(k) is None:
                out.append((obj.name, "has no %s, which says %s" % (k, why)))
        if mo.cloths and "vtmb_cloth" not in obj:
            out.append((obj.name, "has no vtmb_cloth, which says whether its Cloth box is "
                                  "ticked"))
        if mo.cloths and (got is None or got < CLOTH_PIN_VERSION):
            out.append((obj.name, "was imported by addon %s, whose cloth pin group came from "
                                  "the mesh binding and pins fewer particles than the file"
                                  % (".".join(str(x) for x in got) if got
                                     else "0.2.49 or older")))
    names = [a.name for a in m.anims]
    for i, act in sorted(match_indices(names, source, arm_obj)[0].items()):
        if act.name != names[i] and act.get("vtmb_anim_name") is None:
            out.append((act.name, "writes animation %r and has no vtmb_anim_name, which tells "
                                  "a rename from Blender's duplicate suffix" % names[i]))
    if arm_obj is not None:
        bare = [x for x in arm_obj.get("vtmb_sequences") or ()
                if not x.get("new") and x.get("src") is None]
        if bare:
            out.append((arm_obj.name, "stashes %d sequence%s with no src, which says which "
                                      "file sequence each is"
                        % (len(bare), "" if len(bare) == 1 else "s")))
    return out


def _old_import_text(old):
    said = "; ".join("%s %s" % x for x in old[:6])
    if len(old) > 6:
        said += "; and %d more" % (len(old) - 6)
    return ("an import older than what this export reads made part of this scene: %s. "
            "Nothing tells an edit there from what that import left. Re-import the model"
            % said)


def read_meshes(m, source, fields, boxes=None, blank=None):
    """{(bodypart, model): vertices} for every model of `m` the scene supplies, plus the
    models it does not, the fields the file cannot carry, the objects holding a vertex whose
    fifth bone group no record can hold, the vertices whose normal moved by more than the
    round trip's own error and less than `NORMAL_EPS`, which are written from the stash and
    so lose the edit, the UV layers no model wrote, the donor's own triangles, the
    vertices no bone drives any more, which go to bone 0 at full weight, and the vertex
    groups that skin something and name no bone.

    `blank`, a list where the scene's import made an object for every model with a
    vertex, collects (bodypart, model, name) for each such model whose object is gone
    instead of naming it missing."""
    found = mesh_objects(m, source)
    bone_index = {b.name: b.index for b in m.bones}
    donor_meshes = {}
    donor_faces = _donor_faces(source, m, donor_meshes)
    edits, rebuild, missing, unsupported, renormals = {}, {}, [], set(), 0
    blind_normals = unskinned = 0
    crowded, uv_spare, stray = [], [], []
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        ok, no = mesh_mod.supported(mo.filetype, fields)
        obj = found.get((bi, mi))
        if obj is None:
            # The import makes no object for a model of no LOD-0 triangle, which 97 shipped
            # models over 91 files are, so the scene has nothing of it to supply.
            name = mo.name or "model %d of bodypart %d" % (mi, bi)
            if drawn_model(mo, donor_faces, bi, mi) and blank is not None:
                blank.append((bi, mi, name))
            elif drawn_model(mo, donor_faces, bi, mi):
                missing.append(name)
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
        if (_must_rebuild(obj, mo, ok, donor_faces.get((bi, mi)),
                          (boxes or {}).get((bi, mi)), donor_meshes.get((bi, mi)))
                or ("weights" in fields and off_bone0(obj, mo, bone_index))):
            # The split writes 44-byte records, so a quantised model gains the weights and
            # normals its own record has no field for; nothing is unsupported there.
            rebuild[(bi, mi)] = obj
            continue
        unsupported |= set(no)
        if ok:
            edits[(bi, mi)], n, nb, nu = read_mesh(obj, mo, bone_index, ok)
            renormals += n
            blind_normals += nb
            unskinned += nu if skin_counted(mo) else 0
    return (edits, rebuild, missing, sorted(unsupported), renormals, crowded,
            blind_normals, uv_spare, donor_faces, unskinned, stray)


def drawn_model(mo, donor_faces=None, bi=None, mi=None):
    """Whether the import made an object for this model: it has vertices and, where the
    donor's .vtx was read, LOD-0 triangles."""
    if not mo.numvertices:
        return False
    return not donor_faces or bool(donor_faces.get((bi, mi)))


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
            raise SceneRefused("%s has no animation named %r" % (m.path, name))
        return names.index(name)
    i = name_index(names, action)
    if i is None:
        raise build_mod.Refused(
            "action %r is not named after any animation of %s and carries no usable "
            "index: tick \"Add the rest as new\" to append it, or pick the animation it "
            "replaces" % (action.name, os.path.basename(m.path)))
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
    """{index: the name Blender now gives it}.

    The index comes off `match_indices`, which is already what decides which action writes
    which animation, so this adds no second rule -- it reads that map and reports the slots
    whose action is no longer called after the record.  An action that took its slot by
    name renames nothing by construction.

    `vtmb_anim_name` is what makes a rename sayable, and the file's own name at the slot is
    not enough: Blender deduplicates, so the two animations called `walk` that 3 of the
    4445 shipped models carry import as `walk` and `walk.001`, and the second differs from
    the record's name on a plain re-export that renamed nothing.  The stamp also has to
    still agree with the file, or it describes an animation some earlier export already
    renamed and the scene is not the authority any more. An action with no stamp is one an
    import before it made, which `old_imports` refuses.
    """
    found = match_indices(anim_names, source, arm_obj)[0]
    out = {}
    for i, act in sorted(found.items()):
        was = act.get("vtmb_anim_name")
        if act.name != anim_names[i] and was is not None and str(was) == anim_names[i]:
            out[i] = act.name
    return out


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


def skin_table(arm_obj):
    """(rows, names, nfile): the armature's skin table as rows of record indices, the name
    of each record, and how many of those records the imported file carried.

    A record at or past `nfile` is one a skin-family cell named and the file had not got,
    so the export finds or adds it by name.
    """
    flat = [int(x) for x in arm_obj.get("vtmb_skin_table") or ()]
    nref = int(arm_obj.get("vtmb_skin_refs") or 0)
    rows = ([flat[f:f + nref] for f in range(0, len(flat), nref)]
            if flat and nref > 0 else [])
    names = [str(x) for x in arm_obj.get("vtmb_skin_materials") or ()]
    return rows, names, int(arm_obj.get("vtmb_skin_textures", len(names)))


def skin_edited(arm_obj):
    """Whether a skin-family operator has edited the table, which is what makes the export
    write the armature's table over the file's."""
    return bool(arm_obj.get("vtmb_skin_edited"))


def armature_meshes(scene, arm_obj):
    """The mesh objects this armature owns, parented or bound by an Armature modifier, less
    its collision hulls."""
    return [o for o in scene.objects if o.type == "MESH" and not o.get("vtmb_phy_hull") and (
        o.parent is arm_obj or any(getattr(mo, "object", None) is arm_obj
                                   for mo in o.modifiers))]


def skin_slots(objs):
    """[(object, slot it is in now, skinref)] for every slot an import or export stamped."""
    out = []
    for o in objs:
        refs = [int(x) for x in (o.get("vtmb_skinrefs") or ())]
        # Through the import's stamp, since a reordered slot holds a different material
        # and writing by the stamped index would put a family's picture on the wrong mesh.
        moves = slot_moves(o)[0]
        for slot, r in enumerate(refs):
            at = slot if moves is None else moves.get(slot)
            if at is not None and at < len(o.data.materials):
                out.append((o, at, r))
    return out


def write_skin_table(arm_obj, rows, names=None):
    if "vtmb_skin_textures" not in arm_obj:
        arm_obj["vtmb_skin_textures"] = len(arm_obj.get("vtmb_skin_materials") or ())
    arm_obj["vtmb_skin_table"] = [r for row in rows for r in row]
    arm_obj["vtmb_skin_families"] = len(rows)
    if names is not None:
        arm_obj["vtmb_skin_materials"] = names
    arm_obj["vtmb_skin_edited"] = True


def sync_skin_table(arm_obj, objs, materials):
    """Take what each shown slot holds into the skin table, before anything re-points it.

    A material renamed while its family showed renames its texture record in every family
    that draws it; the record's old name is then on no material in the blend. On a model
    of more than one family, any other material put into a slot is the shown family's
    choice at that position, as the panel's picker makes it, so the other families keep
    theirs. On a model of one family the slot renames the record, as the export always
    has. `vtmb_skin_was` keeps the file's names, which is what lets the export tell a
    rename from another donor's record. Returns how many positions changed.
    """
    rows, names, nfile = skin_table(arm_obj)
    fam = int(arm_obj.get("vtmb_skin_family") or 0)
    if not 0 <= fam < len(rows):
        return 0
    many = len(rows) > 1
    renamed, picked, done = 0, 0, set()
    for o, at, r in skin_slots(objs):
        mat = o.data.materials[at]
        if mat is None or not 0 <= r < len(rows[fam]) or r in done:
            continue
        done.add(r)
        rec = rows[fam][r]
        if not 0 <= rec < len(names) or names[rec] == mat.name:
            continue
        if not many or (materials.get(names[rec]) is None and mat.name not in names):
            if "vtmb_skin_was" not in arm_obj:
                arm_obj["vtmb_skin_was"] = names[:nfile]
            names[rec] = mat.name
            renamed += 1
            continue
        if mat.name not in names:
            names.append(mat.name)
        rows[fam][r] = names.index(mat.name)
        picked += 1
    if picked:
        write_skin_table(arm_obj, rows, names)
    elif renamed:
        arm_obj["vtmb_skin_materials"] = names
    return renamed + picked


def resolve_skin_table(d, arm_obj):
    """The armature's skin table with every record index resolved against `d`, and the
    names of the records it had to add.

    A file record keeps its index; a record a cell added is found by name, else appended
    with no skinref of its own, the form 648 shipped models carry. A file wider than the
    table is one an earlier export appended a model to, and the appended column draws
    one record in every family, so row 0's tail extends each row.
    """
    rows, names, nfile = skin_table(arm_obj)
    have = {t.name: k for k, t in enumerate(d.textures)}
    added, out = [], []
    width = len(d.skin[0]) if d.skin else 0
    if rows and len(rows[0]) > width:
        raise SceneRefused("the armature's skin table has %d texture positions and the file "
                           "has %d, so its families do not fit this file"
                           % (len(rows[0]), width))
    for f, row in enumerate(rows):
        new = []
        for r, idx in enumerate(row):
            if 0 <= idx < nfile:
                if idx >= len(d.textures):
                    raise SceneRefused("skin family %d draws texture record %d at position "
                                       "%d, and the file has %d" % (f, idx, r, len(d.textures)))
                new.append(idx)
                continue
            name = names[idx] if 0 <= idx < len(names) else ""
            if not name:
                raise SceneRefused("skin family %d names no material at position %d" % (f, r))
            if name not in have:
                have[name] = build_mod.add_texture(d, name)
                added.append(name)
            new.append(have[name])
        out.append(new + list(d.skin[0][len(new):]))
    return out, added


def skin_renames(d, arm_obj, reached):
    """{record: name} for the records no slot reaches now that a slot renamed while another
    family was showing.

    `vtmb.set_skin_family` copies each shown slot's name into `vtmb_skin_materials` before
    it switches, and `vtmb_skin_was` keeps what the file called each record. A record
    whose file name is no longer the donor's is another file's and is left alone.
    """
    was = [str(x) for x in arm_obj.get("vtmb_skin_was") or ()]
    _rows, names, nfile = skin_table(arm_obj)
    out = {}
    for k in range(min(nfile, len(was), len(names), len(d.textures))):
        if k in reached or not names[k] or names[k] == was[k]:
            continue
        if d.textures[k].name == was[k]:
            out[k] = names[k]
    return out


def scratch_skin_rows(d, arm_obj, mesh_objs):
    """The armature's skin families laid over a no-donor file, whose one row the scratch
    writer built as identity. Returns how many families the file then carries.

    Each column of that row is a texture named after a slot's material, and the slot's
    `vtmb_skinrefs` stamp says which position of the armature's table it shows. The
    family on screen is the identity row itself; another family's record there is found
    by name or added with no skinref. A column no stamped slot shows, or one the families
    agree on, draws the same record in every family.
    """
    rows, names, _nfile = skin_table(arm_obj)
    if len(rows) < 2 or not d.skin:
        return len(d.skin)
    fam = int(arm_obj.get("vtmb_skin_family") or 0)
    if not 0 <= fam < len(rows):
        raise SceneRefused("the armature's skin family is %d, but its skin table holds %d"
                           % (fam, len(rows)))
    have = {t.name: k for k, t in enumerate(d.textures)}
    col_ref = {}
    for obj in mesh_objs:
        refs = [int(x) for x in obj.get("vtmb_skinrefs") or ()]
        moves = slot_moves(obj)[0]
        mats = obj.data.materials
        for slot, r in enumerate(refs):
            at = slot if moves is None else moves.get(slot)
            if at is None or not 0 <= at < len(mats) or mats[at] is None:
                continue
            t = have.get(mats[at].name)
            if t is None or not 0 <= r < len(rows[0]):
                continue
            was = col_ref.setdefault(t, r)
            if was != r and any(row[was] != row[r] for row in rows):
                raise SceneRefused("material %r shows skin positions %d and %d, which the "
                                   "other skin families paint differently"
                                   % (mats[at].name, was, r))
    width = len(d.skin[0])
    out = []
    for f, row in enumerate(rows):
        new = []
        for t in range(width):
            r = col_ref.get(t)
            name = names[row[r]] if r is not None and 0 <= row[r] < len(names) else ""
            if f == fam or not name:
                new.append(d.skin[0][t])
                continue
            if name not in have:
                have[name] = build_mod.add_texture(d, name)
            new.append(have[name])
        out.append(new)
    d.skin = out
    return len(out)


def skin_family(m, arm_obj):
    """Which row of the skin table the scene's material slots are showing.

    Out of range is refused, not clamped: the engine clamps to row 0 silently, and a
    rename read back through the wrong row renames every mstudiotexture_t in the file.
    """
    fam = int(arm_obj.get("vtmb_skin_family") or 0)
    if skin_edited(arm_obj):
        n = len(skin_table(arm_obj)[0])
        if not 0 <= fam < n:
            raise SceneRefused("the armature's skin family is %d, but its skin table holds %d"
                               % (fam, n))
        return fam
    if m.skins and not 0 <= fam < len(m.skins):
        raise SceneRefused("the armature's skin family is %d, but %s carries %d"
                           % (fam, os.path.basename(m.path), len(m.skins)))
    return fam


def slot_faces(me):
    """(x, y, z, face count) for each material slot of `me`, xyz its first face's centre;
    (0, 0, 0, 0) where no face uses it. A position and not an index, since deleting faces
    renumbers every later one."""
    idx = array.array("i", [0]) * len(me.polygons)
    me.polygons.foreach_get("material_index", idx)
    counts = collections.Counter(idx)
    out = []
    for j in range(len(me.materials)):
        if counts[j]:
            c = me.polygons[idx.index(j)].center
            out.append((c[0], c[1], c[2], float(counts[j])))
        else:
            out.append((0.0, 0.0, 0.0, 0.0))
    return out


def stamp_slots(obj, moves=None):
    """What each slot holds and which faces use it, which is what `slot_moves` reads.

    `moves` is `slot_moves`' map from before the slots were re-pointed: the stamp stays in
    the import's slot order, which `vtmb_skinrefs`, `vtmb_meshes` and `vtmb_eye_slots`
    are indexed by, so a reordered slot is stamped where it came from. A stamped slot no
    slot holds keeps its name and no faces.
    """
    me = obj.data
    names = [m.name if m else "" for m in me.materials]
    now = slot_faces(me)
    if moves is None:
        obj["vtmb_slot_mats"] = names
        obj["vtmb_slot_faces"] = [x for pair in now for x in pair]
        return
    old = [str(x) for x in (obj.get("vtmb_slot_mats") or ())]
    at = [moves.get(k) for k in range(len(old))]
    obj["vtmb_slot_mats"] = [old[k] if j is None else names[j] for k, j in enumerate(at)]
    obj["vtmb_slot_faces"] = [x for j in at for x in ((0.0,) * 4 if j is None else now[j])]


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

    Names come second. Blender's Move up/down takes a slot's faces with it, and a material
    put into a slot moves none, so names alone read two slots given each other's materials
    as a reorder (BUGS 179). `obj["vtmb_slot_faces"]` holds each slot's first face's centre
    and face count, and the one slot now matching both is that slot. A slot with no face, a
    file mesh whose triangles all sit in a lower LOD, any slot an edit has given or taken
    faces or whose first face it moved, and slots stamped alike, go by name. Two meshes over
    one surface stamp alike (`panel_cloth.mdl`'s two sides), and once an edit takes faces
    from one, both stamps match the other alone (BUGS 181).

    None rather than a map where the object carries no stamp -- a `.blend` saved before
    2026-09-09 -- since the index is then all there is and reading every slot as deleted
    would be worse than the defect.
    """
    stamp = [str(x) for x in (obj.get("vtmb_slot_mats") or ())]
    if not stamp:
        return None, []
    mats = obj.data.materials
    out = {}
    faces = [float(x) for x in (obj.get("vtmb_slot_faces") or ())]
    if len(faces) == 4 * len(stamp):
        keys = [tuple(faces[4 * slot:4 * slot + 4]) for slot in range(len(stamp))]
        at = {}
        for j, key in enumerate(slot_faces(obj.data)):
            if key[3]:
                at.setdefault(key, []).append(j)
        for slot, key in enumerate(keys):
            if key[3] and keys.count(key) == 1 and len(at.get(key, ())) == 1:
                out[slot] = at[key][0]
    # One slot per stamped slot, and a slot still holding its own stamped name keeps it:
    # a family drawing one material at two positions stamps the name twice.
    taken = set(out.values())
    for slot, name in enumerate(stamp):
        if (slot not in out and slot not in taken and slot < len(mats)
                and mats[slot] is not None and mats[slot].name == name):
            out[slot] = slot
            taken.add(slot)
    here = {}
    for j, mm in enumerate(mats):
        if mm is not None and j not in taken:
            here.setdefault(mm.name, []).append(j)
    for slot, name in enumerate(stamp):
        if slot not in out:
            out[slot] = here[name].pop(0) if here.get(name) else None
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


def read_materials(m, source, family=0, rows=None):
    """({file texture index: name}, slots reordered, [slots deleted]).

    `rows` is the skin table the slots show where the scene edited it, else the file's.

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
            table = m.skins if rows is None else rows
            row = table[family] if 0 <= family < len(table) else None
            if row and 0 <= ref < len(row):
                ref = row[ref]
            name = mats[slot].name
            if out.get(ref, name) != name:
                raise SceneRefused("texture %d is named both %r and %r by the scene's "
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


def spring_end_bone(arm_obj, end):
    """The pose bone a stored spring end means: the one Blender calls `end`, else the one
    whose `vtmb_bone_name` says it was `end` in the file, else None.

    The end is stored as a name, the picker offers Blender's names, and Blender has no
    rename event to rewrite the key with, so either kind of name can be standing there.
    """
    pb = arm_obj.pose.bones.get(end)
    if pb is not None:
        return pb
    hits = [p for p in arm_obj.pose.bones if str(p.get("vtmb_bone_name") or "") == end]
    return hits[0] if len(hits) == 1 else None


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
           "added": [], "removed": [], "stamps": {}, "gone": [], "over_mask": 0,
           "end_renamed": []}

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
            # `m.bones` keeps the file's names, so the end resolves through its bone's stamp.
            eb = spring_end_bone(arm_obj, str(end))
            fname = str(end) if eb is None else str(eb.get("vtmb_bone_name") or eb.name)
            if eb is not None and eb.name != end:
                out["end_renamed"].append((k, str(end), eb.name))
            # +0x00 carries the start bone as -1-bone when the chain starts disabled, and
            # want0 is the form this export is writing rather than the one it read.
            want = spring_endbone(k, want0 if want0 >= 0 else -1 - want0, fname, m)
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


def stray_eyeballs(m, arm_obj):
    """Eyeball empties of this file's models, in the armature's scene, under no armature.

    Clear Parent (Alt+P) takes an empty out of `eyeball_objects`, which would read it as
    deleted; nothing then says which bone it moves with.
    """
    models = {mo.name for bp in m.bodyparts for mo in bp.models}
    scenes = set(arm_obj.users_scene)
    out = []
    for obj in bpy.data.objects:
        if (obj.get("vtmb_eyeball") is None
                or str(obj.get("vtmb_eyeball_model") or "") not in models
                or not scenes & set(obj.users_scene)):
            continue
        up = obj.parent
        while up is not None and up.type != "ARMATURE":
            up = up.parent
        if up is None:
            out.append(obj.name)
    return sorted(out)


# The stamp each kind of accessory empty carries, and what the message calls it.
ACCESSORY_KEYS = (("vtmb_attachment", "attachment"), ("vtmb_hitbox_group", "hitbox"),
                  ("vtmb_hitboxset", "hitbox set"), ("vtmb_cloth_volume_model", "cloth volume"),
                  ("vtmb_phy_hull", "collision hull"))


def stray_accessories(arm_obj, eyes=False):
    """[(kind, name)]: accessory empties in the armature's scene under no armature.

    Clear Parent (Alt+P) leaves the empty in the outliner and takes it out of what the
    export reads, which would write it as deleted. An armature is the only owner an empty
    names, so one stray stops the export of every armature in its scene.
    """
    keys = ACCESSORY_KEYS + ((("vtmb_eyeball", "eyeball"),) if eyes else ())
    scenes = set(arm_obj.users_scene)
    out = []
    for obj in bpy.data.objects:
        kind = next((k for key, k in keys if obj.get(key) is not None), None)
        if kind is None or not scenes & set(obj.users_scene):
            continue
        up = obj.parent
        while up is not None and up.type != "ARMATURE":
            up = up.parent
        if up is None:
            out.append((kind, obj.name))
    return sorted(out)


def stray_text(stray):
    one = len(stray) == 1
    names = ", ".join("%s %r" % x for x in stray[:6])
    if len(stray) > 6:
        names += " and %d more" % (len(stray) - 6)
    it = "it" if one else "them"
    return ("%s %s under no armature, so the export would write %s as deleted. Parent %s "
            "back under the armature (Ctrl+P), or delete %s"
            % (names, "is" if one else "are", it, it, it))


def bone_space(obj):
    """An accessory empty's matrix in its bone's space, where the viewport draws it.

    Blender places a bone child at `bone.matrix_local @ Translation((0, length, 0)) @
    matrix_parent_inverse @ matrix_basis`. Ctrl+P > Bone puts the bone's own inverse in
    the parent inverse and the world matrix in `matrix_basis`, so the whole product is
    read. The import's parent inverse cancels the tail offset, and while it still does,
    `matrix_basis` is read alone: Edit Mode recomputes every bone's length, by up to 7.6e-06
    on andrei, and the product would carry that into every record. On a bone Blender
    deleted, `matrix_basis` alone, where `remove_bone` already moved the file's record.
    """
    arm_obj = obj.parent
    bone = (arm_obj.data.bones.get(obj.parent_bone)
            if arm_obj is not None and arm_obj.type == "ARMATURE"
            and obj.parent_type == "BONE" else None)
    if bone is None or _tail_inverse(obj.matrix_parent_inverse, bone.length):
        return obj.matrix_basis
    return (mathutils.Matrix.Translation((0.0, bone.length, 0.0))
            @ obj.matrix_parent_inverse @ obj.matrix_basis)


def _tail_inverse(inv, length):
    """Whether `inv` is `Translation((0, -length, 0))` up to 16 float32 steps of the length,
    the parent inverse `blender_import._bone_child` sets."""
    if any(inv[r][c] != (1.0 if r == c else 0.0) for r in range(4) for c in range(3)):
        return False
    return (inv[0][3] == 0.0 and inv[2][3] == 0.0 and inv[3][3] == 1.0
            and abs(inv[1][3] + length) <= max(length, 1.0) * 2.0 ** -19)


# Deleting one end of a capsule is itself refused, so the way out names both, and so is
# deleting an eyeball's empty while its eye's faces stay (decision 21).
_DELETE = {"cloth capsule end": "delete both of the capsule's ends",
           "eyeball": "delete it and the eye's faces, which removes the eyeball and its "
                      "mesh"}


def _accessory_bone(obj, m, arm_obj, bmap, what):
    """The file bone index an accessory empty is mounted on.

    An empty whose bone Blender deleted passes to the first surviving ancestor, which is
    where `remove_bone` already put the file's own record.
    """
    fix = ("Parent it to one of the model's bones (Ctrl+P > Bone, or Object properties > "
           "Relations), or %s" % _DELETE.get(what, "delete it"))
    want = obj.parent_bone if obj.parent_type == "BONE" else ""
    if not want:
        raise build_mod.Refused("%s %r is not parented to a bone, so nothing says which bone "
                                "it moves with. %s" % (what, obj.name, fix))
    for b in m.bones:
        if bmap.get(b.name) == want:
            return b.index
    gone = getattr(m, "gone", {})
    name = want
    while name in gone:
        name = gone[name]
        if name in bmap:
            return next(b.index for b in m.bones if b.name == name)
    raise build_mod.Refused("%s %r is on bone %r, which is not one this model carries. %s"
                            % (what, obj.name, want, fix))


def read_attachments(m, arm_obj, scale):
    """[(name, type, bone, 12 floats)] out of the scene, in bone space (`bone_space`)."""
    bmap = bone_map(m, arm_obj)
    out = []
    for obj in accessory_objects(arm_obj)[0]:
        bi = _accessory_bone(obj, m, arm_obj, bmap, "attachment")
        local = bone_space(obj)
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
    local = bone_space(obj)
    mid = [local[r][3] / scale for r in range(3)]
    half = [local.col[c].to_3d().length / scale for c in range(3)]
    # mstudiobbox_t is an AABB in bone space and carries no rotation, so a turned empty
    # would be written as its unturned self. 1e-3 is three orders above the 1e-6 the
    # identity round trip costs and far below any deliberate turn.
    turn = max(abs(local.col[c].to_3d()[r] / (half[c] * scale or 1.0)
                   - (1.0 if r == c else 0.0))
               for c in range(3) for r in range(3))
    if turn > 1e-3:
        raise build_mod.Refused("hitbox %r is turned %.4f out of its bone's axes, and "
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


def face_flexes_refused(d):
    """Refused naming every lid or mouth flex that nothing in the file as written makes.

    The mouth's flex and lid flexes 0-5 are weights only a flex rule writes; lid flexes 6-7
    are weights the eyeball code writes and only a shape key shows (`r_studioflex.cpp:118-144`).
    9 of the 4445 shipped files fail it, named in roadmap section 4c.
    """
    rules = {struct.unpack_from("<i", r.raw, 0)[0] for r in d.flexrules}
    keys = {struct.unpack_from("<i", f.raw, 0)[0] for bp in d.bodyparts for mr in bp.kids
            for mesh in mr.kids for f in mesh.kids}

    def names(ids):
        return ", ".join(repr(d.flexdescs[i].name) if 0 <= i < len(d.flexdescs)
                         else "#%d" % i for i in dict.fromkeys(ids))

    bad = ["the mouth's %s, which no flex rule writes" % names([i])
           for i in (struct.unpack_from("<i", r.raw, 16)[0] for r in d.mouths)
           if i not in rules]
    for bp in d.bodyparts:
        for mr in bp.kids:
            model = bytes(mr.raw[:128]).split(b"\0")[0].decode("latin1")
            for k, e in enumerate((mr.extra or {}).get("eyes") or ()):
                ids = struct.unpack_from("<6i", e.raw, 0x44) + struct.unpack_from(
                    "<2i", e.raw, 0x74)
                if not any(ids):
                    continue
                inputs = [i for i in ids[:6] if i not in rules]
                morphs = [i for i in ids[6:] if i not in keys]
                if inputs:
                    bad.append("eyeball %d of %r: lid flexes %s, which no flex rule writes"
                               % (k, model, names(inputs)))
                if morphs:
                    bad.append("eyeball %d of %r: lid flexes %s, which no shape key carries"
                               % (k, model, names(morphs)))
    if bad:
        raise build_mod.Refused(
            "nothing makes these face flexes, so the game would weight them for nothing: "
            "%s. Write a flex rule of that name (Add flex line, in the armature's VTMB flex "
            "controllers panel) or a shape key of that name, or give the mouth a Jaw flex a "
            "rule writes (VTMB eyeballs and mouth), or set the eyeball empty's Lid flexes to "
            "None (VTMB eyeball)"
            % "; ".join(bad))


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
    if lo == hi:
        raise ValueError("flex controller %r has an empty range: the server normalises a "
                         "value to (v - min) / (max - min), which divides by zero" % line)
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


def cloth_sigma_edges(obj, pv, springs, ns0, pos):
    """{group-0 spring: the sigma the mesh edge carrying it holds}, empty where none does.

    `cloth.edge_keys` is the resolution and the import stamps the attribute through the same
    call, so a spring reaches the edge it was written to. 289 of the corpus's 30 623 group-0
    springs reach none and are absent here, which leaves them the value the file holds.
    `pv` names Blender vertices and `pos` is the file's position per Blender vertex.
    """
    me = obj.data
    att = me.attributes.get("vtmb_cloth_sigma")
    if att is None or att.domain != "EDGE" or len(att.data) != len(me.edges):
        return {}
    buf = [0.0] * len(me.edges)
    att.data.foreach_get("value", buf)
    at = {cloth_mod.pair(*tuple(e.vertices)): e.index for e in me.edges}
    keys, _missed = cloth_mod.edge_keys(pv, springs, ns0, pos, set(at))
    return {q: buf[at[k]] for q, k in keys.items() if buf[at[k]] != cloth_mod.SIGMA_UNSET}


# The addon version at or past which `vtmb_pinned` holds pv[:numfixed]. Before 2026-09-12 the
# import built that group from the mesh binding, which pins fewer -- 44 against 55 on
# `panel_cloth` -- so a pin set read off an older scene regenerates the object around the wrong
# pins, and the group existing before its meaning changed is what absence cannot separate.
CLOTH_PIN_VERSION = (0, 2, 51)

def cloth_flip_bits(obj, origins):
    """The scene's cloth normal flip per output vertex, or None where the scene never said.

    Bit 15 of `mstudiomesh_t +0x34` negates that vertex's normal, and 63 of the 84 shipped
    meshes set it on some vertices and not others, so the import stamps it per vertex. A scene
    carrying neither the attribute nor the object key has said nothing and the file's own bit
    stands: `blender_scratch.flip_flags` answers False there, which is right for a sheet
    authored from nothing and would clear every flip in the file on a donor.
    """
    from . import blender_scratch as scratch_mod
    if (obj.data.attributes.get("vtmb_cloth_flip") is None
            and obj.get("vtmb_cloth_flip") is None):
        return None
    return scratch_mod.flip_flags(obj, origins)


def cloth_flip_moves(cl, offsets, flips, col):
    """([(mesh, mesh-local vertex, the bit the scene wants)], [(mesh, vertices left alone)]).

    The binding's own entry is one owner byte and one ushort per mesh vertex -- low 15 bits
    the particle, bit 15 the flip. The owner byte is the COLUMN of the row-0 cloth object
    that drives the vertex, so 0xff is a vertex this model draws rigidly and any other
    column is a vertex a sibling object owns. `blender_import._stamp_cloth` puts row 0's
    FIRST object on the mesh and stamps the attribute only where the owner byte is that
    object's column, so every other column reads False whatever the file holds and writing
    it would clear the file's own bits -- 293 of them on `tremere_female_armor_3`, the one
    shipped model of 60 carrying two objects in row 0 (BUGS 154). Those are counted and
    left alone.

    `flips` is one answer per model vertex of the file being written and `offsets` each
    mesh's `vertexoffset` in it, which a geometry rebuild moves.
    """
    if flips is None:
        return [], []
    data, out, other = cl["data"], [], []
    for k in sorted(cl["meshes"]):
        (own_at, p34_at, _p38_at), n = cl["meshes"][k]
        off = offsets[k] if k < len(offsets) else 0
        skipped = 0
        for j in range(n):
            own = data[own_at + j]
            if own == 0xff:
                continue
            if own != col:
                skipped += 1
                continue
            raw = struct.unpack_from("<H", data, p34_at + j * 2)[0]
            want = flips if isinstance(flips, bool) else bool(flips[off + j])
            if want != bool(raw & 0x8000):
                out.append((k, j, want))
        if skipped:
            other.append((k, skipped))
    return out, other


def cloth_pin_set(obj, pv, npart):
    """(the particles the scene pins, the group members naming no particle), or None.

    The pin set is spelled by the particle ORDER -- [0, numfixed) is pinned and nothing else
    marks a pin -- so a scene changes it by naming a different set of anchor vertices, and
    `pv[p]` is a Blender vertex index because a Blender object is one mstudiomodel_t carrying
    every mesh of it with `vertexoffset` applied. None where the object names no such group,
    which is a scene that has never said and keeps the file's own pins.
    """
    from . import blender_scratch as scratch_mod
    name = str(obj.get("vtmb_cloth_pin_group") or scratch_mod.PIN_GROUP)
    vg = obj.vertex_groups.get(name)
    if vg is None:
        return None
    anchors = {}
    for q in range(min(npart, len(pv))):
        anchors.setdefault(pv[q], q)
    members = set(v.index for v in obj.data.vertices
                  if any(g.group == vg.index for g in v.groups))
    return (set(anchors[v] for v in members if v in anchors),
            sorted(members - set(anchors)))


def cloth_pin_why(obj, cl, pinned, npart):
    """Why this object's pin edit cannot be written, or None where it can. A scene older
    than `CLOTH_PIN_VERSION` never reaches it: `old_imports` refuses it."""
    if cl["cols"] != 1:
        return ("this model's cloth table is %d columns wide and a regeneration emits one"
                % cl["cols"])
    if not pinned:
        return "the pin group holds no particle's anchor vertex, so every particle would be free"
    if len(pinned) >= npart:
        return "the pin group pins every one of the %d particles, so nothing would move" % npart
    return None


def cloth_scene(m, mo, me, cmap=None):
    """(Blender vertex per written vertex, rest position per written vertex, rest position
    per Blender vertex, each mesh's written `vertexoffset`), or None.

    `cmap` is `rebuild_cell`'s map for a model the geometry rebuild wrote. A carried vertex
    rests where the donor had it, so a particle the scene moved still reads as moved; an
    added vertex, or any vertex of a regenerated object, rests where it is written.
    """
    if cmap is None:
        if len(me.vertices) != mo.numvertices:
            return None
        pos = [v.pos for v in m.vertices(mo)]
        return (range(len(pos)), pos, pos, [ms.vertexoffset for ms in mo.meshes])
    vertex, rest = cmap["vertex"], list(cmap["written"])
    offsets, at = [], 0
    for n in cmap["runs"]:
        offsets.append(at)
        at += n
    carry = cmap["carry"]
    if carry is not None and not cmap["regenerated"]:
        donor = m.vertices(mo)
        for k, w0 in enumerate(offsets):
            if k >= len(mo.meshes):
                break
            ms, cm = mo.meshes[k], carry[k]
            for j in range(cmap["runs"][k]):
                o = j if cm is None else cm[j]
                if o is not None and o < ms.numvertices:
                    rest[w0 + j] = donor[ms.vertexoffset + o].pos
    rest_b = [tuple(v.co) for v in me.vertices]
    for x, vi in enumerate(vertex):
        rest_b[vi] = rest[x]
    return vertex, rest, rest_b, offsets


def remove_cloth(d, m, source):
    """Takes the cloth object off every model whose scene object has the Cloth box unticked.

    The import stamps `vtmb_cloth` 1 on a model carrying cloth and the box writes 0, and a
    scene imported before the stamp carries no key, which `old_imports` refuses. Only the
    cloth object goes -- the
    collision capsules and spheres name bones and no vertex, and the gather at
    `StudioRender 0x2c014a00` collides them with every cloth object of the model, so they
    stay where the file has them. With no cloth left the mesh binding arrays at
    `mstudiomesh_t +0x30..+0x38` emit zero and `studiohdr_t.flags` 0x400 clears.

    Answers one (object, model, objects, rows, meshes) per model whose cloth went.
    """
    found = mesh_objects(m, source)
    out = []
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        rec = d.bodyparts[bi].kids[mi]
        cl = rec.extra.get("cloth")
        obj = found.get((bi, mi))
        if not cl or obj is None or obj.get("vtmb_cloth", 1):
            continue
        rec.extra["cloth"] = None
        out.append((obj.name, mo.name, len(cl["slots"]), cl["rows"], len(cl["meshes"])))
    return out


def cloth_bumpmaps(m, content):
    """The cloth-bound meshes of a written file whose material carries a `$bumpmap`.

    Answers (hits, unread). A hit is (model name, mesh index, bumped materials, own,
    records) over row 0's bound vertices: `own` counts those whose +0x38 names the
    particle's own normal, where the shader's tangent equals the normal and the bump map
    has no visible effect, and `records` those naming a blend record, which only a donor's
    own object carries and which draws the bump frame turned 90 degrees from the rigid
    vertices' (both seen in game, task-vtmb-cloth-blend-table.md). `unread` is every cloth
    material no content root resolves, so its `$bumpmap` is unknown. Every skin family is
    read, since a family swaps the material at runtime. Nothing is written differently for
    a hit -- the export generates no blend record (`cloth.one_to_one`).
    """
    fams = m.skins or [list(range(len(m.materials)))]
    hits, unread, bump = [], [], {}
    for _bi, _mi, _bp, mo in mesh_mod.models_of(m):
        size = {c.col: c.numparticles for c in mo.cloths if c.row == 0}
        for ms in mo.meshes:
            if not ms.clothbind or not ms.clothbind[0]:
                continue
            names = sorted({m.materials[f[ms.material]] for f in fams
                            if 0 <= ms.material < len(f)
                            and 0 <= f[ms.material] < len(m.materials)})
            for name in names:
                if name in bump:
                    continue
                blob = None
                for stem in paths_mod.material_stems(name, m.material_paths):
                    blob = content.read(stem + ".vmt")
                    if blob is not None:
                        break
                bump[name] = (None if blob is None
                              else paths_mod.vmt_value(blob, "$bumpmap") is not None)
                if blob is None:
                    unread.append(name)
            bumped = [n for n in names if bump[n]]
            if not bumped:
                continue
            nrm = (ms.clothnormal or [{}])[0]
            own = sum(1 for v, (col, _p, _f) in ms.clothbind[0].items()
                      if nrm.get(v, 0) < size.get(col, 0))
            hits.append((mo.name, ms.index, bumped, own, len(ms.clothbind[0]) - own))
    return hits, unread


def cloth_edits(m, source, d, rebuilt=None):
    """What each cloth-bound model's scene says that its donor bytes do not.

    One entry per model carrying cloth: which of the three authored numbers moved, how many
    particles sit somewhere else than the file compiled them at, and the reason the object
    cannot be spoken for at all. Split from the write so the same comparison reports the
    narrowing when the write is off -- a cloth object left describing the geometry it was
    compiled against simulates the old garment, and nothing on screen says so.

    `rebuilt` is {(bodypart, model): `edits["cloth_map"]`} for every model this export's
    geometry rebuild wrote, whose cloth object names the written numbering.
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
             "springs": 0, "flattens": False, "sigma_edges": 0,
             "flips": [], "flips_other": [], "pins": None, "pins_why": None,
             "repin": False, "scene": None, "rebuilt": (bi, mi) in (rebuilt or {})}
        out.append(e)
        if obj is None:
            e["why"] = "no mesh object in the scene belongs to this model"
            continue
        slot = _cloth_slot(cl)
        if slot is None:
            e["why"] = "the file's cloth table names no object in row 0"
            continue
        if not obj.get("vtmb_cloth", 1):
            # `remove_cloth` takes it out where Cloth from the scene is on, so this is
            # reached only with that option off.
            e["why"] = ("the Cloth box is off and Cloth from the scene is off, so the "
                        "file's cloth object stays")
            continue
        slot_k, at = slot
        scale_f, npart, nfix, ns0, _spoff, pv, springs = _cloth_header(cl["data"], at)
        if not springs or not pv:
            e["why"] = "the file's cloth object carries no springs or no particle map"
            continue
        me = obj.data
        scene = cloth_scene(m, mo, me, (rebuilt or {}).get((bi, mi)))
        if scene is None:
            e["why"] = ("the scene holds %d vertices where the file's model has %d and no "
                        "geometry rebuild wrote this model, so nothing says which scene "
                        "vertex a particle anchors on" % (len(me.vertices), mo.numvertices))
            continue
        vertex, rest, rest_b, offsets = e["scene"] = scene
        if any(v >= len(vertex) for v in pv):
            e["why"] = ("the file's cloth object names a vertex past the %d this model "
                        "writes" % len(vertex))
            continue
        pv_b = [vertex[v] for v in pv]
        was = [rest[v] for v in pv]
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
                1 for q, v in cloth_sigma_edges(obj, pv_b, springs, ns0, rest_b).items()
                if abs(v - (springs[q][2] - springs[q][3]) / 2.0) > CLOTH_TOL)
        e["moved"] = sum(1 for v, w in zip(pv_b, was)
                         if _d2(tuple(me.vertices[v].co), w) > CLOTH_MIN_D2)
        e["springs"] = len(springs)
        # The scene's flip and the scene's pin set, which are what the panel draws and what
        # the donor path read nowhere until now, both read through the written numbering.
        e["flips"], e["flips_other"] = cloth_flip_moves(
            cl, offsets, cloth_flip_bits(obj, list(vertex)), slot_k % cl["cols"])
        e["pins"] = cloth_pin_set(obj, pv_b, npart)
        if e["pins"] is not None and e["pins"][0] != set(range(nfix)):
            e["pins_why"] = cloth_pin_why(obj, cl, e["pins"][0], npart)
            e["repin"] = e["pins_why"] is None
    return out


def cloth_refit(moved, a, b, d2_was, d2_now):
    """Whether a spring's `rest2` is refitted: one of its particles moved, and it has a
    direction to carry.

    It keys off the particle set `cloth_edits` counts `moved` with and never off the
    spring's own separation. A squared displacement is quadratic in a position error where
    a difference of two squared separations is linear in it, so one absolute tolerance over
    both is two different sensitivities: on a filetype-1 donor, whose packed positions
    decode to a different float32 than the ones the file was authored from, the per-vertex
    test reads about 1e-12 and the per-spring one about 4e-6, and any cloth edit at all then
    refit 2340 of bilttablea's 2657 rest lengths against positions nothing wrote (BUGS 153).
    """
    return (a in moved or b in moved) and d2_was > CLOTH_MIN_D2


def apply_cloth(d, m, source, edits=None):
    """Each model's row-0 cloth object rewritten in place from the scene.

    `scale` at +0x00, each spring's `w0`/`w1` from the panel's sigma with the file's own
    mass split kept, and `rest2` refitted on the springs whose own particles the scene
    moved -- and on no others, whatever the arithmetic says the separation is. Nothing
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
           "sigma_edges": 0, "flattened": [], "flips": 0, "repinned": 0, "pinned": 0}
    for e in edits:
        if e["why"] is not None:
            continue
        if not (e["scale"] or e["sigma"] or e["slack"] or e["moved"]
                or e["sigma_edges"] or e["flips"] or e["repin"]):
            continue
        bi, mi = e["key"]
        cl = d.bodyparts[bi].kids[mi].extra["cloth"]
        _k, at = _cloth_slot(cl)
        data = bytearray(cl["data"])
        _scale_f, _npart, _nfixed, ns0, spoff, pv, springs = _cloth_header(data, at)
        me = found[(bi, mi)].data
        vertex, rest, rest_b, _offsets = e["scene"]
        pv_b = [vertex[v] for v in pv]
        now = [tuple(me.vertices[v].co) for v in pv_b]
        was = [rest[v] for v in pv]
        moved = {p for p, (nw, wz) in enumerate(zip(now, was))
                 if _d2(nw, wz) > CLOTH_MIN_D2}
        if e["scale"]:
            struct.pack_into("<f", data, at, e["scale"][1])
            out["scale"] += 1
        sigma = e["sigma"][1] if e["sigma"] else None
        slack = e["slack"][1] if e["slack"] else None
        per_edge = ({} if sigma is not None
                    else cloth_sigma_edges(found[(bi, mi)], pv_b, springs, ns0, rest_b))
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
                elif cloth_refit(moved, a, b, d2_was, d2_now):
                    rest2 = (rest2 / d2_was) * d2_now if CLOTH_KEEP_RATIO else d2_now
                    out["refitted"] += 1
            struct.pack_into("<2H3f", data, at + spoff + q * 16, a, b, w0, w1, rest2)
            out["springs"] += 1
        for k, j, want in e["flips"]:
            (_own_at, p34_at, _p38_at), _n = cl["meshes"][k]
            raw = struct.unpack_from("<H", data, p34_at + j * 2)[0]
            struct.pack_into("<H", data, p34_at + j * 2,
                             (raw & 0x7fff) | (0x8000 if want else 0))
            out["flips"] += 1
        cl["data"] = bytes(data)
        if e["repin"]:
            # After the edits above, so the regeneration carries the sigma and the scale
            # this export just wrote rather than the ones the file came with.
            npin = cloth_repin(d, m, found[(bi, mi)], e)
            out["repinned"] += 1
            out["pinned"] += npin
        out["models"] += 1
        out["sigma"] += 1 if sigma is not None else 0
        out["slack"] += 1 if slack is not None else 0
        if e["flattens"]:
            out["flattened"].append(e["object"])
    return out


def cloth_repin(d, m, obj, e):
    """The model's row-0 cloth object regenerated around the pin set the scene names.

    The pinned set is spelled by the particle ORDER alone -- [0, numfixed) is pinned -- and
    `k0 = r0^2 / (r0^2 + r1^2)` for `r` the ring index from it, so every spring endpoint,
    every ring distance and the collision-triangle block are functions of the pin set and
    none of them can be written in place. `mdl_build._rebuild_cloth` is the same
    regeneration for a geometry edit; here nothing was deleted, so every particle survives
    and its model-local vertex is the one the file gave it.

    Returns how many particles the written object pins.
    """
    bi, mi = e["key"]
    cl = d.bodyparts[bi].kids[mi].extra["cloth"]
    _k, at = _cloth_slot(cl)
    data = cl["data"]
    co = cloth_mod.unpack_object(data, at)
    npart = co[0x04]
    pinned = e["pins"][0]
    keep = ([q for q in range(npart) if q in pinned]
            + [q for q in range(npart) if q not in pinned])
    me = obj.data
    vertex, _rest, _rest_b, offsets = e["scene"]
    pv_new = [co["pv"][q] for q in keep]
    pos = [tuple(me.vertices[vertex[v]].co) for v in pv_new]
    flips = cloth_flip_bits(obj, list(vertex))
    bound = []
    for k in sorted(cl["meshes"]):
        (own_at, p34_at, p38_at), n = cl["meshes"][k]
        off = offsets[k] if k < len(offsets) else 0
        for j in range(n):
            if data[own_at + j] == 0xff:
                continue
            raw = struct.unpack_from("<H", data, p34_at + j * 2)[0]
            if flips is None:
                flip = bool(raw & 0x8000)
            elif isinstance(flips, bool):
                flip = flips
            else:
                flip = bool(flips[off + j])
            bound.append((k, j, raw & 0x7fff, flip,
                          struct.unpack_from("<H", data, p38_at + j * 2)[0]))
    c, bind, _counts = cloth_mod.rebuild(co, keep, pv_new, pos, bound, npin=len(pinned))
    full = {}
    for k in sorted(cl["meshes"]):
        ent = [None] * cl["meshes"][k][1]
        for v, rec in bind.get(k, {}).items():
            ent[v] = rec
        full[k] = ent
    d.bodyparts[bi].kids[mi].extra["cloth"] = cloth_mod.region(
        cloth_mod.pack(c), c.numparticles, full)
    return len(pinned)


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


def apply_face(d, m, arm_obj, scale, source=None):
    """The eyeball records and the one mouth record, written only where they differ.

    Answers `{changed, materials, radius_keys, lid_targets, eye_empties, eye_records,
    eyes_added, eye_meshes}` -- the record count, the material names an eyeball put in the
    file, the eyeballs whose `vtmb_eyeball_radius` says something neither the sphere nor
    the file says, the eyeballs holding lid flexes without aim points or the other way
    round, the empties no record took as (name, model, index), the records no empty
    claimed as (model, index), which keep the donor's bytes, the records appended as
    (name, model, index, position, some mesh draws it), and the meshes whose eye fields
    were rewritten.

    An empty naming a donor model and claiming an index past that model's records is
    appended, in index order, and its position is what an eye mesh's `materialparam`
    names. An empty naming no model edits records and never appends: the model it would
    land on is a guess. `source` is what finds the mesh objects whose `vtmb_eye_slots`
    point a mesh at its eyeball; without it no mesh is touched.

    Every index a scene holds is a NAME here and is resolved against this file: the two
    materials against mstudiotexture_t, the eight lid flexes and the mouth's flex against
    mstudioflexdesc_t, the mouth's bone against the bone array. The corpus is why -- the
    mouth's flexdesc is 16 on 196 of 200 records and 0 on 4 that list `mouth` first, and
    the eyeball's own upperlidflexdesc is a genuine 0 on the right eye, so a stored index
    would say the wrong thing about either the moment a slot moves.

    zoffset (+0x14) and pitch/yaw (+0x7c, +0x84) are left at whatever the donor holds:
    they are 0 on 602 of 602 shipped records and the QC `eyeball` command has no token
    that sets them, so the scene draws neither.

    Raises Refused naming every lid target component at or past its eyeball's radius.
    """
    out = {"changed": 0, "materials": [], "radius_keys": [], "lid_targets": [],
           "eye_empties": [], "eye_records": [], "lid_past_radius": [],
           "eyes_added": [], "eye_meshes": 0, "mouth_added": False, "eyes_removed": [],
           "mouth_removed": False, "eye_gone": {}}
    stray = stray_eyeballs(m, arm_obj)
    if stray:
        raise build_mod.Refused(
            "eyeball empty %s is not parented to a bone, so nothing says which bone it moves "
            "with. Parent it to one of the model's bones (Ctrl+P > Bone, or Object properties "
            "> Relations), or %s" % (", ".join(repr(n) for n in stray), _DELETE["eyeball"]))
    eyes = eyeball_objects(arm_obj)
    used = set()
    at, added, gone = {}, [], {}
    bmap = None
    removal = bool(arm_obj.get("vtmb_eyeballs"))
    parts = eye_mesh_parts(m, source, blanks_here(arm_obj, source)) if removal else {}
    # A model record carries no name of its own in the description, so the name comes off
    # the parsed file walked beside it -- same file, same order.
    for bi, (bp, dbp) in enumerate(zip(m.bodyparts, d.bodyparts)):
        for mi, (model, mr) in enumerate(zip(bp.models, dbp.kids)):
            recs = mr.extra.setdefault("eyes", [])
            named = eyes.get(model.name) or []
            drew = {x.materialparam for x in model.meshes if x.materialtype == 1}
            have = parts.get((bi, mi), {})
            keep = []
            for i, rec, obj in _eye_pairs(recs, named or eyes.get("") or []):
                if obj is None and not removal:
                    out["eye_records"].append((model.name, i))
                    keep.append((i, rec, None))
                    continue
                if obj is None:
                    if i in have:
                        raise build_mod.Refused(
                            "eyeball %d of model %r has no empty, and %s still draws it. "
                            "Delete the eye's faces as well to remove the eyeball, which "
                            "drops its mesh, or restore the empty"
                            % (i, model.name, " and ".join(have[i])))
                    gone.setdefault((bi, mi), set()).add(i)
                    out["eyes_removed"].append((model.name, i))
                    continue
                # 4 of the 602 shipped records are drawn by no mesh, so only one the donor
                # drew has a mesh to lose.
                if removal and i in drew and i not in have:
                    raise build_mod.Refused(
                        "eyeball %d of model %r still has its empty %r, and no mesh draws "
                        "it any more. Delete the empty as well to remove the eyeball, or "
                        "restore the eye's faces" % (i, model.name, obj.name))
                keep.append((i, rec, obj))
            if (bi, mi) in gone:
                recs[:] = [rec for _i, rec, _o in keep]
                out["changed"] += 1
            pos = at[bi, mi] = {i: k for k, (i, _r, _o) in enumerate(keep)}
            for i, rec, obj in keep:
                if obj is None:
                    continue
                used.add(obj.name)
                if bmap is None:
                    bmap = bone_map(m, arm_obj)
                out["changed"] += _apply_eyeball(
                    d, rec, obj, scale, out,
                    _accessory_bone(obj, m, arm_obj, bmap, "eyeball"))
            for obj in named:
                i = int(obj["vtmb_eyeball"])
                if i in pos:
                    continue
                if bmap is None:
                    bmap = bone_map(m, arm_obj)
                rec = build_mod.Rec(bytearray(140))
                struct.pack_into("<i", rec.raw, 0x04,
                                 _accessory_bone(obj, m, arm_obj, bmap, "eyeball"))
                _apply_eyeball(d, rec, obj, scale, out)
                pos[i] = len(recs)
                recs.append(rec)
                used.add(obj.name)
                added.append((obj.name, model.name, i, bi, mi, pos[i]))
                out["changed"] += 1
    _lid_targets_refused(out["lid_past_radius"])
    out["eye_empties"] = [(obj.name, model, int(obj["vtmb_eyeball"]))
                          for model, objs in sorted(eyes.items()) for obj in objs
                          if obj.name not in used]
    meshed = set()
    out["eye_gone"] = gone
    if source is not None:
        out["eye_meshes"], meshed = _apply_eye_meshes(d, m, source, at,
                                                      gone if removal else None)
    out["eyes_added"] = [(name, model, i, k, (bi, mi, k) in meshed)
                         for name, model, i, bi, mi, k in added]
    out["changed"] += _apply_mouth(d, m, arm_obj, out)
    return out


def apply_appended_eyes(d, m, arm_obj, scale, rows):
    """Eyeball records for the models this export appended, and their eye meshes' fields.

    `apply_face` runs before the append and walks the donor's models only, so an empty
    naming an appended object's model (`<object>.smd`) is written here: one record per
    eyeball index in index order, and each mesh its object's `vtmb_eye_slots` names set to
    `materialtype` 1 and the record's position. The slot in that stamp is the object's own,
    which is what `split_mesh`'s runs carry. Answers apply_face's keys for what it wrote,
    plus `used`, the empties it took.
    """
    out = {"changed": 0, "materials": [], "radius_keys": [], "lid_targets": [],
           "lid_past_radius": [], "eyes_added": [], "eye_meshes": 0, "used": set()}
    eyes = eyeball_objects(arm_obj)
    bmap = None
    for row in rows:
        model = row["name"] + ".smd"
        named = sorted(eyes.get(model) or (), key=lambda o: (int(o["vtmb_eyeball"]), o.name))
        if not named:
            continue
        mr = d.bodyparts[row["bodypart"]].kids[0]
        recs = mr.extra.setdefault("eyes", [])
        pos = {}
        for obj in named:
            i = int(obj["vtmb_eyeball"])
            if i in pos:
                continue
            if bmap is None:
                bmap = bone_map(m, arm_obj)
            rec = build_mod.Rec(bytearray(140))
            struct.pack_into("<i", rec.raw, 0x04,
                             _accessory_bone(obj, m, arm_obj, bmap, "eyeball"))
            _apply_eyeball(d, rec, obj, scale, out)
            pos[i] = len(recs)
            recs.append(rec)
            out["used"].add(obj.name)
        meshed = set()
        for slot, param in row["object"].get("vtmb_eye_slots") or ():
            k = pos.get(int(param))
            if k is None:
                continue
            for j, (s, _r, _n) in enumerate(row["slots"]):
                if s == int(slot):
                    struct.pack_into("<2i", mr.kids[j].raw, 0x18, 1, k)
                    out["eye_meshes"] += 1
                    meshed.add(k)
        out["eyes_added"] += [(obj.name, model, int(obj["vtmb_eyeball"]),
                               pos[int(obj["vtmb_eyeball"])],
                               pos[int(obj["vtmb_eyeball"])] in meshed)
                              for obj in named if obj.name in out["used"]]
    _lid_targets_refused(out["lid_past_radius"])
    return out


def blanks_here(arm_obj, source):
    """Whether a model of `source` whose object is gone is written with no vertex: the
    scene's import made an object for every model of that same file."""
    return bool(source) and same_file(arm_obj.get("vtmb_mesh_models"), source)


def eye_mesh_drops(m, source, gone):
    """{(bodypart, model): {file mesh}}: the eye meshes of the eyeballs `apply_face`
    removed, by the `vtmb_eye_slots` entry naming each. `apply_face` refused any whose
    slot still holds a face, so these are written with no vertex and dropped."""
    out = {}
    for (bi, mi), obj in mesh_objects(m, source).items():
        stash, stamp = obj.get("vtmb_meshes") or (), obj.get("vtmb_eye_slots") or ()
        for slot, param in stamp:
            if int(param) not in gone.get((bi, mi), ()):
                continue
            for j, row in enumerate(stash):
                if int(row[0]) == int(slot):
                    out.setdefault((bi, mi), set()).add(j)
    return out


def eye_mesh_parts(m, source, blanks=False):
    """{(bodypart, model): {eyeball index: [what draws it]}}: the eye meshes in the scene.

    A `vtmb_eye_slots` entry counts while its slot, resolved through `slot_moves`, still
    holds a face, so deleting an eye's faces is what takes its mesh away. A model no scene
    object supplies is carried as the file has it, so its file's eye meshes count -- unless
    `blanks`, where the export writes such a model with no vertex and so no eye.
    """
    out = {}
    objs = mesh_objects(m, source) if source is not None else {}
    for bi, bp in enumerate(m.bodyparts):
        for mi, mo in enumerate(bp.models):
            at = out.setdefault((bi, mi), {})
            obj = objs.get((bi, mi))
            if obj is None and blanks and drawn_model(mo):
                continue
            if obj is None:
                for j, x in enumerate(mo.meshes):
                    if x.materialtype == 1:
                        at.setdefault(x.materialparam, []).append(
                            "file mesh %d of %r, which no object in the scene supplies"
                            % (j, mo.name))
                continue
            moves = slot_moves(obj)[0]
            faces = collections.Counter(p.material_index for p in obj.data.polygons)
            mats = obj.data.materials
            for slot, param in obj.get("vtmb_eye_slots") or ():
                here = int(slot) if moves is None else moves.get(int(slot))
                if here is not None and faces[here]:
                    mat = mats[here].name if here < len(mats) and mats[here] else "?"
                    at.setdefault(int(param), []).append(
                        "material slot %d (%s) of %r" % (here, mat, obj.name))
    return out


def _apply_eye_meshes(d, m, source, at, gone=None):
    """Each mesh a `vtmb_eye_slots` stamp names, drawn as the eyeball it names.

    The stamp is [slot, eyeball index] per eye mesh, the slot being the one `vtmb_meshes`
    records for that file mesh, so it reads the same after a slot reorder. `at` is
    {(bodypart, model): {eyeball index: record position}}. A mesh no stamp names keeps
    the donor's fields, so an untouched import writes what it read. Answers (meshes
    rewritten, {(bodypart, model, record position) some mesh draws}).

    `gone` is {(bodypart, model): {eyeball index removed}}, given where the scene can
    remove an eyeball. Then every donor eye mesh is rewritten: an ordinary mesh, (0, 0) as
    on 9143 of 9143 shipped, where a scene object supplies the model, since its stamp says
    which are eyes, and its eyeball's new position where none does.
    """
    objs = mesh_objects(m, source)
    want = {}
    if gone is not None:
        for bi, dbp in enumerate(d.bodyparts):
            for mi, mr in enumerate(dbp.kids):
                pos = at.get((bi, mi), {})
                for j, kid in enumerate(mr.kids):
                    kind, param = struct.unpack_from("<2i", kid.raw, 0x18)
                    if kind == 1 and (bi, mi) in objs:
                        want[bi, mi, j] = (0, 0)
                    elif kind == 1 and param in pos:
                        want[bi, mi, j] = (1, pos[param])
    for (bi, mi), obj in sorted(objs.items()):
        stash, stamp = obj.get("vtmb_meshes"), obj.get("vtmb_eye_slots")
        if not stash or not stamp:
            continue
        kids = d.bodyparts[bi].kids[mi].kids
        pos = at.get((bi, mi), {})
        for slot, param in stamp:
            # A removed eyeball's entry is one whose faces went, or `apply_face` refused.
            if gone is not None and int(param) in gone.get((bi, mi), ()):
                continue
            for j in range(min(len(stash), len(kids))):
                if int(stash[j][0]) != int(slot):
                    continue
                k = pos.get(int(param))
                if k is None:
                    raise build_mod.Refused(
                        "%s: file mesh %d is the eye of eyeball %d, and the model has no "
                        "such record and no eyeball empty claims it. Restore the empty, or "
                        "delete vtmb_eye_slots from the object's custom properties to "
                        "write the mesh as an ordinary one" % (obj.name, j, int(param)))
                want[bi, mi, j] = (1, k)
    changed, meshed = 0, set()
    for (bi, mi, j), v in sorted(want.items()):
        raw = d.bodyparts[bi].kids[mi].kids[j].raw
        if v[0] == 1:
            meshed.add((bi, mi, v[1]))
        if struct.unpack_from("<2i", raw, 0x18) != v:
            struct.pack_into("<2i", raw, 0x18, *v)
            changed += 1
    return changed, meshed


def _lid_targets_refused(bad):
    """Refused over [(eyeball empty, target component, value, radius)], if any."""
    if bad:
        raise build_mod.Refused(
            "a lid target has to lie inside its eyeball's radius, since the game takes "
            "asin(target / radius) and past the radius that has no value and Blender "
            "disables the lid driver: %s. Move the target inside or enlarge the sphere"
            % "; ".join("%s %s is %g, radius %g" % x for x in bad))


def _eye_pairs(recs, want):
    """[(index, record, empty or None)] over one model's eyeball records.

    Record i is the empty stamped `vtmb_eyeball` i, the first by name when a copy claims
    the same index; `want` comes sorted that way from `eyeball_objects`.
    """
    claim = {}
    for obj in want:
        claim.setdefault(int(obj["vtmb_eyeball"]), obj)
    return [(i, rec, claim.get(i)) for i, rec in enumerate(recs)]


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
    if not mouth:
        # Decision 21: on a scene the import stamped, no stash is a mouth removed. Nothing
        # indexes a mouth record, so the rest of the file stands.
        if d.mouths and arm_obj.get("vtmb_mouths"):
            del d.mouths[:]
            out["mouth_removed"] = True
            return 1
        return 0
    if not d.mouths:
        return _add_mouth(d, m, arm_obj, mouth, out)
    raw = d.mouths[0].raw
    have = struct.unpack_from("<i3fi", raw, 0)
    # A key absent means the scene never said, so the file's own value stands -- the rule
    # a spring bone's end bone and an out-of-range blend cell already follow.
    bone = str(mouth.get("bone") or "")
    bi = have[0]
    if bone:
        bi = next((b.index for b in m.bones if b.name == bone), None)
        if bi is None:
            raise SceneRefused("the mouth is on bone %r, which is not one this model "
                               "carries" % bone)
    fwd = (tuple(float(c) for c in mouth["forward"])
           if mouth.get("forward") is not None else have[1:4])
    flex = str(mouth.get("flex") or "")
    fi = build_mod.flexdesc_index(d, flex) if flex else have[4]
    if (have[0] != bi or have[4] != fi
            or not all(_same(a, b) for a, b in zip(have[1:4], fwd))):
        struct.pack_into("<i3fi", raw, 0, bi, fwd[0], fwd[1], fwd[2], fi)
        return 1
    return 0


def _add_mouth(d, m, arm_obj, mouth, out):
    """One mstudiomouth_t on a donor that carries none, off `vtmb_mouth` alone.

    No record holds a value to keep, so a field the stash lacks is refused, as the no-donor
    export refuses it. No mesh field marks a mouth's mesh on any shipped file, so the record
    is the whole of it. The bone is a file name where the import stamped it and a Blender
    name where the panel's picker set it, so both are tried.
    """
    bone = str(mouth.get("bone") or "")
    flex = str(mouth.get("flex") or "")
    fwd = mouth.get("forward")
    lack = [k for k, v in (("bone", bone), ("forward", fwd), ("flex", flex))
            if v is None or v == ""]
    if lack:
        raise build_mod.Refused("the armature's mouth carries no %s, and the donor has no "
                                "mouth to keep a value from -- set it in the armature's VTMB "
                                "eyeballs and mouth panel" % " and no ".join(lack))
    bmap = bone_map(m, arm_obj)
    bi = next((b.index for b in m.bones if b.name == bone), None)
    if bi is None:
        bi = next((b.index for b in m.bones if bmap.get(b.name) == bone), None)
    if bi is None:
        raise SceneRefused("the mouth is on bone %r, which is not one this model carries"
                           % bone)
    raw = bytearray(20)
    struct.pack_into("<i3fi", raw, 0, bi, *[float(c) for c in fwd],
                     build_mod.flexdesc_index(d, flex))
    d.mouths.append(build_mod.Rec(raw))
    out["mouth_added"] = True
    return 1


def _eye_material(d, raw, off, name, added):
    """The iris or glint index at `off`, resolved by name, or the record's own if none.

    The import stamps "" where the index names no texture: walkie_talki1 and sabbat_hand
    hold iris 8 and 10, glint 9, on 4 textures. Resolving "" appended a texture named "".
    """
    name = str(name or "")
    if not name:
        return struct.unpack_from("<i", raw, off)[0]
    return _material_index(d, name, added)


def _apply_eyeball(d, rec, obj, scale, out, bone=None):
    """Rewrite one mstudioeyeball_t off its empty; 1 if any field moved. `bone`, where
    given, is the file bone index the empty is on, written to +0x04."""
    raw = rec.raw
    local = bone_space(obj)
    org = tuple(local[r][3] / scale for r in range(3))
    up = tuple(local.col[1][:3])
    fw = tuple(local.col[2][:3])
    radius = eyeball_radius(obj, scale)
    iris = _eye_material(d, raw, 0x38, obj.get("vtmb_eyeball_iris"), out["materials"])
    glint = _eye_material(d, raw, 0x40, obj.get("vtmb_eyeball_glint"), out["materials"])
    iscale = float(obj.get("vtmb_eyeball_iris_scale") or 0.0)
    upt = [float(c) for c in (obj.get("vtmb_eyeball_uppertarget") or (0.0, 0.0, 0.0))]
    lot = [float(c) for c in (obj.get("vtmb_eyeball_lowertarget") or (0.0, 0.0, 0.0))]
    lids = [str(x) for x in (obj.get("vtmb_eyeball_lidflexes") or ())]
    # Compared as the float32 the record stores, which is what the game divides.
    out["lid_past_radius"] += [
        (obj.name, "%s[%d]" % (nm, k), t, radius)
        for nm, v in (("uppertarget", upt), ("lowertarget", lot)) for k, t in enumerate(v)
        if abs(write_mod.f32(t)) >= write_mod.f32(radius)]
    # A shipped record says "no lid flexes" by holding zero in all eight, never -1, so an
    # empty scene list writes eight zeros and not a sentinel.
    ids = ([build_mod.flexdesc_index(d, x) if x else 0 for x in lids]
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
    rebone = bone is not None and struct.unpack_from("<i", raw, 0x04)[0] != bone
    if not rebone and all((a == b) if i in ints else
                          (_same_dir(a, b) if 4 <= i <= 9 else _same(a, b))
                          for i, (a, b) in enumerate(zip(have, want))):
        return 0
    if bone is not None:
        struct.pack_into("<i", raw, 0x04, bone)
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


def cloth_volume_objects(arm_obj):
    """({(model, capsule): {end: empty}}, {(model, sphere): empty}) off the import's stamps.

    The model is the flat index `vtmb_index` uses, since model names repeat and can be empty.
    """
    caps, sphs = {}, {}
    for obj in arm_obj.children_recursive:
        gi = obj.get("vtmb_cloth_volume_model")
        if gi is None:
            continue
        if obj.get("vtmb_cloth_capsule") is not None:
            key = (int(gi), int(obj["vtmb_cloth_capsule"]))
            end = int(obj.get("vtmb_cloth_capsule_end") or 0)
            ends = caps.setdefault(key, {})
            if end in ends:
                raise build_mod.Refused(
                    "%r and %r both say they are end %d of cloth capsule %d on model %d; "
                    "delete one, or add a capsule with Add Cloth Capsule"
                    % (ends[end].name, obj.name, end, key[1], key[0]))
            ends[end] = obj
        elif obj.get("vtmb_cloth_sphere") is not None:
            key = (int(gi), int(obj["vtmb_cloth_sphere"]))
            if key in sphs:
                raise build_mod.Refused("%r and %r both say they are cloth sphere %d on model "
                                        "%d; delete one, or add a sphere with Add Cloth Sphere"
                                        % (sphs[key].name, obj.name, key[1], key[0]))
            sphs[key] = obj
    return caps, sphs


def cloth_volume_radius(obj, scale):
    """A volume's radius in model units: the SPHERE empty's display size times its scale."""
    s = [abs(c) for c in obj.scale]
    if max(s) - min(s) > ACC_EPS * max(1.0, max(s)):
        raise build_mod.Refused("cloth volume %r is scaled %.4g %.4g %.4g, and a sphere has "
                                "one radius" % (obj.name, *obj.scale))
    r = obj.empty_display_size * s[0] / (scale or 1.0)
    if not math.isfinite(r) or r <= 0.0:
        raise build_mod.Refused("cloth volume %r has radius %r, and a collision radius has "
                                "to be positive" % (obj.name, r))
    return r


def cloth_volume_centre(obj, arm_obj, scale):
    """The empty's origin in its bone's space, in model units.

    Through `matrix_parent_inverse` so an empty parented with Ctrl+P reads back where it
    stands; on a bone Blender deleted, `matrix_basis` alone, which is the attachments' rule.
    """
    bone = arm_obj.data.bones.get(obj.parent_bone) if obj.parent_type == "BONE" else None
    local = obj.matrix_basis
    if bone is not None:
        local = (mathutils.Matrix.Translation((0.0, bone.length, 0.0))
                 @ obj.matrix_parent_inverse @ local)
    return tuple(local[r][3] / (scale or 1.0) for r in range(3))


def _capsule_radius(ends, gi, k, scale):
    """One capsule's radius off its two ends: the end whose size moved from the file's wins."""
    r = [cloth_volume_radius(o, scale) for o in ends]
    if _same(r[0], r[1]):
        return r[0]
    was = [float(o.get("vtmb_cloth_radius", math.nan)) for o in ends]
    if _same(r[0], was[0]):
        return r[1]
    if _same(r[1], was[1]):
        return r[0]
    raise build_mod.Refused("cloth capsule %d on model %d has radius %.6g at %r and %.6g at "
                            "%r, and a capsule has one radius; make the two ends the same size"
                            % (k, gi, r[0], ends[0].name, r[1], ends[1].name))


def _volume_blob(recs, was, fmt, nint):
    """(bytes, records differing from the file's), keeping the file's value in every field
    that agrees, so an edit to one end of a capsule leaves the other end's bytes alone."""
    out, n = [], max(0, len(was) - len(recs))
    for i, r in enumerate(recs):
        f = was[i] if i < len(was) else None
        if f is None:
            out.append(struct.pack(fmt, *r))
            n += 1
            continue
        keep = [a == b if k < nint else _same(a, b) for k, (a, b) in enumerate(zip(f, r))]
        out.append(struct.pack(fmt, *[a if s else b for a, b, s in zip(f, r, keep)]))
        n += not all(keep)
    return b"".join(out), n


def apply_cloth_volumes(d, m, arm_obj, scale):
    """Every model's cloth collision capsules and spheres, rebuilt from the scene's empties.

    Answers `{changed, unread, two_bone}`. An armature without `vtmb_cloth_volumes` was
    imported before volumes were drawn or lacks a volume's bone, so its file keeps its own
    and any volume empty on it is `unread`. `two_bone` is every capsule written with its
    ends on two bones, `(model, capsule, bone name, bone name)`: 0x2c014a00 moves each end
    by its own bone, and none of the 227 shipped capsules does it.
    """
    caps, sphs = cloth_volume_objects(arm_obj)
    if not arm_obj.get("vtmb_cloth_volumes"):
        return {"changed": 0, "unread": sum(len(e) for e in caps.values()) + len(sphs),
                "two_bone": []}
    flat = [mr for dbp in d.bodyparts for mr in dbp.kids]
    for gi, _k in list(caps) + list(sphs):
        if not 0 <= gi < len(flat):
            raise build_mod.Refused("a cloth volume empty names model %d, and the file has %d"
                                    % (gi, len(flat)))
    bmap = bone_map(m, arm_obj)
    changed, two = 0, []
    for gi, mr in enumerate(flat):
        recs = []
        for key in sorted(k for k in caps if k[0] == gi):
            if set(caps[key]) != {0, 1}:
                (end, obj), = caps[key].items()
                raise build_mod.Refused("%r is end %d of cloth capsule %d on model %d and the "
                                        "other end is gone; delete it too, or add a capsule "
                                        "with Add Cloth Capsule" % (obj.name, end, key[1], gi))
            ends = (caps[key][0], caps[key][1])
            b = [_accessory_bone(o, m, arm_obj, bmap, "cloth capsule end") for o in ends]
            if b[0] != b[1]:
                two.append((gi, key[1], ends[0].parent_bone, ends[1].parent_bone))
            p = [cloth_volume_centre(o, arm_obj, scale) for o in ends]
            recs.append((b[0], b[1], _capsule_radius(ends, gi, key[1], scale)) + p[0] + p[1])
        old = mr.extra.get("clothcollide") or b""
        was = [struct.unpack_from("<2i7f", old, i * 36) for i in range(len(old) // 36)]
        blob, n = _volume_blob(recs, was, "<2i7f", 2)
        if blob != old:
            mr.extra["clothcollide"] = blob
            changed += n
        recs = [(_accessory_bone(sphs[key], m, arm_obj, bmap, "cloth sphere"),
                 cloth_volume_radius(sphs[key], scale))
                + cloth_volume_centre(sphs[key], arm_obj, scale)
                for key in sorted(k for k in sphs if k[0] == gi)]
        old = mr.extra.get("clothsphere") or b""
        was = [struct.unpack_from("<if3f", old, i * 20) for i in range(len(old) // 20)]
        blob, n = _volume_blob(recs, was, "<if3f", 1)
        if blob != old:
            mr.extra["clothsphere"] = blob
            changed += n
    return {"changed": changed, "unread": 0, "two_bone": two}


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


def seq_tombstones(arm_obj):
    """[(label, src or None)] for each removal `vtmb_seqs_removed` holds, in the order made.

    An entry is a bare label where a scene before 2026-10-27 wrote it, else {label, src}:
    `src` is the file ordinal and outlives a rename of the entry, which the label does not.
    """
    out = []
    for x in arm_obj.get("vtmb_seqs_removed") or ():
        if isinstance(x, str):
            out.append((x, None))
        else:
            src = x.get("src")
            out.append((str(x.get("label") or ""), None if src is None else int(src)))
    return out


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

    The list itself is the scene's as well. `vtmb_seqs_removed` names what a removal
    tombstoned, by `src` where it carries one and else by label, `src` on an entry is the
    file ordinal it came from and is what a reorder is resolved through, and a trailing
    entry marked `new` is one the scene authored, which `add_sequence` appends. All three
    are positive marks rather than absences, the spring-chain rule: a stash shorter than
    the file with nothing tombstoned leaves the trailing records alone. A stash with no
    `src` is one an import before it made, which `old_imports` refuses, so a `src` past
    the file's list is what matches by position.

    Returns (what moved, those cells, what the list edits did).
    """
    stash = list(arm_obj.get("vtmb_sequences") or ())
    moves = {"added": [], "removed": [], "reordered": 0, "orphans": [],
             "autolayers_lost": [], "dangling": [], "missing": [], "twice": [],
             "scene_label": {}}
    index = {n: k for k, n in enumerate(anim_names)}
    seen, carried_twice = set(), set()
    for name in anim_names:
        (carried_twice if name in seen else seen).add(name)
    n = 0
    # The donor's own: 39 of the 4445 shipped models play an animation from two sequences
    # and 19 carry one no sequence plays, so only what this export changes is reported.
    was_cited = _cited_counts(d)
    was_orphan = {anim_names[k] for k in range(len(anim_names)) if k not in was_cited}

    # Resolved to the record OBJECT before anything is dropped: a removal renumbers every
    # later ordinal, and a `src` names the file as the import read it.
    byord = list(d.seqs)
    for label, src in seq_tombstones(arm_obj):
        if src is not None and 0 <= src < len(byord):
            at = [k for k, r in enumerate(d.seqs) if r is byord[src]]
        else:
            at = [k for k, r in enumerate(d.seqs) if (r.name or "") == label]
        if not at:
            # The donor no longer has it, so the tombstone has nothing to answer. Reported
            # rather than refused: an export to another path leaves the mark standing, and
            # the next export to that path removes the sequence off the donor again.
            moves["missing"].append(label)
            continue
        was, lost, orphans, naming = build_mod.remove_sequence(d, at[0])
        moves["removed"].append(was)
        if label and label != was:
            moves["scene_label"][was] = label
        moves["autolayers_lost"] += lost
        moves["dangling"] += naming
        moves["orphans"] = [a for a in orphans if a not in was_orphan]
        n += 1

    fresh = 0
    while fresh < len(stash) and stash[len(stash) - 1 - fresh].get("new"):
        fresh += 1
    head = stash[:len(stash) - fresh]
    for k, x in enumerate(head):
        if x.get("new"):
            raise SceneRefused("the armature's sequence %d of %d is one the scene added, and "
                               "an added sequence is appended, so it has to be the last"
                               % (k, len(stash)))
    if len(head) > len(d.seqs):
        raise SceneRefused("the armature carries %d sequence%s the file is supposed to already "
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
            raise SceneRefused("the sequence the scene added blends animation %r, which the "
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
        hint = [list(col) for col in s.get("blend_index") or []]
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
                    raise SceneRefused("sequence %r blends animation %r, which the file does "
                                       "not have" % (rec.name, name))
                if name in carried_twice:
                    # The name cannot say which copy, and `index` holds the last: the
                    # stash's own index where it still names this animation, else the
                    # word the record carries.
                    h = hint[x][y] if x < len(hint) and y < len(hint[x]) else None
                    w = struct.unpack_from("<h", rec.raw, at)[0]
                    if (h is not None and 0 <= int(h) < len(anim_names)
                            and anim_names[int(h)] == name):
                        a = int(h)
                    elif 0 <= w < len(anim_names) and anim_names[w] == name:
                        a = w
                    else:
                        raise SceneRefused(
                            "sequence %r blends animation %r, which the file carries %d "
                            "times, and neither the stash nor the record says which"
                            % (rec.name, name, anim_names.count(name)))
                if struct.unpack_from("<h", rec.raw, at)[0] != a:
                    n += 1
                struct.pack_into("<h", rec.raw, at, a)
        n += _apply_events(rec, s.get("events"))
        n += _apply_params(rec, s.get("params"), pp, rec.name)
        n += _apply_seq_tail(rec, s, labels, bones)
    # After the grid, not inside the loop: which animation a sequence names is what picks
    # the action, and the stash has just rewritten it.
    for k, acts in _first_citers(d, actions).items():
        n += _apply_seq_action(d.seqs[k], acts)
    # Last, off the grid the stash has just written: `_action_seq` reaches only the
    # earlier of two sequences playing one animation.
    moves["twice"] = [anim_names[a] if 0 <= a < len(anim_names) else "animation %d" % a
                      for a, c in sorted(_cited_counts(d).items())
                      if c > 1 and was_cited.get(a, 0) < 2]
    return n, out_of_range, moves


def _cited_counts(d):
    """{animation index: how many sequences play it}."""
    cited = {}
    for rec in d.seqs:
        for a in set(build_mod._seq_anims(rec.raw)):
            cited[a] = cited.get(a, 0) + 1
    return cited


def _first_citers(d, actions):
    """{sequence index: [action, ...]}, each animation going to the first sequence naming it.

    The import stamps `vtmb_seq_flags` and `vtmb_activity` from the first sequence citing
    each animation, so that is the one sequence an action can be read back into. Handing
    the value to every citer instead would let one sequence's flags overwrite another's on
    a plain re-export, which is the one thing this must not cost. A blend grid is the first
    citer of every animation in it, so its sequence gets all of their actions -- eight on
    andrei's `run`, 591 animations over 23 shipped models.
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
                    out.setdefault(k, []).append(actions[a])
    return out


def _seq_action_value(rec, acts, key, have, conv):
    """The one value `acts` set `key` to where it differs from `have`, else `have`.

    Every action of one sequence was stamped with the same value, so the one that differs
    is the edit. Two that differ from the file and from each other cannot both be written.
    """
    got = {}
    for act in acts:
        v = act.get(key)
        if v is not None and conv(v) != have:
            got.setdefault(conv(v), []).append(act.name)
    if len(got) > 1:
        raise build_mod.Refused(
            "sequence %r is the first to cite %s, which set %s to %s; a sequence holds one"
            % (rec.name, ", ".join(sorted(n for names in got.values() for n in names)),
               key, " and ".join(repr(v) for v in got)))
    return next(iter(got), have)


def _apply_seq_action(rec, acts):
    """The two sequence fields the panel puts on an Action, after the stash has had its
    say. Returns what moved.

    An action carrying neither key says nothing and the stash stands; `vtmb_activity`
    present and empty is a sequence with no activity, which is the only way a donor's own
    comes off.
    """
    n = 0
    have = struct.unpack_from("<i", rec.raw, mdl_mod.SEQ_FLAGS)[0]
    flags = _seq_action_value(rec, acts, "vtmb_seq_flags", have, int)
    if flags != have:
        struct.pack_into("<i", rec.raw, mdl_mod.SEQ_FLAGS, flags)
        n += 1
    have = rec.extra.get("activity")
    activity = _seq_action_value(rec, acts, "vtmb_activity", have, str)
    if activity != have:
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


def cut_vtx_lods(source, dest, written=(), carry=True):
    """Every `.vtx` beside the written model turned down to one LOD. One row per file.

    Every export runs it: the addon writes LOD 0 and nothing else. The LOD 1..n payload
    leaves the file -- `vtx_write.cut_lods` re-lays the bytes rather than patching the two
    counts -- so `bytes` is what the file lost. A file already at one LOD re-lays byte for
    byte and is not rewritten in place.

    Every flavour beside `dest` is cut, not only the ones the geometry pass rewrote: the
    engine picks by `-dxlevel` and a flavour left at seven LODs still swaps to a coarse
    mesh. A flavour missing there is copied across from the donor only under `carry`, which
    the caller clears once any `.vtx` was rewritten: a flavour that pass did not write still
    describes the donor's geometry, and copying it would put a stale file beside a new model.
    """
    out = []
    for flavour in VTX_FLAVOURS:
        src_p, dst_p = vtx_path(source, flavour), vtx_path(dest, flavour)
        at = dst_p if flavour in written else src_p if carry else dst_p
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
                    "bytes": len(data) - len(cut)})
    return out


def vtx_eye_flags(d, dest):
    """MESH_IS_EYES re-derived from each written mesh's `materialtype` in every `.vtx`
    beside `dest`: [(flavour, mesh headers changed)] per file rewritten.

    The donor's `.vtx` is rewritten only when geometry moved, so a mesh an added eyeball
    made an eye would otherwise keep the donor's 0 there. A file a writer could not walk
    is left alone: `cut_vtx_lods` has already reported it.
    """
    eyed = {(i, j, k) for i, bp in enumerate(d.bodyparts) for j, mr in enumerate(bp.kids)
            for k, mesh in enumerate(mr.kids)
            if struct.unpack_from("<i", mesh.raw, 0x18)[0] == 1}
    out = []
    for flavour in VTX_FLAVOURS:
        p = vtx_path(dest, flavour)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            data = f.read()
        try:
            new, n = vtxw_mod.set_eye_flags(data, eyed)
        except struct.error:
            continue
        if n:
            with open(p, "wb") as f:
                f.write(new)
            out.append((flavour, n))
    return out


def phy_path(p):
    return (p[:-4] if p[-4:].lower() == ".mdl" else p) + ".phy"


def donor_phy(source, content=None):
    """(where, bytes) of the donor's `.phy`, or (None, None) when it has none.

    Beside the donor first, then with `content` the same relative path on every root and in
    its packs, as the engine finds it: 102 of the 505 loose models in the shipped install
    keep their `.phy` only in a pack or under another root.
    """
    src = phy_path(source)
    if not os.path.isfile(src):
        if content is None:
            return None, None
        src, blob = content.companion(types.SimpleNamespace(path=source), (".phy",))
        if src is None:
            return None, None
        if blob is not None:
            return src, blob
    with open(src, "rb") as f:
        return src, f.read()


def phy_bone_names(path, names, data=None):
    """Which of `names` the sibling `.phy` cites, or None when there is no `.phy`. `data` is
    that file's bytes when they were read elsewhere, from a pack.

    Collision and ragdoll live in that file, which nothing here writes but `rewrite_phy`'s
    bone rename. It references the model by bone name and by nothing else -- over 2929
    shipped files no key names a vertex, a face, a mesh or a bone index -- so a moved
    vertex leaves collision merely wrong, while a renamed or deleted bone leaves
    `CRagdollProp::CreateObjects` unable to look the solid up at all. Anomalies I.9.

    Matching every quoted value rather than a key list means a key some other tool added
    cannot hide a reference from this.
    """
    ph = phy_path(path)
    if data is None and not os.path.exists(ph):
        return None
    try:
        if data is None:
            with open(ph, "rb") as f:
                data = f.read()
        b = data
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


def phy_moves(d, m, donor_pose):
    """`phy.drop_solids`' `moves` for the bones this export removed: {removed bone: (the
    parent it is followed up to, the first that survives; the 3x4 from the removed bone's
    donor space to that bone's space as written; that bone's rest transform in model space)},
    file names throughout. `donor_pose` is every donor bone's `posetobone`, read before the
    removal restamped `d`'s."""
    at = {b.name: k for k, b in enumerate(m.bones)}
    out = {}
    for b in getattr(m, "gone", {}):
        p = m.gone[b]
        while p in m.gone:
            p = m.gone[p]
        if p not in at:
            continue
        flat = struct.unpack_from("<12f", d.bones[at[p]].raw, 0x58)
        p2b = [list(flat[0:4]), list(flat[4:8]), list(flat[8:12])]
        out[b] = (p, mdl_mod.mat_mul(p2b, mdl_mod.mat_inverse(donor_pose[b])),
                  mdl_mod.mat_inverse(p2b))
    return out


# Each `solid` key a hull object carries as `vtmb_phy_<key>`, and the joint limits on a
# joint's child. `volume` is not one: the export takes it off the hull it writes.
HULL_KEYS = (("mass", float), ("surfaceprop", str), ("damping", float),
             ("rotdamping", float), ("inertia", float), ("massbias", float))
JOINT_KEYS = tuple(a + k for a in "xyz" for k in ("min", "max", "friction"))


def hull_objects(arm_obj):
    """The armature's collision hulls: its mesh children carrying `vtmb_phy_hull`."""
    return sorted((o for o in arm_obj.children
                   if o.type == "MESH" and o.get("vtmb_phy_hull")), key=lambda o: o.name)


def hull_matrix(obj):
    """A hull's matrix into its solid's space: its bone's for a bone child, read as
    `bone_space` reads an empty, else the armature's."""
    if obj.parent_type == "BONE":
        return bone_space(obj)
    return obj.matrix_parent_inverse @ obj.matrix_basis


def loose_parts(mesh):
    """Each loose part's vertex indices, ascending, the parts in order of their first."""
    up = list(range(len(mesh.vertices)))

    def find(i):
        while up[i] != i:
            up[i] = up[up[i]]
            i = up[i]
        return i

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            up[max(a, b)] = min(a, b)

    ev = array.array("i", [0]) * (2 * len(mesh.edges))
    mesh.edges.foreach_get("vertices", ev)
    for a, b in zip(ev[0::2], ev[1::2]):
        join(a, b)
    for p in mesh.polygons:
        vs = p.vertices
        for a in vs[1:]:
            join(vs[0], a)
    parts = {}
    for i in range(len(up)):
        parts.setdefault(find(i), []).append(i)
    return [parts[k] for k in sorted(parts)]


def _mesh_co(mesh):
    co = array.array("f", [0.0]) * (3 * len(mesh.vertices))
    mesh.vertices.foreach_get("co", co)
    return co


def hull_unchanged(obj, s, scale):
    """Whether `obj` is still what `blender_import.build_hulls` made of surface `s`: the
    same vertices to the bit, in the same loose parts, and no transform."""
    verts, _faces, starts = phy_mod.ledge_mesh(s, scale)
    me = obj.data
    if len(me.vertices) != len(verts) or hull_matrix(obj) != mathutils.Matrix.Identity(4):
        return False
    if _mesh_co(me) != array.array("f", [c for v in verts for c in v]):
        return False
    ends = starts[1:] + [len(verts)]
    return loose_parts(me) == [list(range(a, b)) for a, b in zip(starts, ends)]


def hull_surface(obj, scale, tag=phy_mod.IVPS):
    """`obj`'s loose parts as one surface, each part's convex hull one ledge, the points
    rounded to the floats the file holds before they are hulled: (surface, [(part, vertices
    inside its hull)]). A part with a vertex deeper inside its hull than `phy.DENT_TOL` of
    its size is one the ledge would fill, and stays out of the surface."""
    me = obj.data
    co = _mesh_co(me)
    m = [list(r) for r in hull_matrix(obj)]
    parts = loose_parts(me)
    at = {}
    for n, part in enumerate(parts):
        for j, i in enumerate(part):
            at[i] = (n, j)
    faces = [[] for _ in parts]
    me.calc_loop_triangles()
    for t in me.loop_triangles:
        n = at[t.vertices[0]][0]
        faces[n].append(tuple(at[i][1] for i in t.vertices))
    ledges, dents = [], []
    for n, part in enumerate(parts):
        pts = []
        for i in part:
            x, y, z = co[3 * i:3 * i + 3]
            hl = [(r[0] * x + r[1] * y + r[2] * z + r[3]) / scale for r in m[:3]]
            pts.append(tuple(phy_mod._f32(c) for c in phy_mod.hl_to_ivp(hl)))
        try:
            tris, deep = phy_mod.part_hull(pts, faces[n])
        except phy_mod.PhyError as e:
            raise SceneRefused("%s: loose part %d makes no convex piece (%s). Nothing was "
                               "written" % (obj.name, n, e)) from None
        if deep:
            dents.append((n, len(deep)))
            continue
        ledges.append([tuple(pts[i] for i in t) for t in tris])
    if dents or not ledges:
        return None, dents
    try:
        return phy_mod.build_surface(ledges, tag), dents
    except phy_mod.PhyError as e:
        raise SceneRefused("%s: %s. Nothing was written" % (obj.name, e)) from None


def _float_changed(value, text):
    try:
        return phy_mod._f32(float(value)) != phy_mod._f32(float(text))
    except (TypeError, ValueError):
        return True


def read_hulls(arm_obj, m, data, scale, names):
    """What the armature's collision hulls change in the donor's `.phy` bytes `data`, or
    None when the import did not give every solid its object (`vtmb_phy_solids`).

    `names` is {armature bone name: file bone name} over the bones being written. A hull
    on a bone the armature lost is left to `drop_solids`. Raises `SceneRefused` on a
    dent, a part that makes no convex piece, two objects claiming one solid, or two
    solids on one bone, before anything is written.

    {"surfaces": {ordinal: rebuilt Surface}, "values": {ordinal: {key: text}}, "joints":
    {child ordinal: {key: text}}, "drop": [solid names], "add": [(file bone, Surface,
    object)], "rebuilt": [(solid, volume was, volume now)], "edited": [(solid, key, was,
    now)]}, keys as `phy.edit_solids` takes them."""
    if not arm_obj.get("vtmb_phy_solids"):
        return None
    f = phy_mod.Phy(data)
    solids, joints, _edit = phy_mod.text_values(data)
    low = {}
    for b in m.bones:
        low.setdefault(b.name.lower(), b.name)
    claims, new, gone_bone = {}, [], set()
    for o in hull_objects(arm_obj):
        fb = None
        if o.parent_type == "BONE":
            fb = names.get(o.parent_bone)
        k = o.get("vtmb_phy_solid")
        k = int(k) if isinstance(k, int) and 0 <= k < len(f.solids) else None
        if o.parent_type == "BONE" and fb is None:
            if k is not None:
                gone_bone.add(k)
            continue
        if k is None:
            new.append((o, fb))
        else:
            claims.setdefault(k, []).append((o, fb))
    for k, objs in sorted(claims.items()):
        if len(objs) > 1:
            raise SceneRefused(
                "%s all carry vtmb_phy_solid %d, the solid of %r, and the file holds one "
                "solid for it. A Shift+D copy keeps the stamp: delete the copies, or clear "
                "vtmb_phy_solid on them to add each as a solid of its own. Nothing was "
                "written" % (", ".join(repr(o.name) for o, _ in objs), k,
                             solids[k].get("name", "")))
    out = {"surfaces": {}, "values": {}, "joints": {}, "drop": [], "add": [],
           "rebuilt": [], "edited": []}
    dents = []
    for k, (s, kv) in enumerate(zip(f.solids, solids)):
        if k in gone_bone:
            continue
        label = kv.get("name", "")
        if k not in claims:
            out["drop"].append(label)
            continue
        o, fb = claims[k][0]
        if fb != low.get(label.lower()):
            out["drop"].append(label)
            new.append((o, fb))
            continue
        vals = {}
        for key, kind in HULL_KEYS:
            v = o.get("vtmb_phy_" + key)
            if v is None or key not in kv:
                continue
            if kind is float and _float_changed(v, kv[key]):
                vals[key] = phy_mod.float_text(float(v))
            elif kind is str and str(v) != kv[key]:
                vals[key] = str(v)
        if k in joints:
            jv = {}
            for key in JOINT_KEYS:
                v = o.get("vtmb_phy_" + key)
                if v is not None and key in joints[k] and _float_changed(v, joints[k][key]):
                    jv[key] = phy_mod.float_text(float(v))
            if jv:
                out["joints"][k] = jv
                out["edited"] += [(label, key, joints[k][key], t) for key, t in jv.items()]
        if not hull_unchanged(o, s, scale):
            # The donor's tag; `rewrite_phy` tags every solid of a file with more than one.
            surf, bad = hull_surface(o, scale, s.dummy[8:])
            dents += [(o.name, label, n, c) for n, c in bad]
            if surf is not None:
                out["surfaces"][k] = surf
                if "volume" in kv and "volume" not in vals:
                    vals["volume"] = phy_mod.float_text(phy_mod.surface_volume(surf))
                    out["rebuilt"].append((label, kv["volume"], vals["volume"]))
        if vals:
            out["values"][k] = vals
            out["edited"] += [(label, key, kv[key], t) for key, t in vals.items()
                              if key != "volume" or k not in out["surfaces"]]
    taken = {n.lower() for n in (solids[k].get("name", "") for k in range(len(solids)))}
    taken -= {n.lower() for n in out["drop"]}
    for o, fb in new:
        if fb is None:
            raise SceneRefused(
                "%s is a collision hull parented to the armature itself, and a solid is "
                "named for its bone: parent it to one (Ctrl+P > Bone). Nothing was written"
                % o.name)
        if fb.lower() in taken:
            raise SceneRefused(
                "%s would be a second solid on bone %r, and the engine finds a bone's solid "
                "by name, first match winning: delete one of the two hulls. Nothing was "
                "written" % (o.name, fb))
        taken.add(fb.lower())
        surf, bad = hull_surface(o, scale)
        dents += [(o.name, fb, n, c) for n, c in bad]
        if surf is not None:
            out["add"].append((fb, surf, o))
    if dents:
        raise SceneRefused(
            "every loose part of a collision hull is written as its convex hull, and %s. "
            "Move those vertices back out, or cut the part into convex pieces, each a loose "
            "part of the same object. Nothing was written"
            % "; ".join("%s (solid %r) loose part %d is concave at %d %s"
                        % (o, label, n, c, "vertex" if c == 1 else "vertices")
                        for o, label, n, c in dents))
    return out


def add_hull_solids(data, adds, m, rest):
    """`data` with a solid appended for each of `adds`, `read_hulls`' (file bone, surface,
    object): its parent the solid of the bone's nearest ancestor that owns one, a joint to
    it with the object's nine limits (0 each unless set, rigid), and the `solid` values
    off the object where it carries them, else studiomdl's: the mass `phy.masses` gives
    it among the file's, the first solid's surfaceprop, damping 0, rotdamping 0, inertia
    1. `rest` is {file bone: its rest transform, a 3x4 in model space}, written as
    `origin`/`angles` when the file's solids carry them. ([solid names], bytes)."""
    at = {b.name: k for k, b in enumerate(m.bones)}
    added = []
    for fb, surf, o in adds:
        solids, _joints, edit = phy_mod.text_values(data)
        own = {}
        for k, kv in enumerate(solids):
            own.setdefault(kv.get("name", "").lower(), k)
        up = m.bones[at[fb]].parent
        while up >= 0 and m.bones[up].name.lower() not in own:
            up = m.bones[up].parent
        if up < 0:
            raise SceneRefused(
                "%s on bone %r would be a solid with no parent solid, a ragdoll in pieces: "
                "give one of its ancestor bones a hull first. Nothing was written"
                % (o.name, fb))
        parent = own[m.bones[up].name.lower()]
        vol = phy_mod.surface_volume(surf)

        def num(key, default):
            v = o.get("vtmb_phy_" + key)
            return float(v) if v is not None else default

        def text(kv, key, default):
            try:
                return float(kv.get(key, default))
            except ValueError:
                return default

        if o.get("vtmb_phy_mass") is not None:
            mass = float(o["vtmb_phy_mass"])
        else:
            mass = phy_mod.masses([text(kv, "volume", 0.0) for kv in solids] + [vol],
                                  [text(kv, "massbias", 1.0) for kv in solids] + [1.0],
                                  text(edit, "totalmass", 1.0))[-1]
        pairs = [("name", fb), ("parent", solids[parent].get("name", ""))]
        if any("origin" in kv for kv in solids) and fb in rest:
            r = rest[fb]
            pairs += [("origin", " ".join(phy_mod.float_text(x[3]) for x in r)),
                      ("angles", " ".join(phy_mod.float_text(a)
                                          for a in phy_mod.matrix_angles(r)))]
        pairs += [("mass", phy_mod.float_text(mass)),
                  ("surfaceprop", _text_prop(o, "surfaceprop",
                                             solids[0].get("surfaceprop", "default"))),
                  ("damping", phy_mod.float_text(num("damping", 0.0))),
                  ("rotdamping", phy_mod.float_text(num("rotdamping", 0.0))),
                  ("inertia", phy_mod.float_text(num("inertia", 1.0))),
                  ("volume", phy_mod.float_text(vol))]
        joint = [("parent", str(parent)), ("child", str(len(solids)))]
        joint += [(key, phy_mod.float_text(num(key, 0.0))) for key in JOINT_KEYS]
        data, _k = phy_mod.add_solid(data, surf, pairs, joint)
        added.append(fb)
    return added, data


def _text_prop(o, key, default):
    """The object's `vtmb_phy_<key>`, an empty one included: 1309 of the 7353 shipped solids
    carry `"surfaceprop" ""`."""
    v = o.get("vtmb_phy_" + key)
    return default if v is None else str(v)


def _dents_text(dents):
    """The error for [(object name, loose part, vertices in a dent)]."""
    return ("every loose part of a collision hull is written as its convex hull, and %s. "
            "Move those vertices back out, or cut the part into convex pieces, each a loose "
            "part of the same object. Nothing was written"
            % "; ".join("%s loose part %d is concave at %d %s"
                        % (o, n, c, "vertex" if c == 1 else "vertices")
                        for o, n, c in dents))


def collision_crc(s):
    """What the import stamps on the collision text's Text as `vtmb_phy_crc`, or None for
    text with no latin-1 form, which no import made."""
    try:
        return "%08x" % zlib.crc32(s.encode("latin-1"))
    except UnicodeEncodeError:
        return None


def scene_collision_text(arm_obj, source=None, content=None):
    """The text to write after `editparams`, out of the Text the import put on the armature
    as `vtmb_phy_collisiontext` (`phy.collision_text`, the `break` blocks): its bytes, b""
    when that Text was deleted, or None when the import made none, which keeps the donor's.
    Raises `SceneRefused` when text that is neither the import's nor, with `source`, the
    donor `.phy`'s fails `phy.check_collision_text`; text as shipped is never checked, and
    two shipped files fail it."""
    if "vtmb_phy_collisiontext" not in arm_obj:
        return None
    t = arm_obj["vtmb_phy_collisiontext"]
    if not isinstance(t, bpy.types.Text):
        return b""
    s = t.as_string()
    was = None
    if source is not None:
        try:
            _where, data = donor_phy(source, content)
            was = phy_mod.collision_text(data) if data is not None else None
        except (OSError, phy_mod.PhyError):
            pass
    crc = collision_crc(s)
    if s != was and (crc is None or t.get("vtmb_phy_crc") != crc):
        errs = phy_mod.check_collision_text(s)
        if errs:
            raise SceneRefused(
                "the collision text %r, the break blocks written after editparams: %s. Fix it "
                "in the Text Editor, or delete the Text to write none. Nothing was written"
                % (t.name, "; ".join(errs)))
    return s.encode("latin-1")


def hulls_phy(arm_obj, d, scale, stem):
    """A `.phy` of the armature's collision hulls alone, for a model with no donor, or a
    donor with no `.phy`, as studiomdl writes one (source-2003 `collisionmodel.cpp:2100`),
    or None when it has no hull. A hull parented to a bone of `d`, named as Blender names
    it, is a solid named for it; the solids go in bone order, each one's parent the solid of
    its bone's nearest ancestor that owns one, with a joint carrying the object's nine
    limits, 0 (rigid) where unset, and `origin`/`angles` the bone's rest transform, as on
    7297 of the 7321 shipped solids. A hull parented to the armature is a single-body prop's
    one solid, in model space, `origin`/`angles` zero as on 2535 of the 2564 shipped, named
    `vtmb_phy_name` or else `<stem>_phy`, the name 1082 of the 2605 shipped single-solid
    files carry. The `solid` values come off the object, else studiomdl's: a share of the
    armature's `vtmb_phy_totalmass` (1.0, its default) by volume, damping 0, rotdamping 0,
    inertia 1, and the surfaceprop of the solid's bone record, or the header's for the
    prop's solid, since studiomdl writes all three from one `GetSurfaceProp`
    (`write.cpp:195`): 4727 of the 4757 shipped solids named for a bone carry their bone's,
    and 2268 of the 2564 others the header's. `scene_collision_text` goes after
    `editparams`. Raises `SceneRefused` on a dent, a hull on a bone the armature has not
    got, two on one bone, a hull on the armature beside any other, more than one solid with
    no parent, or collision text that fails its check.

    {"surfaces": [Surface], "text": bytes, "solids": [bone names], "joints": count,
    "totalmass": float, "parts": loose parts written, "breaks": `break` blocks written}"""
    objs = hull_objects(arm_obj)
    if not objs:
        return None
    tail = scene_collision_text(arm_obj) or b""
    at = {r.name: k for k, r in enumerate(d.bones)}
    whole = [o for o in objs if o.parent_type != "BONE"]
    if whole and len(objs) > 1:
        raise SceneRefused(
            "%s is a collision hull on the armature itself, a single-body prop's one solid, "
            "beside %d other hull%s: parent it to a bone (Ctrl+P > Bone), or delete the "
            "others. Nothing was written"
            % (whole[0].name, len(objs) - 1, "" if len(objs) == 2 else "s"))
    if whole:
        o = whole[0]
        surf, bad = hull_surface(o, scale)
        if bad:
            raise SceneRefused(_dents_text([(o.name, n, c) for n, c in bad]))
        totalmass = float(arm_obj.get("vtmb_phy_totalmass", 1.0))
        vol = phy_mod.surface_volume(surf)

        def num1(key, default):
            v = o.get("vtmb_phy_" + key)
            return float(v) if v is not None else default

        name = str(o.get("vtmb_phy_name") or "%s_phy" % stem)
        solid = {"name": name, "origin": (0.0, 0.0, 0.0), "angles": (0.0, 0.0, 0.0),
                 "mass": num1("mass", phy_mod.masses([vol], [1.0], totalmass)[0]),
                 "surfaceprop": _text_prop(o, "surfaceprop", d.surfaceprop or ""),
                 "damping": num1("damping", 0.0), "rotdamping": num1("rotdamping", 0.0),
                 "inertia": num1("inertia", 1.0), "volume": vol,
                 "massbias": (None if o.get("vtmb_phy_massbias") is None
                              else num1("massbias", 1.0))}
        text = phy_mod.text_half([solid], (), str(arm_obj.get("vtmb_phy_rootname", "")),
                                 totalmass, concave=len(surf.ledges) > 1)
        return {"surfaces": [surf], "text": text[:-1] + tail + b"\0", "solids": [name],
                "joints": 0, "totalmass": totalmass, "parts": len(surf.ledges),
                "breaks": phy_mod.break_count(tail)}
    owner, dents = {}, []
    for o in objs:
        k = at.get(o.parent_bone)
        if k is None:
            raise SceneRefused(
                "%s is a collision hull on bone %r, which the armature has not got, and a "
                "solid is named for its bone: parent it to one (Ctrl+P > Bone). Nothing was "
                "written" % (o.name, o.parent_bone))
        if k in owner:
            raise SceneRefused(
                "%s and %s are both collision hulls on bone %r, and the engine finds a "
                "bone's solid by name, first match winning: delete one of the two. Nothing "
                "was written" % (owner[k][0].name, o.name, o.parent_bone))
        surf, bad = hull_surface(o, scale)
        dents += [(o.name, n, c) for n, c in bad]
        owner[k] = (o, surf)
    if dents:
        raise SceneRefused(_dents_text(dents))
    order = sorted(owner)
    ordinal = {k: i for i, k in enumerate(order)}

    def parent_of(k):
        p = struct.unpack_from("<i", d.bones[k].raw, 0x04)[0]
        while p >= 0 and p not in owner:
            p = struct.unpack_from("<i", d.bones[p].raw, 0x04)[0]
        return p

    roots = [d.bones[k].name for k in order if parent_of(k) < 0]
    if len(roots) > 1:
        raise SceneRefused(
            "the collision hulls on %s each have no hull on any ancestor bone, so the "
            "ragdoll would be in %d pieces: give a bone above them all a hull. Nothing was "
            "written" % (", ".join(repr(n) for n in roots), len(roots)))
    totalmass = float(arm_obj.get("vtmb_phy_totalmass", 1.0))
    vols = [phy_mod.surface_volume(owner[k][1]) for k in order]

    def num(o, key, default):
        v = o.get("vtmb_phy_" + key)
        return float(v) if v is not None else default

    bias = [num(owner[k][0], "massbias", 1.0) for k in order]
    share = phy_mod.masses(vols, bias, totalmass)
    solids, joints = [], []
    for i, k in enumerate(order):
        o = owner[k][0]
        flat = struct.unpack_from("<12f", d.bones[k].raw, 0x58)
        rest = mdl_mod.mat_inverse([list(flat[0:4]), list(flat[4:8]), list(flat[8:12])])
        p = parent_of(k)
        solids.append({"name": d.bones[k].name,
                       "parent": d.bones[p].name if p >= 0 else None,
                       "origin": [r[3] for r in rest],
                       "angles": phy_mod.matrix_angles(rest),
                       "mass": num(o, "mass", share[i]),
                       "surfaceprop": _text_prop(
                           o, "surfaceprop",
                           d.bones[k].extra.get("surfaceprop") or d.surfaceprop or ""),
                       "damping": num(o, "damping", 0.0),
                       "rotdamping": num(o, "rotdamping", 0.0),
                       "inertia": num(o, "inertia", 1.0),
                       "volume": vols[i],
                       "massbias": (None if o.get("vtmb_phy_massbias") is None
                                    else bias[i])})
        if p >= 0:
            joints.append((ordinal[p], i, [num(o, key, 0.0) for key in JOINT_KEYS]))
    surfaces = [owner[k][1] for k in order]
    parts = sum(len(s.ledges) for s in surfaces)
    text = phy_mod.text_half(solids, joints, str(arm_obj.get("vtmb_phy_rootname", "")),
                             totalmass, concave=parts > len(surfaces))
    return {"surfaces": surfaces, "text": text[:-1] + tail + b"\0",
            "solids": [s["name"] for s in solids], "joints": len(joints),
            "totalmass": totalmass, "parts": parts, "breaks": phy_mod.break_count(tail)}


def scene_hulls(arm_obj, m, source, content, scale):
    """`read_hulls` over the donor's `.phy`, or None when the armature has no hull object
    or the donor no `.phy`, which `hulls_phy` then makes from the hull objects. Raises
    `SceneRefused` on an import that did not give every solid its object, so no hull can be
    matched to its solid, or when the import made the hulls and every one is gone, which
    would leave a `.phy` of no solid."""
    objs = hull_objects(arm_obj)
    try:
        _where, data = donor_phy(source, content)
    except OSError:
        data = None
    if data is None:
        return None
    if not objs and arm_obj.get("vtmb_phy_solids"):
        raise SceneRefused(
            "every collision hull object the import made is gone, and a .phy of no solid "
            "is not something this export writes: undo the deletion, or keep one hull. "
            "Nothing was written")
    if not objs:
        return None
    if not arm_obj.get("vtmb_phy_solids"):
        raise SceneRefused(
            "the armature has %d collision hull object%s and its import did not make one "
            "per solid of the file's .phy (Collision was off, or a solid names a bone the "
            "armature has not got), so none can be matched to its solid. Re-import with "
            "Collision on, or delete the hull objects to keep the .phy as it is. Nothing "
            "was written" % (len(objs), "" if len(objs) == 1 else "s"))
    names = {v: k for k, v in bone_map(m, arm_obj).items()}
    return read_hulls(arm_obj, m, data, scale, names)


def bone_rests(d, m):
    """{file bone name: its rest transform in model space, the inverse of the `posetobone`
    `d` holds}."""
    out = {}
    for k, b in enumerate(m.bones):
        flat = struct.unpack_from("<12f", d.bones[k].raw, 0x58)
        out[b.name] = mdl_mod.mat_inverse([list(flat[0:4]), list(flat[4:8]), list(flat[8:12])])
    return out


def rewrite_phy(source, dest, pairs, content=None, removed=(), moves=None, hulls=None,
                m=None, rest=None, collision=None):
    """The donor's `.phy` as this export leaves it: the edits `read_hulls` found in the
    hulls (`hulls`), then every solid named for a bone in `removed` or dropped by a hull
    taken away dropped (the user, 2026-10-29), or moved by `moves` when it has no parent and
    more than one child solid (the user, the same day), then a solid added for each new
    hull (`add_hull_solids`, with `m` and `rest`), then its bone names following `pairs`,
    every remaining bone's (file name, name now) in file order, then `collision`, bytes from
    `scene_collision_text`, as its `phy.collision_text` where it differs, then `IVPS` on
    every solid of a file with more than one (`phy.tag_solids`). `write_phy` puts it beside
    `dest` once the `.mdl` is written.

    It is written when it changed, and on a save-as whether or not it did: the new model has
    no collision and no ragdoll without it (the user, 2026-10-29). None when the donor has
    no `.phy`, or an export in place changes nothing in it; else {"data": the bytes, None on
    an error, "renamed": [(old, new)], "dropped": [solid names], "moved": [(solid name, bone
    now)], "added": [bones], "rebuilt": [(solid, volume was, volume now)], "edited":
    [(solid, key, was, now)], "collision": (`break` blocks was, now) when `collision` was
    written, "copied": where it came from when nothing changed, "tagged": solids that
    lacked the tag, "error"}.
    A drop that would leave the ragdoll in pieces raises `SceneRefused`, so nothing is
    written.
    """
    h = hulls or {}
    try:
        where, data = donor_phy(source, content)
        if data is None:
            return None
        out = phy_mod.edit_solids(data, h.get("surfaces"), h.get("values"), h.get("joints"))
        out, _remap, dropped, moved = phy_mod.drop_solids(
            out, list(removed) + list(h.get("drop", ())), moves)
        added = []
        if h.get("add"):
            added, out = add_hull_solids(out, h["add"], m, rest or {})
        out, changed = (phy_mod.rename_bones(out, pairs) if any(a != b for a, b in pairs)
                        else (out, []))
        retext = None
        if collision is not None:
            was = phy_mod.collision_text(out).encode("latin-1")
            if collision != was:
                out = phy_mod.with_collision_text(out, collision)
                retext = (phy_mod.break_count(was), phy_mod.break_count(collision))
        out, tagged = phy_mod.tag_solids(out)
    except phy_mod.RagdollSplit as e:
        raise SceneRefused("%s: %s. Nothing was written"
                           % (os.path.basename(phy_path(dest)), e)) from None
    except (OSError, phy_mod.PhyError) as e:
        return {"data": None, "renamed": [], "dropped": [], "moved": [], "added": [],
                "rebuilt": [], "edited": [], "collision": None, "copied": None,
                "tagged": 0, "error": str(e)}
    edited = bool(h.get("surfaces") or h.get("values") or h.get("joints"))
    touched = changed or dropped or moved or added or edited or retext or tagged
    if not touched and same_file(source, dest):
        return None
    return {"data": out, "renamed": sorted(set(changed)), "dropped": dropped,
            "moved": moved, "added": added, "rebuilt": list(h.get("rebuilt", ())),
            "edited": list(h.get("edited", ())), "collision": retext,
            "copied": None if touched else where, "tagged": tagged, "error": None}


def built_phy(checksum, built):
    """`rewrite_phy`'s result for a `.phy` `hulls_phy` made, the donor having none."""
    return {"data": phy_mod.phy_file(checksum, built["surfaces"], built["text"]),
            "renamed": [], "dropped": [], "moved": [], "added": [], "rebuilt": [],
            "edited": [], "collision": None, "copied": None, "error": None,
            "built": {k: built[k] for k in ("solids", "joints", "totalmass", "parts",
                                            "breaks")}}


def write_phy(dest, rewrite):
    """`rewrite_phy`'s bytes beside `dest`, and its result with any write error in it."""
    if rewrite is None or rewrite["data"] is None:
        return rewrite
    try:
        with open(phy_path(dest), "wb") as f:
            f.write(rewrite["data"])
    except OSError as e:
        return dict(rewrite, data=None, renamed=[], dropped=[], moved=[], added=[],
                    rebuilt=[], edited=[], collision=None, copied=None, built=None,
                    tagged=0, error=str(e))
    return rewrite


def phy_report(source, dest, removed, renamed, geometry_moved, rewrite=None, content=None):
    """What the `.phy` the engine will load for `dest` no longer describes, or None when
    there is nothing to say.

    `rewrite` is `write_phy`'s result. Once it has written, the renames are its to report:
    a swap of two names leaves each old name cited in the file, now for the other bone. With
    nothing beside `dest`, the donor's `.phy` out of a pack is the one in force."""
    done = rewrite is not None and rewrite["error"] is None
    extra = {"rewritten": rewrite["renamed"] if done else [],
             "dropped": rewrite["dropped"] if done else [],
             "moved": rewrite["moved"] if done else [],
             "added": rewrite["added"] if done else [],
             "rebuilt": rewrite["rebuilt"] if done else [],
             "edited": rewrite["edited"] if done else [],
             "collision": rewrite.get("collision") if done else None,
             "copied": rewrite["copied"] if done else None,
             "built": rewrite.get("built") if done else None,
             "tagged": rewrite.get("tagged", 0) if done else 0,
             "error": rewrite["error"] if rewrite is not None else None}
    if done:
        renamed = []
    old_names = [g["name"] for g in removed or ()] + [a for a, _ in renamed or ()]
    cited = phy_bone_names(dest, old_names)
    if cited is None and same_file(source, dest):
        try:
            _where, data = donor_phy(source, content)
        except OSError:
            data = None
        if data is not None:
            cited = phy_bone_names(dest, old_names, data)
    if cited is None:
        if extra["error"]:
            return dict({"file": os.path.basename(phy_path(dest)), "removed": [],
                         "renamed": [], "geometry": geometry_moved}, **extra)
        return None
    gone = sorted(g["name"] for g in removed or () if g["name"] in cited)
    moved = sorted((a, b) for a, b in renamed or () if a in cited)
    if not (gone or moved or geometry_moved or extra["rewritten"] or extra["dropped"]
            or extra["moved"] or extra["added"] or extra["rebuilt"] or extra["edited"]
            or extra["collision"] or extra["copied"] or extra["built"] or extra["error"]):
        return None
    return dict({"file": os.path.basename(phy_path(dest)), "removed": gone, "renamed": moved,
                 "geometry": geometry_moved}, **extra)


def revise_vtx(source, dest, data, revised, flavours=("dx80",), drop=None):
    """Rewrite the .vtx for a model whose geometry moved, and return what each cost.

    The donor's own file supplies every strip group the edit did not touch, so a change to
    one mesh leaves the others byte for byte as the compiler emitted them. Each flavour is
    revised from its own donor: the two files cap bones per triangle at 9 and at 2, so
    they do not share a strip-group partition and neither can be derived from the other.

    LOD 0 alone is revised. The lower LODs are left as read, naming vertices by the donor's
    numbering, and `cut_vtx_lods` drops them right after.

    A flavour asked for but absent beside the source is skipped. A donor with no
    `.dx80.vtx` at all gets none written, and the export says so.
    """
    written = mdl_mod.Mdl("<written>", data=data)
    out = []
    for flavour in flavours:
        src = vtx_path(source, flavour)
        if not os.path.exists(src):
            continue
        blob, st = vtxr_mod.revise(written, src, revised, lod0_only=True, drop=drop)
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
                   write_cloth=True, vtx_flavours=("dx80",), placed=None,
                   before_write=None, content=None):
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

    `placed` is {object name: the 4x4 `placed` baked into it}, which turns the donor's
    tangents with the object rather than recomputing every one.

    `mesh_fields` says which per-vertex fields a model that still has the file's own
    triangles takes from the scene. It does not decide whether the geometry is read: a
    changed triangle set rebuilds the model from the scene whatever it says, triangles
    living in the .vtx alone and no field naming one. A rebuild writes the scene's
    positions and skinning as well, an original keeping what was imported only for the
    fields `mesh_fields` leaves out.

    `before_write(d, added)` runs on the finished model before the `.vtx` or the `.mdl` is
    written, so a caller that refuses over what the edit produced leaves nothing on disk.
    `added` is the material names the export added, the list the `.vmt` writer reads.

    `content` is a `blender_import.Content` over the donor's roots, where `donor_phy` looks
    for a `.phy` that is not beside it.
    """
    m = mdl_mod.Mdl(source)
    if verify:
        bad = verify_writer(bytes(m.d))
        if bad:
            raise SceneRefused("the writer does not reproduce %s, so nothing was written: %s"
                               % (os.path.basename(source), "; ".join(bad)))
    old = old_imports(m, source, arm_obj)
    if old:
        raise build_mod.Refused(_old_import_text(old))
    stray = stray_accessories(arm_obj)
    if stray:
        raise build_mod.Refused(stray_text(stray))
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
    new_objs = new_model_objects(m, source, arm_obj)
    appending = set(o.name for o in new_objs)
    dup_models = [x for x in dup_models if x[0] not in appending]
    # Before the first reader: `read_poses` and `read_bones` both walk the FILE's bone list
    # and refuse by name on one the armature does not have, so a bone deleted in Blender has
    # to leave the description here or it never reaches the writer.
    d = build_mod.from_bytes(bytes(m.d))
    donor_pose = {b.name: b.posetobone for b in m.bones}
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
    # Every bone, renamed or not: a `.phy` name follows the first bone it matches.
    bone_pairs = [(b.name, d.bones[k].name) for k, b in enumerate(m.bones)]
    renamed = [(m.bones[k].name, name) for k, name in sorted(renamed.items())]

    # Both names are `char name[128]` inline, so writing either moves nothing.
    remodelled = renamed_models(m, source)
    for (bi, mi), name in sorted(remodelled.items()):
        build_mod.set_model_name(d, bi, mi, name)
    # A bodypart's name is a string-table offset, and `mdl_build.emit` lays the table out.
    rebodied = renamed_bodyparts(m, source)
    for bi, name in sorted(rebodied.items()):
        d.bodyparts[bi].name = name
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
    # Compared as the f32 bytes, so an unedited stamp writes nothing. A typed hull moves no
    # vertex and leaves the sequence boxes alone; Fit hull, when ticked, wins over it.
    typed = header_vectors(arm_obj, scale)
    header = {}
    for off, v in sorted(typed.items()):
        if hull is not None and off != mdl_mod.HDR_EYEPOSITION:
            continue
        if struct.pack("<3f", *v) != bytes(d.hdr[off:off + 12]):
            struct.pack_into("<3f", d.hdr, off, *v)
            header[off] = v
    typed_hull = hull is None and bool({mdl_mod.HDR_HULL_MIN, mdl_mod.HDR_HULL_MAX} & set(header))
    if typed_hull:
        hull = (struct.unpack_from("<3f", d.hdr, mdl_mod.HDR_HULL_MIN),
                struct.unpack_from("<3f", d.hdr, mdl_mod.HDR_HULL_MAX))

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
        raise SceneRefused("%s has no animation data" % arm_obj.name)
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
                raise SceneRefused("%s: empty frame range" % action.name)

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
                raise SceneRefused("%s: empty frame range" % action.name)
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
             "added_materials": [], "skin_families": 0,
             "slots_moved": 0, "slots_gone": [],
             "renamed_anims": [],
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
    # The table before the renames, so a slot showing a record a cell added reads through
    # the row that names it, and before new_materials, whose records append a column to
    # every row this replaces.
    if len(skin_table(arm_obj)[0]) > 1:
        sync_skin_table(arm_obj, armature_meshes(context.scene, arm_obj), bpy.data.materials)
    if skin_edited(arm_obj):
        d.skin, added = resolve_skin_table(d, arm_obj)
        scene["skin_families"] = len(d.skin)
        scene["added_materials"] += added
    if skin_edited(arm_obj):
        named, scene["slots_moved"], scene["slots_gone"] = read_materials(
            m, source, skin_family(m, arm_obj), rows=d.skin)
    else:
        named, scene["slots_moved"], scene["slots_gone"] = read_materials(
            m, source, skin_family(m, arm_obj))
    named.update(skin_renames(d, arm_obj, set(named)))
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
                "gone", "over_mask", "end_renamed"):
        scene["spring_" + key] = sp[key]
    scene["accessories"] = apply_accessories(d, m, arm_obj, scale)
    vol = apply_cloth_volumes(d, m, arm_obj, scale)
    scene["cloth_volumes"], scene["cloth_volumes_unread"] = vol["changed"], vol["unread"]
    scene["cloth_two_bone"] = vol.get("two_bone") or []
    # Deleting every box empty writes numhitboxsets 0, which nothing else would say.
    scene["hitboxsets"] = len(d.hitboxsets)
    face = apply_face(d, m, arm_obj, scale, source)
    scene["face"] = face["changed"]
    scene["eyes_added"] = face["eyes_added"]
    scene["mouth_added"] = face["mouth_added"]
    scene["mouth_removed"] = face["mouth_removed"]
    scene["eye_meshes"] = face["eye_meshes"]
    scene["eye_radius_keys"] = face["radius_keys"]
    scene["eye_lid_targets"] = face["lid_targets"]
    scene["eye_empties"] = face["eye_empties"]
    scene["eye_records"] = face["eye_records"]
    scene["eyes_removed"] = face["eyes_removed"]
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
    renames = renamed_anims(anim_names, source, arm_obj)
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
            raise SceneRefused("%s has no animation %r to remove"
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

    # Before the geometry, so a rebuilt model regenerates no cloth the scene removed.
    cloth_removed = remove_cloth(d, m, source) if write_cloth else []
    mesh = {"fields": tuple(mesh_fields), "verts": 0, "models": 0,
            "missing": [], "unsupported": [], "normals": 0, "rebuilt": [],
            "unskinned": 0, "renumbered": 0, "crowded": [], "blind_normals": 0,
            "rebuilt_uvs": 0, "rebuilt_added": 0, "rebuilt_deleted": 0, "blanked": [],
            "flex_dropped": 0, "flex_emptied": 0, "uv_spare": [], "tangents": 0,
            "cloth_rebuilt": [], "cloth_renumbered": [], "cloth_carried": [],
            "cloth_promoted": [], "cloth_rigid": [],
            "flexes": 0, "flex_records": 0, "flex_skipped": 0, "flex_refused": [],
            "flex_carried": [], "flex_gone": [], "flex_unkeyed": [],
            "stray_groups": [], "dequantised": [], "rebased": []}
    revised = {}
    placed = placed or {}
    scene_objs = mesh_objects(m, source)
    place_of = {k: placed[o.name] for k, o in scene_objs.items() if o.name in placed}
    # Only with Positions ticked: without it the file keeps its old placement, box and all.
    boxes = {k: b for k, p in place_of.items() if "positions" in mesh_fields
             for b in [rebased_box(m.bodyparts[k[0]].models[k[1]], p)] if b is not None}
    # Whatever the four checkboxes say. Triangles live in the .vtx alone, so a face
    # added, deleted or re-wound moves no .mdl byte and no checkbox names one -- and a
    # mesh the scene changed is rebuilt from the scene rather than from the file, which
    # is what leaves the written model independent of the donor's own geometry.
    blank = [] if blanks_here(arm_obj, source) else None
    cells, rebuild, mesh["missing"], mesh["unsupported"], mesh["normals"], \
        mesh["crowded"], mesh["blind_normals"], mesh["uv_spare"], donor_faces, \
        mesh["unskinned"], mesh["stray_groups"] = read_meshes(
            m, source, mesh_fields, boxes, blank)
    if (mesh_fields and not cells and not rebuild and not mesh["missing"] and not blank
            and any(drawn_model(mo, donor_faces, bi, mi)
                    for bi, mi, _bp, mo in mesh_mod.models_of(m))):
        # Only where a field was asked for: with none, an untouched scene reaches neither
        # list and that is the whole of a default export rather than a scene that belongs
        # to another file. A file of no vertex at all, null.mdl's form, has nothing to
        # supply.
        raise SceneRefused("no scene mesh belongs to %s" % os.path.basename(source))
    # Decision 21: a removed eyeball's mesh and a deleted object's model leave the file.
    # Dropped just before the emit, so every pass above walks the donor's mesh numbering.
    drops = {}
    eye_drops = eye_mesh_drops(m, source, face["eye_gone"])
    for key, obj in mesh_objects(m, source).items() if eye_drops else ():
        if key in eye_drops:
            cells.pop(key, None)
            rebuild[key] = obj
    for bi, mi, name in blank or ():
        drops[(bi, mi)] = set(range(len(d.bodyparts[bi].kids[mi].kids)))
        mesh["blanked"].append(name)
    for (bi, mi), verts in sorted(cells.items()):
        rec = d.bodyparts[bi].kids[mi]
        mo = m.bodyparts[bi].models[mi]
        was_vb = rec.extra["verts"]
        place = place_of.get((bi, mi))
        box = boxes.get((bi, mi))
        if box is not None:
            # In the moved box nothing moved, so the tangents need no turning either.
            mo = copy.copy(mo)
            mo.quant_offset, mo.quant_scale = box
            struct.pack_into("<3f", rec.raw, 0xa0, *box[0])
            struct.pack_into("<3f", rec.raw, 0xac, *box[1])
            mesh["rebased"].append((scene_objs[(bi, mi)].name, mo.filetype))
            place = None
        rec.extra["verts"], n = mesh_mod.pack_model(mo, verts, mesh_fields, was_vb)
        # _must_rebuild sends every changed triangle set to the rebuild, so on this
        # path the donor's triangles are still the written vertices' own.
        tri = donor_faces.get((bi, mi))
        tb = rec.extra.get("tangents") or b""
        rec.extra["tangents"], t = build_mod.retangent_block(
            mo.filetype, was_vb, rec.extra["verts"],
            tb if place is None else build_mod.turn_tangents(tb, place),
            tri.elements() if tri else (), mo.quant_offset, mo.quant_scale, place=place)
        mesh["verts"] += n
        mesh["tangents"] += t
    mesh["models"] = len(cells)
    bone_index = {b.name: b.index for b in m.bones}
    cloth_maps = {}
    for (bi, mi), obj in sorted(rebuild.items()):
        mo = m.bodyparts[bi].models[mi]
        was = mo.numvertices
        if mo.filetype in mdl_mod.QUANT_MAX:
            mesh["dequantised"].append((obj.name, mo.filetype,
                                        outside_box(obj.data, mo, boxes.get((bi, mi))),
                                        off_bone0(obj, mo, bone_index)
                                        if "weights" in mesh_fields else 0))
        tri = donor_faces.get((bi, mi))
        faces, unskinned, kept, edits = rebuild_cell(
            d, obj, bi, mi, bone_index, mesh_fields,
            list(tri.elements()) if tri else None, place=placed.get(obj.name),
            empty=eye_drops.get((bi, mi), ()))
        if (bi, mi) in eye_drops:
            drops[(bi, mi)] = set(eye_drops[(bi, mi)])
        for k, tris in enumerate(faces):
            revised[(bi, mi, k)] = tris
        now = sum(struct.unpack_from("<i", x.raw, 0x08)[0]
                  for x in d.bodyparts[bi].kids[mi].kids)
        mesh["rebuilt"].append((obj.name, was, now))
        mesh["unskinned"] += unskinned if skin_counted(mo) else 0
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
        if "cloth_map" in edits:
            cloth_maps[(bi, mi)] = edits["cloth_map"]
        ce = edits.get("cloth")
        if ce and ce.get("rigid"):
            mesh["cloth_rigid"].append((obj.name, ce["rigid"], ce["why"]))
        if ce and ce.get("promoted"):
            mesh["cloth_promoted"].append((obj.name, ce))
        elif ce and ce.get("carried"):
            mesh["cloth_carried"].append((obj.name, ce))
        elif ce and "renumbered" in ce:
            if ce["renumbered"]:
                mesh["cloth_renumbered"].append((obj.name, ce["renumbered"]))
        elif ce:
            mesh["cloth_rebuilt"].append((obj.name, ce))

    # After the donor's own models, so every stamped `vtmb_index` keeps its model.
    appended, add_faces, mesh["not_appended"] = append_models(
        d, new_objs, bone_index, skin_family(m, arm_obj), scene["added_materials"], scale)
    revised.update(add_faces)
    eyes = apply_appended_eyes(d, m, arm_obj, scale, appended)
    if eyes["used"]:
        scene["eyes_added"] = scene["eyes_added"] + eyes["eyes_added"]
        scene["eye_meshes"] += eyes["eye_meshes"]
        scene["eye_empties"] = [x for x in scene["eye_empties"] if x[0] not in eyes["used"]]
        scene["eye_radius_keys"] = scene["eye_radius_keys"] + eyes["radius_keys"]
        scene["eye_lid_targets"] = scene["eye_lid_targets"] + eyes["lid_targets"]
        for name in eyes["materials"]:
            if name not in scene["added_materials"]:
                scene["added_materials"].append(name)
    for x in appended:
        mesh["unskinned"] += x["unskinned"]
        over = crowded_vertices(x["object"], bone_index)
        if over:
            mesh["crowded"].append((x["name"], over))
        odd = stray_groups(x["object"], bone_index)
        if odd:
            mesh["stray_groups"].append((x["name"], odd))

    if write_flexes:
        # After the geometry, because add_flex bounds every key against the mesh's
        # numvertices as written and a rebuild has just changed it.
        found = mesh_objects(m, source)
        for (bi, mi, _bp, _mo) in mesh_mod.models_of(m):
            obj = found.get((bi, mi))
            if obj is None:
                continue
            k, r, skipped, why, two = shape_key_flexes(d, obj, bi, mi, scale)
            mesh["flex_carried"].extend(two["carried"])
            mesh["flex_gone"].extend((obj.name,) + g for g in two["gone"])
            if two["unkeyed"]:
                mesh["flex_unkeyed"].append((obj.name, two["unkeyed"]))
            mesh["flexes"] += k
            mesh["flex_records"] += r
            mesh["flex_skipped"] += skipped
            if why:
                mesh["flex_refused"].append((obj.name, why))
    # Whether or not the keys were written: a rule deleted in the panel leaves its flex
    # undriven just the same.
    mesh["flex_undriven"] = build_mod.undriven_flexes(d)
    # After the keys and the rules, which come later here than the face pass and can make a
    # name a face record appended.
    face_flexes_refused(d)

    # After the geometry: a rest length is refitted against where the vertex now sits, and
    # a rebuilt model has just moved them.
    cloth_was = cloth_edits(m, source, d, cloth_maps)
    cloth = apply_cloth(d, m, source, cloth_was) if write_cloth else None

    for (bi, mi), ks in sorted(drops.items()):
        build_mod.drop_meshes(d, bi, mi, ks)
    # Before `before_write`, which writes `.vmt` files: a dent refuses with nothing on disk.
    hulls = scene_hulls(arm_obj, m, source, content, scale)
    # A donor with no `.phy`: the hull objects are the whole of it (the user, 2026-10-29).
    built = (hulls_phy(arm_obj, d, scale, os.path.splitext(os.path.basename(dest))[0])
             if hulls is None else None)
    collision = scene_collision_text(arm_obj, source, content) if built is None else None
    if before_write is not None:
        before_write(d, scene["added_materials"])
    phy_rewrite = rewrite_phy(source, dest, bone_pairs, content,
                              [g["name"] for g in removed], phy_moves(d, m, donor_pose),
                              hulls, m, bone_rests(d, m) if hulls and hulls["add"] else None,
                              collision)
    if phy_rewrite is not None and (phy_rewrite["dropped"] or phy_rewrite["moved"]
                                    or phy_rewrite["added"]):
        build_mod.set_physicsbones(d, phy_mod.solid_names(phy_rewrite["data"]))
    if built is not None:
        build_mod.set_physicsbones(d, built["solids"])
        phy_rewrite = built_phy(d.checksum, built)
    # The donor checksum is kept whether or not the .vtx is rewritten: the pair only
    # has to agree with each other, and the engine draws nothing when it does not.
    data = build_mod.emit(d, checksum=d.checksum)
    # A bone removal or an append moves no vertex and no triangle, so `revised` is empty --
    # and the .vtx still has to be rewritten, because every strip group's bone data is bound
    # to the .mdl's numbering, and a group without flag 0x02 carries the bone id per vertex.
    # `vtx_rebuild.revise` rebinds all of them off the model as written whether or not a cell
    # was named, so an empty face map is the whole of what either needs.
    touched = (revise_vtx(source, dest, data, revised, vtx_flavours, drops)
               if (revised or removed or added_bones or cloth_removed or drops) else [])
    vtx = next((x for x in touched if x["flavour"] == "dx80"), None)
    # After the geometry pass, which writes whole .vtx files: the cut re-lays what that pass
    # wrote.
    lods = cut_vtx_lods(source, dest, [x["flavour"] for x in touched], carry=not touched)
    eye_flags = vtx_eye_flags(d, dest)
    no_vtx =not os.path.exists(vtx_path(source)) and not os.path.exists(vtx_path(dest))
    with open(dest, "wb") as f:
        f.write(data)
    # After the .mdl, so a write that fails there leaves the donor's `.phy` as it was.
    phy_rewrite = write_phy(dest, phy_rewrite)
    # A save-as leaves the armature describing the donor, which has no such model.
    if same_file(source, dest):
        stamp_appended(d, appended, arm_obj)
    mesh["appended"] = [dict((k, v) for k, v in x.items() if k != "object")
                        for x in appended]
    return {"wrote": wrote, "added": added, "dropped": dropped, "unfitted": unfitted,
            "unkeepable": unkeepable, "bones": len(m.bones),
            "mesh": mesh, "scene": scene, "bytes": len(data), "was": len(m.d),
            "anims": len(m.anims), "sequences": len(d.seqs),
            "vtx": vtx, "removed": removed, "added_bones": added_bones,
            "vtx_more": [x for x in touched if x["flavour"] != "dx80"],
            "renamed": renamed,
            "model_name": model_name, "hull": hull, "cdtexture": cdtex,
            "boxes": (d.refit_count, len(d.seqs)) if hull is not None and not typed_hull
            else None,
            "eyeposition": header.get(mdl_mod.HDR_EYEPOSITION),
            "hull_inverted": inverted_axes(d),
            # One scale set serves every animation in the file and `fit_scales` never
            # narrows, so a pose this export authored can widen a bone for every animation
            # already in it. `emit` is what runs the fit, so this is read after it.
            "requant_bones": d.requant_bones, "requant_carried": d.requant_carried,
            "remodelled": sorted((v, k) for k, v in remodelled.items()),
            "rebodied": sorted((v, k) for k, v in rebodied.items()),
            "includes": [r.name for r in d.includes],
            "stale": (stale_flavours(dest, [x["flavour"] for x in touched])
                      if (revised or removed or added_bones or cloth_removed) else []),
            "cloth": cloth, "cloth_edits": cloth_was, "cloth_removed": cloth_removed,
            "lods": lods, "no_vtx": no_vtx, "vtx_eye_flags": eye_flags,
            "phy": phy_report(source, dest, removed, renamed, bool(revised), phy_rewrite,
                              content)}


def export_action(context, arm_obj, source, dest, anim_name="", **kw):
    """One active action into one animation. Kept because it is what the round-trip and
    self-check scripts drive."""
    ad = arm_obj.animation_data
    action = ad.action if ad else None
    if action is None:
        raise SceneRefused("%s has no action assigned" % arm_obj.name)
    index = resolve_target(mdl_mod.Mdl(source), action, anim_name)
    r = export_actions(context, arm_obj, source, dest, {index: action}, **kw)
    i, name, _, frames, movements = r["wrote"][0]
    r.update(anim=name, index=i, frames=frames, movements=movements)
    return r
