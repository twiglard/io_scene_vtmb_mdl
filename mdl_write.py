#!/usr/bin/env python3
"""MDL v2531 animation encoder. No bpy.

Poses to int16 channels and channels to RLE bytes, plus the scale fitting the two need.
It lays nothing out: `mdl_build` decides where a block goes and computes the offsets that
reach it, so there is no path here that edits a file in place. `owned_spans`/`_extents`
stay because attributing the donor's animation bytes is how `mdl_rebuild` and
`mdl-coverage.py` tell a live span from a dead one. Evidence for the encoding is in
todo-vtmb-mdl-animation.md section 2.22.
"""

import math
import os
import struct
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mdl as M
else:
    from . import mdl as M


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Tracks:
    """One animation as int16 channels plus the animdesc fields that describe them.

    `chan` is keyed (bone index, channel 0..6); a missing key is an absent channel,
    which is not the same as a zero one -- an absent rotation channel means the bone's
    bind quaternion, an absent translation means its bind position.
    """

    __slots__ = ("name", "fps", "flags", "numframes", "chan", "weights", "movements")

    def __init__(self):
        self.name = ""
        self.fps = 30.0
        self.flags = 0
        self.numframes = 1
        self.chan = {}
        self.weights = {}
        self.movements = []


def read_tracks(m, a):
    """Decode one shipped animation to int16, losslessly. Values stay quantised: they
    are only ever rescaled by a ratio, never routed through the decoded float."""
    t = Tracks()
    t.name, t.fps, t.flags, t.numframes = a.name, a.fps, a.flags, a.numframes
    t.movements = list(a.movements)
    if a.numframes <= 0:
        return t
    d = m.d
    for b, ch in zip(m.bones, m.anim_channels(a)):
        o = a.base + b.index * M.ANIM_STRIDE
        t.weights[b.index] = struct.unpack_from("<f", d, o)[0]
        for c in range(M.NUM_CHANNELS):
            if ch[c] is not None:
                # Copied: anim_channels hands back the decode cache's own lists, which
                # local_pose keeps reading until the animation moves on.
                t.chan[(b.index, c)] = list(ch[c])
    return t


def runs_of(v):
    """studiomdl's RLE loop, transcribed from utils/studiomdl/simplify.cpp:4948-4993.

    A run stores `valid` values and spans `total` frames, the tail repeating the last
    stored one. A value equal to its predecessor is still stored while the run has
    absorbed no repeat yet and the next value differs again, which is what keeps a
    two-long repeat from splitting a run.
    """
    n = len(v)
    if n == 0:
        return []
    runs = [[1, 1, [v[0]]]]
    for m in range(1, n):
        cur = runs[-1]
        if cur[1] == 255:
            runs.append([1, 0, [v[m]]])
            cur = runs[-1]
        elif (v[m] != v[m - 1]
              or (cur[1] == cur[0] and m < n - 1 and v[m] != v[m + 1])):
            if cur[1] != cur[0]:
                runs.append([0, 0, []])
                cur = runs[-1]
            cur[0] += 1
            cur[2].append(v[m])
        cur[1] += 1
    return runs


def encode_channel(v):
    """Byte-identical to studiomdl on the shipped corpus; recon section 22."""
    out = bytearray()
    for valid, total, vals in runs_of(v):
        out.append(valid)
        out.append(total)
        for x in vals:
            out += struct.pack("<h", x)
    return bytes(out)


def is_droppable(v):
    """studiomdl discards a channel that came out as one header and one zero value
    (`simplify.cpp:4997`), which is the only reason an animation channel is ever absent."""
    r = runs_of(v)
    return len(r) == 1 and r[0][0] == 1 and r[0][2][0] == 0


def clamp16(x):
    return -32768 if x < -32768 else (32767 if x > 32767 else x)


def steps_needed(v):
    """The smallest scale that represents `v`. The range is asymmetric and Troika used all
    of it -- 319534 shipped values sit at -32768 -- so fitting a negative against 32767
    widens a scale that already fits, and every animation in the file is requantised."""
    return v / 32767.0 if v >= 0.0 else v / -32768.0


def fits(v, s):
    """Whether `v` quantises at scale `s` without the clamp costing more than the rounding
    already does. Comparing scales instead would widen on a value a hair past the rail,
    and widening one bone requantises that bone in every animation of the file."""
    return bool(s) and -32768 <= int(round(v / s)) <= 32767


def local_from_world(m, world):
    """Exact inverse of Mdl.world_matrices, engine flags included.

    A BONE_ROTATION_FROM_ROOT bone's world rotation *is* its local one -- the branch at
    client.dll+0x8ffc5 never composed it against the parent -- while its translation did
    come down the chain and has to be undone against the parent frame.
    """
    out = []
    for b in m.bones:
        w = world[b.index]
        if b.parent < 0:
            local = w
        else:
            pinv = M.mat_inverse(world[b.parent])
            if b.flags & M.BONE_ROTATION_FROM_ROOT:
                t = [w[i][3] - world[b.parent][i][3] for i in range(3)]
                local = [w[i][0:3] + [sum(pinv[i][k] * t[k] for k in range(3))]
                         for i in range(3)]
            else:
                local = M.mat_mul(pinv, w)
        out.append(([local[i][3] for i in range(3)], M.quat_from_mat(local)))
    return out


def unwind_signs(poses):
    """A quaternion and its negation are the same rotation but not the same track. Left
    alone, a sign flip between frames makes the engine interpolate the long way round."""
    for f in range(1, len(poses)):
        for i, (pos, q) in enumerate(poses[f]):
            prev = poses[f - 1][i][1]
            if sum(q[c] * prev[c] for c in range(4)) < 0.0:
                poses[f][i] = (pos, [-c for c in q])
    return poses


def quantise(m, poses, scales, drop_zero_pos=True, drop_bind_rot=True):
    """Float local poses to int16 channels, against the scales they will be decoded by.

    An all-zero translation channel is dropped and an all-zero rotation channel is not.
    Translation is additive over the bind position, so dropping it decodes to the same
    thing; rotation is a bare product, so a dropped channel decodes to the bind
    quaternion instead of to zero. Troika applied studiomdl's drop rule to translation
    only -- 1930 shipped rotation channels are in the exact form it would have dropped.

    A rotation component that never leaves the bind value is dropped instead, which is
    what the shipped files do and what keeps a re-authored animation the donor's size.
    """
    chan = {}
    nframes = len(poses)
    for b in m.bones:
        for c in range(3):
            s = scales[b.index][c]
            if not s:
                continue
            v = [clamp16(int(round((poses[f][b.index][0][c] - b.pos[c]) / s)))
                 for f in range(nframes)]
            if drop_zero_pos and not any(v):
                continue
            chan[(b.index, c)] = v
        for c in range(4):
            s = scales[b.index][3 + c]
            if not s:
                continue
            v = [clamp16(int(round(poses[f][b.index][1][c] / s)))
                 for f in range(nframes)]
            # Against the bind float, not its int16: rounding both first lets noise either
            # side of .5 emit a channel that then never drops again.
            if drop_bind_rot and all(abs(poses[f][b.index][1][c] - b.quat[c]) <= 0.5 * s
                                     for f in range(nframes)):
                continue
            chan[(b.index, 3 + c)] = v
    return chan


def rescale(t, old, new):
    """Requantise every channel of `t` from one set of per-bone scales to another.

    `old`/`new` are [bone][channel 0..6]. A ratio of exactly 1.0 leaves the int16s bit
    for bit, so an untouched animation in a file whose scales did not move re-encodes to
    the original bytes.
    """
    for (bi, c), v in t.chan.items():
        o, n = old[bi][c], new[bi][c]
        if o == n or not n:
            continue
        r = o / n
        t.chan[(bi, c)] = [clamp16(int(round(x * r))) for x in v]


def qmul(a, b):
    """Hamilton product in the file's own (x, y, z, w) order, magnitudes preserved.

    Not normalised: a shipped bind quaternion is unit only to ~4.4e-5, and re-basing a
    rotation through a matrix instead would normalise it -- `quat_from_mat` does -- which
    moves a rotation that a pure translation edit cannot touch.
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def bind_delta(bo, bn):
    """`(dq, D)` taking one bone's old parent-local bind to its new one.

    `dq = q_new . q_old^-1`, the true inverse and not the conjugate, so a bind quaternion
    that is not quite unit does not scale the result. `D` is the same delta as a 3x4, which
    is what the translation half needs.
    """
    if list(bn.quat) == list(bo.quat):
        # A bind that only moved keeps every rotation channel and every translation int16:
        # `q_new . q_old^-1` is identity and `D` is a pure translation, so `pos_new` comes
        # out `p_new + (pos_old - p_old)` and re-encodes to the value already stored.
        # Computed rather than asserted it would round back, the delta is exact here.
        return [0.0, 0.0, 0.0, 1.0], M.mat_identity()
    n2 = sum(c * c for c in bo.quat) or 1.0
    dq = qmul(bn.quat, [-bo.quat[0] / n2, -bo.quat[1] / n2, -bo.quat[2] / n2,
                        bo.quat[3] / n2])
    D = M.mat_mul(M.mat_from_quat_pos(bn.quat, bn.pos),
                  M.mat_inverse(M.mat_from_quat_pos(bo.quat, bo.pos)))
    return dq, D


def rebased_channels(t, old, new, moved):
    """The float values the channels of the moved bones have to carry after a bind change,
    as {bone: (pos per frame, quat per frame)}.

    An animation stores the parent-local animated matrix `A` while the bind is `B`, and
    `blender_import.build_actions` composes the two as `A = B . R` -- so preserving `R`,
    which is what "the animation follows the moved bone" means, needs
    `A_new = B_new . B_old^-1 . A_old`, premultiplied. In the stored components that is
    `quat_new = dq . quat_old` and `pos_new = D . (pos_old - p_old) + p_new`.

    **A bind translation needs nothing.** Translation decodes additively, `b.pos + int16 *
    posscale`, so once the record holds the new `pos` the stored int16 already means the
    re-based value: `pos_new - p_new` is `pos_old - p_old`, which is what it encoded. Only
    rotation drifts, because `quat[c] = int16 * rotscale[c]` carries no bind base at all.
    So a bone whose bind quaternion did not move is skipped whole and its bytes come back
    identical.

    A bone with no rotation channel at all needs none created either: `local_pose` falls
    back to the record's own quaternion, which is already the new bind, and `dq . q_old =
    q_new` exactly. But a bone keying *some* of the four has the rest filled from the bind
    at decode time, so once the quaternion turns they all have to be written out -- and its
    translation channels have to be written too, since `D` rotates the offset.

    Only the bones whose own record changed are in `moved`, and that is the whole set that
    needs anything. Every channel is parent-local, so a child's `A` and `B` are both
    untouched and `R` with them; its world pose moves down the chain instead. That holds
    for a BONE_ROTATION_FROM_ROOT child too, which takes its rotation from the root frame
    and is therefore meant not to follow its parent -- measured over the corpus at 597 of
    597 such bones left exactly unmoved.
    """
    out = {}
    for k in sorted(moved):
        bo, bn = old.bones[k], new.bones[k]
        if list(bo.quat) == list(bn.quat):
            continue
        has_p = any(t.chan.get((k, c)) for c in range(3))
        has_q = any(t.chan.get((k, 3 + c)) for c in range(4))
        if not has_p and not has_q:
            continue
        dq, D = bind_delta(bo, bn)
        nf = t.numframes
        pos_f, quat_f = [], []
        for f in range(nf):
            # Decoded against the OLD bind: the block was encoded against it, and
            # set_bone_poses has already overwritten the record.
            pos = [bo.pos[c] + t.chan[(k, c)][f] * bo.posscale[c] if t.chan.get((k, c))
                   else bo.pos[c] for c in range(3)]
            quat = [t.chan[(k, 3 + c)][f] * bo.rotscale[c] if t.chan.get((k, 3 + c))
                    else bo.quat[c] for c in range(4)]
            off = [pos[c] - bo.pos[c] for c in range(3)]
            pos_f.append([bn.pos[c] + sum(D[c][x] * off[x] for x in range(3))
                          for c in range(3)])
            quat_f.append(qmul(dq, quat))
        out[k] = (pos_f if has_p else None, quat_f if has_q else None)
    return out


def fit_rebase_scales(m, authored):
    """Widen each bone's scales until every re-based value is representable, never
    narrowing -- the same rule as `fit_scales`, over the sparse per-bone lists
    `rebased_channels` produces.

    A widened scale requantises that bone in every animation of the file, so the bones this
    returns as widened are a consequence the export has to name rather than absorb.
    """
    out = file_scales(m)
    widened = set()
    for chans in authored:
        for k, (pos_f, quat_f) in chans.items():
            b = m.bones[k]
            for f in range(len(pos_f or quat_f or ())):
                if pos_f is not None:
                    for c in range(3):
                        v = pos_f[f][c] - b.pos[c]
                        if not fits(v, out[k][c]):
                            out[k][c] = max(out[k][c], steps_needed(v))
                            widened.add(k)
                if quat_f is not None:
                    for c in range(4):
                        if not fits(quat_f[f][c], out[k][3 + c]):
                            out[k][3 + c] = max(out[k][3 + c],
                                                steps_needed(quat_f[f][c]))
                            widened.add(k)
    return [[f32(x) for x in row] for row in out], widened


def apply_rebase(t, m, chans, scales):
    """Write the re-based values back into `t` as int16 at `scales`.

    A component whose scale is zero can carry only zero, which is what an absent channel
    already decodes to, so it is left absent rather than written as a zero channel.
    """
    for k, (pos_f, quat_f) in chans.items():
        b = m.bones[k]
        if pos_f is not None:
            for c in range(3):
                s = scales[k][c]
                if not s:
                    continue
                t.chan[(k, c)] = [clamp16(int(round((p[c] - b.pos[c]) / s)))
                                  for p in pos_f]
        if quat_f is not None:
            for c in range(4):
                s = scales[k][3 + c]
                if not s:
                    continue
                t.chan[(k, 3 + c)] = [clamp16(int(round(q[c] / s))) for q in quat_f]


def file_scales(m):
    return [list(b.posscale) + list(b.rotscale) for b in m.bones]


def fit_scales(m, authored):
    """Widen each bone's scales until every authored value is representable, never
    narrowing: the shipped scales are fitted tight to the shipped poses (section 3.4),
    so anything new clips without this, and every other animation in the file shares
    them and has to be requantised when they move.
    """
    out = file_scales(m)
    for poses in authored:
        for frame in poses:
            for b in m.bones:
                pos, quat = frame[b.index]
                for c in range(3):
                    v = pos[c] - b.pos[c]
                    if not fits(v, out[b.index][c]):
                        out[b.index][c] = max(out[b.index][c], steps_needed(v))
                for c in range(4):
                    if not fits(quat[c], out[b.index][3 + c]):
                        out[b.index][3 + c] = max(out[b.index][3 + c],
                                                  steps_needed(quat[c]))
    return [[f32(x) for x in row] for row in out]


LINEAR_AXIS = (0x40, 0x80, 0x100)
# Undefined in ref/2531/studio.h and never read at runtime, but set on 23442 of 23861
# shipped blocks; motionflags as a whole reaches no engine code.
MOTION_EXTRA = 0x1000


def fit_movements(path, eps=1e-4):
    """The straight ground-plane ramp behind one `mstudiomovement_t`, and the per-frame
    displacement it accounts for.

    studiomdl extracts the *net* motion, never the path: one block along the frame-0 to
    last-frame displacement, its speed ramping as `d(t) = v0 t + (v1 - v0) t^2 / 2` over
    normalised time, which is exactly what `anim_position` integrates back
    (`utils/studiomdl/simplify.cpp:315`, `extractLinearMotion`). `v0` and `v1` come from
    the midframe -- distance `d1` there and `d2` at the end give `v0 = 4 d1 - d2` and
    `v1 = 3 d2 - 4 d1`, each clamped to non-negative, and collapsed to a constant `d2`
    when they agree within 10%. andrei's `run_0` carries `v0` 113.687 over 93.519, which
    that fit reproduces.

    Z is never extracted: `position.z` is nonzero in 68 of the corpus's 23 861 blocks, over
    8 files. A run's rise and its fore-aft sway belong in the keys, where a viewer that
    ignores movement blocks -- which is every one not driving an NPC -- can still see them.
    `angle` stays 0: the 21 shipped blocks that carry one are pure yaw with no translation
    at all, which is a different record and not something a linear fit produces.

    One block per animation, so a path that curves keeps its curve in the keys and only
    the chord becomes entity motion. Troika splits a turning walk into a block per span,
    which this does not.

    Returns `([], [])` when the ground-plane displacement is under `eps`.
    """
    n = len(path) - 1
    if n < 1:
        return [], []
    p2 = [path[n][i] - path[0][i] for i in range(2)]
    d2 = math.sqrt(p2[0] * p2[0] + p2[1] * p2[1])
    if d2 <= eps:
        return [], []
    mid = n // 2
    s = n / 2.0 - mid
    p1 = [path[mid][i] * (1 - s) + path[mid + 1][i] * s - path[0][i] for i in range(2)]
    d1 = math.sqrt(p1[0] * p1[0] + p1[1] * p1[1])
    v0, v1 = 4 * d1 - d2, 3 * d2 - 4 * d1
    if v0 < 0.0:
        v0, v1 = 0.0, d2 * 2.0
    elif v1 < 0.0:
        v0, v1 = d2 * 2.0, 0.0
    elif v0 + v1 > 0.01 and abs(v0 - v1) / (v0 + v1) < 0.2:
        v0 = v1 = d2
    vec = (p2[0] / d2, p2[1] / d2, 0.0)
    ramp = []
    for f in range(n + 1):
        t = f / n
        ramp.append([c * (v0 * t + 0.5 * (v1 - v0) * t * t) for c in vec])
    mv = M.Movement()
    mv.endframe = n
    mv.motionflags = MOTION_EXTRA | LINEAR_AXIS[0] | LINEAR_AXIS[1]
    mv.v0 = v0
    mv.v1 = v1
    mv.angle = 0.0
    mv.vector = vec
    mv.position = tuple(ramp[n])
    return [mv], ramp


def extract_root_motion(m, poses):
    """Move the root bone's net ground-plane motion out of the poses and into one movement
    block, subtracting only the ramp `fit_movements` fitted.

    Whatever the ramp does not account for stays in the keys: andrei's `run_0` keeps its
    21 units of rise and 6.7 of fore-aft sway on the root while the block carries the 93.5
    forward, so the bound is still in the animation. Pinning the root instead takes the
    bound out with the motion and the character runs flat.

    Sound only because `angle` is 0: the engine applies anim_position as a pure translation
    of every world matrix, and translating the root of a hierarchy translates all of it, so
    no other bone's pose changes.
    """
    path = [list(p[0][0]) for p in poses]
    mvs, ramp = fit_movements(path)
    if not mvs:
        return [], poses
    out = []
    for f, p in enumerate(poses):
        q = [(list(pp), list(qq)) for pp, qq in p]
        q[0] = ([path[f][i] - ramp[f][i] for i in range(3)], q[0][1])
        out.append(q)
    return mvs, out


def _anim_block(m, t):
    """The mstudioanim_t array followed by its streams, packed the way studiomdl packs
    them: bone-major, channel-minor, no padding, first stream right after the array."""
    n = len(m.bones)
    head = bytearray(n * M.ANIM_STRIDE)
    body = bytearray()
    for b in m.bones:
        o = b.index * M.ANIM_STRIDE
        struct.pack_into("<f", head, o, t.weights.get(b.index, 0.0))
        for c in range(M.NUM_CHANNELS):
            v = t.chan.get((b.index, c))
            if not v:
                continue
            struct.pack_into("<i", head, o + 4 + c * 4, n * M.ANIM_STRIDE + len(body) - o)
            body += encode_channel(v)
    return bytes(head + body)


def _align(buf, at=0, n=4):
    while (at + len(buf)) % n:
        buf.append(0)


def movement_bytes(mv):
    """One mstudiomovement_t record."""
    return (struct.pack("<iifff", mv.endframe, mv.motionflags, mv.v0, mv.v1, mv.angle)
            + struct.pack("<3f", *mv.vector) + struct.pack("<3f", *mv.position))


def _movement_bytes(t):
    return b"".join(movement_bytes(mv) for mv in t.movements)


def _block_extent(m, a):
    """One past the last byte of an animation's mstudioanim_t array and its streams."""
    d, end = m.d, a.base + len(m.bones) * M.ANIM_STRIDE
    for b in m.bones:
        o = a.base + b.index * M.ANIM_STRIDE
        for off in struct.unpack_from("<7i", d, o + 4):
            if not off:
                continue
            p, left = o + off, a.numframes
            while left > 0:
                left -= d[p + 1]
                p += (d[p] + 1) * 2
            end = max(end, p)
    return end


def owned_spans(m, animindex):
    """The bytes each animation owns now, keyed ("block"|"move", animation index).

    None when two animations share a span or two spans partially overlap, since then
    neither can be handed its own bytes to write into. One shipped model does that,
    `weapons/rifle_rem700/view/v_rifle_rem700.mdl`, whose block 7 walks over blocks 8
    and 9.
    """
    spans = {}
    for a in m.anims:
        if a.numframes <= 0:
            continue
        try:
            end = _block_extent(m, a)
        except (IndexError, struct.error):
            return None
        if a.base % 4 or end > len(m.d):
            return None
        spans[("block", a.index)] = (a.base, end)
        desc = animindex + a.index * M.ANIMDESC_STRIDE
        nummove, moveindex = struct.unpack_from("<ii", m.d, desc + 0x10)
        if nummove > 0 and moveindex:
            at = desc + moveindex
            if at % 4 or at + nummove * M.MOVEMENT_STRIDE > len(m.d):
                return None
            spans[("move", a.index)] = (at, at + nummove * M.MOVEMENT_STRIDE)
    ordered = sorted(spans.values())
    for (_, prev_end), (start, _) in zip(ordered, ordered[1:]):
        if start < prev_end:
            return None
    return spans


def _extents(spans):
    """Maximal runs of bytes the animations hold between them.

    Blocks abut modulo 4-byte alignment: over 4412 shipped models every gap between two
    animation spans is 0 or 2 bytes, bar the four Gangrel female armors. So merging
    across 3 claims padding and never a section.
    """
    out = []
    for s, e in sorted(spans.values()):
        if out and s - out[-1][1] <= 3:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
