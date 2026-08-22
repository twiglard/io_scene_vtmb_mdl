#!/usr/bin/env python3
"""MDL v2531 animation writer. No bpy.

Nothing this reader does not parse ever moves. Each animation is reached through its own
`mstudioanimdesc_t.animindex` -- animdesc-relative and a signed int -- so a block is
rewritten into the bytes it already occupies whenever the new one fits there, and only
what has outgrown its own span goes past the end of the file. Evidence for the layout it
reproduces is in todo-vtmb-mdl-animation.md section 2.22.
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
    for b in m.bones:
        o = a.base + b.index * M.ANIM_STRIDE
        t.weights[b.index] = struct.unpack_from("<f", d, o)[0]
        offs = struct.unpack_from("<7i", d, o + 4)
        for c in range(M.NUM_CHANNELS):
            if offs[c]:
                t.chan[(b.index, c)] = [m.extract(o + offs[c], f)
                                        for f in range(a.numframes)]
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
# Undefined in ref/2531/studio.h and never read at runtime, but set on 10717 of 11013
# shipped blocks; motionflags as a whole reaches no engine code.
MOTION_EXTRA = 0x1000


def fit_movements(path, eps=1e-4):
    """mstudiomovement_t blocks reproducing a per-frame model-space path.

    v0/v1 are distance per block span, not per second: (v0+v1)/2 equals the block's own
    step in all 11013 shipped blocks. One block per frame is exact at every integer frame,
    which is the only place the engine is asked for; merging would only save bytes.
    `angle` stays 0. It is not unauthored -- 21 blocks in a patched install carry one,
    the gargoyle, hengeyokai and creation1 turn animations, all flagged 0x800 -- but a
    root path alone cannot say which part of a turn was the entity's, so this never
    invents one and the exporter warns when it replaces blocks that had one.
    """
    rel = [[p[i] - path[0][i] for i in range(3)] for p in path]
    if all(max(abs(c) for c in p) <= eps for p in rel):
        return []
    used = 0
    for p in rel:
        for i in range(3):
            if abs(p[i]) > eps:
                used |= LINEAR_AXIS[i]
    out, prev, vec = [], rel[0], [1.0, 0.0, 0.0]
    for f in range(1, len(rel)):
        d = [rel[f][i] - prev[i] for i in range(3)]
        step = math.sqrt(sum(c * c for c in d))
        if step > eps:
            vec = [c / step for c in d]
        mv = M.Movement()
        mv.endframe = f
        mv.motionflags = MOTION_EXTRA | used
        mv.v0 = mv.v1 = step
        mv.angle = 0.0
        mv.vector = tuple(vec)
        mv.position = tuple(rel[f])
        out.append(mv)
        prev = rel[f]
    return out


def extract_travel(m, poses):
    """Move the root bone's path out of the poses and into movement blocks.

    Equivalent whenever `angle` is 0, which is every shipped block but 21: the engine
    then applies anim_position as a pure translation of every world matrix, and
    translating the root of a hierarchy translates all of it, so no other bone's pose
    changes. A donor block carrying an angle has no equivalent here -- the caller is the
    one who can see the donor's blocks and owns that warning.
    """
    path = [list(p[0][0]) for p in poses]
    mvs = fit_movements(path)
    if not mvs:
        return [], poses
    out = []
    for p in poses:
        q = [(list(pp), list(qq)) for pp, qq in p]
        q[0] = (list(path[0]), q[0][1])
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


def _movement_bytes(t):
    buf = bytearray()
    for mv in t.movements:
        buf += struct.pack("<iifff", mv.endframe, mv.motionflags, mv.v0, mv.v1, mv.angle)
        buf += struct.pack("<3f", *mv.vector)
        buf += struct.pack("<3f", *mv.position)
    return bytes(buf)


def _emit_region(m, animindex, tracks, at):
    """The animation region as it would lie starting at file offset `at`, with the
    animdesc fields that point into it. Nothing outside the region is touched, so the
    caller can splice this over any span that holds only animation data."""
    buf = bytearray()
    patch = []
    for a in m.anims:
        t = tracks[a.index]
        desc = animindex + a.index * M.ANIMDESC_STRIDE
        patch.append((desc + 4, "<fii", (t.fps, t.flags, t.numframes)))
        if t.numframes <= 0:
            continue
        _align(buf, at)
        patch.append((desc + 0x30, "<i", (at + len(buf) - desc,)))
        buf += _anim_block(m, t)
        patch.append((desc + 0x10, "<i", (len(t.movements),)))
        if t.movements:
            _align(buf, at)
            patch.append((desc + 0x14, "<i", (at + len(buf) - desc,)))
            buf += _movement_bytes(t)
    return bytes(buf), patch


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


def _plan(m, tracks, spans, eof):
    """Where each piece of animation data goes, and how long the file ends up.

    Laid down one after another over the bytes the animations already hold between them,
    with whatever does not fit going past the end of the file. Packing the run afresh
    rather than each block into its own span is what stops a block that grew by ten bytes
    from stranding its whole old copy, and on an unchanged rewrite it reproduces the
    layout that was there.
    """
    items = []
    for a in m.anims:
        t = tracks[a.index]
        if t.numframes <= 0:
            continue
        items.append((("block", a.index), _anim_block(m, t)))
        if t.movements:
            items.append((("move", a.index), _movement_bytes(t)))
    # Studiomdl's order: every block, then every movement array. Fixed rather than read
    # off current addresses, so placement cannot feed back into the next write's order.
    items.sort(key=lambda kv: (kv[0][0] != "block", kv[0][1]))
    holes = _extents(spans)
    # The run reaching the end of the file has no section behind it to run into, and
    # nothing below it either -- so the file ends wherever this run does, and shrinks.
    tail = eof
    if holes and holes[-1][1] >= eof:
        tail, holes[-1][1] = holes[-1][0], float("inf")
    at = {}
    for key, buf in items:
        at[key] = tail + -tail % 4
        for hole in holes:
            start = hole[0] + -hole[0] % 4
            if start + len(buf) <= hole[1]:
                at[key], hole[0] = start, start + len(buf)
                break
        tail = max(tail, at[key] + len(buf))
    return items, at, tail


def region_start(m):
    """Where the whole animation region may be re-emitted in one piece, compacting it.

    Only a region this writer laid down itself qualifies, and the test is byte identity:
    re-emitting the animations exactly as they are must reproduce everything from the
    first block to EOF. A file straight from studiomdl fails it -- its animation data is
    followed by mesh and texture blocks this code does not parse -- and then this falls
    back to the end of the file, where `Writer` uses it only as the last resort behind
    packing the payload into the bytes it already holds.
    """
    starts = [a.base for a in m.anims if a.numframes > 0]
    if not starts:
        return len(m.d)
    at = min(starts)
    animindex = struct.unpack_from("<ii", m.d, M.HDR_NUMANIM)[1]
    if at <= animindex + len(m.anims) * M.ANIMDESC_STRIDE:
        return len(m.d)
    buf, _ = _emit_region(m, animindex, [read_tracks(m, a) for a in m.anims], at)
    return at if m.d[at:] == buf else len(m.d)


class Writer:
    """Rewrites every animation of one .mdl into the bytes they already hold between them.

    The payload is packed end to end there and only the overflow -- more frames, more
    channels, a longer encoding -- goes past the end of the file. Ahead of that: a region
    this writer laid down itself is re-emitted whole at its own start. Behind it: if the
    spans cannot be told apart at all, the whole payload is appended.
    """

    def __init__(self, m):
        if not m.anims:
            raise ValueError("%s has no animations to write" % m.path)
        self.m = m
        self.animindex = struct.unpack_from("<ii", m.d, M.HDR_NUMANIM)[1]
        self.at = region_start(m)
        self.spans = owned_spans(m, self.animindex) if self.at == len(m.d) else None

    def build(self, tracks, scales):
        m = self.m
        out = bytearray(m.d) if self.spans is not None else bytearray(m.d[:self.at])
        for b in m.bones:
            off = struct.unpack_from("<ii", m.d, M.HDR_NUMBONES)[1] + b.index * M.BONE_STRIDE
            struct.pack_into("<3f", out, off + M.BONE_POSSCALE, *scales[b.index][0:3])
            struct.pack_into("<4f", out, off + M.BONE_ROTSCALE, *scales[b.index][3:7])

        if self.spans is None:
            region, patch = _emit_region(m, self.animindex, tracks, self.at)
            for off, fmt, values in patch:
                struct.pack_into(fmt, out, off, *values)
            out += region
        else:
            self._place(out, tracks)
        struct.pack_into("<i", out, M.HDR_LENGTH, len(out))
        return bytes(out)

    def _place(self, out, tracks):
        m = self.m
        items, at, tail = _plan(m, tracks, self.spans, len(out))
        if tail > len(out):
            out.extend(b"\0" * (tail - len(out)))
        for key, buf in items:
            out[at[key]:at[key] + len(buf)] = buf
        del out[tail:]
        for a in m.anims:
            t = tracks[a.index]
            desc = self.animindex + a.index * M.ANIMDESC_STRIDE
            struct.pack_into("<fii", out, desc + 4, t.fps, t.flags, t.numframes)
            if t.numframes <= 0:
                continue
            struct.pack_into("<i", out, desc + 0x30, at[("block", a.index)] - desc)
            struct.pack_into("<i", out, desc + 0x10, len(t.movements))
            if t.movements:
                struct.pack_into("<i", out, desc + 0x14, at[("move", a.index)] - desc)


def write_many(m, edits):
    """Replace any number of animations with authored local poses and re-emit the file.

    `edits` maps animation index to a dict holding `poses` and optionally `fps`, `flags`
    and `movements`. Widening a bone's scales to fit the new poses changes how *every*
    animation in the file decodes, since they share `mstudiobone_t`, so all of them are
    requantised here -- and that is also why this takes them all at once. Per-animation
    calls each append a whole fresh animation region, which on move_and_ranged.mdl's 674
    overruns the signed int in `studiohdr_t.length` before it finishes.
    """
    for i in edits:
        if not 0 <= i < len(m.anims):
            raise ValueError("no animation %d in %s" % (i, m.path))
    old = file_scales(m)
    scales = fit_scales(m, [e["poses"] for e in edits.values()])
    tracks = [read_tracks(m, a) for a in m.anims]
    for t in tracks:
        rescale(t, old, scales)
    for i, e in edits.items():
        t = tracks[i]
        t.numframes = len(e["poses"])
        t.chan = quantise(m, e["poses"], scales)
        for field in ("fps", "flags"):
            if e.get(field) is not None:
                setattr(t, field, e[field])
        if e.get("movements") is not None:
            t.movements = list(e["movements"])
    return Writer(m).build(tracks, scales)


def write_poses(m, index, poses, fps=None, flags=None, movements=None):
    return write_many(m, {index: {"poses": poses, "fps": fps, "flags": flags,
                                  "movements": movements}})


def rewrite(m, edits=None):
    """Re-emit every animation. `edits` maps animation index to a Tracks whose channels
    are already quantised against `scales`; anything absent is carried over unchanged.

    With no edits and no scale change this reproduces the original animation region byte
    for byte, relocated -- which is the only test that covers the packing rules.
    """
    tracks = [read_tracks(m, a) for a in m.anims]
    if edits:
        for i, t in edits.items():
            tracks[i] = t
    return Writer(m).build(tracks, file_scales(m))
