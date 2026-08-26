#!/usr/bin/env python3
"""MDL v2531 reader: skeleton, animation, geometry. No bpy.

Clean room: written from the disassembly and from parsing shipped files. Format
evidence is in plans/todo-ghidra-vtmb-recon.md sections 19 and 21.
"""

import math
import os
import struct
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import normal_table as NT
else:
    from . import normal_table as NT

MAGIC = b"IDST"
VERSION = 2531

HDR_LENGTH = 140
HDR_NUMBONES = 240
HDR_NUMANIM = 264
HDR_NUMSEQ = 272
HDR_NUMTEXTURES = 292
HDR_NUMCDTEXTURES = 300
HDR_NUMSKINREF = 308
HDR_NUMBODYPARTS = 320
HDR_NUMHITBOXSETS = 256
HDR_NUMINCLUDEMODELS = 404

# Measured, not from Valve: the record is 116 bytes with the name offset at +0, and no
# other stride parses the corpus. The 24 trailing -1s are the pose-parameter remap.
INCLUDE_STRIDE = 116

HITBOXSET_STRIDE = 12
BBOX_STRIDE = 32

BONE_STRIDE = 160
BONE_POS = 0x20
BONE_QUAT = 0x2C
BONE_POSSCALE = 0x3C
BONE_ROTSCALE = 0x48
BONE_POSETOBONE = 0x58
BONE_FLAGS = 0x88

# Troika's meaning. Valve's studio.h calls 0x2 BONE_PHYSICS_PROCEDURAL, which it is not here.
BONE_ROTATION_FROM_ROOT = 0x2

ANIMDESC_STRIDE = 72
SEQDESC_STRIDE = 764
SEQ_BBMIN = 0x1c
SEQ_ANIM = 0x38
SEQ_GROUPSIZE = 0x23C

MOVEMENT_STRIDE = 44

ANIM_STRIDE = 32
NUM_CHANNELS = 7

TEXTURE_STRIDE = 20
BODYPART_STRIDE = 16
MODEL_STRIDE = 224
MESH_STRIDE = 60

VERTEX_STRIDE = {0: 44, 1: 12, 2: 8}
# What a packed position scalar is multiplied by before the model's own quant_scale.
# filetype 2's byte goes through g_byteToFloatTable (i/255) at StudioRender.dll 0x2c013be7,
# filetype 1's ushort is FILDed raw at 0x2c013ac8, so only filetype 2's scale is an extent.
QUANT_NORM = {1: 1.0, 2: 1.0 / 255.0}
QUANT_MAX = {1: 65535, 2: 255}
# 3 weight bytes, the bone-count byte, 4 bone shorts, then pos/normal/uv as 8 floats.
_VERT0 = struct.Struct("<4B4h8f")
# Tabulated rather than divided, and kept as the same expressions so the bits do not move.
_W255 = [i / 255.0 for i in range(256)]
_W4TH = [(255 - s) / 255.0 for s in range(766)]


def anim_position(a, frame):
    """(translation, yaw degrees) that studiomdl extracted out of the animation and left
    for the engine to re-apply. Studio_AnimPosition, shared/bone_setup.cpp:3162."""
    pos, yaw, prev = [0.0, 0.0, 0.0], 0.0, 0.0
    for mv in a.movements:
        if mv.endframe >= frame:
            span = mv.endframe - prev
            f = (frame - prev) / span if span else 0.0
            d = mv.v0 * f + 0.5 * (mv.v1 - mv.v0) * f * f
            return [pos[i] + d * mv.vector[i] for i in range(3)], yaw * (1 - f) + mv.angle * f
        prev = float(mv.endframe)
        pos, yaw = list(mv.position), mv.angle
    return pos, yaw


def root_motion_matrix(a, frame):
    """anim_position as a 3x4 model-space offset. Yaw is QAngle.y, so about Z here.

    Left-multiply every world matrix by this, not just the root's: the engine puts the
    offset on the entity transform, above the whole skeleton, and a BONE_ROTATION_FROM_ROOT
    bone takes its rotation from the model frame rather than from its parent, so offsetting
    only the root would leave those bones behind.
    """
    pos, yaw = anim_position(a, frame)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return [[c, -s, 0.0, pos[0]],
            [s, c, 0.0, pos[1]],
            [0.0, 0.0, 1.0, pos[2]]]


def mat_identity():
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]


def mat_mul(a, b):
    # Unrolled: the sum()-over-genexpr form was 96% of a chained import. Same association order
    # and same floats, bar an all-negative-zero row, which quantise() maps to the same int16.
    a0, a1, a2 = a
    b0, b1, b2 = b
    a00, a01, a02, a03 = a0
    a10, a11, a12, a13 = a1
    a20, a21, a22, a23 = a2
    b00, b01, b02, b03 = b0
    b10, b11, b12, b13 = b1
    b20, b21, b22, b23 = b2
    return [
        [a00 * b00 + a01 * b10 + a02 * b20,
         a00 * b01 + a01 * b11 + a02 * b21,
         a00 * b02 + a01 * b12 + a02 * b22,
         a00 * b03 + a01 * b13 + a02 * b23 + a03],
        [a10 * b00 + a11 * b10 + a12 * b20,
         a10 * b01 + a11 * b11 + a12 * b21,
         a10 * b02 + a11 * b12 + a12 * b22,
         a10 * b03 + a11 * b13 + a12 * b23 + a13],
        [a20 * b00 + a21 * b10 + a22 * b20,
         a20 * b01 + a21 * b11 + a22 * b21,
         a20 * b02 + a21 * b12 + a22 * b22,
         a20 * b03 + a21 * b13 + a22 * b23 + a23],
    ]


def mat_from_quat_pos(q, p):
    """3x4 row-major affine, rows [x y z t]; matches the engine's matrix3x4_t."""
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), p[0]],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), p[1]],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), p[2]],
    ]


def mat_inverse(m):
    """Inverse of a 3x4 affine. Not a transpose: file quaternions are unit only to
    ~4.4e-5, so these matrices carry that much scale and transposing leaves 7e-3 units
    of error by the end of a leg chain."""
    a, b, c = (m[0][0], m[0][1], m[0][2])
    d, e, f = (m[1][0], m[1][1], m[1][2])
    g, h, i = (m[2][0], m[2][1], m[2][2])
    co = [[e * i - f * h, c * h - b * i, b * f - c * e],
          [f * g - d * i, a * i - c * g, c * d - a * f],
          [d * h - e * g, b * g - a * h, a * e - b * d]]
    det = a * co[0][0] + b * co[1][0] + c * co[2][0]
    r = [[co[j][k] / det for k in range(3)] for j in range(3)]
    t = [-sum(r[j][k] * m[k][3] for k in range(3)) for j in range(3)]
    return [r[j] + [t[j]] for j in range(3)]


def quat_from_mat(m):
    """(x, y, z, w) from the rotation part; the inverse of mat_from_quat_pos. The four
    branches each divide by the largest component, which the near-180-degree cases need."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
             (m[1][0] - m[0][1]) / s, 0.25 * s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s,
             (m[2][1] - m[1][2]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s,
             (m[0][2] - m[2][0]) / s]
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s,
             (m[1][0] - m[0][1]) / s]
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def mat_max_dev_from_identity(m):
    ident = mat_identity()
    return max(abs(m[i][j] - ident[i][j]) for i in range(3) for j in range(4))


class Bone:
    __slots__ = ("index", "name", "parent", "pos", "quat", "posscale", "rotscale",
                 "posetobone", "flags")


class Anim:
    __slots__ = ("index", "name", "fps", "flags", "numframes", "base", "movements")


class Movement:
    __slots__ = ("endframe", "motionflags", "v0", "v1", "angle", "vector", "position")


class Seq:
    __slots__ = ("index", "label", "activity", "flags", "groupsize", "blends",
                 "bbmin", "bbmax")


class Mesh:
    __slots__ = ("index", "material", "numvertices", "vertexoffset", "materialtype",
                 "materialparam")


class Model:
    __slots__ = ("index", "name", "nummeshes", "numvertices", "vertexbase",
                 "tangentbase", "filetype", "quant_offset", "quant_scale", "meshes")


class BodyPart:
    __slots__ = ("index", "name", "base", "models")


class HitboxSet:
    __slots__ = ("index", "name", "boxes")


class Hitbox:
    __slots__ = ("index", "bone", "group", "bbmin", "bbmax")


class Vertex:
    __slots__ = ("pos", "normal", "uv", "bones", "weights", "numbones")


class Mdl:
    def __init__(self, path, data=None):
        self.path = path
        if data is None:
            with open(path, "rb") as f:
                data = f.read()
        self.d = data
        d = self.d
        if d[0:4] != MAGIC:
            raise ValueError("not an MDL: magic is %r" % d[0:4])
        self.version = struct.unpack_from("<i", d, 4)[0]
        if self.version != VERSION:
            raise ValueError("expected STUDIO_VERSION %d, got %d"
                             % (VERSION, self.version))
        self.checksum = struct.unpack_from("<i", d, 8)[0]
        self.name = self._cstr(12)
        self.length = struct.unpack_from("<i", d, HDR_LENGTH)[0]
        if self.length > len(d):
            raise ValueError("header length %d exceeds file size %d"
                             % (self.length, len(d)))
        self.trailer = d[self.length:]
        # One animation's expanded RLE channels, dropped when the block moves.
        self._chan, self._chan_base, self._chan_table = {}, None, []

        numbones, boneindex = struct.unpack_from("<ii", d, HDR_NUMBONES)
        numanim, animindex = struct.unpack_from("<ii", d, HDR_NUMANIM)
        numseq, seqindex = struct.unpack_from("<ii", d, HDR_NUMSEQ)

        self.bones = [self._read_bone(boneindex + i * BONE_STRIDE, i)
                      for i in range(numbones)]
        self.anims = [self._read_anim(animindex + i * ANIMDESC_STRIDE, i)
                      for i in range(numanim)]
        self.seqs = [self._read_seq(seqindex + i * SEQDESC_STRIDE, i)
                     for i in range(numseq)]
        self._read_materials()
        self._read_bodyparts()
        self._read_hitboxsets()
        self._read_includes()

    def _cstr(self, off):
        end = self.d.find(b"\x00", off)
        return self.d[off:end].decode("latin1")

    def _read_bone(self, off, i):
        d, b = self.d, Bone()
        b.index = i
        b.name = self._cstr(off + struct.unpack_from("<i", d, off)[0])
        b.parent = struct.unpack_from("<i", d, off + 4)[0]
        b.pos = struct.unpack_from("<3f", d, off + BONE_POS)
        b.quat = struct.unpack_from("<4f", d, off + BONE_QUAT)
        b.posscale = struct.unpack_from("<3f", d, off + BONE_POSSCALE)
        b.rotscale = struct.unpack_from("<4f", d, off + BONE_ROTSCALE)
        flat = struct.unpack_from("<12f", d, off + BONE_POSETOBONE)
        b.posetobone = [list(flat[0:4]), list(flat[4:8]), list(flat[8:12])]
        b.flags = struct.unpack_from("<i", d, off + BONE_FLAGS)[0]
        return b

    def _read_anim(self, off, i):
        d, a = self.d, Anim()
        a.index = i
        a.name = self._cstr(off + struct.unpack_from("<i", d, off)[0])
        a.fps, a.flags, a.numframes = struct.unpack_from("<fii", d, off + 4)
        nummove, moveindex = struct.unpack_from("<ii", d, off + 0x10)
        a.movements = [self._read_movement(off + moveindex + k * MOVEMENT_STRIDE)
                       for k in range(nummove)] if moveindex else []
        a.base = off + struct.unpack_from("<i", d, off + 0x30)[0]
        return a

    def _read_movement(self, off):
        mv = Movement()
        (mv.endframe, mv.motionflags, mv.v0, mv.v1,
         mv.angle) = struct.unpack_from("<iifff", self.d, off)
        mv.vector = struct.unpack_from("<3f", self.d, off + 20)
        mv.position = struct.unpack_from("<3f", self.d, off + 32)
        return mv

    def _read_seq(self, off, i):
        d, s = self.d, Seq()
        s.index = i
        s.label = self._cstr(off + struct.unpack_from("<i", d, off)[0])
        s.activity = self._cstr(off + struct.unpack_from("<i", d, off + 4)[0])
        # STUDIO_LOOPING is bit 0 of THIS field, not of mstudioanimdesc_t.flags at the
        # same offset of that struct: a file carrying the animdesc bit and not this one
        # plays once and stops -- measured in game 2026-08-22 on ztest/zskel_loop.mdl.
        s.flags = struct.unpack_from("<i", d, off + 8)[0]
        # Mod_GetBounds (engine.dll 200b5970) only *widens* hull_min/hull_max with this,
        # so a zero one does not stop the model drawing.  vampire.dll 10090c80 is the one
        # place it is read alone, sizing an entity's collision hull, and substitutes 0.0
        # per component where this box is the smaller.
        s.bbmin = struct.unpack_from("<3f", d, off + SEQ_BBMIN)
        s.bbmax = struct.unpack_from("<3f", d, off + SEQ_BBMIN + 12)
        gx, gy = struct.unpack_from("<ii", d, off + SEQ_GROUPSIZE)
        s.groupsize = (gx, gy)
        s.blends = [[struct.unpack_from("<h", d, off + SEQ_ANIM + x * 0x20 + y * 2)[0]
                     for y in range(gy)] for x in range(gx)]
        return s

    def _read_materials(self):
        d = self.d
        ntex, texindex = struct.unpack_from("<ii", d, HDR_NUMTEXTURES)
        ncd, cdindex = struct.unpack_from("<ii", d, HDR_NUMCDTEXTURES)
        nskinref, nskinfam, skinindex = struct.unpack_from("<3i", d, HDR_NUMSKINREF)
        self.materials = []
        for i in range(ntex):
            off = texindex + i * TEXTURE_STRIDE
            self.materials.append(
                self._cstr(off + struct.unpack_from("<i", d, off)[0]))
        self.material_paths = [
            self._cstr(struct.unpack_from("<i", d, cdindex + i * 4)[0])
            for i in range(ncd)]
        self.skins = [
            [struct.unpack_from("<h", d, skinindex + (f * nskinref + s) * 2)[0]
             for s in range(nskinref)] for f in range(nskinfam)]

    def _read_bodyparts(self):
        d = self.d
        nbp, bpindex = struct.unpack_from("<ii", d, HDR_NUMBODYPARTS)
        self.bodyparts = []
        for i in range(nbp):
            off = bpindex + i * BODYPART_STRIDE
            szname, nmodels, base, modelindex = struct.unpack_from("<4i", d, off)
            bp = BodyPart()
            bp.index, bp.name, bp.base = i, self._cstr(off + szname), base
            bp.models = [self._read_model(off + modelindex + j * MODEL_STRIDE, j)
                         for j in range(nmodels)]
            self.bodyparts.append(bp)

    def _read_model(self, off, j):
        d, m = self.d, Model()
        m.index = j
        m.name = self._cstr(off)
        m.nummeshes, meshindex = struct.unpack_from("<ii", d, off + 136)
        m.numvertices, vertexindex, tangentsindex, m.filetype = \
            struct.unpack_from("<4i", d, off + 144)
        m.vertexbase = off + vertexindex
        m.tangentbase = off + tangentsindex
        # filetype 1 and 2 quantise position; unkvect is min[3] then (max-min)/range[3].
        m.quant_offset = struct.unpack_from("<3f", d, off + 160)
        m.quant_scale = struct.unpack_from("<3f", d, off + 172)
        m.meshes = []
        for k in range(m.nummeshes):
            eo = off + meshindex + k * MESH_STRIDE
            e = Mesh()
            e.index = k
            e.material, _, e.numvertices, e.vertexoffset = \
                struct.unpack_from("<4i", d, eo)
            e.materialtype, e.materialparam = struct.unpack_from("<2i", d, eo + 24)
            m.meshes.append(e)
        return m

    def _read_hitboxsets(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMHITBOXSETS)
        self.hitboxsets = []
        for i in range(n):
            off = idx + i * HITBOXSET_STRIDE
            s = HitboxSet()
            s.index = i
            nb, bi = struct.unpack_from("<ii", d, off + 4)
            s.name = self._cstr(off + struct.unpack_from("<i", d, off)[0])
            s.boxes = []
            for k in range(nb):
                bo = off + bi + k * BBOX_STRIDE
                x = Hitbox()
                x.index = k
                x.bone, x.group = struct.unpack_from("<ii", d, bo)
                x.bbmin = struct.unpack_from("<3f", d, bo + 8)
                x.bbmax = struct.unpack_from("<3f", d, bo + 20)
                s.boxes.append(x)
            self.hitboxsets.append(s)

    def _read_includes(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMINCLUDEMODELS)
        self.includes = []
        for i in range(n):
            off = idx + i * INCLUDE_STRIDE
            self.includes.append(
                self._cstr(off + struct.unpack_from("<i", d, off)[0]))

    def vertices(self, model):
        """Decoded vertices for one model, in model.vertexbase order."""
        d, out = self.d, []
        stride = VERTEX_STRIDE.get(model.filetype)
        if stride is None:
            raise ValueError("unknown vertex filetype %d" % model.filetype)
        need = model.numvertices * stride
        # One iter_unpack over the span rather than six unpack_from per vertex; the
        # per-vertex loop below still runs on a short file so its error stays the same one.
        if model.filetype == 0 and model.vertexbase + need <= len(d):
            for t in _VERT0.iter_unpack(
                    memoryview(d)[model.vertexbase:model.vertexbase + need]):
                v = Vertex()
                v.numbones = t[3] % 5
                v.weights = [_W255[t[0]], _W255[t[1]], _W255[t[2]],
                             _W4TH[t[0] + t[1] + t[2]]]
                v.bones = list(t[4:8])
                v.pos = t[8:11]
                v.normal = t[11:14]
                v.uv = t[14:16]
                out.append(v)
            return out
        for i in range(model.numvertices):
            o = model.vertexbase + i * stride
            v = Vertex()
            if model.filetype == 0:
                w = d[o:o + 3]
                # Four bones: the fourth weight is not stored, it is the shortfall, and
                # byte +3 % 5 is how many of the four count (StudioRender+0x172b9).
                v.numbones = d[o + 3] % 5
                v.weights = [x / 255.0 for x in w] + [(255 - sum(w)) / 255.0]
                v.bones = list(struct.unpack_from("<4h", d, o + 4))
                v.pos = struct.unpack_from("<3f", d, o + 12)
                v.normal = struct.unpack_from("<3f", d, o + 24)
                v.uv = struct.unpack_from("<2f", d, o + 36)
            else:
                q = struct.unpack_from("<3H", d, o) if model.filetype == 1 \
                    else struct.unpack_from("<3B", d, o)
                nrm = QUANT_NORM[model.filetype]
                v.pos = tuple(model.quant_offset[c] + q[c] * nrm * model.quant_scale[c]
                              for c in range(3))
                # An index into one of StudioRender's two tables, not a vector. Out of
                # range is a file this will not invent a normal for, and (0,0,1) is what
                # every one of these read before the tables were known.
                raw = (struct.unpack_from("<H", d, o + 6)[0]
                       if model.filetype == 1 else d[o + 3])
                v.normal = NT.decode(model.filetype, raw) or (0.0, 0.0, 1.0)
                uv = struct.unpack_from("<2H", d, o + 8) if model.filetype == 1 \
                    else struct.unpack_from("<2H", d, o + 4)
                v.uv = (uv[0] / 65535.0, uv[1] / 65535.0)
                v.bones, v.weights = [0] * 4, [1.0, 0.0, 0.0, 0.0]
                v.numbones = 1
            out.append(v)
        return out

    def extract(self, off, frame):
        """One int16 out of an RLE channel. num.total is a byte, so runs cap at 255."""
        d, k = self.d, frame
        while d[off + 1] <= k:
            k -= d[off + 1]
            off += (d[off] + 1) * 2
        valid = d[off]
        return struct.unpack_from("<h", d, off + (k + 1 if valid > k else valid) * 2)[0]

    def channel(self, off, numframes):
        """One RLE channel as numframes int16s, decoded once and kept.

        extract() restarts at the first run for every frame, so reading a whole animation
        costs O(frames**2 / run length) per channel and unpacks one int16 at a time. Same
        values, same held-tail rule -- a run stores `valid` of its `total` frames and the
        rest repeat the last one -- but each run is unpacked once and each byte walked
        once. Degenerate runs are reproduced rather than rejected: valid == 0 reads the
        header word as extract's arithmetic does, total == 0 contributes nothing and the
        walk moves on.
        """
        out = self._chan.get(off)
        if out is not None:
            return out
        d, out, o = self.d, [], off
        while len(out) < numframes:
            valid, total = d[o], d[o + 1]
            if valid:
                vals = struct.unpack_from("<%dh" % valid, d, o + 2)
                hold = vals[valid - 1]
            else:
                vals, hold = (), struct.unpack_from("<h", d, o)[0]
            for k in range(min(total, numframes - len(out))):
                out.append(vals[k] if k < valid else hold)
            o += (valid + 1) * 2
        self._chan[off] = out
        return out

    def anim_channels(self, anim):
        """Per bone, its seven channels as lists of numframes int16s, None where absent.

        Built once per animation and kept until the next one asks, which is the access
        order every caller has: animation outer, frame inner. It is what takes the
        per-frame path down to an index -- the header unpack, the offset arithmetic and
        the channel decode all happen here instead of once per bone per frame.
        """
        if anim.base != self._chan_base:
            self._chan, self._chan_base = {}, anim.base
            d, nf, tab = self.d, max(1, anim.numframes), []
            for b in self.bones:
                o = anim.base + b.index * ANIM_STRIDE
                offs = struct.unpack_from("<7i", d, o + 4)
                tab.append([self.channel(o + x, nf) if x else None for x in offs])
            self._chan_table = tab
        return self._chan_table

    def local_pose(self, anim, frame):
        """Parent-relative (pos, quat) per bone. Rotation is a bare product, no base."""
        out = []
        for b, ch in zip(self.bones, self.anim_channels(anim)):
            pos = [b.pos[c] + ch[c][frame] * b.posscale[c] if ch[c] else b.pos[c]
                   for c in range(3)]
            quat = [ch[3 + c][frame] * b.rotscale[c] if ch[3 + c] else b.quat[c]
                    for c in range(4)]
            out.append((pos, quat))
        return out

    def world_matrices(self, local, engine_flags=True):
        """FK down the hierarchy, in model space, as the engine draws it.

        Parent-before-child holds in every model checked; checked anyway because a
        violation would silently use a stale matrix. With engine_flags, a bone carrying
        BONE_ROTATION_FROM_ROOT gets the branch at client.dll+0x8ffc5: rotation composed
        against the root frame, translation still from the chain.
        """
        world = []
        for b in self.bones:
            m = mat_from_quat_pos(local[b.index][1], local[b.index][0])
            if b.parent >= 0:
                if b.parent >= b.index:
                    raise ValueError("bone %d parent %d is not earlier"
                                     % (b.index, b.parent))
                chained = mat_mul(world[b.parent], m)
                if engine_flags and b.flags & BONE_ROTATION_FROM_ROOT:
                    m = [m[i][0:3] + [chained[i][3]] for i in range(3)]
                else:
                    m = chained
            world.append(m)
        return world

    def bone_remap(self, src):
        """Index in src per bone of self, -1 where absent. The engine matches a chained
        model to its chaining model by bone name (client.dll `102ad3b6`)."""
        idx = {b.name: b.index for b in src.bones}
        return [idx.get(b.name, -1) for b in self.bones]

    def retarget_pose(self, src, anim, frame, remap=None):
        """A chained animation in self's bone order. It decodes against src's own bones
        -- rotscale and the additive pos base both come from the file it lives in."""
        if remap is None:
            remap = self.bone_remap(src)
        local = src.local_pose(anim, frame)
        return [local[remap[b.index]] if remap[b.index] >= 0 else (b.pos, b.quat)
                for b in self.bones]

    def rest_world_matrices(self):
        """Plain FK, no engine_flags: posetobone is the inverse of *this*, and skinning
        needs the pair to cancel. selfcheck.py asserts that over the whole corpus."""
        return self.world_matrices([(b.pos, b.quat) for b in self.bones],
                                   engine_flags=False)
