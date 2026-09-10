#!/usr/bin/env python3
"""MDL v2531 reader: skeleton, animation, geometry. No bpy.

Clean room: written from the disassembly and from parsing shipped files. Format
evidence is in plans/todo-ghidra-vtmb-recon.md sections 19 and 21.
"""

import math
import os
import re
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
HDR_NUMFLEXDESC = 344
HDR_NUMFLEXCONTROLLERS = 352
HDR_NUMFLEXRULES = 360
HDR_NUMHITBOXSETS = 256
HDR_NUMATTACHMENTS = 328
HDR_NUMINCLUDEMODELS = 404
HDR_NUMSPRINGBONES = 396
HDR_NUMPOSEPARAMS = 384
HDR_NUMMOUTHS = 376

# Measured, not from Valve: the record is 116 bytes with the name offset at +0, and no
# other stride parses the corpus. The 24 trailing -1s are the pose-parameter remap.
INCLUDE_STRIDE = 116

FLEXCTRL_STRIDE = 20
FLEXRULE_STRIDE = 12
FLEXOP_STRIDE = 8

# client.dll 100c3dd3 dispatches op-1 through the 7-entry table at 100c3ee8, so the numbering
# is measured and not inherited from the 2003 tree.
STUDIO_CONST, STUDIO_FETCH1, STUDIO_FETCH2 = 1, 2, 3
STUDIO_ADD, STUDIO_SUB, STUDIO_MUL, STUDIO_DIV = 4, 5, 6, 7
_BINOP = {STUDIO_ADD: "+", STUDIO_SUB: "-", STUDIO_MUL: "*", STUDIO_DIV: "/"}
_PREC = {STUDIO_ADD: 1, STUDIO_SUB: 1, STUDIO_MUL: 2, STUDIO_DIV: 2}
_OPOF = {v: k for k, v in _BINOP.items()}


def flex_expr(ops, controllers, flexdescs):
    """The op block as studiomdl's own QC expression, or None if the stack does not balance.

    Option_Flexrule (studiomdl.cpp:2914) is the syntax: a bare identifier is a flex
    controller, `%name` is a flexdesc, and precedence is +- 1, */ 2.
    """
    st = []
    for op, raw in ops:
        if op == STUDIO_CONST:
            st.append((repr(round(struct.unpack("<f", struct.pack("<i", raw))[0], 6)), 3))
        elif op == STUDIO_FETCH1:
            if not 0 <= raw < len(controllers):
                return None
            st.append((controllers[raw].name, 3))
        elif op == STUDIO_FETCH2:
            if not 0 <= raw < len(flexdescs):
                return None
            st.append(("%" + flexdescs[raw], 3))
        elif op in _BINOP:
            if len(st) < 2:
                return None
            (rt, rp), (lt, lp) = st.pop(), st.pop()
            p = _PREC[op]
            # A right operand at equal precedence has to keep its parentheses: - and / are
            # not associative, and + and * only look it up to float rounding.
            st.append(("%s %s %s" % (("(%s)" % lt) if lp < p else lt, _BINOP[op],
                                     ("(%s)" % rt) if rp <= p else rt), p))
        else:
            return None
    return st[0][0] if len(st) == 1 else None


_TOK = re.compile(r"\s*(%?[A-Za-z_][A-Za-z0-9_]*|[0-9.][0-9.eE+-]*|[()+*/-])")


def parse_flex_expr(text, controllers, flexdescs):
    """A QC expression back to [(op, raw)], raising ValueError with the offending token.

    Names are matched case-insensitively, the way all three stricmp lookups in
    Option_Flexrule are.
    """
    toks, i = [], 0
    while i < len(text):
        m = _TOK.match(text, i)
        if not m:
            if text[i:].strip():
                raise ValueError("cannot read %r" % text[i:])
            break
        toks.append(m.group(1))
        i = m.end()
    ctl = {c.name.lower(): k for k, c in enumerate(controllers)}
    fdx = {n.lower(): k for k, n in enumerate(flexdescs)}
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def atom():
        t = peek()
        if t is None:
            raise ValueError("the expression ends where a value was expected")
        pos[0] += 1
        if t == "(":
            v = expr(1)
            if peek() != ")":
                raise ValueError("unclosed (")
            pos[0] += 1
            return v
        if t == "-":
            # Unary minus: studiomdl has none, so it is written out as 0 - x.
            return [(STUDIO_CONST, struct.unpack("<i", struct.pack("<f", 0.0))[0])]                 + atom() + [(STUDIO_SUB, 0)]
        if t[0] == "%":
            k = fdx.get(t[1:].lower())
            if k is None:
                raise ValueError("no flex named %r" % t[1:])
            return [(STUDIO_FETCH2, k)]
        if t[0].isdigit() or t[0] == ".":
            return [(STUDIO_CONST, struct.unpack("<i", struct.pack("<f", float(t)))[0])]
        k = ctl.get(t.lower())
        if k is None:
            raise ValueError("no flex controller named %r" % t)
        return [(STUDIO_FETCH1, k)]

    def expr(p):
        out = atom() if p > 2 else expr(p + 1)
        while peek() in _OPOF and _PREC[_OPOF[peek()]] == p:
            op = _OPOF[toks[pos[0]]]
            pos[0] += 1
            out = out + (atom() if p > 1 else expr(p + 1)) + [(op, 0)]
        return out

    out = expr(1)
    if pos[0] != len(toks):
        raise ValueError("trailing %r" % " ".join(toks[pos[0]:]))
    return out


class FlexController:
    __slots__ = ("index", "name", "type", "link", "min", "max")

    def __repr__(self):
        return "<FlexController %s %s>" % (self.type, self.name)


class FlexRule:
    __slots__ = ("index", "flex", "name", "ops", "expr")

    def __repr__(self):
        return "<FlexRule %s = %s>" % (self.name, self.expr)


HITBOXSET_STRIDE = 12
ATTACHMENT_STRIDE = 60
SPRINGBONE_STRIDE = 28
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
SEQ_ACTIVITY = 0x04
SEQ_FLAGS = 0x08
SEQ_BBMIN = 0x1c
SEQ_ANIM = 0x38
SEQ_GROUPSIZE = 0x23C
SEQ_NUMEVENTS = 0x14
SEQ_EVENTINDEX = 0x18

# mstudioevent_t: float cycle, int event, int type, char options[64].
EVENT_STRIDE = 76
EVENT_OPTIONS = 0x0c
EVENT_OPTIONS_LEN = 64

# mstudioposeparamdesc_t: int sznameindex (struct-relative), int flags, float start,
# end, loop. 0 of these on 3994 of the 4445 corpus models, 2 on 449 and 4 on 2.
POSEPARAM_STRIDE = 20
SEQ_PARAMINDEX = 0x244
SEQ_PARAMSTART = 0x24c
SEQ_PARAMEND = 0x254
SEQ_ENTRYNODE = 0x270
SEQ_ENTRYPHASE = 0x27c
SEQ_NUMAUTOLAYERS = 0x294
SEQ_STATREQUIRED = 0x2b8
SEQ_NUMHITVOLUMES = 0x2bc
SEQ_NUMKNOCKBACKS = 0x2c4
SEQ_MELEERANGE = 0x2cc
SEQ_SEQSELECTMASK = 0x2d4
SEQ_SZDODGE = 0x2dc
SEQ_SZBLOCK = 0x2e4
SEQ_SZNAME2E8 = 0x2e8
SEQ_SZNAME2EC = 0x2ec
SEQ_CYCLEWINDOW = 0x2f0

HITVOLUME_STRIDE = 24
KNOCKBACK_STRIDE = 188

MOVEMENT_STRIDE = 44

ANIM_STRIDE = 32
NUM_CHANNELS = 7

TEXTURE_STRIDE = 20
BODYPART_STRIDE = 16
MODEL_STRIDE = 224
MESH_STRIDE = 60
EYEBALL_STRIDE = 140
MOUTH_STRIDE = 20
MESH_NUMFLEXES = 0x10

FLEX_STRIDE = 32
# mstudioflex_t.vertanimtype at +0x1c picks the payload record: 1 -> 8 bytes, 0 -> 20.
VERTANIM_STRIDE = {0: 20, 1: 8}
# g_flexPositionDeltaScale, StudioRender.dll 0x2c06c4fc, read only at 0x2c050bc7. The 8-byte
# form's position byte is multiplied by it, so 188/255*8 = 5.898 units is all it reaches.
FLEX_POSITION_DELTA_SCALE = 8.0
# FADD ST0,ST0 at 0x2c050a4a and 0x2c050b90 -- the normal term is one term doubled, not two.
FLEX_NORMAL_DELTA_SCALE = 2.0
_VERTANIM0 = struct.Struct("<hh3fHBB")
_VERTANIM1 = struct.Struct("<hHHBB")

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


def flex_delta(v, vertanimtype):
    """One vertanim record as (position delta, normal delta), at flex weight 1.

    `CStudioRender_R_StudioFlexVerts` StudioRender.dll 0x2c050860, the only reader:

        position                 +=      w * (delta_scale/255) * 8.0 * table[delta_offset]
        normal and tangentS.xyz  +=  2 * w * (ndelta_scale/255)     * table[ndelta_offset]

    with the 20-byte form carrying the position delta as raw floats instead. Returns None
    for a direction the table does not hold -- an offset that is not a multiple of 12, or
    one past the last entry. Nothing in the binary bounds either ushort.
    """
    nd = NT.decode(1, v.ndelta_offset)
    if nd is None:
        return None
    ns = FLEX_NORMAL_DELTA_SCALE * v.ndelta_scale / 255.0
    ndelta = (nd[0] * ns, nd[1] * ns, nd[2] * ns)
    if vertanimtype == 0:
        return v.delta, ndelta
    pd = NT.decode(1, v.delta_offset)
    if pd is None:
        return None
    ps = v.delta_scale / 255.0 * FLEX_POSITION_DELTA_SCALE
    return (pd[0] * ps, pd[1] * ps, pd[2] * ps), ndelta


def flex_pack(v, vertanimtype):
    """A VertAnim back to its record bytes. The inverse of `Mdl._read_vertanim`."""
    if vertanimtype:
        return _VERTANIM1.pack(v.index, v.delta_offset, v.ndelta_offset,
                               v.delta_scale, v.ndelta_scale)
    return _VERTANIM0.pack(v.index, 0, v.delta[0], v.delta[1], v.delta[2],
                           v.ndelta_offset, v.ndelta_scale, 0)


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
                 "bbmin", "bbmax", "events", "paramindex", "paramstart", "paramend",
                 "autolayers", "hitvolumes", "knockbacks", "entrynode", "exitnode",
                 "nodeflags", "entryphase", "exitphase", "dodge", "block", "name2e8",
                 "name2ec", "statrequired", "seqselectmask", "meleerange", "cyclewindow")


class HitVolume:
    """One of `numhitvolumes` axis-aligned boxes a melee sequence sweeps.

    Read as AABB-vs-AABB overlap against the target's world box by
    `CBaseCombatCharacter::ChooseMeleeAttackSequence`, vampire.dll 10347180.  About 14%
    of the shipped records are axis-inverted and can never pass that test.
    """
    __slots__ = ("bbmin", "bbmax")


class Knockback:
    """One `mstudioknockback_t`, 188 bytes off `mstudioseqdesc_t` +0x2c4/+0x2c8.

    `activities` is the 4x4 grid of activity-name strings at +0x78, each row cut to the
    count at +0x28 + row*4, matching what `Studio_ResolveSequenceActivities_vtmb` walks.
    `raw` is the whole record, because most of it has no scene representation and has to
    go back out verbatim.
    """
    __slots__ = ("index", "bone", "cycleend", "counts", "activities", "raw")


class PoseParam:
    """One mstudioposeparamdesc_t -- a runtime input a sequence blends along.

    `move_yaw` and `hit_yaw` on all 451 corpus models that carry any, plus `aim_yaw`
    and `aim_pitch` on the two that carry four. `loop` is 0 for none, 360 for a
    rotation.
    """
    __slots__ = ("index", "name", "flags", "start", "end", "loop")


class Event:
    """One mstudioevent_t. `cycle` is 0..1 along the sequence, not a frame.

    `options` is a 64-byte char array read to its first NUL. Over the 4445-model corpus
    no record carries a byte after that NUL and none fills all 64 without one, so the
    string is the whole field and re-encoding it loses nothing.
    """
    __slots__ = ("cycle", "event", "type", "options")


class Mesh:
    __slots__ = ("index", "material", "numvertices", "vertexoffset", "materialtype",
                 "materialparam", "flexes", "cloth", "clothbind")


class Flex:
    """One morph target's contribution to one mesh. `flexdesc` names it globally."""
    __slots__ = ("index", "flexdesc", "name", "target", "vertanimtype", "verts")


class VertAnim:
    """One vertex's delta inside a flex. `index` is MESH-LOCAL and signed.

    The direction fields are unscaled byte offsets into `g_normalOffsetTable`, the same
    encoding as `mstudiovertex1_t.norm`, so they go through `normal_table` filetype 1.
    `delta` is the raw Vector of the 20-byte form and None in the 8-byte one.
    """
    __slots__ = ("index", "delta", "delta_offset", "delta_scale",
                 "ndelta_offset", "ndelta_scale")


class Model:
    __slots__ = ("index", "name", "nummeshes", "numvertices", "vertexbase",
                 "tangentbase", "filetype", "quant_offset", "quant_scale", "meshes",
                 "cloths", "clothrows", "clothcols", "eyeballs")


class Eyeball:
    """One mstudioeyeball_t.

    `iris_name` and `glint_name` are the mstudiotexture_t names the two material indices
    land on; a writer resolves them back by name. `lidflexes` is the eight lid indices
    resolved through mstudioflexdesc_t, or None where all eight are 0.

    Over the 602 shipped records not one of the eight is negative and none is partly
    written: all eight are set on 392 records and all eight are 0 on 210. The two written
    families are the right eye -- upperflexdesc (1, 2, 3), lowerflexdesc (5, 6, 7),
    upperlidflexdesc 0, lowerlidflexdesc 4 -- and the left, the same eight plus 8. WHICH
    EYEBALL SLOT CARRIES WHICH SIDE IS NOT FIXED: 158 models put the right eye first and 38
    the left, so the side comes from the resolved names and never from the slot.
    """
    __slots__ = ("index", "name", "bone", "org", "zoffset", "radius", "up", "forward",
                 "texture", "iris_material", "iris_name", "iris_scale", "glint_material",
                 "glint_name", "upperflexdesc", "lowerflexdesc", "uppertarget",
                 "lowertarget", "upperlidflexdesc", "lowerlidflexdesc", "lidflexes",
                 "pitch", "yaw")


class Mouth:
    """The one mstudiomouth_t a model may carry -- nummouths is 0 or 1, never more.

    `name` is the mstudioflexdesc_t the index lands on, which is `mouth` on 200 of 200
    shipped records; the four whose index is 0 are models that list `mouth` first, not
    models that left the field unwritten. `bone` resolves to `Bip01 Head` on 200 of 200 and
    `forward` is (0, -1, 0) on 200 of 200.
    """
    __slots__ = ("bone", "forward", "flexdesc", "name")


class Cloth:
    """One mstudiocloth_t: the 0x5c header's numbers, its springs and its particle map.

    `pv` is the +0x10 block, one MODEL-local vertex index per particle. It inverts the
    meshes' +0x34 exactly: over the corpus's 150 objects, every one of the 25 564 particles
    some vertex names has that vertex in `pv`. The other 2471 particles are named by no
    vertex, 2297 of them pinned, so the pinned set is NOT recoverable as a vertex group.

    Particles are ordered pinned-first and nothing else marks a pin, so particle `p` is
    pinned exactly when `p < numfixed` (todo-vtmb-cloth-solver.md section 1.1).

    A spring is `(a, b, w0, w1, rest2)`. `sigma` is `(w0 - w1) / 2` and the group-1 slack is
    `rest2` over the squared rest separation -- `cloth.recover` does both.
    """
    __slots__ = ("slot", "row", "col", "scale", "numparticles", "numfixed", "numfree",
                 "numsprings", "ns0", "ns1", "pv", "springs")


class BodyPart:
    __slots__ = ("index", "name", "base", "models")


class SpringBone:
    __slots__ = ("index", "bone", "endbone", "disabled", "unk08", "gravity",
                 "damping", "springexp", "maxangledeg")


class HitboxSet:
    __slots__ = ("index", "name", "boxes")


class Hitbox:
    __slots__ = ("index", "bone", "group", "bbmin", "bbmax")


class Attachment:
    __slots__ = ("index", "name", "type", "bone", "local")


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
        self._read_poseparams()
        self._read_materials()
        self._read_flexdescs()
        self._read_flexcontrollers()
        self._read_flexrules()
        self._read_bodyparts()
        self._read_mouth()
        self._read_hitboxsets()
        self._read_attachments()
        self._read_includes()
        self._read_springbones()

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
        # `> 0` the way the four string fields below are read: a zero word is what a
        # writer leaves for "no activity", and adding it to `off` would read the record's
        # own sznameindex as a string. No shipped sequence carries one -- 0 of 14012 --
        # 2433 of them instead pointing at an empty string, which is the file's own idiom
        # and what a writer reproduces.
        v = struct.unpack_from("<i", d, off + SEQ_ACTIVITY)[0]
        s.activity = self._cstr(off + v) if v > 0 else ""
        # STUDIO_LOOPING is bit 0 of THIS field, not of mstudioanimdesc_t.flags at the
        # same offset of that struct: a file carrying the animdesc bit and not this one
        # plays once and stops -- measured in game 2026-08-22 on ztest/zskel_loop.mdl.
        s.flags = struct.unpack_from("<i", d, off + SEQ_FLAGS)[0]
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
        s.events = self._read_events(off)
        # Which pose parameter drives each blend axis, -1 for none -- which is what
        # 13724 of the 14012 shipped sequences carry and the only value
        # Studio_LocalPoseParameter short-circuits on.
        s.paramindex = list(struct.unpack_from("<2i", d, off + SEQ_PARAMINDEX))
        s.paramstart = list(struct.unpack_from("<2f", d, off + SEQ_PARAMSTART))
        s.paramend = list(struct.unpack_from("<2f", d, off + SEQ_PARAMEND))
        s.entrynode, s.exitnode, s.nodeflags = struct.unpack_from("<3i", d, off + SEQ_ENTRYNODE)
        s.entryphase, s.exitphase = struct.unpack_from("<2f", d, off + SEQ_ENTRYPHASE)
        s.statrequired = struct.unpack_from("<i", d, off + SEQ_STATREQUIRED)[0]
        s.seqselectmask = struct.unpack_from("<i", d, off + SEQ_SEQSELECTMASK)[0]
        s.meleerange = list(struct.unpack_from("<2f", d, off + SEQ_MELEERANGE))
        s.cyclewindow = list(struct.unpack_from("<3f", d, off + SEQ_CYCLEWINDOW))
        # `> 0`, not `!= -1`: the engine guards with `if (-1 < value)`, so 0 passes there
        # and dereferences the record's own first byte.
        for name, at in (("dodge", SEQ_SZDODGE), ("block", SEQ_SZBLOCK),
                         ("name2e8", SEQ_SZNAME2E8), ("name2ec", SEQ_SZNAME2EC)):
            v = struct.unpack_from("<i", d, off + at)[0]
            setattr(s, name, self._cstr(off + v) if v > 0 else None)
        s.autolayers = self._read_autolayers(off)
        s.hitvolumes = self._read_hitvolumes(off)
        s.knockbacks = self._read_knockbacks(off)
        return s

    def _read_autolayers(self, off):
        """`int[numautolayers]`, each a sequence index this one drags along."""
        d = self.d
        n, idx = struct.unpack_from("<2i", d, off + SEQ_NUMAUTOLAYERS)
        if not 0 < n < 64 or off + idx + n * 4 > len(d):
            return []
        return list(struct.unpack_from("<%di" % n, d, off + idx))

    def _read_hitvolumes(self, off):
        """Six floats per record, bbmin then bbmax.

        Seven shipped sequences hold studiomdl's 768-byte write cursor in both the count
        and the index and decode to denormals; all seven carry `numknockbacks` 0, which is
        the same test vampire.dll 103ea982 puts in front of the only reader that is
        bounded by it.
        """
        d = self.d
        if struct.unpack_from("<i", d, off + SEQ_NUMKNOCKBACKS)[0] <= 0:
            return []
        n, idx = struct.unpack_from("<2i", d, off + SEQ_NUMHITVOLUMES)
        if n <= 0 or off + idx + n * HITVOLUME_STRIDE > len(d):
            return []
        out = []
        for k in range(n):
            o = off + idx + k * HITVOLUME_STRIDE
            hv = HitVolume()
            hv.bbmin = struct.unpack_from("<3f", d, o)
            hv.bbmax = struct.unpack_from("<3f", d, o + 12)
            out.append(hv)
        return out

    def _read_knockbacks(self, off):
        d = self.d
        n, idx = struct.unpack_from("<2i", d, off + SEQ_NUMKNOCKBACKS)
        if n <= 0 or off + idx + n * KNOCKBACK_STRIDE > len(d):
            return []
        out = []
        for k in range(n):
            o = off + idx + k * KNOCKBACK_STRIDE
            kb = Knockback()
            kb.index = k
            kb.bone = struct.unpack_from("<i", d, o + 0x08)[0]
            kb.cycleend = struct.unpack_from("<f", d, o + 0x04)[0]
            kb.counts = list(struct.unpack_from("<4i", d, o + 0x28))
            kb.activities = []
            for g in range(4):
                row = []
                for t in range(4):
                    v = struct.unpack_from("<i", d, o + 0x78 + (g * 4 + t) * 4)[0]
                    row.append(self._cstr(o + v) if t < min(abs(kb.counts[g]), 4) else None)
                kb.activities.append(row)
            kb.raw = bytes(d[o:o + KNOCKBACK_STRIDE])
            out.append(kb)
        return out

    def _read_poseparams(self):
        """studiohdr_t's mstudioposeparamdesc_t array, empty on 3994 of 4445 models.

        A sequence names one of these by index, and Studio_LocalPoseParameter indexes
        the array with that number and no comparison against numposeparameters, in all
        three modules that carry the function -- so an out-of-range index is a read
        past the array and nothing in the engine stops it.
        """
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMPOSEPARAMS)
        self.poseparams = []
        for i in range(max(0, n)):
            o = idx + i * POSEPARAM_STRIDE
            if o + POSEPARAM_STRIDE > len(d):
                break
            pp = PoseParam()
            pp.index = i
            pp.name = self._cstr(o + struct.unpack_from("<i", d, o)[0])
            pp.flags = struct.unpack_from("<i", d, o + 4)[0]
            pp.start, pp.end, pp.loop = struct.unpack_from("<3f", d, o + 8)
            self.poseparams.append(pp)

    def _read_events(self, off):
        """The sequence's mstudioevent_t array, empty on the 12862 sequences with none.

        eventindex is relative to the seqdesc, like every other pointer in the record.
        """
        d = self.d
        n, ei = struct.unpack_from("<2i", d, off + SEQ_NUMEVENTS)
        out = []
        for j in range(max(0, n)):
            o = off + ei + j * EVENT_STRIDE
            if o + EVENT_STRIDE > len(d):
                break
            e = Event()
            e.cycle, e.event, e.type = struct.unpack_from("<fii", d, o)
            opt = d[o + EVENT_OPTIONS:o + EVENT_OPTIONS + EVENT_OPTIONS_LEN]
            z = opt.find(bytes([0]))
            e.options = opt[:z if z >= 0 else len(opt)].decode("latin1")
            out.append(e)
        return out

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
            # +0x30/+0x34/+0x38 bind this mesh's vertices to a cloth object -- owner column,
            # particle index, cloth normal index.  All three are 0 on a mesh without cloth,
            # and the .vtx strip group of one that has them must carry SG_IS_CLOTH.
            e.cloth = all(struct.unpack_from("<3i", d, eo + 0x30))
            e.clothbind = None
            e.flexes = self._read_flexes(eo)
            m.meshes.append(e)
        self._read_cloths(off, m, meshindex)
        self._read_eyeballs(off, m)
        return m

    def _read_eyeballs(self, off, m):
        """Every mstudioeyeball_t of one model, materials and lid flexes resolved by name."""
        d = self.d
        m.eyeballs = []
        n, idx = struct.unpack_from("<2i", d, off + 0xc0)
        for k in range(n):
            eo = off + idx + k * EYEBALL_STRIDE
            e = Eyeball()
            e.index = k
            sz = struct.unpack_from("<i", d, eo)[0]
            e.name = self._cstr(eo + sz) if sz else ""
            e.bone = struct.unpack_from("<i", d, eo + 0x04)[0]
            e.org = struct.unpack_from("<3f", d, eo + 0x08)
            e.zoffset, e.radius = struct.unpack_from("<2f", d, eo + 0x14)
            e.up = struct.unpack_from("<3f", d, eo + 0x1c)
            e.forward = struct.unpack_from("<3f", d, eo + 0x28)
            e.texture, e.iris_material = struct.unpack_from("<2i", d, eo + 0x34)
            e.iris_scale = struct.unpack_from("<f", d, eo + 0x3c)[0]
            e.glint_material = struct.unpack_from("<i", d, eo + 0x40)[0]
            e.iris_name = self._texname(e.iris_material)
            e.glint_name = self._texname(e.glint_material)
            e.upperflexdesc = struct.unpack_from("<3i", d, eo + 0x44)
            e.lowerflexdesc = struct.unpack_from("<3i", d, eo + 0x50)
            e.uppertarget = struct.unpack_from("<3f", d, eo + 0x5c)
            e.lowertarget = struct.unpack_from("<3f", d, eo + 0x68)
            e.upperlidflexdesc, e.lowerlidflexdesc =                 struct.unpack_from("<2i", d, eo + 0x74)
            e.pitch = struct.unpack_from("<2f", d, eo + 0x7c)
            e.yaw = struct.unpack_from("<2f", d, eo + 0x84)
            ids = (list(e.upperflexdesc) + list(e.lowerflexdesc)
                   + [e.upperlidflexdesc, e.lowerlidflexdesc])
            # All eight zero is the only way a shipped record says "no lid flexes"; the
            # right eye's upperlidflexdesc is a genuine 0, so a lone 0 is not absence.
            e.lidflexes = None if not any(ids) else tuple(
                self.flexdescs[i] if 0 <= i < len(self.flexdescs) else None for i in ids)
            m.eyeballs.append(e)

    def _texname(self, i):
        return self.materials[i] if 0 <= i < len(self.materials) else None

    def _read_mouth(self):
        """The one mstudiomouth_t, or None. nummouths is 0 or 1 over the whole corpus."""
        d = self.d
        self.mouth = None
        n, idx = struct.unpack_from("<2i", d, HDR_NUMMOUTHS)
        if n <= 0:
            return
        w = Mouth()
        w.bone = struct.unpack_from("<i", d, idx)[0]
        w.forward = struct.unpack_from("<3f", d, idx + 4)
        w.flexdesc = struct.unpack_from("<i", d, idx + 0x10)[0]
        w.name = (self.flexdescs[w.flexdesc]
                  if 0 <= w.flexdesc < len(self.flexdescs) else None)
        self.mouth = w

    def _read_cloths(self, off, m, meshindex):
        """Every mstudiocloth_t of one model, and the per-mesh binding that names them.

        `mstudiomodel_t+0xcc` is a rows x cols table of MODEL-relative offsets and +0xc8 is
        cols. The table stores no length, so the first object it points at bounds it and the
        row count falls out of that (anomalies section B10). A slot is `row * cols + col`,
        where `col` is the owner byte a mesh's +0x30 array holds -- 150 of 150 shipped
        objects agree, read the other way round 60 do.

        A mesh's three arrays are `rows * numvertices` entries laid out row-major, and the
        0x8000 bit on a +0x34 entry is per VERTEX and not per mesh: 30 361 entries carry it
        against 25 367 that do not.
        """
        d = self.d
        m.cloths, m.clothrows, m.clothcols = [], 0, 0
        cols, tbloff = struct.unpack_from("<2i", d, off + 0xc8)
        if cols <= 0 or tbloff <= 0:
            return
        tbl, offs, k = off + tbloff, [], 0
        while True:
            v = struct.unpack_from("<i", d, tbl + k * 4)[0]
            if v and tbl < off + v < len(d) - 0x5c:
                offs.append((k, off + v))
            k += 1
            if k > 512 or (offs and tbl + k * 4 >= min(a for _k, a in offs)):
                break
        if not offs:
            return
        rows = ((min(a for _k, a in offs) - tbl) // 4) // cols
        m.clothrows, m.clothcols = rows, cols
        for k, at in offs:
            c = Cloth()
            c.slot, c.row, c.col = k, k // cols, k % cols
            c.scale = struct.unpack_from("<f", d, at)[0]
            c.numparticles, c.numfixed, c.numfree, pvoff =                 struct.unpack_from("<4i", d, at + 0x04)
            c.numsprings, c.ns0, c.ns1, spoff = struct.unpack_from("<4i", d, at + 0x14)
            c.pv = list(struct.unpack_from("<%dH" % c.numparticles, d, at + pvoff))                 if pvoff else []
            c.springs = [struct.unpack_from("<2H3f", d, at + spoff + q * 16)
                         for q in range(c.ns0 + c.ns1)] if spoff else []
            m.cloths.append(c)
        for k, e in enumerate(m.meshes):
            eo = off + meshindex + k * MESH_STRIDE
            own, par = struct.unpack_from("<2i", d, eo + 0x30)
            if not own or not par:
                continue
            n = e.numvertices
            e.clothbind = []
            for r in range(rows):
                row = {}
                for v in range(n):
                    col = d[eo + own + r * n + v]
                    if col == 0xff:
                        continue
                    raw = struct.unpack_from("<H", d, eo + par + (r * n + v) * 2)[0]
                    row[v] = (col, raw & 0x7fff, bool(raw & 0x8000))
                e.clothbind.append(row)

    def _read_flexdescs(self):
        n, idx = struct.unpack_from("<ii", self.d, HDR_NUMFLEXDESC)
        self.flexdescs = [self._cstr(idx + i * 4
                                     + struct.unpack_from("<i", self.d, idx + i * 4)[0])
                          for i in range(n)]

    def _read_flexcontrollers(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMFLEXCONTROLLERS)
        self.flexcontrollers = []
        for i in range(n):
            o = idx + i * FLEXCTRL_STRIDE
            c = FlexController()
            c.index = i
            # Both string offsets are relative to the START of the record, which
            # client.dll 100c435b reads that way for sznameindex.  A field-relative read
            # shifts every name four characters and still decodes.
            c.type = self._cstr(o + struct.unpack_from("<i", d, o)[0])
            c.name = self._cstr(o + struct.unpack_from("<i", d, o + 4)[0])
            c.link, c.min, c.max = struct.unpack_from("<iff", d, o + 8)
            self.flexcontrollers.append(c)

    def _read_flexrules(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMFLEXRULES)
        self.flexrules = []
        for i in range(n):
            o = idx + i * FLEXRULE_STRIDE
            r = FlexRule()
            r.index = i
            r.flex, numops, opindex = struct.unpack_from("<3i", d, o)
            r.name = (self.flexdescs[r.flex]
                      if 0 <= r.flex < len(self.flexdescs) else None)
            r.ops = [struct.unpack_from("<2i", d, o + opindex + k * FLEXOP_STRIDE)
                     for k in range(numops)]
            r.expr = flex_expr(r.ops, self.flexcontrollers, self.flexdescs)
            self.flexrules.append(r)

    def _read_flexes(self, mesh_off):
        """Every mstudioflex_t of one mesh, payload decoded."""
        d = self.d
        n, idx = struct.unpack_from("<ii", d, mesh_off + MESH_NUMFLEXES)
        out = []
        for r in range(n):
            off = mesh_off + idx + r * FLEX_STRIDE
            f = Flex()
            f.index = r
            f.flexdesc = struct.unpack_from("<i", d, off)[0]
            f.name = (self.flexdescs[f.flexdesc]
                      if 0 <= f.flexdesc < len(self.flexdescs) else None)
            f.target = struct.unpack_from("<4f", d, off + 4)
            numverts, vertindex = struct.unpack_from("<ii", d, off + 0x14)
            # The reader loops on the low short of both numverts and vertanimtype, so the
            # high half of each is unreachable and this reads what the engine reads.
            numverts &= 0xFFFF
            f.vertanimtype = 1 if (struct.unpack_from("<i", d, off + 0x1c)[0] & 0xFFFF) else 0
            base = off + vertindex
            f.verts = [self._read_vertanim(base, k, f.vertanimtype)
                       for k in range(numverts)]
            out.append(f)
        return out

    def _read_vertanim(self, base, k, vertanimtype):
        v = VertAnim()
        if vertanimtype:
            at = base + k * 8
            (v.index, v.delta_offset, v.ndelta_offset,
             v.delta_scale, v.ndelta_scale) = _VERTANIM1.unpack_from(self.d, at)
            v.delta = None
        else:
            at = base + k * 20
            (v.index, _pad, dx, dy, dz, v.ndelta_offset,
             v.ndelta_scale, _pad2) = _VERTANIM0.unpack_from(self.d, at)
            v.delta = (dx, dy, dz)
            v.delta_offset = v.delta_scale = None
        return v

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

    def _read_attachments(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMATTACHMENTS)
        self.attachments = []
        for i in range(n):
            off = idx + i * ATTACHMENT_STRIDE
            a = Attachment()
            a.index = i
            a.name = self._cstr(off + struct.unpack_from("<i", d, off)[0])
            a.type, a.bone = struct.unpack_from("<ii", d, off + 4)
            flat = struct.unpack_from("<12f", d, off + 12)
            a.local = [list(flat[0:4]), list(flat[4:8]), list(flat[8:12])]
            self.attachments.append(a)

    def _read_includes(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMINCLUDEMODELS)
        self.includes = []
        for i in range(n):
            off = idx + i * INCLUDE_STRIDE
            self.includes.append(
                self._cstr(off + struct.unpack_from("<i", d, off)[0]))

    def _read_springbones(self):
        d = self.d
        n, idx = struct.unpack_from("<ii", d, HDR_NUMSPRINGBONES)
        self.springbones = []
        for i in range(n):
            off = idx + i * SPRINGBONE_STRIDE
            s = SpringBone()
            s.index = i
            bone, s.endbone = struct.unpack_from("<ii", d, off)
            # A negative start bone carries the index as -1-bone and separately makes
            # CBaseAnimating::SetModel set m_nPhysicsChainDisableMask, so the chain also
            # starts switched off and LookupPhysicsChain can never match it by name.
            # 0 of the corpus's 600 records use it.
            s.disabled = bone < 0
            s.bone = -1 - bone if bone < 0 else bone
            (s.unk08, s.gravity, s.damping, s.springexp,
             s.maxangledeg) = struct.unpack_from("<5f", d, off + 8)
            self.springbones.append(s)

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
