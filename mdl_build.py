"""Author a v2531 .mdl from a description of its records.  No bpy.

The rebuilder (`mdl_rebuild.py`) moves spans of a donor file: it can relocate anything it
can attribute, but it cannot add a bone, drop a sequence or change a vertex count, because
every count in the output is the count the donor already had.  This takes a list of records
per array instead, sizes each array from `len()`, and derives every count and every offset
from the layout it chose.  A description assembled from nothing emits a file that had no
donor.

Each record carries its scalar bytes with every pointer zeroed, plus its decoded children.
Authoring from nothing means supplying `DEFAULTS` for the scalars nothing has named;
re-authoring a shipped model means supplying that model's own.  The two paths differ in
where the scalars come from and nowhere else, which is what lets the corpus check a writer
whose real input is a Blender scene.

Sections whose internal structure is unimplemented are *carried*, not dropped: cloth moves
as one rigid region with its pointers recomputed (`_cloth_region`), and a cloth-bound mesh
whose vertex count changed has its three per-vertex arrays rebuilt for that count
(`_regrow_cloth`).  `_drop_stale_cloth` is the backstop for a count that moved without
saying so.  `Desc.dropped` counts everything not carried, and `emit` refuses unless the
caller passes `drop=True`; over `gamedata\\models` nothing populates it, all 3263
procedural bones being proctype 1.

`replace_model` is the third entry point, between the two: it rewrites one existing model's
geometry mesh by mesh and keeps everything else those records hold, including per vertex the
donor's own `bonecountcode` high bits and tangent (`_carry_vertex_fields`, `_skin_key`),
which nothing in a scene supplies.  It is what an exporter calls for a mesh whose vertex
count or UV seams moved.
"""

import os
import struct
import sys

try:
    from . import mdl as M
    from . import mdl_write as W
    from . import normal_table as NT
    from . import relocs as R
    from . import sections as S
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mdl as M
    import mdl_write as W
    import normal_table as NT
    import relocs as R
    import sections as S

HDR_SIZE = 424
ALIGN = 4

# 20000, not the SDK header's 25000: CCachedRenderData's four arrays are MAXSTUDIOVERTS+1
# and all four spans agree on 20001.  SetupComputation accumulates mstudiomesh_t.numvertices
# between two StartModel calls, so the cap is per mstudiomodel_t, and a model past it passes
# every offline check and then draws nothing.  Recon §35.1.
MAXSTUDIOVERTS = 20000

# 350, not the SDK header's 1024 nor Valve's 128.  Nothing in the game validates numbones
# against it, so an over-cap file indexes client.dll's per-bone arrays out of range rather than
# being refused.  Corpus max is 343.  Measured four ways in recon §45.3.
MAXSTUDIOBONES = 350

# 20, and in no header at all.  All three vampire.dll melee walkers clamp the count
# themselves -- `if (numknockbacks >= 0x14) numknockbacks = 0x14` in
# CBaseCombatCharacter__MeleeSwingStep_vtmb 0x10343020, ..IsMeleeSwingActive_vtmb 0x103451d0
# and ..IsMeleeSwingOver_vtmb 0x10345330 -- and the two per-record entity arrays behind them
# are sized for 20: six floats at entity[i*6 + 0x2b0] and five dwords at entity[i*5 + 0x32b].
# Record 21 and up are written, pass every offline reader, and never run.  Corpus maximum is
# 17 over 1587 records, so nothing shipped reaches it.  Recon §54.1.
MAXSTUDIOKNOCKBACKS = 20

# A bone carrying no bit of the 0xfffc used-by mask gets no bone matrix, so anything skinned
# to it draws nothing.  0x10 is the corpus norm: 62439 of 62702 bones carry it.
BONE_USED = 0x10

# Every record ships as this, on all 32742 blocks of all 311 corpus models that carry an
# include -- one distinct value, no exceptions.  Nothing in the file reads it and the engine
# fills it at load, but the span has to be there: removing it crashed from a site with both
# __strcmpi operands wild.  Anomalies §L.
BONEMAP_RECORD = struct.pack("<ii", 0x0000FFFF, -1) + b"\0" * 48

# 24, not Valve's 56: Studio_FindBoneControllerByInput (client.dll:0x1008b360) matches on
# +0x14 and advances by 0x18, and its unrolled lookahead reads [EAX+0x2c] == 0x18 + 0x14,
# which closes at no other stride.  Zero of the 4464 corpus models carry a record, so only a
# writer is affected.
BONECONTROLLER_STRIDE = 24

# The scalars nothing has named, as the corpus states them.  A from-scratch caller starts
# from these; a caller re-authoring a shipped model overwrites them with that model's own.
DEFAULTS = {
    "hdr.unk144": (0.5, 0.5, 0.5),      # on 4438 of 4445 models
    # Lipsync blend-width clamp in seconds, @232 min and @236 max.  client.dll 0x100c3be0
    # clamps a phoneme's own duration into this pair whenever either phonemefilter ConVar
    # reads 0.0, so zero here shuts the window rather than meaning unset.  Four values over
    # the corpus -- (0,0) x3787, this x525, (0.08,0.10) x150, (0.08,0.105) x2.
    "hdr.phonemefilter": (0.065, 0.100),
    "hdr.unhz": (0, 0, 1),              # ints, not floats; on 4438 of 4445 models
    "seqgroup": ("default", ""),        # on every model
}


class Refused(Exception):
    """The description holds something this cannot emit.  Never degrade to a copy."""



class Out(object):
    """Chunks in emission order, plus the fixups whose values layout decides.

    A target is `(key, delta)` rather than a bare key so a fixup can name a record inside an
    array without every record needing a chunk of its own.
    """

    def __init__(self):
        self.keys = []
        self.data = {}
        self.align = {}
        self.fix = []
        self.pool = bytearray()
        self.poolat = {}
        self.pos = {}

    def add(self, key, data, align=ALIGN):
        if key in self.data:
            raise Refused("duplicate chunk %r" % key)
        self.keys.append(key)
        self.data[key] = bytearray(data)
        self.align[key] = align
        return key

    def patch(self, at, target, base=None):
        """at = (key, off).  Writes (target - base), or target itself when base is None."""
        self.fix.append((at, target, base))

    def string(self, text):
        """(key, delta) for one interned C string.  The pool is emitted last, which is also
        what satisfies relocs.POSITIVE_ONLY: those four seqdesc fields may not go negative,
        so their strings have to land after the seqdesc, and everything does."""
        if text not in self.poolat:
            self.poolat[text] = len(self.pool)
            self.pool += text.encode("latin1") + b"\0"
        return ("strings", self.poolat[text])

    def build(self):
        self.add("strings", self.pool, 1)
        at = 0
        for k in self.keys:
            n = self.align[k]
            at += (-at) % n
            self.pos[k] = at
            at += len(self.data[k])
        buf = bytearray(at)
        for k in self.keys:
            p = self.pos[k]
            buf[p:p + len(self.data[k])] = self.data[k]
        for (ak, ao), (tk, td), base in self.fix:
            v = self.pos[tk] + td
            if base is not None:
                v -= self.pos[base[0]] + base[1]
            struct.pack_into("<i", buf, self.pos[ak] + ao, v)
        return bytes(buf)



class Rec(object):
    """Scalar bytes with every pointer field zeroed, plus whatever hangs off them."""

    __slots__ = ("raw", "name", "kids", "extra")

    def __init__(self, raw, name=None, kids=None, extra=None):
        self.raw = bytearray(raw)
        self.name = name
        self.kids = kids if kids is not None else []
        self.extra = extra if extra is not None else {}


class Desc(object):
    def __init__(self):
        self.hdr = bytearray(HDR_SIZE)
        self.name = ""
        self.checksum = 0
        self.surfaceprop = None
        self.bones = []
        self.bonecontrollers = []
        self.hitboxsets = []
        self.attachments = []
        self.includes = []
        # `emit` re-sweeps every sequence box rather than only the zero ones. Off by
        # default so a file read out of bytes keeps the boxes it shipped.
        self.refit_boxes = False
        self.refit_count = 0
        self.anims = []
        self.seqs = []
        self.seqgroups = []
        self.flexdescs = []
        self.flexcontrollers = []
        self.flexrules = []
        self.ikchains = []
        self.mouths = []
        self.poseparams = []
        self.springbones = []
        self.bodyparts = []
        self.textures = []
        self.cdtextures = []
        self.skin = []
        # The .mdl holds no triangle list; this is carried so a scratch model can emit the
        # .vtx that does.
        self.faces = []
        self.dropped = {}

    def drop(self, what, n=1):
        if n:
            self.dropped[what] = self.dropped.get(what, 0) + n


def _cloth_region(b, mo):
    """One model's cloth as a single blob plus the offsets into it to re-aim at, or None.

    Rigid by construction: the table stores no count, so its length is how far the first
    object sits past it and that divided by `cols` is the row count every per-mesh array is
    sized by, and each object's nine payload offsets are object-relative (anomalies §B10,
    §M).  Carried whole -- holes and alignment pads included -- because nothing inside may
    move relative to anything else.
    """
    m = S.Map(b)
    tbl, rows, objs = S.cloth_objects(m, mo)
    if not objs:
        return None
    # The table's own entries are model-relative, so they are the one thing inside the
    # region that a move invalidates. Recorded per slot and recomputed at layout.
    nslot = (min(objs) - tbl) // 4
    slots = [(k, mo + m.i(tbl + k * 4)) for k in range(nslot) if m.i(tbl + k * 4)]
    lo, hi = tbl, tbl + nslot * 4
    for at in objs:
        lo, hi = min(lo, at), max(hi, at + 0x5c)
        h = {o: m.i(at + o) for o in range(0, 0x5c, 4)}
        for off, cnts, w, _what in S.CLOTH_BLOCKS:
            if h[off]:
                lo = min(lo, at + h[off])
                hi = max(hi, at + h[off] + sum(h[c] for c in cnts) * w)
    binds = {}
    mesh = mo + m.i(mo + 0x8c)
    for k in range(m.i(mo + 0x88)):
        s = mesh + k * 60
        if not m.i(s + 0x30):
            continue
        nv = m.i(s + 8) * rows
        spans = []
        for f, n in ((0x30, (nv + 3) // 4 * 4), (0x34, nv * 2), (0x38, nv * 2)):
            at = s + m.i(s + f)
            lo, hi = min(lo, at), max(hi, at + n)
            spans.append(at)
        binds[k] = (spans, m.i(s + 8))
    if lo % ALIGN:
        raise Refused("cloth region starts at %d, not %d-aligned" % (lo, ALIGN))
    return {"data": bytes(b[lo:hi]), "cols": m.i(mo + 0xc8), "table": tbl - lo,
            "rows": rows,
            "slots": [(k, at - lo) for k, at in slots],
            "meshes": dict((k, ([x - lo for x in v], n)) for k, (v, n) in binds.items())}


def _drop_stale_cloth(d):
    """Cloth on a mesh whose vertex count moved is dropped, and the drop is reported.

    `mstudiomesh_t+0x30/+0x34/+0x38` are per-vertex arrays of `rows * numvertices` entries
    and the engine reads that many whatever the array's real length is, so a carried array
    against a changed count is read past its end -- anomalies §M, where a mis-sized region
    faulted in `Cloth_BuildSpringBatches_vtmb` on render rather than on load.

    This is the backstop, not the answer: `_regrow_cloth` rebuilds the arrays for the new
    count and `replace_model` calls it, so a path that reaches here changed a count without
    saying so.
    """
    for bp in d.bodyparts:
        for mr in bp.kids:
            cl = mr.extra.get("cloth")
            if not cl:
                continue
            for k, (_spans, was) in cl["meshes"].items():
                if k < len(mr.kids) and \
                        struct.unpack_from("<i", mr.kids[k].raw, 8)[0] == was:
                    continue
                mr.extra["cloth"] = None
                mr.extra["clothcollide"] = b""
                mr.extra["clothsphere"] = b""
                d.drop("cloth on a mesh whose vertex count changed")
                break


def _regrow_cloth(mr, k, new_n):
    """Mesh `k`'s three per-vertex cloth arrays, rebuilt for a new vertex count.

    Row-major -- row r's slice starts at `numvertices * r`, recon 29.3 -- so a count change
    moves every row and the arrays cannot be carried as they stand. Each row keeps its own
    entries and a vertex the edit added takes the format's own "none": 0xff on +0x30, 0 on
    +0x34 and +0x38. Nothing inside needs remapping, because +0x34 holds a particle index
    and +0x38 a cloth-normal index, neither of them a vertex.

    Grown arrays are appended and the three mesh fields re-aimed. Nothing inside the region
    moves: the table stores no count and every object payload offset is object-relative, so
    only the mesh pointers change, which `_emit_model` patches anyway.
    """
    cl = mr.extra.get("cloth")
    if not cl or k not in cl["meshes"]:
        return False
    spans, old_n = cl["meshes"][k]
    if old_n == new_n:
        return False
    rows, data, out = cl["rows"], bytearray(cl["data"]), []
    for at, width, none in zip(spans, (1, 2, 2), (bytes((0xff,)), bytes(2), bytes(2))):
        buf = bytearray()
        for r in range(rows):
            src = at + old_n * r * width
            buf += data[src:src + min(old_n, new_n) * width]
            buf += none * max(0, new_n - old_n)
        while len(data) % ALIGN:
            data += bytes(1)
        out.append(len(data))
        data += buf
    cl["data"] = bytes(data)
    cl["meshes"][k] = (out, new_n)
    return True


def _sized(b, base, cnt_off, idx_off, stride):
    """A model-relative array carried verbatim: its bytes, or b'' when the count is 0."""
    n, at = struct.unpack_from("<i", b, base + cnt_off)[0], \
        struct.unpack_from("<i", b, base + idx_off)[0]
    if n <= 0 or not at:
        return b""
    return bytes(b[base + at:base + at + n * stride])


def _z(raw, *offs):
    """Zero the pointer fields: a scalar template must carry no stale offset."""
    b = bytearray(raw)
    for o in offs:
        struct.pack_into("<i", b, o, 0)
    return b



def from_bytes(b):
    """A Desc holding everything this can author, and a count of everything it cannot."""
    if b[:4] != b"IDST":
        raise Refused("not IDST")

    def i(o):
        return struct.unpack_from("<i", b, o)[0]

    def s(o):
        if o <= 0 or o >= len(b):
            return None
        e = b.find(b"\0", o)
        return b[o:e if e >= 0 else len(b)].decode("latin1")

    d = Desc()
    d.hdr = bytearray(b[:HDR_SIZE])
    d.checksum = i(8)
    e = b.find(b"\0", 12)
    d.name = b[12:e if 0 <= e < 140 else 140].decode("latin1")
    d.surfaceprop = s(i(392))

    nb, bi = i(240), i(244)
    for k in range(nb):
        o = bi + k * 160
        r = Rec(_z(b[o:o + 160], 0x00, 0x90, 0x98), s(o + i(o)))
        r.extra["surfaceprop"] = s(o + i(o + 0x98)) if i(o + 0x98) else None
        if i(o + 0x8c) == 1:
            at = o + i(o + 0x90)
            r.extra["proc"] = bytes(b[at:at + 176])
        elif i(o + 0x8c):
            # Only proctype 1 has a measured record; another value sizes nothing, so the
            # helper goes and the bone stays.
            struct.pack_into("<i", r.raw, 0x8c, 0)
            d.drop("procedural bone, proctype %d" % i(o + 0x8c))
        d.bones.append(r)

    for k in range(i(248)):
        bc = i(252) + k * BONECONTROLLER_STRIDE
        d.bonecontrollers.append(Rec(b[bc:bc + BONECONTROLLER_STRIDE]))

    for k in range(i(256)):
        o = i(260) + k * 12
        boxes = [Rec(b[o + i(o + 8) + j * 32:o + i(o + 8) + (j + 1) * 32])
                 for j in range(i(o + 4))]
        d.hitboxsets.append(Rec(_z(b[o:o + 12], 0x00, 0x08), s(o + i(o)), boxes))

    for k in range(i(328)):
        o = i(332) + k * 60
        d.attachments.append(Rec(_z(b[o:o + 60], 0x00), s(o + i(o))))

    for k in range(i(404)):
        o = i(408) + k * 116
        d.includes.append(Rec(_z(b[o:o + 116], 0x00, 0x10), s(o + i(o))))

    for k in range(i(264)):
        o = i(268) + k * 72
        r = Rec(_z(b[o:o + 72], 0x00, 0x14, 0x30, 0x38), s(o + i(o)))
        mv = o + i(o + 0x14)
        r.extra["movements"] = [bytes(b[mv + j * 44:mv + (j + 1) * 44])
                                for j in range(i(o + 0x10))]
        nf, blk = i(o + 0x0c), o + i(o + 0x30)
        r.extra["block"] = bytes(b[blk:blk + _block_len(b, blk, nb, nf)]) if nf > 0 else b""
        r.extra["block_bones"] = nb
        if i(o + 0x38):
            # mstudioikrule_t has no measured stride, so the payload cannot be sized and
            # carried. No shipped file reaches this: all 4464 read ikruleindex 0.
            raise Refused("%s: animdesc %d carries %d IK rules and mstudioikrule_t has no "
                          "measured stride" % (d.name, k, i(o + 0x34)))
        d.anims.append(r)

    for k in range(i(272)):
        o = i(276) + k * 764
        r = Rec(_z(b[o:o + 764], 0x000, 0x004, 0x018, 0x298, 0x2c0, 0x2c8,
                   0x2dc, 0x2e4, 0x2e8, 0x2ec), s(o + i(o)))
        r.extra["activity"] = s(o + i(o + 4)) if i(o + 4) else None
        ev = o + i(o + 0x18)
        r.extra["events"] = [bytes(b[ev + j * 76:ev + (j + 1) * 76])
                             for j in range(i(o + 0x14))]
        na = i(o + 0x294)
        r.extra["autolayers"] = ([i(o + i(o + 0x298) + j * 4) for j in range(na)]
                                 if 0 < na < 64 else [])
        # vampire.dll 103ea982 skips the sequence unless numknockbacks >= 1, so the 7 props
        # whose +0x2bc/+0x2c0 both hold studiomdl's 768-byte write cursor carry no table.
        r.extra["hitvolumes"] = _sized(b, o, 0x2bc, 0x2c0, 24) if i(o + 0x2c4) > 0 else b""
        kbs = []
        kb = o + i(o + 0x2c8)
        for j in range(max(0, i(o + 0x2c4))):
            rec = kb + j * 188
            names = []
            for g in range(4):
                for t in range(4):
                    v = i(rec + 0x78 + (g * 4 + t) * 4)
                    names.append(s(rec + v) if t < min(abs(i(rec + 0x28 + g * 4)), 4) else None)
            kbs.append(Rec(_z(b[rec:rec + 188], *[0x78 + t * 4 for t in range(16)]),
                           None, None, {"names": names}))
        r.extra["knockbacks"] = kbs
        for f, nm in ((0x2dc, "dodge"), (0x2e4, "block"),
                      (0x2e8, "seq2e8"), (0x2ec, "seq2ec")):
            r.extra[nm] = s(o + i(o + f)) if i(o + f) > 0 else None
        d.seqs.append(r)

    for k in range(i(284)):
        o = i(288) + k * 16
        d.seqgroups.append(Rec(_z(b[o:o + 16], 0x00, 0x04),
                               s(o + i(o)) or "", None, {"name": s(o + i(o + 4)) or ""}))

    for k in range(i(344)):
        o = i(348) + k * 4
        d.flexdescs.append(Rec(_z(b[o:o + 4], 0x00), s(o + i(o))))
    for k in range(i(352)):
        o = i(356) + k * 20
        # +0x00 is sztypeindex and +0x04 sznameindex, not the other way round.
        d.flexcontrollers.append(Rec(_z(b[o:o + 20], 0x00, 0x04), s(o + i(o + 4)),
                                     None, {"type": s(o + i(o))}))
    for k in range(i(360)):
        o = i(364) + k * 12
        ops = o + i(o + 8)
        d.flexrules.append(Rec(_z(b[o:o + 12], 0x08), None, None,
                               {"ops": bytes(b[ops:ops + i(o + 4) * 8])}))
    for k in range(i(368)):
        o = i(372) + k * 16
        li = o + i(o + 0x0c)
        d.ikchains.append(Rec(_z(b[o:o + 16], 0x00, 0x0c), s(o + i(o)), None,
                              {"links": bytes(b[li:li + i(o + 8) * 28])}))
    for k in range(i(376)):
        d.mouths.append(Rec(b[i(380) + k * 20:i(380) + (k + 1) * 20]))
    for k in range(i(384)):
        o = i(388) + k * 20
        d.poseparams.append(Rec(_z(b[o:o + 20], 0x00), s(o + i(o))))
    for k in range(i(396)):
        d.springbones.append(Rec(b[i(400) + k * 28:i(400) + (k + 1) * 28]))

    for bp in range(i(320)):
        p = i(324) + bp * 16
        models = []
        mi = p + i(p + 0xc)
        for md in range(i(p + 4)):
            mo = mi + md * 224
            nv, nmesh, ft = i(mo + 0x90), i(mo + 0x88), i(mo + 0x9c)
            stride = R.VERTEX_STRIDE.get(ft)
            if nv and stride is None:
                raise Refused("model filetype %d has no vertex stride" % ft)
            meshes = []
            mesh = mo + i(mo + 0x8c)
            for k in range(nmesh):
                so = mesh + k * 60
                flexes = []
                fx = so + i(so + 0x14)
                for r in range(i(so + 0x10)):
                    f = fx + r * 32
                    n, at = i(f + 0x14), f + i(f + 0x18)
                    w = 8 if i(f + 0x1c) else 20
                    flexes.append(Rec(_z(b[f:f + 32], 0x18), None, None,
                                      {"payload": bytes(b[at:at + n * w])}))
                meshes.append(Rec(_z(b[so:so + 60], 0x04, 0x14, 0x30, 0x34, 0x38),
                                  None, flexes))
            eyes = []
            for k in range(i(mo + 0xc0)):
                eo = mo + i(mo + 0xc4) + k * 140
                eyes.append(Rec(b[eo:eo + 140]))
            vi, ti = mo + i(mo + 0x94), mo + i(mo + 0x98)
            r = Rec(_z(b[mo:mo + 224], 0x8c, 0x94, 0x98, 0xc4, 0xcc, 0xd4, 0xdc),
                    None, meshes,
                    {"verts": bytes(b[vi:vi + nv * (stride or 0)]),
                     "tangents": bytes(b[ti:ti + nv * 16]), "eyes": eyes,
                     "cloth": _cloth_region(b, mo),
                     "clothcollide": _sized(b, mo, 0xd0, 0xd4, 36),
                     "clothsphere": _sized(b, mo, 0xd8, 0xdc, 20)})
            models.append(r)
        d.bodyparts.append(Rec(_z(b[p:p + 16], 0x00, 0x0c), s(p + i(p)), models))

    for k in range(i(292)):
        o = i(296) + k * 20
        d.textures.append(Rec(_z(b[o:o + 20], 0x00), s(o + i(o))))
    d.cdtextures = [s(i(i(304) + k * 4)) for k in range(i(300))]
    nref, nfam = i(308), i(312)
    d.skin = [[struct.unpack_from("<h", b, i(316) + (f * nref + r) * 2)[0]
               for r in range(nref)] for f in range(nfam)]
    return d


def _block_len(b, blk, numbones, numframes):
    """How far one animation's mstudioanim_t array plus its RLE streams reaches.

    The array has no length field; its extent is the furthest byte any of its channels
    reaches, which is what studiomdl's own packing makes contiguous.
    """
    end = blk + numbones * 32
    for j in range(numbones):
        e = blk + j * 32
        for c in range(7):
            off = struct.unpack_from("<i", b, e + 4 + c * 4)[0]
            if not off:
                continue
            at = e + off
            n = 0
            while n < numframes:
                valid, total = b[at], b[at + 1]
                at += 2 + valid * 2
                if total <= 0:
                    break
                n += total
            end = max(end, at)
    return end - blk



def model_vertex_counts(d):
    """(bodypart index, model name, numvertices) per mstudiomodel_t, in emission order.

    The count is the one `_emit_model` writes at +0x90, so a guard on it and the field
    cannot disagree.
    """
    out = []
    for i, bp in enumerate(d.bodyparts):
        for mr in bp.kids:
            name = bytes(mr.raw[0:128]).split(b"\0")[0].decode("latin1", "replace")
            out.append((i, name, len(mr.extra.get("tangents") or b"") // 16))
    return out


STUDIOHDR_FLAGS_CLOTH = 0x0400


def _stamp_cloth_flag(d):
    """`studiohdr_t.flags` bit 0x400 is what gets the model a StudioRender instance handle.

    `engine.dll 0x200a7d35` tests it on the header and only then calls `IStudioRender`
    vtable slot 38 to allocate a handle, storing it at `ModelInstance_t +0x0a`.  Without
    one the handle stays 0xffff, and the mesh-draw dispatcher `StudioRender.dll
    0x2c01aaf0` compares exactly that at `0x2c01ab39` and takes the Plain half of
    `g_SoftwareProcessFunc` whatever the `.vtx` cloth bit says -- so the cloth object is
    built and stepped and nothing on screen moves.  Over the 4445-model corpus the bit and
    a cloth object are the same set with no exceptions: 60 of 60 models that bind a mesh to
    cloth carry it, and 4385 of 4385 that do not, leave it clear.

    Derived rather than authored, so a writer cannot forget it.  Cleared as well as set:
    dropping the last cloth region has to clear the bit or the engine hands out a handle
    for a model with nothing to put in it.
    """
    fl = struct.unpack_from("<i", d.hdr, 228)[0]
    has = any(mr.extra.get("cloth") for r in d.bodyparts for mr in r.kids)
    fl = (fl | STUDIOHDR_FLAGS_CLOTH) if has else (fl & ~STUDIOHDR_FLAGS_CLOTH)
    struct.pack_into("<i", d.hdr, 228, fl)


def emit(d, checksum=None, drop=False):
    """Bytes for the description.  Every count comes from a len() and every offset from
    where its target landed, so a description with a record added emits a valid file.

    `drop` is the caller stating it accepts losing whatever `Desc.dropped` names.  Without
    it a description carrying cloth is refused rather than quietly emitted without it.
    """
    _drop_stale_cloth(d)
    if d.dropped and not drop:
        raise Refused("description drops %s"
                      % ", ".join("%s x%d" % (k, v) for k, v in sorted(d.dropped.items())))
    _check_blocks(d)
    if len(d.bones) > MAXSTUDIOBONES:
        raise Refused("armature carries %d bones against MAXSTUDIOBONES %d, so the game "
                      "indexes its per-bone arrays past the end"
                      % (len(d.bones), MAXSTUDIOBONES))
    over_kb = [(k, r) for k, r in enumerate(d.seqs)
               if len(r.extra.get("knockbacks") or []) > MAXSTUDIOKNOCKBACKS]
    if over_kb:
        k, r = over_kb[0]
        raise Refused("sequence %r carries %d knockback records against "
                      "MAXSTUDIOKNOCKBACKS %d, and every melee walk clamps at that, so the "
                      "records past it are written and never run"
                      % (r.name if r.name else k,
                         len(r.extra["knockbacks"]), MAXSTUDIOKNOCKBACKS))
    counts = model_vertex_counts(d)
    over = [c for c in counts if c[2] > MAXSTUDIOVERTS]
    if over:
        raise Refused("model %r carries %d vertices against MAXSTUDIOVERTS %d, so the "
                      "renderer draws nothing%s"
                      % (over[0][1], over[0][2], MAXSTUDIOVERTS,
                         "" if len(counts) == 1 else
                         " (%s)" % ", ".join("%s %d" % (n, c) for _, n, c in counts)))
    d.refit_count = stamp_sequence_boxes(d, force=bool(d.refit_boxes))
    _stamp_cloth_flag(d)
    quantise(d)
    o = Out()
    nb = len(d.bones)
    hdr = bytearray(d.hdr)
    struct.pack_into("<i", hdr, 0, 0x54534449)          # 'IDST'
    struct.pack_into("<i", hdr, 4, M.VERSION)
    hdr[12:140] = d.name.encode("latin1")[:127].ljust(128, b"\0")
    o.add("hdr", hdr)

    def count(off, n):
        struct.pack_into("<i", hdr, off, n)

    def hdrptr(off, key, when):
        """A studiohdr_t index is FILE-relative, and stays 0 when its count is 0."""
        if when:
            o.patch(("hdr", off), (key, 0))

    bones = bytearray()
    for r in d.bones:
        bones += r.raw
    o.add("bones", bones)
    for k, r in enumerate(d.bones):
        base = ("bones", k * 160)
        o.patch(base, o.string(r.name or ""), base)
        sp = r.extra.get("surfaceprop")
        if sp is not None:
            o.patch(("bones", k * 160 + 0x98), o.string(sp), base)
        if r.extra.get("proc"):
            o.add("proc%d" % k, r.extra["proc"])
            o.patch(("bones", k * 160 + 0x90), ("proc%d" % k, 0), base)
    count(240, nb)
    hdrptr(244, "bones", nb)

    _simple(o, hdr, d.bonecontrollers, "bonecontrollers",
            BONECONTROLLER_STRIDE, 248, 252)

    if d.hitboxsets:
        raw = bytearray()
        for r in d.hitboxsets:
            raw += r.raw
        o.add("hitboxsets", raw)
        for k, r in enumerate(d.hitboxsets):
            base = ("hitboxsets", k * 12)
            o.patch(("hitboxsets", k * 12), o.string(r.name or ""), base)
            struct.pack_into("<i", o.data["hitboxsets"], k * 12 + 4, len(r.kids))
            if r.kids:
                boxes = bytearray()
                for x in r.kids:
                    boxes += x.raw
                o.add("hbox%d" % k, boxes)
                o.patch(("hitboxsets", k * 12 + 8), ("hbox%d" % k, 0), base)
    count(256, len(d.hitboxsets))
    hdrptr(260, "hitboxsets", len(d.hitboxsets))

    _named(o, hdr, d.attachments, "attachments", 60, 328, 332)

    if d.includes:
        raw = bytearray()
        for r in d.includes:
            raw += r.raw
        o.add("includes", raw)
        for k, r in enumerate(d.includes):
            base = ("includes", k * 116)
            o.patch(("includes", k * 116), o.string(r.name or ""), base)
            o.add("bonemap%d" % k, BONEMAP_RECORD * nb)
            o.patch(("includes", k * 116 + 0x10), ("bonemap%d" % k, 0), base)
    count(404, len(d.includes))
    hdrptr(408, "includes", len(d.includes))

    if d.anims:
        raw = bytearray()
        for r in d.anims:
            raw += r.raw
        o.add("animdescs", raw)
        for k, r in enumerate(d.anims):
            base = ("animdescs", k * 72)
            o.patch(("animdescs", k * 72), o.string(r.name or ""), base)
            mv = r.extra.get("movements") or []
            struct.pack_into("<i", o.data["animdescs"], k * 72 + 0x10, len(mv))
            if mv:
                o.add("mv%d" % k, b"".join(mv))
                o.patch(("animdescs", k * 72 + 0x14), ("mv%d" % k, 0), base)
            blk = r.extra.get("block") or b""
            if blk:
                o.add("anim%d" % k, blk)
                o.patch(("animdescs", k * 72 + 0x30), ("anim%d" % k, 0), base)
    count(264, len(d.anims))
    hdrptr(268, "animdescs", len(d.anims))

    if d.seqs:
        raw = bytearray()
        for r in d.seqs:
            raw += r.raw
        o.add("seqdescs", raw)
        for k, r in enumerate(d.seqs):
            base = ("seqdescs", k * 764)
            buf = o.data["seqdescs"]
            o.patch(("seqdescs", k * 764), o.string(r.name or ""), base)
            if r.extra.get("activity") is not None:
                o.patch(("seqdescs", k * 764 + 4), o.string(r.extra["activity"]), base)
            ev = r.extra.get("events") or []
            struct.pack_into("<i", buf, k * 764 + 0x14, len(ev))
            if ev:
                o.add("ev%d" % k, b"".join(ev))
                o.patch(("seqdescs", k * 764 + 0x18), ("ev%d" % k, 0), base)
            al = r.extra.get("autolayers") or []
            struct.pack_into("<i", buf, k * 764 + 0x294, len(al))
            if al:
                o.add("al%d" % k, struct.pack("<%di" % len(al), *al))
                o.patch(("seqdescs", k * 764 + 0x298), ("al%d" % k, 0), base)
            hv = r.extra.get("hitvolumes") or b""
            struct.pack_into("<i", buf, k * 764 + 0x2bc, len(hv) // 24)
            if hv:
                o.add("hv%d" % k, hv)
                o.patch(("seqdescs", k * 764 + 0x2c0), ("hv%d" % k, 0), base)
            kbs = r.extra.get("knockbacks") or []
            struct.pack_into("<i", buf, k * 764 + 0x2c4, len(kbs))
            if kbs:
                o.add("kb%d" % k, b"".join(bytes(x.raw) for x in kbs))
                o.patch(("seqdescs", k * 764 + 0x2c8), ("kb%d" % k, 0), base)
                for j, x in enumerate(kbs):
                    for t, nm in enumerate(x.extra.get("names") or []):
                        if nm is None:
                            continue
                        at = j * 188 + 0x78 + t * 4
                        o.patch(("kb%d" % k, at), o.string(nm), ("kb%d" % k, j * 188))
            for f, nm in ((0x2dc, "dodge"), (0x2e4, "block"),
                          (0x2e8, "seq2e8"), (0x2ec, "seq2ec")):
                v = r.extra.get(nm)
                # -1, not 0: the engine reads these with `-1 < value`, so 0 would pass the
                # test and dereference the seqdesc's own first byte.  relocs.POSITIVE_ONLY.
                struct.pack_into("<i", buf, k * 764 + f, -1)
                if v is not None:
                    o.patch(("seqdescs", k * 764 + f), o.string(v), base)
    count(272, len(d.seqs))
    hdrptr(276, "seqdescs", len(d.seqs))

    if d.seqgroups:
        raw = bytearray()
        for r in d.seqgroups:
            raw += r.raw
        o.add("seqgroups", raw)
        for k, r in enumerate(d.seqgroups):
            base = ("seqgroups", k * 16)
            o.patch(("seqgroups", k * 16), o.string(r.name or ""), base)
            o.patch(("seqgroups", k * 16 + 4), o.string(r.extra.get("name") or ""), base)
    count(284, len(d.seqgroups))
    hdrptr(288, "seqgroups", len(d.seqgroups))

    _named(o, hdr, d.flexdescs, "flexdescs", 4, 344, 348)
    if d.flexcontrollers:
        raw = bytearray()
        for r in d.flexcontrollers:
            raw += r.raw
        o.add("flexcontrollers", raw)
        for k, r in enumerate(d.flexcontrollers):
            base = ("flexcontrollers", k * 20)
            o.patch(("flexcontrollers", k * 20), o.string(r.extra.get("type") or ""), base)
            o.patch(("flexcontrollers", k * 20 + 4), o.string(r.name or ""), base)
    count(352, len(d.flexcontrollers))
    hdrptr(356, "flexcontrollers", len(d.flexcontrollers))

    if d.flexrules:
        raw = bytearray()
        for r in d.flexrules:
            raw += r.raw
        o.add("flexrules", raw)
        for k, r in enumerate(d.flexrules):
            ops = r.extra.get("ops") or b""
            struct.pack_into("<i", o.data["flexrules"], k * 12 + 4, len(ops) // 8)
            if ops:
                o.add("flexops%d" % k, ops)
                o.patch(("flexrules", k * 12 + 8), ("flexops%d" % k, 0),
                        ("flexrules", k * 12))
    count(360, len(d.flexrules))
    hdrptr(364, "flexrules", len(d.flexrules))

    if d.ikchains:
        raw = bytearray()
        for r in d.ikchains:
            raw += r.raw
        o.add("ikchains", raw)
        for k, r in enumerate(d.ikchains):
            base = ("ikchains", k * 16)
            o.patch(("ikchains", k * 16), o.string(r.name or ""), base)
            links = r.extra.get("links") or b""
            struct.pack_into("<i", o.data["ikchains"], k * 16 + 8, len(links) // 28)
            if links:
                o.add("iklinks%d" % k, links)
                o.patch(("ikchains", k * 16 + 0x0c), ("iklinks%d" % k, 0), base)
    count(368, len(d.ikchains))
    hdrptr(372, "ikchains", len(d.ikchains))

    _simple(o, hdr, d.mouths, "mouths", 20, 376, 380)
    _named(o, hdr, d.poseparams, "poseparams", 20, 384, 388)
    _simple(o, hdr, d.springbones, "springbones", 28, 396, 400)

    if d.bodyparts:
        raw = bytearray()
        for r in d.bodyparts:
            raw += r.raw
        o.add("bodyparts", raw)
        for bp, r in enumerate(d.bodyparts):
            base = ("bodyparts", bp * 16)
            o.patch(("bodyparts", bp * 16), o.string(r.name or ""), base)
            struct.pack_into("<i", o.data["bodyparts"], bp * 16 + 4, len(r.kids))
            mk = "models%d" % bp
            o.add(mk, b"".join(bytes(x.raw) for x in r.kids))
            o.patch(("bodyparts", bp * 16 + 0x0c), (mk, 0), base)
            for md, mr in enumerate(r.kids):
                _emit_model(o, bp, md, mk, mr)
    count(320, len(d.bodyparts))
    hdrptr(324, "bodyparts", len(d.bodyparts))

    _named(o, hdr, d.textures, "textures", 20, 292, 296)
    if d.cdtextures:
        o.add("cdtextures", b"\0" * (4 * len(d.cdtextures)))
        for k, t in enumerate(d.cdtextures):
            o.patch(("cdtextures", k * 4), o.string(t or ""))
    count(300, len(d.cdtextures))
    hdrptr(304, "cdtextures", len(d.cdtextures))

    nfam = len(d.skin)
    nref = len(d.skin[0]) if nfam else 0
    if nfam:
        o.add("skin", b"".join(struct.pack("<%dh" % nref, *row) for row in d.skin), 2)
    count(308, nref)
    count(312, nfam)
    hdrptr(316, "skin", nfam)

    if d.surfaceprop is not None:
        o.patch(("hdr", 392), o.string(d.surfaceprop))
    else:
        struct.pack_into("<i", hdr, 392, 0)

    # Nothing in the corpus carries one, so there is nothing to emit and nothing to guess.
    count(336, 0)
    struct.pack_into("<i", hdr, 340, 0)

    o.data["hdr"][:] = hdr
    buf = bytearray(o.build())
    struct.pack_into("<i", buf, 140, len(buf))
    struct.pack_into("<i", buf, 8, d.checksum if checksum is None else checksum)
    return bytes(buf)


def _simple(o, hdr, recs, key, stride, cnt_off, idx_off):
    """An array with no pointer inside it."""
    if recs:
        o.add(key, b"".join(bytes(r.raw) for r in recs))
        o.patch(("hdr", idx_off), (key, 0))
    struct.pack_into("<i", hdr, cnt_off, len(recs))
    return stride


def _named(o, hdr, recs, key, stride, cnt_off, idx_off):
    """An array whose only pointer is a struct-relative name at +0x00."""
    if recs:
        o.add(key, b"".join(bytes(r.raw) for r in recs))
        for k, r in enumerate(recs):
            o.patch((key, k * stride), o.string(r.name or ""), (key, k * stride))
        o.patch(("hdr", idx_off), (key, 0))
    struct.pack_into("<i", hdr, cnt_off, len(recs))


def _emit_model(o, bp, md, mk, mr):
    tag = "%d_%d" % (bp, md)
    base = (mk, md * 224)
    buf = o.data[mk]
    meshes, verts = mr.kids, mr.extra.get("verts") or b""
    tangents = mr.extra.get("tangents") or b""
    eyes = mr.extra.get("eyes") or []
    struct.pack_into("<i", buf, md * 224 + 0x88, len(meshes))
    nv = len(tangents) // 16
    struct.pack_into("<i", buf, md * 224 + 0x90, nv)
    sk = "meshes%s" % tag
    if meshes:
        o.add(sk, b"".join(bytes(x.raw) for x in meshes))
        o.patch((mk, md * 224 + 0x8c), (sk, 0), base)
        for k, x in enumerate(meshes):
            o.patch((sk, k * 60 + 4), base, (sk, k * 60))
            struct.pack_into("<i", o.data[sk], k * 60 + 0x10, len(x.kids))
            if x.kids:
                fk = "flex%s_%d" % (tag, k)
                o.add(fk, b"".join(bytes(f.raw) for f in x.kids))
                o.patch((sk, k * 60 + 0x14), (fk, 0), (sk, k * 60))
                for r, f in enumerate(x.kids):
                    pay = f.extra.get("payload") or b""
                    if not pay:
                        continue
                    pk = "vertanim%s_%d_%d" % (tag, k, r)
                    o.add(pk, pay)
                    o.patch((fk, r * 32 + 0x18), (pk, 0), (fk, r * 32))
    if nv:
        o.add("verts%s" % tag, verts)
        o.patch((mk, md * 224 + 0x94), ("verts%s" % tag, 0), base)
        o.add("tang%s" % tag, tangents)
        o.patch((mk, md * 224 + 0x98), ("tang%s" % tag, 0), base)
    struct.pack_into("<i", buf, md * 224 + 0xc0, len(eyes))
    if eyes:
        o.add("eyes%s" % tag, b"".join(bytes(x.raw) for x in eyes))
        o.patch((mk, md * 224 + 0xc4), ("eyes%s" % tag, 0), base)
    for off in (0xc8, 0xcc, 0xd0, 0xd4, 0xd8, 0xdc):
        struct.pack_into("<i", buf, md * 224 + off, 0)
    for key, cnt_off, idx_off, stride in (("clothcollide", 0xd0, 0xd4, 36),
                                          ("clothsphere", 0xd8, 0xdc, 20)):
        blob = mr.extra.get(key) or b""
        if not blob:
            continue
        kk = "%s%s" % (key, tag)
        o.add(kk, blob)
        struct.pack_into("<i", buf, md * 224 + cnt_off, len(blob) // stride)
        o.patch((mk, md * 224 + idx_off), (kk, 0), base)
    cl = mr.extra.get("cloth")
    if cl:
        ck = "cloth%s" % tag
        o.add(ck, cl["data"])
        struct.pack_into("<i", buf, md * 224 + 0xc8, cl["cols"])
        o.patch((mk, md * 224 + 0xcc), (ck, cl["table"]), base)
        for slot, rel in cl["slots"]:
            o.patch((ck, cl["table"] + slot * 4), (ck, rel), base)
        for k, (spans, _was) in sorted(cl["meshes"].items()):
            for f, rel in zip((0x30, 0x34, 0x38), spans):
                o.patch((sk, k * 60 + f), (ck, rel), (sk, k * 60))


def build(b, checksum=None, drop=False):
    """(bytes, dropped) -- read a shipped file into a description and author it again."""
    d = from_bytes(b)
    return emit(d, checksum, drop), dict(d.dropped)


def apply_anims(d, edits, path=""):
    """Put authored poses into `d`: {animation index: {poses, fps, flags, movements}},
    any of the last three optional.  The poses are held, not encoded -- one scale set
    serves the whole file, so the channel a value needs is not decided until `emit`."""
    for i, e in sorted(edits.items()):
        if not 0 <= i < len(d.anims):
            raise ValueError("no animation %d in %s" % (i, path))
        r = d.anims[i]
        r.extra["poses"] = e["poses"]
        r.extra["block"] = b""
        struct.pack_into("<i", r.raw, 0x0c, len(e["poses"]))
        if e.get("movements") is not None:
            r.extra["movements"] = [W.movement_bytes(x) for x in e["movements"]]
        if e.get("fps") is not None:
            struct.pack_into("<f", r.raw, 0x04, float(e["fps"]))
        if e.get("flags") is not None:
            struct.pack_into("<i", r.raw, 0x08, int(e["flags"]))
    return d


def write_many(m, edits, checksum=None):
    """`m` authored again with `edits` applied, as bytes.

    The donor's checksum is kept by default: the `.vtx` beside the file carries it and the
    engine draws nothing at all when the two disagree.
    """
    d = apply_anims(from_bytes(bytes(m.d)), edits, m.path)
    return emit(d, checksum=m.checksum if checksum is None else checksum)


FLT_MIN = 1.17549435e-38
FLT_MAX = 3.40282347e+38


def new(name, surfaceprop="flesh"):
    """An empty description: header scalars, one sequence group, no records.

    The unnamed scalars take their corpus value rather than zero -- `unk144` is
    (0.5, 0.5, 0.5) on 4438 of 4445 models and `unhz` is (0, 0, 1) on the same 4438. The
    phoneme filter is written for the same reason and a stronger one: it is a clamp, so
    leaving it zero does not mean "unset", it means every lipsync blend collapses to zero
    width.
    """
    d = Desc()
    d.name = name
    h = d.hdr
    struct.pack_into("<i", h, 0, 0x54534449)
    struct.pack_into("<i", h, 4, M.VERSION)
    struct.pack_into("<3f", h, 144, *DEFAULTS["hdr.unk144"])
    struct.pack_into("<2f", h, 232, *DEFAULTS["hdr.phonemefilter"])
    struct.pack_into("<3f", h, 180, -16.0, -16.0, 0.0)
    struct.pack_into("<3f", h, 192, 16.0, 16.0, 72.0)
    struct.pack_into("<3i", h, 412, *DEFAULTS["hdr.unhz"])
    d.surfaceprop = surfaceprop
    lab, nm = DEFAULTS["seqgroup"]
    d.seqgroups.append(Rec(bytearray(16), lab, None, {"name": nm}))
    d.faces = []
    return d


def _block_bones(blk):
    """The bone count a carried block was encoded for, or None if it has no live channel.

    Where the first RLE stream starts: every offset is struct-relative, so the lowest
    `bone * 32 + offset` is where the `mstudioanim_t` array ends.  Over the 4445-model
    corpus all 10989 animations carrying a live channel put it at exactly `numbones * 32`
    bar 39, all in `move_and_ranged.mdl`, which sit 4 bytes past -- so this rounds down and
    is a lower bound rather than the count itself.
    """
    n = len(blk) // M.ANIM_STRIDE
    lo = None
    for j in range(n):
        o = j * M.ANIM_STRIDE
        for x in struct.unpack_from("<7i", blk, o + 4):
            if x and (lo is None or o + x < lo):
                lo = o + x
    return None if lo is None else lo // M.ANIM_STRIDE


def _check_blocks(d):
    """A carried animation block must describe the bone list it is about to be written
    against.  `mstudioanim_t[numbones]` is sized off the header, so a block encoded for
    fewer bones hands the engine records read out of the RLE stream that follows -- six
    struct-relative offsets made of animation data, not a wrong bone.

    Every site that writes `extra["block"]` stamps `extra["block_bones"]` beside it, so
    this is one integer comparison per animation.  Deriving it from the offsets instead
    costs 47% of a whole `emit`, and it only sees the block growing SHORT of the bone list:
    a shrunk one still puts its stream past the end of the shorter array, and nothing in
    the bytes says how many records were meant.  The scan is the fallback for a block some
    caller assembled without the stamp.
    """
    nb = len(d.bones)
    for k, r in enumerate(d.anims):
        blk = r.extra.get("block") or b""
        if not blk:
            continue
        was = r.extra.get("block_bones")
        if was is None:
            was = _block_bones(blk)
            if was is None or was >= nb:
                continue
        if was != nb:
            raise Refused("animation %d %r was encoded for %d bones and is being written "
                          "against %d, so the engine reads %s"
                          % (k, r.name, was, nb,
                             "its last %d mstudioanim_t out of the RLE stream"
                             % (nb - was) if nb > was else
                             "%d records fewer than the block holds" % (was - nb)))


def add_bone(d, name, parent=-1, pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0),
             flags=BONE_USED, surfaceprop="flesh"):
    """Append one bone, return its index.  A parent must already be in `d.bones`: the
    engine walks the array once in order, so a child emitted first reads a world matrix
    that has not been computed.

    Every animation already in the file grows an entry for it, holding the bind pose.
    `mstudioanim_t[numbones]` is sized off the header, so a block left at the old count
    hands the engine a record read out of the RLE stream that follows: on andrei that is
    seven offsets of 63822..64775 against the 0 a still bone carries, each one taken from
    the record and landing about 66 KB past the end of the block.  The weight is 1.0
    rather than `_anim_block`'s 0.0 default, which is what 54320 of 56376 shipped
    channel-less records carry, and what all 158530 with a channel do.
    """
    if parent >= len(d.bones):
        raise Refused("bone %r cites parent %d, not emitted yet" % (name, parent))
    old = _skeleton(d)
    tracks = []
    for r in d.anims:
        blk = r.extra.get("block")
        tracks.append(W.read_tracks(_Block(blk, old.bones), _AnimHdr(r)) if blk else None)
    raw = bytearray(160)
    struct.pack_into("<i", raw, 0x04, parent)
    struct.pack_into("<6i", raw, 0x08, *([-1] * 6))
    struct.pack_into("<3f", raw, 0x20, *pos)
    struct.pack_into("<4f", raw, 0x2c, *quat)
    struct.pack_into("<3f", raw, 0x3c, 1.0 / 256, 1.0 / 256, 1.0 / 256)
    struct.pack_into("<4f", raw, 0x48, 1e-5, 1e-5, 1e-5, 1.0 / 32768)
    struct.pack_into("<i", raw, 0x88, flags)
    # physicsbone +0x94 indexes the sibling .phy's solid list.  A bone owning no solid
    # takes its nearest ancestor's, else 0, which is what studiomdl does
    # (collisionmodel.cpp:1955-1976); -1 is a value 0 of 61925 shipped bones carry.
    struct.pack_into("<i", raw, 0x94,
                     struct.unpack_from("<i", d.bones[parent].raw, 0x94)[0]
                     if parent >= 0 else 0)
    d.bones.append(Rec(raw, name, None, {"surfaceprop": surfaceprop}))
    _stamp_posetobone(d)
    i = len(d.bones) - 1
    new = _skeleton(d)
    for r, t in zip(d.anims, tracks):
        if t is None:
            for f in r.extra.get("poses") or ():
                f.append(None)
            continue
        t.weights[i] = 1.0
        r.extra["block"] = W._anim_block(new, t)
        r.extra["block_bones"] = len(d.bones)
    return i


def _stamp_posetobone(d):
    """The inverse of the bone's world bind matrix, recomputed whenever the chain changes
    rather than asked of the caller."""
    world = []
    for r in d.bones:
        pos = struct.unpack_from("<3f", r.raw, 0x20)
        quat = struct.unpack_from("<4f", r.raw, 0x2c)
        parent = struct.unpack_from("<i", r.raw, 0x04)[0]
        m = M.mat_from_quat_pos(quat, pos)
        world.append(M.mat_mul(world[parent], m) if parent >= 0 else m)
    for r, w in zip(d.bones, world):
        inv = M.mat_inverse(w)
        struct.pack_into("<12f", r.raw, 0x58, *[x for row in inv for x in row])


def set_bone_poses(d, poses):
    """Overwrite the bind pose of bones already in `d`.  `poses` is
    {index: (pos, quat, flags, parent)}; `flags` may be None to keep the record's own and
    `parent` is None on all but a reparented bone.

    A reparent is carried by `rebase_carried` for nothing extra: it re-encodes against the
    bind each record now holds, and a bone whose parent changed has a different
    parent-relative bind by construction, so it is in `moved` already.  Every channel is
    parent-local, so `A_new = B_new . B_old^-1 . A_old` preserves the animation's offset
    from the bind whichever parent each side is relative to.

    The rest of each record stays: the quantisation scales an existing animation is
    already encoded against, surfaceprop, and the procedural helper.  `poseToBone` is
    restamped for every bone, not only the named ones, because a moved parent changes its
    children's world matrices too.

    Returns `rebase_carried`'s report.  A rotation channel is `int16 * rotscale` with no
    bind base, so an animation nobody re-exported would otherwise keep pointing where the
    old bind put it while the rest pose moved out from under it; re-basing is unconditional
    and there is no option for it, because one file answering the same edit two ways -- the
    re-exported animations following it and the carried ones not -- is the defect.
    """
    old = _skeleton(d)
    for k, (pos, quat, flags, parent) in poses.items():
        raw = d.bones[k].raw
        struct.pack_into("<3f", raw, 0x20, *pos)
        struct.pack_into("<4f", raw, 0x2c, *quat)
        if flags is not None:
            struct.pack_into("<i", raw, 0x88, flags)
        if parent is not None:
            struct.pack_into("<i", raw, 0x04, parent)
    _stamp_posetobone(d)
    return rebase_carried(d, old)


def rebase_carried(d, old):
    """Re-encode every carried animation against the bind `set_bone_poses` just wrote.

    `old` is the skeleton as it was, which is the only thing that can still say what the
    blocks were encoded against -- the records themselves have already been overwritten.

    Only a bone whose bind actually moved is re-based.  A record written for a flags-only
    edit carries Blender's own `matrix_local` rather than the file's, which is ~1.3e-4
    away, and re-basing on that would requantise the file for storage noise.

    An edited animation is left alone: it holds `extra["poses"]` in float and has not been
    quantised yet, so `quantise` will encode it against the new bind anyway.
    """
    new = _skeleton(d)
    moved = [k for k in range(len(d.bones))
             if list(old.bones[k].pos) != list(new.bones[k].pos)
             or list(old.bones[k].quat) != list(new.bones[k].quat)]
    turned = [k for k in moved if list(old.bones[k].quat) != list(new.bones[k].quat)]
    report = {"moved": moved, "turned": turned, "anims": 0, "widened": [],
              "root_turned": []}
    if not moved:
        return report
    # A movement block is an offset on the entity transform, above the whole skeleton, so
    # no bind edit invalidates one.  Turning a parentless bone's bind does rotate every
    # frame under a travel direction the block still states unrotated, and that is a
    # consequence to name rather than a value to derive.
    if any(r.extra.get("movements") for r in d.anims):
        report["root_turned"] = [k for k in turned
                                 if struct.unpack_from("<i", d.bones[k].raw, 0x04)[0] < 0]
    carried = [r for r in d.anims if r.extra.get("block")]
    if not carried:
        return report
    tracks = [W.read_tracks(_Block(r.extra["block"], new.bones), _AnimHdr(r))
              for r in carried]
    chans = [W.rebased_channels(t, old, new, moved) for t in tracks]
    scales, widened = W.fit_rebase_scales(new, chans)
    for k, br in enumerate(d.bones):
        struct.pack_into("<3f", br.raw, 0x3c, *scales[k][:3])
        struct.pack_into("<4f", br.raw, 0x48, *scales[k][3:])
        new.bones[k].posscale = scales[k][:3]
        new.bones[k].rotscale = scales[k][3:]
    for r, t, ch in zip(carried, tracks, chans):
        if not ch:
            continue
        W.apply_rebase(t, new, ch, scales)
        r.extra["block"] = W._anim_block(new, t)
        r.extra["block_bones"] = len(d.bones)
        report["anims"] += 1
    report["widened"] = sorted(d.bones[k].name for k in widened)
    return report


def _renumber_verts(mr, i):
    """Drop bone `i` from every vertex of one model and renumber the slots above it.

    A slot that named `i` goes and the surviving weights rescale to 255, which is what
    Blender's armature modifier draws once the group matches no bone. A vertex left with
    none becomes `numbones` 0 and rides bone 0 -- the skinning block transforms by
    `bone[0]` before it reads the count at all, StudioRender 0x2c0172b9, so the format has
    no model space to leave it in. Blender leaves it where the rest pose put it, measured
    at 1.267855 units from where the bone had been carrying it.
    """
    if struct.unpack_from("<i", mr.raw, 0x9c)[0] != 0:
        return 0, 0                     # filetype 1 and 2 carry no bone field at all
    vb = bytearray(mr.extra.get("verts") or b"")
    moved = rigid = 0
    for o in range(0, len(vb) - 43, 44):
        n = vb[o + 3] % 5
        if not n:
            continue
        w = [vb[o], vb[o + 1], vb[o + 2]]
        w.append(255 - sum(w))
        bones = list(struct.unpack_from("<4h", vb, o + 4))
        keep = [(b - 1 if b > i else b, w[k]) for k, b in enumerate(bones[:n])
                if b != i]
        if len(keep) == n:
            if max(bones[:n]) <= i:
                continue
            struct.pack_into("<4h", vb, o + 4,
                             *([b for b, _x in keep] + [0] * (4 - len(keep))))
            moved += 1
            continue
        moved += 1
        if not keep:
            rigid += 1
            vb[o] = vb[o + 1] = vb[o + 2] = 0
            vb[o + 3] -= n
            struct.pack_into("<4h", vb, o + 4, 0, 0, 0, 0)
            continue
        tot = sum(x for _b, x in keep) or 1
        q = [int(round(255.0 * x / tot)) for _b, x in keep]
        while len(q) < 3:
            q.append(0)
        # The three stored bytes sum to 255 and the engine derives the fourth from them,
        # so a vertex short of four bones puts the rounding shortfall on its first slot --
        # `_pack_verts` does the same and the two have to agree.
        if len(keep) < 4:
            q[0] += 255 - sum(q[:3])
        vb[o], vb[o + 1], vb[o + 2] = (max(0, min(255, x)) for x in q[:3])
        vb[o + 3] -= n - len(keep)
        struct.pack_into("<4h", vb, o + 4,
                         *([b for b, _x in keep] + [0] * (4 - len(keep))))
    mr.extra["verts"] = bytes(vb)
    return moved, rigid


def _model_label(bi, mi, mr):
    """How to name a model in a message. Every model carrying a cloth volume has an empty
    name, so the indices are the only thing that identifies one to the user."""
    return ("model %r" % mr.name) if mr.name else ("model %d.%d" % (bi, mi))


def remove_bone(d, i):
    """Drop bone `i` and renumber every index that named a bone above it.

    Children reparent onto the removed bone's own parent and keep their world position,
    which is what Blender's `armature.delete` does -- measured on 5.2, deleting the middle
    of a three-bone chain leaves the child's armature-space head and tail unchanged and
    clears its connection. Composing the removed bone's local transform into each child's
    is that same operation in a format whose bind pose is parent-relative.

    Nothing renumbers on its own: `emit` writes each `Rec.raw` verbatim, so an index left
    alone names a different bone and the file still writes. `physicsbone` is the one
    field naming something outside the bone array deliberately untouched -- it indexes the
    sibling `.phy`'s solid list, not a bone, reaching 26 at most over 61 925 corpus bones
    in files whose `numbones` runs to 343, and 0 of them are negative.

    Returns (name, [child names], vertex slots repointed, vertices left rigid, records
    rebound onto the parent).
    """
    if not 0 <= i < len(d.bones):
        raise Refused("no bone %d to remove: the file has %d" % (i, len(d.bones)))
    if len(d.bones) == 1:
        raise Refused("bone %r is the file's only one" % d.bones[i].name)
    name = d.bones[i].name

    for k, r in enumerate(d.bonecontrollers):
        if struct.unpack_from("<i", r.raw, 0x00)[0] == i:
            raise Refused("bone controller %d drives bone %r, and the engine resolves it "
                          "by index with nothing to fall back on" % (k, name))
    for r in d.ikchains:
        links = r.extra.get("links") or b""
        for j in range(len(links) // 28):
            if struct.unpack_from("<i", links, j * 28)[0] == i:
                raise Refused("IK chain %r link %d is bone %r" % (r.name, j, name))
    for k, r in enumerate(d.springbones):
        # +0x00 is -1-bone when the sign is set (client.dll:0x100ac115); +0x04's -1 is the
        # walk-to-the-leaf sentinel, not an encoded index, so it is compared raw.
        v, e = struct.unpack_from("<2i", r.raw, 0x00)
        if (v if v >= 0 else -1 - v) == i or e == i:
            raise Refused("spring bone chain %d is anchored to bone %r" % (k, name))

    parent = struct.unpack_from("<i", d.bones[i].raw, 0x04)[0]
    if parent < 0:
        # Raised here rather than where the record is patched: the walk below mutates as it
        # goes, and every other refusal in this function runs before anything is written.
        bound = []
        for k, r in enumerate(d.hitboxsets):
            bound += ["hitbox %d.%d" % (k, j) for j, x in enumerate(r.kids)
                      if struct.unpack_from("<i", x.raw, 0x00)[0] == i]
        bound += ["attachment %r" % r.name for r in d.attachments
                  if struct.unpack_from("<i", r.raw, 0x08)[0] == i]
        bound += ["mouth %d" % k for k, r in enumerate(d.mouths)
                  if struct.unpack_from("<i", r.raw, 0x00)[0] == i]
        for bi, bp in enumerate(d.bodyparts):
            for mi, mr in enumerate(bp.kids):
                where = _model_label(bi, mi, mr)
                bound += ["eyeball %d of %s" % (j, where)
                          for j, x in enumerate(mr.extra.get("eyes") or [])
                          if struct.unpack_from("<i", x.raw, 0x04)[0] == i]
                for key, stride, fields in (("clothcollide", 36, (0x00, 0x04)),
                                            ("clothsphere", 20, (0x00,))):
                    buf = mr.extra.get(key) or b""
                    bound += ["%s %d of %s" % (key, j, where)
                              for j in range(len(buf) // stride) for f in fields
                              if struct.unpack_from("<i", buf, j * stride + f)[0] == i]
        if bound:
            raise Refused("bone %r is a root, so there is no parent to pass %d record(s) "
                          "to: %s" % (name, len(bound), ", ".join(bound[:4])))

    old = _skeleton(d)
    tracks = []
    for r in d.anims:
        blk = r.extra.get("block")
        tracks.append(W.read_tracks(_Block(blk, old.bones), _AnimHdr(r)) if blk else None)

    gone = M.mat_from_quat_pos(struct.unpack_from("<4f", d.bones[i].raw, 0x2c),
                               struct.unpack_from("<3f", d.bones[i].raw, 0x20))
    kids = []
    for r in d.bones:
        if struct.unpack_from("<i", r.raw, 0x04)[0] != i:
            continue
        kids.append(r.name)
        w = M.mat_mul(gone, M.mat_from_quat_pos(
            struct.unpack_from("<4f", r.raw, 0x2c),
            struct.unpack_from("<3f", r.raw, 0x20)))
        struct.pack_into("<3f", r.raw, 0x20, w[0][3], w[1][3], w[2][3])
        struct.pack_into("<4f", r.raw, 0x2c, *M.quat_from_mat(w))
        struct.pack_into("<i", r.raw, 0x04, parent)

    def ren(x):
        return x - 1 if x > i else x

    rebound = []

    def patch(buf, at, what, signed=False):
        """A record that named the removed bone passes to its parent rather than keeping
        an index that now names a different bone. Parents always precede their children,
        so the parent's own index never renumbers.

        `signed` is the spring bone's -1-bone form: decode, renumber, re-encode."""
        v = struct.unpack_from("<i", buf, at)[0]
        neg = signed and v < 0
        if neg:
            v = -1 - v
        if v > i:
            struct.pack_into("<i", buf, at, -v if neg else v - 1)
        elif v == i:
            if parent < 0:
                raise Refused("%s is bound to bone %r, which is a root: there is no "
                              "parent to pass it to" % (what, name))
            struct.pack_into("<i", buf, at, -1 - parent if neg else parent)
            rebound.append(what)

    for r in d.bones:
        patch(r.raw, 0x04, "bone %r" % r.name)
        # mstudioaxisinterpbone_t.control @+0x00. Carried as one 176-byte blob, so it is
        # the one bone index in the file no other loop here reaches.
        if r.extra.get("proc"):
            buf = bytearray(r.extra["proc"])
            patch(buf, 0x00, "procedural bone %r" % r.name)
            r.extra["proc"] = bytes(buf)
    for k, r in enumerate(d.bonecontrollers):
        patch(r.raw, 0x00, "bone controller %d" % k)
    for k, r in enumerate(d.hitboxsets):
        for j, x in enumerate(r.kids):
            patch(x.raw, 0x00, "hitbox %d.%d" % (k, j))
    for r in d.attachments:
        patch(r.raw, 0x08, "attachment %r" % r.name)
    for k, r in enumerate(d.mouths):
        patch(r.raw, 0x00, "mouth %d" % k)
    for k, r in enumerate(d.springbones):
        patch(r.raw, 0x00, "spring bone chain %d" % k, signed=True)
        patch(r.raw, 0x04, "spring bone chain %d end" % k)
    for r in d.ikchains:
        links = bytearray(r.extra.get("links") or b"")
        for j in range(len(links) // 28):
            patch(links, j * 28, "IK chain %r link %d" % (r.name, j))
        r.extra["links"] = bytes(links)
    # mstudioknockback_t +0x08. The trail is drawn off this bone's world matrix, and
    # Studio_ChainedBoneToLocal_vtmb (client.dll 0x1008dc70) hands the index back unchanged
    # when the sequence is below hdr->numseq -- which every record reached through this
    # file's own seqdesc array is -- so the number is this file's own numbering and
    # renumbers with it. A record reached through an include belongs to the include's
    # removal instead.
    for k, r in enumerate(d.seqs):
        for j, x in enumerate(r.extra.get("knockbacks") or []):
            patch(x.raw, 0x08, "knockback %d of sequence %r" % (j, r.name if r.name else k))

    moved = rigid = 0
    for bi, bp in enumerate(d.bodyparts):
        for mi, mr in enumerate(bp.kids):
            where = _model_label(bi, mi, mr)
            for j, x in enumerate(mr.extra.get("eyes") or []):
                patch(x.raw, 0x04, "eyeball %d of %s" % (j, where))
            for key, stride, fields in (("clothcollide", 36, (0x00, 0x04)),
                                        ("clothsphere", 20, (0x00,))):
                buf = bytearray(mr.extra.get(key) or b"")
                for j in range(len(buf) // stride):
                    for f in fields:
                        patch(buf, j * stride + f, "%s %d of %s" % (key, j, where))
                mr.extra[key] = bytes(buf)
            a, b = _renumber_verts(mr, i)
            moved += a
            rigid += b

    del d.bones[i]
    _stamp_posetobone(d)

    new = _skeleton(d)
    for r, t in zip(d.anims, tracks):
        if t is None:
            poses = r.extra.get("poses")
            if poses:
                for f in poses:
                    del f[i]
            continue
        t.chan = dict(((ren(b), c), v) for (b, c), v in t.chan.items() if b != i)
        t.weights = dict((ren(b), x) for b, x in t.weights.items() if b != i)
        r.extra["block"] = W._anim_block(new, t)
        r.extra["block_bones"] = len(d.bones)
    return name, kids, moved, rigid, rebound


def add_hitbox(d, boxes, name="default", at=None):
    """Append or extend a hitbox set from [(bone, group, bbmin, bbmax), ...].

    `at` extends an existing set rather than adding another; `emit` takes the count from
    `len(kids)`, so nothing here writes +0x04 or either pointer.
    """
    if at is None:
        d.hitboxsets.append(Rec(bytearray(12), name))
        at = len(d.hitboxsets) - 1
    rec = d.hitboxsets[at]
    for bone, group, lo, hi in boxes:
        if not 0 <= bone < len(d.bones):
            raise Refused("hitbox cites bone %d of %d" % (bone, len(d.bones)))
        if any(a > c for a, c in zip(lo, hi)):
            raise Refused("hitbox on bone %d has bbmin past bbmax: %s past %s"
                          % (bone, tuple(lo), tuple(hi)))
        raw = bytearray(32)
        struct.pack_into("<ii", raw, 0, bone, group)
        struct.pack_into("<3f", raw, 8, *lo)
        struct.pack_into("<3f", raw, 20, *hi)
        rec.kids.append(Rec(raw))
    return at


def add_attachment(d, name, bone, local=None, type=0):
    """Append one mount point, return its index.

    `local` is the 3x4 rotation-translation the mount sits at in the bone's own space, as
    three rows of four; None is the identity, which is what 1306 of the 1334 shipped records
    carry -- the 28 that do not are all view-model weapons.  `type` is 0 on all 1334, so
    nothing here derives it and the default is the only value the corpus has.
    """
    if not 0 <= bone < len(d.bones):
        raise Refused("attachment %r cites bone %d of %d" % (name, bone, len(d.bones)))
    raw = bytearray(60)
    struct.pack_into("<ii", raw, 4, type, bone)
    rows = local or ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    struct.pack_into("<12f", raw, 12, *[f for row in rows for f in row])
    d.attachments.append(Rec(raw, name))
    return len(d.attachments) - 1


def add_include(d, name):
    """Append one chained model, return its index.  `name` is the path the engine resolves,
    'models/character/pc/male/pcidles_allsequences.mdl'.

    All 116 bytes go out zero. `emit` patches the only two the file supplies -- the name at
    +0x00 and the bone map at +0x10 -- and every other field is runtime scratch that
    engine.dll `Studio_BuildChainedModelBoneMaps` @2000ce40 initialises before it reads any
    of them, so the zeros are correct by construction rather than merely tolerated.
    """
    d.includes.append(Rec(bytearray(116), name))
    return len(d.includes) - 1


def set_cdtextures(d, paths):
    """The directories under `materials/` a name is tried against, in order.

    A list and not a string: the lookup is a cross product, each of `numtextures` names
    against each of these until one resolves, and nothing binds a prefix to a particular
    material -- `toreador_female_armor_0` carries 11 for 10 materials. No separator is
    inserted at the join, so an entry naming a directory ends in a slash. Normalising
    separators and dropping repeats differs from the shipped bytes on 106 of 4445 models,
    which resolve the same either way.
    """
    out = []
    for p in ([paths] if isinstance(paths, str) else paths):
        p = _norm_cdtexture(p)
        if p not in out:
            out.append(p)
    d.cdtextures = out
    return out


def _norm_cdtexture(p):
    return str(p).replace("\\", "/").strip()


def add_cdtexture(d, path):
    """Append a directory the list does not already carry. True if it was new.

    Resolution is a cross product -- every material name against every directory until one
    resolves -- so a new entry at the end cannot change what already resolved, and order
    decides only which of two candidates wins. Normalising here is what lets a repeat
    compare equal to an entry `set_cdtextures` wrote.
    """
    p = _norm_cdtexture(path)
    if p in d.cdtextures:
        return False
    d.cdtextures.append(p)
    return True


def add_material(d, name, cdtexture=None):
    """One `mstudiotexture_t`, with the one-family skin table rebuilt around it.

    `cdtexture` is the directory this material's `.vmt` lives in and is appended when the
    file does not already carry it. The list is the file's and not this material's, so an
    existing entry is never replaced and never reordered. `None` leaves a populated list
    alone and seeds an empty one with `models/`, which only `new()` can produce -- 4442 of
    the 4445 shipped models are not `models/`, so that default is a marker and not a guess.

    The new texture is appended to every skin family rather than the table being rebuilt as
    one. 201 shipped models carry more than one family, and each is a whole row of
    `numskinref` shorts, so rebuilding would drop every family but the first. Rebuilding is
    only correct where there is nothing to lose, which is a table with no rows at all.
    """
    if cdtexture is not None:
        add_cdtexture(d, cdtexture)
    elif not d.cdtextures:
        set_cdtextures(d, "models/")
    d.textures.append(Rec(bytearray(20), name))
    ref = len(d.textures) - 1
    if d.skin:
        for row in d.skin:
            row.append(ref)
    else:
        d.skin = [list(range(len(d.textures)))]
    return ref


# The largest position delta the 8-byte record can spell: 255/255 * g_flexPositionDeltaScale.
MAX_VERTANIM1_DELTA = M.FLEX_POSITION_DELTA_SCALE
# 32767, from the signed short the reader sign-extends at 0x2c0509a8 and 0x2c050af0.
MAXSTUDIOFLEXVERTS = 32767


def _flex_records(records, vertanimtype):
    """[(mesh-local index, delta, ndelta)] as the payload bytes of `vertanimtype`.

    The direction of each delta goes to the nearest `g_normalOffsetTable` entry and its
    length to a byte -- x8.0 for the position in the 8-byte form, x2.0 for the normal in
    both, which is the FADD ST0,ST0 the reader does. The 20-byte form keeps the position
    delta as raw floats and is the only way past 8.0 units.
    """
    out = bytearray()
    for index, delta, ndelta in records:
        nl = (ndelta[0] ** 2 + ndelta[1] ** 2 + ndelta[2] ** 2) ** 0.5
        if nl > 0.0:
            no = NT.encode(1, ndelta)
            ns = min(255, int(round(nl / M.FLEX_NORMAL_DELTA_SCALE * 255.0)))
        else:
            no, ns = 0, 0
        if vertanimtype:
            dl = (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5
            if dl > 0.0:
                do = NT.encode(1, delta)
                ds = min(255, int(round(dl / M.FLEX_POSITION_DELTA_SCALE * 255.0)))
            else:
                do, ds = 0, 0
            out += M._VERTANIM1.pack(index, do, no, ds, ns)
        else:
            out += M._VERTANIM0.pack(index, 0, delta[0], delta[1], delta[2], no, ns, 0)
    return bytes(out)


def flexdesc_index(d, name):
    """The `mstudioflexdesc_t` slot holding `name`, appended if the file has none."""
    for i, r in enumerate(d.flexdescs):
        if r.name == name:
            return i
    d.flexdescs.append(Rec(bytearray(4), name))
    return len(d.flexdescs) - 1


def add_flex(d, bi, mi, k, name, records, target=(0.0, 0.0, 0.0, 0.0)):
    """One `mstudioflex_t` on mesh `k`, its payload encoded from `records`.

    `records` is [(mesh-local index, position delta, normal delta), ...], the deltas at
    flex weight 1 in the file's own units. The form is chosen by what the deltas need:
    the 8-byte one unless some position delta is longer than 8.0 units, which only
    mingxiao_transformation reaches among the shipped models.

    A key must lie in [0, mesh->numvertices) -- the reader adds it to the mesh's first
    vertex with nothing bounding it -- so a caller that renumbered the mesh must rebuild
    these rather than carry them.
    """
    mr = d.bodyparts[bi].kids[mi]
    if not 0 <= k < len(mr.kids):
        raise Refused("bodypart %d model %d has no mesh %d" % (bi, mi, k))
    mesh = mr.kids[k]
    numvertices = struct.unpack_from("<i", mesh.raw, 0x08)[0]
    if len(records) > MAXSTUDIOFLEXVERTS:
        raise Refused("flex %r carries %d vertices; mstudioflex_t.numverts is read as a "
                      "signed short, so %d is the ceiling"
                      % (name, len(records), MAXSTUDIOFLEXVERTS))
    for index, _delta, _ndelta in records:
        if not 0 <= index < numvertices:
            raise Refused("flex %r names mesh-local vertex %d, outside [0, %d)"
                          % (name, index, numvertices))
    vertanimtype = 1
    for _index, delta, _ndelta in records:
        if max(abs(c) for c in delta) > MAX_VERTANIM1_DELTA or                 (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5 >                 MAX_VERTANIM1_DELTA:
            vertanimtype = 0
            break
    raw = bytearray(32)
    struct.pack_into("<i", raw, 0x00, flexdesc_index(d, name))
    struct.pack_into("<4f", raw, 0x04, *target)
    struct.pack_into("<i", raw, 0x14, len(records))
    struct.pack_into("<i", raw, 0x1c, vertanimtype)
    mesh.kids.append(Rec(raw, None, None,
                         {"payload": _flex_records(records, vertanimtype)}))
    return len(mesh.kids) - 1


def clear_flexes(d, bi, mi):
    """Drop every flex of one model. Returns how many went."""
    n = 0
    for mesh in d.bodyparts[bi].kids[mi].kids:
        n += len(mesh.kids)
        mesh.kids = []
    return n


def _tangent_basis(verts, tris):
    """One (x, y, z, w) per vertex, None where no triangle gave a usable UV basis.

    The array is read: `R_AddVertexToMesh` StudioRender.dll:0x2c013ca3 copies all sixteen
    bytes into the hardware vertex buffer for every vertex of every model, outside every
    filetype and material branch. `w` is the bitangent handedness, which the corpus carries
    negative on 3 814 304 of 5 815 551 records, so a constant 1.0 is wrong on two thirds of
    them.
    """
    acc = [[0.0] * 6 for _ in verts]
    for tri in tris:
        try:
            a, b, c = (verts[i] for i in tri)
        except IndexError:
            continue
        e1 = [b[0][k] - a[0][k] for k in range(3)]
        e2 = [c[0][k] - a[0][k] for k in range(3)]
        du1, dv1 = b[2][0] - a[2][0], b[2][1] - a[2][1]
        du2, dv2 = c[2][0] - a[2][0], c[2][1] - a[2][1]
        det = du1 * dv2 - du2 * dv1
        if not det:
            continue
        pu = [(e1[k] * dv2 - e2[k] * dv1) / det for k in range(3)]
        pv = [(e2[k] * du1 - e1[k] * du2) / det for k in range(3)]
        for i in tri:
            if 0 <= i < len(acc):
                r = acc[i]
                for k in range(3):
                    r[k] += pu[k]
                    r[3 + k] += pv[k]
    out = []
    for v, r in zip(verts, acc):
        n = _unit(v[1]) or (0.0, 0.0, 1.0)
        d = n[0] * r[0] + n[1] * r[1] + n[2] * r[2]
        t = _unit((r[0] - n[0] * d, r[1] - n[1] * d, r[2] - n[2] * d))
        if t is None:
            out.append(None)
            continue
        cx = (n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2],
              n[0] * t[1] - n[1] * t[0])
        h = cx[0] * r[3] + cx[1] * r[4] + cx[2] * r[5]
        out.append(t + (-1.0 if h < 0.0 else 1.0,))
    return out


def _tangents(verts, tris):
    """One `mstudiotangent_t` per vertex, off the mesh's own triangles.

    The array is read: `R_AddVertexToMesh` StudioRender.dll:0x2c013ca3 copies all sixteen
    bytes into the hardware vertex buffer for every vertex of every model, outside every
    filetype and material branch. `w` is the bitangent handedness, which the corpus carries
    negative on 3 814 304 of 5 815 551 records, so a constant 1.0 is wrong on two thirds of
    them.

    A vertex no triangle with a usable UV determinant reaches gets an arbitrary
    perpendicular, which is all a from-scratch write has to give it. A rewrite of an
    existing file keeps the file's own vector there instead -- `retangent_block`.
    """
    return [b if b is not None else _perp(_unit(v[1]) or (0.0, 0.0, 1.0)) + (1.0,)
            for v, b in zip(verts, _tangent_basis(verts, tris))]


# Which bytes of a vertex record a tangent is computed from. filetype 0 keeps its skinning
# in the first twelve; 1 and 2 carry nothing but position, normal and UV.
_GEOM_SPAN = {0: (12, 44), 1: (0, 12), 2: (0, 8)}


def moved_vertices(filetype, donor, new, n):
    """Which of `n` written vertices differ from the donor in position, normal or UV.

    The written bytes and not the scene values, so a move the model's own quantisation
    cannot express does not count, and neither does a field the export was not asked for.
    A vertex the donor block is too short to hold counts as moved.
    """
    lo, hi = _GEOM_SPAN[filetype]
    stride = M.VERTEX_STRIDE[filetype]
    out = set()
    for i in range(n):
        o = i * stride
        if o + hi > len(donor) or new[o + lo:o + hi] != donor[o + lo:o + hi]:
            out.add(i)
    return out


def _ring(moved, tris, n):
    """`moved` plus every vertex sharing a triangle with one of them."""
    out = set(moved)
    for t in tris:
        if any(x in moved for x in t):
            out.update(x for x in t if 0 <= x < n)
    return out


def retangent_block(filetype, donor_vb, new_vb, donor_tb, tris,
                    quant_offset=None, quant_scale=None, also=()):
    """(the model's tangent block, how many records it rewrote).

    A tangent is a function of the geometry, so an edit stales it -- and not only on the
    vertex that moved. `dP/du` is accumulated over the triangles a vertex belongs to, so
    every corner of a triangle one end of which moved is stale too, which is what `_ring`
    adds.

    Every other vertex keeps the file's own vector, and a wholesale rewrite is not
    available: the shipped array reproduces from the UVs on some models and not on others
    -- dot above 0.99 on 95.9% of `andrei`'s 2477 vertices, 50.8% of
    `blueblood_female`'s 9129 and 0.0% of `tray`'s 142 -- so rewriting an untouched
    vertex replaces a shipped vector with a different one for no reason.
    `plans/tangent-basis.py` has the corpus measurement.

    `also` is the vertices a changed triangle set stales, which moves no vertex byte. They
    are not `_ring`-expanded: dropping a triangle changes the accumulation of its own three
    corners and of nothing else.
    """
    stride = M.VERTEX_STRIDE.get(filetype)
    n = len(new_vb) // stride if stride else 0
    tris = list(tris)
    if not n or len(donor_tb) < n * 16 or not tris:
        return donor_tb, 0
    want = _ring(moved_vertices(filetype, donor_vb, new_vb, n), tris, n) | \
        {x for x in also if 0 <= x < n}
    if not want:
        return donor_tb, 0
    vs = M.decode_vertices(filetype, new_vb, 0, n, quant_offset, quant_scale)
    got = _tangent_basis([(v.pos, v.normal, v.uv, ()) for v in vs], tris)
    out, done = bytearray(donor_tb), 0
    for i in sorted(want):
        # None is a vertex every triangle of which has a zero UV determinant, or which no
        # triangle reaches at all. The file's vector stands rather than an invented one.
        if got[i] is None:
            continue
        b = struct.pack("<4f", *got[i])
        if bytes(out[i * 16:i * 16 + 16]) != b:
            out[i * 16:i * 16 + 16] = b
            done += 1
    return bytes(out), done


def _canon3(t):
    """A triangle rotated to start at its lowest corner: rotation-blind, winding-keeping."""
    t = tuple(t)
    i = t.index(min(t))
    return t[i:] + t[:i]


def _face_moved(old_tris, tris, was, new_n):
    """Corners of every triangle this mesh gained or lost, in the new numbering.

    `old_tris` is model-local, so a triangle is this mesh's when all three corners fall in
    its donor run, and re-bases on it. Multiplicity counts -- 11 shipped models carry a
    repeated triangle.
    """
    old_n, old_off = was
    a, b = {}, {}
    for t in old_tris:
        if all(old_off <= x < old_off + old_n for x in t):
            k = _canon3(tuple(x - old_off for x in t))
            a[k] = a.get(k, 0) + 1
    for t in tris:
        k = _canon3(t)
        b[k] = b.get(k, 0) + 1
    out = set()
    for k in set(a) | set(b):
        if a.get(k, 0) != b.get(k, 0):
            out.update(x for x in k if 0 <= x < new_n)
    return out


def _retangent_run(ptb, pvb, old_vb, old_tb, was, new_n, tris, old_tris=None):
    """Rewrite one mesh run's tangents: the donor's, patched where the edit reached.

    `_pack_verts` has already put a fresh accumulation in `ptb`, which is right for a
    from-scratch write and wrong here -- it disagrees with the shipped vector on most
    models, and where the UV basis is degenerate it is an arbitrary perpendicular. So the
    donor's run goes back underneath and `retangent_block` patches over it.
    """
    old_n, old_off = was
    keep = max(0, min(old_n, new_n, len(old_tb) // 16 - old_off))
    base = bytearray(ptb)
    base[:keep * 16] = old_tb[old_off * 16:(old_off + keep) * 16]
    out, n = retangent_block(0, old_vb[old_off * 44:(old_off + keep) * 44],
                             bytes(pvb), bytes(base), tris,
                             also=() if old_tris is None
                             else _face_moved(old_tris, tris, was, new_n))
    ptb[:] = out
    return n


def _pack_verts(verts, tris=()):
    """`mstudiovertex0_t[]` and its tangent array for one run of vertices."""
    vb, tb = bytearray(), bytearray()
    tang = _tangents(verts, tris)
    for i, (pos, nrm, uv, binding) in enumerate(verts):
        b = sorted(binding, key=lambda x: -x[1])[:4]
        tot = sum(w for _i, w in b) or 1.0
        q = [int(round(255 * w / tot)) for _i, w in b]
        while len(q) < 3:
            q.append(0)
        # The three bytes sum to 255 exactly; a fourth bone keeps its index and the engine
        # derives its weight as the remainder.
        if len(b) < 4:
            q[0] += 255 - sum(q[:3])
        vb += bytes(max(0, min(255, x)) for x in q[:3]) + bytes((len(b),))
        vb += struct.pack("<4h", *([i for i, _w in b] + [0] * (4 - len(b))))
        vb += struct.pack("<3f", *pos) + struct.pack("<3f", *nrm)
        vb += struct.pack("<2f", *uv)
        tb += struct.pack("<4f", *tang[i])
    return vb, tb


def add_model(d, meshes, bodypart="studio", model="model"):
    """One bodypart holding one model whose meshes share a vertex array.

    `meshes` is [(material, verts, faces), ...]; `verts` is
    [(pos, normal, uv, [(bone, weight), ...]), ...] and `faces` triples of indices into
    that mesh's own `verts`.  A mesh owns a contiguous run of the model's array, which is
    what `mstudiomesh_t.vertexoffset` addresses, so the runs concatenate in order.

    Only filetype 0 is emitted: 1 and 2 quantise through the model's
    quant_offset/quant_scale, which a from-scratch caller would have to fit.
    """
    vb, tb, recs, face_lists, offset, radius = \
        bytearray(), bytearray(), [], [], 0, 0.0
    for material, verts, faces in meshes:
        pvb, ptb = _pack_verts(verts, faces)
        vb += pvb
        tb += ptb
        mraw = bytearray(60)
        struct.pack_into("<i", mraw, 0x00, material)
        struct.pack_into("<i", mraw, 0x08, len(verts))
        struct.pack_into("<i", mraw, 0x0c, offset)
        cx = [sum(v[0][k] for v in verts) / max(1, len(verts)) for k in range(3)]
        struct.pack_into("<3f", mraw, 0x24, *cx)
        recs.append(Rec(mraw))
        face_lists.append(list(faces))
        offset += len(verts)
        radius = max([radius] + [sum(x * x for x in v[0]) ** 0.5 for v in verts])

    moraw = bytearray(224)
    moraw[0:128] = model.encode("latin1")[:127].ljust(128, b"\0")
    struct.pack_into("<f", moraw, 0x84, radius)

    mo = Rec(moraw, None, recs,
             {"verts": bytes(vb), "tangents": bytes(tb), "eyes": []})
    bp = Rec(bytearray(16), bodypart, [mo])
    struct.pack_into("<i", bp.raw, 0x08, 1)
    d.bodyparts.append(bp)
    d.faces.append([face_lists])
    return len(d.bodyparts) - 1


def add_mesh(d, verts, faces, material=0, bodypart="studio", model="model"):
    """One bodypart holding one model holding one mesh."""
    return add_model(d, [(material, verts, faces)], bodypart, model)


def _skin_key(rec, at):
    """What a filetype-0 record's skin block says, without its slot order.

    `_pack_verts` sorts by descending weight, so a vertex whose two bones differ by 1/255
    comes back with its slots swapped -- the same skinning, different bytes. The file's
    order is not derivable from anything, which is why it is kept rather than reproduced.
    """
    w = list(rec[at:at + 3])
    w.append(255 - sum(w))
    n = rec[at + 3] % 5
    return sorted(zip(struct.unpack_from("<4h", rec, at + 4)[:n], w[:n]))


def _carry_vertex_fields(pvb, old_vb, was, new_n):
    """Give each vertex an edit did not add its donor `bonecountcode` and tangent back.

    Only `code % 5` is read, and the high bits are never zero for a given count and mean
    something unread, so `mesh_write.count_code` keeps them and `_pack_verts` cannot --
    it has no donor to keep them from. The tangent is `_retangent_run`'s, which needs the
    fresh accumulation this leaves alone.
    """
    old_n, old_off = was
    for j in range(min(old_n, new_n)):
        src, at = (old_off + j) * 44, j * 44
        if src + 44 > len(old_vb):
            break
        donor, ours = _skin_key(old_vb, src), _skin_key(pvb, at)
        # A rigid vertex is `numbones == 0`, which the skinning block draws as bone 0 at
        # full weight, and all 5716 shipped ones name bone 0 with weight bytes (255, 0, 0).
        # Nothing in a Blender scene spells the count-0 form -- `split_mesh` gives a vertex
        # in no group `[(0, 1.0)]` -- so the donor's reading is kept where the two only
        # differ that way. 1453 of 4423 shipped models are rigid.
        if donor == ours or (not donor and ours == [(0, 255)]):
            pvb[at:at + 12] = old_vb[src:src + 12]
        else:
            pvb[at + 3] = (old_vb[src + 3] - old_vb[src + 3] % 5) + pvb[at + 3] % 5


def set_model_name(d, bi, mi, name):
    """`mstudiomodel_t.name`, `char name[128]` inline at +0x00 of the 224-byte record.

    Inline and not a string-table offset, so this moves nothing: the same class of edit as
    `studiohdr_t.name`. 128 bytes is not a limit in practice -- the longest of the corpus's
    4567 model names is 69.
    """
    r = d.bodyparts[bi].kids[mi]
    b = name.encode("latin-1", "replace")
    if len(b) > 127:
        raise Refused("model name %r is %d bytes, and the field holds 127 plus a "
                      "terminator" % (name, len(b)))
    r.raw[0:128] = b + b"\x00" * (128 - len(b))


def replace_model(d, bi, mi, meshes, keep_center=True, donor_tris=None):
    """Rewrite one existing model's geometry, keeping everything else its records carry.

    `meshes` is `add_model`'s -- [(material, verts, faces), ...] -- one entry per mesh the
    model already holds, in that order. The partition is what `mstudiomesh_t.vertexoffset`
    addresses and what the .vtx indexes, so a caller that cannot preserve it has renumbered
    every vertex and must say so rather than call this.

    Flexes, eyeballs, materialtype, meshid and the mesh centre stay with the record they
    were read from. `mstudiomesh_t.center` is not the centroid over the corpus, and
    `boundingradius` is 0.0 on every shipped record, so neither is refitted.
    Returns the per-mesh face lists and how many tangents were rewritten.
    """
    mr = d.bodyparts[bi].kids[mi]
    if len(meshes) != len(mr.kids):
        raise Refused("%d meshes in the scene against %d in bodypart %d model %d; the "
                      "file's mesh partition is what the .vtx indexes"
                      % (len(meshes), len(mr.kids), bi, mi))
    # Read before the loop overwrites them: under the preserved-numbering contract new
    # local index j < the mesh's old count is old index j, which is what lets the two
    # fields below come back rather than be re-derived.
    was = [struct.unpack_from("<2i", x.raw, 0x08) for x in mr.kids]
    old_ft = struct.unpack_from("<i", mr.raw, 0x9c)[0]
    old_vb = mr.extra.get("verts") or b""
    old_tb = mr.extra.get("tangents") or b""
    vb, tb, offset, faces, retang = bytearray(), bytearray(), 0, [], 0
    for k, (material, verts, tris) in enumerate(meshes):
        pvb, ptb = _pack_verts(verts, tris)
        if old_ft == 0:
            _carry_vertex_fields(pvb, old_vb, was[k], len(verts))
            retang += _retangent_run(ptb, pvb, old_vb, old_tb, was[k], len(verts),
                                     tris, donor_tris)
        vb += pvb
        tb += ptb
        raw = mr.kids[k].raw
        # None keeps the donor's: the material a mesh draws with is the file's own index
        # into mstudiotexture_t[], and a Blender material slot number is not that.
        if material is not None:
            struct.pack_into("<i", raw, 0x00, material)
        struct.pack_into("<i", raw, 0x08, len(verts))
        struct.pack_into("<i", raw, 0x0c, offset)
        if not keep_center:
            struct.pack_into("<3f", raw, 0x24,
                             *[sum(v[0][c] for v in verts) / max(1, len(verts))
                               for c in range(3)])
        _regrow_cloth(mr, k, len(verts))
        offset += len(verts)
        faces.append(list(tris))
    mr.extra["verts"] = bytes(vb)
    mr.extra["tangents"] = bytes(tb)
    # 44-byte records are filetype 0 whatever the donor was: 1 and 2 carry no weight or
    # bone field at all, so a quantised donor gains skinning here rather than losing it.
    struct.pack_into("<i", mr.raw, 0x9c, 0)
    return faces, retang


def _unit(v):
    L = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5
    return (v[0] / L, v[1] / L, v[2] / L) if L > 1e-12 else None


def _perp(n):
    a = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0)
    t = (n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0])
    L = sum(x * x for x in t) ** 0.5 or 1.0
    return (t[0] / L, t[1] / L, t[2] / L)


class _Bones(object):
    """What `mdl_write` reads of a model: the bone list and nothing else."""

    def __init__(self, bones):
        self.bones = bones


class _Block(object):
    """A carried animation block addressed on its own, so `read_tracks` can decode one
    without the file it came out of.  `base` is 0 because the block starts at byte 0, which
    is also why the borrowed decode cache is good for one block only: reuse a `_Block` and
    every block after the first would read the first one's channels."""

    extract = M.Mdl.extract
    channel = M.Mdl.channel
    anim_channels = M.Mdl.anim_channels
    local_pose = M.Mdl.local_pose

    def __init__(self, data, bones):
        self.d = data
        self.bones = bones
        self._chan, self._chan_base, self._chan_table = {}, None, []


class _AnimHdr(object):
    """The `mstudioanimdesc_t` fields `read_tracks` reads, off a Rec's scalar bytes."""

    def __init__(self, r):
        self.base = 0
        self.name = r.name
        self.fps, self.flags, self.numframes = struct.unpack_from("<fii", r.raw, 0x04)
        self.movements = []


def add_animation(d, name, poses, fps=30.0, flags=0, movements=()):
    """`poses[frame][bone]` is (pos, quat) in parent-local space, or None to hold the bind
    pose.  The poses are kept, not encoded: `mstudiobone_t`'s scales are file-wide, so the
    channel a value needs cannot be chosen until every animation in the file is known.

    `movements` is `mstudiomovement_t` records, which the engine applies to the entity
    rather than to the skeleton.  Empty means the animation plays where it stands: leave
    the motion in the poses instead and the skeleton itself walks away from the origin, and
    snaps back when the sequence loops.
    """
    nb = len(d.bones)
    for f, frame in enumerate(poses):
        if len(frame) != nb:
            raise Refused("animation %r frame %d carries %d pose%s against the file's %d "
                          "bone%s; `quantise` indexes it per bone, so a short frame raises "
                          "an IndexError naming nothing and a long one is dropped without "
                          "a word"
                          % (name, f, len(frame), "" if len(frame) == 1 else "s",
                             nb, "" if nb == 1 else "s"))
    raw = bytearray(72)
    struct.pack_into("<f", raw, 0x04, fps)
    struct.pack_into("<i", raw, 0x08, flags)
    struct.pack_into("<i", raw, 0x0c, len(poses))
    d.anims.append(Rec(raw, name, None,
                       {"movements": [W.movement_bytes(x) for x in movements],
                        "block": b"", "poses": [list(f) for f in poses]}))
    return len(d.anims) - 1


def _skeleton(d):
    """The bone list `mdl_write` reads: index, bind pose and the two scales.

    Translation decodes additively over `pos` and rotation as a bare product, so the two
    conventions differ and `mdl_write.quantise` is the one place that knows it.
    """
    bones = []
    for k, r in enumerate(d.bones):
        b = M.Bone()
        b.index, b.name = k, r.name
        b.parent = struct.unpack_from("<i", r.raw, 0x04)[0]
        b.pos = list(struct.unpack_from("<3f", r.raw, 0x20))
        b.quat = list(struct.unpack_from("<4f", r.raw, 0x2c))
        b.posscale = list(struct.unpack_from("<3f", r.raw, 0x3c))
        b.rotscale = list(struct.unpack_from("<4f", r.raw, 0x48))
        b.posetobone = None
        b.flags = struct.unpack_from("<i", r.raw, 0x88)[0]
        bones.append(b)
    return _Bones(bones)


def quantise(d):
    """Fit every bone's scales to the widest authored pose, then encode.

    One scale per bone serves every animation in the file, so a per-animation fit would
    silently requantise the others -- which is why the poses are held until here.
    """
    pending = [r for r in d.anims if r.extra.get("poses")]
    if not pending:
        return 0
    m = _skeleton(d)
    old = W.file_scales(m)
    filled = []
    for r in pending:
        out = []
        for f in r.extra["poses"]:
            out.append([f[b.index] if f[b.index] is not None else (b.pos, b.quat)
                        for b in m.bones])
        filled.append(out)
    scales = W.fit_scales(m, filled)
    for k, br in enumerate(d.bones):
        struct.pack_into("<3f", br.raw, 0x3c, *scales[k][:3])
        struct.pack_into("<4f", br.raw, 0x48, *scales[k][3:])
        m.bones[k].posscale = scales[k][:3]
        m.bones[k].rotscale = scales[k][3:]
    for r, poses in zip(pending, filled):
        t = W.Tracks()
        t.name, t.numframes = r.name, len(poses)
        t.fps = struct.unpack_from("<f", r.raw, 0x04)[0]
        t.flags = struct.unpack_from("<i", r.raw, 0x08)[0]
        t.weights = dict((b.index, 1.0) for b in m.bones)
        t.chan = W.quantise(m, poses, scales)
        r.extra["block"] = W._anim_block(m, t)
        r.extra["block_bones"] = len(d.bones)
        r.extra["poses"] = None
    # One scale set serves the whole file, so widening it for a new pose leaves every
    # animation already in it decoding against scales it was not encoded for.
    if old != scales:
        fresh = set(id(r) for r in pending)
        for r in d.anims:
            if id(r) in fresh or not r.extra.get("block"):
                continue
            t = W.read_tracks(_Block(r.extra["block"], m.bones), _AnimHdr(r))
            W.rescale(t, old, scales)
            r.extra["block"] = W._anim_block(m, t)
            r.extra["block_bones"] = len(d.bones)
    return len(pending)


def _bone_vertex_boxes(d):
    """Per bone, the box of the vertices it carries, in that bone's own space.

    `posetobone` puts a bind-pose vertex where its bone can carry it, so the box is rigid:
    a frame moves it without changing its extents, and the sweep costs one 3x4 per bone per
    frame instead of one per vertex. It contains the skinned point set rather than equalling
    it -- a blended vertex is a convex combination of its bones' placements, so it stays
    inside their union. Measured against the per-vertex sweep on 108 frames of the andrei
    scratch model: conservative on every axis, widest by 3.15 units.
    """
    lo, hi = {}, {}
    stride = M.VERTEX_STRIDE[0]
    p2b = [struct.unpack_from("<12f", r.raw, 0x58) for r in d.bones]
    for bp in d.bodyparts:
        for mo in bp.kids:
            if struct.unpack_from("<i", mo.raw, 156)[0] != 0:
                continue
            vb = mo.extra.get("verts") or b""
            for o in range(0, len(vb) - stride + 1, stride):
                w = vb[o:o + 3]
                weights = (w[0], w[1], w[2], 255 - w[0] - w[1] - w[2])
                bones = struct.unpack_from("<4h", vb, o + 4)
                x, y, z = struct.unpack_from("<3f", vb, o + 12)
                # An unskinned vertex writes numbones 0 and rides bone 0 at full weight.
                for k in range(vb[o + 3] % 5 or 1):
                    if weights[k] <= 0 or not 0 <= bones[k] < len(p2b):
                        continue
                    t = p2b[bones[k]]
                    q = (t[0] * x + t[1] * y + t[2] * z + t[3],
                         t[4] * x + t[5] * y + t[6] * z + t[7],
                         t[8] * x + t[9] * y + t[10] * z + t[11])
                    b = bones[k]
                    if b not in lo:
                        lo[b], hi[b] = list(q), list(q)
                        continue
                    for c in range(3):
                        if q[c] < lo[b][c]:
                            lo[b][c] = q[c]
                        if q[c] > hi[b][c]:
                            hi[b][c] = q[c]
    return [(b, [(lo[b][c] + hi[b][c]) * 0.5 for c in range(3)],
             [(hi[b][c] - lo[b][c]) * 0.5 for c in range(3)]) for b in sorted(lo)]


def _pose_frames(r, m):
    """Parent-local poses per frame: the ones this pass authored, else the donor's own
    block decoded back.  It decodes through a `_Block` and not through the `Mdl` the export
    opened, because that one is wrapped by `_EditedBones` and would hand back the file's
    original bone list.  One `_Block` per animation -- the cache keys on `base`, 0 for every
    block."""
    poses = r.extra.get("poses")
    if poses:
        for f in poses:
            yield [f[b.index] if f[b.index] is not None else (b.pos, b.quat)
                   for b in m.bones]
        return
    blk = r.extra.get("block") or b""
    if not blk:
        return
    hdr = _AnimHdr(r)
    sh = _Block(blk, m.bones)
    for k in range(max(1, hdr.numframes)):
        yield sh.local_pose(hdr, k)


def _anim_boxes(d, carried, want=None):
    """{animation index: (min, max)} over every frame of every animation, authored this
    pass or decoded out of the donor.  `want` limits it to the animations some sequence is
    about to be stamped from, so an export with nothing to stamp decodes nothing.

    Root motion is not applied: `movements` carries the model away from the origin and the
    engine offsets the whole entity, so a sequence box that already included that offset
    would be counted twice.  `_AnimHdr` carries no movements, so a decoded animation gets
    that for free.  studiomdl does the same -- `extractLinearMotion` runs long before the
    sweep at `utils/studiomdl/simplify.cpp:5278`."""
    m = _skeleton(d)
    out = {}
    for i, r in enumerate(d.anims):
        if want is not None and i not in want:
            continue
        lo, hi = [float("inf")] * 3, [float("-inf")] * 3
        for local in _pose_frames(r, m):
            world = M.Mdl.world_matrices(m, local)
            for b, ctr, half in carried:
                t = world[b]
                for c in range(3):
                    row = t[c]
                    mid = (row[0] * ctr[0] + row[1] * ctr[1] + row[2] * ctr[2] + row[3])
                    ext = (abs(row[0]) * half[0] + abs(row[1]) * half[1]
                           + abs(row[2]) * half[2])
                    if mid - ext < lo[c]:
                        lo[c] = mid - ext
                    if mid + ext > hi[c]:
                        hi[c] = mid + ext
        if lo[0] <= hi[0]:
            out[i] = (lo, hi)
    return out


def _seq_anims(raw):
    """The blend table: `groupsize[0]*groupsize[1]` animation indices from +0x38, rows of
    0x20 bytes. A sequence built here is 1x1, a donor's can be a blend grid."""
    gx, gy = struct.unpack_from("<2i", raw, 0x23c)
    return [struct.unpack_from("<h", raw, 0x38 + x * 0x20 + y * 2)[0]
            for x in range(max(1, gx)) for y in range(max(1, gy))]


def stamp_sequence_boxes(d, force=False):
    """`mstudioseqdesc_t.bbmin`/`bbmax` @+0x1c/+0x28 -- the volume the engine widens the
    model's own bounds with, which `GetRenderBounds` returns on its own, and which an
    entity's collision radius is taken off, so zero there costs the radius and leaves the
    model drawn only while its origin is on screen. The union over the sequence's blend
    animations of the skinned vertex sweep over every frame
    (`utils/studiomdl/simplify.cpp:5276` and `:5349`).

    A zero or inverted box is replaced by the sweep.  `force` takes the valid ones too, for
    a donor whose geometry moved, but there it UNIONS rather than replaces: a shipped box
    is not reproducible from a shipped file -- `simplify.cpp:5314` sweeps source meshes
    with uncompressed poses, and only compiled LOD meshes and RLE blocks ship -- so it runs
    tens of units wide of this sweep, and replacing one would shrink the render bounds and
    the collision radius against geometry nothing here can see.  A zero box is replaced and
    not unioned, since unioning with an empty box at the origin would reach back to it.

    Returns how many sequences the bytes actually moved for.
    """
    if not d.seqs:
        return 0
    # Decided before anything is decoded, so an export with nothing to stamp pays for no
    # sweep at all.
    pending = {}
    for k, r in enumerate(d.seqs):
        box = struct.unpack_from("<6f", r.raw, 0x1c)
        valid = any(box) and all(box[c] <= box[c + 3] for c in range(3))
        if valid and not force:
            continue
        pending[k] = valid
    if not pending:
        return 0
    want = set()
    for k in pending:
        want.update(_seq_anims(d.seqs[k].raw))
    boxes = _anim_boxes(d, _bone_vertex_boxes(d), want)
    if not boxes:
        return 0
    n, first = 0, False
    for k, valid in sorted(pending.items()):
        r = d.seqs[k]
        want = _seq_anims(r.raw)
        cited = [boxes[i] for i in want if i in boxes]
        if len(cited) != len(want):
            continue
        lo = [min(b[0][c] for b in cited) for c in range(3)]
        hi = [max(b[1][c] for b in cited) for c in range(3)]
        if valid:
            was = struct.unpack_from("<6f", r.raw, 0x1c)
            lo = [min(lo[c], was[c]) for c in range(3)]
            hi = [max(hi[c], was[c + 3]) for c in range(3)]
        # bbmax at +0x28 follows bbmin, so the pair is one compare -- on the bytes, which
        # keeps a difference below float32 out of the count.
        new = struct.pack("<6f", *(lo + hi))
        if new == bytes(r.raw[0x1c:0x34]):
            continue
        r.raw[0x1c:0x34] = new
        n += 1
        first = first or k == 0
    if first:
        # studiomdl's default when the .qc has no $illumposition: the centre of sequence 0's
        # box, "Only use the 0th sequence; that should be the idle sequence"
        # (simplify.cpp:5379). It is the centre of hull_min/hull_max on 3845 of 4423 shipped
        # models only because the hull defaults to that same box.
        lo = struct.unpack_from("<3f", d.seqs[0].raw, 0x1c)
        hi = struct.unpack_from("<3f", d.seqs[0].raw, 0x28)
        struct.pack_into("<3f", d.hdr, 168, *[(a + b) * 0.5 for a, b in zip(lo, hi)])
    return n


def add_sequence(d, label, anim, activity=None, flags=0):
    raw = bytearray(764)
    struct.pack_into("<i", raw, 0x008, int(flags))
    struct.pack_into("<i", raw, 0x00c, -1)
    struct.pack_into("<i", raw, 0x010, 1)
    struct.pack_into("<i", raw, 0x034, 1)
    struct.pack_into("<h", raw, 0x038, anim)
    struct.pack_into("<2i", raw, 0x23c, 1, 1)
    struct.pack_into("<2i", raw, 0x244, -1, -1)
    struct.pack_into("<3f", raw, 0x264, 0.2, 0.2, 0.2)
    struct.pack_into("<i", raw, 0x2b8, -1)
    struct.pack_into("<f", raw, 0x2cc, FLT_MIN)
    struct.pack_into("<f", raw, 0x2d0, FLT_MAX)
    struct.pack_into("<3i", raw, 0x2d4, -1, -1, -1)
    struct.pack_into("<i", raw, 0x2e0, -1)
    struct.pack_into("<3f", raw, 0x2f0, 0.0, 1.0, 1.0)
    d.seqs.append(Rec(raw, label, None,
                      {"activity": activity, "events": [], "autolayers": [],
                       "knockbacks": [], "dodge": None, "block": None,
                       "seq2e8": None, "seq2ec": None}))
    return len(d.seqs) - 1


def encode_params(params, names, where=""):
    """[{name, start, end}] x2 -> (paramindex, paramstart, paramend) for one seqdesc.

    -1 is what 13724 of the 14012 shipped sequences carry at paramindex and the only
    value Studio_LocalPoseParameter short-circuits on, so an axis naming nothing gets
    it. A name the file does not carry is refused rather than written as -1: that
    function compares no index against numposeparameters in any of the three modules
    that carry it, so a wrong index reads past the array -- and 0 of 14012 shipped
    sequences name a pose parameter their model does not have.

    The array itself is carried through the rebuild and is not authored from a scene:
    what reads `move_yaw` and `hit_yaw` at runtime is not located, so an invented name
    would drive nothing.
    """
    idx, start, end = [-1, -1], [0.0, 0.0], [0.0, 0.0]
    for k, p in enumerate(list(params or ())[:2]):
        name = str(p.get("name") or "")
        if not name:
            continue
        if name not in names:
            raise Refused("%spose parameter %r on axis %d is not one the file carries -- "
                          "it has %s"
                          % (where, name, k, ", ".join(n for n in names if n) or "none"))
        idx[k] = names.index(name)
        start[k] = float(p.get("start") or 0.0)
        end[k] = float(p.get("end") or 0.0)
    return idx, start, end


def encode_events(events, where=""):
    """[{cycle, event, type, options}] -> the 76-byte mstudioevent_t records.

    Re-encoding the whole record is byte-safe and measured, not assumed: the four fields
    cover all 76 bytes, and of the 1872 event records in the 4445-model corpus 0 carry a
    byte after options' terminating NUL and 0 fill all 64 without one, so the zero
    padding reproduces every shipped record.

    Options past 63 bytes is refused rather than trimmed. The field has to hold a NUL for
    the engine to stop reading -- CBaseAnimating::HandleAnimEvent takes ids 2070 and 2071
    through LookupPhysicsChain and the rest through atoi -- and a silent trim is the kind
    of narrowing roadmap 4c rates worse than a refusal.
    """
    out = []
    for e in events or []:
        opt = str(e.get("options") or "").encode("latin1", "replace")
        if len(opt) > M.EVENT_OPTIONS_LEN - 1:
            raise Refused("%sevent %d options is %d bytes and the field holds %d plus a "
                          "terminator" % (where, int(e.get("event") or 0), len(opt),
                                          M.EVENT_OPTIONS_LEN - 1))
        out.append(struct.pack("<fii", float(e.get("cycle") or 0.0),
                               int(e.get("event") or 0), int(e.get("type") or 0))
                   + opt.ljust(M.EVENT_OPTIONS_LEN, bytes([0])))
    return out


def remove_animation(d, i):
    """Drop animation `i` and every sequence left with nothing to play.

    A sequence's blend grid is the only thing in the file that names an animation, so a
    delete costs that table and whatever the sequences it empties are themselves named by:
    autolayers are sequence indices and renumber behind them. A grid citing the dropped
    animation *beside* others is refused -- one corner of a blend needs a replacement this
    cannot invent -- and so is the last animation of a sequence group the file still has to
    have. Every count is a `len()` at emit, so nothing else moves.

    Returns (removed sequence indices, remaining animation count).
    """
    if not 0 <= i < len(d.anims):
        raise Refused("no animation %d to remove: the file has %d" % (i, len(d.anims)))
    name = d.anims[i].name
    doomed = []
    for k, r in enumerate(d.seqs):
        cites = _seq_anims(r.raw)
        if i not in cites:
            continue
        others = sorted(set(cites) - {i})
        if others:
            raise Refused("sequence %r blends animation %r with %d other%s, so removing it "
                          "would leave a hole in the blend grid"
                          % (r.name, name, len(others), "" if len(others) == 1 else "s"))
        doomed.append(k)
    gone = set(doomed)

    for k in reversed(doomed):
        del d.seqs[k]
    for r in d.seqs:
        gx, gy = struct.unpack_from("<2i", r.raw, 0x23c)
        for x in range(max(1, gx)):
            for y in range(max(1, gy)):
                at = 0x38 + x * 0x20 + y * 2
                a = struct.unpack_from("<h", r.raw, at)[0]
                if a > i:
                    struct.pack_into("<h", r.raw, at, a - 1)
        al = r.extra.get("autolayers") or []
        if al:
            r.extra["autolayers"] = [x - sum(1 for g in gone if g < x)
                                     for x in al if x not in gone]
    del d.anims[i]
    return doomed, len(d.anims)
