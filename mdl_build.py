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
as one rigid region with its pointers recomputed (`_cloth_region`).  Where a description
changes a cloth-bound mesh's vertex count the arrays cannot be resized, so that model's
cloth is dropped and named (`_drop_stale_cloth`).  `Desc.dropped` counts everything not
carried, and `emit` refuses unless the caller passes `drop=True`; over `gamedata\\models`
nothing populates it, all 3263 procedural bones being proctype 1.
"""

import os
import struct
import sys

try:
    from . import mdl as M
    from . import mdl_write as W
    from . import relocs as R
    from . import sections as S
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mdl as M
    import mdl_write as W
    import relocs as R
    import sections as S

HDR_SIZE = 424
ALIGN = 4

# A bone carrying no bit of the 0xfffc used-by mask gets no bone matrix, so anything skinned
# to it draws nothing.  0x10 is the corpus norm: 62439 of 62702 bones carry it.
BONE_USED = 0x10

# Every record ships as this, on all 32742 blocks of all 311 corpus models that carry an
# include -- one distinct value, no exceptions.  Nothing in the file reads it and the engine
# fills it at load, but the span has to be there: removing it crashed from a site with both
# __strcmpi operands wild.  Anomalies §L.
BONEMAP_RECORD = struct.pack("<ii", 0x0000FFFF, -1) + b"\0" * 48

# The scalars nothing has named, as the corpus states them.  A from-scratch caller starts
# from these; a caller re-authoring a shipped model overwrites them with that model's own.
DEFAULTS = {
    "hdr.unk144": (0.5, 0.5, 0.5),      # on 4416 of 4423 models
    "hdr.unk232": 0.0,
    "hdr.unk236": 0.0,
    "hdr.unhz": (0, 0, 1),              # on every model
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
            "slots": [(k, at - lo) for k, at in slots],
            "meshes": dict((k, ([x - lo for x in v], n)) for k, (v, n) in binds.items())}


def _drop_stale_cloth(d):
    """Cloth on a mesh whose vertex count moved is dropped, and the drop is reported.

    `mstudiomesh_t+0x30/+0x34/+0x38` are per-vertex arrays of `rows * numvertices` entries
    and the engine reads that many whatever the array's real length is, so a carried array
    against a changed count is read past its end -- anomalies §M, where a mis-sized region
    faulted in `Cloth_BuildSpringBatches_vtmb` on render rather than on load.  Growing them
    with zero entries is the better answer and cannot be written yet: neither the element
    order nor the meaning of a zero entry has been read out of the code, and `+0x38`'s
    element size is a fit that already fails on one shipped mesh.
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
        d.bonecontrollers.append(Rec(b[i(252) + k * 56:i(252) + (k + 1) * 56]))

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

    _simple(o, hdr, d.bonecontrollers, "bonecontrollers", 56, 248, 252)

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
    (0.5, 0.5, 0.5) on 4416 of 4423 models and `unhz` is (0, 0, 1) on every one.
    """
    d = Desc()
    d.name = name
    h = d.hdr
    struct.pack_into("<i", h, 0, 0x54534449)
    struct.pack_into("<i", h, 4, M.VERSION)
    struct.pack_into("<3f", h, 144, *DEFAULTS["hdr.unk144"])
    struct.pack_into("<3f", h, 180, -16.0, -16.0, 0.0)
    struct.pack_into("<3f", h, 192, 16.0, 16.0, 72.0)
    struct.pack_into("<3i", h, 412, *DEFAULTS["hdr.unhz"])
    d.surfaceprop = surfaceprop
    lab, nm = DEFAULTS["seqgroup"]
    d.seqgroups.append(Rec(bytearray(16), lab, None, {"name": nm}))
    d.faces = []
    return d


def add_bone(d, name, parent=-1, pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0),
             flags=BONE_USED, surfaceprop="flesh"):
    """Append one bone, return its index.  A parent must already be in `d.bones`: the
    engine walks the array once in order, so a child emitted first reads a world matrix
    that has not been computed."""
    if parent >= len(d.bones):
        raise Refused("bone %r cites parent %d, not emitted yet" % (name, parent))
    raw = bytearray(160)
    struct.pack_into("<i", raw, 0x04, parent)
    struct.pack_into("<6i", raw, 0x08, *([-1] * 6))
    struct.pack_into("<3f", raw, 0x20, *pos)
    struct.pack_into("<4f", raw, 0x2c, *quat)
    struct.pack_into("<3f", raw, 0x3c, 1.0 / 256, 1.0 / 256, 1.0 / 256)
    struct.pack_into("<4f", raw, 0x48, 1e-5, 1e-5, 1e-5, 1.0 / 32768)
    struct.pack_into("<i", raw, 0x88, flags)
    struct.pack_into("<i", raw, 0x94, -1)
    d.bones.append(Rec(raw, name, None, {"surfaceprop": surfaceprop}))
    _stamp_posetobone(d)
    return len(d.bones) - 1


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


def add_material(d, name, cdtexture="models/"):
    if not d.cdtextures:
        d.cdtextures = [cdtexture]
    d.textures.append(Rec(bytearray(20), name))
    d.skin = [list(range(len(d.textures)))]
    return len(d.textures) - 1


def _pack_verts(verts):
    """`mstudiovertex0_t[]` and its tangent array for one run of vertices."""
    vb, tb = bytearray(), bytearray()
    for pos, nrm, uv, binding in verts:
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
        t = _perp(nrm)
        tb += struct.pack("<4f", t[0], t[1], t[2], 1.0)
    return vb, tb


def add_model(d, meshes, bodypart="studio", model="model"):
    """One bodypart holding one model whose meshes share a vertex array.

    `meshes` is [(material, verts, faces), ...]; `verts` is
    [(pos, normal, uv, [(bone, weight), ...]), ...] and `faces` triples of indices into
    that mesh's own `verts`.  A mesh owns a contiguous run of the model's array, which is
    what `mstudiomesh_t.vertexoffset` addresses, so the runs concatenate in order.

    Only filetype 0 is emitted: 1 and 2 quantise through the model's
    quant_offset/quant_scale and filetype 2's scale reading is unconfirmed.
    """
    vb, tb, recs, face_lists, offset, radius = \
        bytearray(), bytearray(), [], [], 0, 0.0
    for material, verts, faces in meshes:
        pvb, ptb = _pack_verts(verts)
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
    without the file it came out of.  `base` is 0 because the block starts at byte 0."""

    extract = M.Mdl.extract

    def __init__(self, data, bones):
        self.d = data
        self.bones = bones


class _AnimHdr(object):
    """The `mstudioanimdesc_t` fields `read_tracks` reads, off a Rec's scalar bytes."""

    def __init__(self, r):
        self.base = 0
        self.name = r.name
        self.fps, self.flags, self.numframes = struct.unpack_from("<fii", r.raw, 0x04)
        self.movements = []


def add_animation(d, name, poses, fps=30.0, flags=0):
    """`poses[frame][bone]` is (pos, quat) in parent-local space, or None to hold the bind
    pose.  The poses are kept, not encoded: `mstudiobone_t`'s scales are file-wide, so the
    channel a value needs cannot be chosen until every animation in the file is known."""
    raw = bytearray(72)
    struct.pack_into("<f", raw, 0x04, fps)
    struct.pack_into("<i", raw, 0x08, flags)
    struct.pack_into("<i", raw, 0x0c, len(poses))
    d.anims.append(Rec(raw, name, None, {"movements": [], "block": b"",
                                         "poses": [list(f) for f in poses]}))
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
    return len(pending)


def add_sequence(d, label, anim, activity=None):
    raw = bytearray(764)
    struct.pack_into("<i", raw, 0x00c, -1)
    struct.pack_into("<i", raw, 0x010, 1)
    struct.pack_into("<i", raw, 0x034, 1)
    struct.pack_into("<h", raw, 0x038, anim)
    struct.pack_into("<2i", raw, 0x23c, 1, 1)
    struct.pack_into("<2f", raw, 0x264, 0.2, 0.2)
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
