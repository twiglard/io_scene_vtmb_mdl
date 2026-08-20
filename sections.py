"""Every array a v2531 .mdl holds, in file order, with the gaps between them.

Answers "what would have to move if this array changed size", and is what the rebuilder
enumerates to decide which bytes it is able to emit.  Entries whose stride is not
established print `?`, contribute no extent, and are therefore indistinguishable from a
hole -- the blind spot `plans/todo-vtmb-mdl-anomalies.md` §L records.

`plans/mdl-section-map.py` is the command-line front end.
"""

import struct

HDR = {"length": 140, "numbones": 240, "boneindex": 244,
       "numbonecontrollers": 248, "bonecontrollerindex": 252,
       "numhitboxsets": 256, "hitboxsetindex": 260,
       "numanim": 264, "animdescindex": 268, "numseq": 272, "seqindex": 276,
       "numseqgroups": 284, "seqgroupindex": 288,
       "numtextures": 292, "textureindex": 296,
       "numcdtextures": 300, "cdtextureindex": 304,
       "numskinref": 308, "numskinfamilies": 312, "skinindex": 316,
       "numbodyparts": 320, "bodypartindex": 324,
       "numattachments": 328, "attachmentindex": 332,
       "numflexdesc": 344, "flexdescindex": 348,
       "numflexcontrollers": 352, "flexcontrollerindex": 356,
       "numflexrules": 360, "flexruleindex": 364,
       "numikchains": 368, "ikchainindex": 372,
       "nummouths": 376, "mouthindex": 380,
       "numposeparameters": 384, "poseparamindex": 388, "surfacepropindex": 392,
       "numspringbones": 396, "springboneindex": 400,
       "numincludemodels": 404, "includemodelindex": 408}

HDR_SIZE = 424


class Map(object):
    def __init__(self, b):
        self.b = b
        self.secs = []
        self.strs = []

    def i(self, o):
        return struct.unpack_from("<i", self.b, o)[0]

    def add(self, name, off, n, stride):
        if n and off:
            self.secs.append((off, n * (stride or 0), name,
                              "%d x %s" % (n, stride if stride else "?")))

    def stringat(self, off):
        """Each string separately. A file can hold two string groups far apart with the mesh
        and vertex arrays between them, so one span from the lowest to EOF is not the table."""
        if 0 < off < len(self.b):
            end = self.b.find(b"\0", off)
            self.strs.append((off, (len(self.b) if end < 0 else end + 1) - off))

    def stringruns(self):
        out = []
        for off, n in sorted(self.strs):
            if out and off <= out[-1][0] + out[-1][1]:
                out[-1][1] = max(out[-1][1], off + n - out[-1][0])
            else:
                out.append([off, n])
        return out


# Not (tangentsindex - vertexindex) / numvertices: some files write the tangent array
# ahead of the vertex array, where that fit is negative.
VERTEX_STRIDE = {0: 44, 1: 12, 2: 8}

CLOTH_BLOCKS = ((0x10, (0x04,), 2, "particle->vertex u16"),
                (0x20, (0x18, 0x1c), 16, "mstudioclothspring_t"),
                (0x2c, (0x24, 0x28), 1, "spring batch count u8"),
                (0x34, (0x30,), 6, "tri 3x u16 particle"),
                (0x40, (0x38,), 4, "edge 2x u16 particle"),
                (0x48, (0x44,), 10, "face 2x u16 edge 3x u16 particle"),
                (0x50, (0x4c,), 12, "blend u16 a,b float w0,w1"),
                (0x54, (0x0c,), 16, "seam to row-1"),
                (0x58, (0x0c,), 16, "seam to row+1"))


def cloth_objects(m, mo):
    """(table offset, rows, [object offsets]) for one mstudiomodel_t, or (0, 0, [])."""
    cols = m.i(mo + 0xc8)
    if not cols or not m.i(mo + 0xcc):
        return 0, 0, []
    tbl, objs, k = mo + m.i(mo + 0xcc), [], 0
    while True:
        v = m.i(tbl + k * 4)
        # The table stores no length; the first object it points at bounds it.
        if v and tbl < mo + v < len(m.b) - 0x5c:
            objs.append(mo + v)
        k += 1
        if k > 512 or (objs and tbl + k * 4 >= min(objs)):
            break
    if not objs:
        return tbl, 0, []
    return tbl, ((min(objs) - tbl) // 4) // cols, objs


def cloth(m, mo, tag):
    """mstudiomodel_t+0xcc is a rows x cols table of offsets and +0xc8 is cols. Recon 29."""
    m.add("%s mstudioclothcollide_t[]" % tag, mo + m.i(mo + 0xd4), m.i(mo + 0xd0), 36)
    m.add("%s mstudioclothsphere_t[]" % tag, mo + m.i(mo + 0xdc), m.i(mo + 0xd8), 20)
    cols = m.i(mo + 0xc8)
    tbl, rows, objs = cloth_objects(m, mo)
    if not objs:
        return
    m.add("%s cloth offset table" % tag, tbl, rows * cols, 4)
    for at in objs:
        h = {o: m.i(at + o) for o in range(0, 0x5c, 4)}
        m.secs.append((at, 0x5c, "%s mstudiocloth_t" % tag, "1 x 92"))
        for off, cnts, w, what in CLOTH_BLOCKS:
            if h[off]:
                m.add("%s   cloth %s" % (tag, what), at + h[off],
                      sum(h[c] for c in cnts), w)
    mesh = mo + m.i(mo + 0x8c)
    for k in range(m.i(mo + 0x88)):
        s = mesh + k * 60
        if not m.i(s + 0x30):
            continue
        nv = m.i(s + 8) * rows
        m.add("%s.mesh%d cloth owner u8" % (tag, k), s + m.i(s + 0x30),
              (nv + 3) // 4 * 4, 1)
        m.add("%s.mesh%d cloth particle u16" % (tag, k), s + m.i(s + 0x34), nv, 2)
        # index into the cloth object's normals, the numblends extra ones appended past the
        # numparticles real ones; 0 on the vertices with none. Overruns by 138 B on one file.
        m.add("%s.mesh%d cloth normal u16" % (tag, k), s + m.i(s + 0x38), nv, 2)


def build(b, want_anims, want_seqs):
    m = Map(b)
    h = {k: m.i(o) for k, o in HDR.items()}
    m.secs.append((0, HDR_SIZE, "studiohdr_t", "1"))
    m.stringat(h["surfacepropindex"])

    m.add("mstudiobone_t[]", h["boneindex"], h["numbones"], 160)
    for k in range(h["numbones"]):
        s = h["boneindex"] + k * 160
        m.stringat(s + m.i(s))
        m.stringat(s + m.i(s + 0x98))
        # proctype is 1 on every one of the corpus's 3124 procedural bones; another
        # value carries a struct this does not know and must stay unsized.
        if m.i(s + 0x8c) == 1:
            m.add("  bone%d mstudioaxisinterpbone_t" % k, s + m.i(s + 0x90), 1, 176)
    m.add("mstudiobonecontroller_t[]", h["bonecontrollerindex"],
          h["numbonecontrollers"], 24)
    m.add("mstudiohitboxset_t[]", h["hitboxsetindex"], h["numhitboxsets"], 12)
    for k in range(h["numhitboxsets"]):
        s = h["hitboxsetindex"] + k * 12
        m.stringat(s + m.i(s))
        m.add("  mstudiobbox_t[] set%d" % k, s + m.i(s + 8), m.i(s + 4), 32)

    m.add("mstudioanimdesc_t[]", h["animdescindex"], h["numanim"], 72)
    for k in range(h["numanim"]):
        a = h["animdescindex"] + k * 72
        m.stringat(a + m.i(a))
        if want_anims:
            m.add("  anim%d mstudioanim_t[numbones]" % k, a + m.i(a + 0x30),
                  h["numbones"], 32)
            m.add("  anim%d mstudiomovement_t[]" % k, a + m.i(a + 0x14),
                  m.i(a + 0x10), 44)

    m.add("mstudioseqdesc_t[]", h["seqindex"], h["numseq"], 764)
    for k in range(h["numseq"]):
        s = h["seqindex"] + k * 764
        m.stringat(s + m.i(s))
        m.stringat(s + m.i(s + 4))
        # +0x2c4/+0x2c8 is numknockbacks/knockbackindex, recon 31. +0x2c0 is studiomdl's
        # write cursor, always (numseq-k)*764 plus what had been appended, and is not a pointer.
        nkb = m.i(s + 0x2c4)
        kb = s + m.i(s + 0x2c8)
        if nkb > 0:
            m.add("  seq%d mstudioknockback_t[]" % k, kb, nkb, 188)
            if 0 < kb and kb + nkb * 188 <= len(m.b):
                for r in range(nkb):
                    rec = kb + r * 188
                    for g in range(4):
                        # count[] is stored negative; the loader stores abs() back
                        for j in range(min(abs(m.i(rec + 0x28 + g * 4)), 4)):
                            m.stringat(rec + m.i(rec + 0x78 + (g * 4 + j) * 4))
        for f in (0x2dc, 0x2e4, 0x2e8, 0x2ec):
            if m.i(s + f) > 0:
                m.stringat(s + m.i(s + f))
        if want_seqs:
            m.add("  seq%d mstudioevent_t[]" % k, s + m.i(s + 0x18), m.i(s + 0x14), 76)
            # 7 scenery models hold 764 here, the same stale write cursor as +0x18 and +0x298,
            # rather than a count. Real counts are 0, 1 or 2 on the other 12464 sequences.
            if 0 < m.i(s + 0x294) < 64:
                m.add("  seq%d autolayer int[]" % k, s + m.i(s + 0x298), m.i(s + 0x294), 4)

    # 16, not the SDK's 48: 4412 of 4423 files start the bodypart array immediately after the
    # 16 meaningful bytes, and the 11 that do not carry 32 zero bytes (Valve's padding[32]).
    m.add("mstudioseqgroup_t[]", h["seqgroupindex"], h["numseqgroups"], 16)
    for k in range(h["numseqgroups"]):
        s = h["seqgroupindex"] + k * 16
        for f in (0, 4):
            if m.i(s + f):
                m.stringat(s + m.i(s + f))
    m.add("mstudioflexdesc_t[]", h["flexdescindex"], h["numflexdesc"], 4)
    for k in range(h["numflexdesc"]):
        s = h["flexdescindex"] + k * 4
        m.stringat(s + m.i(s))
    m.add("mstudioflexcontroller_t[]", h["flexcontrollerindex"],
          h["numflexcontrollers"], 20)
    for k in range(h["numflexcontrollers"]):
        s = h["flexcontrollerindex"] + k * 20
        m.stringat(s + m.i(s))
        m.stringat(s + m.i(s + 4))
    m.add("mstudioflexrule_t[]", h["flexruleindex"], h["numflexrules"], 12)
    for k in range(h["numflexrules"]):
        s = h["flexruleindex"] + k * 12
        m.add("  rule%d mstudioflexop_t[]" % k, s + m.i(s + 8), m.i(s + 4), 8)
    m.add("mstudioikchain_t[]", h["ikchainindex"], h["numikchains"], 16)
    for k in range(h["numikchains"]):
        s = h["ikchainindex"] + k * 16
        m.stringat(s + m.i(s))
        m.add("  chain%d mstudioiklink_t[]" % k, s + m.i(s + 12), m.i(s + 8), 28)
    m.add("mstudiomouth_t[]", h["mouthindex"], h["nummouths"], 20)

    m.add("mstudiobodyparts_t[]", h["bodypartindex"], h["numbodyparts"], 16)
    for bp in range(h["numbodyparts"]):
        p = h["bodypartindex"] + bp * 16
        m.stringat(p + m.i(p))
        nm, mi = m.i(p + 4), p + m.i(p + 0xc)
        m.add("  bp%d mstudiomodel_t[]" % bp, mi, nm, 224)
        for md in range(nm):
            mo = mi + md * 224
            nv, nmesh = m.i(mo + 0x90), m.i(mo + 0x88)
            vi, ti = mo + m.i(mo + 0x94), mo + m.i(mo + 0x98)
            mesh = mo + m.i(mo + 0x8c)
            m.add("    bp%d.m%d mstudiomesh_t[]" % (bp, md), mesh, nmesh, 60)
            ft = m.i(mo + 0x9c)
            m.add("    bp%d.m%d vertex[] filetype %d" % (bp, md, ft),
                  vi, nv, VERTEX_STRIDE.get(ft))
            m.add("    bp%d.m%d tangent[]" % (bp, md), ti, nv, 16)
            for k in range(nmesh):
                s = mesh + k * 60
                m.add("    bp%d.m%d.mesh%d mstudioflex_t[]" % (bp, md, k),
                      s + m.i(s + 0x14), m.i(s + 0x10), 32)
            m.add("    bp%d.m%d mstudioeyeball_t[]" % (bp, md),
                  mo + m.i(mo + 0xc4), m.i(mo + 0xc0), 140)
            cloth(m, mo, "    bp%d.m%d" % (bp, md))

    m.add("mstudiotexture_t[]", h["textureindex"], h["numtextures"], 20)
    for k in range(h["numtextures"]):
        s = h["textureindex"] + k * 20
        m.stringat(s + m.i(s))
    m.add("cdtexture int[]", h["cdtextureindex"], h["numcdtextures"], 4)
    for k in range(h["numcdtextures"]):
        m.stringat(m.i(h["cdtextureindex"] + k * 4))
    m.add("skin short[fam][ref]", h["skinindex"],
          h["numskinfamilies"] * h["numskinref"], 2)
    m.add("mstudioattachment_t[]", h["attachmentindex"], h["numattachments"], 60)
    for k in range(h["numattachments"]):
        s = h["attachmentindex"] + k * 60
        m.stringat(s + m.i(s))
    m.add("mstudioposeparamdesc_t[]", h["poseparamindex"], h["numposeparameters"], 20)
    for k in range(h["numposeparameters"]):
        s = h["poseparamindex"] + k * 20
        m.stringat(s + m.i(s))
    m.add("mstudiospringbone_t[]", h["springboneindex"], h["numspringbones"], 28)
    m.add("mstudiomodelgroup_t[]", h["includemodelindex"], h["numincludemodels"], 116)
    for k in range(h["numincludemodels"]):
        s = h["includemodelindex"] + k * 116
        m.stringat(s + m.i(s))
        m.add("  include%d bone map" % k, s + m.i(s + 0x10), h["numbones"], 56)

    for off, n in m.stringruns():
        m.secs.append((off, n, "string run", "packed C strings"))
    return h, m
