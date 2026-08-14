"""Every pointer field in a v2531 .mdl, with the base each one is measured from.

v2531 stores two kinds of offset.  A FILE-relative one (studiohdr_t's own *index fields,
and the cdtexture entries) breaks whenever its target moves.  A STRUCT-relative one breaks
only when the struct and its target land on opposite sides of the change.

This list is the single point of failure for any write path that moves bytes: a field
missing from it is copied verbatim and then points at whatever landed there instead.
`plans/mdl-reloc-audit.py` is the command-line front end.
"""

import struct

CLOTH_OFFS = (0x10, 0x20, 0x2c, 0x34, 0x40, 0x48, 0x50, 0x54, 0x58)

HDR_PTRS = [("boneindex", 244), ("bonecontrollerindex", 252), ("hitboxsetindex", 260),
            ("animdescindex", 268), ("seqindex", 276), ("seqgroupindex", 288),
            ("textureindex", 296), ("cdtextureindex", 304), ("skinindex", 316),
            ("bodypartindex", 324), ("attachmentindex", 332), ("transitionindex", 340),
            ("flexdescindex", 348), ("flexcontrollerindex", 356), ("flexruleindex", 364),
            ("ikchainindex", 372), ("mouthindex", 380), ("poseparamindex", 388),
            ("surfacepropindex", 392), ("springboneindex", 400), ("includemodelindex", 408)]

# A stale index with count 0 is not a pointer and must not be relocated.
HDR_COUNTED = {"boneindex": 240, "bonecontrollerindex": 248, "hitboxsetindex": 256,
               "animdescindex": 264, "seqindex": 272, "seqgroupindex": 284,
               "textureindex": 292, "cdtextureindex": 300, "skinindex": 308,
               "bodypartindex": 320, "attachmentindex": 328, "transitionindex": 336,
               "flexdescindex": 344, "flexcontrollerindex": 352, "flexruleindex": 360,
               "ikchainindex": 368, "mouthindex": 376, "poseparamindex": 384,
               "springboneindex": 396, "includemodelindex": 404}

# Marked unconfirmed in studio-verified.h; counted apart so a resize plan cannot rest on
# them silently.
UNVERIFIED = ("seqdesc.szseqname2e8", "seqdesc.szseqname2ec",
              "model.+0xcc", "model.+0xd4", "model.+0xdc", "animdesc.ikruleindex",
              "mesh.clothownerindex", "mesh.clothparticleindex", "mesh.clothnormalindex")

VERTEX_STRIDE = {0: 44, 1: 12, 2: 8}

# Struct-relative pointers the engine guards with `-1 < value`, so every negative reads as
# absent and the string must physically follow its seqdesc. 0 is unusable too: it passes the
# test and dereferences the seqdesc's own first byte. Read off
# Studio_ResolveSequenceActivities_vtmb, client.dll:0x10079520, which tests +0x2dc and +0x2e4
# that way; +0x2e8 and +0x2ec carry the same -1 sentinel and no reader has been found for
# them, so they are held to the same rule.
POSITIVE_ONLY = ("seqdesc.szdodgeactivity", "seqdesc.szblockactivity",
                 "seqdesc.szseqname2e8", "seqdesc.szseqname2ec")


class Audit(object):
    def __init__(self, b):
        self.b = b
        self.ptrs = []          # (field_addr, base, target, name, verified)

    def i(self, o):
        return struct.unpack_from("<i", self.b, o)[0]

    def rel(self, base, field_off, name, verified=True):
        """A STRUCT-relative pointer stored at base+field_off."""
        v = self.i(base + field_off)
        if v:
            self.ptrs.append((base + field_off, base, base + v, name, verified))

    def abs_(self, field_addr, name, verified=True):
        """A FILE-relative pointer stored at field_addr."""
        v = self.i(field_addr)
        if v:
            self.ptrs.append((field_addr, 0, v, name, verified))


def collect(b, cloth_vertex=True):
    """(audit, [(bodypart, model, model_off, numvertices, nummeshes)]).

    cloth_vertex registers mstudiomesh_t+0x30/+0x34/+0x38 mesh-relative.  Nothing in the
    file counts them and the engine reaches them only through the mesh, so they are the one
    class a relayout has to guess at; anomalies §M left it open.
    """
    a = Audit(b)
    for name, off in HDR_PTRS:
        c = HDR_COUNTED.get(name)
        if c is not None and not a.i(c):
            continue
        a.abs_(off, "studiohdr_t." + name)

    nbones, bi = a.i(240), a.i(244)
    for k in range(nbones):
        s = bi + k * 160
        a.rel(s, 0x00, "bone.sznameindex")
        a.rel(s, 0x98, "bone.surfacepropidx")
        if a.i(s + 0x8c):                       # proctype
            a.rel(s, 0x90, "bone.procindex")

    for k in range(a.i(256)):
        s = a.i(260) + k * 12
        a.rel(s, 0x00, "hitboxset.sznameindex")
        if a.i(s + 4):
            a.rel(s, 0x08, "hitboxset.hitboxindex")

    for k in range(a.i(264)):
        s = a.i(268) + k * 72
        a.rel(s, 0x00, "animdesc.sznameindex")
        if a.i(s + 0x10):
            a.rel(s, 0x14, "animdesc.movementindex")
        a.rel(s, 0x30, "animdesc.animindex")
        a.rel(s, 0x38, "animdesc.ikruleindex", False)
        blk = s + a.i(s + 0x30)
        for j in range(nbones):
            e = blk + j * 32
            for c in range(7):
                a.rel(e, 0x04 + c * 4, "anim.offset[%d]" % c)

    for k in range(a.i(272)):
        s = a.i(276) + k * 764
        a.rel(s, 0x000, "seqdesc.szlabelindex")
        a.rel(s, 0x004, "seqdesc.szactivitynameindex")
        if a.i(s + 0x014):
            a.rel(s, 0x018, "seqdesc.eventindex")
        # 7 sequences hold 764 at +0x294, the write cursor and not a count.
        if 0 < a.i(s + 0x294) < 64:
            a.rel(s, 0x298, "seqdesc.autolayerindex")
        # +0x2c0 is a write cursor and +0x2b8 is -1 on 12464 of 12471, so neither is a pointer.
        nkb = a.i(s + 0x2c4)
        if nkb > 0:
            a.rel(s, 0x2c8, "seqdesc.knockbackindex")
            kb = s + a.i(s + 0x2c8)
            for r in range(nkb):
                rec = kb + r * 188
                for g in range(4):
                    # count[] is stored negative; the loader stores abs() back
                    for j in range(min(abs(a.i(rec + 0x28 + g * 4)), 4)):
                        a.rel(rec, 0x78 + (g * 4 + j) * 4, "knockback.szactivity")
        for f, nm in ((0x2dc, "szdodgeactivity"), (0x2e4, "szblockactivity"),
                      (0x2e8, "szseqname2e8"), (0x2ec, "szseqname2ec")):
            if a.i(s + f) > 0:
                a.rel(s, f, "seqdesc." + nm)

    for k in range(a.i(292)):
        a.rel(a.i(296) + k * 20, 0x00, "texture.sznameindex")
    for k in range(a.i(300)):
        a.abs_(a.i(304) + k * 4, "cdtexture[] -> path string")
    for k in range(a.i(328)):
        a.rel(a.i(332) + k * 60, 0x00, "attachment.sznameindex")
    for k in range(a.i(384)):
        a.rel(a.i(388) + k * 20, 0x00, "poseparam.sznameindex")
    for k in range(a.i(404)):
        s = a.i(408) + k * 116
        a.rel(s, 0x00, "modelgroup.szlabelindex")
        # +0x10 -> mstudiobone_t[numbones] worth of 56-byte records the engine fills in at load;
        # 147 corpus gaps between consecutive blocks are all exactly numbones*56.
        a.rel(s, 0x10, "modelgroup.bonemapindex")

    for k in range(a.i(284)):
        s = a.i(288) + k * 16
        for f in (0x00, 0x04):
            if a.i(s + f):
                a.rel(s, f, "seqgroup.szname")
    for k in range(a.i(344)):
        a.rel(a.i(348) + k * 4, 0x00, "flexdesc.sznameindex")
    for k in range(a.i(352)):
        s = a.i(356) + k * 20
        a.rel(s, 0x00, "flexcontroller.sztypeindex")
        a.rel(s, 0x04, "flexcontroller.sznameindex")
    for k in range(a.i(360)):
        s = a.i(364) + k * 12
        if a.i(s + 0x04):
            a.rel(s, 0x08, "flexrule.opindex")
    for k in range(a.i(368)):
        s = a.i(372) + k * 16
        a.rel(s, 0x00, "ikchain.sznameindex")
        if a.i(s + 0x08):
            a.rel(s, 0x0c, "ikchain.linkindex")

    models = []
    for bp in range(a.i(320)):
        p = a.i(324) + bp * 16
        a.rel(p, 0x00, "bodypart.sznameindex")
        a.rel(p, 0x0c, "bodypart.modelindex")
        mi, nm = p + a.i(p + 0xc), a.i(p + 4)
        for md in range(nm):
            mo = mi + md * 224
            nmesh, nv = a.i(mo + 0x88), a.i(mo + 0x90)
            models.append((bp, md, mo, nv, nmesh))
            if nmesh:
                a.rel(mo, 0x8c, "model.meshindex")
            if nv:
                a.rel(mo, 0x94, "model.vertexindex")
                a.rel(mo, 0x98, "model.tangentsindex")
            if a.i(mo + 0xc0):
                a.rel(mo, 0xc4, "model.eyeballindex")
            for cnt, idx in ((0xc8, 0xcc), (0xd0, 0xd4), (0xd8, 0xdc)):
                if a.i(mo + cnt):
                    a.rel(mo, idx, "model.+0x%02x" % idx, False)
            # Two indirections past +0xcc: table entries are MODEL-relative, the payload offsets
            # inside each cloth object are OBJECT-relative.
            cols = a.i(mo + 0xc8)
            if cols and a.i(mo + 0xcc):
                tbl, lim = mo + a.i(mo + 0xcc), len(a.b)
                seen, k = [], 0
                while True:
                    eoff = tbl + k * 4
                    if eoff + 4 > lim:
                        break
                    v = a.i(eoff)
                    at = mo + v
                    if v and tbl < at and at + 0x5c <= lim:
                        off = [a.i(at + o) for o in CLOTH_OFFS]
                        if not any(x < 0 or at + x > lim for x in off):
                            seen.append(at)
                            a.rel(mo, eoff - mo, "model.clothtable[%d]" % k)
                            for o, x in zip(CLOTH_OFFS, off):
                                if x:
                                    a.rel(at, o, "cloth[%d].+0x%02x" % (k, o))
                    k += 1
                    # The table stores no length; the first object is what ends it.
                    if k > 512 or (seen and tbl + k * 4 >= min(seen)):
                        break
            mesh = mo + a.i(mo + 0x8c)
            for k in range(nmesh):
                s = mesh + k * 60
                a.rel(s, 0x04, "mesh.modelindex (negative)")
                if a.i(s + 0x10):
                    a.rel(s, 0x14, "mesh.flexindex")
                    fx = s + a.i(s + 0x14)
                    for r in range(a.i(s + 0x10)):
                        f = fx + r * 32
                        if a.i(f + 0x14):
                            a.rel(f, 0x18, "flex.vertindex")
                if cloth_vertex and a.i(s + 0x30):
                    a.rel(s, 0x30, "mesh.clothownerindex", False)
                    a.rel(s, 0x34, "mesh.clothparticleindex", False)
                    a.rel(s, 0x38, "mesh.clothnormalindex", False)
    return a, models


def shift_of(inserts, x):
    return sum(d for pos, d in inserts if x >= pos)


def vertex_inserts(a, models, k, nverts):
    """Both insert points in ORIGINAL coordinates, each carrying only its own delta.

    Some files write the tangent array ahead of the vertex array, so neither array's
    extent may be taken as the distance to the other.
    """
    bp, md, mo, nv, _ = models[k]
    if not nv:
        return None, "bp%d.m%d has no vertices" % (bp, md)
    stride = VERTEX_STRIDE.get(a.i(mo + 0x9c))
    if stride is None:
        return None, "bp%d.m%d unknown filetype %d" % (bp, md, a.i(mo + 0x9c))
    vend = mo + a.i(mo + 0x94) + nv * stride
    tend = mo + a.i(mo + 0x98) + nv * 16
    return [(vend, nverts * stride), (tend, nverts * 16)], None
