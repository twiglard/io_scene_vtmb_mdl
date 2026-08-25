"""Re-emit a v2531 .mdl from its own parts, computing every offset rather than adjusting it.

The write path roadmap step 6 and anomalies §K describe: enumerate every span the format
code can attribute, lay those spans out again, and compute each pointer from where its
target landed.  A span nothing reaches is not carried and not blanked -- it is absent from
the output, and every offset that crossed it is a fresh number.

Three layout orders, and the difference between them is the whole point of having more
than one:

  file     ascending donor offset, dead spans closed up.  Least disturbance, so this is
           what a rebuild ships.
  reverse  every block after the header emitted back to front.
  shuffle  a fixed permutation, deterministic from the block count alone.

The last two are instruments, not products.  A struct-relative pointer this code does not
know about survives `file` order by luck -- its struct and its target shift by nearly the
same amount -- and cannot survive a permutation.  So `reverse` and `shuffle` verifying is
evidence that `relocs.collect` is complete, which nothing else here supplies, and a file
that verifies under `file` but not under a permutation names the gap.
"""

import bisect
import os
import struct
import sys

try:
    # The package copies, not fresh top-level ones: register()'s reload loop only reaches
    # these, so importing by bare name would pin whatever was on disk at first import.
    from . import checksum as CK
    from . import mdl as M
    from . import mdl_write as W
    from . import relocs as R
    from . import sections as S
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import checksum as CK
    import mdl as M
    import mdl_write as W
    import relocs as R
    import sections as S

ALIGN = 4
HDR_LENGTH = 140


class Refused(Exception):
    """The file holds something this cannot emit. Never degrade to a copy."""


def attributed(b):
    """Every span of `b` some section, animation run or flex payload accounts for.

    The same three sources `mdl-coverage.py` grades against, so a span it calls covered is
    one this carries and a hole it reports is one this drops.
    """
    _h, m = S.build(b, True, True)
    cov = [(o, o + s) for o, s, _n, _c in m.secs if s > 0]

    mdl = M.Mdl("<rebuild>", data=b)
    owned = W.owned_spans(mdl, S.Map(b).i(268))
    if owned:
        cov += [tuple(x) for x in W._extents(owned)]
    else:
        # Overlapping blocks leave no single owner, which stops the writer but not this:
        # a byte two animations both reach is still attributed.
        for a in mdl.anims:
            if a.numframes > 0:
                cov.append((a.base, W._block_extent(mdl, a)))
    cov += _vertanim(S.Map(b))
    cov += cloth_regions(S.Map(b))
    return cov


def cloth_regions(m):
    """One span per model covering its whole cloth region, holes included.

    Cloth stores none of its own lengths.  The row count is read back from how far the
    first object sits past its offset table, and cross-checked against the gap between a
    mesh's owner and particle arrays -- both positional.  So the region is rigid: moving
    anything inside it relative to anything else changes what the file says, and claiming
    it whole is what stops a relayout from doing that.
    """
    out = []
    for bp in range(m.i(320)):
        p = m.i(324) + bp * 16
        mi = p + m.i(p + 0xc)
        for md in range(m.i(p + 4)):
            mo = mi + md * 224
            tbl, rows, objs = S.cloth_objects(m, mo)
            if not objs:
                continue
            lo, hi = tbl, tbl + 4
            for at in objs:
                lo, hi = min(lo, at), max(hi, at + 0x5c)
                h = {o: m.i(at + o) for o in range(0, 0x5c, 4)}
                for off, cnts, w, _what in S.CLOTH_BLOCKS:
                    if h[off]:
                        lo = min(lo, at + h[off])
                        hi = max(hi, at + h[off] + sum(h[c] for c in cnts) * w)
            mesh = mo + m.i(mo + 0x8c)
            for k in range(m.i(mo + 0x88)):
                s = mesh + k * 60
                if not m.i(s + 0x30):
                    continue
                nv = m.i(s + 8) * rows
                for f, n in ((0x30, (nv + 3) // 4 * 4), (0x34, nv * 2), (0x38, nv * 2)):
                    lo = min(lo, s + m.i(s + f))
                    hi = max(hi, s + m.i(s + f) + n)
            out.append((lo, hi))
    return out


def _vertanim(a):
    """mstudioflex_t payloads: vertanimtype at +0x1c picks 8 or 20 bytes per record."""
    out = []
    for bp in range(a.i(320)):
        p = a.i(324) + bp * 16
        mi = p + a.i(p + 0xc)
        for md in range(a.i(p + 4)):
            mo = mi + md * 224
            mesh = mo + a.i(mo + 0x8c)
            for k in range(a.i(mo + 0x88)):
                s = mesh + k * 60
                for r in range(a.i(s + 0x10)):
                    f = s + a.i(s + 0x14) + r * 32
                    n, at = a.i(f + 0x14), f + a.i(f + 0x18)
                    if n > 0:
                        out.append((at, at + n * (8 if a.i(f + 0x1c) else 20)))
    return out


def blocks_of(b):
    """Attributed spans merged where they overlap, as the movable units of the rebuild.

    Only overlaps are merged, never abutting spans: fusing neighbours would grow one block
    until it swallowed the file and no permutation would move anything.
    """
    out = []
    for s, e in sorted(x for x in attributed(b) if x[1] > x[0]):
        s, e = max(s, 0), min(e, len(b))
        if e <= s:
            continue
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    if not out or out[0][0] != 0:
        raise Refused("no section at offset 0: the header is unattributed")
    return [(s, e) for s, e in out]


def _riffle(xs):
    """One perfect shuffle. A permutation for every length, which a modular stride is not."""
    h = (len(xs) + 1) // 2
    a, b = xs[:h], xs[h:]
    out = []
    for k in range(h):
        out.append(a[k])
        if k < len(b):
            out.append(b[k])
    return out


def _permute(n, order):
    """Emission order as block indices. Index 0 stays first: the header lives at offset 0."""
    if order == "file":
        return list(range(n))
    if order == "reverse":
        return [0] + list(range(n - 1, 0, -1))
    if order == "shuffle":
        rest = list(range(1, n))
        for _ in range(3):
            rest = _riffle(rest)
        return [0] + rest
    raise Refused("unknown layout order %r" % order)


class Plan(object):
    """Where every attributed byte of the donor ends up."""

    def __init__(self, blocks, order):
        self.blocks = blocks
        self.starts = [s for s, _e in blocks]
        self.new = [0] * len(blocks)
        at = self.pad = 0
        for k in _permute(len(blocks), order):
            s, e = blocks[k]
            aligned = (at + ALIGN - 1) // ALIGN * ALIGN
            self.pad += aligned - at
            self.new[k] = aligned
            at = aligned + (e - s)
        self.total = at

    def at(self, x, what):
        k = bisect.bisect_right(self.starts, x) - 1
        if k < 0:
            raise Refused("%s: offset %d precedes every section" % (what, x))
        s, e = self.blocks[k]
        if x > e:
            raise Refused("%s: offset %d falls in an unattributed span starting %d"
                          % (what, x, e))
        return self.new[k] + (x - s)


def cursor_fields(b):
    """Index fields whose count companion is zero, plus the two that never had one.

    studiomdl leaves its write cursor in these, so they are neither pointers nor derived
    and rule 3 of the contract writes them 0.  Carrying the donor's number instead would
    put a stale offset into a file where it addresses something else entirely.
    """
    a = R.Audit(b)
    out = []

    def pair(count_at, index_at):
        if not a.i(count_at):
            out.append(index_at)

    for name, off in R.HDR_PTRS:
        c = R.HDR_COUNTED.get(name)
        if c is not None:
            pair(c, off)

    for k in range(a.i(240)):
        s = a.i(244) + k * 160
        pair(s + 0x8c, s + 0x90)
    for k in range(a.i(256)):
        s = a.i(260) + k * 12
        pair(s + 0x04, s + 0x08)
    for k in range(a.i(264)):
        s = a.i(268) + k * 72
        pair(s + 0x10, s + 0x14)
    for k in range(a.i(272)):
        s = a.i(276) + k * 764
        pair(s + 0x14, s + 0x18)
        pair(s + 0x2c4, s + 0x2c8)
        # vampire.dll 103ea982 skips the melee tail unless numknockbacks >= 1, so
        # +0x2c0 addresses a hit-volume table only there; the 7 props that reach it
        # with none hold studiomdl's write cursor in both +0x2bc and +0x2c0.
        if a.i(s + 0x2c4) > 0:
            pair(s + 0x2bc, s + 0x2c0)
        else:
            out.append(s + 0x2c0)
        # 764 at +0x294 is the same write cursor as +0x18 and +0x298, not a count, and an
        # index with no readable count is worth less than no autolayers at all.
        if not 0 < a.i(s + 0x294) < 64:
            out += [s + 0x294, s + 0x298]
    for k in range(a.i(360)):
        s = a.i(364) + k * 12
        pair(s + 0x04, s + 0x08)
    for k in range(a.i(368)):
        s = a.i(372) + k * 16
        pair(s + 0x08, s + 0x0c)

    for bp in range(a.i(320)):
        p = a.i(324) + bp * 16
        mi = p + a.i(p + 0xc)
        for md in range(a.i(p + 4)):
            mo = mi + md * 224
            pair(mo + 0x88, mo + 0x8c)
            pair(mo + 0x90, mo + 0x94)
            pair(mo + 0x90, mo + 0x98)
            pair(mo + 0xc0, mo + 0xc4)
            pair(mo + 0xc8, mo + 0xcc)
            pair(mo + 0xd0, mo + 0xd4)
            pair(mo + 0xd8, mo + 0xdc)
            mesh = mo + a.i(mo + 0x8c)
            for k in range(a.i(mo + 0x88)):
                s = mesh + k * 60
                pair(s + 0x10, s + 0x14)
                if not a.i(s + 0x30):
                    out += [s + 0x30, s + 0x34, s + 0x38]
    return out


def rebuild(b, order="file", checksum=None):
    """(new bytes, stats). Raises Refused rather than emitting anything it cannot place.

    checksum None keeps the donor's value, which is what a rebuild wants: the `.vtx` beside
    it on disk still carries that number and the engine draws nothing if the two disagree.
    "compute" runs studiomdl's loop over the output and is only correct when the companion
    files are being emitted too; an int stamps that value.
    """
    if b[:4] != b"IDST":
        raise Refused("not IDST")
    blocks = blocks_of(b)
    plan = Plan(blocks, order)

    out = bytearray(plan.total)
    for (s, e), n in zip(blocks, plan.new):
        out[n:n + (e - s)] = b[s:e]

    audit, _models = R.collect(b)
    rewritten = 0
    for field, base, target, name, _verified in audit.ptrs:
        nf = plan.at(field, name + " field")
        nt = plan.at(target, name + " target")
        nb = plan.at(base, name + " base") if base else 0
        if nt - nb <= 0 and name in R.POSITIVE_ONLY:
            raise Refused("%s would be %d, and the engine reads anything below 1 there as "
                          "absent: its string has to land after the seqdesc" % (name, nt - nb))
        if struct.unpack_from("<i", out, nf)[0] != nt - nb:
            struct.pack_into("<i", out, nf, nt - nb)
            rewritten += 1

    cursors = cursor_fields(b)
    for field in cursors:
        struct.pack_into("<i", out, plan.at(field, "cursor field"), 0)

    struct.pack_into("<i", out, HDR_LENGTH, len(out))
    if checksum == "compute":
        ck = CK.stamp_mdl(out)
    elif checksum is not None:
        CK.stamp_at(out, CK.MDL_OFF, checksum)
        ck = CK.read_at(out, CK.MDL_OFF)
    else:
        ck = CK.read_at(out, CK.MDL_OFF)
    stats = {"blocks": len(blocks), "kept": sum(e - s for s, e in blocks),
             "dropped": len(b) - sum(e - s for s, e in blocks),
             "padding": plan.pad, "pointers": len(audit.ptrs),
             "rewritten": rewritten, "cursors": len(cursors), "checksum": ck}
    return bytes(out), stats


def _bone(x):
    return (x.name, x.parent, x.pos, x.quat, x.posscale, x.rotscale, x.posetobone, x.flags)


def _mesh(x):
    return (x.material, x.numvertices, x.vertexoffset, x.materialtype, x.materialparam)


def _model(x):
    return (x.name, x.nummeshes, x.numvertices, x.filetype,
            x.quant_offset, x.quant_scale, [_mesh(y) for y in x.meshes])


def _vert(v):
    return (v.pos, v.normal, v.uv, v.bones, v.weights, v.numbones)


def strings(b):
    """Every packed C string, in the order the pointers that name them are walked."""
    _h, m = S.build(b, True, True)
    return [bytes(b[o:o + n - 1]) for o, n in m.strs]


def cloth_layout(b):
    """What each cloth object's nine payload offsets resolve to, as bytes.

    The offsets themselves are excluded -- they are what a relayout changes. Nothing else
    here reads cloth, so without this a rebuild that left the second indirection stale
    would pass every check in the file (anomalies §M).
    """
    m = S.Map(b)
    out = []
    for bp in range(m.i(320)):
        p = m.i(324) + bp * 16
        mi = p + m.i(p + 0xc)
        for md in range(m.i(p + 4)):
            mo = mi + md * 224
            _tbl, rows, objs = S.cloth_objects(m, mo)
            for at in objs:
                h = {o: m.i(at + o) for o in range(0, 0x5c, 4)}
                scalars = tuple(v for o, v in sorted(h.items()) if o not in R.CLOTH_OFFS)
                payload = []
                for off, cnts, w, _what in S.CLOTH_BLOCKS:
                    if h[off]:
                        n = sum(h[c] for c in cnts) * w
                        payload.append(bytes(b[at + h[off]:at + h[off] + n]))
                out.append((rows, scalars, tuple(payload)))
            mesh = mo + m.i(mo + 0x8c)
            for k in range(m.i(mo + 0x88)):
                s = mesh + k * 60
                if not m.i(s + 0x30):
                    continue
                nv = m.i(s + 8) * rows
                out.append((bytes(b[s + m.i(s + 0x30):s + m.i(s + 0x30) + nv]),
                            bytes(b[s + m.i(s + 0x34):s + m.i(s + 0x34) + nv * 2]),
                            bytes(b[s + m.i(s + 0x38):s + m.i(s + 0x38) + nv * 2])))
    return out


def vertanims(b):
    """Every mstudioflex_t payload as bytes, in walk order.

    Nothing else here reads one.  `_vertanim` moves these spans, so without this a stale
    `flex.vertindex` passes every other comparison in the file while pointing at whatever
    landed where the payload used to be.
    """
    a = S.Map(b)
    out = []
    for at, end in _vertanim(a):
        out.append(bytes(b[at:end]))
    return out


def masked_sections(b):
    """Every sized array's bytes with the pointer fields blanked, in walk order.

    The semantic comparisons below name nine things; the section map sizes thirty-odd, and a
    stale pointer into any of the rest reads back as a perfectly good file.  Blanking exactly
    the fields a relayout is allowed to change leaves an invariant that holds for all of them
    at once: everything else must be identical.  String runs are excluded because their merge
    boundaries move legitimately, and `strings` already compares their contents.
    """
    _h, m = S.build(b, True, True)
    audit, _models = R.collect(b)
    d = bytearray(b)
    blank = set(f for f, _b, _t, _n, _v in audit.ptrs)
    blank.update(off for _n, off in R.HDR_PTRS)
    blank.update(cursor_fields(b))
    blank.add(HDR_LENGTH)
    for f in blank:
        if 0 <= f + 4 <= len(d):
            struct.pack_into("<i", d, f, 0)
    return [(n, bytes(d[o:o + s])) for o, s, n, _c in m.secs
            if s > 0 and n != "string run"]


def _span(b, at, n):
    """Bounded, so a stale offset compares unequal instead of raising."""
    if at < 0 or n < 0 or at + n > len(b):
        return None
    return bytes(b[at:at + n])


def melee(b):
    """Each sequence's hit-volume and knockback records, szactivity words blanked.

    A sequence is compared below as five scalars, so the whole melee tail rode on
    `masked_sections` happening to size it.  BUGS §17 is a writer that kept the record
    count at +0x2bc and wrote the offset at +0x2c0 as 0, which leaves the reader walking
    count*24 bytes from the seqdesc's own first byte.  Both halves are stated here: the
    count travels as itself and the offset as whatever it resolves to.  The offset word is
    excluded -- like every other index in the file it is what a relayout moves.
    """
    m = S.Map(b)
    out = []
    for k in range(m.i(272)):
        s = m.i(276) + k * 764
        nkb = m.i(s + 0x2c4)
        if nkb <= 0:
            # vampire.dll 103ea982 jumps past the whole tail, so neither word addresses
            # anything and the 7 props holding a write cursor in both are not a table.
            out.append(None)
            continue
        nhv = m.i(s + 0x2bc)
        hv = _span(b, s + m.i(s + 0x2c0), nhv * 24) if nhv > 0 else b""
        kb = s + m.i(s + 0x2c8)
        recs = []
        for j in range(nkb):
            rec = bytearray(_span(b, kb + j * 188, 188) or b"")
            for t in range(16 if len(rec) == 188 else 0):
                struct.pack_into("<i", rec, 0x78 + t * 4, 0)
            recs.append(bytes(rec))
        out.append((nhv, hv, tuple(recs)))
    return out


def verify(src, out, same_checksum=True):
    """Everything a reader can see must read back the same.

    Absolute positions are excluded on purpose -- they are what moved.
    """
    a = M.Mdl("<src>", data=src)
    g = M.Mdl("<out>", data=out)
    bad = []

    def cmp(what, x, y):
        if x != y:
            bad.append(what)

    if same_checksum:
        cmp("checksum", a.checksum, g.checksum)
    cmp("name", a.name, g.name)
    # Every name the engine resolves, not only the ones this reader parses: a struct-relative
    # string pointer left unrewritten still lands in the file, so only its text shows it.
    sa, sg = strings(src), strings(out)
    if sa != sg:
        n = sum(1 for x, y in zip(sa, sg) if x != y) + abs(len(sa) - len(sg))
        bad.append("%d of %d strings resolve differently" % (n, max(len(sa), len(sg))))
    cmp("bones", [_bone(x) for x in a.bones], [_bone(x) for x in g.bones])
    cmp("sequences", [(s.label, s.activity, s.flags, s.groupsize, s.blends) for s in a.seqs],
                     [(s.label, s.activity, s.flags, s.groupsize, s.blends) for s in g.seqs])
    cmp("melee tail", melee(src), melee(out))
    cmp("animdescs", [(x.name, x.fps, x.flags, x.numframes) for x in a.anims],
                     [(x.name, x.fps, x.flags, x.numframes) for x in g.anims])
    cmp("movements",
        [[(m.endframe, m.motionflags, m.v0, m.v1, m.angle, m.vector, m.position)
          for m in x.movements] for x in a.anims],
        [[(m.endframe, m.motionflags, m.v0, m.v1, m.angle, m.vector, m.position)
          for m in x.movements] for x in g.anims])
    for x, y in zip(a.anims, g.anims):
        if x.numframes <= 0:
            continue
        ta, tb = W.read_tracks(a, x), W.read_tracks(g, y)
        if ta.chan != tb.chan or ta.weights != tb.weights:
            bad.append("channels of %s" % x.name)
            break
    cmp("materials", a.materials, g.materials)
    cmp("include chain", a.includes, g.includes)
    cmp("bodyparts", [(p.name, [_model(m) for m in p.models]) for p in a.bodyparts],
                     [(p.name, [_model(m) for m in p.models]) for p in g.bodyparts])
    va = [_vert(v) for p in a.bodyparts for m in p.models for v in a.vertices(m)]
    vg = [_vert(v) for p in g.bodyparts for m in p.models for v in g.vertices(m)]
    cmp("vertices", va, vg)
    cmp("cloth", cloth_layout(src), cloth_layout(out))
    cmp("flex vertanim", vertanims(src), vertanims(out))
    ma, mg = masked_sections(src), masked_sections(out)
    if ma != mg:
        if len(ma) != len(mg):
            bad.append("%d sections against %d" % (len(ma), len(mg)))
        else:
            n = [x[0] for x, y in zip(ma, mg) if x != y]
            bad.append("%d section(s) differ: %s" % (len(n), sorted(set(n))[:4]))

    audit, _ = R.collect(out)
    oob = [nm for _f, _base, t, nm, _v in audit.ptrs if not 0 <= t <= len(out)]
    if oob:
        bad.append("%d pointer(s) out of bounds: %s" % (len(oob), sorted(set(oob))[:4]))
    return bad, len(va)
