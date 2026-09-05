#!/usr/bin/env python3
"""Writer for the .vtx strip file that carries MDL v2531 triangles. No bpy.

The file is parsed into arrays and written back in the donor's own file order, so a
rewrite that changes nothing reproduces the bytes exactly. Every span no header points at
-- alignment slack, and in twelve corpus files a whole compiled-but-unreferenced LOD
payload -- is carried verbatim at its own place in that order.

Arrays may change length: offsets are recomputed from the new layout rather than patched,
which is what lets vertex and face counts move. Structure and evidence in
plans/vtx-format.md; the reader is vtx.py.
"""

import itertools
import struct

VTX_VERSION = 107

HEADER_SIZE = 36
BODYPART_STRIDE = 8
MODEL_STRIDE = 8
LOD_STRIDE = 12
MESH_STRIDE = 8
GROUP_STRIDE = 20
STRIP_STRIDE = 16
BSC_STRIDE = 4
MATREPL_STRIDE = 8
# Packed {short materialID; int nameOffset}, read out of studiorender.dll at 0x2c0138f2,
# 0x2c0138fc and 0x2c01390f.  Not 4, and not Valve's aligned 8.
MATREPL_ENTRY_STRIDE = 6

SG_IS_FLEXED = 0x01
SG_IS_HW_SKINNED = 0x02
# Troika's, absent from the 2003 tree. The shipped files set it only on cloth carriers and
# every cloth particle lands in one; what else it takes is unread.
SG_IS_CLOTH = 0x04
SG_VERTS_ARE_BONED = 0x08
SG_VERTS_ARE_PLAIN = 0x10

STRIP_IS_TRILIST = 0x01
STRIP_IS_TRISTRIP = 0x02

SHORT_MAX = 0x7fff


def _bone_map_table():
    """g_boneMapPermutationTable, StudioRender.dll+0x6ce30, as (live count, slot order).

    The arrangements P(4, k) for k = 0..4 in lexicographic order, each completed by its
    unused slots ascending -- byte-identical to the module's own 1300 bytes.
    """
    rows = []
    for k in range(5):
        for pick in itertools.permutations(range(4), k):
            rows.append((k, pick + tuple(x for x in range(4) if x not in pick)))
    return rows


BONE_MAP_TABLE = _bone_map_table()
BONE_MAP_ROW = {row: i for i, row in enumerate(BONE_MAP_TABLE)}
BONE_MAP_IDENTITY = {k: BONE_MAP_ROW[k, (0, 1, 2, 3)] for k in range(5)}


def bone_map_index(count, order=(0, 1, 2, 3)):
    """The entry binding `count` bones with hardware slot i taking .mdl weight order[i]."""
    try:
        return BONE_MAP_ROW[count, tuple(order)]
    except KeyError:
        raise ValueError("no table entry binds %d bones as %s" % (count, tuple(order)))


def bone_map_numbones(index):
    """How many bones the table entry at `index` binds. Raises outside the 65 entries."""
    if 0 <= index < len(BONE_MAP_TABLE):
        return BONE_MAP_TABLE[index][0]
    raise ValueError("boneMapIndex %d is past the 65-entry table" % index)


def _u(d, off, fmt):
    return struct.unpack_from("<" + fmt, d, off)


class Strip:
    __slots__ = ("num_indices", "first_index", "num_verts", "first_vert",
                 "numbones", "flags", "bone_state")

    def __init__(self, num_indices=0, first_index=0, num_verts=0, first_vert=0,
                 numbones=0, flags=STRIP_IS_TRILIST, bone_state=()):
        self.num_indices, self.first_index = num_indices, first_index
        self.num_verts, self.first_vert = num_verts, first_vert
        self.numbones, self.flags = numbones, flags
        self.bone_state = list(bone_state)


class Group:
    __slots__ = ("flags", "unk07", "verts", "indices", "strips")

    def __init__(self, flags=SG_VERTS_ARE_PLAIN, unk07=0, verts=b"", indices=(),
                 strips=()):
        self.flags, self.unk07 = flags, unk07
        self.verts = bytes(verts)
        self.indices = list(indices)
        self.strips = list(strips)

    @property
    def stride(self):
        return 12 if self.flags & SG_VERTS_ARE_BONED else 2

    @property
    def numverts(self):
        return len(self.verts) // self.stride

    def orig_vert_ids(self):
        s = self.stride
        return [_u(self.verts, i * s + s - 2, "h")[0] for i in range(self.numverts)]


class Mesh:
    __slots__ = ("flags", "groups")

    def __init__(self, flags=0, groups=()):
        self.flags, self.groups = flags, list(groups)


class Lod:
    __slots__ = ("switch_point", "meshes")

    def __init__(self, switch_point=0.0, meshes=()):
        self.switch_point, self.meshes = switch_point, list(meshes)


class Model:
    __slots__ = ("lods",)

    def __init__(self, lods=()):
        self.lods = list(lods)


class BodyPart:
    __slots__ = ("models",)

    def __init__(self, models=()):
        self.models = list(models)


class _Blob:
    __slots__ = ("old", "size", "new", "data", "old_size", "fresh")

    def __init__(self, old, size, old_size=None, fresh=False):
        self.old, self.size, self.new, self.data = old, size, None, None
        self.old_size = size if old_size is None else old_size
        self.fresh = fresh


class VtxFile:
    """A parsed .vtx that can be written back out."""

    def __init__(self, path=None, data=None):
        if data is None:
            with open(path, "rb") as f:
                data = f.read()
        self.path = path
        self.size = len(data)
        (self.version, self.vertcachesize, self.maxbones_strip, self.maxbones_tri,
         self.maxbones_vert, self.checksum, self.numlods, matreploffset,
         numbodyparts, bodypartoffset) = _u(data, 0, "iiHHiiiiii")
        if self.version != VTX_VERSION:
            raise ValueError("%s is .vtx version %d, not %d"
                             % (path, self.version, VTX_VERSION))

        self.bodyparts = []
        self.matrepl = []
        self.has_matrepl = bool(matreploffset)
        # Original offset of every array, so the write can reproduce the file order.
        # _rel keeps the offset field as written: an array whose count is 0 holds
        # studiomdl's write cursor there, not a pointer, and it has to survive.
        self._at = {}
        self._rel = {}
        # Keys whose donor count was 0. Their offset is a cursor, so one that gains
        # content cannot be laid out there -- another array already owns that span.
        self._empty = set()
        self._at["bodyparts"] = bodypartoffset
        self._at["matrepl"] = matreploffset

        for i in range(numbodyparts):
            bo = bodypartoffset + i * BODYPART_STRIDE
            nmodels, modeloffset = _u(data, bo, "2i")
            bp = BodyPart()
            self._put(("models", i), bo, modeloffset, nmodels)
            for j in range(nmodels):
                mo = bo + modeloffset + j * MODEL_STRIDE
                nlods, lodoffset = _u(data, mo, "2i")
                mdl = Model()
                self._put(("lods", i, j), mo, lodoffset, nlods)
                for l in range(nlods):
                    lo = mo + lodoffset + l * LOD_STRIDE
                    nmesh, meshoffset, switch = _u(data, lo, "2if")
                    lod = Lod(switch)
                    self._put(("meshes", i, j, l), lo, meshoffset, nmesh)
                    for k in range(nmesh):
                        eo = lo + meshoffset + k * MESH_STRIDE
                        ngroups, mflags, groupoffset = _u(data, eo, "hhi")
                        mesh = Mesh(mflags)
                        self._put(("groups", i, j, l, k), eo, groupoffset, ngroups)
                        for g in range(ngroups):
                            go = eo + groupoffset + g * GROUP_STRIDE
                            mesh.groups.append(
                                self._read_group(data, go, (i, j, l, k, g)))
                        lod.meshes.append(mesh)
                    mdl.lods.append(lod)
                bp.models.append(mdl)
            self.bodyparts.append(bp)

        if matreploffset:
            for l in range(self.numlods):
                ro = matreploffset + l * MATREPL_STRIDE
                nrepl, reploffset = _u(data, ro, "2i")
                self._put(("matrepl_entries", l), ro, reploffset, nrepl)
                self.matrepl.append(
                    data[ro + reploffset:
                         ro + reploffset + nrepl * MATREPL_ENTRY_STRIDE]
                    if nrepl else b"")

        self._donor_sizes = self._sizes()
        self._gaps = self._find_gaps(data)

    def _put(self, key, base, rel, count=1):
        self._at[key] = base + rel
        self._rel[key] = rel
        if not count:
            self._empty.add(key)

    def _read_group(self, data, go, key):
        numverts, numindices, numstrips, flags, unk07 = _u(data, go, "3hBB")
        voff, ioff, soff = _u(data, go + 8, "3i")
        if numverts < 0 or numindices < 0 or numstrips < 0:
            raise ValueError("strip group at %d has a negative count" % go)
        boned = bool(flags & SG_VERTS_ARE_BONED)
        if numverts > 0 and boned == bool(flags & SG_VERTS_ARE_PLAIN):
            raise ValueError("stripgroup flags %#04x sets neither vertex format nor "
                             "one alone" % flags)
        stride = 12 if boned else 2
        g = Group(flags, unk07)
        g.verts = data[go + voff:go + voff + numverts * stride]
        g.indices = list(_u(data, go + ioff, "%dH" % numindices)) if numindices else []
        self._put(("verts",) + key, go, voff, numverts)
        self._put(("indices",) + key, go, ioff, numindices)
        self._put(("strips",) + key, go, soff, numstrips)
        for s in range(numstrips):
            so = go + soff + s * STRIP_STRIDE
            ni, fi = _u(data, so, "2h")
            nv, fv = _u(data, so + 4, "2h")
            numbones, sflags = data[so + 8], data[so + 9]
            nbsc = _u(data, so + 10, "h")[0]
            bsco = _u(data, so + 12, "i")[0]
            st = Strip(ni, fi, nv, fv, numbones, sflags)
            self._put(("bsc",) + key + (s,), so, bsco, nbsc)
            if nbsc > 0:
                st.bone_state = [_u(data, so + bsco + r * BSC_STRIDE, "2h")
                                 for r in range(nbsc)]
            g.strips.append(st)
        return g

    # ---- layout -------------------------------------------------------------------

    def _sizes(self):
        """Every array the headers point at, as key -> byte length."""
        out = {"header": HEADER_SIZE,
               "bodyparts": len(self.bodyparts) * BODYPART_STRIDE}
        if self.has_matrepl:
            out["matrepl"] = self.numlods * MATREPL_STRIDE
            for l, blob in enumerate(self.matrepl):
                out["matrepl_entries", l] = len(blob)
        for i, bp in enumerate(self.bodyparts):
            out["models", i] = len(bp.models) * MODEL_STRIDE
            for j, mo in enumerate(bp.models):
                out["lods", i, j] = len(mo.lods) * LOD_STRIDE
                for l, lod in enumerate(mo.lods):
                    out["meshes", i, j, l] = len(lod.meshes) * MESH_STRIDE
                    for k, mesh in enumerate(lod.meshes):
                        out["groups", i, j, l, k] = len(mesh.groups) * GROUP_STRIDE
                        for g, grp in enumerate(mesh.groups):
                            key = (i, j, l, k, g)
                            out[("verts",) + key] = len(grp.verts)
                            out[("indices",) + key] = len(grp.indices) * 2
                            out[("strips",) + key] = len(grp.strips) * STRIP_STRIDE
                            for s, st in enumerate(grp.strips):
                                if st.bone_state:
                                    out[("bsc",) + key + (s,)] = \
                                        len(st.bone_state) * BSC_STRIDE
        return out

    def _find_gaps(self, data):
        """Spans no header points at, as (offset, bytes). Carried verbatim."""
        owned = bytearray(self.size)
        for key, size in self._donor_sizes.items():
            if not size:
                continue
            a = 0 if key == "header" else self._at.get(key)
            if a is None or a < 0 or a + size > self.size:
                continue
            owned[a:a + size] = b"\x01" * size
        gaps, x = [], 0
        while x < self.size:
            if owned[x]:
                x += 1
                continue
            y = x
            while y < self.size and not owned[y]:
                y += 1
            gaps.append((x, bytes(data[x:y])))
            x = y
        return gaps

    def _layout(self):
        """key -> new offset, keeping the donor's file order and its shared arrays."""
        sizes = self._sizes()
        blobs, by_span, order = {}, {}, []
        fresh = 0
        for key, size in sorted(sizes.items(), key=lambda kv: str(kv[0])):
            if not size:
                continue
            old = 0 if key == "header" else self._at.get(key)
            isfresh = old is None or key in self._empty
            if isfresh:
                # An array the donor did not have, or had empty -- a strip that gained
                # bone state changes. Nothing can follow it, so it goes past the last byte.
                fresh += 1
                old = self.size + fresh
            span = (old, size)
            blob = by_span.get(span)
            if blob is None:
                blob = by_span[span] = _Blob(
                    old, size, size if isfresh else self._donor_sizes.get(key, size),
                    isfresh)
                order.append(blob)
            blobs[key] = blob
        for old, payload in self._gaps:
            blob = _Blob(old, len(payload))
            blob.data = payload
            order.append(blob)
        order.sort(key=lambda b: (b.old, -b.size))
        # A span wholly inside another rides along in it: one shipped strip keeps its bone
        # state changes on the first, zero-valued entry of the material replacement list.
        # Tested with the DONOR's extents: with grown sizes two adjacent arrays look
        # nested, and the second is then packed on top of the first.
        laid, inner = [], []
        for blob in order:
            # A fresh blob has no donor extent, so `old` is a sentinel past the last byte
            # and `old_size` was 0: without this every one of them tests as nested and is
            # placed one byte after the last, on top of its neighbour.
            host = None if blob.fresh else next(
                (h for h in laid
                 if h.old <= blob.old
                 and blob.old + blob.old_size <= h.old + h.old_size), None)
            if host is None:
                laid.append(blob)
            else:
                inner.append((blob, host))
        cursor = 0
        for blob in laid:
            blob.new = cursor
            cursor += blob.size
        for blob, host in inner:
            blob.new = host.new + (blob.old - host.old)
        return {k: b.new for k, b in blobs.items()}, laid, cursor

    def check_relocatable(self):
        """Spans that overlap another span, which a relayout cannot preserve."""
        owned, bad = {}, []
        for key, size in self._sizes().items():
            if not size:
                continue
            a = 0 if key == "header" else self._at.get(key)
            if a is None:
                continue
            for x in range(a, a + size):
                other = owned.get(x)
                if other is not None and other != (a, size):
                    bad.append((key, a, size))
                    break
                owned[x] = (a, size)
        return bad

    # ---- write --------------------------------------------------------------------

    def to_bytes(self):
        at, order, total = self._layout()
        out = bytearray(total)

        def put(key, fmt, *vals):
            struct.pack_into("<" + fmt, out, key, *vals)

        def rel(key, base):
            return at[key] - base if key in at else self._rel.get(key, 0)

        struct.pack_into("<iiHHiiiiii", out, at["header"],
                         self.version, self.vertcachesize, self.maxbones_strip,
                         self.maxbones_tri, self.maxbones_vert, self.checksum,
                         self.numlods, at.get("matrepl", 0),
                         len(self.bodyparts), at["bodyparts"])

        for i, bp in enumerate(self.bodyparts):
            bo = at["bodyparts"] + i * BODYPART_STRIDE
            mbase = bo + rel(("models", i), bo)
            put(bo, "2i", len(bp.models), mbase - bo)
            for j, mo_ in enumerate(bp.models):
                mo = mbase + j * MODEL_STRIDE
                lbase = mo + rel(("lods", i, j), mo)
                put(mo, "2i", len(mo_.lods), lbase - mo)
                for l, lod in enumerate(mo_.lods):
                    lo = lbase + l * LOD_STRIDE
                    ebase = lo + rel(("meshes", i, j, l), lo)
                    put(lo, "2if", len(lod.meshes), ebase - lo, lod.switch_point)
                    for k, mesh in enumerate(lod.meshes):
                        eo = ebase + k * MESH_STRIDE
                        gbase = eo + rel(("groups", i, j, l, k), eo)
                        put(eo, "hhi", len(mesh.groups), mesh.flags, gbase - eo)
                        for g, grp in enumerate(mesh.groups):
                            self._write_group(out, rel, put, grp, gbase + g *
                                              GROUP_STRIDE, (i, j, l, k, g))

        if self.has_matrepl:
            rbase = at["matrepl"]
            for l in range(self.numlods):
                ro = rbase + l * MATREPL_STRIDE
                blob = self.matrepl[l] if l < len(self.matrepl) else b""
                n = len(blob) // MATREPL_ENTRY_STRIDE
                dst = ro + rel(("matrepl_entries", l), ro)
                put(ro, "2i", n, dst - ro)
                out[dst:dst + len(blob)] = blob

        for blob in order:
            if blob.data is not None:
                out[blob.new:blob.new + blob.size] = blob.data
        return bytes(out)

    def _write_group(self, out, rel, put, grp, go, key):
        stride = grp.stride
        nv, ni, ns = grp.numverts, len(grp.indices), len(grp.strips)
        for name, n in (("vertices", nv), ("indices", ni), ("strips", ns)):
            if n > SHORT_MAX:
                raise ValueError("strip group has %d %s; the count field is a short"
                                 % (n, name))
        vbase = go + rel(("verts",) + key, go)
        ibase = go + rel(("indices",) + key, go)
        sbase = go + rel(("strips",) + key, go)
        put(go, "3hBB", nv, ni, ns, grp.flags, grp.unk07)
        put(go + 8, "3i", vbase - go, ibase - go, sbase - go)
        out[vbase:vbase + nv * stride] = grp.verts
        if ni:
            struct.pack_into("<%dH" % ni, out, ibase, *grp.indices)
        for s, st in enumerate(grp.strips):
            so = sbase + s * STRIP_STRIDE
            for name, v in (("numIndices", st.num_indices),
                            ("indexOffset", st.first_index),
                            ("numVerts", st.num_verts),
                            ("vertOffset", st.first_vert)):
                if not -0x8000 <= v <= SHORT_MAX:
                    raise ValueError("strip %s is %d, outside a short" % (name, v))
            put(so, "4h", st.num_indices, st.first_index, st.num_verts, st.first_vert)
            out[so + 8] = st.numbones & 0xff
            out[so + 9] = st.flags & 0xff
            bsc = so + rel(("bsc",) + key + (s,), so)
            put(so + 10, "h", len(st.bone_state))
            put(so + 12, "i", bsc - so)
            for r, (hw, bone) in enumerate(st.bone_state):
                put(bsc + r * BSC_STRIDE, "2h", hw, bone)

    def write(self, path):
        data = self.to_bytes()
        with open(path, "wb") as f:
            f.write(data)
        return len(data)


# ---- building new geometry ------------------------------------------------------

def pack_vert0(orig_id):
    return struct.pack("<h", orig_id)


def pack_vert1(orig_id, bone_slots, max_per_vert=3):
    """A 12-byte record. `bone_slots` are hardware slots, in the .mdl's bone order.

    Slots the vertex does not use are 0 up to `maxBonesPerVert` and -1 past it, which is
    what the donors hold; only the first `BONE_MAP_IDENTITY` entries are ever read.
    """
    live = list(bone_slots)[:4]
    slots = (live + [0] * max(0, max_per_vert - len(live)))[:4]
    slots = (slots + [-1] * 4)[:4]
    return struct.pack("<5h", BONE_MAP_IDENTITY[len(live)], *slots) \
        + struct.pack("<h", orig_id)


MIN_BONE_INFLUENCE = 1.0


def reduce_bone_influence(bones, weights, max_bones):
    """Drop the lightest bone until few enough remain, giving up on a whole influence.

    studiomdl does this only for a hardware unflexed pass of a file that is not the
    fixed-function flavour, so the two-bone files keep every bone the .mdl gave them.
    """
    keep = list(range(len(bones)))
    while len(keep) > max_bones:
        j = min(range(len(keep)), key=lambda i: weights[keep[i]])
        if weights[keep[j]] >= MIN_BONE_INFLUENCE:
            break
        del keep[j]
    return [bones[i] for i in keep]


def assign_groups(triangles, flexed, vert_bones, vert_weights, max_tri, max_vert,
                  fixed_function=False, force_no_flex=False):
    """Which strip group each triangle belongs to, as studiomdl decides it.

    Four passes per mesh -- hardware then software, flexed then not -- each seeing only
    the triangles no earlier pass took, and an empty one dropped. A flexed pass caps at
    one bone per triangle and per vertex; an unflexed one takes `max_tri`/`max_vert` from
    the .vtx header. A hardware pass also wants the three corners within those caps.

    `flexed` is the mesh-local vertices some morph target names, `force_no_flex` the
    per-LOD facial-animation setting, which a donor states by having no flexed group.
    Returns [(is_hw, is_flexed, [triangle, ...]), ...] in that order, and raises if a
    triangle survives all four, which the software passes make impossible.
    """
    left = list(triangles)
    out = []
    for hw in (True, False):
        for fx in (True, False):
            cap_tri, cap_vert = (1, 1) if fx else (max_tri, max_vert)
            take, rest = [], []
            for tri in left:
                if (not force_no_flex and any(v in flexed for v in tri)) != fx:
                    rest.append(tri)
                    continue
                if hw:
                    per = []
                    for v in tri:
                        b = vert_bones[v]
                        if not fixed_function and not fx and len(b) > cap_vert:
                            b = reduce_bone_influence(b, vert_weights[v], cap_vert)
                        per.append(b)
                    if (len(set(x for b in per for x in b)) > cap_tri
                            or max(len(b) for b in per) > cap_vert):
                        rest.append(tri)
                        continue
                take.append(tri)
            left = rest
            if take:
                out.append((hw, fx, take))
    if left:
        raise ValueError("%d triangles fell out of every strip group pass" % len(left))
    return out


def partition(triangles, vert_bones=None, max_bones=16):
    """Triangles -> runs of (triangles, bones), each run within the hardware bone budget.

    `vert_bones` maps a group-local vertex to the model bones it is weighted to. Without
    it everything is one run carrying no bones, which is what a 2-byte group wants.
    """
    if not triangles:
        return []
    if vert_bones is None:
        return [(list(triangles), None)]
    runs, cur, bones = [], [], []
    for tri in triangles:
        # First-use order, not sorted: it is what the donors hold, and the slot number a
        # vertex records is this list's index.
        need = []
        for v in tri:
            for b in vert_bones.get(v, ()):
                if b not in need:
                    need.append(b)
        merged = bones + [b for b in need if b not in bones]
        if len(merged) > max_bones:
            if not cur:
                raise ValueError("one triangle needs %d bones, over the %d a strip can "
                                 "name" % (len(merged), max_bones))
            runs.append((cur, bones))
            cur, bones, merged = [], [], sorted(need)
        cur.append(tri)
        bones = merged
    if cur:
        runs.append((cur, bones))
    return runs


def rebuild_group(group, orig_ids, triangles, vert_bones=None, max_bones=16,
                  max_per_vert=3):
    """Replace one strip group's vertices, indices and strips.

    `orig_ids` is one entry per group-local vertex, indexing the owning mesh's vertex
    range in the .mdl; `triangles` are group-local triples, and a 12-byte group needs
    `vert_bones` giving each vertex its model bones. With SG_IS_HW_SKINNED the record
    stores slots only the strip's own bone state changes can resolve, which is why a
    vertex two strips share is emitted once per strip.
    """
    n = len(orig_ids)
    for tri in triangles:
        for v in tri:
            if not 0 <= v < n:
                raise ValueError("triangle names vertex %d of %d" % (v, n))
    # Ahead of the packing loop: past it, struct.error fires first and names the format
    # string rather than the mesh.
    top = max(orig_ids) if n else -1
    if top > SHORT_MAX:
        raise ValueError("origMeshVertID %d, and the field is a short: a mesh cannot "
                         "carry more than %d vertices. This strip group holds %d of "
                         "them" % (top, SHORT_MAX + 1, n))
    boned = bool(group.flags & SG_VERTS_ARE_BONED)
    hw = bool(group.flags & SG_IS_HW_SKINNED)
    if boned and vert_bones is None:
        raise ValueError("a 12-byte strip group needs vert_bones")
    if boned and hw:
        # studiomdl drops the .mdl's fourth bone rather than raising maxBonesPerVert.
        # The slot map has to lose it too, or the strip reserves a bone nothing reads.
        vert_bones = dict((v, list(b)[:max_per_vert]) for v, b in vert_bones.items())
    runs = partition(triangles, vert_bones if boned and hw else None, max_bones)

    indices, strips, recs = [], [], []
    for tris, bones in runs:
        slot = dict((b, i) for i, b in enumerate(bones)) if bones is not None else {}
        local, first, base = {}, len(indices), len(recs)
        for tri in tris:
            for v in tri:
                if v not in local:
                    local[v] = len(recs)
                    if not boned:
                        recs.append(pack_vert0(orig_ids[v]))
                    else:
                        # Never sorted: the table at +0x6ce30 pairs slot i with weight i.
                        # Without SG_IS_HW_SKINNED there are no slots -- these are bones.
                        ids = [slot[b] for b in vert_bones[v]] if hw else vert_bones[v]
                        recs.append(pack_vert1(orig_ids[v], ids, max_per_vert))
                indices.append(local[v])
        state = [(i, b) for i, b in enumerate(bones or ())]
        # numBones is bones per VERTEX, capped at MAX_NUM_BONES_PER_VERT -- not the
        # hardware bone count, which is numBoneStateChanges.
        per_vert = max([len(vert_bones[v]) for tri in tris for v in tri] or [0]) \
            if boned else 0
        strips.append(Strip(len(tris) * 3, first, len(recs) - base, base,
                            min(per_vert, 4), STRIP_IS_TRILIST, state))
    for name, c in (("vertices", len(recs)), ("indices", len(indices)),
                    ("strips", len(strips))):
        if c > SHORT_MAX:
            raise ValueError("%d %s in one strip group, split from %d mesh vertices; "
                             "the count field is a short, so the bound is %d"
                             % (c, name, n, SHORT_MAX))
    group.verts = b"".join(recs)
    group.indices = indices
    group.strips = strips
    return group
