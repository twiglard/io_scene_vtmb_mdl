#!/usr/bin/env python3
"""Reader and writer for the .phy collision file, and the ledge tree studiomdl builds. No bpy.

plans/phy-read.py is the check that drives it over the shipped corpus.

The layout is Valve's and Ipion's, read off source-2003: `public/phyfile.h`,
`vphysics/physics_collide.cpp` (`VCollideLoad`), `ivp/ivp_surface_manager/ivp_compact_surface.hxx`
and `ivp/ivp_collision/ivp_compact_ledge.hxx`. VTMB's `vphysics.dll` is IVP and carries the
`IVPS` tag.

    phyheader_t            16 B   int size (16), int id (0), int solidCount, long checkSum
    solidCount times:      int size, then `size` bytes of one IVP_Compact_Surface
    the rest:              keyvalue text, studiomdl's "solid {...}" blocks, then a NUL

An IVP_Compact_Surface is a 48-byte header and the ledges and ledge-tree nodes it points at,
every offset relative to the structure that holds it:

    surface  48 B  float mass_center[3], rotation_inertia[3], upper_limit_radius;
                   u32 max_factor_surface_deviation:8 | byte_size:24;
                   int offset_ledgetree_root; int dummy[3]        dummy[2] is "IVPS"
    node     28 B  int offset_right_node (0 = terminal), int offset_compact_ledge,
                   float center[3], radius; u8 box_sizes[3], free_0
                   the left son is the next node, `this + 1`
    ledge    16 B  int c_point_offset; int ledgetree_node_offset (client_data when it has children);
                   u32 has_children:2 | is_compact:2 | dummy:4 | size_div_16:24;
                   short n_triangles, for_future_use
                   then n_triangles triangles
    triangle 16 B  u32 tri_index:12 | pierce_index:12 | material_index:7 | is_virtual:1;
                   three edges, each u32 start_point_index:16 | opposite_index:15 (signed) |
                   is_virtual:1
    point    16 B  float x, y, z; one word IVP keeps client data in at run time

An edge's opposite_index counts 4-byte words from the edge itself, so a triangle header takes
a slot: edge e of triangle t is word 4t + 1 + e of the triangle array. Points are IVP metres in
IVP's axes, HL = (x, z, -y) / 0.0254.
"""

import decimal
import math
import struct

HEADER = struct.Struct("<iiil")
SURFACE = struct.Struct("<3f3ffIi12s")
NODE = struct.Struct("<ii4f3sB")
LEDGE = struct.Struct("<iiIhh")
TRIANGLE = struct.Struct("<4I")
POINT = struct.Struct("<3fI")

IVPS = b"IVPS"


class PhyError(ValueError):
    pass


class RagdollSplit(PhyError):
    """`drop_solids` would leave more solids with no parent than the file had."""


class Edge:
    __slots__ = ("start", "opposite", "virtual")

    def __init__(self, word):
        self.start = word & 0xffff
        o = (word >> 16) & 0x7fff
        self.opposite = o - 0x8000 if o & 0x4000 else o
        self.virtual = word >> 31

    def word(self):
        return self.start | ((self.opposite & 0x7fff) << 16) | (self.virtual << 31)


class Triangle:
    __slots__ = ("index", "pierce", "material", "virtual", "edges")

    def __init__(self, words):
        h = words[0]
        self.index = h & 0xfff
        self.pierce = (h >> 12) & 0xfff
        self.material = (h >> 24) & 0x7f
        self.virtual = h >> 31
        self.edges = [Edge(w) for w in words[1:]]

    def words(self):
        h = (self.index | (self.pierce << 12) | (self.material << 24)
             | (self.virtual << 31))
        return (h,) + tuple(e.word() for e in self.edges)


class Ledge:
    """One convex hull, or with `children` set, the hull over a subtree's ledges."""

    def __init__(self, buf, base, offset):
        if offset < 0 or base + offset + LEDGE.size > len(buf):
            raise PhyError(f"ledge at {offset} is outside the surface")
        (self.point_offset, self.node_offset, flags, self.n_triangles,
         self.future) = LEDGE.unpack_from(buf, base + offset)
        self.offset = offset
        self.children = flags & 3
        self.compact = (flags >> 2) & 3
        self.dummy = (flags >> 4) & 15
        self.size = (flags >> 8) * 16
        t0 = base + offset + LEDGE.size
        if self.n_triangles < 0 or t0 + self.n_triangles * TRIANGLE.size > len(buf):
            raise PhyError(f"ledge at {offset}: {self.n_triangles} triangles overrun")
        self.triangles = [Triangle(TRIANGLE.unpack_from(buf, t0 + TRIANGLE.size * i))
                          for i in range(self.n_triangles)]

    def point_indices(self):
        return {e.start for t in self.triangles for e in t.edges}

    def flags(self):
        return (self.children | (self.compact << 2) | (self.dummy << 4)
                | ((self.size // 16) << 8))


class Node:
    def __init__(self, buf=None, base=0, offset=0):
        self.left = self.right = None
        self.ledge = None
        self.offset = offset
        if buf is None:
            self.right_offset = self.ledge_offset = 0
            self.center, self.radius, self.box, self.free = (0.0, 0.0, 0.0), 0.0, (0, 0, 0), 0
            return
        if offset < 0 or base + offset + NODE.size > len(buf):
            raise PhyError(f"ledge-tree node at {offset} is outside the surface")
        r, l, cx, cy, cz, self.radius, box, self.free = NODE.unpack_from(buf, base + offset)
        self.right_offset = r
        self.ledge_offset = l
        self.center = (cx, cy, cz)
        self.box = tuple(box)


class Surface:
    def __init__(self, buf, base, size):
        if size < SURFACE.size:
            raise PhyError(f"solid of {size} B is smaller than its header")
        v = SURFACE.unpack_from(buf, base)
        self.mass_center = v[0:3]
        self.rotation_inertia = v[3:6]
        self.upper_limit_radius = v[6]
        self.max_deviation = v[7] & 0xff
        self.byte_size = v[7] >> 8
        self.root_offset = v[8]
        self.dummy = v[9]
        self.size = size
        view = memoryview(buf)[base:base + size]
        self.nodes = []
        self.ledges = []
        seen = {}
        self.root = self._node(view, self.root_offset, seen, 0)
        # File order, which differs from the tree's on 131 shipped surfaces.
        self.ledges.sort(key=lambda l: l.offset)
        # Every ledge of a surface points into one array: IVP pools the points of all its
        # ledges, and two ledges may share one.
        starts = {l.offset + l.point_offset for l in self.ledges}
        if len(starts) != 1:
            raise PhyError(f"ledges point at {len(starts)} point arrays")
        self.points_offset = starts.pop()
        n = 1 + max(i for l in self.ledges for i in l.point_indices())
        if self.points_offset < 0 or self.points_offset + n * POINT.size > size:
            raise PhyError(f"{n} points at {self.points_offset} overrun the surface")
        self.points = [POINT.unpack_from(view, self.points_offset + POINT.size * i)
                       for i in range(n)]

    def _node(self, view, offset, seen, depth):
        if offset in seen or depth > 64:
            raise PhyError(f"ledge tree revisits node {offset}")
        n = Node(view, 0, offset)
        seen[offset] = n
        self.nodes.append(n)
        if n.right_offset:
            n.left = self._node(view, offset + NODE.size, seen, depth + 1)
            n.right = self._node(view, offset + n.right_offset, seen, depth + 1)
        if n.ledge_offset:
            n.ledge = Ledge(view, 0, offset + n.ledge_offset)
            self.ledges.append(n.ledge)
        return n


class Phy:
    def __init__(self, data):
        if len(data) < HEADER.size:
            raise PhyError("shorter than its header")
        self.header_size, self.id, self.solid_count, self.checksum = HEADER.unpack_from(data)
        if self.header_size != HEADER.size or self.solid_count <= 0:
            raise PhyError(f"header size {self.header_size}, {self.solid_count} solids")
        pos = self.header_size
        self.solids = []
        for i in range(self.solid_count):
            if pos + 4 > len(data):
                raise PhyError(f"solid {i} starts past the end")
            size, = struct.unpack_from("<i", data, pos)
            if size <= 0 or pos + 4 + size > len(data):
                raise PhyError(f"solid {i} of {size} B overruns the file")
            self.solids.append(Surface(data, pos + 4, size))
            pos += 4 + size
        self.text = bytes(data[pos:])


def read(path):
    with open(path, "rb") as f:
        return Phy(f.read())


def surface_bytes(s):
    """An IVP_Compact_Surface with every offset recomputed from the layout.

    The layout is studiomdl's: the header, each ledge with its triangles in `s.ledges` order,
    the shared point array, then the nodes in preorder. Stored offsets are not consulted.
    """
    ledge_at = {}
    pos = SURFACE.size
    for l in s.ledges:
        ledge_at[id(l)] = pos
        pos += LEDGE.size + TRIANGLE.size * l.n_triangles
    points_at = pos
    pos += POINT.size * len(s.points)
    node_at = {}
    for n in _preorder(s.root):
        node_at[id(n)] = pos
        pos += NODE.size
    out = bytearray(pos)
    SURFACE.pack_into(out, 0, *s.mass_center, *s.rotation_inertia, s.upper_limit_radius,
                      s.max_deviation | (pos << 8), node_at[id(s.root)], s.dummy)
    for l in s.ledges:
        at = ledge_at[id(l)]
        LEDGE.pack_into(out, at, points_at - at, l.node_offset, l.flags(), l.n_triangles,
                        l.future)
        for i, t in enumerate(l.triangles):
            TRIANGLE.pack_into(out, at + LEDGE.size + TRIANGLE.size * i, *t.words())
    for i, p in enumerate(s.points):
        POINT.pack_into(out, points_at + POINT.size * i, *p)
    for n in _preorder(s.root):
        at = node_at[id(n)]
        right = node_at[id(n.right)] - at if n.right is not None else 0
        ledge = ledge_at[id(n.ledge)] - at if n.ledge is not None else 0
        NODE.pack_into(out, at, right, ledge, *n.center, n.radius, bytes(n.box), n.free)
    return bytes(out)


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


# IVP_COMPACT_BOUNDINGBOX_STEP_SIZE is (1.0f / 250.0f), yet the shipped trees were built with
# the double: rounded to a float it gets 228 of 7353 trees wrong.
STEP = 1.0 / 250.0
FLOAT_RES = _f32(1e-6)      # P_FLOAT_RES


class _Sphere:
    __slots__ = ("center", "radius", "box", "ledge")


def _sphere(lo, hi, ledge=None):
    s = _Sphere()
    s.center = [(hi[k] + lo[k]) * 0.5 for k in range(3)]
    s.radius = math.sqrt(sum((hi[k] - s.center[k]) ** 2 for k in range(3)))
    work = s.radius * STEP
    s.box = [int((hi[k] - s.center[k]) / work) + 1 for k in range(3)]
    s.ledge = ledge
    return s


def ledge_sphere(points, ledge):
    """`ledges_to_boxes_and_spheres`: the box over the ledge's points, the sphere round the box."""
    pts = [points[i] for i in ledge.point_indices()]
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    return _sphere(lo, hi, ledge)


def _bounding_box(spheres):
    # IVP_U_Float_Point extents: every bound is rounded to a float as it is stored.
    lo = [_f32(1e6)] * 3
    hi = [_f32(-1e6)] * 3
    for s in spheres:
        work = STEP * s.radius
        for k in range(3):
            a = s.center[k] - s.box[k] * work
            b = s.center[k] + s.box[k] * work
            if a < lo[k]:
                lo[k] = _f32(a)
            if b > hi[k]:
                hi[k] = _f32(b)
    return lo, hi


def _median_cut(spheres):
    if len(spheres) == 1:
        n = Node()
        s = spheres[0]
        n.ledge = s.ledge
    else:
        n = Node()
        s = _sphere(*_bounding_box(spheres))
        if len(spheres) == 2:
            n.left, n.right = _median_cut(spheres[:1]), _median_cut(spheres[1:])
        else:
            n.left, n.right = _median_cut_split(spheres)
    n.center = tuple(_f32(c) for c in s.center)
    n.radius = _f32(s.radius)
    n.box = tuple(s.box)
    return n


def _median_cut_split(spheres):
    left, right, volumes = [], [], []
    for axis in range(3):
        median = 0.0
        for s in spheres:
            median += s.center[axis]
        median /= len(spheres)
        lo, hi, toggle = [], [], True
        for x, s in enumerate(spheres):
            c = s.center[axis]
            if c < median - FLOAT_RES:
                lo.append(s)
            elif c > median + FLOAT_RES:
                hi.append(s)
            else:
                ref = spheres[x - 1 if x == len(spheres) - 1 else x + 1].center[axis]
                if ref < median - FLOAT_RES:
                    hi.append(s)
                elif ref > median + FLOAT_RES:
                    lo.append(s)
                elif not lo:
                    lo.append(s)
                elif not hi:
                    hi.append(s)
                else:
                    (lo if toggle else hi).append(s)
                    toggle = not toggle
        v = []
        for side in (lo, hi):
            a, b = _bounding_box(side)
            d = [_f32(abs(b[k] - a[k])) for k in range(3)]
            v.append(_f32(d[0] * d[1] * d[2]))
        left.append(lo)
        right.append(hi)
        volumes.append(_f32(v[1] + v[0]))
    # The best is recorded even where that axis split nothing off, and then only a later,
    # smaller axis with both sides filled is taken.
    chosen, best = 0, math.inf
    for axis in range(3):
        if volumes[axis] < best:
            if left[axis] and right[axis]:
                chosen = axis
            best = volumes[axis]
    return _median_cut(left[chosen]), _median_cut(right[chosen])


def build_tree(points, ledges):
    """The ledge tree studiomdl's IVP builds over `ledges`, in that order.

    `IVP_SurfaceBuilder_Ledge_Soup` with `IVP_CLUSTER_SHORTRANGE_TOPDOWN` and
    `IVP_CLUSTER_SHORTRANGE_BALANCED` defined: each node's box is the union of the boxes of
    the ledges under it, and a set splits at the mean of its centres on whichever axis leaves
    the two halves' boxes the least volume. `points` is what the ledges' point indices index;
    only x, y and z are read.
    """
    return _median_cut([ledge_sphere(points, l) for l in ledges])


def _preorder(n):
    out = []
    stack = [n]
    while stack:
        n = stack.pop()
        out.append(n)
        if n.right is not None:
            stack.append(n.right)
            stack.append(n.left)
    return out


def phy_bytes(f):
    """The whole file, each surface written by `surface_bytes` and the text half as read."""
    out = [HEADER.pack(HEADER.size, f.id, len(f.solids), f.checksum)]
    for s in f.solids:
        b = surface_bytes(s)
        out.append(struct.pack("<i", len(b)))
        out.append(b)
    out.append(f.text)
    return b"".join(out)


INCH = 0.0254


def ivp_to_hl(p):
    """IVP metres in IVP's axes to HL inches."""
    return (p[0] / INCH, p[2] / INCH, -p[1] / INCH)


def hl_to_ivp(p):
    return (p[0] * INCH, -p[2] * INCH, p[1] * INCH)


def ledge_triangles(ledge):
    """Each triangle's three start point indices, edge 0 first."""
    return [tuple(e.start for e in t.edges) for t in ledge.triangles]


def _cross(p0, p1, p2):
    """(p1 - p0) x (p2 - p0) in doubles: IVP's `calc_hesse(p0, p2, p1)`, outward on a ledge."""
    a0, a1, a2 = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    b0, b1, b2 = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    return (b1 * a2 - a1 * b2, b2 * a0 - a2 * b0, b0 * a1 - a0 * b1)


PIERCE_MAX = _f32(-1e-6)    # IVP_MAX_SCAL_VAL


def pierce_indices(points, tris):
    """`IVP_Object_Polygon_Tetra::insert_pierce_info` over `tris` in order: each triangle not
    yet paired takes the first whose unit normal has the least dot with its own, below
    -1e-6, and that one takes it back, overwriting whatever it held."""
    hesse = []
    for i0, i1, i2 in tris:
        n = _cross(points[i0], points[i1], points[i2])
        q = n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
        if q <= 0.0:
            raise PhyError(f"triangle {len(hesse)} has no area")
        f = 1.0 / math.sqrt(q)
        hesse.append((n[0] * f, n[1] * f, n[2] * f))
    pierced = [None] * len(tris)
    for i, (h0, h1, h2) in enumerate(hesse):
        if pierced[i] is not None:
            continue
        best, found = PIERCE_MAX, None
        for j, (g0, g1, g2) in enumerate(hesse):
            s = h0 * g0 + h1 * g1 + h2 * g2
            if s < best:
                best, found = s, j
        if found is None:
            raise PhyError(f"triangle {i} faces no other: the ledge is not a closed hull")
        pierced[i] = found
        pierced[found] = i
    return pierced


MAX_TRIANGLES = 8192    # IVP_MAX_TRIANGLES_PER_LEDGE


def opposite_indices(tris):
    """Each edge's `opposite_index`: the word of the edge running the other way, counted
    from this one, a triangle's header taking a word (`IVP_Compact_Ledge_Generator`)."""
    at = {}
    for i, t in enumerate(tris):
        for j in range(3):
            e = (t[j], t[(j + 1) % 3])
            if e in at:
                raise PhyError(f"edge {e} is used twice the same way")
            at[e] = i * 4 + j + 1
    out = []
    for i, t in enumerate(tris):
        row = []
        for j in range(3):
            o = at.get((t[(j + 1) % 3], t[j]))
            if o is None:
                raise PhyError(f"triangle {i} edge {j} has no opposite: the ledge is not closed")
            rel = o - (i * 4 + j + 1)
            if not -0x4000 <= rel < 0x4000:
                raise PhyError(f"triangle {i} edge {j}: opposite {rel} words away does not "
                               f"fit 15 bits")
            row.append(rel)
        out.append(row)
    return out


def _new_ledge(points, tris):
    """A compact ledge over `tris`, start point indices into the surface's pooled `points`."""
    if not 2 <= len(tris) <= MAX_TRIANGLES:
        raise PhyError(f"a ledge of {len(tris)} triangles")
    opp = opposite_indices(tris)
    pierce = pierce_indices(points, tris)
    l = Ledge.__new__(Ledge)
    l.point_offset = l.node_offset = 0
    l.offset = None
    l.children, l.compact, l.dummy, l.future = 0, 1, 0, 0
    l.n_triangles = len(tris)
    l.size = 16 + 16 * len(tris) + 16 * len({i for t in tris for i in t})
    l.triangles = []
    for k, t in enumerate(tris):
        tri = Triangle.__new__(Triangle)
        tri.index, tri.pierce, tri.material, tri.virtual = k, pierce[k], 0, 0
        tri.edges = []
        for j in range(3):
            e = Edge.__new__(Edge)
            e.start, e.opposite, e.virtual = t[j], opp[k][j], 0
            tri.edges.append(e)
        l.triangles.append(tri)
    return l


def _terminals(n):
    return [m.ledge for m in _preorder(n) if m.ledge is not None]


def surface_header(points, ledges):
    """(mass_center, rotation_inertia, upper_limit_radius, max_factor_surface_deviation) as
    `insert_radius_in_compact_surface` sets them; `ledges` in the tree's terminal order,
    which `get_all_ledges` returns and every loop here walks backwards."""
    pts = [p[:3] for p in points]
    order = list(reversed(ledges))
    sum_det = qsum = 0.0
    sm = [0.0, 0.0, 0.0]
    for l in order:
        for t in l.triangles:
            p0, p1, p2 = (pts[e.start] for e in t.edges)
            n = tuple(_f32(c) for c in _cross(p0, p1, p2))     # an IVP_U_Float_Point
            det = n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]
            qsum += n[0] * n[0] + n[1] * n[1] + n[2] * n[2]
            sum_det += det
            for p in (p0, p1, p2):
                for k in range(3):
                    sm[k] += p[k] * det * 0.25
    surface = math.sqrt(qsum)
    if sum_det > surface * surface * surface * 1e-9:
        center = [c / sum_det for c in sm]
        a, b, c = (v * v for v in _second_moments(pts, order, center))
        inertia = (math.sqrt(b + c), math.sqrt(a + c), math.sqrt(a + b))
    else:
        used = [pts[e.start] for l in ledges for t in l.triangles for e in t.edges]
        lo = [min(p[k] for p in used) for k in range(3)]
        hi = [max(p[k] for p in used) for k in range(3)]
        center = [(lo[k] + hi[k]) * 0.5 for k in range(3)]
        r = 0.5 * math.sqrt(sum((hi[k] - lo[k]) ** 2 for k in range(3)))
        inertia = (r * r * 0.5,) * 3
    q_max = dev_max = 0.0
    for l in order:
        for t in l.triangles:
            p0, p1, p2 = (pts[e.start] for e in t.edges)
            h = _cross(p0, p1, p2)
            ih = 1.0 / (h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
            for p in (p0, p1, p2):
                d = (p[0] - center[0], p[1] - center[1], p[2] - center[2])
                q_max = max(q_max, d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                c = (d[1] * h[2] - d[2] * h[1], d[2] * h[0] - d[0] * h[2],
                     d[0] * h[1] - d[1] * h[0])
                dev_max = max(dev_max, (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) * ih)
    radius, dev = math.sqrt(q_max), math.sqrt(dev_max)
    return (tuple(_f32(c) for c in center), tuple(_f32(c) for c in inertia), _f32(radius),
            int(1.0 + dev / (radius * _f32(1.0 / 250.0))))


def _second_moments(pts, ledges, center):
    """The volume's mean squared distance from `center` along each axis, what
    `IVP_Rot_Inertia_Solver::find_center_given_xyz` integrates edge by edge. Summed here
    over the tetrahedra the triangles make with the centre, exactly. The stored inertia
    differs from this by over 1e-5 on 464 of the 7353 shipped surfaces, up to 0.52; the
    2003 code ported line by line misses as often, mostly on the same ones, so what made
    those values is unknown."""
    vol, m = 0.0, [0.0, 0.0, 0.0]
    for l in ledges:
        for t in l.triangles:
            a, b, d = ([pts[e.start][i] - center[i] for i in range(3)] for e in t.edges)
            v = (a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0])
                 + a[2] * (b[0] * d[1] - b[1] * d[0])) / 6.0
            vol += v
            for k in range(3):
                m[k] += v / 10.0 * (a[k] * a[k] + b[k] * b[k] + d[k] * d[k]
                                    + a[k] * b[k] + a[k] * d[k] + b[k] * d[k])
    return [x / vol if vol >= 1e-19 else 1.0 for x in m]


def build_surface(ledges, tag=IVPS):
    """A compact surface over `ledges`, each a list of triangles, each three (x, y, z) points
    in IVP metres wound counter-clockwise seen from outside, as a ledge must be closed and
    convex. Points pool across the ledges in first-use order, equal coordinates shared, as
    `IVP_SurfaceBuilder_Ledge_Soup` pools them; triangles keep the order given."""
    points, at, tri_sets = [], {}, []
    for tris in ledges:
        rows = []
        for tri in tris:
            row = []
            for p in tri:
                key = (_f32(p[0]), _f32(p[1]), _f32(p[2]))
                if key not in at:
                    at[key] = len(points)
                    points.append(key + (0,))
                row.append(at[key])
            rows.append(tuple(row))
        tri_sets.append(rows)
    if len(points) > 0x10000:
        raise PhyError(f"{len(points)} points do not fit a 16-bit start index")
    s = Surface.__new__(Surface)
    s.points = points
    s.ledges = [_new_ledge(points, rows) for rows in tri_sets]
    s.root = build_tree(points, s.ledges)
    (s.mass_center, s.rotation_inertia, s.upper_limit_radius,
     s.max_deviation) = surface_header(points, _terminals(s.root))
    if not 0 < s.max_deviation < 256:
        raise PhyError(f"surface deviation step {s.max_deviation} does not fit a byte")
    s.dummy = bytes(8) + tag
    return s


def surface_volume(s):
    """The surface's volume in cubic inches, summed over its ledges' triangles."""
    v = 0.0
    for l in s.ledges:
        for t in l.triangles:
            a, b, c = (ivp_to_hl(s.points[e.start]) for e in t.edges)
            v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                  + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    return v


def convex_hull(points):
    """Triangles (i, j, k) indexing `points`, counter-clockwise seen from outside, over their
    hull; a point inside or on it is left out, and of equal points the first is used.

    Quickhull in exact integer arithmetic (each float scaled by one common power of two), so
    every side-of-plane test is exact; hull the coordinates the file will hold. A face whose
    plane holds the new point is replaced too, so no triangle has a point on the line
    through a hull edge. A flat set gives IVP's two-sided ledge (`_flat_hull`). Raises on
    fewer than three distinct points, a set on one line, or a result some point lies outside
    of."""
    first, pts = {}, []
    for i, p in enumerate(points):
        key = (float(p[0]), float(p[1]), float(p[2]))
        if key not in first:
            first[key] = i
            pts.append(key)
    index = list(first.values())
    if len(pts) < 3:
        raise PhyError(f"{len(pts)} distinct points make no hull")
    ratios = [x.as_integer_ratio() for p in pts for x in p]
    den = max(d for _, d in ratios)
    flat = [n * (den // d) for n, d in ratios]
    P = [tuple(flat[3 * i:3 * i + 3]) for i in range(len(pts))]

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def crs(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    ax = max(range(3), key=lambda k: max(q[k] for q in P) - min(q[k] for q in P))
    i0 = min(range(len(P)), key=lambda i: P[i][ax])
    i1 = max(range(len(P)), key=lambda i: P[i][ax])
    d01 = sub(P[i1], P[i0])
    i2 = max(range(len(P)), key=lambda i: dot(crs(d01, sub(P[i], P[i0])),
                                              crs(d01, sub(P[i], P[i0]))))
    n = crs(d01, sub(P[i2], P[i0]))
    if n == (0, 0, 0):
        raise PhyError("every point is on one line")
    i3 = max(range(len(P)), key=lambda i: abs(dot(n, sub(P[i], P[i0]))))
    if dot(n, sub(P[i3], P[i0])) == 0:
        return [tuple(index[i] for i in t) for t in _flat_hull(P, n)]

    # faces[f] = [(a, b, c), normal, offset, outside points]; a point q is outside when
    # dot(normal, q) > offset.
    faces = {}
    edge_face = {}
    serial = [0]

    def make(a, b, c):
        nn = crs(sub(P[b], P[a]), sub(P[c], P[a]))
        f = serial[0]
        serial[0] += 1
        faces[f] = [(a, b, c), nn, dot(nn, P[a]), []]
        for e in ((a, b), (b, c), (c, a)):
            edge_face[e] = f
        return f

    def outward(a, b, c, d):
        if dot(crs(sub(P[b], P[a]), sub(P[c], P[a])), sub(P[d], P[a])) < 0:
            return a, b, c
        return a, c, b

    new = [make(*outward(i0, i1, i2, i3)), make(*outward(i0, i1, i3, i2)),
           make(*outward(i0, i2, i3, i1)), make(*outward(i1, i2, i3, i0))]

    def assign(idx, cands):
        for i in idx:
            for f in cands:
                fc = faces[f]
                if dot(fc[1], P[i]) > fc[2]:
                    fc[3].append(i)
                    break

    assign([i for i in range(len(P)) if i not in (i0, i1, i2, i3)], new)
    todo = list(new)
    while todo:
        f = todo.pop()
        if f not in faces or not faces[f][3]:
            continue
        fc = faces[f]
        p = max(fc[3], key=lambda i: dot(fc[1], P[i]))
        visible, stack = {f}, [f]
        while stack:
            a, b, c = faces[stack.pop()][0]
            for e in ((b, a), (c, b), (a, c)):
                h = edge_face[e]
                if h not in visible and dot(faces[h][1], P[p]) >= faces[h][2]:
                    visible.add(h)
                    stack.append(h)
        horizon, orphans = [], []
        for g in visible:
            a, b, c = faces[g][0]
            for e in ((a, b), (b, c), (c, a)):
                if edge_face[(e[1], e[0])] not in visible:
                    horizon.append(e)
            orphans += faces[g][3]
        if len({a for a, _ in horizon}) != len(horizon):
            raise PhyError(f"the faces point {index[p]} sees do not make one patch")
        for g in visible:
            a, b, c = faces[g][0]
            for e in ((a, b), (b, c), (c, a)):
                if edge_face.get(e) == g:
                    del edge_face[e]
            del faces[g]
        new = [make(a, b, p) for a, b in horizon]
        assign([i for i in orphans if i != p], new)
        todo += new
    out = []
    for f in sorted(faces):
        fc = faces[f]
        if fc[1] == (0, 0, 0):
            raise PhyError(f"hull face {fc[0]} has no area")
        for e in ((fc[0][0], fc[0][1]), (fc[0][1], fc[0][2]), (fc[0][2], fc[0][0])):
            if (e[1], e[0]) not in edge_face:
                raise PhyError(f"hull edge {e} has no face across it")
        for i, q in enumerate(P):
            if dot(fc[1], q) > fc[2]:
                raise PhyError(f"point {index[i]} lies outside hull face {fc[0]}")
        out.append(tuple(index[i] for i in fc[0]))
    return out


def _flat_hull(P, n):
    """IVP's ledge for a flat set: qhull fails on it, the retry with `QJ` joggles the points
    into a hull, and the ledge keeps the points unjoggled, so it is the outline triangulated
    once facing `n` and once facing away, 2V - 4 triangles. Three points give the same two
    triangles `convert_triangle_to_compace_ledge` writes. Exact: the outline is taken in
    the two axes `n` leans least along, where the turn of three points keeps its sign."""
    k = max(range(3), key=lambda m: abs(n[m]))
    u, v = (k + 1) % 3, (k + 2) % 3

    def turn(o, a, b):
        return ((P[a][u] - P[o][u]) * (P[b][v] - P[o][v])
                - (P[a][v] - P[o][v]) * (P[b][u] - P[o][u]))

    order = sorted(range(len(P)), key=lambda i: (P[i][u], P[i][v]))
    ring = []
    for seq in (order, order[::-1]):
        c = []
        for i in seq:
            while len(c) >= 2 and turn(c[-2], c[-1], i) <= 0:
                c.pop()
            c.append(i)
        ring += c[:-1]
    if n[k] < 0:
        ring.reverse()
    # The far side fans from the next corner, so the two sides share no diagonal and every
    # edge is used once each way, as on the joggled hull.
    far = ring[1:] + ring[:1]
    return ([(ring[0], ring[m], ring[m + 1]) for m in range(1, len(ring) - 1)]
            + [(far[0], far[m + 1], far[m]) for m in range(1, len(far) - 1)])


SIX_PLACES = decimal.Decimal("0.000001")


def float_text(x):
    """A value as studiomdl's `KeyWriteFloat` prints it: a float through `%f`. A float exactly
    halfway between two sixth decimals rounds away from zero, as the shipped text does
    (4610.9765625 is 4610.976563) and Python's `%f` does not."""
    return format(decimal.Decimal(_f32(x)).quantize(SIX_PLACES, decimal.ROUND_HALF_UP), "f")


def keyvalues(blocks):
    """The text half from (block name, [(key, value)]) pairs, each line as studiomdl's
    `fprintf` writes it, then the NUL `VCollideLoad` stops at."""
    out = []
    for name, pairs in blocks:
        out.append(f"{name} {{\n")
        for k, v in pairs:
            if '"' in k or '"' in v or "\0" in k or "\0" in v:
                raise PhyError(f"a quote or NUL in {k!r} {v!r}")
            out.append(f'"{k}" "{v}"\n')
        out.append("}\n")
    return "".join(out).encode("latin-1") + b"\0"


def masses(volumes, massbiases, totalmass):
    """studiomdl's `WriteCollisionModel`: each solid's share of `totalmass` by volume times
    massbias, at least 1."""
    total = sum(v * b for v, b in zip(volumes, massbiases))
    if total <= 0:
        total = 1.0
    return [max(1.0, v * b / total * totalmass) for v, b in zip(volumes, massbiases)]


AXES = ("x", "y", "z")


def text_half(solids, joints=(), rootname="", totalmass=1.0, concave=False, jointmerge=()):
    """The text half of a new `.phy`, in the key order of every shipped one.

    `solids`: dicts with `name`, `surfaceprop`, `volume` (cubic inches), and optionally
    `parent`, `origin` and `angles` (three floats each, written together or not at all),
    `massbias`, and `mass`, `damping`, `rotdamping`, `inertia`, which default to `masses`'
    share and studiomdl's 0, 0, 1. `joints`: (parent ordinal, child ordinal, limits), the
    limits x, y, z each min, max, friction; studiomdl's memset leaves all nine 0, rigid."""
    if solids and "mass" in solids[0]:
        mass = [s["mass"] for s in solids]
    else:
        mass = masses([s["volume"] for s in solids], [s.get("massbias", 1.0) for s in solids],
                      totalmass)
    out = []
    for i, s in enumerate(solids):
        kv = [("index", str(i)), ("name", s["name"])]
        if s.get("parent") is not None:
            kv.append(("parent", s["parent"]))
        if s.get("origin") is not None:
            kv.append(("origin", " ".join(float_text(x) for x in s["origin"])))
            kv.append(("angles", " ".join(float_text(x) for x in s["angles"])))
        kv += [("mass", float_text(mass[i])), ("surfaceprop", s["surfaceprop"]),
               ("damping", float_text(s.get("damping", 0.0))),
               ("rotdamping", float_text(s.get("rotdamping", 0.0))),
               ("inertia", float_text(s.get("inertia", 1.0))),
               ("volume", float_text(s["volume"]))]
        if s.get("massbias") is not None:
            kv.append(("massbias", float_text(s["massbias"])))
        out.append(("solid", kv))
    for parent, child, limits in joints:
        kv = [("parent", str(parent)), ("child", str(child))]
        for a, k in enumerate(AXES):
            for m, key in enumerate(("min", "max", "friction")):
                kv.append((k + key, float_text(limits[3 * a + m])))
        out.append(("ragdollconstraint", kv))
    kv = [("rootname", rootname), ("totalmass", float_text(totalmass))]
    if concave:
        kv.append(("concave", "1"))
    kv += [("jointmerge", "%s,%s" % pair) for pair in jointmerge]
    out.append(("editparams", kv))
    return keyvalues(out)


def phy_file(checksum, surfaces, text):
    """A whole `.phy` from surfaces made by `build_surface` (or read), and `keyvalues`'
    text."""
    out = [HEADER.pack(HEADER.size, 0, len(surfaces), checksum)]
    for s in surfaces:
        b = surface_bytes(s)
        out.append(struct.pack("<i", len(b)))
        out.append(b)
    out.append(text)
    return b"".join(out)


BREAKS = frozenset(b"{}()':")


def tokens(text):
    """The text half cut as `ParseFile` cuts it (source-2003 `public/filesystem_helpers.cpp`).

    Each token is (bytes, start, end, quoted), start and end spanning its content without
    the quotes. Whitespace is every byte up to 0x20, `//` and `/* */` are comments, a quoted
    string runs to the next quote, each of `{}()':` is a token of its own, and the first NUL
    ends the text.
    """
    out = []
    i, n = 0, len(text)
    while True:
        while i < n and text[i] <= 0x20:
            if text[i] == 0:
                return out
            i += 1
        if i >= n:
            return out
        c = text[i]
        if text.startswith(b"//", i):
            while i < n and text[i] not in (0, 0x0a):
                i += 1
            continue
        if text.startswith(b"/*", i):
            j = text.find(b"*/", i + 2)
            k = text.find(bytes(1), i + 2)
            i = n if j < 0 or 0 <= k < j else j + 2
            continue
        if c == 0x22:
            j = i + 1
            while j < n and text[j] not in (0, 0x22):
                j += 1
            out.append((text[i + 1:j], i + 1, j, True))
            if j >= n or text[j] == 0:
                return out
            i = j + 1
            continue
        if c in BREAKS:
            out.append((text[i:i + 1], i, i + 1, False))
            i += 1
            continue
        j = i + 1
        while j < n and text[j] > 0x20 and text[j] not in BREAKS:
            j += 1
        out.append((text[i:j], i, j, False))
        i = j


def blocks(text):
    """The text half as `CVPhysicsParse` walks it: [(name, [(key, value), ...]), ...].

    Keys, values and names are `tokens` entries, so a caller can rewrite one in place. A
    block opens on any key whose value is `{` and closes on a key starting with `}`, whose
    value is read unless the key is `}` alone (`ParseKeyvalue`, `ParseSolid`). The engine
    lowercases both halves; this keeps the bytes as written.
    """
    t = tokens(text)
    out = []
    i = 0
    while i < len(t):
        key = t[i]
        if key[0] == b"}":
            i += 1
            continue
        if i + 1 >= len(t):
            break
        value = t[i + 1]
        i += 2
        if value[0] != b"{":
            continue
        pairs = []
        while i < len(t):
            k = t[i]
            if k[0] == b"}":
                i += 1
                break
            v = t[i + 1] if i + 1 < len(t) else None
            i += 2
            if k[0][:1] == b"}":
                break
            pairs.append((k, v))
        out.append((key, pairs))
    return out


# `jointmerge` holds two, "parent,child" (studiomdl `collisionmodel.cpp:2209`). No other value
# in the 2897 shipped files with a `.mdl` beside them is a bone name.
BONE_KEYS = {b"solid": (b"name", b"parent"), b"editparams": (b"rootname", b"jointmerge")}


def text_start(data):
    """Where the text half begins, found from the solids' sizes alone, so a surface this
    module cannot read does not stop a rename."""
    if len(data) < HEADER.size:
        raise PhyError("shorter than its header")
    size, _id, count, _sum = HEADER.unpack_from(data)
    if size != HEADER.size or count <= 0:
        raise PhyError(f"header size {size}, {count} solids")
    pos = size
    for i in range(count):
        if pos + 4 > len(data):
            raise PhyError(f"solid {i} starts past the end")
        n, = struct.unpack_from("<i", data, pos)
        if n <= 0 or pos + 4 + n > len(data):
            raise PhyError(f"solid {i} of {n} B overruns the file")
        pos += 4 + n
    return pos


def _close(t):
    """Where the token `t` ends, past its closing quote when it has one."""
    return t[2] + (1 if t[3] else 0)


def _open(t):
    return t[1] - (1 if t[3] else 0)


def block_spans(text):
    """`blocks`, with each block's byte span and each pair's: [(name, [(key, value,
    (start, end))], (start, end))], start at the block's name and end past its `}`."""
    t = tokens(text)
    out = []
    i = 0
    while i < len(t):
        key = t[i]
        if key[0] == b"}":
            i += 1
            continue
        if i + 1 >= len(t):
            break
        value = t[i + 1]
        i += 2
        if value[0] != b"{":
            continue
        pairs = []
        end = _close(value)
        while i < len(t):
            k = t[i]
            if k[0] == b"}":
                end = _close(k)
                i += 1
                break
            v = t[i + 1] if i + 1 < len(t) else None
            i += 2
            end = _close(v if v is not None else k)
            if k[0][:1] == b"}":
                break
            pairs.append((k, v, (_open(k), end)))
        out.append((key, pairs, (_open(key), end)))
    return out


# What studiomdl writes from the collision model. Anything after them is its `$collisiontext`,
# copied in after `editparams` (source-2003 `collisionmodel.cpp:1708`).
OWN_BLOCKS = (b"solid", b"ragdollconstraint", b"editparams")
MAX_KEYVALUE = 1024     # vcollide_parse_private.h: a token this long leaves the last key


def _tail(text):
    """[lo, hi) of the text after the last `OWN_BLOCKS` block's line, up to the NUL."""
    lo = 0
    for name, _p, span in block_spans(text):
        if name[0].lower() in OWN_BLOCKS:
            lo = _line(text, *span)[1]
    hi = text.find(b"\0", lo)
    return lo, len(text) if hi < 0 else hi


def collision_text(data):
    """The text half after its last `solid`, `ragdollconstraint` or `editparams` block, up to
    the NUL, latin-1: studiomdl's `$collisiontext`, the `break` blocks of 132 shipped files."""
    text = bytes(data[text_start(data):])
    lo, hi = _tail(text)
    return text[lo:hi].decode("latin-1")


def with_collision_text(data, new):
    """`data` with `collision_text` replaced by the bytes `new`."""
    start = text_start(data)
    text = bytes(data[start:])
    lo, hi = _tail(text)
    return bytes(data[:start]) + text[:lo] + new + text[hi:]


def walk_collision_text(text):
    """`collision_text` as the engine reads it after `editparams` (`CVPhysicsParse::NextBlock`
    and `ParseCustom`, source-2003 `vphysics/vcollide_parse.cpp`): ([(name, [(key, value)],
    line)], [what is wrong]), names and keys lowercased as the engine lowercases them, values
    as written. A `{` inside a block opens no block of its own, so the blocks after it are
    read as part of it, and a pair outside any block can swallow the next block's name."""
    t = tokens(text)

    def line(tok):
        return text.count(b"\n", 0, tok[1]) + 1

    out, errs = [], []
    for tok in t:
        if len(tok[0]) >= MAX_KEYVALUE:
            errs.append("line %d has a %d-character token, and the engine reads none of %d "
                        "or more" % (line(tok), len(tok[0]), MAX_KEYVALUE))
    i, stray = 0, None
    while i < len(t):
        key = t[i]
        if key[0] == b"}":
            i += 1
            continue
        value = t[i + 1] if i + 1 < len(t) else None
        i += 2
        if value is None or value[0] != b"{":
            if line(key) != stray:
                stray = line(key)
                errs.append("line %d, from %r on, is outside any block, and the engine skips "
                            "it up to a name followed by {" % (stray, key[0].decode("latin-1")))
            continue
        pairs, depth, closed = [], 0, False
        while i < len(t):
            k = t[i]
            if k[0] == b"}":
                i += 1
                depth -= 1
                if depth < 0:
                    closed = True
                    break
                continue
            if k[0] == b"{":
                i += 1
                depth += 1
                errs.append("the %s block on line %d has a { on line %d" % (
                    key[0].decode("latin-1"), line(key), line(k)))
                continue
            v = t[i + 1] if i + 1 < len(t) else None
            i += 2
            if v is not None and v[0] == b"{":
                depth += 1
                errs.append("the %s block on line %d has a %r { on line %d, so what follows "
                            "is read as part of it: a key before it has no value, or a } is "
                            "missing" % (key[0].decode("latin-1"), line(key),
                                         k[0].decode("latin-1"), line(k)))
                continue
            if k[0][:1] == b"}":
                depth -= 1
                if depth < 0:
                    closed = True
                    break
                continue
            pairs.append((k[0].lower().decode("latin-1"),
                          v[0].decode("latin-1") if v is not None else None))
        name = key[0].lower()
        if not closed:
            errs.append("the %s block on line %d has no closing }" % (name.decode("latin-1"),
                                                                     line(key)))
        if name in OWN_BLOCKS:
            errs.append("the %s block on line %d is written from the collision hulls, and a "
                        "second one here would be read as another" % (name.decode(), line(key)))
        out.append((name.decode("latin-1"), pairs, line(key)))
    return out, errs


def break_count(text):
    """How many `break` blocks the engine reads in `collision_text` bytes `text`."""
    return sum(1 for name, _p, _l in walk_collision_text(text)[0] if name == "break")


def break_model(pairs):
    """The `model` a `break` block's pairs leave in `CBreakParser`'s `modelName`, or None with
    no `model` key: each one overwrites it, so the last (source-2003 `dlls/props.cpp:1330`)."""
    out = None
    for k, v in pairs:
        if k == "model":
            out = v
    return out


def check_collision_text(s):
    """What stops `s` going in as `collision_text`: [message]. `walk_collision_text`'s, then a
    `break` block, one piece the prop breaks into (`CBreakParser`, source-2003
    `dlls/props.cpp:1330`), whose `model` is missing or empty, which the engine spawns as
    `models/.mdl`."""
    if "\0" in s:
        return ["a NUL, which ends the text there"]
    try:
        b = s.encode("latin-1")
    except UnicodeEncodeError as e:
        return ["%r has no latin-1 form" % s[e.start:e.end]]
    out, errs = walk_collision_text(b)
    for name, pairs, line in out:
        if name == "break" and not break_model(pairs):
            errs.append("the break block on line %d has no model" % line)
    return errs


def _line(text, lo, hi):
    """[lo, hi) widened over the whitespace after it, through the first newline, so a
    removed line leaves no blank one."""
    while hi < len(text) and text[hi] in b" \t\r":
        hi += 1
    if hi < len(text) and text[hi] == 0x0a:
        hi += 1
    return lo, hi


def matrix_angles(m):
    """(pitch, yaw, roll) in degrees of a 3x4's rotation, as mathlib's `MatrixAngles` gives
    them (source-2003 `public/mathlib.cpp:1108`), roll 0 when the x axis is nearly vertical."""
    fx, fy, fz = m[0][0], m[1][0], m[2][0]
    xy = math.sqrt(fx * fx + fy * fy)
    pitch = math.degrees(math.atan2(-fz, xy))
    if xy > 0.001:
        return pitch, math.degrees(math.atan2(fy, fx)), math.degrees(math.atan2(m[2][1], m[2][2]))
    return pitch, math.degrees(math.atan2(-m[0][1], m[1][1])), 0.0


def move_surface(s, m):
    """`s` rebuilt with every point taken through `m`, a 3x4 in HL inches, its ledges and
    triangles in their order and its `dummy` words kept."""
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if not det > 0:
        raise PhyError(f"a matrix of determinant {det} would turn the hull inside out")

    def go(p):
        x, y, z = ivp_to_hl(p)
        return hl_to_ivp([r[0] * x + r[1] * y + r[2] * z + r[3] for r in m])

    out = build_surface([[tuple(go(s.points[i]) for i in tri) for tri in ledge_triangles(l)]
                         for l in s.ledges])
    out.dummy = s.dummy
    return out


def drop_solids(data, names, moves=None):
    """`data` with every solid named for one of `names` taken out or moved, and what that did.

    The engine matches a solid to its bone with `stricmp`, so `names` match case-blind. A
    solid with no parent and more than one child solid moves when `moves` names its bone:
    {bone: (the bone taking it, the 3x4 from the old bone's space to its in HL inches, its
    rest transform in model space)}, unless a solid already carries that name. Its points go
    through the matrix, and its `name` and every `parent`, `rootname` and `jointmerge` part
    naming it take the new name, its `origin`/`angles` the rest transform. A joint's limits
    are about its child's own bone axes (source-2003 `ragdoll_shared.cpp:83`), so the joints
    are copied.

    Every other named solid is dropped: its binary record and `solid` block go; the
    survivors' `index` values close up; a survivor whose `parent` named a dropped solid takes
    that solid's parent, up the chain to the first that survives. A `ragdollconstraint` whose
    child was dropped goes; one whose parent was dropped moves to that same surviving
    ancestor. A `jointmerge` pair naming a dropped solid goes and a `rootname` naming one
    takes its surviving ancestor. Everything else, the other surfaces included, is copied.

    Returns (bytes, {old ordinal: new ordinal or None}, [dropped solid names], [(moved solid
    name, name now)]). Raises `PhyError` when every solid would be dropped, and its
    `RagdollSplit` when a drop would leave more solids with no parent than the file had.
    """
    start = text_start(data)
    size, ident, count, checksum = HEADER.unpack_from(data)
    records, pos = [], size
    for _ in range(count):
        n, = struct.unpack_from("<i", data, pos)
        records.append(bytes(data[pos:pos + 4 + n]))
        pos += 4 + n
    text = bytes(data[start:])
    spans = block_spans(text)

    solids = []
    for name, pairs, span in spans:
        if name[0].lower() != b"solid":
            continue
        kv = {k[0].lower(): (v, s) for k, v, s in pairs if v is not None}
        solids.append((kv, span))
    if len(solids) != count:
        raise PhyError(f"{len(solids)} solid blocks for {count} solids")
    for k, (kv, _) in enumerate(solids):
        got = kv.get(b"index", (None,))[0]
        if got is None or got[0] != str(k).encode():
            raise PhyError(f"solid block {k} carries index "
                           f"{got[0].decode('latin-1') if got else 'none'}")

    low = [kv.get(b"name", ((b"",),))[0][0].lower() for kv, _ in solids]
    drop = {n.lower().encode("latin-1") for n in names}
    named = [k for k in range(count) if low[k] in drop]
    if not named:
        return bytes(data), {k: k for k in range(count)}, [], []
    by_name = {}
    for k in range(count):
        by_name.setdefault(low[k], k)

    def parent_of(k):
        v = solids[k][0].get(b"parent")
        return by_name.get(v[0][0].lower()) if v else None

    kids = [0] * count
    for k in range(count):
        if parent_of(k) is not None:
            kids[parent_of(k)] += 1
    to = {}
    taken = set(low)
    want = {b.lower().encode("latin-1"): v for b, v in (moves or {}).items()}
    for k in named:
        if parent_of(k) is not None or kids[k] < 2 or low[k] not in want:
            continue
        p = want[low[k]][0].encode("latin-1")
        if p.lower() in taken:
            continue
        taken.add(p.lower())
        to[k] = p
    drop -= {low[k] for k in to}
    gone = [k for k in named if k not in to]
    keep = [k for k in range(count) if k not in gone]
    if not keep:
        raise PhyError("every solid would be dropped, and a .phy holds at least one")
    remap = {k: None for k in gone}
    remap.update((k, i) for i, k in enumerate(keep))

    def name_of(k):
        return to.get(k, solids[k][0][b"name"][0][0])

    def survivor(k):
        """The nearest solid up `k`'s parent chain that survives, `k` included."""
        seen = set()
        while k is not None and remap[k] is None and k not in seen:
            seen.add(k)
            k = parent_of(k)
        return k if k is not None and remap[k] is not None else None

    def put(v, b):
        return v[1], v[2], b if v[3] else b'"' + b + b'"'

    edits = []
    loose = []
    for k in gone:
        edits.append(_line(text, *solids[k][1]))
    for k in keep:
        kv = solids[k][0]
        v = kv[b"index"][0]
        if remap[k] != k:
            edits.append((v[1], v[2], str(remap[k]).encode()))
        if k in to:
            edits.append(put(kv[b"name"][0], to[k]))
            rest = want[low[k]][2]
            if b"origin" in kv and b"angles" in kv:
                edits.append(put(kv[b"origin"][0], " ".join(
                    float_text(r[3]) for r in rest).encode()))
                edits.append(put(kv[b"angles"][0], " ".join(
                    float_text(a) for a in matrix_angles(rest)).encode()))
        p = parent_of(k)
        if p is None:
            continue
        s = survivor(p)
        if s is None:
            edits.append(_line(text, *kv[b"parent"][1]))
            loose.append(name_of(k).decode("latin-1"))
        elif s != p or s in to:
            edits.append(put(kv[b"parent"][0], name_of(s)))
    had = sum(parent_of(k) is None for k in range(count))
    if sum(parent_of(k) is None for k in keep) + len(loose) > had:
        raise RagdollSplit(
            "dropping %s would leave %s with no parent solid, a ragdoll in pieces"
            % (", ".join(repr(solids[k][0][b"name"][0][0].decode("latin-1")) for k in gone),
               ", ".join(repr(n) for n in loose)))

    for name, pairs, span in spans:
        n = name[0].lower()
        kv = {k[0].lower(): (v, s) for k, v, s in pairs if v is not None}
        if n == b"ragdollconstraint":
            try:
                parent = int(kv[b"parent"][0][0])
                child = int(kv[b"child"][0][0])
            except (KeyError, ValueError):
                continue
            if not (0 <= child < count and 0 <= parent < count):
                continue
            if remap[child] is None:
                edits.append(_line(text, *span))
                continue
            if remap[parent] is None:
                parent = survivor(parent)
                if parent is None:
                    edits.append(_line(text, *span))
                    continue
            for key, now in ((b"parent", remap[parent]), (b"child", remap[child])):
                v = kv[key][0]
                if v[0] != str(now).encode():
                    edits.append((v[1], v[2], str(now).encode()))
        elif n == b"editparams":
            for k, v, s in pairs:
                if v is None:
                    continue
                key = k[0].lower()
                if key == b"jointmerge":
                    parts = v[0].split(b",")
                    if any(p.lower() in drop for p in parts):
                        edits.append(_line(text, *s))
                        continue
                    now = [name_of(by_name[p.lower()]) if p.lower() in by_name
                           and by_name[p.lower()] in to else p for p in parts]
                    if now != parts:
                        edits.append(put(v, b",".join(now)))
                elif key == b"rootname" and v[0].lower() in by_name:
                    r = by_name[v[0].lower()]
                    if r in to:
                        edits.append(put(v, to[r]))
                    elif v[0].lower() in drop:
                        r = survivor(r)
                        edits.append(put(v, name_of(r) if r is not None else b""))
                elif key == b"rootname" and v[0].lower() in drop:
                    edits.append(put(v, b""))

    edits.sort(reverse=True)
    for lo, hi, *b in edits:
        text = text[:lo] + (b[0] if b else b"") + text[hi:]
    out = [HEADER.pack(size, ident, len(keep), checksum)]
    surfaces = Phy(data).solids if to else None
    for k in keep:
        if k in to:
            b = surface_bytes(move_surface(surfaces[k], want[low[k]][1]))
            out.append(struct.pack("<i", len(b)) + b)
        else:
            out.append(records[k])
    return (b"".join(out) + text, remap,
            [solids[k][0][b"name"][0][0].decode("latin-1") for k in gone],
            [(solids[k][0][b"name"][0][0].decode("latin-1"), to[k].decode("latin-1"))
             for k in sorted(to)])


def solid_names(data):
    """Each solid's `name` in the text half, in solid order."""
    return [dict((k[0].lower(), v[0]) for k, v in kv if v is not None).get(b"name", b"")
            .decode("latin-1")
            for name, kv in blocks(bytes(data[text_start(data):]))
            if name[0].lower() == b"solid"]


def rename_bones(data, pairs):
    """`data` with every bone name in its text half following its bone, and what changed.

    `pairs` is (name in the file, name now) for every bone of the model, in file order.
    The engine finds a solid's bone with `stricmp`, first match winning
    (`Studio_BoneIndexByName`, source-2003 `game_shared/bone_setup.cpp:3318`), so a value
    follows the first bone it matches that way, and one matching no bone -- a single-body
    prop's solid is named for the model -- stays as written. Nothing is touched for a bone
    whose name did not change, case included. The binary half is copied as it was.

    Returns (bytes, [(old, new)] for each value rewritten).
    """
    start = text_start(data)
    text = bytes(data[start:])
    follow = {}
    for old, new in pairs:
        follow.setdefault(old.lower(), (old, new))
    edits, changed = [], []
    for name, kv in blocks(text):
        keys = BONE_KEYS.get(name[0].lower())
        if keys is None:
            continue
        for k, v in kv:
            if v is None or k[0].lower() not in keys:
                continue
            was = v[0].decode("latin-1")
            parts = was.split(",") if k[0].lower() == b"jointmerge" else [was]
            now = []
            for p in parts:
                old, new = follow.get(p.lower(), (p, p))
                now.append(new if old != new else p)
                if now[-1] == p:
                    continue
                if '"' in new or "\0" in new or (len(parts) > 1 and "," in new):
                    raise PhyError(f"bone {new!r} cannot be written as {k[0].decode()!r}: "
                                   "a quote, a NUL or, in a jointmerge pair, a comma ends it")
                changed.append((p, new))
            if now == parts:
                continue
            try:
                b = ",".join(now).encode("latin-1")
            except UnicodeEncodeError as e:
                raise PhyError(f"{','.join(now)!r} has no latin-1 form: {e}") from None
            edits.append((v[1], v[2], b if v[3] else b'"' + b + b'"'))
    for lo, hi, b in sorted(edits, reverse=True):
        text = text[:lo] + b + text[hi:]
    return bytes(data[:start]) + text, changed


# The deepest a shipped ledge point lies inside its own ledge's hull is 4.3e-08 of the ledge's
# size, 5.9e-08 once the import has rounded it to a Blender float at scale 1: 105 of the 166
# points `convex_hull` leaves out of a shipped ledge lie on a face and the rest are float
# rounding. A point inside by more than this share of its part's size is a dent.
DENT_TOL = 1e-5


def part_hull(points, faces=()):
    """(`convex_hull`'s triangles over `points`, [index of each point in a dent]). A ledge
    is convex by format, so the written ledge would fill a dent.

    A point is in one when the hull leaves it inside by more than `DENT_TOL` of the points'
    size, or, with `faces` the triangles the mesh draws over `points`, when it is the far
    corner of a triangle across an edge from another and lies that far in front of the
    other's plane: a concave edge whose corners all stay on the hull. The faces are taken outward
    by their signed volume, so a mesh drawn inside out reads the same. A point on a face,
    or within the tolerance of one, is left out of the ledge and changes nothing."""
    tris = convex_hull(points)
    kept = {i for t in tris for i in t}
    size = max(max(p[k] for p in points) - min(p[k] for p in points) for k in range(3))
    lim = DENT_TOL * size

    def plane(i, j, k, sign=1.0, thin=0.0):
        """The triangle's plane, or None for one no higher than `thin`."""
        a, b, c = points[i], points[j], points[k]
        n = _cross(a, b, c)
        m = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if m == 0 or m / max(math.dist(a, b), math.dist(b, c), math.dist(c, a)) <= thin:
            return None
        return (sign * n[0] / m, sign * n[1] / m, sign * n[2] / m), a

    def front(pl, p):
        n, a = pl
        return n[0] * (p[0] - a[0]) + n[1] * (p[1] - a[1]) + n[2] * (p[2] - a[2])

    planes = [pl for pl in (plane(*t) for t in tris) if pl is not None]
    deep = set(q for q, p in enumerate(points)
               if q not in kept and -max(front(pl, p) for pl in planes) > lim)
    vol = 0.0
    for i, j, k in faces:
        a, b, c = points[i], points[j], points[k]
        vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
    sign = -1.0 if vol < 0 else 1.0
    across = {}
    for f in faces:
        for e in range(3):
            across.setdefault((f[e], f[(e + 1) % 3]), []).append(f)
    # A triangle under the tolerance high is no reference: three points nearly on a line
    # tilt its plane any way, and one in grandpiano.phy reads as an edge 0.29 of its size concave.
    for (a, b), fs in across.items():
        for f in fs:
            for g in across.get((b, a), ()):
                pl = plane(*f, sign, lim)
                far = next((v for v in g if v != a and v != b), None)
                if pl is not None and far is not None and front(pl, points[far]) > lim:
                    deep.add(far)
    return tris, sorted(deep)


def text_values(data):
    """The text half's values: ([{key: value} per `solid` block, in order], {child ordinal:
    {key: value}} per `ragdollconstraint`, {key: value} of `editparams`), keys lowercased
    as the engine reads them and values as written, both latin-1."""
    solids, joints, edit = [], {}, {}
    for name, kv in blocks(bytes(data[text_start(data):])):
        d = dict((k[0].lower().decode("latin-1"), v[0].decode("latin-1"))
                 for k, v in kv if v is not None)
        n = name[0].lower()
        if n == b"solid":
            solids.append(d)
        elif n == b"ragdollconstraint":
            try:
                joints[int(d["child"])] = d
            except (KeyError, ValueError):
                pass
        elif n == b"editparams":
            edit = d
    return solids, joints, edit


def _records(data):
    size, _ident, count, _sum = HEADER.unpack_from(data)
    out, pos = [], size
    for _ in range(count):
        n, = struct.unpack_from("<i", data, pos)
        out.append(bytes(data[pos:pos + 4 + n]))
        pos += 4 + n
    return out


def _latin1(x):
    """`x` as the text half can hold it: latin-1, and no quote or NUL, either of which ends a
    quoted token."""
    if '"' in x or "\0" in x:
        raise PhyError(f"a quote or NUL in {x!r}")
    try:
        return x.encode("latin-1")
    except UnicodeEncodeError as e:
        raise PhyError(f"{x!r} has no latin-1 form: {e}") from None


def edit_solids(data, surfaces=None, values=None, joints=None):
    """`data` with surfaces replaced and values rewritten, every other byte copied.

    `surfaces` is {ordinal: Surface}. `values` is {ordinal: {key: value}} on that solid's
    block and `joints` {child ordinal: {key: value}} on the `ragdollconstraint` whose `child`
    is that ordinal; a value is rewritten in its token's span, and a key the block lacks is
    added on a line before its `}`. Raises `PhyError` naming a solid or joint not there."""
    start = text_start(data)
    text = bytes(data[start:])
    want = {}
    for k, kv in (values or {}).items():
        want[(b"solid", k)] = kv
    for k, kv in (joints or {}).items():
        want[(b"ragdollconstraint", k)] = kv
    edits, found, solid = [], set(), 0
    for name, pairs, span in block_spans(text):
        n = name[0].lower()
        if n == b"solid":
            key, solid = (n, solid), solid + 1
        elif n == b"ragdollconstraint":
            got = [v[0] for k, v, _s in pairs if v is not None and k[0].lower() == b"child"]
            try:
                key = (n, int(got[0]))
            except (IndexError, ValueError):
                continue
        else:
            continue
        kv = want.get(key)
        if kv is None:
            continue
        found.add(key)
        have = {k[0].lower(): v for k, v, _s in pairs if v is not None}
        add = b""
        for k, value in kv.items():
            v = have.get(k.lower().encode("latin-1"))
            if v is None:
                add += b'"%s" "%s"\n' % (_latin1(k), _latin1(value))
                continue
            b = _latin1(value)
            edits.append((v[1], v[2], b if v[3] else b'"' + b + b'"'))
        if add:
            edits.append((span[1] - 1, span[1] - 1, add))
    lost = set(want) - found
    if lost:
        raise PhyError("no %s" % ", ".join("%s %d" % (n.decode(), k) for n, k in sorted(lost)))
    for lo, hi, b in sorted(edits, reverse=True):
        text = text[:lo] + b + text[hi:]
    records = _records(data)
    for k, s in (surfaces or {}).items():
        if not 0 <= k < len(records):
            raise PhyError(f"no solid {k} of {len(records)}")
        b = surface_bytes(s)
        records[k] = struct.pack("<i", len(b)) + b
    return bytes(data[:HEADER.size]) + b"".join(records) + text


def tag_solids(data):
    """`data` with `IVPS` in every solid's `dummy[2]` when it holds more than one, and how
    many solids lacked it.

    `VCollideLoad` (vphysics.dll 0x2600ac70) writes a solid's index into `dummy[0]` only on
    a tagged solid, and `CollideIndex` answers 0 for any other. A saved game records that
    answer and the load rebuilds the object from `solids[iCollide]` (vampire.dll
    0x100458f1, 0x100444a4), so an untagged ragdoll comes back with every element on solid
    0's hull. 98 of 7353 shipped solids are untagged: every solid of four ragdolls and 38
    one-solid props, where index 0 is the right answer anyway."""
    records = _records(data)
    if len(records) < 2:
        return data, 0
    n = 0
    for k, r in enumerate(records):
        if r[48:52] != IVPS:
            records[k] = r[:48] + IVPS + r[52:]
            n += 1
    if not n:
        return data, 0
    return bytes(data[:HEADER.size]) + b"".join(records) + bytes(data[text_start(data):]), n


def add_solid(data, surface, pairs, joint=None):
    """`data` with one solid appended, and its ordinal: the record after the last, a `solid`
    block of `index` then `pairs` [(key, value)] after the last `solid` block, and with
    `joint`, [(key, value)] after `parent` and `child`, a `ragdollconstraint` after the last
    one, else after the new solid."""
    start = text_start(data)
    text = bytes(data[start:])
    size, ident, count, checksum = HEADER.unpack_from(data)
    spans = block_spans(text)
    last = [span for name, _p, span in spans if name[0].lower() == b"solid"]
    if not last:
        raise PhyError("no solid block to add one after")
    at = _line(text, *last[-1])[1]
    block = keyvalues([("solid", [("index", str(count))] + list(pairs))])[:-1]
    text = text[:at] + block + text[at:]
    if joint is not None:
        spans = block_spans(text)
        cons = [span for name, _p, span in spans if name[0].lower() == b"ragdollconstraint"]
        to = _line(text, *cons[-1])[1] if cons else at + len(block)
        text = text[:to] + keyvalues([("ragdollconstraint", list(joint))])[:-1] + text[to:]
    b = surface_bytes(surface)
    return (HEADER.pack(size, ident, count + 1, checksum) + b"".join(_records(data))
            + struct.pack("<i", len(b)) + b + text), count


def ledge_mesh(s, scale=1.0):
    """Surface `s` as the import lays it out: ([vertex], [triangle], [first vertex of each
    ledge]). Each ledge takes its own copy of its points, in index order, HL inches times
    `scale`, so a ledge is one loose part. A triangle on three points an earlier one of its
    ledge already used is left out, which drops a three-point flat ledge's far side."""
    verts, faces, starts = [], [], []
    for l in s.ledges:
        idx = sorted(l.point_indices())
        at = {i: len(verts) + n for n, i in enumerate(idx)}
        starts.append(len(verts))
        verts += [tuple(c * scale for c in ivp_to_hl(s.points[i])) for i in idx]
        seen = set()
        for t in ledge_triangles(l):
            if frozenset(t) in seen:
                continue
            seen.add(frozenset(t))
            faces.append(tuple(at[i] for i in t))
    return verts, faces, starts
