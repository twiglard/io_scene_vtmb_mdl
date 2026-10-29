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
