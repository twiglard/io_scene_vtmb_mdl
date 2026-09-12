"""Authoring a cloth object from a mesh and a pin set, and packing it to bytes.

todo-vtmb-cloth-solver.md is the contract.  Four of its section 10 choices turned out not to
be choices at all -- the spring topology follows from the face list and the mass split from
the ring index -- so a caller supplies a mesh, a pin set, and three numbers:

    sigma   per-spring stiffness, 1.0 unless a region should be softer
    slack   `s`: group 1's rest length squared as a fraction of the squared true separation,
            so the pair may not close below sqrt(s) of its rest separation -- section 3.1
    scale   the per-cloth gravity multiplier, section 1.1

`preset` supplies all three off a shipped garment instead.  Everything else is derived.

plans/cloth-gen.py drives this module and measures it -- regenerating every shipped object
from its own inputs, which is the only honest number, since the corpus is a solved setup.
"""

import collections
import struct

HDR = 0x5c
# The naive pair (p0p1, p1p2), which inverts half the faces because a stored edge is
# ascending and cannot carry the sign; plans/cloth-gen.py --naive-face-edges is the control.
NAIVE_FACE_EDGES = False


class Refused(Exception):
    """Declared here rather than imported, so `plans/cloth-gen.py` can import this module
    flat. `SystemExit` is a BaseException and neither operator's `except Exception` caught
    it."""


def pair(a, c):
    return (a, c) if a < c else (c, a)


def tri_edges(tris):
    return set(pair(t[i], t[(i + 1) % 3]) for t in tris for i in range(3))


# How far apart two model positions may be and still count as one seam's duplicates.  The
# corpus duplicates a vertex exactly, so this only has to survive float32 printing.
EDGE_Q = 4


def edge_keys(pv, springs, ns0, pos, edges):
    """({group-0 spring: the mesh edge carrying its sigma}, the springs no edge reaches).

    A spring names two particles and `pv` names each one's model vertex, but the mesh's
    triangulation is not the cloth's own face list, so the pair is not always an edge of it:
    26 950 of the corpus's 30 623 group-0 springs are one directly, another 3384 once a
    vertex is exchanged for a duplicate a seam left at the same position, and 289 are reached
    by no edge at all and keep whatever the file gave them.  The exchange takes the smallest
    such pair, and it is injective as measured -- those 30 334 springs land on 30 334
    distinct edges, no two claiming one.

    `pos` is one position per model vertex and `edges` the mesh's own ascending pairs.  Both
    sides pass the FILE's positions, so moving a vertex in the scene cannot move a spring
    onto another edge.
    """
    at = collections.defaultdict(list)
    for i, p in enumerate(pos):
        at[tuple(round(x, EDGE_Q) for x in p)].append(i)
    keys, missed = {}, []
    for q in range(min(ns0, len(springs))):
        a, b = springs[q][0], springs[q][1]
        if a >= len(pv) or b >= len(pv):
            missed.append(q)
            continue
        u, w = pv[a], pv[b]
        k = pair(u, w)
        if k not in edges:
            du = at.get(tuple(round(x, EDGE_Q) for x in pos[u]), [u]) if u < len(pos) else [u]
            dw = at.get(tuple(round(x, EDGE_Q) for x in pos[w]), [w]) if w < len(pos) else [w]
            hit = sorted(pair(x, y) for x in du for y in dw if pair(x, y) in edges)
            if not hit:
                missed.append(q)
                continue
            k = hit[0]
        keys[q] = k
    return keys, missed


def adjacency(pairs, n):
    g = collections.defaultdict(set)
    for a, c in pairs:
        if a < n and c < n:
            g[a].add(c)
            g[c].add(a)
    return g


def bfs(g, seeds, limit=None):
    d = {v: 0 for v in seeds}
    q = collections.deque(seeds)
    while q:
        v = q.popleft()
        if limit is not None and d[v] >= limit:
            continue
        for w in g.get(v, ()):
            if w not in d:
                d[w] = d[v] + 1
                q.append(w)
    return d


Preset = collections.namedtuple("Preset", "scale s sigma floor objects springs varying")

# `sigma` reproduces the shipped w0/w1 only where `varying` is 0; on the other 6 it is the mean
# of a continuous quantity no scalar can match (todo-vtmb-cloth-solver.md section 12).
PRESETS = {
    #                        scale      s   sigma   floor  obj  springs  varying
    "misti":         Preset(   1.0,   0.5,    1.0,    1.0,  39,    7039,  0),
    "blueblood_female":
                     Preset(   6.0,   0.2, 0.9444, 0.1010,  26,   12089,  3),
    "stan_gimble":   Preset(   2.0,   0.5, 0.7635, 0.1000,  17,    6269, 13),
    "sheriff":       Preset(   3.0,   0.5, 0.8584, 0.1000,  14,    8677,  4),
    "grouts_wife":   Preset(   3.0,   0.7, 0.4575, 0.0250,   8,    6846,  8),
    "elite_hunter":  Preset(   1.5,   0.7,    1.0,    1.0,   7,    4550,  0),
    "hostess":       Preset(   8.0,  0.05,    1.0,    1.0,   7,    1232,  0),
    "ghost":         Preset(   4.0,   1.0, 0.4954, 0.1000,   5,    3214,  5),
    "lantern":       Preset(   0.3,   0.7,    1.0,    1.0,   3,    1698,  0),
    "andrei":        Preset(   2.0,   0.4,    1.0,    1.0,   3,    1308,  0),
    "doppleganger_female":
                     Preset(   2.0,   1.0,    1.0,    1.0,   3,    1078,  0),
    "yukie":         Preset(   4.0,   0.8,    1.0,    1.0,   3,    4408,  0),
    "toreador_female_armor_2":
                     Preset(   4.0,   0.5, 0.4911, 0.1000,   2,    1290,  2),
    "ventrue_female_armor_2":
                     Preset(   6.0,   1.0, 0.3931, 0.0336,   2,    1094,  2),
    "panel_cloth":   Preset(   1.0,   0.7,    1.0,    1.0,   1,     324,  0),
    "beckett":       Preset(   1.0,   1.0, 0.5179, 0.1000,   1,     330,  1),
    "jeanette":      Preset(   5.0, 0.001, 0.3869, 0.0500,   1,     934,  1),
}

PRESET_MODELS = {
    "misti": ("creation1_full", "misti", "mistidance", "prep_room_curtain",
              "strippdk", "strippdr", "stripper"),
    "blueblood_female": ("blueblood_female", "blueblood_ming", "blueblood_therese",
                         "mingxiao", "mira", "nadia"),
    "stan_gimble": ("goth_female", "prostitute_2_ref", "rosa", "security_guard_skinny",
                    "stan_gimble"),
    "sheriff": ("bum_male", "malkavian_male_armor_3", "nosferatu_female_armor_2", "sheriff",
                "tin_can_bill", "tremere_female_armor_3", "ventrue_male_armor_3",
                "wolf_form", "wolf_form_2"),
    "grouts_wife": ("grouts_wife", "malkavian_female_armor_0", "pisha", "sheriff",
                    "tremere_female_armor_0", "tremere_female_armor_2",
                    "tremere_female_armor_3", "yukie"),
    "elite_hunter": ("bach", "buch", "elite_hunter", "stripper_reduced_01"),
    "hostess": ("hostess",),
    "ghost": ("ghost", "regent", "samantha", "toreador_female_armor_3"),
    "lantern": ("lantern", "lanternskins", "table"),
    "andrei": ("and", "andrei", "andrei_no_mouth"),
    "doppleganger_female": ("doppleganger_female", "imalia", "maria"),
    "yukie": ("yukie",),
    "toreador_female_armor_2": ("toreador_female_armor_2", "ventrue_female_armor_0"),
    "ventrue_female_armor_2": ("ventrue_female_armor_2", "ventrue_female_armor_3"),
    "panel_cloth": ("panel_cloth",),
    "beckett": ("beckett",),
    "jeanette": ("jeanette",),
}


def resolve(preset=None, sigma=None, slack=None, scale=None):
    """The three authored numbers.  A preset supplies whatever the caller left None; with no
    preset the fallbacks are the stiffest sigma and the commonest slack a shipped object uses."""
    if preset is not None and preset not in PRESETS:
        raise KeyError("no preset %r -- the 17 are: %s"
                       % (preset, ", ".join(sorted(PRESETS))))
    p = PRESETS[preset] if preset is not None else None
    return ((p.sigma if p else 1.0) if sigma is None else sigma,
            (p.s if p else 0.5) if slack is None else slack,
            (p.scale if p else 1.0) if scale is None else scale)


Recovered = collections.namedtuple(
    "Recovered", "scale sigma sigma_min sigma_mean uniform slack preset")


def recover(scale, springs, ns0, pos=None, tol=1e-4):
    """The three authored numbers read back off a shipped object, and the preset they name.

    `scale` comes off the header verbatim and needs no derivation. A spring stores
    `w0 = 2*sigma*k0` and `w1 = -2*sigma*(1-k0)`, so `sigma` is `(w0 - w1) / 2` whatever the
    mass split, one value per spring. Group 1 stores `rest2 = slack * d2`, so dividing by the
    squared rest separation gives `slack` -- one value per spring again, and the median is
    what reproduces the authored number.

    `pos` is one rest position per particle, or None where the caller has none (a quantised
    model), which leaves `slack` and `preset` None.

    What this cannot do: `sigma` is per spring and 39 of the corpus's 150 objects vary, so
    `sigma` here is the maximum and `uniform` says whether one number describes the object at
    all. `PRESETS[...].sigma` is a mean over a whole garment's objects and is not comparable
    to it.
    """
    sig = [(w0 - w1) / 2.0 for _a, _b, w0, w1, _r in springs]
    hi = max(sig) if sig else None
    lo = min(sig) if sig else None
    mean = sum(sig) / float(len(sig)) if sig else None
    sl = None
    if pos is not None:
        rat = []
        for a, b, _w0, _w1, rest2 in springs[ns0:]:
            if a < len(pos) and b < len(pos):
                d2 = sum((x - y) ** 2 for x, y in zip(pos[a], pos[b]))
                if d2 > 1e-9:
                    rat.append(rest2 / d2)
        if rat:
            rat.sort()
            n = len(rat)
            sl = rat[n // 2] if n % 2 else (rat[n // 2 - 1] + rat[n // 2]) / 2.0
    uniform = bool(sig) and (hi - lo) <= tol
    name = None
    if sl is not None:
        # (scale, s) is unique over all 17 presets, so the pair names one or none.
        for k, pr in PRESETS.items():
            if abs(pr.scale - scale) <= tol and abs(pr.s - sl) <= tol:
                name = k
                break
    return Recovered(scale, hi, lo, mean, uniform, sl, name)


def list_presets():
    w = max(len(k) for k in PRESETS)
    out = ["%-*s %6s %6s %7s %7s %5s %8s %8s   %s"
           % (w, "preset", "scale", "s", "sigma", "floor", "obj", "springs", "varying",
              "models")]
    for k in sorted(PRESETS, key=lambda k: (-PRESETS[k].objects, k)):
        p = PRESETS[k]
        out.append("%-*s %6g %6g %7g %7g %5d %8d %8d   %s"
                   % (w, k, p.scale, p.s, p.sigma, p.floor, p.objects, p.springs, p.varying,
                      ", ".join(PRESET_MODELS[k])))
    return "\n".join(out)


class Cloth:
    __slots__ = ("scale", "numfixed", "numfree", "pv", "springs", "ns0", "batch",
                 "tris", "edges", "ned0", "faces", "blends")

    @property
    def numparticles(self):
        return self.numfixed + self.numfree


def winding(e0, e1):
    """The triangle ordering whose normal equals E[e0] x E[e1], for E the directed delta.

    A face record names two of its triangle's three edges and the accumulator adds their
    cross product to all three corners (section 5).  Stored edges are ASCENDING -- 60938 of
    60938 over the corpus -- so a face cannot direct them and must instead pick an ordered
    pair that comes out on its own winding.  The four ways two edges can share a particle
    reduce as:

        b == c :  (b-a)x(d-b) = (b-a)x(d-a)        -> (a, b, d)
        a == c :  (b-a)x(d-a)                      -> (a, b, d)
        a == d :  (b-a)x(a-c) = -[(a-c)x(b-a)]     -> (c, b, a)
        b == d :  (b-a)x(b-c) = (a-b)x(c-b)        -> (b, a, c)

    Returned rotated so that two orderings of the same three particles compare equal iff
    they wind the same way.  None when the two edges share no particle.
    """
    a, b = e0
    c, d = e1
    if b == c or a == c:
        t = (a, b, d)
    elif a == d:
        t = (c, b, a)
    elif b == d:
        t = (b, a, c)
    else:
        return None
    i = t.index(min(t))
    return (t[i], t[(i + 1) % 3], t[(i + 2) % 3])


def face_edges(f, seen=()):
    """The ordered edge pair a face record names, as two ASCENDING particle pairs.

    Whichever pair is chosen, the accumulated normal has to wind the way the face's own
    (p0, p1, p2) does -- 40950 of 40950 shipped faces, 144 of 144 objects, no exception.
    Of the six ordered pairs drawn from the triangle's three edges exactly three satisfy
    that, since swapping the two negates the cross product, so one always exists.  Sorting
    the pair, which is what the edge array stores, is what loses the sign: taking
    (p0p1, p1p2) blind inverts a face whenever exactly one of the two comparisons runs
    backwards, which is half of them.

    `seen` is the edge set already emitted; preferring a candidate that adds nothing new
    keeps the face-referenced prefix short, which is what numedges0 counts.
    """
    tri = (f[0], f[1], f[2])
    i = tri.index(min(tri))
    want = (tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3])
    es = (pair(f[0], f[1]), pair(f[1], f[2]), pair(f[2], f[0]))
    if NAIVE_FACE_EDGES:
        return es[0], es[1]
    best = None
    for j, k in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
        if es[j] == es[k] or winding(es[j], es[k]) != want:
            continue
        cost = (es[j] not in seen) + (es[k] not in seen)
        if best is None or cost < best[0]:
            best = (cost, es[j], es[k])
            if not cost:
                break
    if best is None:
        raise ValueError("face %s has no edge pair on its own winding" % (tri,))
    return best[1], best[2]


def generate(pos, faces, npin, pv=None, sigma=None, slack=None, scale=None, collide=None,
             preset=None, sigma0=None):
    """`pos` is one rest position per particle, `faces` triangles over particle indices, and
    the first `npin` particles are the pinned ones -- the array is ordered pinned-first and
    nothing else marks a pin (section 1.1).

    `preset` names one of the 17 shipped (scale, s) pairs and supplies whichever of the three
    numbers the caller left None; passing all three explicitly is the same call as before.

    `sigma0` is one value per ascending particle pair and overrides `sigma` on group 0 alone,
    28 of the 59 shipped row-0 objects varying it spring by spring.  Group 1 keeps the scalar:
    it is 1.0 on all 27 008 shipped group-1 springs, and its pairs are bend partners rather
    than edges, so nothing addresses them."""
    sigma, slack, scale = resolve(preset, sigma, slack, scale)
    c = Cloth()
    c.scale, c.numfixed = scale, npin
    n = len(pos)
    c.numfree = n - npin
    c.pv = list(pv) if pv is not None else list(range(n))

    fe = tri_edges(faces)
    adj = adjacency(fe, n)
    r = bfs(adj, [v for v in range(npin) if v in adj])

    def d2(a, b):
        return sum((x - y) ** 2 for x, y in zip(pos[a], pos[b]))

    def spring(a, b, rest2, s=None):
        # v0 is the end nearer the pins, which is what makes it the heavier one and puts
        # k0 == 0 exactly on a pinned v0 -- section 3.3.
        s = sigma if s is None else s
        if r.get(a, 1 << 30) > r.get(b, 1 << 30):
            a, b = b, a
        r0, r1 = r.get(a, 0), r.get(b, 0)
        k0 = (r0 * r0) / float(r0 * r0 + r1 * r1)
        return (a, b, 2.0 * s * k0, -2.0 * s * (1.0 - k0), rest2)

    g0 = []
    for a, b in sorted(fe):
        if a in r and b in r and not (r[a] == 0 and r[b] == 0):
            g0.append(spring(a, b, d2(a, b),
                             None if sigma0 is None else sigma0.get((a, b))))

    # Group 1 is the bend set: the corners opposite each shared face edge, which is a little
    # under half the full 2-ring -- section 3.2a.  Pairs with both ends pinned are dropped by
    # the ring-0 test below, which is the whole of the corpus's selection rule.
    opp = collections.defaultdict(set)
    for f in faces:
        for i in range(3):
            opp[pair(f[i], f[(i + 1) % 3])].add(f[(i + 2) % 3])
    two = set()
    for corners in opp.values():
        cs = sorted(corners)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                two.add((cs[i], cs[j]))
    two -= fe
    g1 = []
    for a, b in sorted(two):
        if a in r and b in r and not (r[a] == 0 and r[b] == 0):
            g1.append(spring(a, b, slack * d2(a, b)))

    c.springs, c.ns0 = g0 + g1, len(g0)
    c.batch = batches(g0) + batches(g1)
    c.tris = list(collide) if collide is not None else collide_tris(faces, npin)

    ref, order = [], []
    seen = set()
    for f in faces:
        a, b = face_edges(f, seen)
        for e in (a, b):
            if e not in seen:
                seen.add(e)
                order.append(e)
        ref.append((a, b))
    c.ned0 = len(order)
    order += sorted(fe - seen)
    idx = {e: j for j, e in enumerate(order)}
    c.edges = order
    c.faces = [(idx[a], idx[b], f[0], f[1], f[2]) for f, (a, b) in zip(faces, ref)]
    c.blends = []
    return c


def collide_tris(faces, npin):
    """The +0x34 list Cloth_CollideWithVolumes_vtmb walks.  It builds a plane at tri[0] and
    pushes tri[1] and tri[2] onto the outside of it, so tri[0] is a witness that never moves.

    A face ships iff exactly one of its three corners sits at the face's minimum ring, ring
    being the graph distance from the pinned set over the face-edge graph -- section 3.2b.
    No face spans more than one ring step, 40938 of 40938, so a face's corners read (0,0,0),
    (0,0,1) or (0,1,1) as offsets from its own minimum, and only the last form ships.  The
    single inner corner is the witness and the two outer ones are the pair the solver pushes,
    which is outward from the pins -- the same orientation the mass split reads off the ring
    in section 3.3.  Emission order is by witness ring, then face index.

    That subsumes the three rules stated separately before it.  A face with two pinned corners
    reads (0,0,x) and never ships.  The witness is also the triple's minimum, because the
    particle array is non-decreasing in ring on 143 of 143 objects -- so a generator numbering
    particles differently has to use the ring and not min(f).  And what read as "one-pin
    triangles first, in face order" is the ring-0 band of this ordering.

    19996 of 19996 shipped triples in place and in file order, 144 of 144 objects
    byte-identical, 0 over-emitted over all 40938 faces, from the face list and npin alone."""
    n = max((max(f) for f in faces), default=-1) + 1
    adj = adjacency(tri_edges(faces), n)
    r = bfs(adj, [v for v in range(min(npin, n)) if v in adj])
    out = []
    for k, f in enumerate(faces):
        if any(v not in r for v in f):
            continue
        lo = min(r[v] for v in f)
        inner = [j for j in range(3) if r[f[j]] == lo]
        if len(inner) != 1:
            continue
        i = inner[0]
        out.append((lo, k, (f[i],) + tuple(f[j] for j in range(3) if j != i)))
    out.sort()
    return [tri for _, _, tri in out]


def batches(springs):
    """Four wide, and the builder consumes springs in file order with no indirection table --
    section 4 -- so the partition is just a run of counts over that order."""
    out, cur, live = [], 0, set()
    for v0, v1, _, _, _ in springs:
        if cur == 4 or v0 in live or v1 in live:
            out.append(cur)
            cur, live = 0, set()
        cur += 1
        live.update((v0, v1))
    if cur:
        out.append(cur)
    return out


def pack(c):
    """The nine blocks after a 0x5c header, every one 4-aligned, offsets object-relative."""
    body, off = [], HDR
    H = {}

    def put(key, data):
        nonlocal off
        if not data:
            H[key] = 0
            return
        while off % 4:
            body.append(b"\0")
            off += 1
        H[key] = off
        body.append(data)
        off += len(data)

    put(0x10, struct.pack("<%dH" % len(c.pv), *c.pv))
    put(0x20, b"".join(struct.pack("<2H3f", *s) for s in c.springs))
    put(0x2c, bytes(c.batch))
    put(0x34, b"".join(struct.pack("<3H", *t) for t in c.tris))
    put(0x40, b"".join(struct.pack("<2H", *e) for e in c.edges))
    put(0x48, b"".join(struct.pack("<5H", *f) for f in c.faces))
    put(0x50, b"".join(struct.pack("<2H2f", *x) for x in c.blends))
    H[0x54] = H[0x58] = 0

    ns1 = len(c.springs) - c.ns0
    nb0 = len(batches(c.springs[:c.ns0]))
    head = bytearray(HDR)
    struct.pack_into("<f", head, 0x00, c.scale)
    for k, v in ((0x04, c.numparticles), (0x08, c.numfixed), (0x0c, c.numfree),
                 (0x14, len(c.springs)), (0x18, c.ns0), (0x1c, ns1),
                 (0x24, nb0), (0x28, len(c.batch) - nb0),
                 (0x30, len(c.tris)), (0x38, len(c.edges)), (0x3c, c.ned0),
                 (0x44, len(c.faces)), (0x4c, len(c.blends))):
        struct.pack_into("<i", head, k, v)
    for k, v in H.items():
        struct.pack_into("<i", head, k, v)
    return bytes(head) + b"".join(body)




def region(blob, nvert, npart, flip):
    """`mstudiomodel_t`'s cloth as mdl_build's dict: table, object, three per-mesh arrays.

    `data` is laid out as one blob and every offset in the dict is relative to its start;
    `mdl_build.emit` places it and patches the table slots and the three `mstudiomesh_t`
    fields.  The nine payload offsets inside the object are object-relative and `pack`
    already wrote them, so nothing here touches the object's bytes.
    """
    if nvert > npart:
        raise Refused("a cloth region wants at least one particle per vertex, and this "
                      "one has %d vertices against %d particles" % (nvert, npart))
    table = struct.pack("<i", 4)                     # one slot, the object right after it
    # Bit 15 negates that vertex's normal, and 63 of the 84 shipped meshes set it on some
    # vertices and not others, so a shipped object needs one flag per vertex. A bool is
    # the whole mesh, which is what an authored sheet has.
    if isinstance(flip, (bool, int)):
        bits = [0x8000 if flip else 0] * nvert
    else:
        bits = [0x8000 if f else 0 for f in flip]
        if len(bits) != nvert:
            raise Refused("%d flip flags against %d vertices; +0x34 is one entry per "
                          "vertex" % (len(bits), nvert))
    # All three arrays start 4-aligned on 84 of 84 shipped meshes, so an odd vertex count
    # needs the ushort ones padded too, not only the documented byte one.
    def pad(x):
        return x + bytes(-len(x) % 4)
    own = pad(bytes(nvert))                          # column 0 for every vertex
    p34 = pad(b"".join(struct.pack("<H", v | bits[v]) for v in range(nvert)))
    p38 = pad(b"".join(struct.pack("<H", v) for v in range(nvert)))
    at_own = 4 + len(blob)
    at_p34 = at_own + len(own)
    at_p38 = at_p34 + len(p34)
    for name, off in (("object", 4), ("owner", at_own), ("particle", at_p34),
                      ("normal", at_p38)):
        if off % 4:
            raise Refused("the cloth region's %s array lands at %d, which is not "
                          "4-aligned; all three start 4-aligned on 84 of 84 shipped "
                          "meshes" % (name, off))
    return {"data": table + blob + own + p34 + p38, "cols": 1, "rows": 1, "table": 0,
            "slots": [(0, 4)], "meshes": {0: ([at_own, at_p34, at_p38], nvert)}}
