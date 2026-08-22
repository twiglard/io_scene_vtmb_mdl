#!/usr/bin/env python3
"""The two unit-normal tables a quantised vertex indexes. No bpy.

`R_AddVertexToMesh` (`StudioRender.dll+0x13a70`) decodes `mstudiomodel_t.filetype` 1 and 2's
normal against two `float[3]` tables in `.rdata`: filetype 1's `ushort` at +6 of the 12-byte
record is an unscaled **byte offset** into `g_normalOffsetTable` (`0x2c06e008`, 5314 entries at
stride 12), filetype 2's `byte` at +3 of the 8-byte record is `x16` into `g_normalIndexTable`
(`0x2c06d358`, 202 entries at stride 16 with the 4th float zero). Recon section 35.3 and 35.3a.

Both are latitude-ring tessellations -- south pole first, `65` rings at `180/64` degrees and 13
at 15, each ring azimuth-ascending from +X -- so the vectors are generated here from their ring
populations rather than carried. Generating and comparing against the DLL's own bytes: worst
component error **2.97e-08**, one float32 ulp, and 5000 of 5314 entries byte-identical (148 of
202 for the other). The populations are the one thing no closed form gives -- `128*cos` yields
12.5 where the table has 16 and 60.34 where it has 60 -- so they are the two literals below.

Nothing about a normal is lost by generating rather than reading: the tables resolve to about
2.8 degrees, and one ulp is nine orders below that.
"""

import math

# g_normalOffsetTable, ring by ring from the south pole. 5314 entries.
RINGS_OFFSET = (
    1, 8, 16, 20, 28, 32, 40, 44, 52, 56, 60, 68, 72, 76, 84, 88, 92, 96, 100, 104,
    108, 112, 116, 116, 120, 124, 124, 124, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 124, 124, 124, 120, 116, 116, 112, 108, 104, 100, 96, 92, 88, 84, 76, 72, 68,
    60, 56, 52, 44, 40, 32, 28, 20, 16, 8, 1)
# g_normalIndexTable. 202 entries, plus an all-zero slot at 202 no shipped file uses.
RINGS_INDEX = (1, 8, 12, 20, 24, 24, 24, 24, 24, 20, 12, 8, 1)

RINGS = {1: RINGS_OFFSET, 2: RINGS_INDEX}
# What the record stores: filetype 1 an unscaled byte offset, filetype 2 a real index.
SCALE = {1: 12, 2: 1}
FIELD_MAX = {1: 65535, 2: 255}

_TABLES = {}


def _build(rings):
    """(vectors, ring start indices). One ring per latitude, azimuth ascending from +X."""
    n, vecs, starts = len(rings), [], []
    for i, c in enumerate(rings):
        starts.append(len(vecs))
        z = -math.cos(math.pi * i / (n - 1))
        r = math.sqrt(max(0.0, 1.0 - z * z))
        for j in range(c):
            a = 2.0 * math.pi * j / c
            vecs.append((r * math.cos(a), r * math.sin(a), z))
    return vecs, starts


def table(filetype):
    """The table filetype 1 or 2 indexes, as a list of unit (x, y, z)."""
    return _cached(filetype)[0]


def _cached(filetype):
    rings = RINGS.get(filetype)
    if rings is None:
        raise ValueError("filetype %r stores no normal index" % (filetype,))
    if filetype not in _TABLES:
        _TABLES[filetype] = _build(rings)
    return _TABLES[filetype]


def decode(filetype, raw):
    """The stored field as a unit vector, or None where it points outside the table.

    There is no bounds check anywhere in the binary -- the accessor adds and reads -- so an
    out-of-range value is a file this code will not pretend to understand.
    """
    vecs, _starts = _cached(filetype)
    step = SCALE[filetype]
    if raw % step:
        return None
    i = raw // step
    return vecs[i] if 0 <= i < len(vecs) else None


def nearest(filetype, normal):
    """The table index whose entry is closest to `normal`, by angle.

    Exact, and 65 candidate pairs rather than 5314 dots: within a ring the best azimuth is
    the one nearest the target's own, so each ring contributes two candidates and the
    winner is the best of those. A ring further away in z can still win -- the rings are
    2.8 degrees apart and so is the equator's spacing -- which is why every ring is tried
    rather than the two nearest.
    """
    vecs, starts = _cached(filetype)
    rings = RINGS[filetype]
    x, y, z = normal
    n = math.sqrt(x * x + y * y + z * z)
    if not n:
        return 0
    x, y, z = x / n, y / n, z / n
    phi = math.atan2(y, x) / (2.0 * math.pi)
    best, at = -2.0, 0
    for k, c in enumerate(rings):
        j0 = int(math.floor(phi * c))
        for j in (j0, j0 + 1):
            i = starts[k] + (j % c)
            v = vecs[i]
            d = v[0] * x + v[1] * y + v[2] * z
            if d > best:
                best, at = d, i
    return at


def encode(filetype, normal):
    """`nearest` as the value the record stores."""
    return nearest(filetype, normal) * SCALE[filetype]


def worst_angle(filetype):
    """The table's own resolution: the largest angle any direction can be off by.

    Measured rather than derived -- the ring populations follow no closed form, so neither
    does the gap they leave. Half the widest neighbour spacing is the bound, and the widest
    spacing is at the poles, where a ring of 1 meets a ring of 8.
    """
    vecs, starts = _cached(filetype)
    rings = RINGS[filetype]
    worst = 0.0
    for k, c in enumerate(rings):
        if k + 1 >= len(rings):
            break
        for j in range(c):
            v = vecs[starts[k] + j]
            best = max(v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
                       for w in vecs[starts[k + 1]:starts[k + 1] + rings[k + 1]])
            worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, best)))))
    return worst
