#!/usr/bin/env python3
"""MDL v2531 per-vertex writer. No bpy.

Vertex records are fixed stride and index-parallel with what `Mdl.vertices` returns, so a
field is overwritten where it already sits inside the model's own vertex block. Laying that
block back into a file is `mdl_build`'s job. The count is fixed here -- the triangle lists
live in the `.vtx`, so this cannot add or remove geometry.

A filetype-0 vertex carries four bones. The fourth weight is not stored -- it is whatever
the first three leave short of 255 -- and byte +3 is a count code whose `% 5` is how many
of the four count, so only those low bits are ever rewritten. Evidence in
`ref/2531/studio-verified.h` and todo-vtmb-mdl-animation.md section 2.25.
"""

import os
import struct
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mdl as M
else:
    from . import mdl as M

FIELDS = ("positions", "normals", "uvs", "weights")

# Which fields the record can carry at all. filetype 1 and 2 store a normal *index* into
# a table this code does not parse, so a normal cannot be written for them.
BY_FILETYPE = {0: ("positions", "normals", "uvs", "weights"),
               1: ("positions", "uvs"),
               2: ("positions", "uvs")}


def supported(filetype, fields):
    ok = BY_FILETYPE.get(filetype, ())
    return tuple(f for f in fields if f in ok), tuple(f for f in fields if f not in ok)


def u8_weights(weights):
    """The first three of four weights as bytes; the fourth is the shortfall from 255 and
    is never stored, so the three are left short exactly when there is a fourth bone."""
    w = (list(weights) + [0.0] * 4)[:4]
    total = sum(w)
    if total > 1.0 + 1e-6:
        w = [x / total for x in w]
    b = [max(0, min(255, int(round(x * 255.0)))) for x in w[:3]]
    room = 255 - max(0, min(255, int(round(w[3] * 255.0))))
    over = sum(b) - room
    if over > 0:
        i = max(range(3), key=lambda k: b[k])
        b[i] = max(0, b[i] - over)
    return b


def count_code(old, numbones):
    """Byte +3 with its low `% 5` set to `numbones`. The high bits mean something this
    does not know, and are never zero for a given count, so they are kept."""
    return (old - old % 5) + (numbones % 5)


def quantise(v, offset, scale):
    return 0 if not scale else int(round((v - offset) / scale))


def _encode(data, model, verts, fields, base):
    """Write `fields` of `verts` as vertex records into `data` from `base`."""
    stride = M.VERTEX_STRIDE.get(model.filetype)
    if stride is None:
        raise ValueError("unknown vertex filetype %d" % model.filetype)
    if len(verts) != model.numvertices:
        raise ValueError("%s: %d vertices in the scene, %d in the file -- a mesh write "
                         "cannot change the count"
                         % (model.name, len(verts), model.numvertices))
    fields, _ = supported(model.filetype, fields)
    if not fields:
        return 0
    lim = 2 ** 16 - 1 if model.filetype == 1 else 255
    for i, v in enumerate(verts):
        o = base + i * stride
        if model.filetype == 0:
            if "positions" in fields:
                struct.pack_into("<3f", data, o + 12, *v.pos)
            if "normals" in fields and v.normal is not None:
                struct.pack_into("<3f", data, o + 24, *v.normal)
            if "uvs" in fields and v.uv is not None:
                struct.pack_into("<2f", data, o + 36, *v.uv)
            if "weights" in fields:
                data[o:o + 3] = bytes(u8_weights(v.weights))
                n = getattr(v, "numbones", None)
                data[o + 3] = count_code(data[o + 3],
                                         sum(1 for x in v.weights if x > 0)
                                         if n is None else n)
                struct.pack_into("<4h", data, o + 4,
                                 *([int(b) for b in list(v.bones)[:4]] + [0, 0, 0, 0])[:4])
        else:
            if "positions" in fields:
                q = [max(0, min(lim, quantise(v.pos[c], model.quant_offset[c],
                                              model.quant_scale[c]))) for c in range(3)]
                struct.pack_into("<3H" if model.filetype == 1 else "<3B", data, o, *q)
            if "uvs" in fields and v.uv is not None:
                uv = [max(0, min(65535, int(round(x * 65535.0)))) for x in v.uv]
                struct.pack_into("<2H", data,
                                 o + (8 if model.filetype == 1 else 4), *uv)
    return len(verts)


def pack_model(model, verts, fields, donor):
    """(bytes, records written) for one model's vertex block.

    `donor` is that block as the file holds it: a field left unticked keeps its stored
    value rather than being re-derived from the scene.
    """
    buf = bytearray(donor)
    n = _encode(buf, model, verts, fields, 0)
    return bytes(buf), n


def models_of(m):
    """Every (bodypart index, model index, bodypart, model) the file carries."""
    return [(bp.index, mo.index, bp, mo)
            for bp in m.bodyparts for mo in bp.models]
