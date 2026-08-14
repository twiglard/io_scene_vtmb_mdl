#!/usr/bin/env python3
"""Reader for the .vtx strip file that carries MDL v2531 triangles. No bpy.

The .mdl holds vertices but no indices; faces live only here. Clean room: struct sizes
were measured off shipped files before any header was consulted. See
plans/todo-ghidra-vtmb-recon.md section 21.
"""

import struct

SG_STRIDE = 20
STRIP_STRIDE = 16
STRIP_IS_TRILIST = 0x01
STRIP_IS_TRISTRIP = 0x02

VTX_VERSION = 107
# StripGroupHeader_t.flags, the BYTE at +6; bits 1/2/4 are flexed/hwskinned/delta-flexed.
# StudioRender.dll+0x13fa0 tests 0x08, +0x144d0 tests 0x10 with the opposite polarity.
SG_VERTS_ARE_BONED = 0x08   # 12-byte Vertex1_t, origMeshVertID at +10
SG_VERTS_ARE_PLAIN = 0x10   # 2-byte Vertex0_t, a bare array of ids


def _u(d, off, fmt):
    return struct.unpack_from("<" + fmt, d, off)


class StripGroup:
    __slots__ = ("model", "lod", "mesh", "numverts", "vertbase", "indexbase",
                 "stripbase", "numindices", "numstrips", "stride", "flags")


class Vtx:
    def __init__(self, path, data=None):
        if data is None:
            with open(path, "rb") as f:
                data = f.read()
        self.d = d = data
        (self.version, self.vertcachesize, self.maxbones_strip, self.maxbones_tri,
         self.maxbones_vert, self.checksum, self.numlods,
         self.matreplaceoffset, numbodyparts, bodypartoffset) = _u(d, 0, "iiHHiiiiii")
        if self.version != VTX_VERSION:
            raise ValueError("%s is .vtx version %d, not %d"
                             % (path, self.version, VTX_VERSION))

        self.groups = []
        model_i = 0
        for i in range(numbodyparts):
            bo = bodypartoffset + i * 8
            nmodels, modeloffset = _u(d, bo, "2i")
            for j in range(nmodels):
                mo = bo + modeloffset + j * 8
                nlods, lodoffset = _u(d, mo, "2i")
                for l in range(nlods):
                    lo = mo + lodoffset + l * 12
                    nmesh, meshoffset = _u(d, lo, "2i")
                    for k in range(nmesh):
                        eo = lo + meshoffset + k * 8
                        nsg, _mflags, sgoffset = _u(d, eo, "hhi")
                        for g in range(nsg):
                            self.groups.append(
                                self._read_group(eo + sgoffset + g * SG_STRIDE,
                                                 model_i, l, k))
                model_i += 1
        self._resolve_strides()

    def _read_group(self, off, model_i, lod, mesh):
        d, sg = self.d, StripGroup()
        sg.model, sg.lod, sg.mesh = model_i, lod, mesh
        # The engine reads flags as a byte; the one at +7 is a separate field it ignores.
        sg.numverts, sg.numindices, sg.numstrips, sg.flags, _ = _u(d, off, "3hBB")
        voff, ioff, soff = _u(d, off + 8, "3i")
        sg.vertbase, sg.indexbase, sg.stripbase = off + voff, off + ioff, off + soff
        sg.stride = None
        return sg

    def _resolve_strides(self):
        """Vertex stride is 2 or 12, chosen per strip group by the flags byte. The .mdl's
        filetype does not predict it (measured: 2764 groups disagree) and neither does the
        distance to the next array, which is only an upper bound."""
        for g in self.groups:
            boned = bool(g.flags & SG_VERTS_ARE_BONED)
            # The engine decides this twice off two bits and would read such a group
            # inconsistently; every shipped group sets exactly one.
            if g.numverts > 0 and boned == bool(g.flags & SG_VERTS_ARE_PLAIN):
                raise ValueError("stripgroup flags %#04x sets neither vertex format nor "
                                 "one alone" % g.flags)
            g.stride = 12 if boned else 2

    def orig_vert_ids(self, sg):
        """Stripgroup vertex -> index into the owning mesh's vertex range.
        origMeshVertID is the last short of the record in both strides."""
        d, out = self.d, []
        for q in range(sg.numverts):
            out.append(_u(d, sg.vertbase + q * sg.stride + sg.stride - 2, "h")[0])
        return out

    def triangles(self, sg):
        """Stripgroup-local vertex triples, winding as the engine draws them."""
        d, out = self.d, []
        idx = _u(d, sg.indexbase, "%dH" % sg.numindices) if sg.numindices else ()
        for s in range(sg.numstrips):
            so = sg.stripbase + s * STRIP_STRIDE
            numindices, indexoffset = _u(d, so, "2h")
            flags = d[so + 9]
            run = idx[indexoffset:indexoffset + numindices]
            if flags & STRIP_IS_TRILIST:
                out += [run[t:t + 3] for t in range(0, len(run) - 2, 3)]
            elif flags & STRIP_IS_TRISTRIP:
                for t in range(len(run) - 2):
                    tri = (run[t], run[t + 2], run[t + 1]) if t & 1 \
                        else (run[t], run[t + 1], run[t + 2])
                    if len(set(tri)) == 3:
                        out.append(tri)
            else:
                raise ValueError("strip flags %#x is neither list nor strip" % flags)
        return out


# Preference order; dx80 is the richest variant every model has.
SUFFIXES = (".dx80.vtx", ".dx90.vtx", ".dx7_2bone.vtx", ".sw.vtx")


def companion_paths(mdl_path):
    stem = mdl_path[:-4] if mdl_path.lower().endswith(".mdl") else mdl_path
    return [stem + s for s in SUFFIXES]
