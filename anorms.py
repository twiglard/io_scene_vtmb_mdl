#!/usr/bin/env python3
"""The packed-vertex normal tables, read out of the user's own StudioRender.dll. No bpy.

A filetype 1 or 2 vertex stores its normal as a reference into one of two float tables
compiled into StudioRender.dll -- Troika's data, so it is read from the install at import
time and never shipped here, the same line the textures sit behind. Both dereferences are
in the vertex accessor at 0x2c013a70, in the branches that decode the packed positions:

    filetype 1    MOVZX EAX, word [vert+6]
                  ADD   EAX, 0x2c06e008        the u16 is a BYTE OFFSET, unscaled,
                                               into 5314 float[3] entries
    filetype 2    MOVZX EAX, byte [vert+3]
                  SHL   EAX, 0x4
                  ADD   EAX, 0x2c06d358        an INDEX into 203 entries at stride 16,
                                               the fourth float being padding

The counts are pinned by the data on both ends: every filetype-1 u16 in a patched
install is divisible by 12 and spans 0..63756 exactly (5314 entries), and the gap
between the two tables is 0xcb0 = 203 * 16 while the observed filetype-2 index tops out
at 201. Entry 0 of both tables is (0, 0, -1) and the entries are unit vectors, which is
what load() checks -- a rebuilt or different-build DLL fails by name instead of shading
garbage off whatever bytes sit at those addresses.
"""

import os
import struct

IMAGE = "StudioRender.dll"
ANORM_LONG = 0x2C06E008
ANORM_SHORT = 0x2C06D358
N_LONG = 5314
N_SHORT = (ANORM_LONG - ANORM_SHORT) // 16


def _va_reader(d):
    """Map a virtual address into `d` through the PE section table."""
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    if d[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE image")
    nsec = struct.unpack_from("<H", d, pe + 6)[0]
    optsz = struct.unpack_from("<H", d, pe + 20)[0]
    imgbase = struct.unpack_from("<I", d, pe + 24 + 28)[0]
    secs = []
    for i in range(nsec):
        o = pe + 24 + optsz + i * 40
        vsz, va, rsz, ptr = struct.unpack_from("<4I", d, o + 8)
        secs.append((imgbase + va, max(vsz, rsz), ptr))

    def file_off(va):
        for base, size, ptr in secs:
            if base <= va < base + size:
                return ptr + (va - base)
        raise ValueError("VA %#x is not mapped by any section" % va)

    return file_off


class Anorms:
    """Both tables, loaded once and validated."""

    __slots__ = ("path", "long", "short")

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            d = f.read()
        file_off = _va_reader(d)
        lo, so = file_off(ANORM_LONG), file_off(ANORM_SHORT)
        self.long = [struct.unpack_from("<3f", d, lo + i * 12) for i in range(N_LONG)]
        self.short = [struct.unpack_from("<3f", d, so + i * 16) for i in range(N_SHORT)]
        for name, table in (("filetype-1", self.long), ("filetype-2", self.short)):
            if table[0] != (0.0, 0.0, -1.0):
                raise ValueError("%s: %s table entry 0 is %r, not (0, 0, -1) -- not the "
                                 "known build's normal tables" % (path, name, table[0]))
            unit = sum(1 for n in table
                       if abs(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0) < 1e-3)
            if unit < 0.99 * len(table):
                raise ValueError("%s: only %d of %d %s entries are unit length -- not "
                                 "the known build's normal tables"
                                 % (path, unit, len(table), name))

    def normal(self, filetype, ref):
        """The float[3] a packed vertex's stored reference names."""
        if filetype == 1:
            if ref % 12 or not 0 <= ref < N_LONG * 12:
                raise ValueError("filetype-1 normal offset %d is not a multiple of 12 "
                                 "inside %d entries" % (ref, N_LONG))
            return self.long[ref // 12]
        if filetype == 2:
            if not 0 <= ref < N_SHORT:
                raise ValueError("filetype-2 normal index %d of %d" % (ref, N_SHORT))
            return self.short[ref]
        raise ValueError("filetype %d stores no packed normal" % filetype)


def find(roots):
    """The install's StudioRender.dll, or None. The game root holds Bin/ beside
    Vampire/, and a content root is usually Vampire/ itself, so both each root and its
    parent are tried; the name is matched case-insensitively for a mounted install."""
    for r in roots:
        for base in (r, os.path.dirname(r)):
            bindir = os.path.join(base, "Bin")
            if not os.path.isdir(bindir):
                continue
            try:
                names = os.listdir(bindir)
            except OSError:
                continue
            for n in names:
                if n.lower() == IMAGE.lower():
                    return os.path.join(bindir, n)
    return None


