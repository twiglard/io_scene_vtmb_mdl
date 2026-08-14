"""studiomdl's checksum, and stamping it into the three files that must agree.

`studiohdr_t.checksum` gates rendering.  `CStudioRender_R_StudioCreateStaticMeshes`
(StudioRender.dll:0x2c014660) compares `.vtx` +0x10 against `.mdl` +8 and allocates the
mesh array only inside the equal branch, so a model whose pair disagrees draws nothing.
**Only equality is tested** -- a value in no shipped file renders normally as long as every
file carries it (anomalies A3), so a writer picks its own number.

`studio_checksum` is studiomdl's own loop, transcribed from `write.cpp:1538`:

    phdr->checksum = 0;
    for (i = 0; i < total; i += 4)
        phdr->checksum = (phdr->checksum << 1)
                       + ((phdr->checksum & 0x8000000) ? 1 : 0)
                       + *((long *)(pStart + i));

Two things about that loop.  The carry is taken from **bit 27**, not bit 31, so the rotate
it was reaching for does not happen and the top four bits fall off the end.  And
`phdr->checksum` lives at `pStart + 8`, inside the buffer being summed and rewritten every
iteration, so the read at `i == 8` returns the running value: that step is `3c + carry` and
nothing else touches the field.

It does **not** reproduce the values in Troika's files -- `plans/cksum-reproduce.py` is the
check and expects 0 hits over the corpus, since VTMB's compiler is Troika's and only its
output survives.  That costs nothing: what a writer needs is one deterministic number
derived from what it emitted, so that a `.vtx` left over from an earlier build cannot pair
with a newly written `.mdl` and quietly render the wrong geometry.
"""

import struct

MDL_OFF = 8         # studiohdr_t.checksum
VTX_OFF = 0x10      # OptimizedModel::FileHeader_t.checkSum
PHY_OFF = 12        # phyheader_t: size, id, solidCount, checkSum

CARRY_BIT = 0x8000000
MASK = 0xffffffff


def studio_checksum(buf, length=None):
    """studiomdl's rolling sum over the first `length` bytes, as an unsigned 32-bit value.

    The buffer is summed with the checksum field itself zeroed first, which is what
    `phdr->checksum = 0` does before the loop starts.
    """
    n = len(buf) if length is None else min(length, len(buf))
    n -= n % 4
    words = struct.unpack_from("<%dI" % (n // 4), buf, 0)
    c = 0
    for k, w in enumerate(words):
        # word 2 is the checksum field: studiomdl reads back what the previous iteration
        # wrote there, so it sums the running value rather than anything on disk.
        if k == MDL_OFF // 4:
            w = c
        c = (c * 2 + (1 if c & CARRY_BIT else 0) + w) & MASK
    return c


def stamp_mdl(buf):
    """Checksum the .mdl in `buf` and write it into studiohdr_t. Returns the value."""
    b = bytearray(buf)
    struct.pack_into("<I", b, MDL_OFF, 0)
    c = studio_checksum(b)
    struct.pack_into("<I", b, MDL_OFF, c)
    buf[:] = b
    return c


def stamp_at(buf, off, value):
    """Write `value` into a companion file's checksum field."""
    struct.pack_into("<I", buf, off, value & MASK)


def read_at(buf, off):
    return struct.unpack_from("<I", buf, off)[0]
