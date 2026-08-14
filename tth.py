#!/usr/bin/env python3
"""Reader for Troika's .tth/.ttz texture pair. No bpy.

The pair is one VTF split in two: the .tth carries a small header, a mip offset table
and the VTF header with its low-res thumbnail and smallest mips; the .ttz is the rest,
raw zlib. Concatenating them reproduces the .vtf the engine would otherwise ship.

    0x00 "TTH\\0"  0x04 u16 version  0x06 u8 nmips  0x07 u8 mips inlined
    0x08 u32 blobsize                                  bytes of VTF that follow the table
    0x0c (u32 dst, u32 src)[nmips+1]                   smallest mip first, last is the end

dst is an offset into the reconstructed VTF, src into the .ttz. The .ttz is deflated with
a flush at every mip boundary, so src lets a caller inflate one mip without the rest.
"""

import struct
import sys
import zlib

MAGIC = b"TTH\0"
VTF_MAGIC = b"VTF\0"

DXT1, DXT3, DXT5 = 13, 14, 15
BLOCK_BYTES = {DXT1: 8, DXT3: 16, DXT5: 16}
PIXEL_BYTES = {0: 4, 1: 4, 2: 3, 3: 3, 4: 2, 5: 1, 6: 2, 8: 1, 11: 4, 12: 4, 16: 2}
FOURCC = {DXT1: b"DXT1", DXT3: b"DXT3", DXT5: b"DXT5"}

FORMAT_NAMES = {0: "RGBA8888", 1: "ABGR8888", 2: "RGB888", 3: "BGR888", 4: "RGB565",
                5: "I8", 6: "IA88", 8: "A8", 11: "ARGB8888", 12: "BGRA8888",
                13: "DXT1", 14: "DXT3", 15: "DXT5", 16: "UV88"}


def inflate(ttz):
    """Three shipped .ttz carry a wrong adler32 trailer but sound deflate blocks, so a
    checksum failure falls back to whatever the stream already produced."""
    if not ttz:
        return b""
    try:
        return zlib.decompress(ttz)
    except zlib.error:
        return zlib.decompressobj(-zlib.MAX_WBITS).decompress(ttz[2:])


def level_bytes(w, h, fmt):
    if fmt in BLOCK_BYTES:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * BLOCK_BYTES[fmt]
    if fmt not in PIXEL_BYTES:
        raise ValueError("unsupported image format %d" % fmt)
    return w * h * PIXEL_BYTES[fmt]


class Tth:
    def __init__(self, data):
        if data[:4] != MAGIC:
            raise ValueError("not a .tth: magic %r" % data[:4])
        self.d = data
        self.version, self.nmips, self.inlined, self.blob = struct.unpack_from(
            "<HBBI", data, 4)
        self.table = [struct.unpack_from("<II", data, 12 + i * 8)
                      for i in range(self.nmips + 1)]
        self.vtf_at = 12 + (self.nmips + 1) * 8
        if len(data) - self.vtf_at != self.blob:
            raise ValueError("blob says %d, %d bytes follow the table"
                             % (self.blob, len(data) - self.vtf_at))
        v = data[self.vtf_at:]
        if v[:4] != VTF_MAGIC:
            raise ValueError("no VTF header at %d: %r" % (self.vtf_at, v[:4]))
        self.vtf_major, self.vtf_minor, self.header_size = struct.unpack_from("<3I", v, 4)
        self.width, self.height, self.flags = struct.unpack_from("<HHI", v, 16)
        self.frames, self.first_frame = struct.unpack_from("<HH", v, 24)
        self.format, = struct.unpack_from("<I", v, 52)
        self.mipcount = v[56]

    def __repr__(self):
        return "<Tth %dx%d %s mips=%d frames=%d>" % (
            self.width, self.height, FORMAT_NAMES.get(self.format, self.format),
            self.nmips, self.frames)

    def _slice(self, ttz_raw, start, end):
        head = self.d[self.vtf_at:]
        if end <= self.blob:
            return head[start:end]
        if start >= self.blob:
            return ttz_raw[start - self.blob:end - self.blob]
        return head[start:] + ttz_raw[:end - self.blob]

    def top_mip(self, ttz):
        """(level, width, height, bytes) for the largest mip the pair actually stores.

        Taken as the tail of the reconstructed file, since VTF orders mips smallest
        first. The offset table would say the same thing and say it without inflating,
        but it is wrong on files the Unofficial Patch rewrote -- eyeball.tth puts its
        last two entries at the same offset, which would make the top mip empty."""
        raw = inflate(ttz)
        total = self.blob + len(raw)
        frames = max(1, self.frames)
        for k in range(0, 16):
            w, h = max(1, self.width >> k), max(1, self.height >> k)
            n = level_bytes(w, h, self.format) * frames
            if n <= total - self.header_size:
                return k, w, h, self._slice(raw, total - n, total)
            if w == 1 and h == 1:
                break
        raise ValueError("%dx%d %s over %d frames does not fit %d bytes"
                         % (self.width, self.height,
                            FORMAT_NAMES.get(self.format, self.format), frames, total))

    def vtf(self, ttz):
        """The whole reconstructed .vtf."""
        return self.d[self.vtf_at:] + inflate(ttz)


DDS_HEADER = struct.Struct("<4sIIIIIII44sII4sIIIIIIIIII")


def to_dds(w, h, fmt, data):
    """A single-level .dds carrying one mip. DDS orders mips largest-first and VTF
    smallest-first, so emitting the whole chain would mean reversing it for no gain."""
    caps, pitch = 0x1000, len(data)
    if fmt in FOURCC:
        flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000
        pf = (32, 0x4, FOURCC[fmt], 0, 0, 0, 0, 0)
    elif fmt in (12, 0):
        flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x8
        pitch = w * 4
        masks = ((0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000) if fmt == 12
                 else (0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
        pf = (32, 0x41, b"\0\0\0\0", 32) + masks
    elif fmt == 3:
        flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x8
        pitch = w * 3
        pf = (32, 0x40, b"\0\0\0\0", 24, 0x00ff0000, 0x0000ff00, 0x000000ff, 0)
    else:
        raise ValueError("no .dds equivalent for format %d" % fmt)
    return DDS_HEADER.pack(b"DDS ", 124, flags, h, w, pitch, 0, 1,
                           b"\0" * 44, *pf, caps, 0, 0, 0, 0) + data


def _c565(c):
    r, g, b = (c >> 11) & 31, (c >> 5) & 63, c & 31
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def _bc_colors(c0, c1, punchthrough):
    a, b = _c565(c0), _c565(c1)
    if punchthrough and c0 <= c1:
        return (a, b, tuple((a[i] + b[i]) // 2 for i in range(3)), (0, 0, 0))
    return (a, b,
            tuple((2 * a[i] + b[i]) // 3 for i in range(3)),
            tuple((a[i] + 2 * b[i]) // 3 for i in range(3)))


def decode(w, h, fmt, data):
    """Top-down RGBA8888, one byte per channel."""
    out = bytearray(w * h * 4)
    if fmt in BLOCK_BYTES:
        step = BLOCK_BYTES[fmt]
        bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
        for by in range(bh):
            for bx in range(bw):
                o = (by * bw + bx) * step
                blk = data[o:o + step]
                if len(blk) < step:
                    break
                alpha = None
                if fmt == DXT5:
                    a0, a1 = blk[0], blk[1]
                    bits = int.from_bytes(blk[2:8], "little")
                    tbl = [a0, a1]
                    if a0 > a1:
                        tbl += [((6 - i) * a0 + (1 + i) * a1) // 7 for i in range(6)]
                    else:
                        tbl += [((4 - i) * a0 + (1 + i) * a1) // 5 for i in range(4)]
                        tbl += [0, 255]
                    alpha = [tbl[(bits >> (3 * i)) & 7] for i in range(16)]
                elif fmt == DXT3:
                    bits = int.from_bytes(blk[0:8], "little")
                    alpha = [((bits >> (4 * i)) & 15) * 17 for i in range(16)]
                cofs = 0 if fmt == DXT1 else 8
                c0, c1 = struct.unpack_from("<HH", blk, cofs)
                idx = struct.unpack_from("<I", blk, cofs + 4)[0]
                pal = _bc_colors(c0, c1, fmt == DXT1)
                for py in range(4):
                    y = by * 4 + py
                    if y >= h:
                        break
                    for px in range(4):
                        x = bx * 4 + px
                        if x >= w:
                            continue
                        i = py * 4 + px
                        sel = (idx >> (2 * i)) & 3
                        r, g, b = pal[sel]
                        p = (y * w + x) * 4
                        out[p:p + 3] = bytes((r, g, b))
                        out[p + 3] = (alpha[i] if alpha is not None
                                      else (0 if fmt == DXT1 and c0 <= c1 and sel == 3
                                            else 255))
        return out
    n = PIXEL_BYTES.get(fmt)
    if n is None:
        raise ValueError("cannot decode format %d" % fmt)
    for i in range(w * h):
        s, p = data[i * n:i * n + n], i * 4
        if fmt == 12:
            out[p:p + 4] = bytes((s[2], s[1], s[0], s[3]))
        elif fmt == 0:
            out[p:p + 4] = s
        elif fmt == 3:
            out[p:p + 4] = bytes((s[2], s[1], s[0], 255))
        elif fmt == 2:
            out[p:p + 4] = bytes((s[0], s[1], s[2], 255))
        else:
            raise ValueError("cannot decode format %d" % fmt)
    return out


def to_tga(w, h, rgba):
    """Bottom-up 32-bit BGRA, the layout the SDK's own extraction uses."""
    hdr = struct.pack("<3B2HB4H2B", 0, 0, 2, 0, 0, 0, 0, 0, w, h, 32, 8)
    rows = []
    for y in range(h - 1, -1, -1):
        row = rgba[y * w * 4:(y + 1) * w * 4]
        bgra = bytearray(len(row))
        bgra[0::4], bgra[1::4], bgra[2::4], bgra[3::4] = (
            row[2::4], row[1::4], row[0::4], row[3::4])
        rows.append(bytes(bgra))
    return hdr + b"".join(rows)


def _cli(argv):
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vpk

    if len(argv) < 3:
        print("usage: tth.py <game dir> <materials/...> [out.tga|out.dds]")
        return 2
    root, rel = argv[1], argv[2].replace("\\", "/").lower()
    if rel.endswith((".tth", ".ttz", ".vtf")):
        rel = rel[:-4]
    v = vpk.Vpks(root)

    def get(ext):
        p = os.path.join(root, (rel + ext).replace("/", os.sep))
        if os.path.isfile(p):
            return open(p, "rb").read()
        return v.read(rel + ext) if rel + ext in v.index else None

    head = get(".tth")
    if head is None:
        print("no %s.tth under %s" % (rel, root))
        return 1
    t = Tth(head)
    level, w, h, blob = t.top_mip(get(".ttz"))
    print("%r  top mip level %d, %dx%d, %d B" % (t, level, w, h, len(blob)))
    if len(argv) > 3:
        out = argv[3]
        data = (to_dds(w, h, t.format, blob) if out.lower().endswith(".dds")
                else to_tga(w, h, decode(w, h, t.format, blob)))
        with open(out, "wb") as f:
            f.write(data)
        print("wrote %s, %d B" % (out, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
