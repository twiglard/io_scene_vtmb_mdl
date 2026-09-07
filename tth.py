#!/usr/bin/env python3
"""Reader for Troika's .tth/.ttz texture pair. No bpy.

The BC1/BC3 decode rules -- the two interpolated colours, the six interpolated alphas and
the four-plus-two form -- are Microsoft's, from the D3D9 block-compression pages and the
Khronos S3TC specification, both of which specify decompression only. S3TC names no
encoder, so every endpoint fit and index search here was designed for this module and no
existing compressor was consulted; `libtxc_dxtn` in particular was not read.

The pair is one VTF split in two: the .tth carries a small header, a mip offset table
and the VTF header with its low-res thumbnail and smallest mips; the .ttz is the rest,
raw zlib. Concatenating them reproduces the .vtf the engine would otherwise ship.

    0x00 "TTH\\0"  0x04 u16 version  0x06 u8 nmips  0x07 u8 mips inlined
    0x08 u32 blobsize                                  bytes of VTF that follow the table
    0x0c (u32 dst, u32 src)[nmips+1]                   smallest mip first, last is the end

dst is an offset into the reconstructed VTF, src into the .ttz. The .ttz is deflated with
a flush at every mip boundary, so src lets a caller inflate one mip without the rest.

`encode` and `build_vtf` are the write half. plans/tth-encode.py is the check that drives
them over the shipped corpus; it imports them from here rather than carrying a second copy.
"""

import os
import struct
import sys
import zlib

MAGIC = b"TTH\0"
VTF_MAGIC = b"VTF\0"

DXT1, DXT3, DXT5 = 13, 14, 15
BLOCK_BYTES = {DXT1: 8, DXT3: 16, DXT5: 16}
PIXEL_BYTES = {0: 4, 1: 4, 2: 3, 3: 3, 4: 2, 5: 1, 6: 2, 8: 1, 11: 4, 12: 4, 16: 4,
               22: 2, 23: 4}
FOURCC = {DXT1: b"DXT1", DXT3: b"DXT3", DXT5: b"DXT5"}

FORMAT_NAMES = {0: "RGBA8888", 1: "ABGR8888", 2: "RGB888", 3: "BGR888", 4: "RGB565",
                5: "I8", 6: "IA88", 8: "A8", 11: "ARGB8888", 12: "BGRA8888",
                13: "DXT1", 14: "DXT3", 15: "DXT5", 16: "BGRX8888",
                22: "UV88", 23: "UVWQ8888"}


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
    # D3D9 rounds the thirds -- (2*c0 + c1 + 1)/3 -- where a truncating divide does not, so
    # without the +1 every interpolated channel that is not exact comes out one low.
    a, b = _c565(c0), _c565(c1)
    if punchthrough and c0 <= c1:
        return (a, b, tuple((a[i] + b[i]) // 2 for i in range(3)), (0, 0, 0))
    return (a, b,
            tuple((2 * a[i] + b[i] + 1) // 3 for i in range(3)),
            tuple((a[i] + 2 * b[i] + 1) // 3 for i in range(3)))


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
                        tbl += [((6 - i) * a0 + (1 + i) * a1 + 3) // 7 for i in range(6)]
                    else:
                        tbl += [((4 - i) * a0 + (1 + i) * a1 + 2) // 5 for i in range(4)]
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
        elif fmt in (3, 16):
            # 16 is BGRX8888, imageloader.h:45 -- its fourth byte is undefined padding.
            out[p:p + 4] = bytes((s[2], s[1], s[0], 255))
        elif fmt == 2:
            out[p:p + 4] = bytes((s[0], s[1], s[2], 255))
        elif fmt == 23:
            # vtex writes U and V as (char)(source - 127), a literal 127 into W and a
            # copied alpha into Q, so only the first two invert, and modularly: source
            # 255 stores as -128.  W is 127 on all 1460224 pixels of the 11 shipped files.
            out[p:p + 4] = bytes(((s[0] + 127) & 0xFF, (s[1] + 127) & 0xFF, s[2], s[3]))
        else:
            raise ValueError("cannot decode format %d" % fmt)
    return out


def _q565(c):
    """An 8-bit RGB triple as the 565 word, rounded rather than truncated."""
    r = min(31, (c[0] * 31 + 127) // 255)
    g = min(63, (c[1] * 63 + 127) // 255)
    b = min(31, (c[2] * 31 + 127) // 255)
    return (r << 11) | (g << 5) | b


def _axis(px):
    """The block's principal colour axis, by power iteration on its covariance.

    A bounding-box diagonal is the cheaper choice and is wrong on the blocks that matter:
    a smooth gradient across one corner of the box gives a diagonal at 45 degrees to the
    colours actually present, and the two interpolated palette entries then sit off the
    line every pixel is on.
    """
    n = len(px)
    mean = [sum(p[c] for p in px) / n for c in range(3)]
    cov = [[0.0] * 3 for _ in range(3)]
    for p in px:
        d = [p[c] - mean[c] for c in range(3)]
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]
    v = [1.0, 1.0, 1.0]
    for _ in range(8):
        v = [sum(cov[i][j] * v[j] for j in range(3)) for i in range(3)]
        L = max(abs(c) for c in v)
        if L < 1e-12:
            return mean, (1.0, 0.0, 0.0)
        v = [c / L for c in v]
    L = sum(c * c for c in v) ** 0.5
    return mean, tuple(c / L for c in v)


def _fit_endpoints(px):
    """Two RGB endpoints spanning the block, refined against the 1/3 and 2/3 palette.

    The projection extremes alone put both endpoints on real pixels, which is right for
    the two ends and wrong for everything between: the four palette entries are at 0,
    1/3, 2/3 and 1 of the segment, so the least-squares refit below is what places them.
    """
    mean, ax = _axis(px)
    ts = [sum((p[c] - mean[c]) * ax[c] for c in range(3)) for p in px]
    lo, hi = min(ts), max(ts)
    a = [mean[c] + lo * ax[c] for c in range(3)]
    b = [mean[c] + hi * ax[c] for c in range(3)]
    for _ in range(2):
        # w is each pixel's position on the segment, so the normal equations below are
        # the ordinary least-squares fit of a line through the four palette weights.
        ws = []
        for p in px:
            den = sum((b[c] - a[c]) ** 2 for c in range(3))
            t = 0.0 if den < 1e-12 else                 sum((p[c] - a[c]) * (b[c] - a[c]) for c in range(3)) / den
            ws.append(min(1.0, max(0.0, round(t * 3.0) / 3.0)))
        s0 = sum((1.0 - w) ** 2 for w in ws)
        s1 = sum(w * w for w in ws)
        s01 = sum(w * (1.0 - w) for w in ws)
        det = s0 * s1 - s01 * s01
        if abs(det) < 1e-9:
            break
        for c in range(3):
            t0 = sum((1.0 - w) * p[c] for w, p in zip(ws, px))
            t1 = sum(w * p[c] for w, p in zip(ws, px))
            a[c] = (t0 * s1 - t1 * s01) / det
            b[c] = (t1 * s0 - t0 * s01) / det
    clamp = lambda v: int(min(255, max(0, round(v))))                    # noqa: E731
    return tuple(clamp(x) for x in a), tuple(clamp(x) for x in b)


def _bc1_block(px):
    """One 8-byte BC1 block from 16 RGB triples, always in the four-colour mode.

    c0 <= c1 is the three-colour mode, whose fourth entry is transparent black, so the
    words are ordered rather than emitted as fitted -- a swap costs the selectors their
    0<->1 and 2<->3 pairing and nothing else.
    """
    a, b = _fit_endpoints(px)
    c0, c1 = _q565(a), _q565(b)
    if c0 < c1:
        c0, c1 = c1, c0
    pal = _bc_colors(c0, c1, False) if c0 > c1 else (_c565(c0),) * 4
    idx = 0
    for i, p in enumerate(px):
        best, at = None, 0
        for k in range(4 if c0 > c1 else 1):
            e = sum((p[c] - pal[k][c]) ** 2 for c in range(3))
            if best is None or e < best:
                best, at = e, k
        idx |= at << (2 * i)
    return struct.pack("<HHI", c0, c1, idx)


def _bc3_alpha(av):
    """One 8-byte BC3 alpha block from 16 alpha bytes, always in the eight-value mode."""
    a0, a1 = max(av), min(av)
    if a0 == a1:
        return struct.pack("<BB", a0, a1) + b"\0" * 6
    # +3 for the same reason as _bc_colors' +1: D3D9's sevenths are (6*a0 + a1 + 3)/7.
    tbl = [a0, a1] + [((6 - i) * a0 + (1 + i) * a1 + 3) // 7 for i in range(6)]
    bits = 0
    for i, a in enumerate(av):
        best, at = None, 0
        for k, t in enumerate(tbl):
            e = (a - t) ** 2
            if best is None or e < best:
                best, at = e, k
        bits |= at << (3 * i)
    return struct.pack("<BB", a0, a1) + bits.to_bytes(6, "little")


def _block_pixels(w, h, rgba, bx, by):
    """The 16 RGBA tuples of one block, edge-replicated where the image runs out."""
    out = []
    for py in range(4):
        y = min(h - 1, by * 4 + py)
        for px_ in range(4):
            x = min(w - 1, bx * 4 + px_)
            o = (y * w + x) * 4
            out.append(tuple(rgba[o:o + 4]))
    return out


def compress(w, h, fmt, rgba):
    """Top-down RGBA8888 to DXT1 or DXT5 blocks. The inverse of `decode`."""
    if fmt not in (DXT1, DXT5):
        raise ValueError("compress writes DXT1 and DXT5, not format %d" % fmt)
    out = bytearray()
    for by in range(max(1, (h + 3) // 4)):
        for bx in range(max(1, (w + 3) // 4)):
            px = _block_pixels(w, h, rgba, bx, by)
            if fmt == DXT5:
                out += _bc3_alpha([p[3] for p in px])
            out += _bc1_block([p[:3] for p in px])
    return bytes(out)


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


VERSION = 1
HEADER = struct.Struct("<4sHBBI")
ENTRY = struct.Struct("<II")

ENVMAP = 0x4000
DEFAULT_INLINED = 3


def mip_chain(width, height, fmt, mipcount, frames, faces):
    """(level, w, h, bytes) per mip, smallest level first -- the order a VTF stores."""
    out = []
    for k in range(mipcount - 1, -1, -1):
        w, h = max(1, width >> k), max(1, height >> k)
        out.append((k, w, h, level_bytes(w, h, fmt) * frames * faces))
    return out


def vtf_geometry(vtf):
    """(header_size, width, height, fmt, mipcount, frames, faces) off a complete .vtf."""
    if vtf[:4] != VTF_MAGIC:
        raise ValueError("not a .vtf: magic %r" % vtf[:4])
    header_size, = struct.unpack_from("<I", vtf, 12)
    width, height, flags = struct.unpack_from("<HHI", vtf, 16)
    frames, = struct.unpack_from("<H", vtf, 24)
    fmt, = struct.unpack_from("<I", vtf, 52)
    return (header_size, width, height, fmt, vtf[56], max(1, frames),
            7 if flags & ENVMAP else 1)


def mip_offsets(vtf, nmips=None, control=None):
    """The dst column: nmips+1 ascending offsets, the last being len(vtf).

    `nmips` overrides the VTF header's own mip count, which 36 shipped pairs disagree
    with -- gimblesign.tth says 1 and stores 9.
    """
    header_size, width, height, fmt, mipcount, frames, faces = vtf_geometry(vtf)
    if control == "mipcount":
        nmips = None
    if control == "faces":
        faces = 1
    sizes = mip_chain(width, height, fmt, nmips or mipcount, frames, faces)
    total = sum(n for _, _, _, n in sizes)
    start = len(vtf) - total
    if start < header_size:
        raise ValueError("a %d-byte mip chain does not fit under a %d-byte file with a "
                         "%d-byte header" % (total, len(vtf), header_size))
    dst, o = [], start
    for _, _, _, n in sizes:
        dst.append(o)
        o += n
    dst.append(len(vtf))
    return dst, start - header_size


def encode(vtf, inlined=None, level=9, nmips=None, control=None):
    """(tth, ttz) from a complete .vtf, cut so that `inlined` mips stay in the .tth.

    A full flush precedes every mip after the first compressed one, so each src offset
    begins a stream a raw inflate can pick up alone.  Troika cut theirs with a SYNC flush
    instead: over 64027 shipped entries 15760 inflate alone and 45036 only with the
    earlier output as the window, so src there is a seek point and not an entry point.
    """
    dst, _ = mip_offsets(vtf, nmips, control)
    nmips = len(dst) - 1
    if inlined is None:
        inlined = min(DEFAULT_INLINED, max(0, nmips - 1))
    if not 0 <= inlined <= nmips:
        raise ValueError("inlined %d outside 0..%d" % (inlined, nmips))
    blob = dst[inlined]
    src = [0] * (nmips + 1)
    co = zlib.compressobj(level)
    out = bytearray()
    for i in range(inlined, nmips):
        if i > inlined:
            out += co.flush(zlib.Z_FULL_FLUSH)
        src[i] = len(out)
        out += co.compress(vtf[dst[i]:dst[i + 1]])
    out += co.flush()
    src[nmips] = len(out)
    tth = bytearray(HEADER.pack(MAGIC, VERSION, nmips, inlined, blob))
    for i in range(nmips + 1):
        tth += ENTRY.pack(dst[i], src[i])
    tth += vtf[:blob]
    return bytes(tth), bytes(out)


BGRA8888 = 12
VTF_HEADER = struct.Struct("<4sIIIHHIHH4x3f4xfIBiBBx")

ONEBITALPHA = 0x1000
EIGHTBITALPHA = 0x2000


def alpha_flags(rgba, fmt):
    """TEXTUREFLAGS_ONEBITALPHA / _EIGHTBITALPHA for these pixels in this format.

    One bit where the alpha channel holds only 0 and 255, eight where it holds anything
    else, neither for DXT1 -- the split 7593 of 7593 shipped DXT1 files, 2969 of 3014
    DXT5 and all 165 BGRA8888 carry.
    """
    if fmt == DXT1:
        return 0
    av = set(rgba[3::4])
    if av <= {255}:
        return 0
    return ONEBITALPHA if av <= {0, 255} else EIGHTBITALPHA


def thumb_extent(w, h):
    """The low-resolution thumbnail's extent: longest side capped at 16, aspect kept.

    10891 of the 10966 shipped thumbnails. Nothing is fitted to the 75 misses, which
    contradict each other -- 128x64 under a 256x128 image, and 1x1 under a 128x64 one.
    """
    s = max(1, max(w, h) // 16)
    return max(1, w // s), max(1, h // s)


def _box(w, h, rgba, nw, nh):
    """Top-down RGBA resampled to nw x nh by averaging each source rectangle.

    Repeated _halve reaches the extent on every power-of-two pair and on 42871 of the
    1048576 pairs up to 1024x1024; 17x5 halves to 8x2 where the rule says 17x5.
    """
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        y0, y1 = y * h // nh, max(y * h // nh + 1, (y + 1) * h // nh)
        for x in range(nw):
            x0, x1 = x * w // nw, max(x * w // nw + 1, (x + 1) * w // nw)
            acc, n = [0, 0, 0, 0], 0
            for sy in range(y0, y1):
                row = sy * w
                for sx in range(x0, x1):
                    o = (row + sx) * 4
                    for c in range(4):
                        acc[c] += rgba[o + c]
                    n += 1
            o = (y * nw + x) * 4
            for c in range(4):
                out[o + c] = acc[c] // n
    return bytes(out)


def _halve(w, h, rgba):
    nw, nh = max(1, w // 2), max(1, h // 2)
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        r0, r1 = min(2 * y, h - 1) * w, min(2 * y + 1, h - 1) * w
        for x in range(nw):
            c0, c1 = min(2 * x, w - 1), min(2 * x + 1, w - 1)
            o = (y * nw + x) * 4
            for c in range(4):
                out[o + c] = (rgba[(r0 + c0) * 4 + c] + rgba[(r0 + c1) * 4 + c]
                              + rgba[(r1 + c0) * 4 + c] + rgba[(r1 + c1) * 4 + c]) // 4
    return nw, nh, bytes(out)


def _bgra(rgba):
    b = bytearray(rgba)
    b[0::4], b[2::4] = rgba[2::4], rgba[0::4]
    return bytes(b)


def pick_format(rgba):
    """DXT5 where the image uses alpha at all, DXT1 where it does not.

    DXT1 has no alpha in the four-colour mode this writes, and the three-colour one
    spells only fully transparent, so anything between is DXT5's to carry.
    """
    return DXT5 if any(a != 255 for a in rgba[3::4]) else DXT1


def build_vtf(w, h, rgba, flags=0, fmt=None, thumbnail=True):
    """A complete VTF 7.1 from top-down RGBA. `fmt` None picks DXT1 or DXT5 by the alpha.

    The alpha flags follow the pixels, and a DXT1 low-resolution thumbnail sits between
    the header and the mip chain -- 10966 of 11001 readable shipped pairs carry one, 20
    write lowResImageFormat -1 (what `thumbnail=False` emits) and 15 a zero extent.
    """
    if len(rgba) != w * h * 4:
        raise ValueError("%dx%d needs %d bytes, got %d" % (w, h, w * h * 4, len(rgba)))
    if fmt is None:
        fmt = pick_format(rgba)
    if fmt not in (DXT1, DXT5, BGRA8888):
        raise ValueError("build_vtf writes DXT1, DXT5 or BGRA8888, not format %d" % fmt)
    chain = [(w, h, rgba)]
    cw, ch, cur = w, h, rgba
    while cw > 1 or ch > 1:
        cw, ch, cur = _halve(cw, ch, cur)
        chain.append((cw, ch, cur))
    n = len(chain)
    px = len(rgba) // 4
    refl = tuple(sum(rgba[c::4]) / (255.0 * px) for c in range(3))
    flags |= alpha_flags(rgba, fmt)
    low = b""
    lf, lw, lh = -1, 0, 0
    if thumbnail:
        lf = DXT1
        lw, lh = thumb_extent(w, h)
        low = compress(lw, lh, DXT1, _box(w, h, rgba, lw, lh))
    head = VTF_HEADER.pack(VTF_MAGIC, 7, 1, VTF_HEADER.size, w, h, flags, 1, 0,
                           refl[0], refl[1], refl[2], 1.0, fmt, n, lf, lw, lh)
    body = [(_bgra(c[2]) if fmt == BGRA8888 else compress(c[0], c[1], fmt, c[2]))
            for c in reversed(chain)]
    return head + low + b"".join(body)


def write_pair(stem, w, h, rgba, flags=0, overwrite=False, fmt=None):
    """The `.tth` and `.ttz` at `stem`, from top-down RGBA. True if they were written.

    Either half already on disk stops both: the pair is one .vtf cut in two, and a fresh
    .tth beside a stale .ttz reconstructs a file that is neither. A texture under the game
    tree may be Troika's, so nothing is replaced unless `overwrite`.
    """
    if not overwrite and (os.path.exists(stem + ".tth")
                          or os.path.exists(stem + ".ttz")):
        return False
    head, rest = encode(build_vtf(w, h, rgba, flags, fmt))
    parent = os.path.dirname(stem)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(stem + ".tth", "wb") as f:
        f.write(head)
    with open(stem + ".ttz", "wb") as f:
        f.write(rest)
    return True


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
