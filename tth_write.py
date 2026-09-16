#!/usr/bin/env python3
"""Writer for Troika's .tth/.ttz texture pair -- the inverse of tth.py. No bpy.

The container, with the fields a writer needs rather than the ones a reader can get
away with:

    .tth  "TTH\\0"                              4 bytes
          uint16 version         = 1
          uint8  mip_count                      full chain: 512x512 -> 10
          uint8  inline_mips                    the smallest N mips kept in the .tth
          uint32 vtf_blob_len                   bytes from the "VTF\\0" marker to EOF
          (mip_count + 1) x { uint32 raw_offset; uint32 ttz_prefix }
          <standard VTF 7.1 header, 64 bytes>
          <low-res thumbnail, DXT1 at lowres_dims -- 128 bytes for the common 16x16>
          <the inline_mips smallest mips>

    .ttz  one zlib stream: the mips the .tth does not carry, smallest -> largest,
          with a Z_SYNC_FLUSH between each.

The two table columns are the part that is not guessable. `raw_offset` locates a mip in
the reconstructed [header][thumbnail][mips] image counted from the "VTF\\0" marker, so
the first (smallest) mip sits at 64 plus the thumbnail -- 192 for the common 16x16
texture and above, less for the small and narrow. `ttz_prefix` is how many *compressed*
bytes precede that mip -- the flush points are what make
zlib.decompressobj().decompress(ttz[:prefix]) yield exactly the mips before it, which is
how the engine pulls one mip without inflating the rest. Entry mip_count is the pair of
totals: the end offset and the whole .ttz length. Inline mips carry prefix 0.

Both shipped arrangements come out of encode(): DXT with the three smallest mips inline
(retail) and uncompressed with none (the Unofficial Patch's re-exports). Frames stay 1 --
a 7-face cubemap is a different authoring problem and is refused rather than half done.

The DXT compressor is a range fit mapped through tth.py's own palette arithmetic, so
decode(encode(x)) round-trips under one interpolation rule. Confirmed in game: an
authored pair drawn by retail, VtMB having no loose .vtf path to smuggle one in through.
"""

import os
import struct
import sys
import zlib

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tth as T
else:
    from . import tth as T

VTF_HEADER_SIZE = 64

ENCODABLE = (0, 2, 3, 12, T.DXT1, T.DXT3, T.DXT5)


def lowres_dims(w, h):
    """The thumbnail's dimensions: the texture halved until the larger side is <= 16.
    Measured over all 10,667 installed .tth: 10,648 carry a DXT1 thumbnail at exactly
    these dims, and the remaining 19 are the envmap cubemaps, which carry none at all
    (lowResFormat -1, 0x0)."""
    while max(w, h) > 16:
        w, h = max(1, w // 2), max(1, h // 2)
    return w, h


def _to565(r, g, b):
    return (((r * 31 + 127) // 255) << 11) | (((g * 63 + 127) // 255) << 5) \
        | ((b * 31 + 127) // 255)


# Endpoint weights per selector, in the order tth._bc_colors builds its palette:
# entry 0 is c0, 1 is c1, 2 is 2/3 c0 + 1/3 c1, 3 is 1/3 c0 + 2/3 c1.
_COLOR_W = ((1.0, 0.0), (0.0, 1.0), (2 / 3.0, 1 / 3.0), (1 / 3.0, 2 / 3.0))
_ALPHA_W = tuple([(1.0, 0.0), (0.0, 1.0)]
                 + [((6 - i) / 7.0, (1 + i) / 7.0) for i in range(6)])


def _assign(px16, pal):
    return [min(range(len(pal)),
                key=lambda k: sum((p[c] - pal[k][c]) ** 2 for c in range(len(p))))
            for p in px16]


def _refit(px16, sel, weights, dims):
    """Least-squares endpoints given an assignment: minimise sum |a*e0 + b*e1 - p|^2
    per channel, the standard cluster-fit half of a DXT compressor."""
    aa = sum(weights[s][0] * weights[s][0] for s in sel)
    ab = sum(weights[s][0] * weights[s][1] for s in sel)
    bb = sum(weights[s][1] * weights[s][1] for s in sel)
    det = aa * bb - ab * ab
    if abs(det) < 1e-8:
        return None
    e0, e1 = [], []
    for c in range(dims):
        ap = sum(weights[s][0] * p[c] for s, p in zip(sel, px16))
        bp = sum(weights[s][1] * p[c] for s, p in zip(sel, px16))
        e0.append(max(0, min(255, int(round((bp * -ab + ap * bb) / det)))))
        e1.append(max(0, min(255, int(round((ap * -ab + bp * aa) / det)))))
    return e0, e1


def _sse(px, pal, sel):
    return sum((p[c] - pal[s][c]) ** 2 for p, s in zip(px, sel) for c in range(3))


def _color_half(px16):
    """The 8-byte DXT color half: two endpoint seeds -- the channel bounding box and
    the farthest pixel pair -- each refined by least-squares passes, keeping whichever
    lands the smallest error. Pixels map to the four colors tth._bc_colors derives, the
    same palette arithmetic the decoder reads back."""
    rgb = [p[:3] for p in px16]
    hi = [max(p[c] for p in rgb) for c in range(3)]
    lo = [min(p[c] for p in rgb) for c in range(3)]
    far = max(((a, b) for a in rgb for b in rgb),
              key=lambda ab: sum((ab[0][c] - ab[1][c]) ** 2 for c in range(3)))
    best = None
    for e0, e1 in ((hi, lo), far):
        c0, c1 = _to565(*e0), _to565(*e1)
        if c0 < c1:
            c0, c1 = c1, c0
        for _ in range(3):
            if c0 == c1:
                sel, err = [0] * 16, _sse(rgb, [T._c565(c0)], [0] * 16)
            else:
                pal = T._bc_colors(c0, c1, False)
                sel = _assign(rgb, pal)
                err = _sse(rgb, pal, sel)
            if best is None or err < best[0]:
                best = (err, c0, c1, sel)
            if c0 == c1 or err == 0:
                break
            fit = _refit(rgb, sel, _COLOR_W, 3)
            if fit is None:
                break
            n0, n1 = _to565(*fit[0]), _to565(*fit[1])
            if n0 < n1:
                n0, n1 = n1, n0
            if (n0, n1) == (c0, c1):
                break
            c0, c1 = n0, n1
    _err, c0, c1, sel = best
    bits = 0
    if c0 != c1:
        for i, s in enumerate(sel):
            bits |= s << (2 * i)
    # c0 > c1 keeps DXT1 in 4-color mode; an equal pair with index 0 decodes to c0
    # opaque, since the decoder's punch-through arm only fires on selector 3.
    return struct.pack("<HHI", c0, c1, bits)


def _alpha_table(a0, a1):
    # a0 > a1 selects the 8-value table, the same one the decoder builds.
    return [(a0,), (a1,)] + [(((6 - i) * a0 + (1 + i) * a1) // 7,) for i in range(6)]


def _dxt5_alpha_half(a16):
    pts = [(a,) for a in a16]
    a0, a1 = max(a16), min(a16)
    if a0 == a1:
        return bytes((a0, a1)) + (0).to_bytes(6, "little")
    for _ in range(2):
        fit = _refit(pts, _assign(pts, _alpha_table(a0, a1)), _ALPHA_W, 1)
        if fit is None:
            break
        n0, n1 = fit[0][0], fit[1][0]
        # A collapsed or unchanged pair means the fit converged; a flipped one would
        # select the decoder's 6-value table and change every meaning.
        if n0 <= n1 or (n0, n1) == (a0, a1):
            break
        a0, a1 = n0, n1
    bits = 0
    for i, s in enumerate(_assign(pts, _alpha_table(a0, a1))):
        bits |= s << (3 * i)
    return bytes((a0, a1)) + bits.to_bytes(6, "little")


def _dxt3_alpha_half(a16):
    bits = 0
    for i, a in enumerate(a16):
        bits |= ((a + 8) // 17) << (4 * i)
    return bits.to_bytes(8, "little")


def encode_pixels(w, h, fmt, rgba):
    """One level's raw bytes in `fmt`, from top-down RGBA8888 -- the inverse of
    tth.decode. Edge blocks replicate their clamped pixels, which the decoder skips."""
    if fmt in (T.DXT1, T.DXT3, T.DXT5):
        out = bytearray()
        for by in range(max(1, (h + 3) // 4)):
            for bx in range(max(1, (w + 3) // 4)):
                px = []
                for py in range(4):
                    y = min(by * 4 + py, h - 1)
                    for pxi in range(4):
                        x = min(bx * 4 + pxi, w - 1)
                        p = (y * w + x) * 4
                        px.append(tuple(rgba[p:p + 4]))
                if fmt == T.DXT5:
                    out += _dxt5_alpha_half([p[3] for p in px])
                elif fmt == T.DXT3:
                    out += _dxt3_alpha_half([p[3] for p in px])
                out += _color_half(px)
        return bytes(out)
    if fmt not in ENCODABLE:
        raise ValueError("cannot encode format %d (%s)"
                         % (fmt, T.FORMAT_NAMES.get(fmt, "?")))
    out = bytearray()
    for i in range(w * h):
        r, g, b, a = rgba[i * 4:i * 4 + 4]
        if fmt == 12:
            out += bytes((b, g, r, a))
        elif fmt == 0:
            out += bytes((r, g, b, a))
        elif fmt == 3:
            out += bytes((b, g, r))
        else:
            out += bytes((r, g, b))
    return bytes(out)


def halve(w, h, rgba):
    """The next mip down: a 2x2 box average, clamped on an odd or exhausted axis."""
    nw, nh = max(1, w // 2), max(1, h // 2)
    out = bytearray(nw * nh * 4)
    for y in range(nh):
        y0, y1 = min(2 * y, h - 1), min(2 * y + 1, h - 1)
        for x in range(nw):
            x0, x1 = min(2 * x, w - 1), min(2 * x + 1, w - 1)
            o = (y * nw + x) * 4
            for c in range(4):
                out[o + c] = (rgba[(y0 * w + x0) * 4 + c] + rgba[(y0 * w + x1) * 4 + c]
                              + rgba[(y1 * w + x0) * 4 + c]
                              + rgba[(y1 * w + x1) * 4 + c] + 2) // 4
    return nw, nh, bytes(out)


def _reflectivity(w, h, rgba):
    """vtex's stored value is the mean *linear* albedo: over one map's 412 materials the
    decoded image's linear mean matches the field at correlation 1.0000 where the
    gamma-encoded mean reaches only 0.96, so the bytes are linearised before averaging."""
    n = w * h
    return tuple(sum((v / 255.0) ** 2.2 for v in rgba[c::4]) / n for c in range(3))


def _vtf_header(w, h, flags, fmt, mip_count, reflectivity, lw, lh):
    hdr = bytearray(VTF_HEADER_SIZE)
    struct.pack_into("<4sIII", hdr, 0, T.VTF_MAGIC, 7, 1, VTF_HEADER_SIZE)
    struct.pack_into("<HHIHH", hdr, 16, w, h, flags, 1, 0)
    struct.pack_into("<3f", hdr, 32, *reflectivity)
    struct.pack_into("<f", hdr, 48, 1.0)                        # bumpmapScale
    struct.pack_into("<IB", hdr, 52, fmt, mip_count)
    struct.pack_into("<IBB", hdr, 57, T.DXT1, lw, lh)
    return bytes(hdr)


def encode(w, h, rgba, fmt=T.DXT5, flags=0, inline_mips=3, mip_count=None):
    """(tth bytes, ttz bytes) for one top-down RGBA8888 image.

    `flags` carries VTF semantics this code does not interpret; copy them from a shipped
    texture of the same role (encode_like) rather than inventing a value. An all-inline
    result leaves the .ttz a bare empty stream, which the engine's own no-.ttz textures
    show is acceptable to omit entirely.
    """
    if w & (w - 1) or h & (h - 1):
        raise ValueError("%dx%d: the engine's mip chain needs power-of-two sides"
                         % (w, h))
    if len(rgba) != w * h * 4:
        raise ValueError("%d bytes for %dx%d RGBA" % (len(rgba), w, h))
    if mip_count is None:
        mip_count = max(w, h).bit_length()
    inline_mips = min(inline_mips, mip_count)

    chain = [(w, h, bytes(rgba))]
    for _ in range(mip_count - 1):
        chain.append(halve(*chain[-1]))
    # VTF image order is smallest mip first.
    mips = [encode_pixels(cw, ch, fmt, px) for cw, ch, px in reversed(chain)]

    lw, lh = lowres_dims(w, h)
    tw, th, tp = w, h, bytes(rgba)
    while (tw, th) != (lw, lh):
        tw, th, tp = halve(tw, th, tp)
    thumb = encode_pixels(lw, lh, T.DXT1, tp)
    lowres = T.level_bytes(lw, lh, T.DXT1)

    co = zlib.compressobj(9)
    packed, prefixes, total = [], [], 0
    for i, mip in enumerate(mips):
        prefixes.append(0 if i < inline_mips else total)
        if i < inline_mips:
            continue
        chunk = co.compress(mip) + co.flush(zlib.Z_SYNC_FLUSH)
        packed.append(chunk)
        total += len(chunk)
    packed.append(co.flush())
    ttz = b"".join(packed)

    blob = VTF_HEADER_SIZE + lowres + sum(len(m) for m in mips[:inline_mips])
    tth = bytearray()
    tth += struct.pack("<4sHBBI", T.MAGIC, 1, mip_count, inline_mips, blob)
    offset = VTF_HEADER_SIZE + lowres
    for i, mip in enumerate(mips):
        tth += struct.pack("<II", offset, prefixes[i])
        offset += len(mip)
    tth += struct.pack("<II", offset, len(ttz))                 # the pair of totals
    tth += _vtf_header(w, h, flags, fmt, mip_count, _reflectivity(w, h, rgba), lw, lh)
    tth += thumb
    tth += b"".join(mips[:inline_mips])
    return bytes(tth), ttz


def template_params(tth_bytes):
    """The encoding policy of a shipped .tth, for encode(**params)."""
    t = T.Tth(tth_bytes)
    return dict(fmt=t.format, flags=t.flags, inline_mips=t.inlined, mip_count=t.nmips)


def encode_like(template_tth, w, h, rgba):
    """Encode with the format, flags and mip policy of a shipped texture, which is the
    path that never has to guess what a flag bit means."""
    return encode(w, h, rgba, **template_params(template_tth))


def read_tga(data):
    """(w, h, top-down RGBA8888) off an uncompressed type-2 TGA -- the inverse of
    tth.to_tga, plus the top-down orientation bit."""
    idlen, cmap, kind = data[0], data[1], data[2]
    if kind != 2 or cmap:
        raise ValueError("only an uncompressed true-color TGA is read (type %d)" % kind)
    w, h = struct.unpack_from("<HH", data, 12)
    bpp, desc = data[16], data[17]
    if bpp not in (24, 32):
        raise ValueError("TGA depth %d; 24 or 32 expected" % bpp)
    n = bpp // 8
    px = data[18 + idlen:18 + idlen + w * h * n]
    if len(px) < w * h * n:
        raise ValueError("TGA truncated: %d of %d pixel bytes" % (len(px), w * h * n))
    out = bytearray(w * h * 4)
    top_down = bool(desc & 0x20)
    for y in range(h):
        sy = y if top_down else h - 1 - y
        for x in range(w):
            s = (sy * w + x) * n
            o = (y * w + x) * 4
            out[o:o + 4] = bytes((px[s + 2], px[s + 1], px[s],
                                  px[s + 3] if n == 4 else 255))
    return w, h, bytes(out)


def _cli(argv):
    if len(argv) < 3:
        print("usage: tth_write.py <in.tga> <out stem> [template.tth]\n"
              "       writes <out stem>.tth and .ttz; the template supplies format,\n"
              "       flags and mip policy, else DXT5 with three inline mips")
        return 2
    with open(argv[1], "rb") as f:
        w, h, rgba = read_tga(f.read())
    if len(argv) > 3:
        with open(argv[3], "rb") as f:
            tth, ttz = encode_like(f.read(), w, h, rgba)
    else:
        tth, ttz = encode(w, h, rgba)
    with open(argv[2] + ".tth", "wb") as f:
        f.write(tth)
    with open(argv[2] + ".ttz", "wb") as f:
        f.write(ttz)
    t = T.Tth(tth)
    print("%r -> %s.tth %d B + .ttz %d B" % (t, argv[2], len(tth), len(ttz)))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
