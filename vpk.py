#!/usr/bin/env python3
"""Troika VPK reader ("Vampire PacK"), unrelated to Valve's .vpk. No bpy.

    python vpk.py list <game dir> [substring]
    python vpk.py extract <game dir> <path in archive> <dest file>
"""

import os
import struct
import sys

FOOTER = 9
PACK_GLOB = "pack"


def read_dir(path, collisions=None):
    """{name: (offset, size)}; names lowercased and forward-slashed.

    Two entries whose names differ only in case are two keys here and one file on NTFS, and
    the second used to replace the first without being counted as anything. `collisions`
    collects (archive, key, first name, second name) instead. The later entry still wins:
    which one FileSystem_Stdio's own table keeps is unmeasured, so the reader reports the
    ambiguity rather than resolving it.

    The fold stays `.lower()` over a latin1 decode. Nothing shipped needs more -- 0 of the
    71 051 directory entries in the 16 archives collide and 0 carry a byte above 127, so
    the codepage has nothing to decide -- and `str.casefold()` would be wrong for this
    purpose: it folds U+00DF to "ss" where NTFS's upcase table does not, inventing a
    collision the filesystem does not have.
    """
    sz = os.path.getsize(path)
    if sz < FOOTER:
        raise ValueError("%s: %d bytes, too short for a footer" % (path, sz))
    with open(path, "rb") as f:
        f.seek(sz - FOOTER)
        count, diroff = struct.unpack("<II", f.read(8))
        if diroff > sz - FOOTER or count > sz:
            raise ValueError("%s: footer %d/%d does not fit" % (path, count, diroff))
        f.seek(diroff)
        d = f.read(sz - diroff - FOOTER)
    out, raw, o = {}, {}, 0
    for i in range(count):
        (n,) = struct.unpack_from("<I", d, o)
        o += 4
        if o + n + 8 > len(d):
            raise ValueError("%s: entry %d of %d runs off the end" % (path, i, count))
        name = d[o:o + n].decode("latin1")
        o += n
        off, size = struct.unpack_from("<II", d, o)
        o += 8
        if size and off + size > diroff:
            raise ValueError("%s: %r overlaps the directory" % (path, name))
        key = name.replace("\\", "/").lower()
        if key in raw and collisions is not None:
            collisions.append((os.path.basename(path), key, raw[key], name))
        raw[key] = name
        out[key] = (off, size)
    if o != len(d):
        raise ValueError("%s: %d directory bytes left over" % (path, len(d) - o))
    return out


RUNS = (0, 100)


def mount_order(game_dir):
    """Pack paths in the engine's own mount order, highest priority first.

    FileSystem_Stdio.dll AddPackFiles (0x10001ec0) probes pack%.3d.vpk upward from 0 and
    again from 100, each run stopping at the first missing index, and mounts each run
    downward. Lookup walks the search paths forward and takes the first hit, so run 0
    outranks run 100 whole -- pack005 beats pack103, not the other way round.
    """
    out = []
    if not game_dir:
        # "" would probe the cwd, since join("", "pack000.vpk") is a relative path.
        return out
    for start in RUNS:
        run = []
        i = start
        while True:
            p = os.path.join(game_dir, "%s%.3d.vpk" % (PACK_GLOB, i))
            if not os.path.isfile(p):
                break
            run.append(p)
            i += 1
        out += reversed(run)
    return out


class Vpks:
    """Every mounted packNNN.vpk in one dir, merged at the engine's own precedence."""

    def __init__(self, game_dir):
        self.dir = game_dir
        self.index = {}
        self.packs = []
        # Within one archive only. A name in two archives is not a collision --
        # mount order decides it, which is what `index` is built to apply.
        self.collisions = []
        for p in reversed(mount_order(game_dir)):
            entries = read_dir(p, self.collisions)
            self.packs.append((os.path.basename(p), len(entries)))
            for name, (off, size) in entries.items():
                # Zero-length entries are directory names, not files: pack011.vpk is a
                # bare directory index, 3568 of them, and its footer version byte is 1.
                if size:
                    self.index[name] = (p, off, size)
        self.packs.reverse()

    def read(self, name):
        hit = self.index.get(name.replace("\\", "/").lower())
        if hit is None:
            return None
        path, off, size = hit
        with open(path, "rb") as f:
            f.seek(off)
            return f.read(size)


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    v = Vpks(argv[2])
    if argv[1] == "list":
        pat = argv[3].lower() if len(argv) > 3 else ""
        for name in sorted(n for n in v.index if pat in n):
            p, off, size = v.index[name]
            print("%-9s %10d %10d  %s" % (os.path.basename(p), off, size, name))
        print("%d packs, %d files" % (len(v.packs), len(v.index)))
        for pack, _key, first, second in v.collisions:
            print("collision  %s: %r and %r are one name on disk; %r wins"
                  % (pack, first, second, second))
    elif argv[1] == "extract" and len(argv) > 4:
        blob = v.read(argv[3])
        if blob is None:
            sys.exit("not in any pack: %s" % argv[3])
        with open(argv[4], "wb") as f:
            f.write(blob)
        print("wrote %s, %d bytes" % (argv[4], len(blob)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
