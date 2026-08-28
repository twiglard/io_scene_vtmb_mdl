#!/usr/bin/env python3
"""Resolving a v2531 relative path against content roots. No bpy.

Search order comes from the engine's -game argument and is recorded nowhere on disk, so
the roots list is the caller's to state; only the candidates can be discovered.
"""

import os
import re

_VMT_TOKEN = re.compile(r'"([^"]*)"|(\S+)')

MODELS_DIR = "models"
MATERIALS_DIR = "materials"


def root_of(path):
    """The dir that a 'models/...' path is relative to, given a file inside that tree."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        parent, base = os.path.split(d)
        if base.lower() == MODELS_DIR:
            return parent
        if parent == d:
            return None
        d = parent


def content_root(path):
    """Fallback for a file outside a models/ tree: the nearest ancestor holding one."""
    r = root_of(path)
    if r is not None:
        return r
    d = os.path.dirname(os.path.abspath(path))
    while True:
        for sub in (MODELS_DIR, MATERIALS_DIR):
            if os.path.isdir(os.path.join(d, sub)):
                return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def content_relative(path, name):
    """What follows the last `name/` component of `path`, forward-slashed, or None.

    A picked file can sit in the install, in an extraction or in a loose tree of the user's
    own, and none of those roots is knowable from the path alone; the marker component is,
    so it is what the engine-relative part gets cut from.
    """
    parts = os.path.abspath(path).replace("\\", "/").rstrip("/").split("/")
    low = [p.lower() for p in parts]
    if name.lower() not in low:
        return None
    return "/".join(parts[len(low) - low[::-1].index(name.lower()):])


def engine_paths(seq):
    """Separators normalised, blanks dropped, repeats dropped in first-seen order."""
    out = []
    for p in (seq or ()):
        p = str(p).replace("\\", "/").strip()
        if p and p not in out:
            out.append(p)
    return out


def cdtexture_list(value, default=MODELS_DIR + "/"):
    """A `;`-separated field or a sequence, as the list the file stores.

    Never empty: a model with no directory at all resolves no material, so the default
    stands in for a field the user cleared.
    """
    out = engine_paths(value.split(";") if isinstance(value, str) else value)
    return out or engine_paths([default])


def split_list(text):
    return [p.strip() for p in str(text or "").replace(",", ";").split(";") if p.strip()]


def roots(mdl_path, game_root="", mods=(), extra=()):
    """Ordered content roots: the opened file's own tree, then the configured mods."""
    out = []
    own = content_root(mdl_path) if mdl_path else None
    # Without a game root the mod names are not paths: os.path.join("", "Vampire") is
    # relative, and abspath would then resolve it against Blender's cwd.
    mod_dirs = [os.path.join(game_root, m) for m in mods] if game_root else []
    for cand in [own] + mod_dirs + list(extra):
        if not cand:
            continue
        cand = os.path.abspath(cand)
        if os.path.isdir(cand) and cand not in out:
            out.append(cand)
    return out


def resolve(rel, roots_):
    rel = rel.replace("/", os.sep).replace("\\", os.sep).lstrip(os.sep)
    for r in roots_:
        p = os.path.join(r, rel)
        if os.path.isfile(p):
            return p
    return None


def relative(path, roots_):
    p = os.path.abspath(path)
    for r in roots_:
        if p.lower().startswith(r.lower() + os.sep):
            return p[len(r) + 1:].replace(os.sep, "/")
    return None


def companions(mdl_path, roots_, suffixes):
    """Companion candidates beside the .mdl first, then the same relative path under
    each root -- the Unofficial Patch ships .mdl without .vtx and relies on that."""
    stem = mdl_path[:-4] if mdl_path.lower().endswith(".mdl") else mdl_path
    out = [stem + s for s in suffixes]
    rel = relative(mdl_path, roots_)
    if rel:
        rel = rel[:-4] if rel.lower().endswith(".mdl") else rel
        for r in roots_:
            for s in suffixes:
                p = os.path.join(r, rel.replace("/", os.sep) + s)
                if p not in out:
                    out.append(p)
    return out


def material_stems(name, search_paths=()):
    """Extensionless material paths to try, cdtexture search paths first."""
    out = [MATERIALS_DIR + "/" + p.replace("\\", "/").strip("/") + "/" + name
           for p in (search_paths or [])]
    out.append(MATERIALS_DIR + "/" + name)
    return out


def vmt_value(blob, *keys):
    """The value a .vmt assigns to the first of `keys` it carries, or None.

    A // outside a quoted string ends the line. Four shipped files comment one
    "$basetexture" out above the live one, so scanning the quoted tokens alone returns the
    dead value -- gio_spirit.mdl draws gio_spirit.tth where its .vmt says giospiritbody.
    The key is not always quoted: every $dudvmap in the corpus is bare, and that same scan
    sees the values and none of the names.
    """
    toks = []
    for line in blob.decode("latin1").splitlines():
        q = False
        for i, ch in enumerate(line):
            if ch == '"':
                q = not q
            elif ch == "/" and not q and line[i:i + 2] == "//":
                line = line[:i]
                break
        for m in _VMT_TOKEN.finditer(line):
            t = m.group(1) if m.group(1) is not None else m.group(2)
            if t not in ("{", "}"):
                toks.append(t)
    for key in keys:
        k = key.lower()
        for i in range(len(toks) - 1):
            if toks[i].strip().lower() == k:
                return toks[i + 1].strip()
    return None
