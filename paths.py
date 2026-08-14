#!/usr/bin/env python3
"""Resolving a v2531 relative path against content roots. No bpy.

Search order comes from the engine's -game argument and is recorded nowhere on disk, so
the roots list is the caller's to state; only the candidates can be discovered.
"""

import os

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
