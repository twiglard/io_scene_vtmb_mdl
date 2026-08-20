#!/usr/bin/env python3
"""Bone-set templates: names and structure as data, positions as a formula. No bpy.

A template ships what the format needs to chain and nothing a shipped file would supply.
The chain join is `__strcmpi` on bone name (engine.dll `Studio_BuildChainedModelBoneMaps`
@2000ce40), so names and hierarchy are the whole of what makes a generated skeleton play
Troika's animations -- and both are convention, not content. Positions come from a named
generator with parameters, never a table read out of a model.

That is affordable rather than a compromise because bind ROTATION is free: an animation's
quaternion is the bone's local rotation outright, with no base composed under it, so a
generated orientation cannot put a limb at the wrong angle. Only bind POSITION is additive,
which means a generated skeleton plays the shipped animations correctly at its own limb
lengths.

`attach` is the set of parent names a template references but does not itself provide, so
two templates merge when the second's attach points are bones the first supplied. It is
declared in the file and checked against the computed set rather than trusted -- a declared
field drifts, a derived one cannot.
"""

import json
import math
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

KINDS = ("skeleton", "material", "animations")
# `measured` is legal to write down and illegal to ship: the tag exists so a pack-time
# sweep can refuse it, which is what makes the no-shipped-content rule enforced rather
# than remembered.
PROVENANCE = ("convention", "generated", "measured")
SHIPPABLE = ("convention", "generated")
# Every key here must carry a provenance tag when the template has it at all. Gating on
# this set, rather than on whatever tags happen to be present, is what stops an empty
# `provenance` block from passing as audited.
TAGGED_KEYS = ("bones", "proportions", "includes", "surfaceprop", "cdtexture", "material")

BONE_USED = 0x10
BONE_ROTATION_FROM_ROOT = 0x2
# Item 41 measured this as one bit on one name across the corpus, so a template states it
# rather than transcribing a flags word.
FROM_ROOT_NAMES = ("bip01 spine1",)


class Refused(Exception):
    pass


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a, fallback=(0.0, 0.0, 1.0)):
    n = math.sqrt(_dot(a, a))
    return fallback if n < 1e-9 else _mul(a, 1.0 / n)


def _frame(head, tail):
    """(forward, side, up) for a bone, forward along its own length.

    `up` is world +Z made orthogonal to forward, so a bone that happens to point straight
    up falls back to +X rather than producing a zero vector and a degenerate basis.
    """
    fwd = _norm(_sub(tail, head), (0.0, 1.0, 0.0))
    up = _sub((0.0, 0.0, 1.0), _mul(fwd, _dot(fwd, (0.0, 0.0, 1.0))))
    if _dot(up, up) < 1e-12:
        up = _sub((1.0, 0.0, 0.0), _mul(fwd, _dot(fwd, (1.0, 0.0, 0.0))))
    up = _norm(up, (1.0, 0.0, 0.0))
    return fwd, _cross(fwd, up), up


# Fractions of `height`, from the eight-head artistic canon with a Vitruvian span -- head
# 1/8 of stature, femur and tibia equal, fingertip to fingertip one stature.  Nothing here
# is read out of a shipped model.  Axes are the engine's: +X forward, +Y left, +Z up.
# `height` 72.0 is the hull `mdl_build.new` already writes at @180/@192.

# A None tail is taken from the first child listed, so bone ORDER in the JSON is
# load-bearing: `pelvis` lists `spine` before the thighs or its bone points down a leg.
_BIPED = {
    "root":     ((0.000, 0.000, 0.530), (0.060, 0.000, 0.530)),
    "pelvis":   ((0.000, 0.000, 0.530), None),
    "spine":    ((0.000, 0.000, 0.560), None),
    "spine1":   ((0.000, 0.000, 0.640), None),
    "spine2":   ((0.000, 0.000, 0.720), None),
    "neck":     ((0.000, 0.000, 0.815), None),
    "head":     ((0.000, 0.000, 0.875), (0.000, 0.000, 1.000)),
    "clavicle": ((0.000, 0.026, 0.800), None),
    "upperarm": ((0.000, 0.076, 0.800), None),
    "forearm":  ((0.000, 0.265, 0.800), None),
    "hand":     ((0.000, 0.420, 0.800), (0.000, 0.485, 0.800)),
    "thigh":    ((0.000, 0.052, 0.520), None),
    "calf":     ((0.000, 0.052, 0.285), None),
    "foot":     ((0.000, 0.052, 0.050), None),
    "toe":      ((0.070, 0.052, 0.015), (0.105, 0.052, 0.015)),
}

# Same role names, horizontal spine, so no table derives from the other.  `height` is
# withers height here and the body runs along -X behind the shoulder.
_QUADRUPED = {
    "root":     ((0.000, 0.000, 1.000), (0.150, 0.000, 1.000)),
    "pelvis":   ((-0.900, 0.000, 0.960), None),
    "spine":    ((-0.640, 0.000, 0.985), None),
    "spine1":   ((-0.380, 0.000, 1.000), None),
    "spine2":   ((-0.120, 0.000, 1.000), None),
    "neck":     ((0.060, 0.000, 1.010), None),
    "head":     ((0.320, 0.000, 1.120), (0.560, 0.000, 1.080)),
    "clavicle": ((-0.060, 0.130, 0.960), None),
    "upperarm": ((-0.020, 0.170, 0.900), None),
    "forearm":  ((-0.020, 0.170, 0.520), None),
    "hand":     ((-0.020, 0.170, 0.150), (0.110, 0.170, 0.060)),
    "thigh":    ((-0.880, 0.175, 0.900), None),
    "calf":     ((-0.760, 0.175, 0.500), None),
    "foot":     ((-0.860, 0.175, 0.170), None),
    "toe":      ((-0.760, 0.175, 0.040), (-0.640, 0.175, 0.030)),
}

# (across the palm, above it, direction as (fwd, side, up), three joint lengths).  The
# thumb leaves the palm plane, so direction is per-finger rather than shared.
_FINGERS = (
    (0.020, -0.008, (0.72, 0.62, -0.30), (0.020, 0.015, 0.012)),
    (0.014, 0.004, (0.99, 0.14, 0.00), (0.024, 0.015, 0.011)),
    (0.004, 0.004, (1.00, 0.00, 0.00), (0.028, 0.017, 0.012)),
    (-0.006, 0.003, (0.99, -0.12, 0.00), (0.025, 0.015, 0.011)),
    (-0.016, 0.001, (0.98, -0.20, 0.00), (0.019, 0.012, 0.009)),
)


def _table_gen(table):
    def gen(bones, params, anchors):
        h = float(params.get("height", 72.0))
        wide = float(params.get("width", 1.0))
        deep = float(params.get("depth", 1.0))
        out = {}
        for b in bones:
            role = b.get("role")
            if role not in table:
                raise Refused("bone %r has role %r, which this generator has no place for"
                              % (b.get("name"), role))
            mirror = -1.0 if str(b.get("side", "")).upper() == "R" else 1.0

            def place(p):
                return (p[0] * h * deep, p[1] * h * wide * mirror, p[2] * h)

            head, tail = table[role]
            out[b["name"]] = (place(head), None if tail is None else place(tail))
        return out
    return gen


def _digits(bones, params, anchors):
    """Fingers, placed in the hand bone's own frame so a merge lands on a hand this
    template never saw."""
    h = float(params.get("height", 72.0))
    spread = float(params.get("spread", 1.0))
    by_name = {b["name"]: b for b in bones}
    out = {}
    for b in bones:
        if b.get("role") != "finger":
            raise Refused("bone %r has role %r in a digits template"
                          % (b.get("name"), b.get("role")))
        fi, joint = int(b.get("finger", 0)), int(b.get("joint", 0))
        if not 0 <= fi < len(_FINGERS):
            raise Refused("bone %r cites finger %d, of %d" % (b["name"], fi, len(_FINGERS)))
        side, up_off, direction, lengths = _FINGERS[fi]
        if joint >= len(lengths):
            raise Refused("bone %r cites joint %d, of %d"
                          % (b["name"], joint, len(lengths)))
        # Walk back to the bone this finger hangs off, which is an anchor rather than
        # anything this template placed.
        root = b
        while root.get("joint", 0) > 0:
            parent = by_name.get(root.get("parent"))
            if parent is None:
                raise Refused("finger bone %r has no joint 0 above it" % b["name"])
            root = parent
        hand = anchors.get(root.get("parent"))
        if hand is None:
            raise Refused("finger bone %r hangs off %r, which nothing provides"
                          % (b["name"], root.get("parent")))
        fwd, sd, up = _frame(hand[0], hand[1])
        mirror = -1.0 if str(b.get("side", "")).upper() == "R" else 1.0
        base = _add(hand[1], _add(_mul(sd, side * h * spread * mirror),
                                  _mul(up, up_off * h)))
        step = _norm(_add(_add(_mul(fwd, direction[0]),
                               _mul(sd, direction[1] * mirror)),
                          _mul(up, direction[2])))
        for j in range(joint):
            base = _add(base, _mul(step, lengths[j] * h))
        out[b["name"]] = (base, _add(base, _mul(step, lengths[joint] * h)))
    return out


def _strand(bones, params, anchors):
    """A straight run from an anchor -- hair, a tail, a wing spar.  `direction` is in the
    anchor's own frame as (forward, side, up), so one file merges onto a spine or a hand."""
    h = float(params.get("height", 72.0))
    total = float(params.get("length", 0.25))
    taper = float(params.get("taper", 1.0))
    direction = tuple(params.get("direction", (0.0, 0.0, -1.0)))
    by_name = {b["name"]: b for b in bones}
    runs = {}
    for b in bones:
        if b.get("role") != "strand":
            raise Refused("bone %r has role %r in a strand template"
                          % (b.get("name"), b.get("role")))
        root = b
        while root.get("index", 0) > 0:
            parent = by_name.get(root.get("parent"))
            if parent is None:
                raise Refused("strand bone %r has no index 0 above it" % b["name"])
            root = parent
        runs.setdefault(root.get("parent"), []).append(b)
    out = {}
    for anchor_name, members in runs.items():
        anchor = anchors.get(anchor_name)
        if anchor is None:
            raise Refused("strand hangs off %r, which nothing provides" % anchor_name)
        fwd, sd, up = _frame(anchor[0], anchor[1])
        n = max(1, len(members))
        # A geometric taper, so `taper` 1.0 is an even chain and 0.8 shortens each link to
        # four fifths of the one before it.
        weights = [taper ** k for k in range(n)]
        unit = total * h / (sum(weights) or 1.0)
        for b in sorted(members, key=lambda x: int(x.get("index", 0))):
            mirror = -1.0 if str(b.get("side", "")).upper() == "R" else 1.0
            step = _norm(_add(_add(_mul(fwd, direction[0]),
                                   _mul(sd, direction[1] * mirror)),
                              _mul(up, direction[2])))
            k = int(b.get("index", 0))
            if k >= n:
                raise Refused("strand bone %r cites index %d, of %d" % (b["name"], k, n))
            head = _add(anchor[1], _mul(step, unit * sum(weights[:k])))
            out[b["name"]] = (head, _add(head, _mul(step, unit * weights[k])))
    return out


def _single(bones, params, anchors):
    """One bone at the origin. The least that makes a valid spawnable model."""
    h = float(params.get("height", 72.0))
    ln = float(params.get("length", 0.1)) * h
    out = {}
    for b in bones:
        out[b["name"]] = ((0.0, 0.0, 0.0), (0.0, 0.0, ln))
    return out


GENERATORS = {
    "single": _single,
    "biped": _table_gen(_BIPED),
    "quadruped": _table_gen(_QUADRUPED),
    "digits": _digits,
    "strand": _strand,
}


class Bone(object):
    __slots__ = ("name", "parent", "head", "tail", "role", "side", "flags", "hitgroup")

    def __init__(self, name, parent, head, tail, role, side, flags, hitgroup):
        self.name, self.parent = name, parent
        self.head, self.tail = head, tail
        self.role, self.side = role, side
        self.flags, self.hitgroup = flags, hitgroup

    def __repr__(self):
        return "<Bone %s parent=%s>" % (self.name, self.parent)


def attach_points(t):
    """Parent names the bone list references but does not provide, in first-seen order.

    Derived, never read off the file: this is what `attach` is checked against.
    """
    provided, out = set(), []
    for b in t.get("bones") or ():
        provided.add(b.get("name"))
    for b in t.get("bones") or ():
        p = b.get("parent")
        if p and p not in provided and p not in out:
            out.append(p)
    return out


def validate(t, source="", shippable=True):
    """Raise `Refused` on anything wrong with one template. Returns the number of distinct
    checks that actually ran.

    The count is the point: a validator whose loops never execute reports success just as
    loudly as one that passed, so the caller gates on this rather than on the absence of
    an exception.
    """
    where = source or t.get("id") or "<template>"
    n = 0

    def need(cond, msg):
        if not cond:
            raise Refused("%s: %s" % (where, msg))

    for key in ("id", "label", "kind"):
        need(isinstance(t.get(key), str) and t[key], "%s must be a non-empty string" % key)
        n += 1
    need(t["kind"] in KINDS, "kind %r is not one of %s" % (t["kind"], ", ".join(KINDS)))
    n += 1

    prov = t.get("provenance")
    need(isinstance(prov, dict), "provenance must be an object")
    n += 1
    for key in TAGGED_KEYS:
        if key not in t:
            continue
        need(key in prov, "%r is present but carries no provenance tag" % key)
        need(prov[key] in PROVENANCE,
             "%r is tagged %r, not one of %s" % (key, prov[key], ", ".join(PROVENANCE)))
        if shippable:
            need(prov[key] in SHIPPABLE,
                 "%r is tagged %r, which must never ship: a template carries names, "
                 "structure and formulas, never a number read out of a shipped file"
                 % (key, prov[key]))
        n += 1
    for key in prov:
        need(key in TAGGED_KEYS, "provenance tags %r, which is not a content key" % key)
        n += 1

    if t["kind"] == "animations":
        inc = t.get("includes")
        need(isinstance(inc, list) and inc,
             "an animations template is nothing but its includes, so it needs at least one")
        n += 1
        need(not t.get("bones"), "an animations template carries no bones")
        n += 1
        for p in inc:
            need(isinstance(p, str) and p.lower().endswith(".mdl"),
                 "include %r is not a .mdl path" % p)
            n += 1
        return n

    if t["kind"] == "material":
        for key in ("cdtexture", "name"):
            need(isinstance(t.get(key), str), "a material template needs a %r string" % key)
            n += 1
        return n

    bones = t.get("bones")
    need(isinstance(bones, list) and bones, "a skeleton template needs a non-empty bones list")
    n += 1
    seen = set()
    for b in bones:
        need(isinstance(b, dict), "every bone must be an object")
        name = b.get("name")
        need(isinstance(name, str) and name.strip(), "every bone needs a name")
        need(name not in seen, "bone %r is listed twice" % name)
        need(isinstance(b.get("role"), str) and b["role"], "bone %r needs a role" % name)
        side = str(b.get("side", "") or "")
        need(side in ("", "L", "R"), "bone %r has side %r, not L or R" % (name, side))
        hg = b.get("hitgroup")
        need(hg is None or (isinstance(hg, int) and 0 <= hg <= 7),
             "bone %r has hitgroup %r, not 0..7" % (name, hg))
        seen.add(name)
        n += 1
    # A parent must be provided by this template or be an attach point; either way it can
    # never be a forward reference, because the engine walks the bone array once in order.
    placed = set()
    for b in bones:
        p = b.get("parent")
        need(p is None or p in placed or p not in seen,
             "bone %r cites parent %r, which this template lists after it" % (b["name"], p))
        placed.add(b["name"])
        n += 1

    gen = (t.get("proportions") or {}).get("generator")
    need(gen in GENERATORS,
         "proportions.generator is %r, not one of %s" % (gen, ", ".join(sorted(GENERATORS))))
    n += 1
    need(isinstance((t.get("proportions") or {}).get("params", {}), dict),
         "proportions.params must be an object")
    n += 1

    declared = list(t.get("attach") or [])
    computed = attach_points(t)
    need(declared == computed,
         "attach says %s but the bone list needs %s -- the declared field has drifted"
         % (declared or "nothing", computed or "nothing"))
    n += 1

    for p in t.get("includes") or ():
        need(isinstance(p, str) and p.lower().endswith(".mdl"),
             "include %r is not a .mdl path" % p)
        n += 1
    return n


def load_all(directory=None, shippable=True):
    """Every template on disk, keyed by id. Validates each and refuses a duplicate id."""
    directory = directory or TEMPLATE_DIR
    out = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(directory, fn)
        with open(path, "r", encoding="utf-8") as fp:
            t = json.load(fp)
        validate(t, source=fn, shippable=shippable)
        if t["id"] in out:
            raise Refused("%s: id %r is already taken by another file" % (fn, t["id"]))
        t["_source"] = fn
        out[t["id"]] = t
    return out


def _flags_for(name):
    f = BONE_USED
    if name.lower() in FROM_ROOT_NAMES:
        f |= BONE_ROTATION_FROM_ROOT
    return f


def resolve(t, anchors=None, params=None):
    """Placed bones for one template, parent-by-name, in emit order.

    `anchors` is {bone name: (head, tail)} for every attach point -- what an already-built
    armature supplies when a second template merges onto it.
    """
    if t.get("kind") != "skeleton":
        raise Refused("%r is a %s template, not a skeleton" % (t.get("id"), t.get("kind")))
    anchors = dict(anchors or {})
    missing = [p for p in attach_points(t) if p not in anchors]
    if missing:
        raise Refused("%r attaches to %s, which the scene does not have"
                      % (t["id"], ", ".join(missing)))
    prop = t.get("proportions") or {}
    merged = dict(prop.get("params") or {})
    merged.update(params or {})
    placed = GENERATORS[prop["generator"]](t["bones"], merged, anchors)

    kids = {}
    for b in t["bones"]:
        kids.setdefault(b.get("parent"), []).append(b["name"])
    out = []
    for b in t["bones"]:
        name = b["name"]
        head, tail = placed[name]
        if tail is None:
            first = (kids.get(name) or [None])[0]
            tail = placed[first][0] if first else None
        # Blender drops a zero-length bone without a word, and a degenerate one is exactly
        # what a bone sharing its only child's head produces -- `Bip01` over `Bip01 Pelvis`
        # is that case in every biped here.
        if tail is None or _dot(_sub(tail, head), _sub(tail, head)) < 1e-12:
            raise Refused("%s: bone %r would have zero length; the generator owes it an "
                          "explicit tail" % (t["id"], name))
        out.append(Bone(name, b.get("parent"), head, tail, b["role"],
                        str(b.get("side", "") or ""), _flags_for(name), b.get("hitgroup")))
    return out


def merged_includes(templates):
    """The chain over several templates, first-seen order, no repeats."""
    out = []
    for t in templates:
        for p in t.get("includes") or ():
            if p not in out:
                out.append(p)
    return out
