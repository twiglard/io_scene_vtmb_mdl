"""Drivers that show in Blender what the face rules draw.

Decision 18's option 2, built from drivers on the user's word 2026-10-15: a .blend with
drivers animates without the addon installed, and changing data from a frame-change
handler during a render can crash Blender. Every expression here is in Blender's
simple-expression set, so none needs Auto Run Python Scripts.

Three layers, the first two as properties of the armature object:
- one per flex controller, in its min..max units, which is what a rule reads: the client
  maps the server's 0..1 back before RunFlexRules (`C_BaseFlex__SetupWeights`);
- one per flexdesc weight, driven by its rule, by an eyeball's lid code, or by nothing;
- a driver on each shape key's value, the renderer's window over its flexdesc's weight.

The export reads key coordinates and never a key's value, so none of this reaches a file.
"""

import struct

from . import blender_export
from . import mdl as mdl_mod
from . import mdl_build as build_mod

CTL = "vtmb_ctl:"
FLEX = "vtmb_flex:"
ON = "vtmb_preview"
SAVED = "vtmb_preview_values"
# ChannelDriver.expression is 256 bytes, and Blender 5.2 cuts a longer one at 255 without
# a word. The longest shipped rule comes to 187, gargoyle.mdl's `mouth`.
EXPR_MAX = 255
# An ID property name is at most 63 characters (Blender 5.2's own error says so).
NAME_MAX = 63
# RunFlexRules, 2003 c_baseflex.cpp:126-136: a divisor at or under this gives 0.
DIV_MIN = 0.0001
# R_StudioFlexVerts skips a record whose windowed weight is inside +-0.001.
CUT = 0.001


def _num(v):
    """The shortest fixed-point decimal reading back as the same float32."""
    f = struct.unpack("<f", struct.pack("<f", v))[0]
    for p in range(1, 60):
        s = "%.*f" % (p, f)
        if struct.unpack("<f", struct.pack("<f", float(s)))[0] == f:
            break
    return "(%s)" % s if s.startswith("-") else s


def _lit(v):
    """A ramp's width or cut to 9 significant digits. A ramp 0.008 wide (smiling_jack.mdl's
    `lower_right`) multiplies an error 125 times, and the shortest float32 form is off by
    1.3e-6 there."""
    s = "%.9g" % v
    return ("%.15f" % v).rstrip("0") if "e" in s else s


def _bound(v):
    """A window bound exactly. At rest a lid weight is `r * sin(asin(t / r))`, which rounds
    to the target itself, and a target is often a bound: `w >= t3` has to see them equal."""
    return repr(float(v))


def _rule_expr(ops, ctl_names, desc_names, readable):
    """(expression, [property]) for one rule's op stream, the i-th property being `vi`.

    `readable` is the flexdescs an earlier rule wrote. RunFlexRules zeroes every weight and
    runs the rules in file order, so any other `%name` reads 0 and is written as 0.0.
    """
    props, st = [], []

    def var(prop):
        if prop not in props:
            props.append(prop)
        return "v%d" % props.index(prop)

    for op, raw in ops:
        if op == mdl_mod.STUDIO_CONST:
            st.append((_num(struct.unpack("<f", struct.pack("<i", raw))[0]), 3))
        elif op == mdl_mod.STUDIO_FETCH1:
            st.append((var(CTL + ctl_names[raw]), 3))
        elif op == mdl_mod.STUDIO_FETCH2:
            n = desc_names[raw]
            st.append((var(FLEX + n) if n in readable else "0.0", 3))
        else:
            (rt, rp), (lt, lp) = st.pop(), st.pop()
            if op == mdl_mod.STUDIO_DIV:
                # The conditional compiles to a jump in Blender's evaluator, so the branch
                # not taken never divides.
                st.append(("(%s / %s if %s > %s else 0.0)"
                           % (lt if lp >= 2 else "(%s)" % lt, rt if rp > 2 else "(%s)" % rt,
                              rt, _num(DIV_MIN)), 3))
                continue
            p = 1 if op in (mdl_mod.STUDIO_ADD, mdl_mod.STUDIO_SUB) else 2
            sym = {mdl_mod.STUDIO_ADD: "+", mdl_mod.STUDIO_SUB: "-",
                   mdl_mod.STUDIO_MUL: "*"}[op]
            st.append(("%s %s %s" % (lt if lp >= p else "(%s)" % lt, sym,
                                     rt if rp > p else "(%s)" % rt), p))
    if len(st) != 1:
        raise ValueError("the op stream does not balance")
    return st[0][0], props


def _window_expr(t):
    """The renderer's window over the weight `w`, 2003 r_studioflex.cpp:190-217.

    The renderer drops a ramp value under 0.001, which here is a bound on `w` so each
    literal is said once. A ramp whose two ends meet or cross is never reached by the
    renderer's tests, and is left out (32 of 18 060 shipped records).
    """
    t0, t1, t2, t3 = (float(c) for c in t)
    inner = "1.0"
    if t3 > t2:
        inner = "(%s - w) / %s * (w <= %s) if w > %s else %s" % (
            _bound(t3), _lit(t3 - t2), _lit(t3 - CUT * (t3 - t2)), _bound(t2), inner)
    if t1 > t0:
        inner = "(w - %s) / %s * (w >= %s) if w < %s else (%s)" % (
            _bound(t0), _lit(t1 - t0), _lit(t0 + CUT * (t1 - t0)), _bound(t1), inner)
    return "0.0 if w <= %s or w >= %s else (%s)" % (_bound(t0), _bound(t3), inner)


def window(t, w):
    """The renderer's window at the weight `w`, what each key's driver computes."""
    t0, t1, t2, t3 = (float(c) for c in t)
    if w <= t0 or w >= t3:
        return 0.0
    x = (w - t0) / (t1 - t0) if w < t1 else (t3 - w) / (t3 - t2) if w > t2 else 1.0
    return 0.0 if -CUT < x < CUT else x


def _lid_expr(inputs, readable):
    """R_StudioEyeballPosition's lid height for a gaze straight ahead, or None.

    `r * sin(sum of w_k * asin(t_k / r))`, the three inputs' weights `wk`, targets `tk` and
    radius `r` read live off the eyeball empty. 2003 r_studioflex.cpp:50-144.
    """
    terms = ["w%d * asin(t%d / r)" % (k, k) for k, n in enumerate(inputs) if n in readable]
    return "r * sin(%s)" % " + ".join(terms) if terms else None


def key_owners(arm_obj):
    """[(Key datablock, the object whose stamps name its keys)] under one armature.

    Alt+D shares the datablock, which is driven once.
    """
    out, seen = [], set()
    for obj in arm_obj.children_recursive:
        if obj.type != "MESH" or obj.data.shape_keys is None:
            continue
        key = obj.data.shape_keys
        if len(key.key_blocks) < 2 or key.name in seen:
            continue
        seen.add(key.name)
        out.append((key, obj))
    return out


def plan(arm_obj):
    """Everything `start` builds, or ValueError naming the line it cannot preview.

    {"ctls": [(name, lo, hi)], "flex": {name: (expr or None, [(var, id, path)])},
     "keys": [(key, block name, expr, flexdesc)]}
    """
    ctls = []
    for line in arm_obj.get("vtmb_flexcontrollers") or ():
        _t, lo, hi, name = blender_export.flex_controller_fields(str(line))
        ctls.append((name, lo, hi))
    rules = []
    for line in arm_obj.get("vtmb_flexrules") or ():
        line = str(line)
        if "=" not in line:
            raise ValueError("flex rule %r has no `=`" % line)
        name, expr = (x.strip() for x in line.split("=", 1))
        rules.append((name, expr, line))
    eyes = [o for objs in blender_export.eyeball_objects(arm_obj).values() for o in objs]
    lids = {}
    for e in eyes:
        names = [str(x) for x in (e.get("vtmb_eyeball_lidflexes") or ())]
        if len(names) != 8:
            continue
        for side, inputs, lid in (("upper", names[0:3], names[6]),
                                  ("lower", names[3:6], names[7])):
            if lid:
                if lid in lids:
                    raise ValueError("eyeballs %r and %r both write lid flex %r"
                                     % (lids[lid][0].name, e.name, lid))
                lids[lid] = (e, side, inputs)

    owners = key_owners(arm_obj)
    keyed = []
    for key, obj in owners:
        if blender_export.prestamp_flexes(obj):
            raise ValueError(blender_export._summed_keys(obj.name))
        names = list(obj.get("vtmb_flex_names") or [])
        descs = obj.get("vtmb_flex_descs")
        targets = obj.get("vtmb_flex_targets") or []
        for kb in key.key_blocks[1:]:
            fd, t = blender_export.key_flex(kb.name, names, descs, targets)
            keyed.append((key, kb.name, t if t is not None else build_mod.FLEX_TARGET, fd))

    descs = []
    for n in [r[0] for r in rules] + list(lids) + [k[3] for k in keyed]:
        if n not in descs:
            descs.append(n)
    for n in [c[0] for c in ctls]:
        if len(CTL + n) > NAME_MAX:
            raise ValueError("flex controller %r is too long a name for a property" % n)
    for n in descs:
        if len(FLEX + n) > NAME_MAX:
            raise ValueError("flex %r is too long a name for a property" % n)

    class _C:
        def __init__(self, name):
            self.name = name
    ctl_objs = [_C(c[0]) for c in ctls]
    flex, readable = {}, set()
    for name, expr, line in rules:
        if name in flex:
            raise ValueError("two flex rules write %r; the game keeps the later, which the "
                             "preview cannot show to a rule reading between them" % name)
        if name in lids:
            raise ValueError("flex rule %r writes eyeball %r's lid flex, which the eyeball "
                             "code writes after every rule" % (line, lids[name][0].name))
        ops = mdl_mod.parse_flex_expr(expr, ctl_objs, descs)
        text, props = _rule_expr(ops, [c[0] for c in ctls], descs, readable)
        if len(text) > EXPR_MAX:
            raise ValueError("flex rule %r comes to a %d-character driver expression, and "
                             "Blender keeps %d" % (line, len(text), EXPR_MAX))
        flex[name] = (text, [("v%d" % k, arm_obj, '["%s"]' % p)
                             for k, p in enumerate(props)])
        readable.add(name)
    for lid, (e, side, inputs) in lids.items():
        text = _lid_expr(inputs, readable)
        if text is None:
            flex[lid] = (None, [])
            continue
        tkey = "vtmb_eyeball_%starget" % side
        var = [("r", e, '["vtmb_eyeball_radius"]')]
        for k, n in enumerate(inputs):
            if n in readable:
                var += [("w%d" % k, arm_obj, '["%s%s"]' % (FLEX, n)),
                        ("t%d" % k, e, '["%s"][%d]' % (tkey, k))]
        flex[lid] = (text, var)
    keys = []
    for key, name, t, fd in keyed:
        flex.setdefault(fd, (None, []))
        text = _window_expr(t)
        if len(text) > EXPR_MAX:
            raise ValueError("shape key %r's window %r comes to a %d-character driver "
                             "expression, and Blender keeps %d"
                             % (name, tuple(t), len(text), EXPR_MAX))
        keys.append((key, name, text, fd))
    return {"ctls": ctls, "flex": flex, "keys": keys}


def _drive(id_, path, expr, variables):
    fc = id_.driver_add(path)
    # Blender 5.2 hands back the curve keyed at (0, 0) and (1, 1), which maps the driver's
    # value through a Bezier: 4.4e-05 came back 0. With no key and no modifier it does not.
    for kp in reversed(list(fc.keyframe_points)):
        fc.keyframe_points.remove(kp, fast=True)
    for mod in list(fc.modifiers):
        fc.modifiers.remove(mod)
    d = fc.driver
    d.type = "SCRIPTED"
    for name, tid, tpath in variables:
        v = d.variables.new()
        v.name, v.type = name, "SINGLE_PROP"
        v.targets[0].id_type = "OBJECT"
        v.targets[0].id = tid
        v.targets[0].data_path = tpath
    d.expression = expr


def start(arm_obj):
    """Build the preview, replacing any earlier one. (controllers, driven, undriven, keys).

    The plan is made before anything is touched, so a line it cannot preview raises with
    the scene as it was. A controller keeps its value across a rebuild while in range.
    """
    p = plan(arm_obj)
    was = {k: float(arm_obj[k]) for k in arm_obj.keys() if k.startswith(CTL)}
    stop(arm_obj)
    for name, lo, hi in p["ctls"]:
        k = CTL + name
        a, b = sorted((lo, hi))
        rest = min(max(0.0, a), b)
        v = was.get(k, rest)
        arm_obj[k] = v if a <= v <= b else rest
        arm_obj.id_properties_ui(k).update(
            min=a, max=b, soft_min=a, soft_max=b,
            description="flex controller %s, in the units a rule reads it" % name)
    driven = 0
    for name, (expr, var) in p["flex"].items():
        arm_obj[FLEX + name] = 0.0
        if expr is not None:
            _drive(arm_obj, '["%s%s"]' % (FLEX, name), expr, var)
            driven += 1
    for key in {id(k[0]): k[0] for k in p["keys"]}.values():
        key[SAVED] = {kb.name: kb.value for kb in key.key_blocks}
    for key, name, expr, fd in p["keys"]:
        _drive(key, 'key_blocks["%s"].value' % name.replace('"', '\\"'), expr,
               [("w", arm_obj, '["%s%s"]' % (FLEX, fd))])
    arm_obj[ON] = 1
    arm_obj.update_tag()
    return len(p["ctls"]), driven, len(p["flex"]) - driven, len(p["keys"])


def _ours(fc, arm_obj):
    return any(t.id == arm_obj and t.data_path.startswith('["%s' % FLEX)
               for v in fc.driver.variables for t in v.targets)


def stop(arm_obj):
    """Remove every driver and property `start` made and put each key's value back."""
    ad = arm_obj.animation_data
    if ad is not None:
        for path in [fc.data_path for fc in ad.drivers
                     if fc.data_path.startswith('["%s' % FLEX)]:
            arm_obj.driver_remove(path)
    for key, _obj in key_owners(arm_obj):
        kad = key.animation_data
        if kad is not None:
            for path in [fc.data_path for fc in kad.drivers if _ours(fc, arm_obj)]:
                key.driver_remove(path)
        saved = key.get(SAVED)
        if saved is not None:
            for kb in key.key_blocks:
                if kb.name in saved:
                    kb.value = saved[kb.name]
            del key[SAVED]
    for k in [k for k in arm_obj.keys() if k.startswith((CTL, FLEX))]:
        del arm_obj[k]
    if ON in arm_obj:
        del arm_obj[ON]
    arm_obj.update_tag()


def refresh(arm_obj):
    """Rebuild a running preview after an edit, or stop it and say why. None when fine."""
    if not arm_obj.get(ON):
        return None
    try:
        start(arm_obj)
    except ValueError as e:
        stop(arm_obj)
        return "the rule preview stopped: %s" % e
    return None
