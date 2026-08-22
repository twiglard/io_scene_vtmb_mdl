#!/usr/bin/env python3
"""Blender side of the VTMB MDL exporter: actions back into a .mdl's animations.

Animation is the only thing written. Mesh, UVs, the skeleton's bind pose, materials,
sequences, hitboxes and everything else keep the bytes of the .mdl being rewritten, so
what comes out is that model with some of its animations replaced -- never a model built
from the scene.
"""

import math
import os

import bpy

from . import mdl as mdl_mod
from . import mdl_write as write_mod
from . import mdl_rebuild as rebuild_mod
from . import mesh_write as mesh_mod


UV_TOL = 1e-6
# Custom split normals are stored as two 16-bit angles, so an untouched mesh's corners
# disagree by up to 9.2e-03; under this a normal is taken as unedited.
NORMAL_EPS = 1.5e-2
NORMAL_ATTR = "vtmb_normal"
SKIN_ATTR = "vtmb_skin"
WEIGHT_ATTR = "vtmb_weight"
COUNT_ATTR = "vtmb_numbones"


def _rows(mat, scale):
    """A mathutils.Matrix as the format's 3x4, with the import scale taken back out."""
    return [[mat[i][0], mat[i][1], mat[i][2], mat[i][3] / scale] for i in range(3)]


def read_poses(context, arm_obj, m, frames, scale, anim=None, root_motion=False):
    """Local (pos, quat) per bone per frame, inverting everything the importer applied.

    `pose.bones[].matrix` is object space, and the importer's rest_local_inv and
    matrix_local factors cancel down the chain, so it is exactly the world matrix the
    importer built -- no reconstruction from matrix_basis needed.
    """
    missing = [b.name for b in m.bones if b.name not in arm_obj.pose.bones]
    if missing:
        raise ValueError("armature has no bone %s (and %d more)"
                         % (missing[0], len(missing) - 1))
    pbs = [arm_obj.pose.bones[b.name] for b in m.bones]
    scene = context.scene
    out = []
    for f in frames:
        scene.frame_set(f)
        world = [_rows(pb.matrix, scale) for pb in pbs]
        if root_motion and anim is not None and anim.movements:
            off = mdl_mod.mat_inverse(mdl_mod.root_motion_matrix(anim, f))
            world = [mdl_mod.mat_mul(off, w) for w in world]
        out.append(write_mod.local_from_world(m, world))
    return write_mod.unwind_signs(out)


def mesh_objects(m, source):
    """{(bodypart index, model index): object} for the scene meshes belonging to `m`.

    The importer stamps `vtmb_bodypart`/`vtmb_model` on each object and `vtmb_source` on
    the armature it parents them to, so the file is identified through the parent rather
    than by object name, which the user is free to change.
    """
    flat = mesh_mod.models_of(m)
    # Model names repeat across bodyparts, so the name pair is only a fallback for objects
    # an older version imported without `vtmb_index`.
    want = {(bp.name, mo.name): (bi, mi) for bi, mi, bp, mo in flat}
    out = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        par = obj.parent
        if par is not None and not same_file(par.get("vtmb_source"), source):
            continue
        gi = obj.get("vtmb_index")
        if gi is not None and 0 <= gi < len(flat):
            key = flat[gi][0], flat[gi][1]
        else:
            key = want.get((obj.get("vtmb_bodypart"), obj.get("vtmb_model")))
        if key is not None:
            out.setdefault(key, obj)
    return out


def _per_vertex(me, count, get):
    """Every distinct per-loop value each vertex carries, as a list per vertex."""
    out = [[] for _ in range(count)]
    for loop in me.loops:
        out[loop.vertex_index].append(get(loop))
    return out


def _one_uv(me, count, uv_layer):
    """One UV per vertex, refusing a vertex whose loops disagree: the file stores one UV
    per vertex and spells a seam by duplicating the vertex, so a seam cut in Blender has
    nowhere to go without changing the count."""
    per = _per_vertex(me, count, lambda l: tuple(uv_layer.data[l.index].uv))
    split = [i for i, vs in enumerate(per)
             if any(max(abs(a - b) for a, b in zip(vs[0], v)) > UV_TOL for v in vs)]
    if split:
        raise ValueError("%s: %d vertices carry more than one UV (first is vertex %d), "
                         "which the format can only store by splitting the vertex and "
                         "that would change the vertex count"
                         % (me.name, len(split), split[0]))
    return [vs[0] if vs else None for vs in per]


def _one_normal(me, count, stash):
    """One normal per vertex, and how many were taken from Blender rather than `stash`.

    Averaging the corners is only ever a fallback: the round trip through custom split
    normals is lossy, so a vertex still within NORMAL_EPS of the normal the importer
    stashed is written back exactly instead of being degraded by re-export.
    """
    per = _per_vertex(me, count, lambda l: tuple(me.corner_normals[l.index].vector))
    out, edited = [], 0
    for i, vs in enumerate(per):
        if not vs:
            out.append(stash[i] if stash else None)
            continue
        avg = [sum(v[c] for v in vs) / len(vs) for c in range(3)]
        n = math.sqrt(sum(x * x for x in avg)) or 1.0
        avg = tuple(x / n for x in avg)
        if stash and max(abs(a - b) for a, b in zip(avg, stash[i])) <= NORMAL_EPS:
            out.append(stash[i])
        else:
            out.append(avg)
            edited += 1
    return out, edited


def _stashed(me, name, count, width=4, field="color"):
    """An attribute the importer wrote, as one tuple per vertex."""
    att = me.attributes.get(name)
    if att is None or len(att.data) != count:
        return None
    flat = [0.0] * (count * width)
    att.data.foreach_get(field, flat)
    return [tuple(flat[i * width:i * width + width]) for i in range(count)]


def _one_skin(v, groups, bone_index, stash):
    """(weights, bones, count) for one vertex, keeping the file's slot order where it
    still describes the same skinning -- the order is not derivable, so losing it would
    move bytes on a rewrite that changed nothing."""
    pairs = sorted(((g.weight, -bone_index[groups[g.group].name])
                    for g in v.groups if groups[g.group].name in bone_index),
                   reverse=True)[:4]
    w = [x for x, _ in pairs] + [0.0] * (4 - len(pairs))
    b = [-x for _, x in pairs] + [0] * (4 - len(pairs))
    n = sum(1 for x in w if x > 0.0)
    if stash is not None:
        sw, sb, sn = stash
        if sorted(zip(mesh_mod.u8_weights(sw), (int(x) for x in sb))) == \
           sorted(zip(mesh_mod.u8_weights(w), b)):
            return list(sw), [int(x) for x in sb], sn
    return w, b, n


def read_mesh(obj, model, bone_index, fields):
    """One `mdl.Vertex` per file vertex, in file order, inverting what the importer did.

    Only the requested fields are read, because the per-loop agreement check refuses a
    mesh a write of some other field would have been fine with.
    """
    me = obj.data
    if len(me.vertices) != model.numvertices:
        raise ValueError("%s: %d vertices in the scene, %d in %s -- a mesh write cannot "
                         "change the count"
                         % (obj.name, len(me.vertices), model.numvertices, model.name))
    n = model.numvertices
    uvs = normals = [None] * n
    edited = 0
    if "uvs" in fields:
        uv_layer = me.uv_layers.active
        if uv_layer is None:
            raise ValueError("%s has no UV layer" % obj.name)
        uvs = _one_uv(me, n, uv_layer)
    if "normals" in fields:
        normals, edited = _one_normal(
            me, n, _stashed(me, NORMAL_ATTR, n, width=3, field="vector"))
    skin = None
    if "weights" in fields:
        sb, sw = _stashed(me, SKIN_ATTR, n), _stashed(me, WEIGHT_ATTR, n)
        sn = _stashed(me, COUNT_ATTR, n, width=1, field="value")
        skin = list(zip(sw, sb, [int(x[0]) for x in sn])) if sb and sw and sn else None
    out = []
    for i, v in enumerate(me.vertices):
        x = mdl_mod.Vertex()
        x.pos = tuple(v.co)
        # None where Blender holds nothing: a vertex no triangle references has no loops,
        # so it has neither UV nor corner normal, and the file's own bytes must stand.
        x.normal = normals[i]
        x.uv = (uvs[i][0], 1.0 - uvs[i][1]) if uvs[i] else None
        x.weights, x.bones, x.numbones = _one_skin(
            v, obj.vertex_groups, bone_index, skin[i] if skin else None)
        out.append(x)
    return out, edited


def read_meshes(m, source, fields):
    """{(bodypart, model): vertices} for every model of `m` the scene supplies, plus the
    models it does not and the fields the file cannot carry."""
    found = mesh_objects(m, source)
    bone_index = {b.name: b.index for b in m.bones}
    edits, missing, unsupported, renormals = {}, [], set(), 0
    for bi, mi, _bp, mo in mesh_mod.models_of(m):
        ok, no = mesh_mod.supported(mo.filetype, fields)
        unsupported |= set(no)
        obj = found.get((bi, mi))
        if obj is None:
            missing.append(mo.name)
            continue
        if ok:
            edits[(bi, mi)], n = read_mesh(obj, mo, bone_index, ok)
            renormals += n
    return edits, missing, sorted(unsupported), renormals


def name_index(anim_names, action):
    """The slot an action lands on when no slot was named, or None.

    Its own name first, `vtmb_anim_index` second: duplicating an action copies the
    property, so it can point at a slot the action is no longer called after. The dialog
    and the write must both come through here, or one displays a slot and the other
    writes a different one.
    """
    for i, n in enumerate(anim_names):
        if n == action.name:
            return i
    i = action.get("vtmb_anim_index")
    return int(i) if i is not None and 0 <= int(i) < len(anim_names) else None


def resolve_target(m, action, name):
    """Which animation of the donor this action replaces."""
    names = [a.name for a in m.anims]
    if name:
        if name not in names:
            raise ValueError("%s has no animation named %r" % (m.path, name))
        return names.index(name)
    i = name_index(names, action)
    if i is None:
        raise ValueError("action %r is not named after any animation of %s and carries "
                         "no usable index, so name the target animation explicitly"
                         % (action.name, os.path.basename(m.path)))
    return i


def same_file(a, b):
    try:
        return bool(a) and bool(b) and os.path.samefile(a, b)
    except OSError:
        return bool(a) and bool(b) and \
            os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def match_indices(anim_names, source):
    """Every action in the blend that names one of `anim_names`, as ({index: action}, ignored).

    The name decides, and `vtmb_anim_index` is only a fallback for an action whose name
    matches nothing. Duplicating an action copies its custom properties, so several
    actions routinely carry one index while showing three different names; letting the
    invisible property outvote the visible name made that an error the user could not
    see the cause of. Takes names rather than an Mdl so the dialog can call it every
    redraw off its cached list.
    """
    by_name = set(anim_names)
    claims = {}
    for act in bpy.data.actions:
        if act.name not in by_name and not same_file(act.get("vtmb_source"), source):
            continue
        i = name_index(anim_names, act)
        if i is None:
            continue
        claims.setdefault(i, []).append(act)
    found, ignored = {}, []
    for i, acts in sorted(claims.items()):
        best = next((a for a in acts if a.name == anim_names[i]), acts[0])
        found[i] = best
        ignored += [(anim_names[i], a.name) for a in acts if a is not best]
    return found, ignored


def match_actions(m, source):
    return match_indices([a.name for a in m.anims], source)


def describe_ignored(ignored, limit=3):
    head = "; ".join("%s kept over %s" % (n, a) for n, a in ignored[:limit])
    more = "" if len(ignored) <= limit else " and %d more" % (len(ignored) - limit)
    return "%d action%s ignored, same target: %s%s" % (
        len(ignored), "" if len(ignored) == 1 else "s", head, more)


def export_actions(context, arm_obj, source, dest, actions, scale=1.0,
                   frame_start=None, frame_end=None, fps=None, keep_travel=True,
                   travel="keep", mesh_fields=(), rebuild=False):
    """Write `actions`, an {animation index: action} map, into a copy of `source`.

    An empty map is meaningful: it re-emits every animation unchanged, which is how the
    dialog's "copy unchanged" spells itself and the only way to rebuild a file without
    touching its content.
    """
    m = mdl_mod.Mdl(source)
    ad = arm_obj.animation_data
    if actions and ad is None:
        raise ValueError("%s has no animation data" % arm_obj.name)
    restore = ad.action if ad else None

    edits, wrote, yaw_lost = {}, [], []
    try:
        for index, action in sorted(actions.items()):
            if ad.action is not action:
                ad.action = action
            anim = m.anims[index]
            lo, hi = action.frame_range
            a = int(round(lo)) if frame_start is None else frame_start
            b = int(round(hi)) if frame_end is None else frame_end
            frames = list(range(a, b + 1))
            if not frames:
                raise ValueError("%s: empty frame range" % action.name)

            # "extract" wants the travel still in the poses, so root motion is only taken
            # back out when the donor's own blocks are the ones being kept.
            poses = read_poses(context, arm_obj, m, frames, scale, anim,
                               root_motion=(travel == "keep" and keep_travel
                                            and bool(action.get("vtmb_root_motion"))))
            movements = None
            if travel == "extract":
                movements, poses = write_mod.extract_travel(m, poses)
                # 21 shipped blocks author a yaw (three monsters' turn animations), and
                # fit_movements never writes one, so replacing those blocks straightens
                # the turn. The user asked for a refit, so it proceeds -- named.
                if any(mv.angle for mv in anim.movements):
                    yaw_lost.append(anim.name)
            elif travel == "none":
                movements = []
            edits[index] = {"poses": poses, "movements": movements,
                            "fps": fps if fps else action.get("vtmb_fps"),
                            "flags": action.get("vtmb_flags")}
            wrote.append((index, anim.name, action.name, len(frames),
                          len(anim.movements if movements is None else movements)))
    finally:
        if ad is not None and ad.action is not restore:
            ad.action = restore

    mesh = {"fields": tuple(mesh_fields), "verts": 0, "models": 0,
            "missing": [], "unsupported": [], "normals": 0}
    if mesh_fields:
        # Before the animation write, so it happens on the file the offsets were read
        # from rather than on a rebuilt one.
        cells, mesh["missing"], mesh["unsupported"], mesh["normals"] = \
            read_meshes(m, source, mesh_fields)
        if not cells and not mesh["missing"]:
            raise ValueError("no scene mesh belongs to %s" % os.path.basename(source))
        if cells:
            patched, mesh["verts"] = mesh_mod.patch(m, cells, mesh_fields)
            mesh["models"] = len(cells)
            m = mdl_mod.Mdl(source, data=patched)

    # Re-emitting animations nobody replaced relocates the region and strands the old one,
    # which on a pristine file costs ~90 KB for no change at all.
    data = write_mod.write_many(m, edits) if edits else bytes(m.d)
    built = None
    if rebuild:
        # The donor's checksum is kept: the .vtx beside this file still carries it, and the
        # engine draws nothing at all when the two disagree.
        before = data
        data, built = rebuild_mod.rebuild(before)
        bad, _n = rebuild_mod.verify(before, data)
        if bad:
            raise ValueError("rebuild changed what the file says: %s" % "; ".join(bad))
    with open(dest, "wb") as f:
        f.write(data)
    return {"wrote": wrote, "bones": len(m.bones), "mesh": mesh, "rebuild": built,
            "bytes": len(data), "was": len(m.d), "anims": len(m.anims),
            "yaw_lost": yaw_lost}


def export_action(context, arm_obj, source, dest, anim_name="", **kw):
    """One active action into one animation. Kept because it is what the round-trip and
    self-check scripts drive."""
    ad = arm_obj.animation_data
    action = ad.action if ad else None
    if action is None:
        raise ValueError("%s has no action assigned" % arm_obj.name)
    index = resolve_target(mdl_mod.Mdl(source), action, anim_name)
    r = export_actions(context, arm_obj, source, dest, {index: action}, **kw)
    i, name, _, frames, movements = r["wrote"][0]
    r.update(anim=name, index=i, frames=frames, movements=movements)
    return r
