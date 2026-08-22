#!/usr/bin/env python3
"""Blender side of the VTMB MDL exporter: actions back into a .mdl's animations.

Animation is the only thing written. Mesh, UVs, the skeleton's bind pose, materials,
sequences, hitboxes and everything else keep the bytes of the .mdl being rewritten, so
what comes out is that model with some of its animations replaced -- never a model built
from the scene.
"""

import math
import os
import struct

import bpy
import mathutils

from . import mdl as mdl_mod
from . import mdl_build as build_mod
from . import mdl_rebuild as rebuild_mod
from . import mdl_write as write_mod
from . import mesh_write as mesh_mod


UV_TOL = 1e-6
# Custom split normals are stored as two 16-bit angles, so an untouched mesh's corners
# disagree by up to 9.2e-03; under this a normal is taken as unedited.
NORMAL_EPS = 1.5e-2
NORMAL_ATTR = "vtmb_normal"
# Against the import stash an untouched bone compares exactly; this only absorbs a
# recomposition of the same float32s.
REST_EPS = 1e-6
# Without a stash the file's record is the baseline, and Blender loses 3.2e-05 of a bone's
# offset to chain composition and 1.4e-03 per quaternion entry to head/tail/roll storage.
REST_POS_EPS = 1e-3
REST_POS_REL = 1e-4
REST_QUAT_EPS = 2e-3
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


def to_file(local, scale):
    """(pos, quat) in the file's convention out of a parent-relative Blender matrix.

    The importer scales translation only, so the rotation comes back untouched and only the
    offset divides out. Blender orders a quaternion w first and the file x first.
    """
    t, q = local.to_translation(), local.to_quaternion()
    return (t.x / scale, t.y / scale, t.z / scale), (q.x, q.y, q.z, q.w)


def bone_flags(arm_obj, name):
    """`vtmb_bone_flags` as the importer stashed it, or None to keep the file's own.

    Verbatim, with none of `blender_scratch`'s zero-used-by-mask guard: there is a file
    value here and it is already what the engine accepts -- manbat's Dummy01 ships 0x0 --
    so forcing a bit in would edit a bone nobody asked to edit.
    """
    pb = arm_obj.pose.bones.get(name)
    f = pb.get("vtmb_bone_flags") if pb is not None else None
    return None if f is None else int(f)


def rest_baseline(arm_obj, name):
    """`vtmb_rest_local` as a Matrix, or None where the importer stashed none."""
    pb = arm_obj.pose.bones.get(name)
    v = pb.get("vtmb_rest_local") if pb is not None else None
    if v is None or len(v) != 16:
        return None
    return mathutils.Matrix([tuple(v[i * 4:i * 4 + 4]) for i in range(4)])


def moved_from_file(m, b, pos, quat):
    """Did this bone leave the file's own record, allowing for what Blender cannot hold.

    Only for a bone the importer stashed no baseline for. The bounds are far coarser than
    an edit, so the answer is one-sided: a real edit under them is missed, silently.
    """
    tol = REST_POS_EPS + REST_POS_REL * max(abs(x) for x in b.pos)
    return not (all(abs(x - y) <= tol for x, y in zip(pos, b.pos))
                and all(abs(x - y) <= REST_QUAT_EPS for x, y in zip(quat, b.quat)))


def read_bones(m, arm_obj, scale):
    """{index: (pos, quat, flags)} for the bones of `m` the armature actually moved.

    A bone still sitting on its imported pose is left out, so an unedited export does not
    touch the bone array and comes back byte for byte -- the same reason the mesh puts an
    unmoved vertex back with its file bytes rather than Blender's.

    The file's parent chain is what the local matrix is taken against, not Blender's. The
    two agree on an imported armature, and where they disagree it is the file the output
    has to stay consistent with.
    """
    dbs = arm_obj.data.bones
    missing = [b.name for b in m.bones if b.name not in dbs]
    if missing:
        raise ValueError("armature has no bone %s (and %d more)"
                         % (missing[0], len(missing) - 1))
    out = {}
    for k, b in enumerate(m.bones):
        local = dbs[b.name].matrix_local
        if b.parent >= 0:
            local = dbs[m.bones[b.parent].name].matrix_local.inverted() @ local
        pos, quat = to_file(local, scale)
        flags = bone_flags(arm_obj, b.name)
        # A quaternion and its negation are one rotation, so the nearer sign wins or a bone
        # the importer flipped reads as moved.
        if sum(x * y for x, y in zip(quat, b.quat)) < 0:
            quat = tuple(-x for x in quat)
        base = rest_baseline(arm_obj, b.name)
        if base is None:
            moved = moved_from_file(m, b, pos, quat)
        else:
            moved = any(abs(local[i][j] - base[i][j]) > REST_EPS
                        for i in range(4) for j in range(4))
        if not moved and (flags is None or flags == b.flags):
            continue
        out[k] = (pos, quat, flags)
    return out


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


def named_index(anim_names, action):
    """The slot whose animation bears this action's exact name, or None."""
    for i, n in enumerate(anim_names):
        if n == action.name:
            return i
    return None


def stamped_index(anim_names, action):
    """The slot `vtmb_anim_index` points at, or None when it is absent or out of range."""
    i = action.get("vtmb_anim_index")
    return int(i) if i is not None and 0 <= int(i) < len(anim_names) else None


def name_index(anim_names, action):
    """The slot one named action lands on, or None.

    Its own name first, `vtmb_anim_index` second: duplicating an action copies the
    property, so it can point at a slot the action is no longer called after. The dialog
    and the write must both come through here, or one displays a slot and the other
    writes a different one. Only for the paths that carry a single action -- with several
    in play the stamp is not enough on its own, which is what `match_indices` decides.
    """
    i = named_index(anim_names, action)
    return stamped_index(anim_names, action) if i is None else i


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


def armature_actions(arm_obj):
    """Every action the armature currently plays -- its own, and its NLA strips'."""
    ad = getattr(arm_obj, "animation_data", None)
    if ad is None:
        return []
    out = [ad.action] if ad.action else []
    for tr in ad.nla_tracks:
        out += [s.action for s in tr.strips if s.action]
    return out


def match_indices(anim_names, source, arm_obj=None):
    """Which action replaces which animation, as ({index: action}, unwritten, addable).

    The name decides; `vtmb_anim_index` answers only for an action nobody else contests.
    `action.copy()` carries custom properties over, so `run_0.001`, `run_0.002` and
    `run_0.003` all read index 0 and all claim animation 0 -- picking one is picking by
    `bpy.data.actions` order, which the author cannot see, so none is written and the tie
    is reported. `unwritten` is (animation, action, kept) for every candidate left out,
    `kept` being the action that took the slot or None when the animation is left alone.

    `addable` is the same actions the `anim is None` rows name, as objects rather than
    names: an action of this file's that matches no animation of it, and one the armature
    plays that claims nothing at all. Those are exactly the ones an append can take, and
    they are collected here so the rule lives in one place rather than in the dialog.

    Takes names rather than an Mdl so the dialog can call it every redraw off its cached
    list.
    """
    named, stamped, loose = {}, {}, []
    for act in bpy.data.actions:
        i = named_index(anim_names, act)
        if i is not None:
            named[i] = act
            continue
        if not same_file(act.get("vtmb_source"), source):
            continue
        j = stamped_index(anim_names, act)
        if j is None:
            loose.append(act)
        else:
            stamped.setdefault(j, []).append(act)

    found, unwritten = dict(named), []
    for j, acts in sorted(stamped.items()):
        if j in found:
            unwritten += [(anim_names[j], a.name, found[j].name) for a in acts]
        elif len(acts) == 1:
            found[j] = acts[0]
        else:
            unwritten += [(anim_names[j], a.name, None) for a in acts]
    # Came from this file and now points at no animation of it -- a rename, or a stamp
    # that outlived the animation it named.
    addable = list(loose)
    unwritten += [(None, a.name, None) for a in loose]
    # Keying into a fresh action leaves neither a name nor a stamp, so it never became a
    # candidate above and would otherwise be dropped without appearing anywhere at all.
    seen = set(found.values()) | {a for a in loose}
    named_out = {u[1] for u in unwritten}
    for act in armature_actions(arm_obj):
        if act not in seen and act.name not in named_out:
            unwritten.append((None, act.name, None))
            named_out.add(act.name)
            addable.append(act)
    return found, unwritten, addable


def match_actions(m, source, arm_obj=None):
    return match_indices([a.name for a in m.anims], source, arm_obj)


def split_unwritten(unwritten, adding):
    """(rows an append would take, rows nothing takes).

    A row with no animation named it is an action matching nothing, which is the append
    candidate; every other row is a contested slot, which adding cannot help.
    """
    if not adding:
        return [], list(unwritten)
    return ([u for u in unwritten if u[0] is None],
            [u for u in unwritten if u[0] is not None])


def unwritten_line(anim, act, kept):
    """One row of `match_indices`' second return, in the dialog's own arrow idiom."""
    if kept is not None:
        return "%s ← %s (not %s)" % (anim, kept, act)
    if anim is None:
        return "%s → nothing" % act
    return "%s ← nothing (%s and others claim it)" % (anim, act)


def describe_unwritten(unwritten, limit=3):
    head = "; ".join(unwritten_line(*u) for u in unwritten[:limit])
    more = "" if len(unwritten) <= limit else " and %d more" % (len(unwritten) - limit)
    return "%d action%s not written: %s%s" % (
        len(unwritten), "" if len(unwritten) == 1 else "s", head, more)


def read_materials(m, source):
    """{file texture index: name} from the scene's material slots.

    `obj["vtmb_meshes"]` records which slot each file mesh landed in, and it is the only
    thing that maps a material back: two meshes may share one, and Blender's per-face slot
    cannot speak for a mesh whose triangles all sit in a lower LOD.
    """
    out = {}
    for (bi, mi), obj in sorted(mesh_objects(m, source).items()):
        stash = obj.get("vtmb_meshes")
        if not stash:
            continue
        mats = obj.data.materials
        for j, mesh in enumerate(m.bodyparts[bi].models[mi].meshes):
            if j >= len(stash):
                break
            slot = int(stash[j][0])
            if not 0 <= slot < len(mats) or mats[slot] is None:
                continue
            ref = mesh.material
            if m.skins and 0 <= ref < len(m.skins[0]):
                ref = m.skins[0][ref]
            name = mats[slot].name
            if out.get(ref, name) != name:
                raise ValueError("texture %d is named both %r and %r by the scene's "
                                 "materials" % (ref, out[ref], name))
            out[ref] = name
    return out


def apply_sequences(d, arm_obj, anim_names):
    """Label, activity, group size and the blend grid out of `arm_obj["vtmb_sequences"]`.

    Matched by position: the stash is written in file order. A file carrying more than the
    stash does is one an append has just grown, and the sequences past its end are ones the
    scene says nothing about, so they keep what `add_sequence` wrote; a stash longer than
    the file claims sequences that are not there and is refused. Blends are stored as
    animation names because an index means nothing once the file is re-emitted, so they
    resolve back through `anim_names`.
    """
    stash = arm_obj.get("vtmb_sequences")
    if not stash:
        return 0
    if len(stash) > len(d.seqs):
        raise ValueError("the armature carries %d sequences and the file has %d"
                         % (len(stash), len(d.seqs)))
    index = {n: k for k, n in enumerate(anim_names)}
    n = 0
    for rec, s in zip(d.seqs, stash):
        label, activity = s.get("label"), s.get("activity")
        if label and rec.name != label:
            rec.name, n = label, n + 1
        if activity and rec.extra.get("activity") != activity:
            rec.extra["activity"], n = activity, n + 1
        blends = [list(col) for col in s.get("blends") or []]
        gx, gy = len(blends), max((len(c) for c in blends), default=0)
        # The grid sits inside the 764-byte record at a 0x20 stride, so one that would run
        # past it is left alone rather than written over the fields behind it.
        if not gx or mdl_mod.SEQ_ANIM + (gx - 1) * 0x20 + gy * 2 > len(rec.raw):
            continue
        was = struct.unpack_from("<ii", rec.raw, mdl_mod.SEQ_GROUPSIZE)
        struct.pack_into("<ii", rec.raw, mdl_mod.SEQ_GROUPSIZE, gx, gy)
        if was != (gx, gy):
            n += 1
        for x, col in enumerate(blends):
            for y, name in enumerate(col):
                at = mdl_mod.SEQ_ANIM + x * 0x20 + y * 2
                a = index.get(name, -1)
                if a < 0:
                    raise ValueError("sequence %r blends animation %r, which the file does "
                                     "not have" % (rec.name, name))
                if struct.unpack_from("<h", rec.raw, at)[0] != a:
                    n += 1
                struct.pack_into("<h", rec.raw, at, a)
    return n


def verify_writer(src):
    """The writer must reproduce this donor before anything from the scene is fed in.

    Grades the relayout alone, which is what makes the differences that follow attributable
    to the scene: an edit is an intended difference and would drown the check, so the
    comparison is `emit(from_bytes(x))` against `x` and never against the edited output.
    """
    plain = build_mod.emit(build_mod.from_bytes(src))
    bad, _n = rebuild_mod.verify(src, plain)
    return bad


def export_actions(context, arm_obj, source, dest, actions, scale=1.0,
                   frame_start=None, frame_end=None, fps=None, keep_travel=True,
                   travel="keep", mesh_fields=(), verify=True, add=()):
    """Author `source` again with `actions`, an {animation index: action} map, applied.

    `add` is actions appended as new animations rather than replacing one, each with a
    sequence of the same name -- the engine reaches an animation only through a sequence,
    so one with none is dead weight. Appending renumbers nothing, every stored index
    pointing below the insertion point, and `emit` requantises the animations already in
    the file when the new pose widens the file-wide `mstudiobone_t` scales.

    The file is always rebuilt from its own decoded records -- every count from a `len()`,
    every offset from where its target landed -- so an empty map is meaningful and re-emits
    the model unchanged rather than copying it.

    The skeleton, the material names and the sequence table come from the scene too, and
    unconditionally: each is compared against what the file already says and written only
    where it differs, so an unedited model still comes back byte for byte.
    """
    m = mdl_mod.Mdl(source)
    if verify:
        bad = verify_writer(bytes(m.d))
        if bad:
            raise ValueError("the writer does not reproduce %s, so nothing was written: %s"
                             % (os.path.basename(source), "; ".join(bad)))
    ad = arm_obj.animation_data
    if actions and ad is None:
        raise ValueError("%s has no animation data" % arm_obj.name)
    restore = ad.action if ad else None

    edits, wrote, pending = {}, [], []
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
            elif travel == "none":
                movements = []
            edits[index] = {"poses": poses, "movements": movements,
                            "fps": fps if fps else action.get("vtmb_fps"),
                            "flags": action.get("vtmb_flags")}
            wrote.append((index, anim.name, action.name, len(frames),
                          len(anim.movements if movements is None else movements)))

        for action in add:
            if ad.action is not action:
                ad.action = action
            lo, hi = action.frame_range
            a = int(round(lo)) if frame_start is None else frame_start
            b = int(round(hi)) if frame_end is None else frame_end
            frames = list(range(a, b + 1))
            if not frames:
                raise ValueError("%s: empty frame range" % action.name)
            # No donor animation behind this one, so there is no travel to take back out
            # and nothing for "keep" to keep: an appended animation either has the blocks
            # fitted here or has none.
            poses = read_poses(context, arm_obj, m, frames, scale)
            movements = ()
            if travel == "extract":
                movements, poses = write_mod.extract_travel(m, poses)
            pending.append((action, poses, tuple(movements), len(frames)))
    finally:
        if ad is not None and ad.action is not restore:
            ad.action = restore

    d = build_mod.apply_anims(build_mod.from_bytes(bytes(m.d)), edits, source)

    scene = {"bones": 0, "materials": 0, "sequences": 0, "stale": 0}
    poses = read_bones(m, arm_obj, scale)
    if poses:
        build_mod.set_bone_poses(d, poses)
        scene["bones"] = len(poses)
        # A rotation channel is `int16 * rotscale` with no bind base, so an animation left
        # un-re-encoded keeps a pose the moved bind disagrees with -- 1.0e-01 against 5.4e-04.
        scene["stale"] = len(d.anims) - len(edits)
    for ref, name in sorted(read_materials(m, source).items()):
        if 0 <= ref < len(d.textures) and d.textures[ref].name != name:
            d.textures[ref].name = name
            scene["materials"] += 1
    # Before the append, not after: apply_sequences refuses outright when the armature's
    # stash and the file disagree on how many sequences there are.
    scene["sequences"] = apply_sequences(d, arm_obj, [r.name for r in d.anims])

    added = []
    for action, poses, movements, nframes in pending:
        rate = float(fps or action.get("vtmb_fps") or context.scene.render.fps)
        i = build_mod.add_animation(d, action.name, poses, rate,
                                    int(action.get("vtmb_flags") or 0), movements)
        build_mod.add_sequence(d, action.name, i, action.get("vtmb_activity"),
                               int(action.get("vtmb_seq_flags") or 0))
        added.append((i, action.name, nframes, len(movements)))

    mesh = {"fields": tuple(mesh_fields), "verts": 0, "models": 0,
            "missing": [], "unsupported": [], "normals": 0}
    if mesh_fields:
        cells, mesh["missing"], mesh["unsupported"], mesh["normals"] = \
            read_meshes(m, source, mesh_fields)
        if not cells and not mesh["missing"]:
            raise ValueError("no scene mesh belongs to %s" % os.path.basename(source))
        for (bi, mi), verts in sorted(cells.items()):
            rec = d.bodyparts[bi].kids[mi]
            rec.extra["verts"], n = mesh_mod.pack_model(
                m.bodyparts[bi].models[mi], verts, mesh_fields, rec.extra["verts"])
            mesh["verts"] += n
        mesh["models"] = len(cells)

    # The donor's checksum is kept: the .vtx beside this file still carries it, and the
    # engine draws nothing at all when the two disagree.
    data = build_mod.emit(d, checksum=d.checksum)
    with open(dest, "wb") as f:
        f.write(data)
    return {"wrote": wrote, "added": added, "bones": len(m.bones), "mesh": mesh,
            "scene": scene, "bytes": len(data), "was": len(m.d), "anims": len(m.anims)}


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
