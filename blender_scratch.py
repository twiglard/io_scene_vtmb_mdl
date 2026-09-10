"""Author a .mdl and its .dx80.vtx from a Blender scene, with no donor file.

`blender_export` rewrites a file that already exists: it opens with `Mdl(source)` and
carries every record the scene does not supply. This does not open anything. The armature
becomes `mstudiobone_t[]`, the mesh objects become bodyparts, the actions become animations
and one sequence each, and everything the format holds that no scene expresses -- cloth,
flex, eyeballs, spring bones, procedural bones, IK -- is emitted with a count of zero.

The rest pose is re-derived from the armature rather than carried. That costs `2.198e-04`
units of bone translation and `6.386e-04` units of vertex displacement at r=8 worst over
2140 bone comparisons (`plans/rest-fidelity.py`), against a `posscale` step of 1/256, and
it converges rather than accumulating.
"""

import os
import struct

import bpy  # noqa: F401

# Package-only, like blender_export, which it reuses: both are bpy-side and neither is
# imported flat by a check script.
from . import blender_export as export_mod
from . import cloth as cloth_mod
from . import mdl as mdl_mod
from . import mdl_build as build_mod
from . import mdl_write as write_mod
from . import paths as paths_mod
from . import vtx_rebuild as vtxr_mod

# A bone needs at least one bit of the 0xFFFC used-by mask or it gets no bone matrix and
# everything weighted to it renders nowhere.
BONE_FLAGS = build_mod.BONE_USED

# Splitting key resolution. Two loops of one vertex that agree to this are one file vertex.
NORMAL_Q = 1e-5
UV_Q = 1e-6

# `blind` is the normals that moved past the round trip's own error and not past
# NORMAL_EPS: written from the file, so the edit is lost. Same band the in-place path reports.
NO_EDITS = (("uvs", 0), ("normals", 0), ("blind", 0), ("added", 0))
# An original that has not been moved sits exactly on its imported position -- float32 both
# sides -- so this is slack, not licence to have moved.
HOME_TOL = 1e-3


class Refused(Exception):
    pass


def bone_order(arm_obj):
    """Armature bones with every parent ahead of its children.

    `add_bone` refuses a forward parent reference, and the engine walks the array once in
    order, so a child emitted first reads a world matrix that has not been computed.
    """
    out, seen = [], set()

    def visit(b):
        if b.name in seen:
            return
        if b.parent is not None:
            visit(b.parent)
        seen.add(b.name)
        out.append(b)

    for b in arm_obj.data.bones:
        visit(b)
    return out


_to_file = export_mod.to_file


def _rest_local(db, parent_db):
    if parent_db is None:
        return db.matrix_local.copy()
    return parent_db.matrix_local.inverted() @ db.matrix_local


def _flags_of(arm_obj, name):
    """`vtmb_bone_flags` if the importer stashed it, and never a zero used-by mask: a bone
    with no bit of 0xFFFC gets no bone matrix and draws nothing weighted to it."""
    pb = arm_obj.pose.bones.get(name)
    f = int(pb.get("vtmb_bone_flags", 0)) if pb is not None else 0
    return f if f & 0xFFFC else f | BONE_FLAGS


def add_bones(d, arm_obj, scale=1.0, surfaceprop="flesh"):
    """Every armature bone as an `mstudiobone_t`. Returns {name: index}."""
    index = {}
    for db in bone_order(arm_obj):
        parent = -1 if db.parent is None else index[db.parent.name]
        pos, quat = _to_file(_rest_local(db, db.parent), scale)
        index[db.name] = build_mod.add_bone(d, db.name, parent, pos, quat,
                                            flags=_flags_of(arm_obj, db.name),
                                            surfaceprop=surfaceprop)
    return index


def _skin(v, groups, bone_index):
    """The four heaviest bone bindings of one Blender vertex, as [(bone, weight), ...]."""
    pairs = []
    for g in v.groups:
        name = groups[g.group].name
        if name in bone_index and g.weight > 0.0:
            pairs.append((g.weight, bone_index[name]))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [(b, w) for w, b in pairs[:4]]


def _point_vectors(me, name, width):
    """A POINT attribute as one tuple per vertex, or None if the mesh has no such layer."""
    att = me.attributes.get(name)
    if att is None or len(att.data) != len(me.vertices):
        return None
    field = "color" if width == 4 else "vector"
    buf = [0.0] * (width * len(me.vertices))
    att.data.foreach_get(field, buf)
    return [tuple(buf[i * width:(i + 1) * width]) for i in range(len(me.vertices))]


def _point_ints(me, name):
    att = me.attributes.get(name)
    if att is None or len(att.data) != len(me.vertices):
        return None
    buf = [0] * len(me.vertices)
    att.data.foreach_get("value", buf)
    return buf


def identity_map(obj, me):
    """{original file vertex -> Blender vertex} for the tags that survived editing.

    A tag only one vertex claims is trusted however far it has been moved. A tag several
    claim resolves only to a claimant still standing where the file put that original,
    read from `obj["vtmb_orig_co"]`, which is keyed by file vertex so no edit rewrites it.

    The ceiling is what stops a subdivided vertex taking the slot of an original dragged
    further from home than the new vertex sits: distance cannot tell those two apart, and
    the wrong one would then be written as the original. Unresolved is the safe answer.
    """
    orig = _point_ints(me, "vtmb_orig")
    home = obj.get("vtmb_orig_co")
    if orig is None or home is None:
        return None
    claims = {}
    for vi, o in enumerate(orig):
        claims.setdefault(o, []).append(vi)
    out = {}
    for o, vs in claims.items():
        if len(vs) == 1:
            out[o] = vs[0]
            continue
        if (o + 1) * 3 > len(home):
            continue
        h = home[o * 3:o * 3 + 3]
        near = sorted((sum((me.vertices[vi].co[c] - h[c]) ** 2
                           for c in range(3)), vi) for vi in vs)
        if near[0][0] < near[1][0] and near[0][0] <= HOME_TOL * HOME_TOL:
            out[o] = near[0][1]
    return out


def original_runs(obj, me, why=None):
    """The file's mesh partition as [(material slot, [Blender vertex, ...]), ...].

    None unless every original vertex is still present: one that has gone cannot be put
    back in its slot, and renumbering around the hole is what the map exists to avoid.
    A slot naming two runs is also refused, since a vertex the edit added is placed by
    slot and there would be no saying which of them it belongs to. Each of those appends
    to `why`, which the caller reports and cannot work out for itself.
    """
    if why is None:
        why = []
    spec = obj.get("vtmb_meshes")
    if not spec:
        why.append("the object carries no vtmb_meshes, so it came from no .mdl import")
        return None
    ident = identity_map(obj, me)
    if ident is None:
        why.append("the object carries no vtmb_orig tags, so no vertex of the scene can "
                   "be matched to one of the file's")
        return None
    # The stamped slot resolved to where that material sits now, because the run is
    # matched against a polygon's live `material_index` below and Blender renumbers slots
    # on a reorder or a delete.
    moves = export_mod.slot_moves(obj)[0]
    runs, at, slots = [], 0, set()
    for slot, n in spec:
        members = [ident.get(at + k) for k in range(int(n))]
        if any(x is None for x in members):
            lost = next(at + k for k in range(int(n)) if members[k] is None)
            why.append("file vertex %d is claimed by no vertex of the scene, so it was "
                       "deleted or its tag was lost" % lost)
            return None
        here = int(slot) if moves is None else moves.get(int(slot))
        if here is None:
            why.append("the material slot one of the file's meshes was imported into is "
                       "no longer in the scene, so a vertex added to that mesh could not "
                       "be placed")
            return None
        if here in slots:
            why.append("material slot %d names two of the file's meshes" % here)
            return None
        slots.add(here)
        runs.append((here, members))
        at += int(n)
    return runs


def wound(corners):
    """`corners` in the order the format wants, which is the reverse of Blender's.

    Blender winds a triangle with its outward normal and the format winds it against, so
    the conversion is unconditional in both directions -- `blender_import` reverses on the
    way in and this reverses on the way out, and a round trip writes the file's own order
    back. Nothing here reads a normal to decide it: the winding is the format's own
    convention and holds whether or not a mesh has a normal to compare against.
    """
    return tuple(reversed(corners))


def split_mesh(obj, bone_index, scale=1.0, fields=("uvs", "normals")):
    """([(material slot, verts, faces), ...], unskinned, kept, why, edits) per corner.

    The format stores one normal, one UV and one skin per vertex, so a seam or a hard edge
    is spelled by duplicating the vertex. `kept` counts the originals written back into
    their own mesh in file order; it is 0 when the partition had to be rebuilt from the
    material slots instead, which puts every vertex in triangle-visit order. `why` names
    the one condition that forced that, and is None where it did not happen -- eight
    separate ones reach it and the caller refuses on the wrong one otherwise. `edits` is
    what the scene moved on the originals, in the keys of NO_EDITS, and is all zero on the
    rebuilt partition, which has no original to compare against.

    `fields` is the export's own, so an original keeps what was imported for any field the
    scene was not asked to supply. A vertex that has no original takes Blender's whatever
    is asked, there being nothing else to take.
    """
    me = obj.data
    if not me.polygons:
        raise Refused("%s has no faces" % obj.name)
    uv_layer = export_mod.uv_layer_of(obj)[0]
    if uv_layer is None:
        raise Refused("%s has no UV layer" % obj.name)
    me.calc_loop_triangles()
    normals = me.corner_normals
    # Custom split normals round-trip through two 16-bit angles, come back up to 9.2e-03
    # out and differ between loops of one vertex, which would split every vertex there is.
    stash = _point_vectors(me, "vtmb_normal", 3)
    uvs = _point_vectors(me, "vtmb_uv", 2)
    skins = [_skin(v, obj.vertex_groups, bone_index) for v in me.vertices]
    unskinned = sum(1 for s in skins if not s)

    def rec(vi, n, u, w):
        co = me.vertices[vi].co
        return ((co.x / scale, co.y / scale, co.z / scale), tuple(n),
                (u, 1.0 - w), skins[vi] or [(0, 1.0)])

    def key_of(vi, n, u, w):
        return (vi, round(n[0] / NORMAL_Q), round(n[1] / NORMAL_Q),
                round(n[2] / NORMAL_Q), round(u / UV_Q), round(w / UV_Q))

    why = []
    if stash is None:
        why.append("nothing stashed the file's own normals on this mesh -- it was "
                   "built by hand, or imported before every filetype carried one")
    elif uvs is None:
        why.append("nothing stashed the file's own UVs on this mesh")
    else:
        runs = original_runs(obj, me, why)
        if runs is not None:
            out = _split_preserved(me, runs, stash, uvs, normals, uv_layer,
                                   rec, key_of, why, fields)
            if out is not None:
                return out[0], unskinned, out[1], None, out[2]

    per_slot = {}
    for tri in me.loop_triangles:
        verts, faces, seen = per_slot.setdefault(tri.material_index, ([], [], {}))
        corners = []
        for li in tri.loops:
            vi = me.loops[li].vertex_index
            n = stash[vi] if stash else normals[li].vector
            u, w = uv_layer.data[li].uv
            key = key_of(vi, n, u, w)
            at = seen.get(key)
            if at is None:
                at = seen[key] = len(verts)
                verts.append(rec(vi, n, u, w))
            corners.append(at)
        faces.append(wound(corners))
    return ([(k, per_slot[k][0], per_slot[k][1]) for k in sorted(per_slot)],
            unskinned, 0, why[0] if why else "the file's partition was not recoverable",
            dict(NO_EDITS))


def _split_preserved(me, runs, stash, uvs, normals, uv_layer, rec, key_of, why, fields):
    """Every original vertex back in its own slot, then whatever an edit added after it.

    A vertex with no original of its own is placed in the run its triangle's material slot
    names, and takes Blender's own normal: `stash` and `uvs` are indexed by Blender vertex
    and an edit copies or interpolates them, so they belong to nobody but the original.
    None when a triangle straddles two runs, which cannot be written without renumbering.

    An original carries what the scene says about it for the fields the export asked for,
    and both halves of that had to be read the same way in the two passes: writing the
    stash into the file's own vertex and then keying the triangle off Blender's corner put
    a moved UV on a key no slot held, so the edit arrived as a second vertex appended
    behind the first and the file's own one was left drawn by nothing.
    """
    take_uv, take_normal = "uvs" in fields, "normals" in fields
    dead = export_mod._dead_loops(me)
    corners_of = {}
    for tri in me.loop_triangles:
        for li in tri.loops:
            corners_of.setdefault(me.loops[li].vertex_index, []).append(li)

    def averaged(vi):
        """(the normal every quiet corner of an original resolves to, how far it moved).

        Averaged rather than read per corner, which is what the in-place path does and for
        the same reason: a custom split normal round-trips through two 16-bit angles and
        comes back up to 9.2e-03 out, differing between loops of one vertex, so reading
        the corners raw would split every edited vertex into one copy per corner. Corners
        on a face with no area are left out -- BUGS 39 is what averaging them costs.

        An exactly-zero stash is kept whatever the corners say: Blender hands back
        (0, 0, 1) for one and no average can be near it, which is BUGS 35 on this path.
        """
        live = [li for li in corners_of.get(vi, ()) if li not in dead]
        if not take_normal or not live or stash[vi] == (0.0, 0.0, 0.0):
            return stash[vi], 0.0
        acc = [0.0, 0.0, 0.0]
        for li in live:
            v = normals[li].vector
            for c in range(3):
                acc[c] += v[c]
        length = (acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2) ** 0.5 or 1.0
        avg = tuple(x / length for x in acc)
        dev = max(abs(a - b) for a, b in zip(avg, stash[vi]))
        return (stash[vi] if dev <= export_mod.NORMAL_EPS else avg), dev

    decided, moved = {}, {}
    for _slot, members in runs:
        for vi in members:
            decided[vi], moved[vi] = averaged(vi)

    def corner(vi, li):
        """(normal, u, w) for one corner of an original.

        A corner further than NORMAL_EPS from what the vertex settled on is a hard edge
        the scene put there and still splits; the rest resolve to the one value, so an
        edit moves the vertex instead of duplicating it. A UV is float32 on both sides and
        needs none of that.
        """
        base = decided[vi]
        n = base if not take_normal else tuple(normals[li].vector)
        u, w = uv_layer.data[li].uv if take_uv else uvs[vi]
        if base == (0.0, 0.0, 0.0) or max(abs(a - b) for a, b in zip(n, base)) \
                <= export_mod.NORMAL_EPS:
            n = base
        return n, u, w

    def home(vi):
        """The corner an original keeps, and how far its UV moved.

        The file keeps this vertex's number and a flex key or a cloth row names it by
        that, so of the corners a seam split apart the original takes the one nearest what
        was imported. Untouched, every corner answers the stash and the first wins.

        No corner at all is a vertex only a stripped lower LOD drew -- 3716 of
        blueblood_female's 9129 -- and it is written anyway, the .vtx still naming it.
        """
        su, sw = uvs[vi]
        best = None
        for li in corners_of.get(vi, ()):
            n, u, w = corner(vi, li)
            duv = max(abs(u - su), abs(w - sw))
            dn = max(abs(a - b) for a, b in zip(n, decided[vi]))
            if best is None or (duv, dn) < best[:2]:
                best = (duv, dn, n, u, w)
        if best is None:
            return decided[vi], su, sw, 0.0
        return best[2], best[3], best[4], best[0]

    edits = dict(NO_EDITS)
    where, run_of_slot, out, seen, kept = {}, {}, [], [], 0
    for ri, (slot, members) in enumerate(runs):
        run_of_slot[slot] = ri
        verts, keys = [], {}
        for vi in members:
            where[vi] = ri
            n, u, w, duv = home(vi)
            if duv > UV_Q:
                edits["uvs"] += 1
            if moved[vi] > export_mod.NORMAL_EPS:
                edits["normals"] += 1
            elif moved[vi] > export_mod.NORMAL_NOISE:
                edits["blind"] += 1
            keys[key_of(vi, n, u, w)] = len(verts)
            verts.append(rec(vi, n, u, w))
            kept += 1
        out.append((verts, []))
        seen.append(keys)

    for tri in me.loop_triangles:
        corners = []
        for li in tri.loops:
            vi = me.loops[li].vertex_index
            ri = where.get(vi)
            if ri is None:
                ri = run_of_slot.get(tri.material_index)
                if ri is None:
                    why.append("a triangle sits on material slot %d, which names no mesh "
                               "of the file" % tri.material_index)
                    return None
                n = tuple(normals[li].vector)
                u, w = uv_layer.data[li].uv
            else:
                n, u, w = corner(vi, li)
            key = key_of(vi, n, u, w)
            at = seen[ri].get(key)
            if at is None:
                at = seen[ri][key] = len(out[ri][0])
                out[ri][0].append(rec(vi, n, u, w))
                edits["added"] += 1
            corners.append((ri, at))
        if corners[0][0] != corners[1][0] or corners[1][0] != corners[2][0]:
            why.append("a triangle spans two of the file's meshes")
            return None
        out[corners[0][0]][1].append(wound([c[1] for c in corners]))
    return [(runs[ri][0], out[ri][0], out[ri][1])
            for ri in range(len(runs))], kept, edits


PIN_GROUP = "vtmb_pinned"


def cloth_params(obj):
    """(preset, sigma, slack, scale, pin group) for a mesh marked as cloth, else None."""
    if not obj.get("vtmb_cloth"):
        return None
    preset = str(obj.get("vtmb_cloth_preset") or "") or None
    if preset is not None and preset not in cloth_mod.PRESETS:
        raise Refused("%s names cloth preset %r, which is not one of the 17"
                      % (obj.name, preset))
    return (preset, obj.get("vtmb_cloth_sigma"), obj.get("vtmb_cloth_slack"),
            obj.get("vtmb_cloth_scale"),
            str(obj.get("vtmb_cloth_pin_group") or PIN_GROUP))


def pin_first(obj, verts, faces, group, scale=1.0):
    """(verts, faces, npin) reordered so the pinned particles come first.

    cloth_mod.generate takes the pin set as a count and nothing else marks a pin, so the
    order is the only channel a mesh has into it.  split_mesh drops the Blender vertex
    index -- a seam duplicates a vertex -- so membership is carried by position, and two
    vertices sharing one position while disagreeing about the group are refused rather
    than resolved.
    """
    vg = obj.vertex_groups.get(group)
    if vg is None:
        raise Refused("%s is marked as cloth and carries no vertex group %r naming the "
                      "pinned particles" % (obj.name, group))
    pinned = set()
    for v in obj.data.vertices:
        if any(g.group == vg.index for g in v.groups):
            pinned.add((round(v.co.x / scale, 5), round(v.co.y / scale, 5),
                        round(v.co.z / scale, 5)))
    if not pinned:
        raise Refused("%s's vertex group %r is empty, so every particle would be free "
                      "and the object would fall away" % (obj.name, group))
    key = [tuple(round(c, 5) for c in v[0]) in pinned for v in verts]
    order = ([i for i in range(len(verts)) if key[i]]
             + [i for i in range(len(verts)) if not key[i]])
    npin = sum(key)
    if npin == len(verts):
        raise Refused("%s pins every one of its %d particles, so nothing would move"
                      % (obj.name, npin))
    at = {o: i for i, o in enumerate(order)}
    return ([verts[o] for o in order],
            [tuple(at[c] for c in f) for f in faces], npin)


def add_cloth(d, obj, verts, faces, npin, flip=False):
    """Generate this model's cloth object and hang it off the model's one mesh."""
    preset, sigma, slack, cscale, _g = cloth_params(obj)
    c = cloth_mod.generate([v[0] for v in verts], faces, npin,
                           pv=list(range(len(verts))), preset=preset,
                           sigma=sigma, slack=slack, scale=cscale)
    blob = cloth_mod.pack(c)
    mr = d.bodyparts[-1].kids[-1]
    mr.extra["cloth"] = cloth_mod.region(blob, len(verts), c.numparticles, flip)
    return c, blob


def add_meshes(d, mesh_objs, bone_index, scale=1.0):
    """One bodypart per object, holding one model whose meshes are its material slots.

    A bodypart is a variant selector -- the engine draws one of its models, chosen by
    `m_nBody` -- so a bodypart per material would declare that many bodygroups. 4381 of
    the 4464 shipped files carry exactly one bodypart; the material unit is the mesh.
    """
    slot_of = {}
    total_unskinned = total_kept = 0
    crowded, cloths = [], []
    for obj in mesh_objs:
        runs, unskinned, kept, _why, _edits = split_mesh(obj, bone_index, scale)
        total_unskinned += unskinned
        total_kept += kept
        over = export_mod.crowded_vertices(obj, bone_index)
        if over:
            crowded.append((obj.name, over))
        mats = obj.data.materials
        meshes = []
        for slot, verts, faces in runs:
            name = mats[slot].name if slot < len(mats) and mats[slot] else obj.name
            if name not in slot_of:
                slot_of[name] = build_mod.add_material(d, name)
            meshes.append((slot_of[name], verts, faces))
        cloth = cloth_params(obj)
        if cloth is not None:
            if len(meshes) != 1:
                raise Refused("%s is marked as cloth and uses %d materials -- a cloth "
                              "object spans one model's whole particle array, so the "
                              "mesh has to be one material" % (obj.name, len(meshes)))
            slot, cverts, cfaces = meshes[0]
            cverts, cfaces, npin = pin_first(obj, cverts, cfaces, cloth[4], scale)
            meshes = [(slot, cverts, cfaces)]
        if meshes:
            build_mod.add_model(d, meshes,
                                bodypart=obj.get("vtmb_bodypart") or obj.name,
                                model=obj.get("vtmb_model") or obj.name + ".smd")
            if cloth is not None:
                add_cloth(d, obj, cverts, cfaces, npin,
                          bool(obj.get("vtmb_cloth_flip")))
                cloths.append((obj.name, npin, len(cverts)))
    return list(d.faces), total_unskinned, total_kept, crowded, cloths


_ARM = ("upperarm", "forearm", "hand", "finger", "thumb")
_LEG = ("thigh", "calf", "foot", "toe")


def _hitgroup_of(name):
    """The hit group a bone falls in: 1 head, 2 chest, 3 stomach, 4/5 arms, 6/7 legs, 0
    generic. Not derivable from a file, so this is the corpus convention over 14333
    shipped boxes -- pelvis 0 on 372 of 372, bare Spine 3 on 512 of 514, Spine1 2 on 501
    of 508, neck and head 1, the limbs 4/5 and 6/7 on 98.9%. A `vtmb_hitgroup` on the
    pose bone wins over it."""
    n = " %s " % name.lower().replace("_", " ")
    if "pelvis" in n:
        return 0
    if "head" in n or "neck" in n or "jaw" in n:
        return 1
    right = " r " in n or "right" in n
    if any(k in n for k in _ARM):
        return 5 if right else 4
    if any(k in n for k in _LEG):
        return 7 if right else 6
    if "spine" in n:
        return 2 if n.rstrip()[-1].isdigit() else 3
    if "clavicle" in n:
        return 2
    return 0


def _hitgroup_prop(arm_obj, bname):
    """The explicit hit group stamped on a bone, or None to let the name decide.

    `Bone` and `PoseBone` are separate ID-property containers, so which one is read
    decides whether a stamp reaches the file at all. Every other `vtmb_*` bone key lives
    on the pose bone -- the flags, the scales, the rotation limit, the rest matrix and the
    seven spring fields -- and that is where `blender_templates.apply_skeleton` writes
    this one too.
    """
    pb = arm_obj.pose.bones.get(bname)
    return None if pb is None else pb.get("vtmb_hitgroup")


def fit_hitboxes(d, mesh_objs, bone_index, arm_obj, scale=1.0, floor=0.05,
                 name="default"):
    """One box per bone that owns geometry, in that bone's own space. Returns the count.

    A vertex counts for every bone binding it at `floor` or more, so a joint's box covers
    the flesh on both sides of it. At 0.50 only 3 of `security_guard`'s 20 shipped boxes
    come out enclosed and at 0.05 it is 18, the two others being a degenerate pair Troika
    shipped on `Bip01 Spine2` (item 41). The bindings are the ones being written, not the
    raw vertex groups, so the box cannot enclose flesh the file does not give that bone.
    """
    inv = [[list(f[0:4]), list(f[4:8]), list(f[8:12])]
           for f in (struct.unpack_from("<12f", r.raw, 0x58) for r in d.bones)]
    lo, hi = {}, {}
    for obj in mesh_objs:
        for v in obj.data.vertices:
            p = (v.co[0] / scale, v.co[1] / scale, v.co[2] / scale)
            for bi, w in _skin(v, obj.vertex_groups, bone_index):
                if w < floor:
                    continue
                m = inv[bi]
                q = [m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]
                     for r in range(3)]
                if bi not in lo:
                    lo[bi], hi[bi] = list(q), list(q)
                    continue
                for c in range(3):
                    lo[bi][c] = min(lo[bi][c], q[c])
                    hi[bi][c] = max(hi[bi][c], q[c])
    boxes = []
    for bi in sorted(lo):
        bname = d.bones[bi].name or ""
        g = _hitgroup_prop(arm_obj, bname)
        boxes.append((bi, int(_hitgroup_of(bname) if g is None else g),
                      tuple(lo[bi]), tuple(hi[bi])))
    if boxes:
        build_mod.add_hitbox(d, boxes, name)
    return len(boxes)


def add_attachments(d, arm_obj, bone_index, scale=1.0):
    """Every attachment empty in the scene, on the bone it is parented to. Returns the count.

    Silently skipping one whose bone the authored skeleton does not carry would lose a mount
    point the user placed, so an unresolvable bone is refused instead.
    """
    n = 0
    for obj in export_mod.accessory_objects(arm_obj)[0]:
        want = obj.parent_bone if obj.parent_type == "BONE" else ""
        if want not in bone_index:
            raise Refused("attachment %r is on bone %r, which is not being written"
                          % (obj.name, want or "<none>"))
        rows = [[obj.matrix_basis[r][c] / (scale if c == 3 else 1.0) for c in range(4)]
                for r in range(3)]
        build_mod.add_attachment(d, str(obj.get("vtmb_attachment") or obj.name),
                                 bone_index[want], rows,
                                 int(obj.get("vtmb_attachment_type") or 0))
        n += 1
    return n


def sample_action(context, arm_obj, action, d, scale=1.0, use_range=False):
    """`poses[frame][bone]` of local (pos, quat), in the bone order already authored.

    Goes through `blender_export.read_poses` rather than composing parent-relative
    matrices here: a BONE_ROTATION_FROM_ROOT bone's local rotation is its *world* one, so
    the naive chain is wrong for it, and that inverse is already written and checked.
    """
    ad = arm_obj.animation_data or arm_obj.animation_data_create()
    was, at = ad.action, context.scene.frame_current
    ad.action = action
    if use_range:
        first, last = context.scene.frame_start, context.scene.frame_end
    else:
        lo, hi = action.frame_range
        first, last = int(round(lo)), int(round(hi))
    try:
        return export_mod.read_poses(context, arm_obj, build_mod._skeleton(d),
                                     range(first, last + 1), scale)
    finally:
        ad.action = was
        context.scene.frame_set(at)


def _ordered(actions):
    """The file's animation order, not Blender's.

    `bpy.data.actions` is sorted by name, while every sequence's blend table cites an
    animation by index, so authoring in collection order renumbers all of them.
    """
    idx = [a.get("vtmb_anim_index") for a in actions]
    if any(i is None for i in idx):
        return list(actions)
    return [a for _i, a in sorted(zip(idx, actions), key=lambda p: p[0])]


def add_actions(context, arm_obj, d, actions, scale=1.0, use_range=False,
                activity="ACT_IDLE", root_motion="extract"):
    """`root_motion` is what becomes of the root bone's path. "extract" takes its net
    ground displacement into one `mstudiomovement_t` and leaves the rest -- the sway and
    rise -- in the keys, which is how a walk cycle is stored: the engine carries the
    entity, the skeleton stays put and still bobs. "none" leaves the whole path in the
    poses, which is right only for an animation that really does translate in model
    space. "in_place" takes the same net displacement out and writes no block, so nothing
    travels on either side. "per_action" reads each action's own stamp instead, where a
    stored "keep" becomes "extract" -- there is no donor file here to keep blocks from."""
    moved, unfitted, unkeepable = 0, [], []
    for act in _ordered(actions):
        poses = sample_action(context, arm_obj, act, d, scale, use_range)
        if not poses:
            raise Refused("action %r has no frames" % act.name)
        mode = export_mod.root_motion_mode(act, root_motion, donor=False,
                                           default="extract")
        movements = ()
        if mode in ("extract", "in_place"):
            mvs, poses = write_mod.extract_root_motion(build_mod._skeleton(d), poses)
            if mode == "extract":
                movements = mvs
                moved += bool(mvs)
                if export_mod.lost_travel(act, mvs):
                    unfitted.append(act.name)
        elif mode == "keep" and export_mod.lost_travel(act, movements):
            # Only a forced "keep" reaches this: a stamped one became "extract" above.
            unkeepable.append(act.name)
        fps = float(act.get("vtmb_fps", context.scene.render.fps))
        flags = int(act.get("vtmb_flags", 0))
        a = build_mod.add_animation(d, act.name, poses, fps, flags, movements)
        # `get(key, default)` and not `or`: an imported action carries "" where its
        # sequence claimed no activity, and that is a value, not a gap for the
        # dialog to fill.
        build_mod.add_sequence(d, act.name, a, act.get("vtmb_activity", activity),
                               int(act.get("vtmb_seq_flags", 0)))
    return len(actions), moved, unfitted, unkeepable


def scene_cdtextures(arm_obj):
    """The material directories an import stamped.

    Empty is not the same as `models/`: 4442 of the 4445 shipped models are not `models/`,
    so a caller with nothing here picks its own default rather than being handed one.
    """
    return paths_mod.engine_paths(arm_obj.get("vtmb_cdtexture"))


def scene_includes(arm_obj):
    """The chain an import stamped, or a template wrote, as a list of engine paths.

    An import stores it verbatim; a bone-set template writes the same key, which is where
    a generated skeleton gets its animations from. Duplicates are dropped -- the engine
    walks the array and a repeat costs a whole second bone map.
    """
    return paths_mod.engine_paths(arm_obj.get("vtmb_includes"))


def build(context, arm_obj, mesh_objs, actions, name, scale=1.0, surfaceprop="flesh",
          cdtexture="models/", hull=None, use_range=False, activity="ACT_IDLE",
          hitboxes=False, root_motion="extract", includes=()):
    """(Desc, faces) for the scene. `hull` is (min, max) and stays the caller's:
    @180/@192 are the movement hull the .qc's $hbox sets, not the mesh's bounds."""
    d = build_mod.new(name, surfaceprop)
    if hull is not None:
        lo, hi = hull
        _set_hull(d, lo, hi)
    for p in includes:
        build_mod.add_include(d, p)
    build_mod.set_cdtextures(d, paths_mod.cdtexture_list(cdtexture))
    bone_index = add_bones(d, arm_obj, scale, surfaceprop)
    if not bone_index:
        raise Refused("the armature has no bones")
    faces, unskinned, kept, crowded, cloths = add_meshes(d, mesh_objs, bone_index, scale)
    if hitboxes:
        fit_hitboxes(d, mesh_objs, bone_index, arm_obj, scale)
    add_attachments(d, arm_obj, bone_index, scale)
    _n, moved, unfitted, unkeepable = add_actions(context, arm_obj, d, actions, scale,
                                                  use_range, activity, root_motion)
    return d, faces, unskinned, kept, moved, crowded, unfitted, unkeepable, cloths


def _set_hull(d, lo, hi):
    struct.pack_into("<3f", d.hdr, 180, *lo)
    struct.pack_into("<3f", d.hdr, 192, *hi)
    # Only reached by a model with no sequence at all: mdl_build.stamp_sequence_boxes
    # rewrites @168 as the centre of sequence 0's box, which is studiomdl's own rule.
    struct.pack_into("<3f", d.hdr, 168, *[(a + b) * 0.5 for a, b in zip(lo, hi)])


def fit_hull(mesh_objs, scale=1.0):
    """(min, max) over every vertex handed in, in file units, or None when there are none.

    A starting point and not a truth: @180/@192 is the movement hull the .qc's `$hbox`
    sets, which is why `build` takes it from the caller rather than deriving it. Without
    one `mdl_build.new` leaves (-16,-16,0)..(16,16,72), so a bat claims a humanoid volume.
    Mesh-local like `split_mesh`, so an object transform is ignored by both alike.
    """
    lo = hi = None
    for obj in mesh_objs:
        for v in obj.data.vertices:
            p = [v.co[c] / scale for c in range(3)]
            if lo is None:
                lo, hi = list(p), list(p)
                continue
            for c in range(3):
                lo[c] = min(lo[c], p[c])
                hi[c] = max(hi[c], p[c])
    return None if lo is None else (tuple(lo), tuple(hi))


def embedded_name(path):
    """What goes in `studiohdr_t.name[128]`: the path BELOW `models/`, forward-slashed.

    Measured over the 4445-model corpus: the field is the file's own path relative to the
    models directory on 4383 of them, and **no `models/` prefix** -- 0 of 4445 carry one.
    Separators are forward slashes on every file that has one (4443; the two without are
    top-level) and no name anywhere contains a backslash. Case comes off the source .qc
    and is not the on-disk case on 1281 of the 4383, so nothing here folds it.
    """
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    slash = stem.replace("\\", "/")
    cut = slash.lower().rfind("/models/")
    base = slash[cut + len("/models/"):] if cut >= 0 else os.path.basename(stem)
    return base + ".mdl"


def write(d, faces, path, checksum):
    """Emit both files. Returns (mdl bytes, vtx bytes, vtx stats)."""
    data = build_mod.emit(d, checksum)
    m = mdl_mod.Mdl("<scene>", data=data)
    vtx, st = vtxr_mod.scratch(m, faces, checksum)
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    with open(stem + ".mdl", "wb") as f:
        f.write(data)
    with open(stem + ".dx80.vtx", "wb") as f:
        f.write(vtx)
    return data, vtx, st


def export_scene(context, arm_obj, mesh_objs, actions, path, checksum, **kw):
    d, faces, unskinned, kept, moved, crowded, unfitted, unkeepable, cloths = build(
        context, arm_obj, mesh_objs, actions, embedded_name(path), **kw)
    data, vtx, st = write(d, faces, path, checksum)
    return {"bytes": len(data), "vtx_bytes": len(vtx), "bones": len(d.bones),
            "with_root_motion": moved, "unfitted": unfitted,
            "unkeepable": unkeepable,
            "bodyparts": len(d.bodyparts), "materials": len(d.textures),
            "anims": len(d.anims), "seqs": len(d.seqs),
            "includes": len(d.includes),
            "hitboxes": sum(len(r.kids) for r in d.hitboxsets),
            "faces": st["tris_out"], "verts": st["verts_out"],
            "model_verts": sum(len(x.extra.get("tangents") or b"") // 16
                               for bp in d.bodyparts for x in bp.kids),
            "kept": kept, "crowded": crowded,
            "unskinned": unskinned, "dropped": dict(d.dropped),
            "cloths": cloths}
