"""Author a .mdl and its .dx80.vtx from a Blender scene, with no donor file.

`blender_export` rewrites a file that already exists: it opens with `Mdl(source)` and
carries every record the scene does not supply. This does not open anything. The armature
becomes `mstudiobone_t[]`, the mesh objects become bodyparts, the actions become animations.
The sequences are the imported list where the armature carries one, `vtmb_sequences` with its
pose parameters, and one per action for every action that list does not cite. Attachments,
hitboxes, the include chain, cloth, spring bone chains, shape keys as flexes with the flex
controllers and rules, eyeballs and the mouth are written from the scene too.
Everything else is emitted with a count of zero, and `SCRATCH_DROPS` in `__init__` is the list
of it the export dialog prints.

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
NO_EDITS = (("uvs", 0), ("normals", 0), ("blind", 0), ("added", 0), ("deleted", 0))
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


def original_runs(obj, me, why=None, carry=None):
    """The file's mesh partition as [(material slot, [Blender vertex, ...]), ...].

    A vertex the edit deleted is left out of its run, so the survivors stay in file order
    and their new local index is their position in `members`. `carry` collects
    `(the mesh's old vertex count, [the old local index of each survivor])` per run, which
    is what lets a caller resolve the new numbering back onto the old -- a flex payload is
    keyed by mesh-local index and cannot be carried positionally across a delete.

    Only an original no vertex of the scene claims at all counts as deleted. One a vertex
    still claims that `identity_map` could not resolve is unrecoverable instead: dropping
    it would take a flex delta off geometry the scene still has, since the claimants are
    then written as added vertices.

    None where the partition itself is unrecoverable: that; a slot naming two runs, since a
    vertex the edit added is placed by slot and there would be no saying which of them it
    belongs to; or a mesh every vertex of which has gone, `numvertices` 0 being a form 0 of
    9741 shipped meshes has. Each of those appends to `why`, which the caller reports and
    cannot work out for itself.
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
    claimed = set(_point_ints(me, "vtmb_orig") or ())
    runs, at, slots = [], 0, set()
    for slot, n in spec:
        members, kept_at = [], []
        for k in range(int(n)):
            vi = ident.get(at + k)
            if vi is not None:
                members.append(vi)
                kept_at.append(k)
            elif at + k in claimed:
                why.append("file vertex %d is claimed by a vertex of the scene that "
                           "cannot be resolved to it -- several claim the tag and none "
                           "stands where the file put that original, so writing either "
                           "would name the wrong vertex" % (at + k))
                return None
        if int(n) and not members:
            why.append("every vertex of one of the file's meshes is claimed by no vertex "
                       "of the scene, and a mesh with numvertices 0 is a form 0 of 9741 "
                       "shipped meshes has")
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
        if carry is not None:
            carry.append((int(n), kept_at))
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


def split_mesh(obj, bone_index, scale=1.0, fields=("uvs", "normals"), carry=None):
    """([(material slot, verts, faces), ...], unskinned, kept, why, edits, origins).

    The format stores one normal, one UV and one skin per vertex, so a seam or a hard edge
    is spelled by duplicating the vertex. `kept` counts the originals written back into
    their own mesh in file order; it is 0 when the partition had to be rebuilt from the
    material slots instead, which puts every vertex in triangle-visit order. `why` names
    the one condition that forced that, and is None where it did not happen -- ten
    separate ones reach it and the caller refuses on the wrong one otherwise. `edits` is
    what the scene moved on the originals, in the keys of NO_EDITS, and is all zero on the
    rebuilt partition, which has no original to compare against.

    `fields` is the export's own, so an original keeps what was imported for any field the
    scene was not asked to supply. A vertex that has no original takes Blender's whatever
    is asked, there being nothing else to take.

    `origins` runs parallel to the runs and names the Blender vertex each output vertex
    came from, several of them naming one where a seam split it apart. Anything the scene
    holds per Blender vertex needs it to reach a written vertex: two vertices a seam
    duplicated share a position, so a position cannot tell them apart, and 18 998 of the
    25 564 named cloth particles are named by vertices that disagree about the flip bit.

    `carry` is an out-parameter and collects one list per run, the old local index of each
    output vertex and None for one the edit added. `mdl_build.replace_model` resolves the
    donor's per-vertex fields through it instead of by position, which is what lets a
    delete keep the file's own numbering for every vertex that survived. It is None per run
    on the rebuilt partition, where no output vertex has an original.
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
        spans = []
        runs = original_runs(obj, me, why, spans)
        if runs is not None:
            out = _split_preserved(me, runs, stash, uvs, normals, uv_layer,
                                   rec, key_of, why, fields)
            if out is not None:
                out[2]["deleted"] = sum(n - len(k) for n, k in spans)
                if carry is not None:
                    carry.extend(
                        kept_at + [None] * (len(out[0][ri][1]) - len(kept_at))
                        for ri, (_n, kept_at) in enumerate(spans))
                return out[0], unskinned, out[1], None, out[2], out[3]

    per_slot = {}
    for tri in me.loop_triangles:
        verts, faces, seen, orig = per_slot.setdefault(tri.material_index,
                                                       ([], [], {}, []))
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
                orig.append(vi)
            corners.append(at)
        faces.append(wound(corners))
    if carry is not None:
        carry.extend([None] * len(per_slot))
    return ([(k, per_slot[k][0], per_slot[k][1]) for k in sorted(per_slot)],
            unskinned, 0, why[0] if why else "the file's partition was not recoverable",
            dict(NO_EDITS), [per_slot[k][3] for k in sorted(per_slot)])


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
        edit moves the vertex instead of duplicating it. A corner on a face with no area
        reads no normal at all and resolves too -- BUGS 158. A UV is float32 on both sides
        and needs none of that.
        """
        base = decided[vi]
        n = base if not take_normal or li in dead else tuple(normals[li].vector)
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
    origins = []
    for ri, (slot, members) in enumerate(runs):
        run_of_slot[slot] = ri
        verts, keys, orig = [], {}, []
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
            orig.append(vi)
            kept += 1
        out.append((verts, []))
        origins.append(orig)
        seen.append(keys)

    for tri in me.loop_triangles:
        # A triangle whose originals all sit in one run but whose slot names another was
        # moved there: it goes to that slot's run and its corners are copied in.
        target = run_of_slot.get(tri.material_index)
        homes = {where[me.loops[li].vertex_index] for li in tri.loops
                 if me.loops[li].vertex_index in where}
        if len(homes) == 1 and target is None:
            why.append("a triangle of the file's was moved onto material slot %d, which names "
                       "no mesh of the file" % tri.material_index)
            return None
        moved = len(homes) == 1 and target not in homes
        corners = []
        for li in tri.loops:
            vi = me.loops[li].vertex_index
            ri = where.get(vi)
            if ri is None:
                ri = target
                if ri is None:
                    why.append("a triangle sits on material slot %d, which names no mesh "
                               "of the file" % tri.material_index)
                    return None
                n = tuple(normals[li].vector)
                u, w = uv_layer.data[li].uv
            else:
                n, u, w = corner(vi, li)
                if moved:
                    ri = target
            key = key_of(vi, n, u, w)
            at = seen[ri].get(key)
            if at is None:
                at = seen[ri][key] = len(out[ri][0])
                out[ri][0].append(rec(vi, n, u, w))
                origins[ri].append(vi)
                edits["added"] += 1
            corners.append((ri, at))
        if corners[0][0] != corners[1][0] or corners[1][0] != corners[2][0]:
            why.append("a triangle spans two of the file's meshes")
            return None
        out[corners[0][0]][1].append(wound([c[1] for c in corners]))
    return [(runs[ri][0], out[ri][0], out[ri][1])
            for ri in range(len(runs))], kept, edits, origins


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
    """(verts, faces, npin, order) reordered so the pinned particles come first.

    cloth_mod.generate takes the pin set as a count and nothing else marks a pin, so the
    order is the only channel a mesh has into it.  Membership is carried by position,
    which two vertices a seam duplicated share; `order` is the permutation itself, so
    anything the caller holds per output vertex follows the reorder rather than being
    re-derived from a position that cannot tell those two apart.
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
            [tuple(at[c] for c in f) for f in faces], npin, order)


def flip_flags(obj, origins):
    """One flip bit per output vertex, from the mesh attribute or from the object key.

    Bit 15 of `mstudiomesh_t +0x34` negates that vertex's normal and 63 of the 84 shipped
    meshes set it on some vertices and not others, so the import stamps it per vertex.
    The object key is what an authored sheet and a `.blend` saved before the attribute
    shipped have, and it covers every vertex.
    """
    per_vertex = _point_ints(obj.data, "vtmb_cloth_flip")
    if per_vertex is None:
        return bool(obj.get("vtmb_cloth_flip"))
    return [bool(per_vertex[vi]) for vi in origins]


def sigma_edges(obj, origins, faces):
    """{ascending output particle pair: the sigma its Blender edge holds}, or None.

    28 of the 59 shipped row-0 objects give each group-0 spring its own sigma, and group 0
    is the face list's edge set, so a float per mesh edge is what an authored one carries.
    `origins` names the Blender vertex each output vertex came from, so a pair a seam split
    reads the edge it was split from.
    """
    me = obj.data
    att = me.attributes.get("vtmb_cloth_sigma")
    if att is None or att.domain != "EDGE" or len(att.data) != len(me.edges):
        return None
    buf = [0.0] * len(me.edges)
    att.data.foreach_get("value", buf)
    at = {cloth_mod.pair(*tuple(e.vertices)): e.index for e in me.edges}
    out = {}
    for f in faces:
        for i in range(3):
            a, b = cloth_mod.pair(f[i], f[(i + 1) % 3])
            if a >= len(origins) or b >= len(origins):
                continue
            j = at.get(cloth_mod.pair(origins[a], origins[b]))
            if j is not None and buf[j] != cloth_mod.SIGMA_UNSET:
                out[(a, b)] = buf[j]
    return out


def add_cloth(d, obj, verts, faces, npin, flip=False, origins=None):
    """Generate this model's cloth object and hang it off the model's one mesh."""
    preset, sigma, slack, cscale, _g = cloth_params(obj)
    c = cloth_mod.generate([v[0] for v in verts], faces, npin,
                           pv=list(range(len(verts))), preset=preset,
                           sigma=sigma, slack=slack, scale=cscale,
                           sigma0=None if origins is None
                           else sigma_edges(obj, origins, faces))
    blob = cloth_mod.pack(c)
    mr = d.bodyparts[-1].kids[-1]
    mr.extra["cloth"] = cloth_mod.region(
        blob, c.numparticles,
        cloth_mod.one_to_one(len(verts), c.numparticles, flip))
    return c, blob


def meshless_model(d, allowed):
    """Where the scene wrote no bodypart, one holding one model of no mesh, as `null.mdl`
    and 11 more shipped files carry, or Refused unless `allowed`. True when it wrote one.

    0 of the 4445 shipped models carry `numbodyparts` 0, and zero bodyparts reach
    `vtx_write.to_bytes` and die on `at["bodyparts"]`, which is BUGS 135.
    """
    if d.bodyparts:
        return False
    if not allowed:
        raise Refused("the armature owns no mesh. Parent a mesh to it or bind one with "
                      "an Armature modifier naming it, or tick Skeleton without mesh for "
                      "an animation-only file")
    build_mod.add_model(d, [], model="null")
    return True


def add_mesh_flexes(d, obj, bi, origins, scale=1.0):
    """One mstudioflex_t per shape key per written mesh it moves. (keys, records).

    Every key is encoded, none carried: the written numbering is Blender's and not the
    file's. A vertex a seam split takes the key's delta on every copy. An object imported
    before `vtmb_flex_descs` raises Refused, since some of its keys sum two steps.
    """
    me = obj.data
    keys = me.shape_keys
    if keys is None or len(keys.key_blocks) < 2:
        return 0, 0
    if export_mod.prestamp_flexes(obj):
        raise Refused(export_mod._summed_keys(obj.name))
    copies = {}
    for k, run in enumerate(origins):
        for j, vi in enumerate(run):
            copies.setdefault(vi, []).append((k, j))
    me.calc_loop_triangles()
    blocks = list(keys.key_blocks)
    base = [tuple(v.co) for v in blocks[0].data]
    base_n = export_mod._vertex_normals(me, base)
    names = list(obj.get("vtmb_flex_names") or [])
    descs = obj.get("vtmb_flex_descs")
    targets = obj.get("vtmb_flex_targets") or []
    nkeys = nrec = 0
    for kb in blocks[1:]:
        co = [tuple(v.co) for v in kb.data]
        moved = export_mod.moved_vertices(co, base)
        per_mesh = {}
        for vi, delta, ndelta in export_mod.key_deltas(me, co, base, base_n, moved, scale):
            for k, j in copies.get(vi, ()):
                per_mesh.setdefault(k, []).append((j, delta, ndelta))
        if not per_mesh:
            continue
        fd, t = export_mod.key_flex(kb.name, names, descs, targets)
        nkeys += 1
        for k in sorted(per_mesh):
            recs = sorted(per_mesh[k])
            build_mod.add_flex(d, bi, 0, k, fd, recs, t)
            nrec += len(recs)
    return nkeys, nrec


def eye_slots(obj, bi, slots):
    """[(bodypart, mesh, eyeball index, object, material)] per slot the import stamped as an
    eye and the export wrote. `slots` is the material slot of each written mesh, in order."""
    moves = export_mod.slot_moves(obj)[0]
    mats = obj.data.materials
    out = []
    for slot, param in obj.get("vtmb_eye_slots") or ():
        here = int(slot) if moves is None else moves.get(int(slot))
        if here in slots:
            mat = mats[here].name if here < len(mats) and mats[here] else obj.name
            out.append((bi, slots.index(here), int(param), obj.name, mat))
    return out


def add_meshes(d, mesh_objs, bone_index, scale=1.0, owners=None, face=None):
    """One bodypart per object, holding one model whose meshes are its material slots.

    A bodypart is a variant selector -- the engine draws one of its models, chosen by
    `m_nBody` -- so a bodypart per material would declare that many bodygroups. 4381 of
    the 4464 shipped files carry exactly one bodypart; the material unit is the mesh.

    `face`, where given, collects `flexes` and `flex_records` from the shape keys and
    `eye_meshes` from `eye_slots`, for `scene_face` to point at its records.
    """
    slot_of = {}
    total_unskinned = total_kept = 0
    crowded, cloths, stray, seams = [], [], [], []
    for obj in mesh_objs:
        runs, unskinned, kept, _why, _edits, origins = split_mesh(obj, bone_index,
                                                                  scale)
        total_unskinned += unskinned
        total_kept += kept
        over = export_mod.crowded_vertices(obj, bone_index)
        if over:
            crowded.append((obj.name, over))
        odd = export_mod.stray_groups(obj, bone_index)
        if odd:
            stray.append((obj.name, odd))
        mats = obj.data.materials
        meshes = []
        for slot, verts, faces in runs:
            name = mats[slot].name if slot < len(mats) and mats[slot] else obj.name
            if name not in slot_of:
                slot_of[name] = build_mod.add_material(d, name)
            meshes.append((slot_of[name], verts, faces))
        cloth = cloth_params(obj)
        # A cloth mesh keeps one vertex per particle, so it is never split.
        if cloth is None:
            copies = 0
            for k, (mi, verts, faces) in enumerate(meshes):
                verts, faces, src = build_mod.split_handedness(verts, faces)
                meshes[k] = (mi, verts, faces)
                origins[k].extend(origins[k][i] for i in src)
                copies += len(src)
            if copies:
                seams.append((obj.name, copies))
        if cloth is not None:
            if len(meshes) != 1:
                raise Refused("%s is marked as cloth and uses %d materials -- a cloth "
                              "object spans one model's whole particle array, so the "
                              "mesh has to be one material" % (obj.name, len(meshes)))
            slot, cverts, cfaces = meshes[0]
            cverts, cfaces, npin, order = pin_first(obj, cverts, cfaces, cloth[4], scale)
            corig = [origins[0][o] for o in order]
            cflip = flip_flags(obj, corig)
            meshes = [(slot, cverts, cfaces)]
        if meshes:
            bi = build_mod.add_model(d, meshes,
                                     bodypart=obj.get("vtmb_bodypart") or obj.name,
                                     model=obj.get("vtmb_model") or obj.name + ".smd")
            if owners is not None:
                owners.append((bi, obj))
            if face is not None:
                nk, nr = add_mesh_flexes(d, obj, bi, origins if cloth is None else [corig],
                                         scale)
                face["flexes"] += nk
                face["flex_records"] += nr
                face["eye_meshes"] += eye_slots(obj, bi, [s for s, _v, _f in runs])
            if cloth is not None:
                add_cloth(d, obj, cverts, cfaces, npin, cflip, corig)
                cloths.append((obj.name, npin, len(cverts)))
    return list(d.faces), total_unskinned, total_kept, crowded, cloths, stray, seams


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


def scene_hitboxes(d, arm_obj, bone_index, scale=1.0):
    """Every hitbox set and box the scene carries, verbatim. Returns (boxes, sets).

    The scene's boxes are the model's, so nothing here refits and nothing merges with a
    refit: `build` runs `fit_hitboxes` only where this returns no set at all.  A set empty
    with no box under it is still a set -- 358 of the 4445 shipped models carry exactly
    that -- so the ordinals come from `set(boxes) | set(names)` and not from the boxes.
    Set names are the scene's, `default` being what 4444 of 4445 shipped models call theirs
    and `mingxiao.mdl` the one with seven.

    An unresolvable bone is refused by name, the rule `add_attachments` already follows,
    because silently dropping a box the user placed is the narrowing this replaces.
    """
    _attach, boxes, names = export_mod.accessory_objects(arm_obj)
    ords = sorted(set(boxes) | set(names))
    nbox = 0
    for k in ords:
        recs = []
        for obj in boxes.get(k, ()):
            want = obj.parent_bone if obj.parent_type == "BONE" else ""
            if want not in bone_index:
                raise Refused("hitbox %r is on bone %r, which is not being written"
                              % (obj.name, want or "<none>"))
            lo, hi = export_mod.hitbox_extent(obj, scale)
            recs.append((bone_index[want], int(obj.get("vtmb_hitbox_group") or 0), lo, hi))
        build_mod.add_hitbox(d, recs, names.get(k, "default"))
        nbox += len(recs)
    return nbox, len(ords)


def add_cloth_volumes(d, arm_obj, owners, bone_index, scale=1.0):
    """Every cloth capsule and sphere empty, on the model its object came from. Returns
    (records, records put on the first model because no imported object carries theirs,
    capsules with their ends on two bones as `apply_cloth_volumes` names them).

    Where one lands decides nothing the engine does: the gather at 0x2c014a00 collects the
    selected submodel of every bodypart into one list, and each bodypart here has one.
    """
    caps, sphs = export_mod.cloth_volume_objects(arm_obj)
    at = {}
    for bi, obj in owners:
        if obj.get("vtmb_index") is not None:
            at.setdefault(int(obj["vtmb_index"]), bi)

    def bone(obj):
        want = obj.parent_bone if obj.parent_type == "BONE" else ""
        if want not in bone_index:
            raise Refused("cloth volume %r is on bone %r, which is not being written"
                          % (obj.name, want or "<none>"))
        return bone_index[want]

    blobs, n, moved, two = {}, 0, 0, []
    for key in sorted(caps):
        ends = caps[key]
        if set(ends) != {0, 1}:
            (end, obj), = ends.items()
            raise Refused("%r is end %d of cloth capsule %d and the other end is gone; "
                          "delete it too, or add a capsule with Add Cloth Capsule"
                          % (obj.name, end, key[1]))
        ends = (ends[0], ends[1])
        if bone(ends[0]) != bone(ends[1]):
            two.append((key[0], key[1], ends[0].parent_bone, ends[1].parent_bone))
        rec = struct.pack("<2i7f", bone(ends[0]), bone(ends[1]),
                          export_mod._capsule_radius(ends, key[0], key[1], scale),
                          *(export_mod.cloth_volume_centre(ends[0], arm_obj, scale)
                            + export_mod.cloth_volume_centre(ends[1], arm_obj, scale)))
        bi = at.get(key[0], 0)
        moved += bool(at) and key[0] not in at
        blobs.setdefault((bi, "clothcollide"), []).append(rec)
        n += 1
    for key in sorted(sphs):
        obj = sphs[key]
        rec = struct.pack("<if3f", bone(obj), export_mod.cloth_volume_radius(obj, scale),
                          *export_mod.cloth_volume_centre(obj, arm_obj, scale))
        bi = at.get(key[0], 0)
        moved += bool(at) and key[0] not in at
        blobs.setdefault((bi, "clothsphere"), []).append(rec)
        n += 1
    for (bi, k), recs in blobs.items():
        d.bodyparts[bi].kids[0].extra[k] = b"".join(recs)
    return n, moved, two


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
        local = export_mod.bone_space(obj)
        rows = [[local[r][c] / (scale if c == 3 else 1.0) for c in range(4)]
                for r in range(3)]
        build_mod.add_attachment(d, str(obj.get("vtmb_attachment") or obj.name),
                                 bone_index[want], rows,
                                 int(obj.get("vtmb_attachment_type") or 0))
        n += 1
    return n


def _model_name(mr):
    return bytes(mr.raw[:128]).split(b"\0")[0].decode("latin1")


def scene_face(d, arm_obj, eye_meshes, bone_index, scale=1.0):
    """The eyeball records and the mouth, written from the scene onto the models written.

    Answers `apply_face`'s keys, its `eye_empties` naming an empty no record took -- a
    copy sharing an index, or one whose model is not written -- plus `eyeballs`, `mouth`,
    and `unmeshed`, the (model, record) no mesh is the eye of, which walkie_talki1 and
    sabbat_hand ship. Records go in index order, the first empty by name taking an index;
    a record's written position is what an eye mesh's `materialparam` names.
    """
    out = {"changed": 0, "materials": [], "radius_keys": [], "lid_targets": [],
           "eye_empties": [], "eye_records": [], "lid_past_radius": [],
           "eyeballs": 0, "unmeshed": [], "mouth": 0}
    models = {}
    for bi, bp in enumerate(d.bodyparts):
        models.setdefault(_model_name(bp.kids[0]), bi)
    per = {}
    for model, objs in export_mod.eyeball_objects(arm_obj).items():
        bi = models.get(model) if model else 0
        if bi is None:
            out["eye_empties"] += [(o.name, model, int(o["vtmb_eyeball"])) for o in objs]
            continue
        per.setdefault(bi, []).extend((int(o["vtmb_eyeball"]), not model, o.name, o)
                                      for o in objs)
    at = {}
    for bi in sorted(per):
        mr = d.bodyparts[bi].kids[0]
        pos = at[bi] = {}
        for i, _added, _name, obj in sorted(per[bi], key=lambda x: x[:3]):
            if i in pos:
                out["eye_empties"].append((obj.name, _model_name(mr), i))
                continue
            want = obj.parent_bone if obj.parent_type == "BONE" else ""
            if want not in bone_index:
                raise Refused("eyeball %r is on bone %r, which is not being written -- "
                              "parent it to a bone of the armature"
                              % (obj.name, want or "<none>"))
            rec = build_mod.Rec(bytearray(140))
            struct.pack_into("<i", rec.raw, 0x04, bone_index[want])
            export_mod._apply_eyeball(d, rec, obj, scale, out)
            pos[i] = len(mr.extra["eyes"])
            mr.extra["eyes"].append(rec)
            out["eyeballs"] += 1
    export_mod._lid_targets_refused(out["lid_past_radius"])
    eyed = set()
    for bi, k, param, name, mat in eye_meshes:
        j = at.get(bi, {}).get(param)
        if j is None:
            raise Refused("%s: material %r is the eye of eyeball %d, and no eyeball empty "
                          "on its model carries vtmb_eyeball %d. Restore the empty, or "
                          "delete vtmb_eye_slots from the object's custom properties to "
                          "write the mesh as an ordinary one" % (name, mat, param, param))
        struct.pack_into("<2i", d.bodyparts[bi].kids[0].kids[k].raw, 0x18, 1, j)
        eyed.add((bi, j))
    out["unmeshed"] = [(_model_name(d.bodyparts[bi].kids[0]), j) for bi in sorted(at)
                       for j in range(len(at[bi])) if (bi, j) not in eyed]
    out["mouth"] = scene_mouth(d, arm_obj, bone_index)
    return out


def eye_mesh_materials(d):
    """The texture name of each mesh drawn as an eyeball, once each."""
    out = []
    for bp in d.bodyparts:
        for mr in bp.kids:
            for mesh in mr.kids:
                if struct.unpack_from("<i", mesh.raw, 0x18)[0]:
                    name = d.textures[struct.unpack_from("<i", mesh.raw, 0)[0]].name
                    if name not in out:
                        out.append(name)
    return out


def eye_materials(d, appended):
    """The texture names an eyeball appended, then each eye mesh's own, once each."""
    return list(dict.fromkeys(list(appended) + eye_mesh_materials(d)))


def scene_mouth(d, arm_obj, bone_index):
    """The one mstudiomouth_t off `vtmb_mouth`, or nothing where the armature has none.

    The bone is a Blender name where the panel's picker set it and the file's where the
    import did, so both are tried. With no file to keep a value from, a field the stash
    lacks raises Refused rather than being guessed.
    """
    mouth = arm_obj.get("vtmb_mouth")
    if not mouth:
        return 0
    bone = str(mouth.get("bone") or "")
    flex = str(mouth.get("flex") or "")
    fwd = mouth.get("forward")
    lack = [k for k, v in (("bone", bone), ("forward", fwd), ("flex", flex))
            if v is None or v == ""]
    if lack:
        raise Refused("the armature's mouth carries no %s, and with no donor there is no "
                      "value to keep -- set it in the armature's VTMB eyeballs and mouth "
                      "panel" % " and no ".join(lack))
    bi = bone_index.get(bone)
    if bi is None:
        bi = stash_bones(arm_obj, bone_index).get(bone)
    if bi is None:
        raise Refused("the mouth is on bone %r, which is not being written" % bone)
    raw = bytearray(20)
    struct.pack_into("<i3fi", raw, 0, bi, *[float(c) for c in fwd],
                     build_mod.flexdesc_index(d, flex))
    d.mouths.append(build_mod.Rec(raw))
    return 1


def add_springbones(d, arm_obj, bone_index):
    """Every spring bone chain the scene authors, on the bone it starts at. Returns the count.

    The donor path matches a chain to a record by the ordinal the import stamped; there is
    no file to match against here, so the chains an import stamped keep their order and the
    ones this scene authored follow in bone order. A tombstoned chain is one the scene has
    taken away and no record is written for it.

    An unresolvable start or end bone is refused rather than skipped, the way an attachment
    on a bone that is not being written is: a chain silently dropped is a garment that goes
    stiff with nothing said.
    """
    claimed, add, _drop = export_mod.spring_chains(arm_obj)
    n = 0
    for pb in [x for _k, x in sorted(claimed.items())] + list(add):
        if pb.name not in bone_index:
            raise Refused("spring bone chain %r starts at a bone that is not being written"
                          % pb.name)
        end = pb.get("vtmb_spring_end")
        if end:
            if end not in bone_index:
                raise Refused("spring bone chain %r ends at %r, which is not being written"
                              % (pb.name, end))
            endbone = bone_index[end]
        else:
            endbone = -1
        fields = dict((field, float(pb[key]))
                      for _at, key, field in export_mod.SPRING_KEYS if key in pb)
        build_mod.add_springbone(d, bone_index[pb.name], endbone,
                                 bool(pb.get("vtmb_spring_disabled")), **fields)
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
                activity="ACT_IDLE", root_motion="extract", bone_index=None):
    """`root_motion` is what becomes of the root bone's path. "extract" takes its net
    ground displacement into one `mstudiomovement_t` and leaves the rest -- the sway and
    rise -- in the keys, which is how a walk cycle is stored: the engine carries the
    entity, the skeleton stays put and still bobs. "none" leaves the whole path in the
    poses, which is right only for an animation that really does translate in model
    space. "in_place" takes the same net displacement out and writes no block, so nothing
    travels on either side. "per_action" reads each action's own stamp instead, where a
    stored "keep" becomes "extract" -- there is no donor file here to keep blocks from.

    Every action becomes an animation; the sequences come after, out of
    `scene_sequences`, and an action no imported sequence cites gets one of its own."""
    moved, unfitted, unkeepable, written = 0, [], [], []
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
        written.append((act, build_mod.add_animation(d, act.name, poses, fps, flags,
                                                     movements)))
    seqs = scene_sequences(d, arm_obj, written, bone_index or {})
    cited = set()
    for rec in d.seqs:
        cited.update(build_mod._seq_anims(rec.raw))
    for act, a in written:
        if a in cited:
            continue
        # `get(key, default)` and not `or`: an imported action carries "" where its
        # sequence claimed no activity, and that is a value, not a gap for the
        # dialog to fill.
        build_mod.add_sequence(d, act.name, a, act.get("vtmb_activity", activity),
                               int(act.get("vtmb_seq_flags", 0)))
    return len(actions), moved, unfitted, unkeepable, seqs


def add_poseparams(d, arm_obj):
    """`mstudioposeparamdesc_t[]` off `vtmb_poseparams`, which an import stamps from the
    file. Carried rather than authored: a blend grid names its axis by one of these, and
    what reads `move_yaw` and `hit_yaw` at runtime is not located."""
    for p in arm_obj.get("vtmb_poseparams") or ():
        build_mod.add_poseparam(d, str(p.get("name") or ""), int(p.get("flags") or 0),
                                float(p.get("start") or 0.0), float(p.get("end") or 0.0),
                                float(p.get("loop") or 0.0))
    return len(d.poseparams)


def stash_bones(arm_obj, bone_index):
    """{the name the file gave a bone: its index in the file being written}.

    A knockback in the stash names its bone by the file's name, and `add_bones` keys by
    Blender's, which a rename moves; `vtmb_bone_name` on the pose bone is the bridge. A
    bone with no stamp answers to its own name.
    """
    out = {}
    for pb in arm_obj.pose.bones:
        if pb.name in bone_index:
            out.setdefault(str(pb.get("vtmb_bone_name") or pb.name), bone_index[pb.name])
    return out


class _Cells(object):
    """Which written animation a stash cell names, decided once per export.

    The name decides, as `match_indices` has it on the donor path: an action called what
    the cell says, from this file or from none. Failing that, the one action this file
    stamped with that animation name, which is a rename. A name the file carried twice
    is matched by `blend_index`, the file's own index, since the name cannot say which.
    """

    def __init__(self, arm_obj, written):
        source = arm_obj.get("vtmb_source")
        self.by_name, self.stamped = {}, {}
        for act, a in written:
            mine = export_mod.same_file(act.get("vtmb_source"), source)
            if mine or not act.get("vtmb_source"):
                self.by_name[act.name] = a
            if mine and act.get("vtmb_anim_name") is not None:
                self.stamped.setdefault(str(act["vtmb_anim_name"]), []).append(
                    (act.get("vtmb_anim_index"), act.name, a))

    def find(self, name, hint):
        """(animation index, None), or (None, why). `why` is None as well where no action
        being written is a candidate at all, which is what leaves a sequence out."""
        got = self.stamped.get(name) or []
        if len(got) > 1:
            hit = [a for i, _n, a in got
                   if hint is not None and i is not None and int(i) == int(hint)]
            if len(hit) == 1:
                return hit[0], None
            return None, ("animation %r, which the file carried %d times -- actions %s -- "
                          "and the stash does not say which"
                          % (name, len(got), ", ".join(sorted(n for _i, n, _a in got))))
        if name in self.by_name:
            return self.by_name[name], None
        if got:
            return got[0][2], None
        return None, None


# The 16 activity-name pointers of an mstudioknockback_t, the only struct-relative words in
# it -- the ones `mdl_build.from_bytes` zeroes.
_KNOCKBACK_NAMES = tuple(0x78 + t * 4 for t in range(16))


def plan_sequences(arm_obj, written):
    """([(stash entry, grid of animation indices)], [labels left out]) for the stash.

    An entry with a cell naming an animation no action being written carries is left out
    and named -- an export handed no actions, or the first few of a file's, writes only the
    sequences whose every cell it has, since writing one short of a cell would play another
    animation in its place -- and an action it cited gets one of its own. A name two actions
    could be is refused, and so is a cell the stash holds as "", the file's own index having
    named nothing.
    """
    cells = _Cells(arm_obj, written)
    out, skipped = [], []
    for k, s in enumerate(arm_obj.get("vtmb_sequences") or ()):
        label = str(s.get("label") or "")
        blends = [[str(v) for v in col] for col in s.get("blends") or ()]
        hint = [list(col) for col in s.get("blend_index") or ()]
        gx = len(blends)
        gy = max((len(col) for col in blends), default=0)
        if not gx or not gy or any(len(col) != gy for col in blends):
            raise Refused("sequence %d, %r, has a blend grid of %s cells, which is not a "
                          "grid" % (k, label, [len(col) for col in blends]))
        if gx > 16 or gy > 16:
            raise Refused("sequence %d, %r, has a %d x %d blend grid; the record holds 16 x "
                          "16" % (k, label, gx, gy))
        grid, why, missing = [], [], False
        for x, col in enumerate(blends):
            row = []
            for y, name in enumerate(col):
                h = hint[x][y] if x < len(hint) and y < len(hint[x]) else None
                a, bad = cells.find(name, h) if name else (None, None)
                if bad is not None:
                    why.append("blend %d, %d names %s" % (x, y, bad))
                missing |= a is None and bool(name)
                row.append(a)
            grid.append(row)
        if why:
            raise Refused("sequence %r: %s" % (label, "; ".join(why)))
        if missing or all(a is None for row in grid for a in row):
            skipped.append(label)
            continue
        empty = [(x, y) for x, row in enumerate(grid) for y, a in enumerate(row)
                 if a is None and not blends[x][y]]
        if empty:
            raise Refused("sequence %r has blend cell%s %s naming no animation -- the "
                          "file's own index there named none. Pick one with the cell's "
                          "button, drawn (kept), in the Dope Sheet's VTMB sequence panel"
                          % (label, "" if len(empty) == 1 else "s",
                             ", ".join("%d, %d" % c for c in empty)))
        out.append((s, grid))
    return out, skipped


def scene_sequences(d, arm_obj, written, bone_index):
    """`arm["vtmb_sequences"]` written as the file's own sequences, in the stash's order.

    `written` is [(action, animation index)]. The label, activity, blend grid, events,
    pose parameters, autolayers, tail fields and knockbacks come off the stash through the
    donor path's own writers, and then each action's Loops and Activity off the action,
    as `apply_sequences` has it. A knockback is rebuilt from the raw record the import
    stashed. A record with none is one an import before 2026-10-15 stashed, and raises
    Refused: its 170 bytes with no scene field are not in the scene.

    Returns {"imported", "written", "skipped"}.
    """
    stash = list(arm_obj.get("vtmb_sequences") or ())
    plan, skipped = plan_sequences(arm_obj, written)
    out = {"imported": len(stash), "written": len(plan), "skipped": skipped}
    if not plan:
        return out
    for s, grid in plan:
        rec = d.seqs[build_mod.add_sequence(d, str(s.get("label") or ""), grid[0][0])]
        struct.pack_into("<ii", rec.raw, mdl_mod.SEQ_GROUPSIZE, len(grid), len(grid[0]))
        for x, row in enumerate(grid):
            for y, a in enumerate(row):
                struct.pack_into("<h", rec.raw, mdl_mod.SEQ_ANIM + x * 0x20 + y * 2, a)
        activity = s.get("activity")
        rec.extra["activity"] = None if activity is None else str(activity)
    first = len(d.seqs) - len(plan)
    # Off the plan and not the stash: an autolayer names a sequence by label, and one the
    # export left out is not in the file to be named.
    labels = {}
    for k, (s, _grid) in enumerate(plan):
        labels.setdefault(str(s.get("label") or ""), first + k)
    bones = stash_bones(arm_obj, bone_index)
    pp = [r.name or "" for r in d.poseparams]
    for k, (s, _grid) in enumerate(plan):
        rec = d.seqs[first + k]
        tail = {key: s[key] for key in s.keys()}
        kbs = list(s.get("knockbacks") or ())
        raws = [kb.get("raw") for kb in kbs]
        if not all(raws):
            raise Refused("sequence %r of %s stashes knockbacks with no raw record, which "
                          "an import before 2026-10-15 left. Re-import the model"
                          % (rec.name, arm_obj.name))
        if kbs:
            for j, kb in enumerate(kbs):
                if str(kb.get("bone") or "") not in bones:
                    raise Refused("knockback %d of sequence %r drives bone %r, which the "
                                  "armature has no bone imported as"
                                  % (j, rec.name, str(kb.get("bone") or "")))
            recs = []
            for j, r in enumerate(raws):
                raw = bytes.fromhex(str(r))
                if len(raw) != mdl_mod.KNOCKBACK_STRIDE:
                    raise Refused("knockback %d of sequence %r stashes %d bytes; the record "
                                  "is %d" % (j, rec.name, len(raw), mdl_mod.KNOCKBACK_STRIDE))
                recs.append(build_mod.Rec(build_mod._z(raw, *_KNOCKBACK_NAMES), None, None,
                                          {"names": [None] * 16}))
            rec.extra["knockbacks"] = recs
        export_mod._apply_events(rec, s.get("events"))
        export_mod._apply_params(rec, s.get("params"), pp, rec.name)
        export_mod._apply_seq_tail(rec, tail, labels, bones)
    for k, acts in export_mod._first_citers(d, {a: act for act, a in written}).items():
        export_mod._apply_seq_action(d.seqs[k], acts)
    return out


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
          hitboxes=False, root_motion="extract", includes=(), no_mesh=False):
    """(Desc, faces, ..., face) for the scene. `hull` is (min, max) and stays the caller's:
    @180/@192 are the movement hull the .qc's $bbox sets, not the mesh's bounds. Without
    one the armature's is written, and without that `new`'s."""
    d = build_mod.new(name, surfaceprop)
    if hull is None:
        hull = scene_hull(arm_obj, scale)
    if hull is not None:
        lo, hi = hull
        _set_hull(d, lo, hi)
    eye = scene_eyeposition(arm_obj, scale)
    if eye is not None:
        struct.pack_into("<3f", d.hdr, mdl_mod.HDR_EYEPOSITION, *eye)
    for p in includes:
        build_mod.add_include(d, p)
    build_mod.set_cdtextures(d, paths_mod.cdtexture_list(cdtexture))
    bone_index = add_bones(d, arm_obj, scale, surfaceprop)
    if not bone_index:
        raise Refused("the armature has no bones")
    owners = []
    face = {"flexes": 0, "flex_records": 0, "eye_meshes": []}
    faces, unskinned, kept, crowded, cloths, stray, face["seams"] = add_meshes(
        d, mesh_objs, bone_index, scale, owners, face)
    face["meshless"] = meshless_model(d, no_mesh)
    if face["meshless"]:
        faces = list(d.faces)
    # Keys first, then rules, then lids and the mouth: each appends the flexdescs it names
    # and the file lacks, and a lid or mouth flex no key or rule made is refused.
    face["controllers"], face["rules"], face["rule_flexdescs"] = export_mod.apply_flex(
        d, None, arm_obj)
    face.update(scene_face(d, arm_obj, face.pop("eye_meshes"), bone_index, scale))
    export_mod.face_flexes_refused(d)
    # The scene is the authority: a box or a set the user placed is written whatever the
    # option says, and the refit fills in only a scene that carries neither.
    nbox, nset = scene_hitboxes(d, arm_obj, bone_index, scale)
    nfit = fit_hitboxes(d, mesh_objs, bone_index, arm_obj, scale) if (
        hitboxes and not nset) else 0
    add_attachments(d, arm_obj, bone_index, scale)
    nvol, vmoved, d.cloth_two_bone = add_cloth_volumes(d, arm_obj, owners, bone_index,
                                                       scale)
    d.cloth_volumes = (nvol, vmoved)
    add_springbones(d, arm_obj, bone_index)
    add_poseparams(d, arm_obj)
    _n, moved, unfitted, unkeepable, seqs = add_actions(
        context, arm_obj, d, actions, scale, use_range, activity, root_motion, bone_index)
    return (d, faces, unskinned, kept, moved, crowded, unfitted, unkeepable, cloths,
            stray, (nbox, nset, nfit), seqs, face)


def _set_hull(d, lo, hi):
    struct.pack_into("<3f", d.hdr, 180, *lo)
    struct.pack_into("<3f", d.hdr, 192, *hi)
    # Only reached by a model with no sequence at all: mdl_build.stamp_sequence_boxes
    # rewrites @168 as the centre of sequence 0's box, which is studiomdl's own rule.
    struct.pack_into("<3f", d.hdr, 168, *[(a + b) * 0.5 for a, b in zip(lo, hi)])


HULL_DEFAULT = build_mod.HULL_DEFAULT


def scene_eyeposition(arm_obj, scale=1.0):
    """@156 off the armature in file units, or None when it carries no key."""
    return export_mod.header_vectors(arm_obj, scale).get(mdl_mod.HDR_EYEPOSITION)


def scene_hull(arm_obj, scale=1.0):
    """The hull the armature carries, in file units, or None when it carries neither key.
    One key alone takes the other from `mdl_build.HULL_DEFAULT`, which is what the panel
    shows for an absent one."""
    typed = export_mod.header_vectors(arm_obj, scale)
    if mdl_mod.HDR_HULL_MIN not in typed and mdl_mod.HDR_HULL_MAX not in typed:
        return None
    return (typed.get(mdl_mod.HDR_HULL_MIN, HULL_DEFAULT[0]),
            typed.get(mdl_mod.HDR_HULL_MAX, HULL_DEFAULT[1]))


def fit_hull(mesh_objs, scale=1.0, arm_obj=None):
    """(min, max) over every vertex handed in, in file units, or None when there are none.

    A starting point and not a truth: @180/@192 is the movement hull the .qc's `$bbox`
    sets, which is why `build` takes it from the caller rather than deriving it. Without
    one `mdl_build.new` leaves (-16,-16,0)..(16,16,72), so a bat claims a humanoid volume.
    Mesh-local like `split_mesh`: both operators call it inside `placed`, which bakes the
    object transform in first. A dialog's `draw` may not write a mesh, so it passes the
    armature instead and each `co` goes through the matrix `placed` would apply.
    """
    lo = hi = None
    inv = arm_obj.matrix_world.inverted() if arm_obj is not None else None
    for obj in mesh_objs:
        rel = None if inv is None else inv @ obj.matrix_world
        if rel is not None and all(abs(rel[r][c] - (r == c)) <= export_mod.PLACED_TOL
                                   for r in range(4) for c in range(4)):
            rel = None
        for v in obj.data.vertices:
            co = v.co if rel is None else rel @ v.co
            p = [co[c] / scale for c in range(3)]
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
    # Read rather than restated: vtx_rebuild.seed writes numLODs at 0x14 and the dialog
    # claims the file is single-LOD, so the claim is graded against the bytes.
    st["lods"] = struct.unpack_from("<i", vtx, 0x14)[0]
    stem = path[:-4] if path.lower().endswith(".mdl") else path
    with open(stem + ".mdl", "wb") as f:
        f.write(data)
    with open(stem + ".dx80.vtx", "wb") as f:
        f.write(vtx)
    return data, vtx, st


def export_scene(context, arm_obj, mesh_objs, actions, path, checksum, before_write=None,
                 **kw):
    """`before_write(d, added)` runs on the built model before any file is written, so a
    caller that refuses over what the build produced leaves nothing on disk. `added` is
    every texture name, since with no donor every material is one the export added."""
    (d, faces, unskinned, kept, moved, crowded, unfitted, unkeepable, cloths,
     stray, hbox, seqs, face) = build(context, arm_obj, mesh_objs, actions,
                                      embedded_name(path), **kw)
    if before_write is not None:
        before_write(d, [r.name for r in d.textures])
    data, vtx, st = write(d, faces, path, checksum)
    return {"bytes": len(data), "vtx_bytes": len(vtx), "bones": len(d.bones),
            "hull_inverted": export_mod.inverted_axes(d),
            "with_root_motion": moved, "unfitted": unfitted,
            "unkeepable": unkeepable,
            "bodyparts": len(d.bodyparts), "materials": len(d.textures),
            # Every material here is one this export added -- there is no donor to have
            # carried a texture record -- so the caller writing .vmt files needs the names
            # and not only the count, and taking them off d.textures is what stops the
            # list drifting from what add_material actually put in the file.
            "material_names": [r.name for r in d.textures],
            # `$iris` is not derivable from the record, so an eye mesh's `.vmt` is copied
            # from its import's and a name only an eyeball record carries gets none.
            "eye_materials": eye_materials(d, face["materials"]),
            "eye_mesh_materials": eye_mesh_materials(d),
            "flexes": face["flexes"], "flex_records": face["flex_records"],
            "flex_controllers": face["controllers"], "flex_rules": face["rules"],
            "eyeballs": face["eyeballs"], "mouths": face["mouth"],
            "eye_unmeshed": face["unmeshed"], "eye_empties": face["eye_empties"],
            "eye_radius_keys": face["radius_keys"], "eye_lid_targets": face["lid_targets"],
            "anims": len(d.anims), "seqs": len(d.seqs),
            # Skipped: labels no written action reaches.
            "seqs_imported": seqs["imported"], "seqs_from_stash": seqs["written"],
            "seqs_skipped": seqs["skipped"],
            "poseparams": len(d.poseparams),
            "includes": len(d.includes),
            "springs": len(d.springbones), "cloth_volumes": d.cloth_volumes,
            "cloth_two_bone": d.cloth_two_bone,
            "hitboxes": sum(len(r.kids) for r in d.hitboxsets),
            "faces": st["tris_out"], "verts": st["verts_out"],
            "model_verts": sum(len(x.extra.get("tangents") or b"") // 16
                               for bp in d.bodyparts for x in bp.kids),
            "kept": kept, "crowded": crowded,
            "unskinned": unskinned, "dropped": dict(d.dropped),
            "cloths": cloths, "stray_groups": stray,
            "scene_hitboxes": hbox[0:2], "refit_hitboxes": hbox[2],
            "lods": st["lods"], "meshless": face["meshless"], "seams": face["seams"]}
