#!/usr/bin/env python3
"""Re-emit a .vtx from its own geometry, taking bone bindings from the paired .mdl. No bpy.

`rebuild` discards the donor's strips and vertex records and builds new ones, so it
exercises everything `vtx_write` decides -- the strip partition, the hardware bone slots and
the bone state changes that name them, where `vtx_write` alone only relays out what it
parsed. That makes it the corpus comparison; it is not what an exporter calls.

`revise` is, and does the opposite: it keeps every strip group the edit did not touch, byte
for byte as the compiler emitted it, and re-emits only the cells named. LOD 0 only. A named
mesh split over several strip groups is dealt out again by `vtx_write.assign_groups`, which
is studiomdl's own rule; a mesh whose groups would change in number or kind, and any mesh
carrying a cloth group, is refused instead.

`seed` and `scratch` write a .vtx for a model that never had one.
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mdl as M
    import vtx as R
    import vtx_write as W
else:
    from . import mdl as M
    from . import vtx as R
    from . import vtx_write as W


def vertex_bones(mesh, orig_ids, verts):
    """Group-local vertex -> its .mdl bones, in the .mdl's own order.

    The table at StudioRender.dll+0x6ce30 pairs hardware slot i with weight i, so sorting
    or deduplicating here binds a bone to another bone's weight.
    """
    out = {}
    for v, orig in enumerate(orig_ids):
        i = mesh.vertexoffset + orig
        if not 0 <= i < len(verts):
            raise ValueError("origMeshVertID %d is outside the mesh" % orig)
        vert = verts[i]
        # numbones 0 stays 0: a rigid prop's records bind nothing and its strip carries
        # no bone state changes. Forcing a bone here gains one the donor never had.
        n = min(4, vert.numbones)
        out[v] = list(vert.bones[:n])
    return out


def rebuild(mdl_path, vtx_path, keep=None):
    """Rebuild every strip group of one .vtx. `keep(triangles)` may drop faces.

    Returns (new bytes, stats). Raises if the .mdl and .vtx disagree about a mesh.
    """
    m = M.Mdl(mdl_path)
    reader = R.Vtx(vtx_path)
    v = W.VtxFile(vtx_path)
    if len(reader.groups) != sum(len(mesh.groups)
                                 for bp in v.bodyparts for mo in bp.models
                                 for lod in mo.lods for mesh in lod.meshes):
        raise ValueError("%s: reader and writer disagree on strip group count"
                         % os.path.basename(vtx_path))
    cache = {}
    st = dict(groups=0, tris_in=0, tris_out=0, verts_in=0, verts_out=0, strips=0)
    ri = 0
    for i, bp in enumerate(v.bodyparts):
        if i >= len(m.bodyparts):
            raise ValueError("%s has more bodyparts than the .mdl" % vtx_path)
        for j, model in enumerate(bp.models):
            src_model = m.bodyparts[i].models[j]
            # Keyed by both indices: Model.index counts within its bodypart, so a file
            # with two bodyparts would otherwise read the wrong one's vertices.
            if (i, j) not in cache:
                cache[i, j] = m.vertices(src_model)
            verts = cache[i, j]
            for lod in model.lods:
                for k, mesh in enumerate(lod.meshes):
                    if k >= len(src_model.meshes):
                        raise ValueError("%s has more meshes than the .mdl" % vtx_path)
                    src_mesh = src_model.meshes[k]
                    for grp in mesh.groups:
                        sg = reader.groups[ri]
                        ri += 1
                        if not grp.numverts:
                            continue
                        orig_ids = grp.orig_vert_ids()
                        tris = [tuple(t) for t in reader.triangles(sg)]
                        st["tris_in"] += len(tris)
                        st["verts_in"] += grp.numverts
                        if keep is not None:
                            tris = keep(tris)
                        vb = vertex_bones(src_mesh, orig_ids, verts) \
                            if grp.flags & W.SG_VERTS_ARE_BONED else None
                        W.rebuild_group(grp, orig_ids, tris, vb,
                                        max_bones=v.maxbones_strip,
                                        max_per_vert=v.maxbones_vert)
                        st["tris_out"] += len(grp.indices) // 3
                        st["verts_out"] += grp.numverts
                        st["strips"] += len(grp.strips)
                        st["groups"] += 1
    return v.to_bytes(), st


def _canon(tri):
    """One triangle as a rotation starting at its lowest corner, so two spellings of the
    same face compare equal while a reversed winding still does not."""
    t = tuple(tri)
    i = t.index(min(t))
    return t[i:] + t[:i]


def _same_faces(a, b):
    return sorted(_canon(t) for t in a) == sorted(_canon(t) for t in b)


def _repartition(mesh, src_mesh, new, verts, v, lod, fixed_function, i, j, k):
    """New triangles dealt out over a split mesh's existing strip groups.

    studiomdl's rule is in `vtx_write.assign_groups`; `plans/vtx-partition-check.py`
    holds it to the shipped corpus. The donor's own groups say which passes the mesh has,
    and a re-partition that would add or drop one is refused rather than guessed at: the
    group array would have to change length, which no writer path does.
    """
    if any(g.flags & W.SG_IS_CLOTH for g in mesh.groups):
        raise ValueError(
            "bodypart %d model %d mesh %d has a strip group flagged %#04x, which the "
            "shipped files set only on cloth, and what puts a triangle in it is unread"
            % (i, j, k, W.SG_IS_CLOTH))
    flexed = set()
    for f in src_mesh.flexes:
        for va in f.verts:
            flexed.add(va.index)
    vb, vw = {}, {}
    for t in new:
        for x in t:
            if x in vb:
                continue
            vert = verts[src_mesh.vertexoffset + x]
            vb[x] = list(vert.bones[:min(4, vert.numbones)])
            vw[x] = list(vert.weights)
    # forceNoFlex is a per-LOD .qc setting the donor states by having no flexed group.
    force_no_flex = not any(g.flags & W.SG_IS_FLEXED
                            for e in lod.meshes for g in e.groups)
    passes = W.assign_groups(new, flexed, vb, vw, v.maxbones_tri, v.maxbones_vert,
                             fixed_function, force_no_flex)
    want = [(bool(g.flags & W.SG_IS_HW_SKINNED), bool(g.flags & W.SG_IS_FLEXED))
            for g in mesh.groups]
    got = [(hw, fx) for hw, fx, _t in passes]
    if want != got:
        raise ValueError(
            "bodypart %d model %d mesh %d is split over %d strip groups and the new "
            "triangles need %d: the donor has %s and the edit wants %s"
            % (i, j, k, len(want), len(got),
               " ".join("hw%d/flex%d" % (a, b) for a, b in want),
               " ".join("hw%d/flex%d" % (a, b) for a, b in got)))
    return [t for _hw, _fx, t in passes]


def revise(mdl, vtx_path, faces, fixed_function=None):
    """One .vtx re-emitted with new triangles for the cells named and the rest untouched.

    `mdl` is the model as it will be written, since a changed mesh's bone bindings come out
    of it; `faces` is {(bodypart, model, mesh): [(a, b, c), ...]} into that mesh's own
    vertices. A cell not named keeps the donor's own triangles, which `rebuild` reproduces
    byte for byte, so editing one mesh leaves every other alone.

    Only LOD 0 is revised; the lower LODs keep their own triangles, which stay valid
    because an addition never moves an original vertex id.

    A mesh split over several strip groups is partitioned again by
    `vtx_write.assign_groups`, studiomdl's own four-pass rule, unless the donor's groups
    would have to change in number or kind, or one of them is flagged as cloth.

    The .mdl's checksum is written through unchanged: the pair only has to agree with each
    other, and both files are written together.
    """
    reader = R.Vtx(vtx_path)
    v = W.VtxFile(vtx_path)
    if fixed_function is None:
        fixed_function = os.path.basename(vtx_path).endswith(".dx7_2bone.vtx")
    st = dict(groups=0, revised=0, tris_out=0, verts_out=0, strips=0)
    ri = 0
    for i, bp in enumerate(v.bodyparts):
        for j, model in enumerate(bp.models):
            src_model = mdl.bodyparts[i].models[j]
            verts = mdl.vertices(src_model)
            for li, lod in enumerate(model.lods):
                for k, mesh in enumerate(lod.meshes):
                    src_mesh = src_model.meshes[k]
                    # LOD 0 only. A lower LOD draws a subset of the same mesh vertices by
                    # their original ids, so an edit that adds vertices leaves its
                    # triangles valid, and handing it LOD 0's would replace it with them.
                    new = faces.get((i, j, k)) if li == 0 else None
                    donor = []
                    for n, grp in enumerate(mesh.groups):
                        sg = reader.groups[ri + n]
                        ids = grp.orig_vert_ids() if grp.numverts else []
                        donor.append((sg, ids,
                                      [tuple(t) for t in reader.triangles(sg)]
                                      if grp.numverts else []))
                    split = None
                    if new is not None and len(mesh.groups) != 1:
                        # The comparison is in mesh-local indices, which is what the caller
                        # speaks; `rebuild_group` wants the group-local ones kept above.
                        flat = [tuple(ids[x] for x in t)
                                for _s, ids, ts in donor for t in ts]
                        if _same_faces(flat, new):
                            new = None
                        else:
                            split = _repartition(mesh, src_mesh, new, verts, v,
                                                 lod, fixed_function, i, j, k)
                    for n, grp in enumerate(mesh.groups):
                        ri += 1
                        if new is None:
                            if not grp.numverts:
                                continue
                            orig, tris = donor[n][1], donor[n][2]
                        else:
                            st["revised"] += 1
                            mine = new if split is None else split[n]
                            orig = sorted(set(x for t in mine for x in t))
                            local = dict((o, m) for m, o in enumerate(orig))
                            tris = [tuple(local[x] for x in t) for t in mine]
                        vb = (vertex_bones(src_mesh, orig, verts)
                              if grp.flags & W.SG_VERTS_ARE_BONED else None)
                        W.rebuild_group(grp, orig, tris, vb,
                                        max_bones=v.maxbones_strip,
                                        max_per_vert=v.maxbones_vert)
                        st["tris_out"] += len(grp.indices) // 3
                        st["verts_out"] += grp.numverts
                        st["strips"] += len(grp.strips)
                        st["groups"] += 1
    return v.to_bytes(), st


def seed(mdl, checksum, boned=True):
    """A structurally valid .vtx for `mdl` with every strip group empty.

    `VtxFile` only ever parses, so a model that never had a .vtx has nothing to hand it.
    This is the smallest file with the right tree: one LOD, one group per mesh, all counts
    zero.  `_layout` treats a zero-count array as fresh and appends it, so filling the
    groups afterwards with `rebuild_group` grows the file rather than colliding with it.
    """
    import struct as _s
    parts = []
    for bp in mdl.bodyparts:
        parts.append([[getattr(e, "cloth", False) for e in m.meshes] for m in bp.models])

    out = bytearray()
    flags = W.SG_VERTS_ARE_BONED if boned else W.SG_VERTS_ARE_PLAIN
    if boned:
        flags |= W.SG_IS_HW_SKINNED
    out += _s.pack("<iiHHiiiiii", W.VTX_VERSION, 24, 16, 9, 3, checksum, 1, 0,
                   len(parts), W.HEADER_SIZE)
    bpo = W.HEADER_SIZE
    out += b"\0" * (len(parts) * W.BODYPART_STRIDE)
    for i, models in enumerate(parts):
        mo = len(out)
        _s.pack_into("<2i", out, bpo + i * W.BODYPART_STRIDE,
                     len(models), mo - (bpo + i * W.BODYPART_STRIDE))
        out += b"\0" * (len(models) * W.MODEL_STRIDE)
        for j, meshcloth in enumerate(models):
            nmesh = len(meshcloth)
            lo = len(out)
            _s.pack_into("<2i", out, mo + j * W.MODEL_STRIDE, 1,
                         lo - (mo + j * W.MODEL_STRIDE))
            out += b"\0" * W.LOD_STRIDE
            eo = len(out)
            _s.pack_into("<2if", out, lo, nmesh, eo - lo, 0.0)
            out += b"\0" * (nmesh * W.MESH_STRIDE)
            for k in range(nmesh):
                go = len(out)
                _s.pack_into("<hhi", out, eo + k * W.MESH_STRIDE, 1, 0,
                             go - (eo + k * W.MESH_STRIDE))
                # A cloth mesh is drawn by the ProcessMesh*_Cloth_* half of
                # g_SoftwareProcessFunc, and StudioRender picks that half only when bit
                # 0x04 has reached the runtime mesh's +0x0c: 0x2c01ab31 tests the byte and
                # 0x2c01ab40 adds 24 to the table index.  Without it the mesh draws from
                # the skinned vertices and the simulated particles never reach the screen.
                # It also cannot be hardware-skinned -- no shipped group carries 0x02 and
                # 0x04 together, over 38 707 groups.
                f = flags
                if meshcloth[k]:
                    f = (f | W.SG_IS_CLOTH) & ~W.SG_IS_HW_SKINNED
                out += _s.pack("<3hBB3i", 0, 0, 0, f, 0,
                               W.GROUP_STRIDE, W.GROUP_STRIDE, W.GROUP_STRIDE)
    # materialReplacementListOffset is indexed unconditionally, so 0 is not "absent": the
    # engine reads the version field as numReplacements and walks off the file.
    _s.pack_into("<i", out, 0x18, len(out))
    out += b"\0" * W.MATREPL_STRIDE
    return W.VtxFile(path="<seed>", data=bytes(out))


def scratch(mdl, faces, checksum=None):
    """Emit a .vtx from nothing for a model that has none.

    `faces[bodypart][model][mesh]` is triangles as indices into that mesh's own vertices.
    """
    v = seed(mdl, mdl.checksum if checksum is None else checksum)
    st = dict(groups=0, tris_out=0, verts_out=0, strips=0)
    for i, bp in enumerate(v.bodyparts):
        verts = mdl.vertices(mdl.bodyparts[i].models[0]) if mdl.bodyparts[i].models else []
        for j, model in enumerate(bp.models):
            src_model = mdl.bodyparts[i].models[j]
            verts = mdl.vertices(src_model)
            for lod in model.lods:
                for k, mesh in enumerate(lod.meshes):
                    src_mesh = src_model.meshes[k]
                    tris = [tuple(t) for t in faces[i][j][k]]
                    orig = sorted(set(x for t in tris for x in t))
                    local = dict((o, n) for n, o in enumerate(orig))
                    grp = mesh.groups[0]
                    vb = vertex_bones(src_mesh, orig, verts)
                    W.rebuild_group(grp, orig, [tuple(local[x] for x in t) for t in tris],
                                    vb, max_bones=v.maxbones_strip,
                                    max_per_vert=v.maxbones_vert)
                    st["tris_out"] += len(grp.indices) // 3
                    st["verts_out"] += grp.numverts
                    st["strips"] += len(grp.strips)
                    st["groups"] += 1
    return v.to_bytes(), st


def resolved_skin(vtxfile):
    """origMeshVertID -> the set of real bone tuples its records resolve to.

    Under SG_IS_HW_SKINNED a record names hardware slots and only the strip's bone state
    changes say which bone a slot holds; without it the record names .mdl bones outright.
    Two files that disagree on slot numbering but agree here bind the same bones.
    """
    import struct
    out = {}
    gi = 0
    for bp in vtxfile.bodyparts:
        for mo in bp.models:
            for lod in mo.lods:
                for mesh in lod.meshes:
                    for g in mesh.groups:
                        gi += 1
                        if not (g.flags & W.SG_VERTS_ARE_BONED):
                            continue
                        hw = bool(g.flags & W.SG_IS_HW_SKINNED)
                        for s in g.strips:
                            slot = dict(s.bone_state)
                            for x in range(s.first_vert,
                                           min(s.first_vert + s.num_verts, g.numverts)):
                                r = struct.unpack_from("<6h", g.verts, x * 12)
                                n = W.bone_map_numbones(r[0])
                                ids = [slot.get(b) for b in r[1:1 + n]] if hw \
                                    else list(r[1:1 + n])
                                out.setdefault((gi, r[5]), set()).add(tuple(ids))
    return out


def face_multiset(reader):
    """Triangles as sorted origMeshVertID triples, per strip group."""
    import collections
    out = collections.Counter()
    for gi, sg in enumerate(reader.groups):
        ids = reader.orig_vert_ids(sg)
        for t in reader.triangles(sg):
            out[(gi, tuple(sorted(ids[x] for x in t)))] += 1
    return out
