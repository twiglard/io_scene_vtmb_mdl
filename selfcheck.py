#!/usr/bin/env python3
"""Format check for MDL v2531. Runs without Blender.

    python -m io_scene_vtmb_mdl.selfcheck X.mdl [...]
    python selfcheck.py --corpus DIR

Checks that cannot pass if a convention is wrong: FK(rest) * poseToBone == identity,
every decoded quaternion is unit, every .vtx face index is in range, and every
origMeshVertID lands inside the mesh it claims.

Each returns (ok, comparisons, why-none), so a check with nothing to compare reports
that instead of the 0.0 that reads as perfection, and the corpus summary carries a
denominator per check rather than one file count for all four.
"""

import collections
import math
import os
import struct
import sys

if __package__ in (None, ""):
    # Import the format modules directly rather than through the package, whose
    # __init__ is the addon entry point and pulls in bpy.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mdl as mdl_mod
    import normal_table as NT
    import vtx as vtx_mod
else:
    from . import mdl as mdl_mod
    from . import normal_table as NT
    from . import vtx as vtx_mod

CHECKS = ("rest", "quats", "vertices", "geometry", "includes")


def check_rest(m, verbose=True):
    if not m.bones:
        return True, 0, "no bones"
    worst_bone, worst_dev, n = None, 0.0, 0
    for b, w in zip(m.bones, m.rest_world_matrices()):
        dev = mdl_mod.mat_max_dev_from_identity(mdl_mod.mat_mul(w, b.posetobone))
        n += 1
        if dev > worst_dev:
            worst_dev, worst_bone = dev, b.name
    if verbose:
        print("  FK(rest) * poseToBone vs identity : over %d bones, worst %.7f  (%s)"
              % (n, worst_dev, worst_bone))
    # 271-bone cinematic rigs accumulate past 1e-3 through float32 in the file itself.
    return worst_dev < 5e-3, n, None


def check_quats(m, verbose=True):
    worst_q, worst_q_at, n = 0.0, None, 0
    for a in m.anims:
        if a.numframes <= 0:
            continue
        for frame in {0, a.numframes // 2, a.numframes - 1}:
            for b, (_, q) in zip(m.bones, m.local_pose(a, frame)):
                dev = abs(math.sqrt(sum(x * x for x in q)) - 1.0)
                n += 1
                if dev > worst_q:
                    worst_q, worst_q_at = dev, (a.name, frame, b.name)
    if not n:
        why = "%d animations over %d bones" % (len(m.anims), len(m.bones))
        if verbose:
            print("  decoded |q| vs 1 : NOT MEASURED, %s" % why)
        return True, 0, why
    if verbose:
        print("  decoded |q| vs 1 : over %d quaternions, worst %.8f  %s"
              % (n, worst_q, worst_q_at))
    return worst_q < 1e-3, n, None


def check_includes(m, verbose=True):
    ok = True
    for rel in m.includes:
        # A wrong INCLUDE_STRIDE lands the name offset inside another record, so
        # "every name is a printable .mdl path" pins it; no other stride passes.
        if not rel.isprintable() or not rel.lower().endswith(".mdl"):
            print("  !!! include model name %r is not a .mdl path" % rel[:60])
            ok = False
    if verbose and m.includes:
        print("  %d include models: %s" % (len(m.includes), ", ".join(m.includes)))
    if not m.includes:
        return True, 0, "no include models"
    return ok, len(m.includes), None


def check_geometry(m, verbose=True):
    path = next((p for p in vtx_mod.companion_paths(m.path) if os.path.exists(p)),
                None)
    if path is None:
        if verbose:
            print("  no .vtx companion -- geometry NOT MEASURED")
        return True, 0, "no .vtx companion"
    v = vtx_mod.Vtx(path)
    models = [mo for bp in m.bodyparts for mo in bp.models]
    if v.checksum != m.checksum:
        print("  !!! vtx checksum %#x != mdl %#x" % (v.checksum & 0xffffffff,
                                                     m.checksum & 0xffffffff))
        return False, 1, None
    tris = verts = 0
    ok = True
    for g in v.groups:
        if g.model >= len(models):
            print("  !!! stripgroup model %d out of range" % g.model)
            return False, tris + verts, None
        model = models[g.model]
        if g.lod == 0 and g.mesh >= len(model.meshes):
            print("  !!! stripgroup mesh %d of %d" % (g.mesh, len(model.meshes)))
            return False, tris + verts, None
        for tri in v.triangles(g):
            if max(tri) >= g.numverts:
                print("  !!! face index %d >= numverts %d" % (max(tri), g.numverts))
                ok = False
                break
            tris += 1
        if g.lod == 0:
            mesh = model.meshes[g.mesh]
            ids = v.orig_vert_ids(g)
            verts += len(ids)
            if ids and (min(ids) < 0 or max(ids) >= mesh.numvertices):
                print("  !!! origMeshVertID %d..%d outside mesh nv %d"
                      % (min(ids), max(ids), mesh.numvertices))
                ok = False
    if verbose:
        strides = sorted({g.stride for g in v.groups})
        print("  vtx v%d: %d stripgroups, strides %s, %d faces, %d strip verts"
              % (v.version, len(v.groups), strides, tris, verts))
    if not tris and not verts:
        return ok, 0, "vtx holds no face and no strip vertex"
    return ok, tris + verts, None


def check_vertices(m, verbose=True):
    ok, n, ftypes = True, 0, set()
    for bp in m.bodyparts:
        for model in bp.models:
            if not model.numvertices:
                continue
            ftypes.add(model.filetype)
            try:
                vs = m.vertices(model)
            except ValueError as exc:
                print("  !!! %s" % exc)
                return False, n, None
            if model.filetype != 0:
                # A packed coordinate cannot leave the model's own quantisation range, so
                # one outside it means the normalisation is wrong. Sorted: quant_scale may
                # be negative.
                span = mdl_mod.QUANT_MAX[model.filetype] * mdl_mod.QUANT_NORM[model.filetype]
                rng = [sorted((model.quant_offset[c],
                               model.quant_offset[c] + span * model.quant_scale[c]))
                       for c in range(3)]
                slack = [1e-3 * max(1.0, hi - lo) for lo, hi in rng]
                out = [v for v in vs
                       if any(not rng[c][0] - slack[c] <= v.pos[c] <= rng[c][1] + slack[c]
                              for c in range(3))]
                # The normal is an index into a table with no count and no bounds check in
                # the binary, so what grades the field's offset and stride is that every
                # stored value lands inside it. A weight there is not: neither quantised
                # record has a weight or bone field, so those models are rigid.
                stride = mdl_mod.VERTEX_STRIDE[model.filetype]
                nbad = 0
                for i in range(model.numvertices):
                    o = model.vertexbase + i * stride
                    raw = (struct.unpack_from("<H", m.d, o + 6)[0]
                           if model.filetype == 1 else m.d[o + 3])
                    nbad += NT.decode(model.filetype, raw) is None
                n += len(vs) * 2
                if verbose:
                    print("  %d verts %r filetype=%d: %d outside the quantisation range "
                          "%s, %d normals outside the table; no weight is in the file"
                          % (len(vs), model.name, model.filetype, len(out),
                             " ".join("%.3f..%.3f" % (a, b) for a, b in rng), nbad))
                ok &= not out and not nbad
                continue
            lens = [math.sqrt(sum(x * x for x in v.normal)) for v in vs]
            # A handful of degenerate verts carry a null normal; only a normal that is
            # present but not unit would mean the layout is wrong.
            null_n = sum(1 for x in lens if x == 0.0)
            worst_n = max([abs(x - 1.0) for x in lens if x > 0.0] or [0.0])
            # There are 4 bone slots but only 3 weights, so a vertex needing 4 bones
            # sums short of 255. A wrong layout would miss on nearly all of them.
            off_w = sum(1 for v in vs if abs(sum(v.weights) - 1.0) > 1e-6)
            bad = [v for v in vs if any(b < 0 or b >= len(m.bones) for b in v.bones)]
            n += len(vs)
            if verbose:
                print("  %d verts %r: worst||n|-1|=%.6f (%d null) "
                      "weights!=1 on %d out-of-range bones=%d"
                      % (len(vs), model.name, worst_n, null_n, off_w, len(bad)))
            ok &= worst_n < 1e-3 and off_w <= max(8, len(vs) // 10) and not bad
    if not n:
        why = ("filetype %s carries no vertex to grade"
               % sorted(ftypes) if ftypes else "no model with vertices")
        if verbose:
            print("  normals and weights : NOT MEASURED, %s" % why)
        return True, 0, why
    return ok, n, None


def check(path, verbose=True):
    """{check name: (ok, comparisons, why-none)} for one model."""
    m = mdl_mod.Mdl(path)
    if verbose:
        nmodels = sum(len(bp.models) for bp in m.bodyparts)
        print("%s\n  v%d  %d bones, %d animations, %d sequences, %d bodyparts/"
              "%d models, %d materials"
              % (path, m.version, len(m.bones), len(m.anims), len(m.seqs),
                 len(m.bodyparts), nmodels, len(m.materials)))
    return {"rest": check_rest(m, verbose), "quats": check_quats(m, verbose),
            "vertices": check_vertices(m, verbose),
            "geometry": check_geometry(m, verbose),
            "includes": check_includes(m, verbose)}


def passed(res):
    return all(r[0] for r in res.values())


def corpus(root, expect_fail=0):
    total = failed = 0
    ran = collections.Counter()
    cmp_ = collections.Counter()
    why = collections.defaultdict(collections.Counter)
    for dirpath, _, names in os.walk(root):
        for n in names:
            if not n.lower().endswith(".mdl"):
                continue
            p = os.path.join(dirpath, n)
            total += 1
            try:
                res = check(p, verbose=False)
            except Exception as exc:
                failed += 1
                print("FAIL %s: %s: %s" % (p, type(exc).__name__, exc))
                continue
            for k, (ok, cnt, reason) in res.items():
                if cnt:
                    ran[k] += 1
                    cmp_[k] += cnt
                else:
                    why[k][reason or "?"] += 1
            if not passed(res):
                failed += 1
                print("FAIL %s" % p)
    print("\n%d models checked, %d failed" % (total, failed))
    print("%-10s %8s %14s   files that compared nothing" % ("check", "files", "compared"))
    empty = []
    for k in CHECKS:
        print("%-10s %8d %14d   %s"
              % (k, ran[k], cmp_[k],
                 ", ".join("%d %s" % (v, r) for r, v in why[k].most_common(3)) or "-"))
        if not ran[k]:
            empty.append(k)
    if not total:
        print("NOT MEASURED: no .mdl found under %s" % root)
    for k in empty:
        print("NOT MEASURED: %s compared nothing on any file" % k)
    if failed != expect_fail:
        print("%d failed against an expected %d" % (failed, expect_fail))
    ok = total and failed == expect_fail and not empty
    print("VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main(argv):
    if len(argv) > 2 and argv[1] == "--corpus":
        expect = int(argv[argv.index("--expect-fail") + 1]) \
            if "--expect-fail" in argv else 0
        sys.exit(0 if corpus(argv[2], expect) else 1)
    if len(argv) < 2:
        sys.exit(__doc__)
    failed = 0
    for path in argv[1:]:
        res = check(path)
        if not passed(res):
            failed += 1
            print("  SELF-CHECK FAILED")
        print("  compared: %s"
              % ", ".join("%s=%d%s" % (k, res[k][1],
                                       "" if res[k][1] else " (%s)" % (res[k][2] or "?"))
                          for k in CHECKS))
        print()
    print("VERDICT: %s" % ("PASS" if not failed else "FAIL"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv)
