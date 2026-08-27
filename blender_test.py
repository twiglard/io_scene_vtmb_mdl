#!/usr/bin/env python3
"""Headless acceptance test for the Blender half.

    blender --background --factory-startup --python blender_test.py -- X.mdl
    blender --background --factory-startup --python blender_test.py -- X.mdl \
        --expect-comparisons N

Passes when, after import, each pose bone's armature-space matrix equals the matrix the
file decodes to for that frame. That is checked through Blender's own dependency graph,
so it exercises the bone hierarchy, roll, and the pose composition -- none of which the
plain-Python selfcheck touches.

--expect-comparisons fails the run unless it made exactly N comparisons. Every reading
here folds into a maximum, and a maximum over nothing reads as no error.
"""

import os
import sys

import bpy
import addon_utils
import mathutils

ADDON = "io_scene_vtmb_mdl"

# Blender evaluates poses in float32; the rest pose alone already differs by ~8e-6.
ROT_TOL = 3e-4
POS_TOL = 5e-3


class Tally(object):
    """Every comparison the run makes, by the kind of thing compared.

    Each count is a pair of matrices held against each other once, not a scalar element:
    `pose` and `export` are one bone at one frame, `rest` is one bone, `skin` is one
    vertex group against the armature's bone names.
    """

    def __init__(self):
        self.pose = self.rest = self.export = self.skin = 0

    @property
    def n(self):
        return self.pose + self.rest + self.export + self.skin


TALLY = Tally()


def pos_bound(rot, radius):
    """A flat translation bound is really a bound on bones near the origin: the bone's own
    rotation error is worth rot*radius of translation, 0.009 at a monster finger's 77."""
    return POS_TOL + rot * radius


def enable():
    # default_set=False leaves the addon out of context.preferences.addons entirely, so
    # the preference path -- and with it the VPK and mod-dir roots -- never runs.
    addon_utils.enable(ADDON, default_set=True, persistent=False)
    mod = sys.modules[ADDON]
    print("addon %s enabled, version %s" % (ADDON, mod.bl_info["version"]))
    return mod


def set_game_root(mod, path):
    """A real install puts the chain in a sibling mod dir and its packs, so point the
    addon at the dir holding both the opened file's tree and Vampire\\."""
    prefs = getattr(bpy.context.preferences.addons.get(ADDON), "preferences", None)
    cr = mod.paths.content_root(path)
    if prefs is None or not cr:
        return ""
    prefs.game_root = os.path.dirname(cr)
    return prefs.game_root


def worst_pose_error(mod, arm, m, anim, frames, src=None, root_motion=True):
    dg = bpy.context.evaluated_depsgraph_get()
    remap = None if src is None or src is m else m.bone_remap(src)
    src = src or m
    in_keys = root_motion and bool(anim.movements)
    worst_rot, worst_pos, at, worst_excess, n = 0.0, 0.0, None, float("-inf"), 0
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        dg.update()
        local = src.local_pose(anim, frame) if remap is None \
            else m.retarget_pose(src, anim, frame, remap)
        world = m.world_matrices(local)
        if in_keys:
            off = mod.mdl.root_motion_matrix(anim, frame)
            world = [mod.mdl.mat_mul(off, w) for w in world]
        for b in m.bones:
            want = mathutils.Matrix([world[b.index][0], world[b.index][1],
                                     world[b.index][2], [0, 0, 0, 1]])
            got = arm.pose.bones[b.name].matrix
            rot = max(abs(want[i][j] - got[i][j])
                      for i in range(3) for j in range(3))
            pos = max(abs(want[i][3] - got[i][3]) for i in range(3))
            excess = pos - pos_bound(rot, want.to_translation().length)
            if rot > worst_rot or excess > worst_excess:
                at = (anim.name, frame, b.name)
            worst_rot, worst_pos = max(worst_rot, rot), max(worst_pos, pos)
            worst_excess = max(worst_excess, excess)
            n += 1
            TALLY.pose += 1
    # float("-inf") <= 0.0 would let a comparison over no bone at all read as a pass.
    if not n:
        return float("inf"), float("inf"), "nothing compared", float("inf")
    return worst_rot, worst_pos, at, worst_excess


def check_chained(mod, arm, m, content, meshes=()):
    """The chain is what a character model's animations actually live in, so an import
    that resolves none of it silently looks like a model with four animations.

    `meshes` is what makes this the mesh-and-chain case: the chain is joined by bone name
    across files, and a skinned mesh names its bones the same way, so a chained import that
    reordered or dropped a bone leaves a vertex group pointing at nothing.
    """
    pairs, missing = mod.blender_import.pick_animations(m, content, "", 0, True)
    chained = [(src, a) for src, a in pairs if src is not m]
    print("  chain: %d animations over %d files, %d unresolved"
          % (len(chained), len({s.path for s, _ in chained}), len(missing)))
    if missing:
        # Not a failure: on a real install most of the chain is VPK-only, and the
        # extracted-content root that would supply it is an addon preference.
        print("  unresolved: %s" % ", ".join(missing[:3]))
    if not chained:
        return True
    have = {b.name for b in arm.pose.bones}
    for o in meshes:
        stray = [g.name for g in o.vertex_groups if g.name not in have]
        TALLY.skin += len(o.vertex_groups)
        print("  mesh %r: %d verts over %d vertex groups, %d naming no bone"
              % (o.name, len(o.data.vertices), len(o.vertex_groups), len(stray)))
        if stray:
            print("  chained import left %r skinned to %s" % (o.name, ", ".join(stray[:4])))
            return False
    src, anim = chained[0]
    act = bpy.data.actions.get(mod.blender_import.action_name(m, src, anim))
    if act is None:
        print("  no action for chained %r" % anim.name)
        return False
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot"):
        for slot in act.slots:
            arm.animation_data.action_slot = slot
            break
    frames = sorted({0, anim.numframes // 2, max(0, anim.numframes - 1)})
    wr, wp, at, ex = worst_pose_error(mod, arm, m, anim, frames, src=src)
    print("  chained pose vs file: rotation %.7f, translation %.7f units (%+.7f over "
          "bound)  %s" % (wr, wp, ex, at))
    return wr < ROT_TOL and ex <= 0.0


def check_export(mod, arm, m, anim, path):
    """Write the imported action straight back out and decode both.

    Compared as poses, not as int16: the error here is Blender's float32 pose evaluation,
    which is a physical quantity, while the same drift in LSB means whatever each bone's
    fitted rotscale happens to be.
    """
    import tempfile
    dst = os.path.join(tempfile.gettempdir(), "vtmb-export-roundtrip.mdl")
    r = mod.blender_export.export_action(bpy.context, arm, path, dst, scale=1.0)
    m2 = mod.mdl.Mdl(dst)
    a2 = m2.anims[r["index"]]
    if a2.numframes != anim.numframes:
        print("  EXPORT frame count %d != %d" % (a2.numframes, anim.numframes))
        return False
    wr = wp = 0.0
    where = None
    for f in range(anim.numframes):
        p0, p1 = m.local_pose(anim, f), m2.local_pose(a2, f)
        for b in m.bones:
            dp = max(abs(p0[b.index][0][c] - p1[b.index][0][c]) for c in range(3))
            q0, q1 = p0[b.index][1], p1[b.index][1]
            dq = min(max(abs(q0[c] - q1[c]) for c in range(4)),
                     max(abs(q0[c] + q1[c]) for c in range(4)))
            if dq > wr:
                wr, where = dq, "%s frame %d" % (b.name, f)
            wp = max(wp, dp)
            TALLY.export += 1
    print("  export round trip: %d frames, rotation %.7f, translation %.7f units  (%s)"
          % (anim.numframes, wr, wp, where))
    print("  file %d -> %d bytes, %d movement blocks kept"
          % (r["was"], r["bytes"], r["movements"]))
    return wr < ROT_TOL and wp < 2.0 * POS_TOL


def check_root_motion(mod, path, m):
    """Re-import one animation that carries root motion, on its own. The offset reaches
    Blender only through the keys, so nothing short of a posed armature says whether it
    got there."""
    moving = [a for a in m.anims if a.movements]
    print("  animations with movement blocks: %d/%d" % (len(moving), len(m.anims)))
    if not moving:
        return True
    # Prefer one that turns: any moving animation covers the translation, but only a
    # nonzero yaw reaches root_motion_matrix's rotation, and 21 of 11052 carry one.
    pick = max(moving, key=lambda a: max(abs(mv.angle) for mv in a.movements))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mod = enable()
    set_game_root(mod, path)
    res = bpy.ops.import_scene.vtmb_mdl(
        filepath=path, with_mesh=False, with_anims=True, with_chained=False,
        anim_filter=pick.name, max_anims=1)
    act = next(iter(bpy.data.actions), None)
    if res != {"FINISHED"} or act is None:
        print("  root-motion re-import failed: %s" % (res,))
        return False
    m = mod.mdl.Mdl(path)
    a = m.anims[act["vtmb_anim_index"]]
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot"):
        for slot in act.slots:
            arm.animation_data.action_slot = slot
            break
    last = max(0, a.numframes - 1)
    pos, yaw = mod.mdl.anim_position(a, last)
    print("  %r: %d frames, root motion %.3f units, yaw %.2f deg"
          % (a.name, a.numframes, mathutils.Vector(pos).length, yaw))
    frames = sorted({0, last // 2, last})
    wr, wp, at, ex = worst_pose_error(mod, arm, m, a, frames)
    print("  root-motion pose vs file: rotation %.7f, translation %.7f units (%+.7f over "
          "bound)  %s" % (wr, wp, ex, at))
    wr0, wp0, _, _ = worst_pose_error(mod, arm, m, a, frames, root_motion=False)
    print("  with the offset left out:  rotation %.7f, translation %.7f units"
          % (wr0, wp0))
    # Without that second number the first passes just as well on an animation that
    # happens not to move, which would test nothing.
    return wr < ROT_TOL and ex <= 0.0 and wp0 > 1.0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    # Ahead of the loop below, which takes every remaining word for a model path.
    expect = None
    if "--expect-comparisons" in argv:
        i = argv.index("--expect-comparisons")
        expect, argv = int(argv[i + 1]), argv[:i] + argv[i + 2:]
    if not argv:
        sys.exit("usage: ... -- X.mdl")
    ok = True
    for path in argv:
        print("\n=== %s" % path)
        # read_factory_settings unregisters non-persistent addons, so enable after it.
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mod = enable()
        print("  game root: %r" % set_game_root(mod, path))
        m0 = mod.mdl.Mdl(path)
        # Enough headroom past the file's own animations to reach the include chain.
        res = bpy.ops.import_scene.vtmb_mdl(
            filepath=path, with_mesh=True, with_anims=True,
            max_anims=len(m0.anims) + 2)
        if res != {"FINISHED"}:
            print("  IMPORT FAILED: %s" % (res,))
            ok = False
            continue
        arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        print("  armature %r: %d bones" % (arm.name, len(arm.pose.bones)))
        opts = arm.get("vtmb_import")
        print("  vtmb_import: %s" % ({k: opts[k] for k in sorted(opts.keys())}
                                     if opts else None))
        for o in meshes:
            uv = len(o.data.uv_layers)
            groups = sum(1 for g in o.vertex_groups)
            print("  mesh %r: %d verts, %d faces, %d uv layers, %d vgroups, "
                  "%d materials, %d modifiers"
                  % (o.name, len(o.data.vertices), len(o.data.polygons), uv,
                     groups, len(o.data.materials), len(o.modifiers)))
            assert uv == 1, "no UV layer"
            assert len(o.data.polygons) > 0, "no faces"
        acts = [a for a in bpy.data.actions]
        print("  actions: %s" % [a.name for a in acts])

        m = mod.mdl.Mdl(path)
        rest = m.rest_world_matrices()
        wr = wp = 0.0
        for b in m.bones:
            got = arm.pose.bones[b.name].bone.matrix_local
            wr = max(wr, max(abs(rest[b.index][i][j] - got[i][j])
                             for i in range(3) for j in range(3)))
            wp = max(wp, max(abs(rest[b.index][i][3] - got[i][3])
                             for i in range(3)))
            TALLY.rest += 1
        print("  rest matrix_local vs file: rotation %.7f, translation %.7f"
              % (wr, wp))
        ok &= wr < ROT_TOL and wp < POS_TOL

        if m.anims and acts:
            anim = m.anims[0]
            # bpy.data.actions is name-sorted, so acts[0] need not be anims[0].
            act = bpy.data.actions.get(anim.name)
            if act is None:
                print("  no action for %r" % anim.name)
                ok = False
                continue
            frames = sorted({0, anim.numframes // 2, max(0, anim.numframes - 1)})
            arm.animation_data.action = act
            if hasattr(arm.animation_data, "action_slot"):
                for slot in act.slots:
                    arm.animation_data.action_slot = slot
                    break
            wr, wp, at, ex = worst_pose_error(mod, arm, m, anim, frames)
            print("  pose vs file: rotation %.7f, translation %.7f units (%+.7f over "
                  "bound)  %s" % (wr, wp, ex, at))
            ok &= wr < ROT_TOL and ex <= 0.0
            ok &= check_export(mod, arm, m, anim, path)
        else:
            print("  no animations to verify")

        if m.includes:
            prefs = bpy.context.preferences.addons[ADDON].preferences
            ok &= check_chained(mod, arm, m, mod.blender_import.Content(
                mod.paths.roots(path, prefs.game_root,
                                mod.paths.split_list(prefs.mods))), meshes)

        # Last: it reloads factory settings, which takes the scene above with it.
        ok &= check_root_motion(mod, path, m)
    print("\n%d comparisons over %d model(s): %d posed bone matrices, %d rest matrices, "
          "%d written bone poses, %d vertex groups"
          % (TALLY.n, len(argv), TALLY.pose, TALLY.rest, TALLY.export, TALLY.skin))
    if expect is not None and TALLY.n != expect:
        print("   FAIL %d comparisons, expected exactly %d" % (TALLY.n, expect))
        ok = False
    print("\nVERDICT: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
