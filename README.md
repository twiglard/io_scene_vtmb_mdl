# io_scene_vtmb_mdl

A Blender importer and exporter for **Vampire: The Masquerade — Bloodlines** models,
`.mdl` format version **2531**.

Clean-room: written from the game's own binaries and shipped files, with no existing
model tool consulted. It ships no game content of any kind — everything it reads comes
from your own installation.

## What it does

**Import** — skeletons, meshes, UVs, vertex weights, textures and animations, including
the chained models a character's animations actually live in. It reads the game's `.vpk`
archives directly and decodes `.tth`/`.ttz` textures itself, so a plain GOG install with
the Unofficial Patch works with no extraction step.

**Export**, two separate operators:

- *File > Export > VTMB Model (.mdl)* rebuilds a `.mdl` from its own decoded records,
  using an existing file as the donor. Animations come from Blender's poses; vertex
  positions, normals, UVs and weights are optional. Vertex and face counts may change: a
  mesh that moved is rebuilt whole and its `.dx80.vtx` rewritten beside the model, while a
  mesh that did not comes back byte for byte. Geometry is written to `.dx80.vtx` only, so
  a `.dx7_2bone.vtx`, `.dx90.vtx` or `.sw.vtx` beside the model goes stale and is named in
  a warning — the engine asks for `.dx80.vtx` first and tests only the flavour it loaded.
  **Turn LODs off** is the exception and edits every flavour that is there, since the
  engine picks one by `-dxlevel` and a flavour left at seven LODs still swaps to a coarse
  mesh at distance.
- *File > Export > VTMB Model, no donor (.mdl)* authors a `.mdl` **and** its matching
  `.dx80.vtx` from the scene alone, with no donor file at all.

Both paths have been confirmed in game.

The donor path also edits the parts of a model that have no geometry: bones added,
renamed or deleted (with the animations re-based onto the new bind and everything indexing
a bone renumbered), an animation replaced, appended or deleted, materials renamed with a
`.vmt` and a `.tth`/`.ttz` pair written beside the model on request, cloth gravity and
stiffness taken from the scene, and **Turn LODs off**, which writes 1 into every `.vtx`
LOD count so the model never swaps to a coarse mesh.

**Bone sets** — *Add > VTMB > Bone set...* drops a skeleton into the scene, or merges one
onto the armature already there. Each template is a small `.json` under `templates/`
holding bone **names and hierarchy** plus a named proportion generator; nothing in it is a
coordinate copied out of a shipped model. Merging is by bone name and applying twice adds
nothing, so `human-core` + `human-hands` + `human-hair` compose, and `wing-pair` lands on
either `human-core` or `quadruped`.

The two `anims-human-*` templates carry no bones at all — only the chain of included models
a character reads its animations through. That chain is what turns a bare armature into
something that walks, and the engine joins it **by bone name, case-insensitively**, so a
generated skeleton of the right names plays the game's own animations at its own limb
lengths. Rename a bone and it silently stops animating.

Adding a template is dropping a file in `templates/`. Every one is validated against the
schema before it ships, and packing refuses any field tagged as measured off a shipped
file.

## Requirements

Blender, and a Bloodlines installation to read content from. Development and testing has
been done on Blender 5.2 against GOG + Unofficial Patch; `bl_info` claims a lower minimum
than has actually been exercised, so older versions may load and then fail rather than
refuse cleanly.

## Setup

Install the `.zip` through *Edit > Preferences > Add-ons > Install from Disk*, then set
two fields in the add-on's preferences:

- **Game root** — the directory holding `Vampire/`.
- **Mod dirs** — the mod directory names to search, in engine precedence order, e.g.
  `Unofficial_Patch`.

Nothing else needs configuring. Loose files beat packed ones and later mod directories
win, matching how the engine itself resolves a path.

## Limits

Stated up front, because hitting one should not be how it gets discovered.

**The format cannot express these.**

- Four bone weights per vertex, maximum. The exporter keeps the heaviest four.
- No weights on the two quantised vertex layouts — those records carry no bone or weight
  field at all, so roughly a third of the game's models are rigid by construction. An
  export that changes such a mesh's vertex count rebuilds it as the unquantised layout, so
  it gains weights rather than losing them.
- 20 000 vertices per model, the renderer's own ceiling, and 32 767 per mesh, which is the
  width of the `.vtx` field naming a vertex. Both are refused by name. Nothing shipped
  comes near either — the heaviest model in the game is 18 000 vertices — but a subdivide
  reaches them quickly, because a vertex carrying more than one UV or normal is duplicated.

**Not implemented on the donor path.** These are boundaries, not permanent limits.

- Editing a mesh the compiler split over several strip groups, if the edit actually moved
  its faces. Which group a *new* triangle belongs to is studiomdl's decision and is
  recorded nowhere. 547 of 9705 top-LOD meshes are split that way, 5.64%; the rest edit
  freely, and a split mesh whose faces did not move is patched like any other.
- Collision and ragdoll. Both live in the sibling `.phy`, which nothing here reads or
  writes, so renaming or deleting a bone leaves that file citing a skeleton the model no
  longer has.
- Authoring a skin family the file does not carry. Every family it *does* carry is listed
  in the armature's Object Data tab and pickable, and the export writes the row you
  picked.

**What the no-donor path leaves out**, listed verbatim in its own export dialog:

- collision and ragdoll — both live in the sibling `.phy`, not written
- flex descs, controllers, rules and every vertanim
- eyeballs, mouths and pose parameters
- spring bones, procedural bones, IK chains and bone controllers
- sequence events and autolayers

The donor path writes every one of those bar the `.phy` — cloth, attachments and hitbox
sets included.

Nothing blocks a rebuild: over the 4464 models of a full install, all 4464 re-author and
verify against the donor with none differing. Cloth carriers included — the region moves
whole, and a cloth-bound mesh whose vertex count changed has its per-vertex arrays regrown
for the new count.

## Licence

See [`COPYING`](COPYING).

## Credits

`mstudiobonecontroller_t` corrected to 24 bytes, and the phoneme filter defaulted, by
João Amaro Lagedo ([#2](https://github.com/twiglard/io_scene_vtmb_mdl/pull/2)).

Bloodlines is Troika Games' and Activision's. This project is not affiliated with either,
and redistributes nothing of theirs.
