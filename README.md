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
  positions, normals, UVs and weights are optional.
- *File > Export > VTMB Model, no donor (.mdl)* authors a `.mdl` **and** its matching
  `.dx80.vtx` from the scene alone, with no donor file at all.

Both paths have been confirmed in game.

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

- One UV per vertex. A seam is spelled by duplicating the vertex, which changes the
  vertex count.
- Four bone weights per vertex, maximum. The exporter keeps the heaviest four.
- No weights on the two quantised vertex layouts — those records carry no bone or weight
  field at all, so roughly a third of the game's models are rigid by construction.

**Not implemented on the donor path.** These are boundaries, not permanent limits.

- Changing the vertex or face count.
- Removing a bone.
- Authoring new materials or textures. Repointing an existing one is a text edit to the
  `.vmt`; supplying a genuinely new image would need a `.tth` encoder, which does not
  exist here.
- Skin families past the first, so alternate looks are invisible in Blender.
- Adding or removing an animation. The dialog replaces; it does not append.

**What the no-donor path leaves out**, listed verbatim in its own export dialog:

- the include chain, so only this scene's animations exist
- collision and ragdoll — both live in the sibling `.phy`, not written
- cloth, flex descs, controllers, rules and every vertanim
- eyeballs, mouths and pose parameters
- spring bones, procedural bones, IK chains and bone controllers
- attachments, sequence events and autolayers

Cloth is the one thing that blocks a rebuild outright: over the 4464 models of a full
install, 4391 rebuild and verify byte-for-byte and the 73 refusals are cloth carriers in
every case.

## Licence

See [`COPYING`](COPYING).

## Credits

`mstudiobonecontroller_t` corrected to 24 bytes, and the phoneme filter defaulted, by
João Amaro Lagedo ([#2](https://github.com/twiglard/io_scene_vtmb_mdl/pull/2)).

Bloodlines is Troika Games' and Activision's. This project is not affiliated with either,
and redistributes nothing of theirs.
