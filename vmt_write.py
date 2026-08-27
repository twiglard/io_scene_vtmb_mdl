"""The `.vmt` beside a material the exporter has just added to a `.mdl`.

A `.mdl` carries a material name and a directory list and nothing else -- no shader, no
texture path, no render state -- so a new `mstudiotexture_t` resolves to the purple
checkerboard until `materials/<dir>/<name>.vmt` exists. That file is plain-text
KeyValues, 105 B on `tor_femamor0head.vmt`, and this writes it.

The recipe is `andrei.vmt`: Troika's own, unpatched, and it works in game.

    VertexLitGeneric
    {
        "$basetexture"              "models\character\monster\andrei\andrei"
        "$selfillum"                "1"
        "$bumpmap"                  "models\character\monster\andrei\andreinormal"
        "$normalmapalphaenvmapmask" "1"
        "$envmap"                   "envmap\asylum"
    }

`$bumpmap` draws nothing without an `$envmap` beside it -- 50 of the 148 bumped
`VertexLitGeneric` materials in the corpus have none and so cost texture memory and draw
nothing. `env_cubemap` is the default here because it is the runtime cubemap and works in
any map that has one built; the 29 shipped alternatives name a static cubemap and tie the
model to the map that cubemap was baked for.

`$normalmapalphaenvmapmask` picks the `_MultByAlpha` shader variant and reads the normal
map's alpha as a per-texel gloss level, so it is only written where the caller says the
map has an alpha channel. No shipped material declares it over a map that has none.

`$envmaptint`, `$envmapcontrast` and `$envmapsaturation` are not written: 0 of the 98
bumped `VertexLitGeneric` that carry an `$envmap` use the last two, the 21 that carry the
first are all `models/scenery/` and none is a character, and the third occurs in no file
under `materials/` at all.
"""

import os

# Shipped `.vmt` write the separator this way, and Source takes either.
SEP = "\\"


def texture_path(directory, name):
    """`models\character\...\andrei` -- a `$basetexture` value, no extension.

    The directory is the same string the `.mdl`'s cdtexture list carries, which is
    relative to `materials/` and already carries its trailing separator on 4442 of the
    4445 shipped models.
    """
    d = (directory or "").replace("/", SEP).strip(SEP)
    return (d + SEP + name) if d else name


def vmt_text(basetexture, bumpmap=None, envmap="env_cubemap", alpha_mask=False,
             selfillum=False, shader="VertexLitGeneric", extra=()):
    """The file's text. `extra` is (key, value) pairs appended in the order given."""
    keys = [("$basetexture", basetexture)]
    if selfillum:
        keys.append(("$selfillum", "1"))
    if bumpmap:
        keys.append(("$bumpmap", bumpmap))
        if alpha_mask:
            keys.append(("$normalmapalphaenvmapmask", "1"))
        if envmap:
            keys.append(("$envmap", envmap))
    keys.extend(extra)
    pad = max(len(k) for k, _v in keys) + 2
    body = "".join('    "%s"%s"%s"\n' % (k, " " * (pad - len(k) - 1), v)
                   for k, v in keys)
    return "%s\n{\n%s}\n" % (shader, body)


def write_vmt(path, text, overwrite=False):
    """True if it was written. An existing file is left alone unless `overwrite`.

    The game tree is a test rig and a `.vmt` under it may be Troika's, so a material this
    addon names after an existing one must not silently replace it.
    """
    if os.path.exists(path) and not overwrite:
        return False
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="latin1", newline="\r\n") as f:
        f.write(text)
    return True
