"""Lets a script answer the addon's dialogs and file browsers in advance, so none is shown."""

skip_dialogs = False
# {bl_idname: {property: value}}, set on the operator before execute, as a user would set them.
dialog_answers = {}
# (bl_idname, the operator's properties when execute ran), one per dialog skipped.
skipped_dialogs = []


def answer(op, context):
    name = type(op).bl_idname
    for prop, value in dialog_answers.get(name, {}).items():
        setattr(op, prop, value)
    skipped_dialogs.append((name, op.as_keywords()))
    return op.execute(context)
