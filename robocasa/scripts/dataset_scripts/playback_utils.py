"""
Shared helpers for dataset playback (instruction resolution, etc.).
Used by playback_dataset.py and playback_dataset_hdf5.py.
"""


def _format_cat_as_lang(cat):
    """Format object category for display (mirrors object_utils.get_obj_lang logic)."""
    if not cat:
        return ""
    lang = cat.replace("_", " ")
    if lang == "kettle electric":
        lang = "electric kettle"
    elif lang == "kettle non electric":
        lang = "kettle"
    elif lang == "bread flat":
        lang = "bread"
    elif lang == "oil and vinegar bottle":
        lang = "oil/vinegar bottle"
    elif lang == "salt and pepper shaker":
        lang = "salt/pepper shaker"
    elif lang == "jug wide opening":
        lang = "jug"
    return lang


def resolve_instruction_from_ep_meta(ep_meta):
    """
    Return the exact instruction for this episode. Uses ep_meta['lang'] when present.
    If the stored lang contains placeholders (e.g. {condiment_lang}), resolve them
    from ep_meta['object_cfgs'] so playback shows the concrete instruction (e.g. 'ketchup').
    """
    lang = ep_meta.get("lang") or ""
    if not lang:
        return None
    # Already resolved (no placeholders)
    if "{" not in lang:
        return lang
    # Build substitutions from object_cfgs: obj_name -> lang, and obj_name_lang -> lang
    object_cfgs = ep_meta.get("object_cfgs") or []
    subs = {}
    for cfg in object_cfgs:
        name = cfg.get("name")
        info = cfg.get("info") or {}
        cat = info.get("cat")
        if name and cat is not None:
            obj_lang = _format_cat_as_lang(cat)
            subs[name] = obj_lang
            subs[f"{name}_lang"] = obj_lang
    if not subs:
        return lang
    try:
        return lang.format(**subs)
    except KeyError:
        return lang
