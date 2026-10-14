"""Shared helper: name the nearest LANDMARK fixture to a point, for nav language.

Used by washing_dishes movers to qualify a counter/cabinet with the appliance next
to it ("the counter by the coffee machine", "the open cabinet by the toaster oven").
"""
import re
import numpy as np

# Fixture classes that are NOT landmarks (structural, storage, overhead, decorative).
_NON_LANDMARK = {
    "Wall", "Floor", "Counter", "Box", "Stool", "Window", "Accessory",
    "WallAccessory", "Hood", "Shelf", "Plant", "Door",
    "Cabinet", "SingleCabinet", "HingeCabinet", "PanelCabinet", "OpenCabinet",
    "Drawer", "HousingCabinet", "Housing", "Framework", "Floorplan",
}
# Class -> display name overrides (otherwise de-camel-cased).
_DISPLAY = {
    "CoffeeMachine": "coffee machine",
    "FridgeSideBySide": "fridge",
    "FridgeFrenchDoor": "fridge",
    "Fridge": "fridge",
    "StandMixer": "stand mixer",
    "ToasterOven": "toaster oven",
}


def _display(cls_name: str) -> str:
    if cls_name in _DISPLAY:
        return _DISPLAY[cls_name]
    # de-camel-case: "Stovetop" -> "stovetop", "CoffeeMachine" -> "coffee machine"
    words = re.findall(r"[A-Z][a-z0-9]*", cls_name) or [cls_name]
    return " ".join(w.lower() for w in words)


def nearest_landmark(env, xy, cap=1.2):
    """Display name of the nearest landmark fixture to point `xy` (within `cap` m),
    or None if none qualifies.
    """
    xy = np.asarray(xy)[:2]
    fixtures = getattr(env, "fixtures", None)
    if not fixtures:
        return None
    best = None
    for fo in fixtures.values():
        cls = type(fo).__name__
        if cls in _NON_LANDMARK or cls.endswith("Cabinet") or cls.endswith("Drawer"):
            continue
        try:
            p = np.asarray(fo.pos)[:2]
        except Exception:
            continue
        d = float(np.linalg.norm(p - xy))
        if best is None or d < best[0]:
            best = (d, _display(cls))
    if best is None or best[0] > cap:
        return None
    return best[1]
