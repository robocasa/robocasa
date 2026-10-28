"""Shared camera names for configuration and simulator observations."""

CAMERAS = {
    "ego_view": "robot0_head_camera",
    "left_wrist": "robot0_left_wrist_camera",
    "right_wrist": "robot0_right_wrist_camera",
}
_LEGACY_NAMES = {"wrist_view": "right_wrist"}


def normalize_camera_names(names):
    """Resolve old configuration names to the published dataset camera keys."""
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) for name in names)):
        raise ValueError("camera_names must be a nonempty list of camera names")
    normalized = [_LEGACY_NAMES.get(name, name) for name in names]
    if any(name not in CAMERAS for name in normalized):
        raise ValueError(f"camera_names must select supported G1 cameras: {list(CAMERAS)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("camera_names must select unique cameras; wrist_view aliases right_wrist")
    return normalized
