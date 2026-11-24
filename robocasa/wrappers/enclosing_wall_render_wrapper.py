from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import yaml
from robosuite.wrappers import Wrapper

import robocasa.models.scenes.scene_registry as SceneRegistry


Key = None
Listener = None


def _load_keyboard_listener():
    global Key, Listener
    if Key is not None and Listener is not None:
        return Key, Listener
    try:
        from pynput.keyboard import Key as PynputKey
        from pynput.keyboard import Listener as PynputListener
    except ImportError as exc:
        print(f"WARNING: enclosing wall hotkeys disabled: {exc}")
        return None
    Key = PynputKey
    Listener = PynputListener
    return Key, Listener


def _unwrap_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _find_enclosing_wall_render_wrapper(env) -> Optional["EnclosingWallRenderWrapper"]:
    """Return the first EnclosingWallRenderWrapper in a Wrapper chain (if any)."""
    base = env
    while hasattr(base, "env"):
        if isinstance(base, EnclosingWallRenderWrapper):
            return base
        base = base.env
    return base if isinstance(base, EnclosingWallRenderWrapper) else None


def _refresh_visualization_and_redraw(env, *, render: bool):
    """Best-effort redraw after mutating wall alpha.

    Some robosuite setups only apply visualization changes on explicit `env.visualize(...)`,
    and mjviewer often redraws via `viewer.update()` inside the main loop. This helper makes
    a best-effort attempt to update both without stepping the sim.
    """
    # Apply (or re-apply) visualize settings if possible
    vis_settings = getattr(env, "vis_settings", None)
    if not vis_settings and hasattr(env, "_visualizations"):
        vis_settings = {vis: True for vis in getattr(env, "_visualizations")}
    if vis_settings and hasattr(env, "visualize"):
        env.visualize(vis_settings)

    # Force a redraw without stepping
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        viewer.update()
    elif render and hasattr(env, "render"):
        env.render()


class EnclosingWallHotkeyHandler:
    """Consume `install_enclosing_wall_hotkeys()` requests and apply wall changes.

    Typical usage inside an interactive control loop:

        handler = EnclosingWallHotkeyHandler(env)
        while True:
            if handler.consume_pending(render=render):
                continue
            # ... normal control loop ...
    """

    def __init__(self, env):
        self.env = env
        self._wrapper = _find_enclosing_wall_render_wrapper(env)

    def consume_pending(self, *, render: bool) -> bool:
        """Apply and clear any queued wall hotkey requests.

        Returns:
            True if a request was consumed (callers typically `continue` their loop).
        """
        force_opaque = bool(getattr(self.env, "_force_enclosing_walls_opaque", False))
        toggle = bool(getattr(self.env, "_toggle_enclosing_walls", False))
        if not force_opaque and not toggle:
            return False

        if force_opaque:
            if self._wrapper is not None:
                self._wrapper.set_enabled(False)
            self.env._force_enclosing_walls_opaque = False
            # If a toggle was also queued, drop it (force-off wins)
            self.env._toggle_enclosing_walls = False
            _refresh_visualization_and_redraw(self.env, render=render)
            return True

        # toggle
        if self._wrapper is not None:
            self._wrapper.toggle()
        self.env._toggle_enclosing_walls = False
        _refresh_visualization_and_redraw(self.env, render=render)
        return True


class EnclosingWallRenderWrapper(Wrapper):
    """Visualization-only transparency override for enclosing walls.

    In interactive viewers (e.g. robosuite's `mjviewer`), the renderer can redraw the scene
    independently of `env.render()` / `env.step()`. To make the effect stable across camera
    switches and user interactions, we apply a persistent alpha override to
    `sim.model.geom_rgba[..., 3]` while enabled, and restore the original alpha values
    when disabled.

    This is still "demo safe" in `collect_demos.py` because demos save MJCF from `env.model.get_xml()`
    (not the live `sim.model` values we adjust here), and state/action logging is unchanged.
    """

    def __init__(self, env, alpha: float = 0.1, enabled: bool = False):
        super().__init__(env)
        self.alpha = float(alpha)
        self.enabled = False
        self._geom_ids: Optional[np.ndarray] = None
        self._saved_alpha: Dict[int, float] = {}
        self._last_model_ptr = None
        self.set_enabled(bool(enabled))

    def __setattr__(self, name, value):
        # Forward non-internal attribute sets to the underlying env when possible.
        # This matters because robosuite Wrapper only forwards __getattr__ (not __setattr__),
        # and callers commonly do things like `env.layout_and_style_ids = ...`.
        if name.startswith("_") or name in {"env", "alpha", "enabled"}:
            object.__setattr__(self, name, value)
            return

        if "env" in self.__dict__ and hasattr(self.__dict__["env"], name):
            setattr(self.__dict__["env"], name, value)
            return

        object.__setattr__(self, name, value)

    def toggle(self):
        self.set_enabled(not self.enabled)

    def set_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self._apply_or_restore()

    def visualize(self, vis_settings):
        """
        Delegates visualization to the wrapped env, then applies / restores the enclosing-wall
        transparency override. This keeps wrapper-specific rendering logic out of the base env.
        """
        self.env.visualize(vis_settings)
        self._apply_or_restore()

    def reset(self, *args, **kwargs):
        # Reset underlying env first, then refresh geom ids for the new model and re-apply if needed
        ret = self.env.reset(*args, **kwargs)
        self._geom_ids = None
        self._saved_alpha = {}
        self._last_model_ptr = None
        if self.enabled:
            self._apply_or_restore()
        return ret

    def _apply_or_restore(self):
        base = _unwrap_env(self.env)
        if not hasattr(base, "sim") or base.sim is None:
            return

        model = getattr(base.sim, "model", None)
        if model is None:
            return

        # detect model reloads (hard resets) so we don't reuse stale ids / alpha
        model_ptr = id(model)
        if self._last_model_ptr != model_ptr:
            self._geom_ids = None
            self._saved_alpha = {}
            self._last_model_ptr = model_ptr

        geom_ids = self._get_enclosing_wall_geom_ids()
        if geom_ids.size == 0:
            return

        if self.enabled:
            # save originals once
            for gi in geom_ids.tolist():
                if int(gi) not in self._saved_alpha:
                    self._saved_alpha[int(gi)] = float(model.geom_rgba[int(gi), 3])
            model.geom_rgba[geom_ids, 3] = self.alpha
        else:
            # force walls fully opaque when disabled
            model.geom_rgba[geom_ids, 3] = 1.0
            self._saved_alpha = {}

    @staticmethod
    def _get_enclosing_wall_names_from_layout(layout_id: int) -> list[str]:
        layout_path = SceneRegistry.get_layout_path(layout_id)
        with open(layout_path, "r") as f:
            layout_data = yaml.safe_load(f) or {}
        walls = (layout_data.get("room") or {}).get("walls") or []
        return [w.get("name") for w in walls if w.get("enclosing_wall", False) is True]

    @staticmethod
    def _match_geom_ids(model, enclosing_names: list[str]) -> np.ndarray:
        """Return MuJoCo geom indices for all geometry belonging to enclosing walls.

        Used to know which geoms to make transparent when the wrapper is enabled.
        Matching is done in two steps:

        1. Find body IDs: bodies whose name equals or is a variant of an enclosing
           wall name (exact match, or prefix/suffix with underscore), then add all
           descendant bodies in the model tree so child meshes are included.
        2. Find geom IDs: geoms that belong to those bodies, or whose geom name
           matches an enclosing wall name (same name rules as above).

        Args:
            model: MuJoCo model (e.g. sim.model).
            enclosing_names: Wall names marked as enclosing in the layout YAML.

        Returns:
            1D array of geom indices (dtype int) for enclosing wall geometry.
        """
        if not enclosing_names:
            return np.array([], dtype=int)

        # Collect bodies that match enclosing wall names (exact or name_prefix_ / _name_suffix)
        body_ids = set()
        for b in range(model.nbody):
            body_name = model.body_id2name(b)
            if body_name is None:
                continue
            for ename in enclosing_names:
                if (
                    body_name == ename
                    or body_name.startswith(ename + "_")
                    or body_name.endswith("_" + ename)
                ):
                    body_ids.add(b)
                    break

        # Add all descendant bodies so child bodies/geoms (e.g. sub-meshes) are included
        added = True
        while added:
            added = False
            for b in range(model.nbody):
                if b in body_ids:
                    continue
                parent = model.body_parentid[b]
                if parent >= 0 and parent in body_ids:
                    body_ids.add(b)
                    added = True

        # Geoms: include if attached to a matched body, or if geom name matches an enclosing name
        geom_ids: list[int] = []
        for i in range(model.ngeom):
            if model.geom_bodyid[i] in body_ids:
                geom_ids.append(i)
                continue
            gname = model.geom_id2name(i)
            if gname is None:
                continue
            for ename in enclosing_names:
                if (
                    gname == ename
                    or gname.startswith(ename + "_")
                    or gname.endswith("_" + ename)
                ):
                    geom_ids.append(i)
                    break
        return np.array(geom_ids, dtype=int)

    def _get_enclosing_wall_geom_ids(self) -> np.ndarray:
        if self._geom_ids is not None:
            return self._geom_ids

        base = _unwrap_env(self.env)
        if (
            not hasattr(base, "sim")
            or base.sim is None
            or not hasattr(base, "layout_id")
        ):
            self._geom_ids = np.array([], dtype=int)
            return self._geom_ids

        enclosing_names = self._get_enclosing_wall_names_from_layout(
            int(base.layout_id)
        )
        model = getattr(base.sim, "model", None)
        if model is None:
            self._geom_ids = np.array([], dtype=int)
            return self._geom_ids

        self._geom_ids = self._match_geom_ids(model, enclosing_names)
        return self._geom_ids


def install_enclosing_wall_hotkeys(env):
    """Install hotkeys that request a wall toggle on `env`.

    Hotkeys:
      - Esc: toggle transparency on / off
      - [ or ]: force transparency OFF (useful since mjviewer uses these for camera cycling)

    Used by interactive loops that call `EnclosingWallHotkeyHandler.consume_pending()`.
    """

    # idempotent-ish: if already installed, no-op
    if getattr(env, "_enclosing_wall_key_listener", None) is not None:
        return

    keyboard_listener = _load_keyboard_listener()
    if keyboard_listener is None:
        env._enclosing_wall_key_listener = False
        return
    key_cls, listener_cls = keyboard_listener

    env._toggle_enclosing_walls = False
    env._force_enclosing_walls_opaque = False

    def _on_release(key):
        if key == key_cls.esc:
            env._toggle_enclosing_walls = True
            return
        if hasattr(key, "char") and key.char in {"[", "]"}:
            env._force_enclosing_walls_opaque = True
            return

    env._enclosing_wall_key_listener = listener_cls(on_release=_on_release)
    env._enclosing_wall_key_listener.start()
