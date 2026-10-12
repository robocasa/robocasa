"""Navigation goal annotation confined to the operator's passive viewer.

Never add sites/geoms to the simulation model or recolor its materials: the
camera publisher renders that same model for LeRobot. Passive-viewer user_scn
geometries and text are independent of both those cameras and raw state/XML.
"""

import warnings

import mujoco
import numpy as np


class NavigationTargetOverlay:
    """Show a static fixture outline and arrow once per episode/viewer.

    Call update after stepping so robosuite can create its passive viewer first.
    Hard resets replace the model, fixture, and viewer; all three are checked.
    """

    _GEOM_COUNT = 13  # twelve bounding-box edges and a downward arrow
    _COLOR = np.array([1.0, 0.75, 0.05, 1.0], dtype=np.float32)

    def __init__(self):
        self._viewer = None
        self._model = None
        self._fixture = None
        self._start = None

    @staticmethod
    def _corners(fixture):
        points = np.asarray(fixture.get_bbox_points(), dtype=np.float64)
        if points.shape != (8, 3) or not np.isfinite(points).all():
            raise ValueError("navigation target has an invalid bounding box")
        # Fixture orders its first four corners as p0, px, py, pz.
        axes = points[1:4] - points[0]
        lengths = np.linalg.norm(axes, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("navigation target has an empty bounding box")
        padding = 0.02 * axes / lengths[:, None]
        origin = points[0] - padding.sum(axis=0)
        axes = axes + 2 * padding
        return np.asarray([
            origin + sum((bool(index & (1 << axis)) * axes[axis] for axis in range(3)))
            for index in range(8)
        ])

    @classmethod
    def _connector(cls, geom, kind, width, start, end):
        mujoco.mjv_initGeom(
            geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), cls._COLOR,
        )
        mujoco.mjv_connector(geom, kind, width, start, end)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0.6

    def update(self, env):
        """Attach to a newly opened viewer or a newly sampled goal, if available."""
        renderer = getattr(env, "viewer", None)
        viewer = getattr(renderer, "viewer", None)
        fixture = getattr(env, "target_fixture", None)
        if viewer is None or fixture is None:
            return False
        model = env.sim.model
        if viewer is self._viewer and model is self._model and fixture is self._fixture:
            return False

        self.close()
        self._viewer, self._model, self._fixture = viewer, model, fixture
        scene = viewer.user_scn
        if scene is None:
            warnings.warn("Navigation target annotation requires a passive viewer user_scn.")
            return False
        try:
            corners = self._corners(fixture)
        except (AttributeError, ValueError) as error:
            warnings.warn(f"Cannot outline navigation target: {error}")
            return False

        with viewer.lock():
            start = scene.ngeom
            if start + self._GEOM_COUNT > scene.maxgeom:
                warnings.warn("Navigation target annotation: viewer geometry buffer is full.")
                return False
            index = start
            for corner in range(8):
                for axis in range(3):
                    other = corner ^ (1 << axis)
                    if other > corner:
                        self._connector(
                            scene.geoms[index], mujoco.mjtGeom.mjGEOM_CAPSULE,
                            0.008, corners[corner], corners[other],
                        )
                        index += 1
            tip = corners.mean(axis=0)
            tip[2] = corners[:, 2].max() + 0.03
            self._connector(
                scene.geoms[index], mujoco.mjtGeom.mjGEOM_ARROW,
                0.025, tip + [0.0, 0.0, 0.35], tip,
            )
            scene.geoms[index].label = f"TARGET: {fixture.nat_lang}"[:99]
            scene.ngeom = start + self._GEOM_COUNT
            self._start = start

        # Text and user_scn belong to this window, never to the camera renderer.
        viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Navigation target:", fixture.nat_lang,
        ))
        viewer.sync()
        return True

    def close(self):
        """Remove only our geometry; leave other viewer decorations intact."""
        viewer = self._viewer
        if viewer is not None and self._start is not None and viewer.is_running():
            with viewer.lock():
                scene = viewer.user_scn
                end = self._start + self._GEOM_COUNT
                for index in range(self._start, min(end, scene.ngeom)):
                    scene.geoms[index].rgba[3] = 0
                    scene.geoms[index].label = ""
                if scene.ngeom == end:
                    scene.ngeom = self._start
            viewer.clear_texts()
            viewer.sync()
        self._viewer = self._model = self._fixture = self._start = None
