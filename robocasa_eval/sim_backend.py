"""Independent RoboCasa / SONIC simulator used by each evaluation worker.

MuJoCo runs at 200 Hz, SONIC decoding and hand commands at 50 Hz. The VLA
receives the live-exporter joint ordering, calibrated RGB cameras, and gravity.
Heavy imports are lazy so mapping tests and launch preflight work without MuJoCo.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np

from .cameras import CAMERAS, normalize_camera_names
from .evaluation import compiler_cache_path


SIM_HZ = 200
POLICY_HZ = 50
SUBSTEPS = SIM_HZ // POLICY_HZ
# Live exporter / Pinocchio *state* is index-first. The action fields contain
# teleop hand commands and already use thumb-first SONIC / DDS motor order.
HAND_MOTOR_TO_DATASET = np.array([3, 4, 5, 6, 0, 1, 2])
HAND_DATASET_TO_MOTOR = np.argsort(HAND_MOTOR_TO_DATASET)


def camera_configuration(config: Mapping) -> tuple[dict[str, str], int, int]:
    """Select policy observations independently of optional video recording."""
    names = normalize_camera_names(config.get("camera_names", ["ego_view"]))
    dimensions = [config.get("camera_height", 480), config.get("camera_width", 640)]
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           or value <= 0 for value in dimensions):
        raise ValueError("camera_height and camera_width must be positive integers")
    return {name: CAMERAS[name] for name in names}, *map(int, dimensions)


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape ({size},), got {a.shape}")
    return a


def hand_motor_to_dataset(value: Any) -> np.ndarray:
    return _finite_vector(value, 7, "hand motor positions")[HAND_MOTOR_TO_DATASET].copy()


def hand_dataset_to_motor(value: Any) -> np.ndarray:
    return _finite_vector(value, 7, "hand dataset positions")[HAND_DATASET_TO_MOTOR].copy()


def state_from_motors(body: Any, left: Any, right: Any, gravity: Any) -> dict:
    """Return named state features, matching the published dataset modality.json."""
    q = _finite_vector(body, 29, "body motor positions")
    result = {
        "left_leg": q[:6].copy(),
        "right_leg": q[6:12].copy(),
        "waist": q[12:15].copy(),
        "left_arm": q[15:22].copy(),
        "right_arm": q[22:29].copy(),
        "left_hand": hand_motor_to_dataset(left),
        "right_hand": hand_motor_to_dataset(right),
        "projected_gravity": _finite_vector(gravity, 3, "projected gravity").copy(),
    }
    return {key: value.astype(np.float32) for key, value in result.items()}


def projected_gravity_from_wxyz(value: Any) -> np.ndarray:
    """R(q).T @ [0, 0, -1], independent of scipy / robot-model packages."""
    q = _finite_vector(value, 4, "base quaternion")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("base quaternion has zero norm")
    w, x, y, z = q / norm
    return np.array(
        [2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1],
        dtype=np.float32,
    )


def model_step_limit(task: Mapping, atomic_seconds: float = 40,
                     composite_seconds: float = 180) -> int:
    category = task.get("category")
    if category not in ("atomic", "composite"):
        raise ValueError(f"Unknown evaluation category: {category!r}")
    default = atomic_seconds if category == "atomic" else composite_seconds
    limit = float(task.get("timeout_seconds", task.get("limit_sec", default)))
    if ("timeout_seconds" in task and "limit_sec" in task
            and not math.isclose(limit, float(task["limit_sec"]), abs_tol=1e-8)):
        raise ValueError("timeout_seconds and limit_sec disagree")
    if not math.isfinite(limit) or limit <= 0 or not math.isclose(
        limit * POLICY_HZ, round(limit * POLICY_HZ), abs_tol=1e-8
    ):
        raise ValueError("limit_sec must be positive and an exact multiple of 1/50 second")
    steps = round(limit * POLICY_HZ)
    if "max_model_steps" in task:
        requested = task["max_model_steps"]
        if (isinstance(requested, bool) or not isinstance(requested, (int, np.integer))
                or requested != steps):
            raise ValueError("max_model_steps must equal timeout_seconds × 50 Hz")
    return steps


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class SonicRoboCasaEnv:
    """One simulator, decoder history, and renderer per spawned CPU worker.

    Required config: ``work_dir`` and ``decoder_path`` (or ``sonic_decoder``).
    Optional: ``renderer`` (osmesa/egl), ``decoder_threads`` (default 1),
    ``layout_ids`` (default -1, the collection layout convention), ``style_ids``,
    ``layout_and_style_ids``, ``obj_instance_split``, ``camera_width/height``.
    No task-success signals or privileged object state are sent to the VLA.
    """

    def __init__(self, config: Mapping | Any):
        self.config = dict(config) if isinstance(config, Mapping) else vars(config).copy()
        self._cameras, self._camera_height, self._camera_width = camera_configuration(self.config)
        if (self.config.get("model_hz", POLICY_HZ) != POLICY_HZ
                or self.config.get("control_hz", SIM_HZ) != SIM_HZ
                or not math.isclose(self.config.get("physics_dt", 1 / SIM_HZ), 1 / SIM_HZ)):
            raise ValueError("SONIC protocol requires model_hz=50, control_hz=200, physics_dt=.005")
        self.work_dir = Path(self.config["work_dir"]).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # MuJoCo's C logger may write MUJOCO_LOG.TXT relative to the process cwd.
        # Every simulator is an isolated worker, so keep even that fallback here.
        os.chdir(self.work_dir)
        for key, subdir in {
            "TMPDIR": "tmp",
            "NUMBA_CACHE_DIR": "cache/numba",
            "XDG_CACHE_HOME": "cache/xdg",
            "MPLCONFIGDIR": "cache/matplotlib",
        }.items():
            target = self.work_dir / subdir
            if key == "NUMBA_CACHE_DIR":
                target = compiler_cache_path(target, os.environ)
            target.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(target)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        renderer = str(self.config.get("renderer", self.config.get("mujoco_gl", "osmesa")))
        if renderer not in ("osmesa", "egl"):
            raise ValueError("renderer must be osmesa or egl")
        os.environ["MUJOCO_GL"] = renderer
        if renderer == "osmesa":
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("MUJOCO_LOG_PATH", str(self.work_dir / "MUJOCO_LOG.TXT"))

        from .sonic_decoder import SonicDecoder
        self.decoder = SonicDecoder(
            self.config.get("decoder_path", self.config.get("sonic_decoder")),
            threads=int(self.config.get("decoder_threads", 1)),
        )
        self.env = self.wall = self.render = self.maps = None
        self._finished = False
        self._success = False
        self._reason = "not_started"
        self._model_steps = self._physics_steps = 0
        self._metadata: dict = {}

    def reset(self, task: Mapping, seed: int) -> dict:
        """Sample a fresh scene from a fixed seed; never replay a training episode."""
        self.close()
        import mujoco
        import robosuite
        import robosuite.macros as macros
        from robosuite.controllers import load_composite_controller_config
        from robosuite.scripts.collect_sonic_g1_demos import match_base_sim_physics, SONIC_CFG
        from robosuite.utils.sonic.controller import G1SonicController
        import robocasa  # noqa: F401: register tasks
        from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper
        from .sonic_decoder import BODY_KP, BODY_KD, HAND_KP, HAND_KD

        if self.config.get("warmup_seconds", 0) != 0:
            raise ValueError("This protocol has no unscored warmup; warmup_seconds must be 0")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.task = dict(task)
        self._limit = model_step_limit(
            task, self.config.get("atomic_seconds", 40),
            self.config.get("composite_seconds", 180),
        )
        self.decoder.reset()
        np.random.seed(int(seed) % 2**32)
        random.seed(int(seed))
        macros.SIMULATION_TIMESTEP = 1 / SIM_HZ
        macros.FILE_LOGGING_LEVEL = None
        # controller=None chooses BASIC in this RoboSuite version even for SonicG1.
        cfg = load_composite_controller_config(controller=SONIC_CFG)
        scene = {}
        pairs = self.config.get("layout_and_style_ids")
        if pairs is not None:
            scene["layout_and_style_ids"] = pairs
        else:
            scene["layout_ids"] = self.config.get("layout_ids", -1)
            scene["style_ids"] = self.config.get("style_ids")
        started = time.monotonic()
        self.env = robosuite.make(
            task["name"], robots=["SonicG1"], controller_configs=cfg,
            initialization_noise=None, has_renderer=False,
            has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False,
            ignore_done=True, horizon=self._limit * SUBSTEPS + 1,
            control_freq=SIM_HZ, mujoco_solver="PGS", seed=int(seed),
            translucent_robot=False, randomize_cameras=False, generative_textures=None,
            obj_instance_split=self.config.get("obj_instance_split"), **scene,
        )
        self.wall = EnclosingWallRenderWrapper(self.env, alpha=0.0, enabled=True)
        # Kitchen defers scene loading to reset; seed before its first sampled scene.
        self.env.rng = np.random.default_rng(int(seed))
        self.env.unset_ep_meta()
        self.wall.reset()
        m = self.env.sim.model._model
        d = self.env.sim.data._data
        match_base_sim_physics(m, floor_friction=1.0, floor_torsion=0.005,
                               timestep=1 / SIM_HZ)
        m.opt.solver = int(mujoco.mjtSolver.mjSOL_PGS)
        self.env.post_action_freq = 10  # the data collector's fixture bookkeeping rate
        self.env.render_freq = 10
        # Do not let a reward cached from a previous reset become a success.
        if hasattr(self.env, "_last_post_action"):
            del self.env._last_post_action
        controller = self.env.robots[0].composite_controller
        controller.set_command_gains({
            "body": (np.asarray(BODY_KP), np.asarray(BODY_KD)),
            "lhand": (np.full(7, HAND_KP), np.full(7, HAND_KD)),
            "rhand": (np.full(7, HAND_KP), np.full(7, HAND_KD)),
        })
        controller.release_band()
        self.maps = G1SonicController(self.env.sim, None, controller._cfg)
        if self.maps._lh is None or self.maps._rh is None or self.maps.free_qadr is None:
            raise RuntimeError("Expected a floating SonicG1 with two 7-motor Dex3 hands")
        self._validate_hand_order(self.maps._lh_names, "left")
        self._validate_hand_order(self.maps._rh_names, "right")
        self._pelvis = next(
            int(m.jnt_bodyid[j]) for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and int(m.jnt_qposadr[j]) == self.maps.free_qadr
        )
        mujoco.mj_forward(m, d)
        if not math.isclose(float(self.env.control_timestep), 1 / SIM_HZ):
            raise RuntimeError("Simulator control timestep differs from the 200 Hz protocol")
        if self.env.action_dim != 43:
            raise RuntimeError(f"Expected 43 motor targets; action_dim={self.env.action_dim}")

        ep_meta = self.env.get_ep_meta()
        self.env.set_ep_meta(ep_meta)  # freeze sampled language / objects during this rollout
        self._instruction = ep_meta.get("lang")
        if not isinstance(self._instruction, str) or not self._instruction.strip():
            raise RuntimeError(f"No sampled language instruction for {task['name']}")
        height, width = self._camera_height, self._camera_width
        self.render = mujoco.Renderer(m, height=height, width=width)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[0] = 0
        self._scene_option.geomgroup[1] = 1
        for camera in self._cameras.values():
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise RuntimeError(f"Missing calibrated camera: {camera}")
        self._model_steps = self._physics_steps = 0
        self._sim_start = float(d.time)
        self._initial_root_height = float(d.qpos[self.maps.free_qadr + 2])
        self._min_root_height = self._initial_root_height
        self._final_root_height = self._initial_root_height
        self._wall_start = time.monotonic()
        self._finished = self._success = False
        self._reason = "running"
        self._metadata = _jsonable({
            "task": task["name"], "category": task["category"],
            "split": task.get("split"), "seed": int(seed),
            "layout_id": self.env.layout_id, "style_id": self.env.style_id,
            "instruction": self._instruction, "episode_metadata": ep_meta,
            "time_limit_seconds": self._limit / POLICY_HZ,
            "sim_hz": SIM_HZ, "decoder_hz": POLICY_HZ, "post_action_hz": 20,
            "warmup_seconds": 0, "startup_band_during_scoring": False,
            "initialization": "fresh seeded SonicG1 reset; decoder history cleared",
            "reset_wall_seconds": time.monotonic() - started,
            "success_criterion": "first task reward > 0 at native 20 Hz bookkeeping",
            "renderer": os.environ["MUJOCO_GL"],
            "rgb_size": [height, width], "rgb_vertical_flip": False,
            "camera_names": self._cameras,
            "left_hand_native_joint_order": self.maps._lh_names,
            "right_hand_native_joint_order": self.maps._rh_names,
            "hand_slot_convention": (
                "match live DDS/model slots; state reorder follows exporter SDK slots; "
                "action slots unchanged"
            ),
            "enclosing_wall_alpha": 0.0,
        })
        return self._observation()

    @staticmethod
    def _validate_hand_order(names: list[str], side: str) -> None:
        # The live sim bridge deliberately preserves native model joint order,
        # even where model index/middle order differs from SDK label ordering.
        # Reordering these anatomical names here would change the collected data
        # convention. Preserve numerical slots, and record names in the report.
        suffixes = [f"{side}_hand_{part}_joint" for part in (
            "thumb_0", "thumb_1", "thumb_2"
        )]
        finger_suffixes = {f"{side}_hand_{part}_joint" for part in (
            "index_0", "index_1", "middle_0", "middle_1"
        )}
        if (len(names) != 7
                or any(not actual.endswith(want) for actual, want in zip(names[:3], suffixes))
                or any(sum(name.endswith(want) for name in names[3:]) != 1
                       for want in finger_suffixes)):
            raise RuntimeError(f"Unexpected {side} hand motor order: {names}")

    def _motor_state(self) -> tuple:
        d = self.env.sim.data._data
        # Match G1SonicController.build_obs -> UnitreeSdk2Bridge.Publish exactly.
        # Freejoint angular qvel is body-local. Derived xquat/cvel can still refer
        # to the pre-integration state immediately after mj_step.
        base_quat = d.qpos[self.maps.free_qadr + 3:self.maps.free_qadr + 7]
        omega = d.qvel[self.maps.free_vadr + 3:self.maps.free_vadr + 6]
        gravity = projected_gravity_from_wxyz(base_quat)
        return (
            d.qpos[self.maps.qpos_adr].copy(), d.qvel[self.maps.qvel_adr].copy(),
            d.qpos[self.maps._lh[0]].copy(), d.qpos[self.maps._rh[0]].copy(),
            omega.copy(), gravity,
        )

    def _observation(self) -> dict:
        body, _, left, right, _, gravity = self._motor_state()
        images = {}
        for key, camera in self._cameras.items():
            self.render.update_scene(self.env.sim.data._data, camera=camera,
                                     scene_option=self._scene_option)
            image = self.render.render()
            if image.dtype != np.uint8:
                raise RuntimeError("MuJoCo RGB renderer returned a non-uint8 image")
            images[key] = np.ascontiguousarray(image).copy()
        return {
            "video": images,
            "state": state_from_motors(body, left, right, gravity),
            "language": self._instruction,
        }

    def advance(self, actions: Mapping[str, Any]) -> dict:
        """Execute up to one VLA chunk, with per-tick closed-loop SONIC decoding."""
        if self.env is None:
            raise RuntimeError("reset() must precede advance()")
        if self._finished:
            return self._result()
        arrays = {}
        for key, dim in (("motion_token", 64), ("left_hand_joints", 7),
                         ("right_hand_joints", 7)):
            a = np.asarray(actions.get(key, actions.get(f"action.{key}")), dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != dim or len(a) == 0 or not np.isfinite(a).all():
                raise ValueError(f"{key} must be finite with shape (T,{dim}), got {a.shape}")
            arrays[key] = a
        if len({len(a) for a in arrays.values()}) != 1:
            raise ValueError("Action chunk lengths differ")
        from .sonic_decoder import clamp_hand_target
        steps = min(len(arrays["motion_token"]), self._limit - self._model_steps)
        for k in range(steps):
            q, dq, left, right, omega, gravity = self._motor_state()
            body_target = self.decoder.step(arrays["motion_token"][k], q, dq, omega, gravity)
            for _ in range(SUBSTEPS):
                # C++ publishes hand commands at 500 Hz; at 200 Hz physics the
                # corresponding measured-position clamp is recomputed each tick.
                d = self.env.sim.data._data
                l_target = clamp_hand_target(arrays["left_hand_joints"][k],
                                             d.qpos[self.maps._lh[0]], is_left=True)
                r_target = clamp_hand_target(arrays["right_hand_joints"][k],
                                             d.qpos[self.maps._rh[0]], is_left=False)
                action = np.concatenate([body_target, l_target, r_target])
                _, reward, _, _ = self.wall.step(action)
                self._physics_steps += 1
                self._final_root_height = float(d.qpos[self.maps.free_qadr + 2])
                self._min_root_height = min(self._min_root_height, self._final_root_height)
                # reward() invokes _check_success; calling it again can mutate task history.
                if float(reward) > 0:
                    self._success = self._finished = True
                    self._reason = "success"
                    break
                if not np.isfinite(self.env.sim.data.qpos).all():
                    raise FloatingPointError("MuJoCo produced non-finite qpos")
            self._model_steps += 1
            if self._finished:
                break
        if not self._finished and self._model_steps >= self._limit:
            self._finished = True
            self._reason = "simulation_time_limit"
        return self._result()

    def _result(self) -> dict:
        sim_seconds = self._physics_steps / SIM_HZ
        if self.env is not None:
            actual_seconds = float(self.env.sim.data._data.time) - self._sim_start
            if not math.isclose(sim_seconds, actual_seconds, abs_tol=1e-6):
                raise RuntimeError(f"Physics time mismatch: expected {sim_seconds}, got {actual_seconds}")
        return {
            "observation": None if self._finished else self._observation(),
            "finished": self._finished, "success": self._success,
            "sim_steps": self._model_steps, "sim_seconds": sim_seconds,
            "reason": self._reason,
            "metadata": {
                **self._metadata, "physics_steps": self._physics_steps,
                "model_steps": self._model_steps,
                "first_physics_step_success": bool(self._success and self._physics_steps == 1),
                "initial_root_height": self._initial_root_height,
                "final_root_height": self._final_root_height,
                "min_root_height": self._min_root_height,
                "root_height_units": "meters; world-frame pelvis z; diagnostic only",
                "rollout_wall_seconds": time.monotonic() - self._wall_start,
            },
        }

    def close(self) -> None:
        if self.render is not None:
            self.render.close()
            self.render = None
        if self.env is not None:
            self.env.close()
            self.env = self.wall = self.maps = None
