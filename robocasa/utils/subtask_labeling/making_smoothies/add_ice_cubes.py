import numpy as np
from robosuite.utils import transform_utils as T

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Bank atomic task: ice cubes go from the bowl on the counter into the blender.
SUBTASK = "PickPlaceCounterToBlender"
PICK_LANG = "pick up an ice cube from the bowl"
PLACE_LANG = "place the ice cube in the blender"

# Bank-extension skills for the bowl-carry strategy: lift the whole bowl and tilt
# it to pour its contents in. Two distinct subtasks, "Pick" then "Tilt" (neither
# is an official bank atomic task; both declared in extra_subtasks below).
PICK_SUBTASK = "Pick"
PICK_BOWL_LANG = "pick up the bowl of ice cubes"
TILT_SUBTASK = "Tilt"
TILT_LANG = "tilt the bowl to pour the ice cubes into the blender"

DONE_LANG = "task complete"


@register_labeler
class AddIceCubesLabeler(BaseSubtaskLabeler):
    """Labeler for the AddIceCubes composite task.

    The robot moves N ice cubes from a bowl on the counter into the blender.
    Teleoperators use one of two strategies, which this labeler distinguishes
    automatically:

    1. Per-cube transport: pick and place each cube individually. Each cube is
       one bank atomic task, PickPlaceCounterToBlender, split into pick/place
       stages -> 2N+1 segments.

    2. Bowl-carry / tilt-pour: lift the whole bowl, carry it to the blender, and
       tilt it to dump every cube at once. Labeled Pick -> Tilt -> done (3
       segments) with the bank-extension skill "Tilt", and attached as a soft
       review note (the strategy is valid but worth a human confirmation).

    Strategy detection is purely geometric: the tilt strategy is the one where
    the bowl's orientation rotates far from its resting pose (>45 deg) while the
    bowl is lifted off the counter. On this dataset that signal separates the
    two strategies with a huge margin (tilt-pours rotate 106-158 deg; per-cube
    demos never exceed ~30 deg, and never while lifted).

    Per-cube labels are derived purely from cube trajectories (each cube's own
    start pose in the bowl and final pose in the blender), so there are no stale
    fixture references and no contact-based grasp checks that false-positive on
    neighbouring cubes sharing the bowl.
    """

    ENV_NAME = "AddIceCubes"

    # --- per-cube transport thresholds ---
    # Height above its resting pose for a cube to count as lifted off the bowl.
    LIFT_TOL = 0.04
    # A cube within this distance of its final pose counts as delivered.
    PLACED_TOL = 0.08
    # Gripper-cube distance above which the gripper counts as released.
    RELEASE_TH = 0.12

    # --- tilt-pour detection thresholds ---
    # Degrees the bowl's up-axis must rotate from rest to count as tilting.
    POUR_TILT_TH = 45.0
    # Height the bowl must rise above its resting pose to count as lifted.
    POUR_LIFT_TH = 0.05
    # Minimum sustained tilted-while-lifted frames to call it a tilt-pour
    # (rejects a single-frame orientation blip).
    POUR_MIN_FRAMES = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.num_cubes = int(ep_meta["refs"]["num_ice_cubes"])
        self.cube_names = [f"ice_cube{i}" for i in range(self.num_cubes)]
        self.bowl_body_id = env.obj_body_id["bowl"]
        # Default to the per-cube expectation; resolve() overrides it to 2 if it
        # detects the tilt-pour strategy.
        self.expected_done_idx = 2 * self.num_cubes
        # "Pick"/"Tilt" are bank-extension skills, not official atomic tasks.
        self.extra_subtasks = {PICK_SUBTASK, TILT_SUBTASK}

    def per_step(self, env, t: int) -> dict:
        cube_pos = np.stack(
            [env.sim.data.body_xpos[env.obj_body_id[n]] for n in self.cube_names]
        )
        gripper_pos = np.array(
            env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        )
        # Bowl up-axis (its local +z in world frame) and height, for tilt-pour
        # detection. mujoco body_xquat is wxyz; transform_utils wants xyzw.
        q = np.array(env.sim.data.body_xquat[self.bowl_body_id])
        bowl_up = T.quat2mat(q[[1, 2, 3, 0]])[:, 2].copy()
        bowl_z = float(env.sim.data.body_xpos[self.bowl_body_id][2])
        return {
            "cube_pos": cube_pos,
            "gripper_pos": gripper_pos,
            "bowl_up": bowl_up,
            "bowl_z": bowl_z,
        }

    def resolve(self, history):
        T_ = len(history)
        pos = np.stack([h["cube_pos"] for h in history])  # (T, N, 3)
        gpos = np.stack([h["gripper_pos"] for h in history])  # (T, 3)
        bowl_up = np.stack([h["bowl_up"] for h in history])  # (T, 3)
        bowl_z = np.array([h["bowl_z"] for h in history])  # (T,)

        # Tilt-pour detection: bowl rotated far from rest AND lifted off counter.
        tilt = np.degrees(np.arccos(np.clip(bowl_up @ bowl_up[0], -1.0, 1.0)))
        lifted = (bowl_z - bowl_z[0]) > self.POUR_LIFT_TH
        pour_mask = (tilt > self.POUR_TILT_TH) & lifted
        if int(pour_mask.sum()) >= self.POUR_MIN_FRAMES:
            return self._resolve_tilt(T_, pour_mask, tilt, lifted)
        return self._resolve_per_cube(T_, pos, gpos)

    def _resolve_tilt(self, T_, pour_mask, tilt, lifted):
        frames = np.where(pour_mask)[0]
        t_pour = int(frames[0])  # first frame clearly tilted (>POUR_TILT_TH)
        # Pick -> Tilt boundary at bowl LIFT-OFF, so the "Tilt" (deliver) instruction
        # leads the whole carry + pour -- preemptive, and consistent with how
        # pick-and-place labels the transport under the 'place' instruction (pick
        # ends at lift-off). Back up from the pour to the last on-counter frame.
        grounded = np.where(~lifted[:t_pour])[0]
        t_tilt_start = int(grounded[-1]) + 1 if grounded.size else 0
        # tilt -> done: bowl returns upright, or the recording ends. These demos
        # often stop the instant success is detected (mid-pour), so cap the tilt
        # segment at T-1 to guarantee a terminal 'done' frame.
        t_tilt_end = min(int(frames[-1]) + 1, T_ - 1)
        self.expected_done_idx = 2  # Pick, Tilt, done
        segments = [
            (t_tilt_start, PICK_SUBTASK, STAGE_EXECUTE, PICK_BOWL_LANG),
            (t_tilt_end, TILT_SUBTASK, STAGE_TILT, TILT_LANG),
            (T_, DONE, STAGE_DONE, DONE_LANG),
        ]
        result = self.build_result(
            T_,
            segments,
            debug={
                "strategy": "tilt",
                "t_tilt_start": t_tilt_start,
                "t_tilt_end": t_tilt_end,
                "max_tilt": float(tilt.max()),
            },
        )
        result.review = [
            f"tilt-pour auto-detected (max bowl tilt {tilt.max():.0f} deg); "
            f"labeled Pick->Tilt -- confirm"
        ]
        return result

    def _resolve_per_cube(self, T_, pos, gpos):
        N = self.num_cubes
        start = pos[0]  # (N, 3) -- resting in the bowl
        final = pos[-1]  # (N, 3) -- resting in the blender

        t_depart = np.zeros(N, dtype=int)
        t_release = np.full(N, T_, dtype=int)
        for i in range(N):
            d_final = np.linalg.norm(pos[:, i, :] - final[i], axis=1)
            d_grip = np.linalg.norm(pos[:, i, :] - gpos, axis=1)
            # t_release: first frame the cube is at the blender (within
            # PLACED_TOL of its final pose) and the gripper has let go.
            delivered = (d_final < self.PLACED_TOL) & (d_grip > self.RELEASE_TH)
            rel = np.where(delivered)[0]
            t_release[i] = int(rel[0]) if rel.size else T_
            # t_depart: start of the final contiguous run (before t_release) in
            # which the cube stays lifted off the bowl floor. Taking the *last*
            # such run ignores transient bumps from handling other cubes.
            lifted = pos[:, i, 2] > start[i, 2] + self.LIFT_TOL
            window = lifted[: t_release[i]]
            if window.any():
                grounded = np.where(~window)[0]
                t_depart[i] = int(grounded[-1]) + 1 if grounded.size else 0
            else:
                t_depart[i] = int(t_release[i])

        # Handling order: cubes are released one after another.
        order = sorted(range(N), key=lambda i: (t_release[i], t_depart[i]))

        segments = []
        for cube in order:
            segments.append((t_depart[cube], SUBTASK, STAGE_PICK, PICK_LANG))
            segments.append((t_release[cube], SUBTASK, STAGE_PLACE, PLACE_LANG))
        segments.append((T_, DONE, STAGE_DONE, DONE_LANG))

        return self.build_result(
            T_,
            segments,
            debug={
                "strategy": "per_cube",
                "t_depart": t_depart,
                "t_release": t_release,
                "order": np.asarray(order),
            },
        )
