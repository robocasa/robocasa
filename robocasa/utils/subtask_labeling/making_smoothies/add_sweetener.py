import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# N sugar cubes go from the plate on the counter into the blender, one at a time.
# Counter -> blender has no bank name -> PickPlaceCounterToBlender (extension). Sister
# task to AddIceCubes; unlike the ice-cube bowl, the sugar-cube PLATE is never lifted or
# tilted (verified: max plate tilt ~12 deg, lift ~0.015 m), so there is only the
# per-cube strategy. The base never drives.
SUBTASK = "PickPlaceCounterToBlender"
PICK_LANG = "pick up a sugar cube from the plate"
PLACE_LANG = "place the sugar cube in the blender"

DONE_LANG = "task complete"


@register_labeler
class AddSweetenerLabeler(BaseSubtaskLabeler):
    """Labeler for the AddSweetener composite task.

    The robot moves N (2-3) sugar cubes from a plate on the counter into the blender,
    one at a time. A fruit already sits in the blender (a distractor, never touched).
    The robot starts at the counter; the base never drives.

    Doc-canonical decomposition (one pick-place per cube, split into pick/place stages
    -> 2N+1 segments). Labels are derived purely from each cube's own trajectory (its
    start pose on the plate, final pose in the blender), so neighbouring cubes sharing
    the plate never false-trigger a grasp.

    Per cube: t_release = first frame the cube is at the blender (within PLACED_TOL of
    its final pose) with the gripper let go; t_depart = the start of the LAST contiguous
    run in which the cube stays lifted off the plate before that release (ignores bumps
    from handling the other cubes). Handling ORDER is the release order.
    """

    ENV_NAME = "AddSweetener"

    LIFT_TOL = 0.04
    PLACED_TOL = 0.08
    RELEASE_TH = 0.12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.num_cubes = int(ep_meta["refs"]["num_sugar_cubes"])
        self.cube_names = [f"sugar_cube{i}" for i in range(self.num_cubes)]
        self.cube_body_ids = [env.obj_body_id[n] for n in self.cube_names]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.extra_subtasks = {SUBTASK}
        self.expected_done_idx = 2 * self.num_cubes
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        cube_pos = np.stack([env.sim.data.body_xpos[b] for b in self.cube_body_ids])
        gripper_pos = np.array(env.sim.data.site_xpos[self.eef_site_id])
        return {"cube_pos": cube_pos, "gripper_pos": gripper_pos}

    def resolve(self, history):
        T_ = len(history)
        pos = np.stack([h["cube_pos"] for h in history])   # (T, N, 3)
        gpos = np.stack([h["gripper_pos"] for h in history])  # (T, 3)
        N = self.num_cubes
        start = pos[0]   # (N, 3) resting on the plate
        final = pos[-1]  # (N, 3) resting in the blender

        t_depart = np.zeros(N, dtype=int)
        t_release = np.full(N, T_, dtype=int)
        for i in range(N):
            d_final = np.linalg.norm(pos[:, i, :] - final[i], axis=1)
            d_grip = np.linalg.norm(pos[:, i, :] - gpos, axis=1)
            delivered = (d_final < self.PLACED_TOL) & (d_grip > self.RELEASE_TH)
            rel = np.where(delivered)[0]
            t_release[i] = int(rel[0]) if rel.size else T_
            lifted = pos[:, i, 2] > start[i, 2] + self.LIFT_TOL
            window = lifted[: t_release[i]]
            if window.any():
                grounded = np.where(~window)[0]
                t_depart[i] = int(grounded[-1]) + 1 if grounded.size else 0
            else:
                t_depart[i] = int(t_release[i])

        order = sorted(range(N), key=lambda i: (t_release[i], t_depart[i]))

        segments = []
        for cube in order:
            segments.append((t_depart[cube], SUBTASK, STAGE_PICK, PICK_LANG))
            segments.append((t_release[cube], SUBTASK, STAGE_PLACE, PLACE_LANG))
        segments.append((T_, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        n_delivered = int(np.sum(t_release < T_))
        review = []
        if n_delivered != N:
            review.append(f"only {n_delivered}/{N} sugar cubes delivered")
        result = self.build_result(T_, segments, debug={
            "strategy": "per_cube", "num_cubes": N,
            "t_depart": t_depart, "t_release": t_release,
            "order": np.asarray(order), "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues  # human-reviewed override: skip the per-cube structural checks
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        npi = int(np.sum(stages == STAGE_PICK))
        npl = int(np.sum(stages == STAGE_PLACE))
        if npi != self.num_cubes:
            issues.append(f"expected {self.num_cubes} picks, found {npi}")
        if npl != self.num_cubes:
            issues.append(f"expected {self.num_cubes} places, found {npl}")
        return issues
