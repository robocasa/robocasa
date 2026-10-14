import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Move the mug from the counter into the (already-open) cabinet, handle facing right, like
# the mugs already inside. Counter -> cabinet is the bank atomic PickPlaceCounterToCabinet.
# Fixed base; no nav, no door (the cabinet stays open).
SUBTASK_MOVE = "PickPlaceCounterToCabinet"

DONE_LANG = "task complete"


@register_labeler
class OrganizeMugsByHandleLabeler(BaseSubtaskLabeler):
    """Labeler for the OrganizeMugsByHandle composite task.

    Task (fixed-base): a mug sits on the counter; two mugs are already in the open cabinet with
    their handles to the right. Put the counter mug in the cabinet with its handle to the right.
    The robot starts at the cabinet; the move is local.

    Doc-canonical decomposition (2 stages: one counter->cabinet move):
      PickPlaceCounterToCabinet [pick]  -> pick up the mug from the counter
      PickPlaceCounterToCabinet [place] -> place the mug in the cabinet with the handle to the right
    plus a terminal done.

    Boundaries: t_pick = the mug first displaces from its rest; t_place = the gripper RELEASE.
    """

    ENV_NAME = "OrganizeMugsByHandle"

    MUG = "mug_counter1"
    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id[self.MUG]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }

    @staticmethod
    def _sustained_start(mask, start, sustain):
        run = 0
        for t in range(int(start), len(mask)):
            if mask[t]:
                run += 1
                if run >= sustain:
                    return t - sustain + 1
            else:
                run = 0
        return None

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        gd = np.linalg.norm(grip - obj, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the mug from the counter")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, "place the mug in the cabinet with the handle to the right")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        return self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_place": int(t_place), "n_segments": len(segments),
        })

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append(f"expected 1 pick, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append(f"expected 1 place, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
