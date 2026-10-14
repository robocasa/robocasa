import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A mug sits on the counter next to an (already-open) cabinet. Pick it up and place it
# upside down in the cabinet to dry. One counter->cabinet pick-place (mobile: No); the
# cabinet is pre-opened (no OpenCabinet). PickPlaceCounterToCabinet is a bank atomic.
# (Two distractor objects, one on the counter and one in the cabinet, are ignored.)
SUBTASK_MOVE = "PickPlaceCounterToCabinet"

DONE_LANG = "task complete"


@register_labeler
class DryDrinkwareLabeler(BaseSubtaskLabeler):
    """Labeler for the DryDrinkware composite task.

    Task (non-mobile): a mug is on the counter beside an open cabinet. Pick it up and set it
    upside down in the cabinet. Doc-canonical count = 2 (the pick/place stages of the one
    PickPlaceCounterToCabinet move). pick + place + done.

    pick/place split at the mug's first sustained displacement; place = lift + carry + invert
    + set-down + release.
    """

    ENV_NAME = "DryDrinkware"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["mug"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.mug = env.get_obj_lang("mug")
        except Exception:
            self.mug = "mug"
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

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

        t_move = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_move = int(mv[0]) if mv.size else max(1, T // 2)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.mug} from the counter")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.mug} upside down in the open cabinet")
        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1
        return self.build_result(T, segments, debug={
            "t_move": int(t_move), "t_placed": int(t_placed), "n_segments": len(segments)})

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 mug pick")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 mug place")
        return issues
