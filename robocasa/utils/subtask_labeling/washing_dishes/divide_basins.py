import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A double-basin sink. An item sits in the LEFT basin (to be moved to the RIGHT basin for
# drying) and a drinkware item sits on the counter (to go into the LEFT basin for washing).
# Move the drying item left->right basin, then the wash item counter->left basin. The
# basin->basin move has no bank name -> extension PickPlaceSinkToSink; counter->basin is
# PickPlaceCounterToSink (bank). The drying move is CONDITIONAL (skipped when nothing starts
# in the left basin). Doc-canonical count = 2.
SUBTASK_DRY = "PickPlaceSinkToSink"      # obj_drying: left basin -> right basin
SUBTASK_WASH = "PickPlaceCounterToSink"  # obj_to_wash: counter -> left basin

DONE_LANG = "task complete"


@register_labeler
class DivideBasinsLabeler(BaseSubtaskLabeler):
    """Labeler for the DivideBasins composite task.

    Task: on a double-basin sink, move any item in the LEFT basin over to the RIGHT basin
    (for drying), then move a drinkware item from the counter into the LEFT basin (for
    washing). The drying move is conditional on an item actually being in the left basin at
    the start. Doc-canonical count = 2.

    Per object the pick/place split is at the first sustained displacement; the drying item
    is only emitted if it is actually relocated.
    """

    ENV_NAME = "DivideBasins"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in ("obj_drying", "obj_to_wash") if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.names:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "item"
        self.extra_subtasks = {SUBTASK_DRY}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

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
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.names}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.names}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        first_move = {}
        for o in self.names:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_wash = 0
        for o in order:
            w = self.obj_lang[o]
            t_move = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            if o == "obj_drying":
                emit(t_move, SUBTASK_DRY, STAGE_PICK, f"pick up the {w} from the left basin")
                emit(t_placed, SUBTASK_DRY, STAGE_PLACE, f"place the {w} in the right basin")
            else:
                emit(t_move, SUBTASK_WASH, STAGE_PICK, f"pick up the {w} from the counter")
                emit(t_placed, SUBTASK_WASH, STAGE_PLACE, f"place the {w} in the left basin of the sink")
                n_wash += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_wash != 1:
            review.append(f"expected 1 wash-item move, found {n_wash}")
        result = self.build_result(T, segments, debug={"order": order, "n_wash": n_wash,
                                                       "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_DRY}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_WASH) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 wash-item pick")
        if int(np.sum((names == SUBTASK_WASH) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 wash-item place")
        return issues
