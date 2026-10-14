import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Gather two vegetables from the sink onto the cutting board on the counter. Sink -> counter is
# the bank atomic PickPlaceSinkToCounter. Fixed base / local (the board is beside the sink).
SUBTASK_MOVE = "PickPlaceSinkToCounter"

DONE_LANG = "task complete"

VEGS = ("veg1", "veg2")


@register_labeler
class GatherVegetablesLabeler(BaseSubtaskLabeler):
    """Labeler for the GatherVegetables composite task.

    Task (fixed-base): two vegetables sit in the sink; a cutting board sits on the counter beside
    it. Move both vegetables onto the cutting board. The robot starts at the sink.

    Doc-canonical decomposition (2 moves), emitted in first-motion order, each vegetable:
      PickPlaceSinkToCounter [pick]  -> pick up the {veg} from the sink
      PickPlaceSinkToCounter [place] -> place the {veg} on the cutting board
    plus a terminal done. `veg` names come from get_obj_lang.

    Boundaries: t_pick = first sustained displacement from a rest re-anchored at the segment start;
    t_place = the gripper RELEASE.
    """

    ENV_NAME = "GatherVegetables"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in VEGS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: env.get_obj_lang(o) for o in self.names}
        self.extra_subtasks = set()
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
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            word = self.word[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the sink")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} on the cutting board")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 vegetables moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
