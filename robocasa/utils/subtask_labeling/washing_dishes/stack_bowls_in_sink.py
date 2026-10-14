import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Stack the bowls in the sink: move each bowl into the sink, the second onto the first.
# Counter -> sink is the bank atomic PickPlaceCounterToSink. Doc-canonical count = 2 (two
# bowl moves). Non-mobile. Bowls are wide (gripper grabs the rim), so no grasp gate -- order
# and pick split off sustained displacement with a cursor-relative rest (as in ClearSink).
SUBTASK_MOVE = "PickPlaceCounterToSink"

DONE_LANG = "task complete"

BOWLS = ("receptacle1", "receptacle2")


@register_labeler
class StackBowlsInSinkLabeler(BaseSubtaskLabeler):
    """Labeler for the StackBowlsInSink composite task.

    Move both bowls into the sink and stack them (the second bowl ends on top of the first).
    Doc-canonical count = 2. Order by first sustained move; same-type bowls read
    "a bowl"/"the other bowl"; the first placed goes "in the sink", the second "on top".

    pick/place split at each bowl's first sustained displacement (cursor-relative rest);
    place ends at the gripper release.
    """

    ENV_NAME = "StackBowlsInSink"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in BOWLS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "bowl"
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

        # Stacking convention: the larger bowl is the base (picked/placed first), the smaller
        # bowl is stacked on top (picked second). Name by that pick order.
        noun = {}
        for k, o in enumerate(order):
            noun[o] = "the larger bowl" if k == 0 else "the smaller bowl"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n = 0
        for k, o in enumerate(order):
            nn = noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the counter")
            if k == 0:
                emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} in the sink")
            else:
                emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"stack {nn} on top in the sink")
            n += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n != 2:
            review.append(f"only {n}/2 bowls moved")
        result = self.build_result(T, segments, debug={"order": order, "n": n,
                                                       "n_segments": len(segments)})
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
            issues.append("expected 2 bowl picks")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append("expected 2 bowl places")
        return issues
