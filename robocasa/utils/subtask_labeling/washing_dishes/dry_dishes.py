import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A cup and a bowl sit in the sink; take each out and set it on the counter to dry. Two
# sink->counter pick-places, no navigation (mobile: No). PickPlaceSinkToCounter is a bank
# atomic. (A distractor object on the counter is ignored.)
SUBTASK_MOVE = "PickPlaceSinkToCounter"

DONE_LANG = "task complete"

OBJS = ("obj1", "obj2")


@register_labeler
class DryDishesLabeler(BaseSubtaskLabeler):
    """Labeler for the DryDishes composite task.

    Task (non-mobile): a cup (obj1) and a bowl (obj2) sit in the sink. Pick each out and set
    it on the counter to dry. Doc-canonical count = 2: one PickPlaceSinkToCounter per item.
    Both relocated; order resolved from which moves first. 2 pick-places + done.

    Per object the pick/place split is at the first sustained displacement (pick = reach +
    grasp; place = lift + carry + set-down + release).
    """

    ENV_NAME = "DryDishes"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.names:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "dish"
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
        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.names}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.names}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        first_move = {}
        for o in self.names:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        from collections import Counter
        counts = Counter(self.obj_lang[o] for o in order)
        seen = {}
        disp_noun = {}
        for o in order:
            w = self.obj_lang[o]
            if counts[w] <= 1:
                disp_noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                disp_noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            noun = disp_noun[o]
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
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} on the counter")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 dishes moved")
        result = self.build_result(T, segments, debug={"order": order, "n_moved": n_moved,
                                                       "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 2:
            issues.append("expected 2 dish picks")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 2:
            issues.append("expected 2 dish places")
        return issues
