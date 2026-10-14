import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Clear non-produce objects out of the sink onto the counter (to make room for washing
# produce). Sink -> counter is the bank atomic PickPlaceSinkToCounter. The produce items
# (food_item*) stay in the sink and are not moved. Doc count = 2 (one object's pick + place).
SUBTASK_MOVE = "PickPlaceSinkToCounter"

DONE_LANG = "task complete"

# objects that may be cleared from the sink (food_item* usually stay -> only movers emit)
OBJS = ("obj0", "obj1", "food_item0", "food_item1")


@register_labeler
class ClearSinkSpaceLabeler(BaseSubtaskLabeler):
    """Labeler for the ClearSinkSpace composite task.

    Task (non-mobile): move the clutter object(s) out of the sink onto the counter; leave the
    produce in the sink. Only objects that actually move are emitted (ordered by first move);
    each is a PickPlaceSinkToCounter. Same-type movers read "a X"/"the other X".
    """

    ENV_NAME = "ClearSinkSpace"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "item"
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

        from collections import Counter
        counts = Counter(self.base[o] for o in order)
        seen, noun = {}, {}
        for o in order:
            w = self.base[o]
            if counts[w] <= 1:
                noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n = 0
        for o in order:
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
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the counter")
            n += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n < 1:
            review.append("no object cleared from the sink")
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
        npick = int(np.sum(stages == STAGE_PICK))
        nplace = int(np.sum(stages == STAGE_PLACE))
        if npick != nplace:
            issues.append(f"pick/place mismatch ({npick} vs {nplace})")
        if npick < 1:
            issues.append("expected at least 1 cleared object")
        return issues
