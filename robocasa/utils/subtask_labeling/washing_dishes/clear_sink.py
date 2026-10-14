import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Two food items are in the sink; a bowl and a cup are on the counter. Take the food out of
# the sink (onto the counter), then move the bowl and cup from the counter into the sink for
# washing. Non-mobile. Food moves are PickPlaceSinkToCounter; dish moves are
# PickPlaceCounterToSink (both bank atomics). Doc-canonical count = 4.
SUBTASK_OUT = "PickPlaceSinkToCounter"   # food: sink -> counter
SUBTASK_IN = "PickPlaceCounterToSink"    # dish: counter -> sink

DONE_LANG = "task complete"

FOODS = ("food0", "food1")
DISHES = ("dish0", "dish1")


@register_labeler
class ClearSinkLabeler(BaseSubtaskLabeler):
    """Labeler for the ClearSink composite task.

    Task (non-mobile): two food items sit in the sink; a bowl and a cup are on the counter.
    Remove both food items from the sink onto the counter, then put the bowl and cup into the
    sink. Four pick-places (doc-canonical count = 4), ordered by first move. Food uses
    PickPlaceSinkToCounter; dishes use PickPlaceCounterToSink.

    pick/place split at each object's first sustained displacement.
    """

    ENV_NAME = "ClearSink"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in (FOODS + DISHES) if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.names:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "item"
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    @staticmethod
    def _is_food(o):
        return o.startswith("food")

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
        noun = {}
        for o in order:
            w = self.obj_lang[o]
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
            # Re-base each object's rest at the cursor: an object can be nudged
            # off its start (e.g. the bowl bumped while the sink is cleared) so
            # that its GLOBAL displacement already exceeds MOVE_TOL by the time
            # its own turn comes -- that would collapse the pick to zero frames.
            # A cursor-relative rest makes "moved" mean "moved from where it is
            # sitting now", so the pick spans the real reach+grab.
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dloc = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dloc >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dloc[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            if self._is_food(o):
                emit(t_move, SUBTASK_OUT, STAGE_PICK, f"pick up {noun[o]} from the sink")
                emit(t_placed, SUBTASK_OUT, STAGE_PLACE, f"place {noun[o]} on the counter")
            else:
                emit(t_move, SUBTASK_IN, STAGE_PICK, f"pick up {noun[o]} from the counter")
                emit(t_placed, SUBTASK_IN, STAGE_PLACE, f"place {noun[o]} in the sink")
            n += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n != 4:
            review.append(f"only {n}/4 objects moved")
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
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        picks = int(np.sum(stages == STAGE_PICK))
        places = int(np.sum(stages == STAGE_PLACE))
        if picks != 4:
            issues.append(f"expected 4 picks, found {picks}")
        if places != 4:
            issues.append(f"expected 4 places, found {places}")
        return issues
