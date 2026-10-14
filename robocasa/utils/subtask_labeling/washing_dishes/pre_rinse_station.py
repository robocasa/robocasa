import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Two items are on the counter; place one in the LEFT basin and the other in the RIGHT basin
# of a double-basin sink (a pre-rinse setup). Counter -> sink is the bank atomic
# PickPlaceCounterToSink. Doc-canonical count = 2. Basin side is detected, not hard-coded.
SUBTASK_MOVE = "PickPlaceCounterToSink"

DONE_LANG = "task complete"

OBJS = ("stackable", "receptacle")


@register_labeler
class PreRinseStationLabeler(BaseSubtaskLabeler):
    """Labeler for the PreRinseStation composite task.

    Task: two items sit on the counter; put one in the left basin and the other in the right
    basin of the sink. Doc-canonical count = 2 (two counter->sink moves). Order by first
    grasped move; the destination basin ("left"/"right") is read from the sink at the place.

    pick/place split at each object's first sustained displacement (cursor-relative rest);
    place ends at the gripper release.
    """

    ENV_NAME = "PreRinseStation"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.env = env
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.obj_lang = {}
        for o in self.names:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "item"
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
            basin = "none"
            if self.sink is not None:
                try:
                    basin = self.sink.get_obj_basin_loc(env, o, partial_check=True)
                except Exception:
                    basin = "none"
            d[o + "_basin"] = basin
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
        basins = {o: [h[o + "_basin"] for h in history] for o in self.names}
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

        def basin_at(o, t):
            # the side the object settles in: majority vote over the last frames before/at t
            lo = max(t - 20, 0)
            vals = [b for b in basins[o][lo: t + 1] if b in ("left", "right")]
            if not vals:
                vals = [b for b in basins[o] if b in ("left", "right")]
            if not vals:
                return None
            return max(set(vals), key=vals.count)

        n = 0
        for o in order:
            w = self.obj_lang[o]
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
            side = basin_at(o, t_placed)
            dest = f"the {side} basin of the sink" if side else "the sink"
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {w} from the counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {w} in {dest}")
            n += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n != 2:
            review.append(f"only {n}/2 items moved")
        placed_sides = [basin_at(o, T - 1) for o in order]
        if any(s is None for s in placed_sides):
            review.append(f"an item did not settle in a detectable basin: {placed_sides}")
        result = self.build_result(T, segments, debug={"order": order, "n": n,
                                                       "sides": placed_sides, "n_segments": len(segments)})
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
            issues.append("expected 2 picks")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append("expected 2 places")
        return issues
