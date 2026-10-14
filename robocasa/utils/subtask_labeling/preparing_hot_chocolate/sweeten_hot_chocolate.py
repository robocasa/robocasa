import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Drop the sugar cube(s) into the saucepan of hot chocolate on the stove. The number of cubes
# varies (1-3). Counter -> saucepan on the stove is the bank atomic PickPlaceCounterToStove. Fixed
# base / local (the counter is beside the stove where the robot starts).
SUBTASK_MOVE = "PickPlaceCounterToStove"

DONE_LANG = "task complete"


@register_labeler
class SweetenHotChocolateLabeler(BaseSubtaskLabeler):
    """Labeler for the SweetenHotChocolate composite task.

    Task (fixed-base): one to three sugar cubes sit on the counter; a saucepan of hot chocolate is
    on the stove. Drop each sugar cube into the saucepan. The robot starts at the stove.

    Doc-canonical count = 2 (`mobile: No`, action-level: pick + place). Emitted per cube, in
    first-motion order:
      PickPlaceCounterToStove [pick]  -> pick up {a|another} sugar cube from the counter
      PickPlaceCounterToStove [place] -> place the sugar cube in the saucepan
    plus a terminal done. The number of cubes varies episode to episode (1-3).

    Boundaries: t_pick = first sustained displacement from a rest re-anchored at the segment start;
    t_place = the gripper RELEASE.
    """

    ENV_NAME = "SweetenHotChocolate"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 10
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = sorted([o for o in env.obj_body_id if o.startswith("sugar")],
                            key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 8

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
        for k, o in enumerate(order):
            art = "a" if k == 0 else "another"
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {art} sugar cube from the counter")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the sugar cube in the saucepan")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved < 1:
            review.append("no sugar cubes moved -- verify")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_cubes": len(self.names), "n_segments": len(segments),
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
        npick = int(np.sum(stages == STAGE_PICK))
        nplace = int(np.sum(stages == STAGE_PLACE))
        if npick != nplace:
            issues.append(f"pick/place mismatch: {npick} picks, {nplace} places")
        if npick not in (1, 2, 3):
            issues.append(f"expected 1-3 sugar moves, found {npick}")
        return issues
