import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Stack a VARIABLE number of same-category items (cups or bowls, 2-3) from largest to
# smallest on the counter. Counter -> stack has no bank name -> PickPlaceCounterToCounter
# (extension). Fixed base; obj_0 (the largest) is the base of the stack and never moves; the
# robot nests obj_1 on obj_0, then obj_2 on obj_1.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"


@register_labeler
class SizeSortingLabeler(BaseSubtaskLabeler):
    """Labeler for the SizeSorting composite task.

    Task (fixed-base): 2-3 differently-sized items of one category (cup or bowl) sit on the
    counter (where the robot starts). Stack them largest-to-smallest. obj_0 (largest) is the
    stationary base; the robot nests obj_1 onto it, then obj_2 onto obj_1.

    Doc-canonical count = 2 (`mobile: No`, action-level: the stack moves). The number of
    moves VARIES with the item count (1 for 2 items, 2 for 3), so validate() checks coverage
    (equal picks/places, 1-2 moves). Emitted in first-motion order.

    Boundaries per moved item:
      - t_pick (pick -> place) = the LIFT ONSET: the item leaves its re-anchored rest
        (small LIFT_TOL -- the sim replay is deterministic, so a resting item reads exactly
        0 and the onset is noise-free).
      - place-end: the FINAL item runs to the end of the episode (the item is held, set on
        the stack, released, and the arm withdraws -- all "stacking", no long "done" tail).
        A NON-FINAL item ends at the gripper RELEASE (when that item is let go), so the next
        item's pick keeps its reach window. (A held item is maneuvered at low speed for a
        long time, so an object-at-rest test cuts the place early on a transient pause --
        the gripper release is the robust set-down signal.)
    """

    ENV_NAME = "SizeSorting"

    MOVE_TOL = 0.08     # displacement that counts an item as "moved" (ordering / inclusion)
    LIFT_TOL = 0.02     # smaller onset threshold for the pick->place split (lift off rest)
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    SETTLE_TH = 0.004   # m/frame; fallback rest test if a non-final release isn't found
    SETTLE_W = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        # obj_0 is the stationary base; only obj_1.. are moved (stacked).
        self.names = [n for n in (f"obj_{i}" for i in range(1, 8)) if n in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cat = env.get_obj_lang("obj_0")
        except Exception:
            self.cat = "object"
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        # a tight stack-to-stack transition makes a legitimately brief reach ("pick"); the
        # zero-length check still guards against a truly collapsed segment.
        self.min_segment_frames = None

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

    def _settle(self, pos, t_pick, T):
        speed = np.concatenate([np.linalg.norm(np.diff(pos, axis=0), axis=1), [0.0]])
        run = 0
        for t in range(int(min(t_pick + self.MIN_CARRY, T - 1)), T):
            if speed[t] <= self.SETTLE_TH:
                run += 1
                if run >= self.SETTLE_W:
                    return t - self.SETTLE_W + 1
            else:
                run = 0
        return T - 1

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}

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

        n = len(order)
        n_moved = 0
        for k, o in enumerate(order):
            # pick -> place boundary = the lift onset (item leaves its re-anchored rest)
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.LIFT_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.LIFT_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.cat}")

            if k == n - 1:
                # the final stack: carry + stack + release + withdraw all run to the end
                t_placed = T - 1
            else:
                # a non-final item ends when it comes to rest on the stack, so the NEXT
                # item's pick keeps a visible window (a gripper-release boundary here would
                # run to that item's LAST release, which for a re-grabbed item is near the
                # end of the demo and would swallow the next pick)
                t_placed = self._settle(opos[o], t_pick, T)
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"stack the {self.cat} on the larger {self.cat}")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != len(self.names):
            review.append(f"only {n_moved}/{len(self.names)} items stacked")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
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
        if npick not in (1, 2):
            issues.append(f"expected 1-2 stack moves, found {npick}")
        return issues
