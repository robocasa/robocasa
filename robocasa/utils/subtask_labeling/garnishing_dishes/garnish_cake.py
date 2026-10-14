import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# One cherry and one strawberry are moved from the fruit plate onto the cake / cake
# plate (all on the dining counter). Source and destination are both the counter, so
# the bank has no name -> PickPlaceCounterToCounter is a declared extension. No nav.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"

CHERRIES = ("cherry1", "cherry2")
STRAWBERRIES = ("strawberry1", "strawberry2")


@register_labeler
class GarnishCakeLabeler(BaseSubtaskLabeler):
    """Labeler for the GarnishCake composite task.

    Task: a cake sits on the cake plate; a fruit plate nearby holds two cherries and
    two strawberries. Move ONE cherry onto the cake (or the cake plate) and ONE
    strawberry onto the cake plate. The base does not drive.

    Doc-canonical decomposition (2 high-level subtasks: cherry, strawberry). Exactly
    one cherry and one strawberry are relocated (the others stay); this labeler picks
    the moved one of each by displacement. Emitted per fruit as a [pick]/[place].
    The task text lists the cherry first, but demos often move the strawberry first,
    so the ORDER is taken from the data (by first move).

    Per fruit: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_placed = the gripper RELEASE.
    """

    ENV_NAME = "GarnishCake"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 30
    RELEASE_MARGIN = 0.05
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in CHERRIES + STRAWBERRIES}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = 4
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in CHERRIES + STRAWBERRIES:
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
        allobjs = CHERRIES + STRAWBERRIES
        opos = {o: np.array([h[o] for h in history]) for o in allobjs}
        grip = np.array([h["grip"] for h in history])
        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in allobjs}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in allobjs}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in allobjs}

        # the moved cherry / strawberry = the one displaced most in its pair
        moved_cherry = max(CHERRIES, key=lambda o: float(disp[o].max()))
        moved_straw = max(STRAWBERRIES, key=lambda o: float(disp[o].max()))
        # noun per body id, so the emit loop can label generically
        noun = {moved_cherry: "cherry", moved_straw: "strawberry"}

        first_move = {}
        for o in (moved_cherry, moved_straw):
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([moved_cherry, moved_straw], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            if first_move[o] >= T:
                continue
            word = noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            # there are two of each fruit on the plate, so the pick is "a {fruit}"
            # (not "the", which would imply a single one); the place refers to the one
            # now in hand -> "the {fruit}".
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up a {word} from the fruit plate")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} on the cake plate")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 fruits moved")
        result = self.build_result(T, segments, debug={
            "moved_cherry": moved_cherry, "moved_straw": moved_straw,
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
            issues.append(f"expected 2 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
