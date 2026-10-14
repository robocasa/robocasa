import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A vegetable and a meat both start ON the cutting board; each is lifted off and set
# on the counter to clear the board for cleaning. The cutting board sits on the
# counter, so board -> counter has no bank name -> PickPlaceCuttingBoardToCounter
# (a declared extension, the same "clear" atomic ClearCuttingBoard uses). Each splits
# pick / place. Non-mobile, so no NavigateKitchen.
SUBTASK_CLEAR = "PickPlaceCuttingBoardToCounter"

DONE_LANG = "task complete"

ITEMS = ("vegetable", "meat")


@register_labeler
class RemoveCuttingBoardItemsLabeler(BaseSubtaskLabeler):
    """Labeler for the RemoveCuttingBoardItems composite task.

    Task (non-mobile): a vegetable and a meat sit on the cutting board. Remove both
    off the board (onto the counter) to prepare it for cleaning. Everything is within
    arm reach on one counter, so the base never drives -- no NavigateKitchen.

    Doc-canonical count = 2 (`mobile: No`): one clear per item. Both are relocated;
    the ORDER is resolved from when each starts moving. 2 pick-places (pick/place
    stages each) + done = 5 segments; expected_done_idx = 4.

    Each item moves ONE AT A TIME, tracked by its OWN displacement + gripper distance
    (first-move / relative-release split, like ClearCuttingBoard -- the release is
    measured relative to each object's carry hold distance, so an elongated food
    grabbed off-centre is handled without an absolute grasp threshold):
      - t_move (pick end)  = the item first displaces MOVE_TOL from its rest pose,
                             after the previous item was released.
      - t_placed (place end) = the gripper RELEASES the item -- gripper->object
                             distance rises RELEASE_MARGIN above the carry level.
    """

    ENV_NAME = "RemoveCuttingBoardItems"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.names:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = o
        self.extra_subtasks = {SUBTASK_CLEAR}
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

    def _move_release(self, opos, grip, disp, cursor, T):
        t_move = self._sustained_start(disp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp[cursor:] >= self.MOVE_TOL)[0]
            t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        t_move = max(t_move, cursor)
        gd = np.linalg.norm(grip - opos, axis=1)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1
        return t_move, t_placed

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.names}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.names}

        first_move = {}
        for o in self.names:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(self.names, key=lambda o: first_move[o])

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
            n_moved += 1
            word = self.obj_lang[o]
            t_move, t_placed = self._move_release(opos[o], grip, disp[o], cursor, T)
            emit(t_move, SUBTASK_CLEAR, STAGE_PICK,
                 f"pick up the {word} from the cutting board")
            emit(t_placed, SUBTASK_CLEAR, STAGE_PLACE,
                 f"place the {word} on the counter")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 items removed from the board")

        result = self.build_result(T, segments, debug={
            "order": ",".join(order),
            "first_move": {o: int(first_move[o]) for o in self.names},
            "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        # setup() is skipped for manual overrides, so re-declare the extension name
        # before the bank check, then short-circuit the structural checks for them.
        self.extra_subtasks = {SUBTASK_CLEAR}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
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
