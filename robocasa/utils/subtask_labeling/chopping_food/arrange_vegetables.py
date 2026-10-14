import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Take the two vegetables out of the sink and set each on the cutting board (which
# sits on the counter by the sink). Two sink->counter pick-places, no navigation
# (mobile: No). PickPlaceSinkToCounter is a bank atomic (the cutting board is a
# receptacle ON the counter).
SUBTASK_MOVE = "PickPlaceSinkToCounter"

DONE_LANG = "task complete"

OBJS = ("vegetable1", "vegetable2")


@register_labeler
class ArrangeVegetablesLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeVegetables composite task.

    Task: two vegetables sit in the sink; a cutting board sits on the counter next
    to the sink. Pick each vegetable out of the sink and place it on the cutting
    board. The base never drives (mobile: No).

    Doc-canonical count = 2 (`mobile: No`): one PickPlaceSinkToCounter per
    vegetable. Both are always relocated; the ORDER varies (resolved from which
    moves first). 2 pick-places (pick/place stages each) + done = 5 segments;
    expected_done_idx = 4.

    The two vegetables are distinct bodies, so each is tracked by its own gripper
    distance (no sticky attribution needed). Per object, the pick/place split is at
    the first sustained displacement (like OrganizeCoffeeCondiments / BoilEggs):
    pick = reach + grasp; place = lift + carry + set-down + release.
      - t_move (pick end): the vegetable first displaces MOVE_TOL from its rest (in-
                 sink) pose, after the previous vegetable was released.
      - t_placed (place end): the gripper RELEASES the vegetable -- gripper->object
                 distance rises above the carry (hold) level -- after the carry.
    """

    ENV_NAME = "ArrangeVegetables"

    MOVE_TOL = 0.10  # object displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_move used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.08  # grip->obj rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20  # a carry lasts at least this many frames before a release counts
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in OBJS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in OBJS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "vegetable"

        self.extra_subtasks = set()
        self.expected_done_idx = 4
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in OBJS:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

    # ---- helpers -------------------------------------------------------

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in OBJS}
        grip = np.array([h["grip"] for h in history])

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in OBJS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in OBJS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in OBJS}

        # order the two objects by which moves first
        first_move = {}
        for o in OBJS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(OBJS, key=lambda o: first_move[o])

        # distinguish the two by type when they differ, else "a"/"the other"
        same_type = self.obj_lang[OBJS[0]] == self.obj_lang[OBJS[1]]

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for k, o in enumerate(order):
            t_move = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)

            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            word = self.obj_lang[o]
            if same_type:
                noun = "the other " + word if k == 1 else "a " + word
            else:
                noun = "the " + word
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} on the cutting board")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": order,
            "first_move": {o: int(first_move[o]) for o in OBJS},
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(stage):
            return int(np.sum((names == SUBTASK_MOVE) & (stages == stage)))

        if count(STAGE_PICK) != 2:
            issues.append(f"expected 2 vegetable picks, found {count(STAGE_PICK)}")
        if count(STAGE_PLACE) != 2:
            issues.append(f"expected 2 vegetable places, found {count(STAGE_PLACE)}")
        return issues
