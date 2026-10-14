import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each pastry is carried from the counter onto one of the two plates. Source and
# destination are both the counter (the plates sit on it), and the bank has no
# PickPlaceCounterToCounter, so the name is a declared extension (same convention
# as ArrangeBreadBasket / OrganizeBakingIngredients). It splits pick / place.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"


@register_labeler
class PastryDisplayLabeler(BaseSubtaskLabeler):
    """Labeler for the PastryDisplay composite task.

    Task: place the two pastries onto the two plates (one pastry per plate). The
    plates (receptacle1/receptacle2) are the stationary targets; only pastry1 and
    pastry2 move. Everything is within arm reach on one counter, so the base never
    drives -- no NavigateKitchen.

    Doc-canonical count = 2 (`mobile: No`): one PickPlaceCounterToCounter per
    pastry. Both are always relocated; the ORDER varies and is resolved from when
    each is released. 2 PickPlace subtasks (2 stages each) + done = 5 segments;
    expected_done_idx = 4.

    Grasp attribution is sticky/exclusive (`_hold_assignment`) so that, while one
    pastry is carried past the other or over the shared plate area, only the
    carried one counts as held. Each pastry's pick/place come from its own carry:
    t_move (first sustained displacement from rest) is the pick->place boundary;
    release is the end of the sticky hold run.

    Language:
      - The pastries are `obj_groups="pastry"`, so each demo samples a concrete
        type (croissant, donut, ...). When the two differ they are named by type
        ("the croissant" / "the donut"); when they are the SAME type they cannot
        be told apart, so the first is "a {type}" and the second "the other
        {type}".
      - The two plates are not distinguished in the task lang (no position info),
        so the target is not named by position: the first pastry placed goes "on
        a plate" and the second "on the other plate".
    """

    ENV_NAME = "PastryDisplay"

    OBJECTS = ("pastry1", "pastry2")
    PLATES = ("receptacle1", "receptacle2")

    GRASP_MAX = 0.06  # closest gripper approach below this == grasped
    HOLD_MARGIN = 0.03  # sticky release fires past GRASP_MAX + this
    MIN_RUN = 5  # min held-run length (frames) to count as a grasp
    MOVE_TH = 0.05  # object displacement from rest counting as transport underway
    MOVE_SUSTAIN = 8  # frames the motion must persist
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.OBJECTS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.OBJECTS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "pastry"

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = 4
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.OBJECTS:
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

    def _hold_assignment(self, gd_arr):
        """Sticky per-frame attribution of the grasp to at most one pastry: once
        acquired (< GRASP_MAX) it stays held until let go (> GRASP_MAX +
        HOLD_MARGIN), so a neighbouring pastry/plate can't steal frames mid-carry."""
        T = gd_arr.shape[1]
        acquire = self.GRASP_MAX
        release = self.GRASP_MAX + self.HOLD_MARGIN
        assign = np.full(T, -1, dtype=int)
        cur = -1
        for t in range(T):
            if cur != -1 and gd_arr[cur, t] < release:
                assign[t] = cur
            else:
                k = int(gd_arr[:, t].argmin())
                if gd_arr[k, t] < acquire:
                    cur = k
                    assign[t] = k
                else:
                    cur = -1
        return assign

    def _carry(self, held, opos):
        """Resolve (t_move, release) for one pastry's transport, or None."""
        T = len(opos)
        if not held.any():
            return None
        runs = []
        i = 0
        while i < T:
            if held[i]:
                j = i
                while j < T and held[j]:
                    j += 1
                if j - i >= self.MIN_RUN:
                    runs.append((i, j - 1))
                i = j
            else:
                i += 1
        if not runs:
            return None
        rest0 = np.median(opos[: self.REST_W], axis=0)
        grasp, last = max(
            runs, key=lambda r: np.linalg.norm(opos[r[0]: r[1] + 1] - rest0, axis=1).max()
        )
        release = last + 1
        rest_pos = np.median(opos[: max(grasp, self.REST_W)], axis=0)
        disp = np.linalg.norm(opos - rest_pos, axis=1)
        moved = disp > self.MOVE_TH
        t_move = self._sustained_start(moved, grasp, self.MOVE_SUSTAIN)
        if t_move is None:
            mf = np.where(moved[grasp:])[0]
            t_move = grasp + int(mf[0]) if mf.size else grasp
        return t_move, release

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.OBJECTS}
        grip = np.array([h["grip"] for h in history])

        # sticky grasp attribution -> per-pastry carry
        gd_arr = np.stack(
            [np.linalg.norm(grip - opos[o], axis=1) for o in self.OBJECTS], axis=0
        )
        assign = self._hold_assignment(gd_arr)
        moves = {}
        for k, o in enumerate(self.OBJECTS):
            c = self._carry(assign == k, opos[o])
            if c is not None:
                moves[o] = c
        order = sorted(moves, key=lambda o: moves[o][0])

        same_type = self.obj_lang[self.OBJECTS[0]] == self.obj_lang[self.OBJECTS[1]]
        moved_order = list(order)

        def pick_lang(o):
            word = self.obj_lang[o]
            if not same_type:
                return f"pick up the {word} from the counter"
            # identical pastries: first is "a", second is "the other"
            if len(moved_order) == 2 and o == moved_order[1]:
                return f"pick up the other {word} from the counter"
            return f"pick up a {word} from the counter"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for k, o in enumerate(order):
            t_move, release = moves[o]
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, pick_lang(o))
            # first pastry placed goes "on a plate", the second "on the other plate"
            plate_ref = "the other plate" if k == 1 else "a plate"
            emit(release, SUBTASK_MOVE, STAGE_PLACE,
                 f"place the {self.obj_lang[o]} on {plate_ref}")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if len(order) != len(self.OBJECTS):
            review.append(f"only {len(order)}/2 pastries had a clean grasp")

        debug = {
            "order": ",".join(order),
            "same_type": int(same_type),
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
        return result

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
            issues.append(f"expected 2 pastry picks, found {count(STAGE_PICK)}")
        if count(STAGE_PLACE) != 2:
            issues.append(f"expected 2 pastry places, found {count(STAGE_PLACE)}")
        return issues
