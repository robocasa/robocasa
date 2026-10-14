import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each bread is carried from the counter onto the cutting board (which sits on the
# counter). Source and destination are both the counter, and the bank has no
# PickPlaceCounterToCounter, so the name is a declared extension (same convention
# as PastryDisplay / OrganizeBakingIngredients). It splits pick / place.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"


@register_labeler
class BreadSetupSlicingLabeler(BaseSubtaskLabeler):
    """Labeler for the BreadSetupSlicing composite task.

    Task: 1-3 pieces of bread sit on the counter; a cutting board (receptacle) sits
    on the same counter. Place every bread onto the cutting board. Everything is
    within arm reach, so the base never drives -- no NavigateKitchen.

    Doc-canonical count = 2 (`mobile: No`): one PickPlaceCounterToCounter per bread.
    But the number of breads VARIES 1-3 per demo (`num_bread` in ep_meta refs), and
    a bread that spawns already on the board is not relocated -- so the actual move
    count varies and expected_done_idx is set per demo; validation is structural.
    The ORDER also varies and is resolved from when each starts moving.

    A bread counts as RELOCATED (gets a PickPlace) iff it was BOTH grasped AND
    displaced. The breads are identical bodies clustered together, so grasp
    attribution is sticky/exclusive (`_hold_assignment`) and each bread's
    pick/place come from its OWN motion:
      - t_move (pick->place boundary) = the start of the bread's transport (first
        sustained displacement from rest -- DISPLACEMENT, not lift, since a bread is
        often slid along the counter).
      - release = the end of the sticky hold run (the gripper lets go).
    """

    ENV_NAME = "BreadSetupSlicing"

    GRASP_MAX = 0.06  # closest gripper approach below this == grasped
    HOLD_MARGIN = 0.03  # sticky release fires past GRASP_MAX + this
    MIN_RUN = 5  # min held-run length (frames) to count as a grasp
    MIN_DISP = 0.06  # max object displacement above this == it was actually moved
    MOVE_TH = 0.05  # object displacement from rest counting as transport underway
    MOVE_SUSTAIN = 8  # frames the motion must persist
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        # number of breads is recorded in ep_meta refs (spawned obj_0 .. obj_{n-1})
        try:
            self.num_bread = int(ep_meta["refs"]["num_bread"])
        except Exception:
            self.num_bread = sum(
                1 for k in env.obj_body_id if k.startswith("obj_")
            )
        self.OBJECTS = tuple(f"obj_{i}" for i in range(self.num_bread))
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.OBJECTS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.OBJECTS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "bread"

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
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
        """Sticky per-frame attribution of the grasp to at most one bread: once
        acquired (< GRASP_MAX) it stays held until let go (> GRASP_MAX +
        HOLD_MARGIN), so a neighbouring bread/board can't steal frames mid-carry."""
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

    def _move_interval(self, held, opos):
        """(t_move, release) for one bread's transport, or None if not relocated."""
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
        rest_pos = np.median(opos[: max(grasp, self.REST_W)], axis=0)
        disp = np.linalg.norm(opos - rest_pos, axis=1)
        if float(disp.max()) < self.MIN_DISP:
            return None  # held but only adjusted in place -- not a relocation
        # cap at T-1 so a demo cut the instant the last bread lands still leaves a
        # frame for the terminal 'done' (validator exempts a single-frame done).
        release = min(last + 1, T - 1)
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

        gd_arr = np.stack(
            [np.linalg.norm(grip - opos[o], axis=1) for o in self.OBJECTS], axis=0
        )
        assign = self._hold_assignment(gd_arr)
        moves = {}
        for k, o in enumerate(self.OBJECTS):
            mi = self._move_interval(assign == k, opos[o])
            if mi is not None:
                moves[o] = mi
        order = sorted(moves, key=lambda o: moves[o][0])  # by transport start

        # noun for each relocated bread: identical bodies get an ordinal scheme,
        # distinct sampled types are named by type.
        words = [self.obj_lang[o] for o in order]
        identical = len(set(words)) <= 1
        n = len(order)

        def bread_noun(k, o):
            word = self.obj_lang[o]
            if not identical:
                return f"the {word}"
            if n == 1:
                return f"the {word}"
            if k == 0:
                return f"a {word}"
            if k == n - 1:
                return f"the last {word}" if n >= 3 else f"the other {word}"
            return f"another {word}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for k, o in enumerate(order):
            t_move, release = moves[o]
            noun = bread_noun(k, o)
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the counter")
            emit(release, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} on the cutting board")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        # Soft review: a bread that did not move must have spawned already on the
        # board -- name the variant so a human can confirm.
        review = []
        not_moved = [o for o in self.OBJECTS if o not in moves]
        if not_moved:
            review.append(
                f"{len(moves)}/{self.num_bread} breads relocated; "
                f"already on the board: {', '.join(not_moved)}"
            )

        debug = {
            "num_bread": self.num_bread,
            "order": ",".join(order),
            "n_moved": len(moves),
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

        n_pick, n_place = count(STAGE_PICK), count(STAGE_PLACE)
        if n_pick != n_place:
            issues.append(f"picks ({n_pick}) != places ({n_place})")
        if not (1 <= n_pick <= self.num_bread):
            issues.append(
                f"expected 1-{self.num_bread} bread moves, found {n_pick}"
            )
        return issues
