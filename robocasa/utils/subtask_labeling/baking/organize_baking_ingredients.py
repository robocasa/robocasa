import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# The three ingredients are each moved from near the sink to next to the bowl.
# Source and destination are both the counter, and the bank has no
# PickPlaceCounterToCounter, so the name is a declared extension (same convention
# as ArrangeBreadBasket). It splits into pick / place stages like any PickPlace.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"


@register_labeler
class OrganizeBakingIngredientsLabeler(BaseSubtaskLabeler):
    """Labeler for the OrganizeBakingIngredients composite task.

    Task: gather the two eggs and the milk (scattered near the sink) and set each
    one down next to the bowl on the counter. The bowl is the stationary
    reference; only egg1, egg2, milk move. Everything is within arm reach (the
    ingredients and bowl share the counter by the sink), so the base never drives
    -- there is no NavigateKitchen.

    Doc-canonical count = 3 (`mobile: Yes`): one PickPlaceCounterToCounter per
    ingredient. But the count is really the number of ingredients OUT of place --
    some demos spawn an egg (or the milk) already next to the bowl, and the
    operator only relocates the others. So the move count VARIES 1-3 and
    expected_done_idx is set per demo; validation is structural. The ORDER also
    varies and is resolved from when each starts moving.

    An ingredient counts as RELOCATED (gets a PickPlace) iff it was BOTH grasped
    AND displaced -- grasp is the reliable discriminator, since a pre-placed
    ingredient is never touched (its closest gripper approach stays >= ~0.09,
    vs <= ~0.06 for a grasped one), while displacement alone misses a genuine
    small relocation of an egg that spawned only just out of place. Its pick/place
    boundaries come from the object's OWN MOTION (not the gripper -- the two eggs
    sit close together and everything ends clustered by the bowl, so gripper
    proximity is ambiguous mid-trajectory):
      - t_move (pick->place boundary) = the start of the object's transport (it
        first leaves rest, backed up to the start of that motion burst).
        DISPLACEMENT, not lift -- the ingredients are frequently SLID along the
        counter rather than lifted (z barely changes), so a lift test never fires.
      - t_settled (place end) = the object first arrives within SETTLE_TOL of its
        final resting spot and stays.
    Relocated objects are ordered by t_move (the gather order). Each contributes a
    pick + place; a terminal 'done' follows.
    """

    ENV_NAME = "OrganizeBakingIngredients"

    OBJECTS = ("egg1", "egg2", "milk")
    TARGET = "bowl"

    # --- which objects were relocated (vs spawned already by the bowl) ---
    GRASP_MAX = 0.08  # closest gripper approach below this == the object was grasped
    # (a pre-placed ingredient is never touched: its min approach stays >= ~0.09,
    # vs <= ~0.06 for a grasped one, so a FIXED threshold cleanly separates them)
    MIN_DISP = 0.06  # max object displacement above this == it was actually moved
    MIN_RUN = 5  # min held-run length (frames) to count as a grasp (not a brush)
    HOLD_MARGIN = 0.03  # release extends while gripper within this of GRASP_MAX
    # --- pick -> place boundary ---
    MOVE_TH = 0.05  # object displacement from rest counting as transport underway
    MOVE_SUSTAIN = 8  # frames the motion must persist (rejects a grasp-time jostle)
    REST_W = 20  # window for rest / final pose medians
    SUCCESS_TH = 0.20  # a non-moved object should still be within this of the bowl

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {
            o: env.obj_body_id[o] for o in (*self.OBJECTS, self.TARGET)
        }
        self.eef_site_id = env.robots[0].eef_site_id["right"]

        self.obj_lang = {}
        for o in self.OBJECTS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "milk" if o == "milk" else "egg"

        # Sink position + the agentview-center camera, to say which side of the
        # sink each ingredient starts on -- projected onto the camera's right axis
        # so "left"/"right" match what the rendered video shows (this correctly
        # flips for mirrored kitchen layouts).
        self.sink_pos = np.array(env.sink.pos, dtype=float)
        try:
            self.cam_id = env.sim.model.camera_name2id("robot0_agentview_center")
        except Exception:
            self.cam_id = None

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in (*self.OBJECTS, self.TARGET):
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        d["cam_right"] = (
            env.sim.data.cam_xmat[self.cam_id].reshape(3, 3)[:, 0].copy()
            if self.cam_id is not None
            else np.array([0.0, 1.0, 0.0])
        )
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
        """Sticky per-frame attribution of the grasp to at most one ingredient.

        A stationary ingredient already sitting at the bowl can be closer to the
        gripper than the one actually being carried to it, so a plain nearest test
        mis-attributes the carried object's frames. Instead, once an ingredient is
        acquired (within GRASP_MAX) it STAYS the held one until the gripper truly
        lets go (distance exceeds GRASP_MAX + HOLD_MARGIN) -- hysteresis that keeps
        a placed neighbour from stealing frames mid-carry."""
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
        """Resolve (t_move, release) for one ingredient's transport, or None if it
        was not relocated (never the held object, or held but not displaced).
        `held` is the sticky per-frame hold mask for this ingredient."""
        T = len(opos)
        if not held.any():
            return None  # never the grasped object -- spawned already by the bowl
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
        # carry run = the held run that displaces the object the most (isolates the
        # real move from an incidental brush near the clustered bowl)
        rest0 = np.median(opos[: self.REST_W], axis=0)
        grasp, last = max(
            runs, key=lambda r: np.linalg.norm(opos[r[0]: r[1] + 1] - rest0, axis=1).max()
        )
        rest_pos = np.median(opos[: max(grasp, self.REST_W)], axis=0)
        disp = np.linalg.norm(opos - rest_pos, axis=1)
        if float(disp.max()) < self.MIN_DISP:
            return None  # held but only adjusted in place -- not a relocation
        release = last + 1  # the sticky mask already ends the run at the let-go
        # pick -> place boundary: the object starts moving (slid or lifted)
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
        bowl = np.array([h[self.TARGET] for h in history])
        bowl_final = np.median(bowl[-self.REST_W:], axis=0)
        # Side of the sink for each ingredient (spawn) and the bowl, from the
        # video's view (projected onto the initial camera right axis).
        cr0 = np.median(np.array([h["cam_right"] for h in history])[: self.REST_W], axis=0)

        def side_of(pos):
            return "right" if float(np.dot(pos - self.sink_pos, cr0)) > 0 else "left"

        spawn = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.OBJECTS}
        side = {o: side_of(spawn[o]) for o in self.OBJECTS}
        bowl_side = side_of(bowl_final)
        second_egg = None  # set below once the gather order is known

        def pick_lang(o):
            """Instruction for picking ingredient o.

            The sink side is a LOCATOR, named whenever the ingredient sits ACROSS
            the sink from the bowl (never when it is on the bowl's own side) -- for
            eggs too, and even for 'the other egg'. Egg noun:
              - the second of two relocated eggs -> "the other egg"
              - else "the egg" if the (opposite-bowl) side uniquely identifies it
                (the eggs are on different sides), otherwise "an egg".
            """
            word = self.obj_lang[o]  # "milk" / "egg"
            opp = side[o] != bowl_side
            if o in ("egg1", "egg2"):
                if o == second_egg:
                    noun = "the other egg"
                else:
                    other = "egg2" if o == "egg1" else "egg1"
                    unique = opp and side[o] != side[other]
                    noun = f"the {word}" if unique else f"an {word}"
            else:
                noun = f"the {word}"
            if opp:
                return f"pick up {noun} to the {side[o]} of the sink"
            return f"pick up {noun} from the counter"

        grip = np.array([h["grip"] for h in history])
        # Sticky, exclusive grasp attribution (see _hold_assignment): each frame's
        # grasp goes to at most one ingredient, held through the carry.
        gd_arr = np.stack(
            [np.linalg.norm(grip - opos[o], axis=1) for o in self.OBJECTS], axis=0
        )  # (n_obj, T)
        assign = self._hold_assignment(gd_arr)

        moves = {}
        for k, o in enumerate(self.OBJECTS):
            mi = self._move_interval(assign == k, opos[o])
            if mi is not None:
                moves[o] = mi
        order = sorted(moves, key=lambda o: moves[o][0])  # by transport start

        # When BOTH eggs are relocated, the second one picked is "the other egg".
        moved_eggs = [o for o in order if o in ("egg1", "egg2")]
        second_egg = moved_eggs[1] if len(moved_eggs) == 2 else None

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for o in order:
            t_move, release = moves[o]
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, pick_lang(o))
            emit(release, SUBTASK_MOVE, STAGE_PLACE,
                 f"place the {self.obj_lang[o]} next to the bowl")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        # Soft review: fewer than three moves is a valid variant (some ingredients
        # spawn already by the bowl) -- name them so a human can confirm. If a
        # NON-moved ingredient is far from the bowl, that is suspicious.
        review = []
        not_moved = [o for o in self.OBJECTS if o not in moves]
        if not_moved:
            far = [
                o for o in not_moved
                if float(np.linalg.norm(np.median(opos[o][-self.REST_W:], axis=0) - bowl_final))
                > self.SUCCESS_TH
            ]
            review.append(
                f"{len(moves)}/3 ingredients relocated; already by the bowl: "
                f"{', '.join(not_moved)}"
                + (f" (WARNING far from bowl: {', '.join(far)})" if far else "")
            )

        debug = {
            "gather_order": ",".join(order),
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
        # The number of relocated ingredients varies 1-3 (some spawn by the bowl),
        # but picks and places must match and there must be at least one move.
        if n_pick != n_place:
            issues.append(f"picks ({n_pick}) != places ({n_place})")
        if not (1 <= n_pick <= 3):
            issues.append(f"expected 1-3 ingredient moves, found {n_pick}")
        return issues
