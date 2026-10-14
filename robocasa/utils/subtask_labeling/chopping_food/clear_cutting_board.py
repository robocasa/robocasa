import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# The non-vegetable is lifted OFF the cutting board and set on the counter (a
# clear); each vegetable is then carried from the counter ONTO the cutting board.
# The cutting board sits on the counter, so both directions are counter<->counter,
# for which the bank has no name -- two declared extensions distinguishing the
# clear from the place. Each splits pick / place.
SUBTASK_CLEAR = "PickPlaceCuttingBoardToCounter"  # non-veg: board -> counter
SUBTASK_VEG = "PickPlaceCounterToCuttingBoard"    # veg: counter -> board

DONE_LANG = "task complete"

NON_VEG = "non_vegetable"
VEGGIES = ("vegetable1", "vegetable2")


@register_labeler
class ClearCuttingBoardLabeler(BaseSubtaskLabeler):
    """Labeler for the ClearCuttingBoard composite task.

    Task: a non-vegetable food item starts ON the cutting board; two vegetables sit
    on the counter beside it. Clear the non-vegetable off the board (set it on the
    counter), then place both vegetables onto the now-clear board. Everything is
    within arm reach on one counter, so the base never drives -- no NavigateKitchen.

    Doc-canonical count = 3 (`mobile: No`): the non-vegetable's clear + one place
    per vegetable. All three are relocated; the ORDER is resolved from when each
    starts moving (the clear leads -- the board must be freed before the vegetables
    go on). 3 pick-places (pick/place stages each) + done = 7 segments;
    expected_done_idx = 6.

    The three objects move ONE AT A TIME in sequence, so each is tracked by its OWN
    displacement + gripper distance (no sticky/exclusive grasp attribution, which
    needs an absolute grasp gate -- and the non-vegetable can be an ELONGATED food,
    e.g. a baguette, grabbed at the tip, whose gripper->body-center distance never
    drops below ~0.064). Per object (first-move / relative-release split, like
    ArrangeVegetables / SweetenCoffee):
      - t_move (pick end) = the object first displaces MOVE_TOL from its rest pose,
        after the previous object was released.
      - t_placed (place end) = the gripper RELEASES the object -- gripper->object
        distance rises RELEASE_MARGIN above the carry (hold) level. Because the
        release is measured relative to each object's own hold distance, a
        tip-grabbed baguette (hold ~0.064) is handled the same as a centre-grabbed
        vegetable (hold ~0.01) with no absolute threshold.
    """

    ENV_NAME = "ClearCuttingBoard"

    MOVE_TOL = 0.10  # object displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_move used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.08  # grip->obj rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20  # a carry lasts at least this many frames before a release counts
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.OBJECTS = (NON_VEG, *VEGGIES)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.OBJECTS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.OBJECTS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "vegetable" if o in VEGGIES else "object"

        self.extra_subtasks = {SUBTASK_CLEAR, SUBTASK_VEG}
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

    def _move_release(self, opos, grip, disp, cursor, T):
        """(t_move, t_placed) for one object: t_move = first sustained displacement
        after cursor; t_placed = the gripper release (relative to this object's own
        carry hold level) after the carry."""
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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.OBJECTS}
        grip = np.array([h["grip"] for h in history])

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.OBJECTS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.OBJECTS}

        # order the three objects by which moves first (the clear leads)
        first_move = {}
        for o in self.OBJECTS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(self.OBJECTS, key=lambda o: first_move[o])

        # vegetable naming: by type when the two differ, else "a"/"the other"
        moved_veg = [o for o in order if o in VEGGIES and first_move[o] < T]
        veg_same = (
            len(moved_veg) == 2
            and self.obj_lang[moved_veg[0]] == self.obj_lang[moved_veg[1]]
        )

        def veg_noun(o):
            word = self.obj_lang[o]
            if not veg_same:
                return f"the {word}"
            return f"the other {word}" if o == moved_veg[1] else f"a {word}"

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
                continue  # never moved (anomaly) -- skip; flagged below
            n_moved += 1
            t_move, t_placed = self._move_release(opos[o], grip, disp[o], cursor, T)
            if o == NON_VEG:
                word = self.obj_lang[o]
                emit(t_move, SUBTASK_CLEAR, STAGE_PICK,
                     f"pick up the {word} from the cutting board")
                emit(t_placed, SUBTASK_CLEAR, STAGE_PLACE,
                     f"place the {word} on the counter")
            else:
                noun = veg_noun(o)
                emit(t_move, SUBTASK_VEG, STAGE_PICK,
                     f"pick up {noun} from the counter")
                emit(t_placed, SUBTASK_VEG, STAGE_PLACE,
                     f"place {noun} on the cutting board")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 objects moved")
        if first_move[NON_VEG] >= T:
            review.append("the non-vegetable was never cleared off the board")

        debug = {
            "order": ",".join(order),
            "first_move": {o: int(first_move[o]) for o in self.OBJECTS},
            "n_moved": n_moved,
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

        def count(name, stage):
            return int(np.sum((names == name) & (stages == stage)))

        if count(SUBTASK_CLEAR, STAGE_PICK) != 1:
            issues.append(
                f"expected 1 non-veg clear pick, found {count(SUBTASK_CLEAR, STAGE_PICK)}"
            )
        if count(SUBTASK_CLEAR, STAGE_PLACE) != 1:
            issues.append(
                f"expected 1 non-veg clear place, found {count(SUBTASK_CLEAR, STAGE_PLACE)}"
            )
        if count(SUBTASK_VEG, STAGE_PICK) != 2:
            issues.append(
                f"expected 2 vegetable picks, found {count(SUBTASK_VEG, STAGE_PICK)}"
            )
        if count(SUBTASK_VEG, STAGE_PLACE) != 2:
            issues.append(
                f"expected 2 vegetable places, found {count(SUBTASK_VEG, STAGE_PLACE)}"
            )
        return issues
