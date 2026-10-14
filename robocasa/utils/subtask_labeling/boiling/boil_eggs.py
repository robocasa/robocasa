import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each egg is carried from the plate into the saucepan on the stove.
SUBTASK_MOVE = "PickPlaceCounterToStove"  # bank atomic: "plate -> pan"

DONE_LANG = "task complete"


@register_labeler
class BoilEggsLabeler(BaseSubtaskLabeler):
    """Labeler for the BoilEggs composite task.

    Task: the saucepan is already on a lit front burner with boiling water (no
    lid). Put the two eggs (from the plate) into the saucepan.

    Doc-canonical count = 2 (`mobile: No` in effect -- the base does not drive
    during any carry, plate is beside the stove): one PickPlaceCounterToStove per
    egg. Both eggs are always relocated; the ORDER varies and is resolved from
    when each is released. 2 PickPlace subtasks (2 stages each) + done = 5
    segments; expected_done_idx = 4.

    A simplified BoilCorn (no lid). Grasp attribution is sticky/exclusive
    (`_hold_assignment`); each egg's carry is the sticky held run merged across a
    brief pre-grasp touch / regrasp (`_carries`), with the pick->place boundary at
    the first sustained displacement from the grasp position. The two eggs share
    the word "egg", so the first is "an egg" and the second "the other egg".
    """

    ENV_NAME = "BoilEggs"

    EGGS = ("egg1", "egg2")

    GRASP_MAX = 0.08
    HOLD_MARGIN = 0.03
    MIN_RUN = 5
    MERGE_GAP = 50
    MIN_CARRY_DISP = 0.10
    MOVE_TH = 0.05
    MOVE_SUSTAIN = 8
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.EGGS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.egg_lang = env.get_obj_lang("egg1")
        except Exception:
            self.egg_lang = "egg"

        self.extra_subtasks = set()
        self.expected_done_idx = 4
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.EGGS:
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
        """Sticky per-frame grasp attribution to at most one egg."""
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

    def _carries(self, opos, held):
        """Real carry runs: grasp runs merged across small gaps, kept only if the
        object actually travels MIN_CARRY_DISP (discards a brief pre-grasp touch)."""
        T = len(held)
        runs = []
        i = 0
        while i < T:
            if held[i]:
                j = i
                while j < T and held[j]:
                    j += 1
                if j - i >= self.MIN_RUN:
                    runs.append([i, j - 1])
                i = j
            else:
                i += 1
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= self.MERGE_GAP:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out = []
        for a, b in merged:
            if np.linalg.norm(opos[a: b + 1] - opos[a], axis=1).max() >= self.MIN_CARRY_DISP:
                out.append((a, b))
        return out

    def _boundary(self, opos, a, b):
        """(t_move, release) for a grasp run [a, b] -- pick/place split at the
        first sustained displacement from the grasp position."""
        ref = np.median(opos[max(0, a - 5): a + 1], axis=0)
        disp = np.linalg.norm(opos - ref, axis=1)
        moved = disp > self.MOVE_TH
        t_move = self._sustained_start(moved, a, self.MOVE_SUSTAIN)
        if t_move is None:
            mf = np.where(moved[a:])[0]
            t_move = a + int(mf[0]) if mf.size else a
        return int(min(max(t_move, a), b + 1)), b + 1

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.EGGS}
        grip = np.array([h["grip"] for h in history])

        gd_arr = np.stack(
            [np.linalg.norm(grip - opos[o], axis=1) for o in self.EGGS], axis=0
        )
        assign = self._hold_assignment(gd_arr)

        events = []
        for k, o in enumerate(self.EGGS):
            for (a, b) in self._carries(opos[o], assign == k):
                events.append((a, b, o))
        events.sort()

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        art = "an" if self.egg_lang[:1].lower() in "aeiou" else "a"
        for i, (a, b, o) in enumerate(events):
            t_move, release = self._boundary(opos[o], a, b)
            which = "the other" if i == 1 else art
            emit(t_move, SUBTASK_MOVE, STAGE_PICK,
                 f"pick up {which} {self.egg_lang} from the plate")
            emit(release, SUBTASK_MOVE, STAGE_PLACE,
                 f"place the {self.egg_lang} in the saucepan")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if len(events) != 2:
            review.append(f"expected 2 egg carries, found {len(events)}")

        debug = {"n_events": len(events), "n_segments": len(segments)}
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
            issues.append(f"expected 2 egg picks, found {count(STAGE_PICK)}")
        if count(STAGE_PLACE) != 2:
            issues.append(f"expected 2 egg places, found {count(STAGE_PLACE)}")
        return issues
