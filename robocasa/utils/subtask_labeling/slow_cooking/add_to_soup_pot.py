import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# The saucepan simmers on a low burner with its lid on. Take the lid off (saucepan ->
# counter), drop two vegetables in (plate -> saucepan), then put the lid back on (counter
# -> saucepan). All four moves are bank PickPlace skills keyed by their endpoints.
SUBTASK_LID_OFF = "PickPlaceStoveToCounter"   # lid: off the saucepan onto the counter
SUBTASK_TO_POT = "PickPlaceCounterToStove"    # veg: plate -> saucepan; lid back on

DONE_LANG = "task complete"


@register_labeler
class AddToSoupPotLabeler(BaseSubtaskLabeler):
    """Labeler for the AddToSoupPot composite task.

    Task (mobile, local): a saucepan simmers on a low front burner with its lid on; two
    vegetables sit on a plate on the adjacent counter. Take the lid off, put both vegetables
    in the saucepan, then put the lid back on. The burner is already low (no knob action).

    Doc-canonical count = 4 (`mobile: Yes`), all plain PickPlace (base does not drive):
      1. lid off:  PickPlaceStoveToCounter [pick]/[place]  saucepan -> counter
      2. veg:      PickPlaceCounterToStove [pick]/[place]  plate -> saucepan
      3. veg:      PickPlaceCounterToStove [pick]/[place]  plate -> saucepan
      4. lid on:   PickPlaceCounterToStove [pick]/[place]  counter -> saucepan
    plus a terminal done -> 9 segments, expected_done_idx = 8. Vegetables in either order.

    The LID (`saucepan_auxiliary`) is grabbed tightly (grip->center ~0.01) and handled
    TWICE, so its two carries are read from its own grasp runs. The VEGETABLES are often
    grabbed off-center (grip->center up to ~0.11, more than the lid-tuned grasp threshold),
    so each veg's single carry is read from DISPLACEMENT (first sustained move -> release),
    which is independent of grasp distance and can't be stolen by the resting lid. All
    events are merged in first-move order.
    """

    ENV_NAME = "AddToSoupPot"

    VEGS = ("vegetable1", "vegetable2")
    LID = "saucepan_auxiliary"

    # lid grasp-run detection
    GRASP_LID = 0.06
    MIN_RUN = 5
    MERGE_GAP = 50
    MIN_CARRY_DISP = 0.10
    MOVE_TH = 0.05        # lid pick->place onset (displacement from grasp position)
    MOVE_SUSTAIN = 8
    # vegetable displacement detection
    VEG_MOVE_TOL = 0.08
    VEG_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 20
    PICK_FLOOR = 8   # a same-plate 2nd veg is grabbed with almost no reach; give its
                     # pick at least this many frames so it can't collapse to zero length

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.movable = tuple(o for o in (*self.VEGS, self.LID) if o in env.obj_body_id)
        self.vegs = [o for o in self.VEGS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.movable}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.veg_word = {}
        for o in self.vegs:
            try:
                self.veg_word[o] = env.get_obj_lang(o)
            except Exception:
                self.veg_word[o] = "vegetable"
        self.extra_subtasks = set()  # both names are bank atomics
        self.expected_done_idx = 8
        # a 2nd vegetable on the same plate is grabbed with a legitimately brief reach
        self.min_segment_frames = None

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.movable:
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

    def _runs(self, held):
        T = len(held)
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
        return runs

    def _carries(self, opos, held):
        runs = self._runs(held)
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

    def _lid_carry(self, opos, a, b):
        ref = np.median(opos[max(0, a - 5): a + 1], axis=0)
        disp = np.linalg.norm(opos - ref, axis=1)
        moved = disp > self.MOVE_TH
        t_move = self._sustained_start(moved, a, self.MOVE_SUSTAIN)
        if t_move is None:
            mf = np.where(moved[a:])[0]
            t_move = a + int(mf[0]) if mf.size else a
        return int(min(max(t_move, a), b + 1)), b + 1

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.movable}
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.movable}

        events = []  # (rough_t_pick, kind, payload)

        # --- LID: two carries (off, then on) from its own grasp runs ---
        lid_carries = self._carries(opos[self.LID], gd[self.LID] < self.GRASP_LID)
        for i, (a, b) in enumerate(lid_carries):
            t_pick, t_place = self._lid_carry(opos[self.LID], a, b)
            events.append((int(t_pick), "lid", (i, int(t_pick), int(t_place))))

        # --- VEGETABLES: one carry each; a rough first-move only decides the ORDER --
        # the exact pick/place is recomputed at emit time with the rest re-anchored at the
        # cursor, so a nudge during the previous carry can't pull the pick early / collapse it.
        for o in self.vegs:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            disp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(disp >= self.VEG_MOVE_TOL, 0, self.VEG_SUSTAIN)
            if fm is None:
                continue
            events.append((int(fm), "veg", o))

        events.sort(key=lambda e: e[0])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        lid_seen = 0
        n_veg = 0
        for (_rough, kind, payload) in events:
            if kind == "lid":
                _i, t_pick, t_place = payload
                if lid_seen == 0:  # removal: saucepan -> counter
                    emit(t_pick, SUBTASK_LID_OFF, STAGE_PICK, "pick up the lid from the simmering saucepan")
                    emit(t_place, SUBTASK_LID_OFF, STAGE_PLACE, "place the lid on the counter")
                else:  # replacement: counter -> saucepan
                    emit(t_pick, SUBTASK_TO_POT, STAGE_PICK, "pick up the lid from the counter")
                    emit(t_place, SUBTASK_TO_POT, STAGE_PLACE, "place the lid on the saucepan")
                lid_seen += 1
            else:
                o = payload
                r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
                dsp = np.linalg.norm(opos[o] - r, axis=1)
                t_pick = self._sustained_start(dsp >= self.VEG_MOVE_TOL, cursor, self.VEG_SUSTAIN)
                if t_pick is None:
                    mv = np.where(dsp[cursor:] >= self.VEG_MOVE_TOL)[0]
                    t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
                t_pick = max(t_pick, cursor)
                hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
                t_place = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                                t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
                if t_place is None:
                    t_place = T - 1
                # a same-plate 2nd veg is grabbed instantly -> the reach can be ~0; carve a
                # minimal non-zero pick window from the carry so the pick can't collapse
                if t_pick - cursor < self.PICK_FLOOR and t_place - cursor > self.PICK_FLOOR:
                    t_pick = cursor + self.PICK_FLOOR
                w = self.veg_word.get(o, "vegetable")
                emit(t_pick, SUBTASK_TO_POT, STAGE_PICK, f"pick up the {w} from the plate")
                emit(t_place, SUBTASK_TO_POT, STAGE_PLACE, f"place the {w} in the saucepan")
                n_veg += 1
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if lid_seen != 2 or n_veg != 2:
            review.append(f"expected lid handled twice + 2 vegetables, got lid x{lid_seen}, veg x{n_veg}")
        result = self.build_result(T, segments, debug={
            "lid_carries": lid_seen, "veg": n_veg, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_LID_OFF, STAGE_PICK) != 1:
            issues.append(f"expected 1 {SUBTASK_LID_OFF}, found {count(SUBTASK_LID_OFF, STAGE_PICK)}")
        if count(SUBTASK_TO_POT, STAGE_PICK) != 3:
            issues.append(f"expected 3 {SUBTASK_TO_POT} picks, found {count(SUBTASK_TO_POT, STAGE_PICK)}")
        n_pick = count(SUBTASK_LID_OFF, STAGE_PICK) + count(SUBTASK_TO_POT, STAGE_PICK)
        n_place = count(SUBTASK_LID_OFF, STAGE_PLACE) + count(SUBTASK_TO_POT, STAGE_PLACE)
        if n_pick != 4 or n_place != 4:
            issues.append(f"expected 4 picks + 4 places, found {n_pick} + {n_place}")
        return issues
