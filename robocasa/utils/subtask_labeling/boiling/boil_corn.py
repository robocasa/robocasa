import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# The saucepan sits on a lit burner with its lid on. To boil the corn the robot
# must: take the lid off (saucepan -> counter), drop each corn in (plate ->
# saucepan), then put the lid back on (counter -> saucepan). All four moves are
# bank PickPlace skills keyed by their endpoints.
SUBTASK_LID_OFF = "PickPlaceStoveToCounter"   # lid: off the saucepan onto the counter
SUBTASK_TO_POT = "PickPlaceCounterToStove"    # corn: plate -> saucepan; lid back on

DONE_LANG = "task complete"


@register_labeler
class BoilCornLabeler(BaseSubtaskLabeler):
    """Labeler for the BoilCorn composite task.

    Task: the saucepan is already on a lit front burner with its lid on. Take the
    lid off, put the two corns (from the plate) into the saucepan, then put the
    lid back on to start boiling.

    Doc-canonical count = 4 (`mobile: Yes`), all plain PickPlace (the base does
    NOT drive during any carry -- verified base displacement ~0 across each grasp
    run -- so there is no NavigateKitchen; base motion is only repositioning
    between carries). Physical order, always:
      1. lid off:   PickPlaceStoveToCounter [pick]/[place]  saucepan -> counter
      2. corn:      PickPlaceCounterToStove [pick]/[place]  plate -> saucepan
      3. corn:      PickPlaceCounterToStove [pick]/[place]  plate -> saucepan
      4. lid on:    PickPlaceCounterToStove [pick]/[place]  counter -> saucepan
    plus a terminal 'done' -- 9 segments, expected_done_idx = 8. The two corns go
    in in either order.

    The LID (`saucepan_auxiliary`) is the only object handled twice: it has two
    grasp runs (the removal, then the replacement). Grasp attribution is
    sticky/exclusive (`_hold_assignment`) so the big lid resting on the counter
    can't steal a corn's frames while it is carried past. Each grasp run becomes
    one PickPlace; the pick->place boundary is when the held object starts moving
    (displacement from where it was grasped), and the release ends the run.
    """

    ENV_NAME = "BoilCorn"

    CORNS = ("corn1", "corn2")
    LID = "saucepan_auxiliary"

    GRASP_MAX = 0.08  # corn is elongated / grabbed off-center (min approach ~0.07)
    HOLD_MARGIN = 0.03
    MIN_RUN = 5
    MERGE_GAP = 50  # same-object grasp runs closer than this are one carry
    MIN_CARRY_DISP = 0.10  # a real carry moves the object at least this far
    MOVE_TH = 0.05
    MOVE_SUSTAIN = 8
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.movable = (*self.CORNS, self.LID)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.movable}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.corn_lang = env.get_obj_lang("corn1")
        except Exception:
            self.corn_lang = "corn"

        self.extra_subtasks = set()  # both names are bank atomics
        self.expected_done_idx = 8
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.movable:
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
        """Sticky per-frame grasp attribution to at most one movable object."""
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

    def _runs(self, held):
        """Contiguous held runs (start, end_inclusive) of at least MIN_RUN."""
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
        """Real carry runs for one object: grasp runs merged across small gaps
        (a brief pre-grasp touch / mid-carry regrasp is one carry) and kept only
        if the object actually travels MIN_CARRY_DISP. The lid's removal and
        replacement stay separate (their gap spans the whole corn phase)."""
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

    def _carry(self, opos, a, b):
        """(t_move, release) for a single grasp run [a, b], with the pick->place
        boundary at the first sustained displacement from the grasp position."""
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
        opos = {o: np.array([h[o] for h in history]) for o in self.movable}
        grip = np.array([h["grip"] for h in history])

        gd_arr = np.stack(
            [np.linalg.norm(grip - opos[o], axis=1) for o in self.movable], axis=0
        )
        assign = self._hold_assignment(gd_arr)

        # events: (grasp_start, end, object) -- corns yield one carry, the lid two
        events = []
        for k, o in enumerate(self.movable):
            for (a, b) in self._carries(opos[o], assign == k):
                events.append((a, b, o))
        events.sort()

        # which lid run is the removal (first) vs the replacement (last)
        lid_runs = [e for e in events if e[2] == self.LID]
        lid_off_start = lid_runs[0][0] if lid_runs else None
        corn_seen = 0

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for (a, b, o) in events:
            t_move, release = self._carry(opos[o], a, b)
            if o == self.LID:
                if a == lid_off_start:  # removal: saucepan -> counter
                    emit(t_move, SUBTASK_LID_OFF, STAGE_PICK, "pick up the lid from the saucepan")
                    emit(release, SUBTASK_LID_OFF, STAGE_PLACE, "place the lid on the counter")
                else:  # replacement: counter -> saucepan
                    emit(t_move, SUBTASK_TO_POT, STAGE_PICK, "pick up the lid from the counter")
                    emit(release, SUBTASK_TO_POT, STAGE_PLACE, "place the lid on the saucepan")
            else:  # corn: plate -> saucepan
                which = "the other" if corn_seen == 1 else "a"
                corn_seen += 1
                emit(t_move, SUBTASK_TO_POT, STAGE_PICK,
                     f"pick up {which} {self.corn_lang} from the plate")
                emit(release, SUBTASK_TO_POT, STAGE_PLACE,
                     f"place the {self.corn_lang} in the saucepan")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        n_lid = len(lid_runs)
        if n_lid != 2 or corn_seen != 2:
            review.append(
                f"expected lid handled twice + 2 corns, got lid x{n_lid}, corn x{corn_seen}"
            )

        debug = {
            "n_events": len(events),
            "lid_runs": n_lid,
            "corns": corn_seen,
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

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        # 1 lid-off (StoveToCounter) + 3 CounterToStove (2 corns + lid-on)
        if count(SUBTASK_LID_OFF, STAGE_PICK) != 1:
            issues.append(f"expected 1 {SUBTASK_LID_OFF}, found {count(SUBTASK_LID_OFF, STAGE_PICK)}")
        if count(SUBTASK_TO_POT, STAGE_PICK) != 3:
            issues.append(f"expected 3 {SUBTASK_TO_POT} picks, found {count(SUBTASK_TO_POT, STAGE_PICK)}")
        n_pick = count(SUBTASK_LID_OFF, STAGE_PICK) + count(SUBTASK_TO_POT, STAGE_PICK)
        n_place = count(SUBTASK_LID_OFF, STAGE_PLACE) + count(SUBTASK_TO_POT, STAGE_PLACE)
        if n_pick != 4 or n_place != 4:
            issues.append(f"expected 4 picks + 4 places, found {n_pick} + {n_place}")
        return issues
