import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# The toaster oven starts OPEN; a bagel sits on a plate on the counter. Pull the rack out,
# put the bagel on it, close the door, then set the timer to turn the oven on. Bank atomics
# (doc count = 5, non-mobile): SlideToasterOvenRack + PickPlaceCounterToToasterOven
# (pick/place) + CloseToasterOvenDoor + TurnOnToasterOven.
SUBTASK_RACK = "SlideToasterOvenRack"
SUBTASK_MOVE = "PickPlaceCounterToToasterOven"
SUBTASK_CLOSE = "CloseToasterOvenDoor"
SUBTASK_TIMER = "TurnOnToasterOven"

DONE_LANG = "task complete"


@register_labeler
class ToastBagelLabeler(BaseSubtaskLabeler):
    """Labeler for the ToastBagel composite task.

    Task (non-mobile): the toaster oven starts open; a bagel is on a plate on the counter.
    Pull the rack out, place the bagel on it, close the door, and set the timer.

    Doc-canonical count = 5: SlideToasterOvenRack (out) + PickPlaceCounterToToasterOven
    [pick]/[place] + CloseToasterOvenDoor + TurnOnToasterOven. -> 6 segments + done.

    Boundaries (from replayed sim state):
      - t_rack:  the rack reaches its OUT plateau (pulled out first; it is pushed back in
                 before/with the door close -- that return folds into the close segment).
      - t_pick:  the bagel LIFTS off the plate (dz > LIFT_TOL) while grasped.
      - t_place: the gripper RELEASES the bagel onto the rack.
      - t_close: the door returns near closed (< CLOSE_ABS).
      - t_timer: the timer knob is advanced (>= TIMER_ON), turning the oven on, last.
    """

    ENV_NAME = "ToastBagel"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.06
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.09
    LIFT_TOL = 0.04
    RACK_OUT_FRAC = 0.85
    CLOSE_ABS = 0.15
    CLOSE_SUSTAIN = 3
    DOOR_DESCEND_FRAC = 0.85  # door below this fraction of its open peak == starting to close
    RACK_IN_FRAC = 0.30    # rack back below this fraction of its out peak == slid in
    RACK_MIN = 0.05        # min out travel for a real slide
    TIMER_ON = 0.10        # timer knob past this == set / oven on
    TIMER_SUSTAIN = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["bagel"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        to = env.toaster_oven
        jn = to._joint_names

        def _adr(key):
            n = jn.get(key)
            return env.sim.model.get_joint_qpos_addr(n) if n and n in env.sim.model.joint_names else None

        self.door_adr = _adr("door")
        self.time_adr = _adr("time")
        prefix = to.naming_prefix
        self.rec_names = [n for n in env.sim.model.joint_names
                          if n and n.startswith(prefix) and ("rack" in n or "tray" in n)]
        self.rec_adrs = [env.sim.model.get_joint_qpos_addr(n) for n in self.rec_names]
        try:
            self.multi_rack = bool(to.has_multiple_rack_levels())
        except Exception:
            self.multi_rack = len(self.rec_names) >= 2
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 5

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(env.sim.data.qpos[self.door_adr]) if self.door_adr is not None else 0.0,
            "time": float(env.sim.data.qpos[self.time_adr]) if self.time_adr is not None else 0.0,
            "recs": np.array([float(env.sim.data.qpos[a]) for a in self.rec_adrs]),
        }

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

    @staticmethod
    def _rack_in_plateau(rec, t_place, t_close, rpeak, W=10, min_frac=0.4, flat=0.08):
        """Frame where the gripper's inward push of the rack PLATEAUS between t_place and
        t_close -- the rack has moved in by >= min_frac of its out travel and then stays flat
        for W frames (before the door-close nudges it the rest). None if the gripper did not
        slide it in (the door-close does the sliding) or the plateau is < MOVE_SUSTAIN after
        the place (it went in during the place)."""
        T = len(rec)
        if rpeak <= 0.05 or T == 0:
            return None
        base = float(rec[min(int(t_place), T - 1)])
        for t in range(int(t_place), max(int(t_place) + 1, int(t_close) - W)):
            if (base - float(rec[t])) >= min_frac * rpeak and \
               abs(float(rec[t]) - float(rec[min(t + W, T - 1)])) < flat * rpeak:
                return t if (t - int(t_place)) >= 5 else None
        return None

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        timer = np.array([h["time"] for h in history])
        recs = np.array([h["recs"] for h in history])
        if recs.shape[1]:
            mover = int(np.argmax(recs.max(axis=0) - recs.min(axis=0)))
            rec = recs[:, mover]
            rec_word = "tray" if "tray" in self.rec_names[mover] else "rack"
        else:
            rec = np.zeros(T)
            rec_word = "rack"
        gd = np.linalg.norm(grip - obj, axis=1)
        review = []

        # bagel lifts off the plate (while grasped) -- computed FIRST so the rack-out window
        # excludes the placement jostle, which can push the rack higher than the slide-out.
        objz = obj[:, 2]
        rest_z = float(np.median(objz[: self.REST_W]))
        lifted = (objz - rest_z > self.LIFT_TOL) & (gd < self.GRASP_TH)
        t_pick = self._sustained_start(lifted, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = self._sustained_start(objz - rest_z > self.LIFT_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(1, T // 2)

        # rack slide out (before the pick); amplitude measured only up to the lift so the
        # placement jostle is excluded. Conditional: some demos barely slide the rack.
        rec_rest = float(np.median(rec[: self.REST_W]))
        peak_out = float(rec[: max(t_pick, 1)].max() - rec_rest)
        has_slide = peak_out > 0.04
        if has_slide:
            t_rack = self._sustained_start(rec - rec_rest >= self.RACK_OUT_FRAC * peak_out, 0, self.MOVE_SUSTAIN)
            t_rack = int(min(t_rack if t_rack is not None else 0, t_pick))
        else:
            t_rack = 0

        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                        t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        # close the door
        t_close = self._sustained_start(door < self.CLOSE_ABS, t_place, self.CLOSE_SUSTAIN)
        if t_close is None:
            t_close = T - 1
        # slide the rack back IN as its own subtask, between the place and the close. The
        # gripper pushes the rack in while the door is still OPEN, then the door-close nudges
        # it the last bit. The slide-in ENDS where the gripper's inward push PLATEAUS (the
        # rack has moved in and stops), so the close starts at that plateau and covers the
        # reach-to-door + door motion. Emit only when the gripper actually slid it in (else
        # the door-close does all the sliding → no distinct slide-in). ##user
        rec_rest = float(np.median(rec[: self.REST_W]))
        rec_peak = float(rec.max() - rec_rest)
        t_rack_in = self._rack_in_plateau(rec, t_place, t_close, rec_peak)

        # set the timer (advanced from ~0, done last)
        if float(timer.max()) > self.TIMER_ON:
            t_timer = self._sustained_start(timer >= self.TIMER_ON, t_close, self.TIMER_SUSTAIN)
        else:
            t_timer = None
            review.append("timer never set -- verify")
        if t_timer is None:
            t_timer = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if has_slide and t_rack > 0:
            art = "a" if self.multi_rack else "the"
            emit(t_rack, SUBTASK_RACK, STAGE_EXECUTE, f"slide {art} toaster oven {rec_word} out")
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the bagel from the plate")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the bagel on the toaster oven {rec_word}")
        if t_rack_in is not None and t_rack_in > cursor:
            # slide-IN is always "the" -- the specific rack was already pulled out
            emit(t_rack_in, SUBTASK_RACK, STAGE_EXECUTE, f"slide the toaster oven {rec_word} in")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, "close the toaster oven door")
        emit(t_timer, SUBTASK_TIMER, STAGE_EXECUTE, "set the timer to turn on the toaster oven")
        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_rack": t_rack, "t_pick": t_pick, "t_place": t_place, "t_close": t_close,
            "t_timer": t_timer, "rec_word": rec_word, "n_segments": len(segments),
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

        if count(SUBTASK_RACK) not in (0, 1, 2):
            issues.append(f"expected 0-2 SlideToasterOvenRack, found {count(SUBTASK_RACK)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append("expected 1 bagel pick")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append("expected 1 bagel place")
        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseToasterOvenDoor, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_TIMER) != 1:
            issues.append(f"expected 1 TurnOnToasterOven, found {count(SUBTASK_TIMER)}")
        return issues
