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


# The toaster oven starts OPEN with its temperature knob turned HIGH; a baguette sits on a
# plate on the counter. Pull the rack out, put the baguette on it, close the door, then turn
# the temperature DOWN to the lowest setting. All four are bank atomics (doc count = 5,
# non-mobile): SlideToasterOvenRack + PickPlaceCounterToToasterOven (pick/place) +
# CloseToasterOvenDoor + AdjustToasterOvenTemperature.
SUBTASK_RACK = "SlideToasterOvenRack"
SUBTASK_MOVE = "PickPlaceCounterToToasterOven"
SUBTASK_CLOSE = "CloseToasterOvenDoor"
SUBTASK_TEMP = "AdjustToasterOvenTemperature"

DONE_LANG = "task complete"


@register_labeler
class ToastBaguetteLabeler(BaseSubtaskLabeler):
    """Labeler for the ToastBaguette composite task.

    Task (non-mobile): the toaster oven starts open with the temperature knob high; a
    baguette is on a plate on the counter. Pull the rack out, place the baguette on it,
    close the door fully, and turn the temperature down to the lowest setting.

    Doc-canonical count = 5: SlideToasterOvenRack (out) + PickPlaceCounterToToasterOven
    [pick]/[place] + CloseToasterOvenDoor + AdjustToasterOvenTemperature. -> 6 segments + done.

    Boundaries (from replayed sim state):
      - t_rack:  the rack reaches its OUT plateau (slid out first; it slides back in later as
                 the door closes -- that return folds into close, not a 2nd SlideRack).
      - t_pick:  the baguette LIFTS off the plate (dz > LIFT_TOL) while grasped.
      - t_place: the gripper RELEASES the baguette onto the rack.
      - t_close: the door returns near closed (< CLOSE_ABS).
      - t_temp:  the temperature knob reaches near its MIN (starts high, turned down last).
    """

    ENV_NAME = "ToastBaguette"

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
    DOOR_OPEN_FRAC = 0.90
    CLOSE_ABS = 0.15
    CLOSE_SUSTAIN = 3
    DOOR_DESCEND_FRAC = 0.85  # door/temp below this fraction of its peak == starting to move
    RACK_IN_FRAC = 0.30    # rack back below this fraction of its out peak == slid in
    RACK_MIN = 0.05        # min out travel for a real slide
    TEMP_MIN_FRAC = 0.12   # temp knob <= this fraction of its START value == turned to min
    TEMP_SUSTAIN = 3
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["baguette"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        to = env.toaster_oven
        jn = to._joint_names

        def _adr(key):
            n = jn.get(key)
            return env.sim.model.get_joint_qpos_addr(n) if n and n in env.sim.model.joint_names else None

        self.door_adr = _adr("door")
        self.temp_adr = _adr("temperature")
        prefix = to.naming_prefix
        self.rec_names = [n for n in env.sim.model.joint_names
                          if n and n.startswith(prefix) and ("rack" in n or "tray" in n)]
        self.rec_adrs = [env.sim.model.get_joint_qpos_addr(n) for n in self.rec_names]
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 5

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(env.sim.data.qpos[self.door_adr]) if self.door_adr is not None else 0.0,
            "temp": float(env.sim.data.qpos[self.temp_adr]) if self.temp_adr is not None else 0.0,
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
    def _rack_in_plateau(rec, t_place, t_end, rpeak, W=10, min_frac=0.4, flat=0.08):
        """Frame where the gripper's inward push of the rack PLATEAUS between t_place and
        t_end -- the rack has moved in by >= min_frac of its out travel and then stays flat
        for W frames (before the door-close nudges it the rest). None if the gripper did not
        slide it in, or the plateau is < 5 frames after the place."""
        T = len(rec)
        if rpeak <= 0.05 or T == 0:
            return None
        base = float(rec[min(int(t_place), T - 1)])
        for t in range(int(t_place), max(int(t_place) + 1, int(t_end) - W)):
            if (base - float(rec[t])) >= min_frac * rpeak and \
               abs(float(rec[t]) - float(rec[min(t + W, T - 1)])) < flat * rpeak:
                return t if (t - int(t_place)) >= 5 else None
        return None

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        temp = np.array([h["temp"] for h in history])
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

        # baguette lifts off the plate (while grasped) -- computed FIRST so the rack-out
        # window excludes the placement jostle (which can exceed the slide-out amplitude).
        objz = obj[:, 2]
        rest_z = float(np.median(objz[: self.REST_W]))
        lifted = (objz - rest_z > self.LIFT_TOL) & (gd < self.GRASP_TH)
        t_pick = self._sustained_start(lifted, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = self._sustained_start(objz - rest_z > self.LIFT_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(1, T // 2)

        # rack slide out (before the pick); amplitude up to the lift only; conditional
        rec_rest = float(np.median(rec[: self.REST_W]))
        peak_out = float(rec[: max(t_pick, 1)].max() - rec_rest)
        has_slide = peak_out > 0.04
        if has_slide:
            t_rack = self._sustained_start(rec - rec_rest >= self.RACK_OUT_FRAC * peak_out, 0, self.MOVE_SUSTAIN)
            t_rack = int(min(t_rack if t_rack is not None else 0, t_pick))
        else:
            t_rack = 0

        # release onto the rack
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                        t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        # close the door and turn the temperature DOWN to min. They overlap; in some demos the
        # temp reaches min before the door finishes shutting, so the two are ordered by
        # whichever completes first (below) rather than forcing close-then-temp. ##demo_25/67
        t_close = self._sustained_start(door < self.CLOSE_ABS, t_place, self.CLOSE_SUSTAIN)
        if t_close is None:
            t_close = T - 1
        temp_start = float(np.median(temp[: self.REST_W]))
        if temp_start > 0.2:
            t_temp = self._sustained_start(temp <= self.TEMP_MIN_FRAC * temp_start, t_place, self.TEMP_SUSTAIN)
        else:
            t_temp = None
            review.append("temperature knob never turned down -- verify")
        if t_temp is None:
            t_temp = T - 1
        # slide the rack back IN as its own subtask, between the place and the close/temp. The
        # gripper pushes the rack in while the door is still open; the slide-in ENDS where that
        # inward push plateaus (before the door-close nudges it the rest), so the close/temp
        # start at that plateau. None when the door-close does all the sliding. ##user
        rec_rest = float(np.median(rec[: self.REST_W]))
        rec_peak = float(rec.max() - rec_rest)
        t_rack_in = self._rack_in_plateau(rec, t_place, min(t_close, t_temp), rec_peak)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if has_slide and t_rack > 0:
            emit(t_rack, SUBTASK_RACK, STAGE_EXECUTE, f"slide the toaster oven {rec_word} out")
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the baguette from the plate")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the baguette on the toaster oven {rec_word}")
        if t_rack_in is not None and t_rack_in > cursor:
            emit(t_rack_in, SUBTASK_RACK, STAGE_EXECUTE, f"slide the toaster oven {rec_word} in")
        # close + temperature-down, in the order they complete (they overlap)
        close_seg = (t_close, SUBTASK_CLOSE, STAGE_EXECUTE, "close the toaster oven door")
        temp_seg = (t_temp, SUBTASK_TEMP, STAGE_EXECUTE,
                    "set the toaster oven temperature to the lowest setting")
        for end, nm, st, lg in sorted([close_seg, temp_seg], key=lambda s: s[0]):
            emit(end, nm, st, lg)
        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_rack": t_rack, "t_pick": t_pick, "t_place": t_place, "t_close": t_close,
            "t_temp": t_temp, "rec_word": rec_word, "temp_start": round(temp_start, 2),
            "n_segments": len(segments),
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
            issues.append("expected 1 baguette pick")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append("expected 1 baguette place")
        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseToasterOvenDoor, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_TEMP) != 1:
            issues.append(f"expected 1 AdjustToasterOvenTemperature, found {count(SUBTASK_TEMP)}")
        return issues
