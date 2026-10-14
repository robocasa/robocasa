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


# Put the fish on the toaster oven's (slid-out) rack/tray, slide it back in, close
# the door, then turn the temperature knob to max. All four are bank atomics -- no
# extra_subtasks.
SUBTASK_MOVE = "PickPlaceCounterToToasterOven"
SUBTASK_RACK = "SlideToasterOvenRack"
SUBTASK_CLOSE = "CloseToasterOvenDoor"
SUBTASK_TEMP = "AdjustToasterOvenTemperature"

CLOSE_LANG = "close the toaster oven door"
TEMP_LANG = "set the toaster oven temperature to the max"
DONE_LANG = "task complete"


@register_labeler
class ToasterOvenBroilFishLabeler(BaseSubtaskLabeler):
    """Labeler for the ToasterOvenBroilFish composite task.

    Task: the toaster oven starts OPEN with its rack/tray slid OUT; a fish (on a
    plate) is on the counter. Place the fish on the rack/tray, slide it back in,
    close the door, then turn the temperature knob to max to start broiling.

    Doc-canonical count = 4 (`mobile: No`): PickPlaceCounterToToasterOven +
    SlideToasterOvenRack + CloseToasterOvenDoor + AdjustToasterOvenTemperature. The
    pick-place has pick/place stages -> 5 segments + done = 6 (expected_done_idx = 5).
    Structure = OvenBroilFish + a final temperature-set step. The receptacle is a
    "rack" or a "tray" depending on the unit (spoken accordingly).

    Boundaries (same rack/door handling as OvenBroilFish):
      - t_place_end: the gripper releases the fish OR the rack starts sliding,
                 whichever first.
      - t_rack_in: the receptacle reaches its inward plateau BEFORE the door starts
                 closing (a door-close can push it the last bit in).
      - t_close: the door returns near closed (< CLOSE_ABS).
      - t_temp: the temperature knob reaches near its max (it starts at ~0 and is
                turned up last).
    """

    ENV_NAME = "ToasterOvenBroilFish"

    MOVE_TOL = 0.12
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.05
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    RACK_MOVE_TH = 0.05
    RACK_SETTLE = 0.03
    DOOR_OPEN_FRAC = 0.90
    CLOSE_ABS = 0.15  # door joint below this == shut (open ~1.47)
    CLOSE_SUSTAIN = 3
    TEMP_FRAC = 0.90  # temp knob >= this fraction of its peak == set to max
    TEMP_SUSTAIN = 3
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.fish_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fish_lang = env.get_obj_lang("obj")
        except Exception:
            self.fish_lang = "fish"

        to = env.toaster_oven
        jn = to._joint_names

        def _adr(key):
            n = jn.get(key)
            return env.sim.model.get_joint_qpos_addr(n) if n and n in env.sim.model.joint_names else None

        self.door_adr = _adr("door")
        self.temp_adr = _adr("temperature")
        # the receptacle is a rack or a tray; a multi-level unit has one joint per
        # level (rack0/rack1 or tray0/tray1), only one of which actually slides, so
        # track them all and let resolve() pick the mover (largest travel).
        prefix = to.naming_prefix
        self.rec_names = [
            n for n in env.sim.model.joint_names
            if n and n.startswith(prefix) and ("rack" in n or "tray" in n)
        ]
        self.rec_adrs = [env.sim.model.get_joint_qpos_addr(n) for n in self.rec_names]

        rack_level = ep_meta.get("refs", {}).get("rack_level", 0)
        if to.has_multiple_rack_levels():
            self.rec_pos = "top" if rack_level == 1 else "bottom"
        else:
            self.rec_pos = None

        self.extra_subtasks = set()
        self.expected_done_idx = 5
        # a couple demos push the rack in very fast (the robot nudges it a few frames,
        # then the door-close finishes sliding it in -- variant B), so allow a short
        # SlideToasterOvenRack rather than flag those genuine fast slides.
        self.min_segment_frames = 5

    def per_step(self, env, t: int) -> dict:
        return {
            "fish": env.sim.data.body_xpos[self.fish_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(env.sim.data.qpos[self.door_adr]) if self.door_adr is not None else 0.0,
            "recs": np.array([float(env.sim.data.qpos[a]) for a in self.rec_adrs]),
            "temp": float(env.sim.data.qpos[self.temp_adr]) if self.temp_adr is not None else 0.0,
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

    def resolve(self, history):
        T = len(history)
        fish = np.array([h["fish"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        temp = np.array([h["temp"] for h in history])
        recs = np.array([h["recs"] for h in history])  # (T, n_rec)
        # the moving receptacle (largest travel) + whether it is a rack or a tray
        if recs.shape[1]:
            mover = int(np.argmax(recs.max(axis=0) - recs.min(axis=0)))
            rec = recs[:, mover]
            self.rec_word = "tray" if "tray" in self.rec_names[mover] else "rack"
        else:
            rec = np.zeros(T)
            self.rec_word = "rack"

        # fish pick -> place
        rest = np.median(fish[: self.REST_W], axis=0)
        disp = np.linalg.norm(fish - rest, axis=1)
        t_move = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_move = int(mv[0]) if mv.size else 0
        gd = np.linalg.norm(grip - fish, axis=1)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1

        # place ends at release OR when the receptacle starts sliding, whichever
        # first. The slide "commits" at the first below-threshold frame that is NOT
        # followed by a return to near-out -- a brief fish bump dips the rack then it
        # recovers, while the real slide stays in (demo_1 dips at ~t=260 but the real
        # slide is ~t=415).
        rec_out = float(np.median(rec[: self.REST_W]))
        below = rec < rec_out - self.RACK_MOVE_TH
        near_out = rec > rec_out - 0.5 * self.RACK_MOVE_TH
        # the receptacle "commits" to sliding after the LAST frame it is still near
        # out -- a brief fish bump dips it then it recovers, the real slide stays in.
        # slide_start = that last-out frame (the motion onset), so the slide segment
        # spans the whole active slide even when the rack only travels a little
        # before the door-close pushes it the rest of the way (variant B, demo_31).
        last_out = int(np.where(near_out)[0][-1]) if np.any(near_out) else -1
        commits = below & (np.arange(T) > last_out)
        if np.any(commits) and last_out >= 0:
            slide_start = last_out
        else:
            slide_start = None
        t_place_end = min(t_placed, slide_start) if slide_start is not None else t_placed
        t_place_end = int(max(t_place_end, t_move))

        # slide the rack/tray in: inward plateau BEFORE the door starts closing
        peak_door = float(door.max())
        descend = np.where(door >= self.DOOR_OPEN_FRAC * peak_door)[0]
        t_descend = int(descend[-1]) + 1 if descend.size else T - 1
        t_descend = int(min(max(t_descend, t_place_end), T))
        win = rec[t_place_end: max(t_descend, t_place_end + 1)]
        win_min = float(win.min()) if win.size else float(rec[min(t_place_end, T - 1)])
        t_rack_in = self._sustained_start(rec <= win_min + self.RACK_SETTLE, t_place_end, 3)
        if t_rack_in is None:
            t_rack_in = t_descend
        t_rack_in = int(min(t_rack_in, t_descend))

        # close the door
        t_close = self._sustained_start(door < self.CLOSE_ABS, t_rack_in, self.CLOSE_SUSTAIN)
        if t_close is None:
            t_close = T - 1

        # set the temperature to max: the temp knob is turned up last
        temp_peak = float(temp.max())
        if temp_peak > 0.1:
            t_temp = self._sustained_start(temp >= self.TEMP_FRAC * temp_peak, t_close, self.TEMP_SUSTAIN)
        else:
            t_temp = None
        if t_temp is None:
            t_temp = T - 1

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.fish_lang} from the counter")
        pos = f"{self.rec_pos} " if self.rec_pos else ""
        emit(t_place_end, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.fish_lang} on the {pos}{self.rec_word} of the toaster oven")
        emit(t_rack_in, SUBTASK_RACK, STAGE_EXECUTE, f"slide the toaster oven {self.rec_word} in")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(t_temp, SUBTASK_TEMP, STAGE_EXECUTE, TEMP_LANG)
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        rrange = float(rec.max() - rec.min())
        if rrange < self.RACK_MOVE_TH:
            review.append("rack/tray never slid in")
        if not np.any(door < self.CLOSE_ABS):
            review.append("door never closed")
        if temp_peak <= 0.1:
            review.append("temperature knob never turned")

        debug = {
            "t_move": t_move, "t_place_end": t_place_end, "t_rack_in": t_rack_in,
            "t_close": t_close, "t_temp": t_temp, "receptacle": self.rec_word,
            "temp_peak": round(temp_peak, 2), "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        # hand-reviewed manual overrides may drop a subtask on purpose (e.g. demo_103
        # has no SlideToasterOvenRack -- the door-close does all the sliding), so skip
        # the structural checks for them.
        if any("manually labeled" in str(r) for r in (getattr(result, "review", None) or [])):
            return issues
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

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 fish pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 fish place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_RACK) != 1:
            issues.append(f"expected 1 SlideToasterOvenRack, found {count(SUBTASK_RACK)}")
        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseToasterOvenDoor, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_TEMP) != 1:
            issues.append(f"expected 1 AdjustToasterOvenTemperature, found {count(SUBTASK_TEMP)}")
        return issues
