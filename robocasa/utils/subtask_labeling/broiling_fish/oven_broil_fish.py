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


# Put the fish on the (slid-out) oven rack, slide the rack back into the oven, then
# close the oven door. All three are bank atomics -- no extra_subtasks.
SUBTASK_MOVE = "PickPlaceCounterToOven"
SUBTASK_RACK = "SlideOvenRack"
SUBTASK_CLOSE = "CloseOven"

RACK_LANG = "slide the oven rack in"
CLOSE_LANG = "close the oven door"
DONE_LANG = "task complete"


@register_labeler
class OvenBroilFishLabeler(BaseSubtaskLabeler):
    """Labeler for the OvenBroilFish composite task.

    Task: the oven starts OPEN with its rack slid OUT (scene setup); a fish (on a
    plate) sits on the counter. Place the fish on the rack, slide the rack back into
    the oven, then close the oven door.

    Doc-canonical count = 3 (`mobile: Yes`, but the base does not drive -- the oven
    is beside the counter): PickPlaceCounterToOven + SlideOvenRack + CloseOven. The
    pick-place has pick/place stages -> 4 segments + done = 5 (expected_done_idx = 4).

    Boundaries:
      - t_move (fish pick end): the fish first displaces MOVE_TOL from the counter.
      - t_placed (fish place end): the gripper releases the fish onto the rack
                 (grip->fish distance rises above the carry hold level).
      - t_rack_in (SlideOvenRack end): the moving rack (auto-detected as the rack
                 joint with the largest travel) reaches its inward plateau -- it
                 starts slid OUT and is pushed IN toward ~0.
      - t_close (CloseOven end): the door joint returns near closed (< CLOSE_ABS);
                 it starts open (~1.15) and is swung shut last.
    """

    ENV_NAME = "OvenBroilFish"

    MOVE_TOL = 0.12  # fish displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_move used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.05  # grip->fish rising this far above carry level == let go
    # (modest: after letting go the gripper hovers near the rack to push it, so
    # grip->fish only rises a little -- a larger margin misses the real release)
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20  # a carry lasts at least this many frames before a release counts
    RACK_MOVE_TH = 0.05  # min rack travel for a real slide to have happened
    RACK_SETTLE = 0.03  # rack is at its inward plateau within this of its min
    DOOR_OPEN_FRAC = 0.90  # door >= this fraction of its peak == still (near) fully open
    CLOSE_ABS = 0.15  # door joint below this == shut (open ~1.15)
    CLOSE_SUSTAIN = 3
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.fish_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fish_lang = env.get_obj_lang("obj")
        except Exception:
            self.fish_lang = "fish"

        # rack destination language: top/bottom for a multi-rack oven, else "the rack"
        oven = env.oven
        rack_level = ep_meta.get("rack_level", 0)
        if oven.has_multiple_rack_levels():
            self.rack_place_lang = "top" if rack_level == 1 else "bottom"
        else:
            self.rack_place_lang = None

        self.door_qpos_adr = env.sim.model.get_joint_qpos_addr(oven._joint_names["door"])
        prefix = oven.naming_prefix
        rack_names = [
            n for n in env.sim.model.joint_names
            if n.startswith(prefix) and "rack" in n and n.endswith("_joint")
        ]
        self.rack_qpos_adrs = [env.sim.model.get_joint_qpos_addr(n) for n in rack_names]

        self.extra_subtasks = set()
        self.expected_done_idx = 4
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "fish": env.sim.data.body_xpos[self.fish_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(env.sim.data.qpos[self.door_qpos_adr]),
            "racks": np.array([float(env.sim.data.qpos[a]) for a in self.rack_qpos_adrs]),
        }

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        fish = np.array([h["fish"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        racks = np.array([h["racks"] for h in history])  # (T, n_rack)

        # fish pick -> place: displacement + gripper release
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

        # slide the rack in: the moving rack (largest travel) is pushed from its
        # slid-out start toward ~0. SlideOvenRack ends when the rack reaches its
        # inward plateau BEFORE the door starts closing -- in some demos the robot
        # only slides the rack partway and the DOOR-CLOSE pushes it the rest of the
        # way in, so a global-min test lands t_rack_in on the door-close and
        # collapses CloseOven (variant B).
        rrange = racks.max(axis=0) - racks.min(axis=0)
        mover = int(np.argmax(rrange)) if racks.shape[1] else 0
        rack = racks[:, mover] if racks.shape[1] else np.zeros(T)

        # the fish place ends at the gripper release OR when the rack first starts
        # sliding in, whichever comes first -- in a few demos the robot pushes the
        # rack in while still lowering/holding the fish, so the release fires after
        # the slide has already begun (which would collapse SlideOvenRack).
        rack_out = float(np.median(rack[: self.REST_W]))
        slide_start = self._sustained_start(rack < rack_out - self.RACK_MOVE_TH, t_move, self.MOVE_SUSTAIN)
        t_place_end = min(t_placed, slide_start) if slide_start is not None else t_placed
        t_place_end = int(max(t_place_end, t_move))

        peak_door = float(door.max())
        descend = np.where(door >= self.DOOR_OPEN_FRAC * peak_door)[0]
        t_descend = int(descend[-1]) + 1 if descend.size else T - 1
        t_descend = int(min(max(t_descend, t_place_end), T))
        win = rack[t_place_end: max(t_descend, t_place_end + 1)]
        win_min = float(win.min()) if win.size else float(rack[min(t_place_end, T - 1)])
        t_rack_in = self._sustained_start(rack <= win_min + self.RACK_SETTLE, t_place_end, 3)
        if t_rack_in is None:
            t_rack_in = t_descend
        t_rack_in = int(min(t_rack_in, t_descend))

        # close the door: it returns near closed after the rack is in
        t_close = self._sustained_start(door < self.CLOSE_ABS, t_rack_in, self.CLOSE_SUSTAIN)
        if t_close is None:
            t_close = T - 1

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.fish_lang} from the counter")
        if self.rack_place_lang is not None:
            place_lang = f"place the {self.fish_lang} on the {self.rack_place_lang} rack of the oven"
        else:
            place_lang = f"place the {self.fish_lang} on the rack of the oven"
        emit(t_place_end, SUBTASK_MOVE, STAGE_PLACE, place_lang)
        emit(t_rack_in, SUBTASK_RACK, STAGE_EXECUTE, RACK_LANG)
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if rrange[mover] < self.RACK_MOVE_TH:
            review.append("rack never slid in -- check SlideOvenRack")
        if not np.any(door < self.CLOSE_ABS):
            review.append("door never closed -- check CloseOven")

        debug = {
            "t_move": t_move,
            "t_placed": t_placed,
            "t_rack_in": t_rack_in,
            "t_close": t_close,
            "rack_mover": mover,
            "rack_travel": round(float(rrange[mover]), 3),
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

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 fish pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 fish place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_RACK) != 1:
            issues.append(f"expected 1 SlideOvenRack, found {count(SUBTASK_RACK)}")
        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseOven, found {count(SUBTASK_CLOSE)}")
        return issues
