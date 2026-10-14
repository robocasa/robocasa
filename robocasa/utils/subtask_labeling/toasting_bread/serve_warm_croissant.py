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


# A warm croissant sits inside the (closed) toaster oven, on its rack; a plate is on the
# counter beside the oven. Open the door, slide the rack out, take the croissant off it and
# put it on the plate. OpenToasterOvenDoor / SlideToasterOvenRack / PickPlaceToasterOvenToCounter
# are all bank atomics (doc-canonical count = 3, non-mobile).
SUBTASK_OPEN = "OpenToasterOvenDoor"
SUBTASK_RACK = "SlideToasterOvenRack"
SUBTASK_MOVE = "PickPlaceToasterOvenToCounter"

DONE_LANG = "task complete"


@register_labeler
class ServeWarmCroissantLabeler(BaseSubtaskLabeler):
    """Labeler for the ServeWarmCroissant composite task.

    Task (non-mobile): the toaster oven starts CLOSED with a croissant on its rack; a plate
    sits on the adjacent counter. Open the door, slide the rack out, lift the croissant off
    the rack and place it on the plate.

    Doc-canonical count = 3: OpenToasterOvenDoor [execute] + SlideToasterOvenRack (out)
    [execute] + PickPlaceToasterOvenToCounter [pick]/[place]. -> 4 segments + done.

    Boundaries (from the replayed sim state):
      - t_open:  door reaches its open plateau (door starts ~0, opens to ~1.37).
      - t_rack:  the rack reaches its out plateau (starts ~0, slides to ~0.19). The croissant
                 rides out WITH the rack, so its pick is measured only AFTER the rack is out.
      - t_pick:  croissant first lifts off the (slid-out) rack while grasped.
      - t_place: the gripper RELEASES the croissant onto the plate.
    """

    ENV_NAME = "ServeWarmCroissant"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.06
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.09
    LIFT_TOL = 0.03
    DOOR_OPEN_FRAC = 0.80
    RACK_OUT_FRAC = 0.85
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["croissant"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.croissant = env.get_obj_lang("croissant")
        except Exception:
            self.croissant = "croissant"
        to = env.toaster_oven
        jn = to._joint_names
        n = jn.get("door")
        self.door_adr = (env.sim.model.get_joint_qpos_addr(n)
                         if n and n in env.sim.model.joint_names else None)
        prefix = to.naming_prefix
        self.rec_names = [nm for nm in env.sim.model.joint_names
                          if nm and nm.startswith(prefix) and ("rack" in nm or "tray" in nm)]
        self.rec_adrs = [env.sim.model.get_joint_qpos_addr(nm) for nm in self.rec_names]
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

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        recs = np.array([h["recs"] for h in history])
        if recs.shape[1]:
            mover = int(np.argmax(recs.max(axis=0) - recs.min(axis=0)))
            rec = recs[:, mover]
            rec_word = "tray" if "tray" in self.rec_names[mover] else "rack"
            import re as _re
            m = _re.search(r"(?:rack|tray)(\d+)", self.rec_names[mover])
            rec_level = int(m.group(1)) if m else None
        else:
            rec = np.zeros(T)
            rec_word = "rack"
            rec_level = None
        # the rack that actually slides out IS the croissant's rack; on a multi-level oven
        # name it (level 1 = top, 0 = bottom), else just "the rack".
        rack_pos = ("top" if rec_level == 1 else "bottom") if (self.multi_rack and rec_level is not None) else None
        gd = np.linalg.norm(grip - obj, axis=1)

        review = []
        # door open
        peak_door = float(door.max())
        if peak_door > 0.2:
            t_open = self._sustained_start(door >= self.DOOR_OPEN_FRAC * peak_door, 0, self.OPEN_SUSTAIN)
        else:
            t_open = None
            review.append("door never opened -- verify")
        if t_open is None:
            t_open = max(1, T // 6)

        # croissant pick: it rides OUT with the rack (horizontal, dz~0, and NOT grasped), so
        # neither the horizontal slide nor a transient grip-brush is the pick. The real pick
        # is the LIFT off the rack -- the croissant rises (dz > LIFT_TOL) while grasped
        # (gd < GRASP_TH). That is where the pick stage ends and the carry/place begins.
        objz = obj[:, 2]
        rest_z = float(np.median(objz[: self.REST_W]))
        lifted = (objz - rest_z > self.LIFT_TOL) & (gd < self.GRASP_TH)
        t_pick = self._sustained_start(lifted, t_open, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = self._sustained_start(objz - rest_z > self.LIFT_TOL, t_open, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(1, T // 2)

        # rack slide out (between the door opening and the croissant lifting off it).
        # Conditional: a few demos take the croissant out without sliding the rack.
        rec_rest = float(np.median(rec[: self.REST_W]))
        peak_rec = float(rec.max() - rec_rest)
        has_slide = peak_rec > 0.04
        if has_slide:
            t_rack = self._sustained_start(rec - rec_rest >= self.RACK_OUT_FRAC * peak_rec,
                                           t_open, self.MOVE_SUSTAIN)
            t_rack = int(min(max(t_rack if t_rack is not None else t_open, t_open), t_pick))
        else:
            t_rack = t_open
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                        t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the toaster oven door")
        if has_slide and t_rack > t_open:
            pos = f"{rack_pos} " if rack_pos else ""
            emit(t_rack, SUBTASK_RACK, STAGE_EXECUTE, f"slide the {pos}toaster oven {rec_word} out")
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.croissant} from the toaster oven")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.croissant} on the plate")
        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "t_rack": t_rack, "t_pick": t_pick, "t_place": t_place,
            "rec_word": rec_word, "n_segments": len(segments),
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

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenToasterOvenDoor, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_RACK) not in (0, 1):
            issues.append(f"expected 0-1 SlideToasterOvenRack, found {count(SUBTASK_RACK)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append("expected 1 croissant pick")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append("expected 1 croissant place")
        return issues
