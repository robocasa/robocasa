import numpy as np
from robosuite.utils import transform_utils as T

import robocasa.utils.object_utils as OU
from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# Bank-extension skills for the pan tilt-pour + return (no bank atomic fits a bare
# pick / a tilt-pour / a return-to-stove).
PICK_SUBTASK = "Pick"
TILT_SUBTASK = "Tilt"
PLACE_SUBTASK = "Place"

PICK_LANG = "pick up the pan from the stove"
PLACE_LANG = "return the pan to the stove"
DONE_LANG = "task complete"


@register_labeler
class PanTransferLabeler(BaseSubtaskLabeler):
    """Labeler for the PanTransfer composite task.

    A pan (`vegetable_container`) holding one vegetable sits on the stove; a plate
    sits on the counter. The robot picks up the pan, tilts it to dump the vegetable
    onto the plate, and returns the pan to the stove. The pan is held throughout,
    so the three subtasks are one continuous grasp split geometrically: Pick (grab
    + lift off the stove) -> Tilt (carry to the plate + tilt-pour) -> Place (carry
    back + set the pan down).

    Detection off privileged sim state: the pan's up-axis (its local +z in world)
    rotates far from rest (>POUR_TILT_TH) while the pan is lifted -> the tilt-pour;
    the Pick->Tilt boundary is the lift-off (last grounded frame before the pour),
    the Tilt->Place boundary is the end of the pour, and the Place ends when the pan
    is released (later of the contact-grasp end +pad and the finger-width open).
    """

    ENV_NAME = "PanTransfer"

    PAN = "vegetable_container"
    VEG = "vegetable"

    LIFT_TH = 0.05          # height above rest for the pan to count as lifted
    POUR_TILT_TH = 30.0     # degrees the pan's up-axis must rotate to count as pouring
                            # (pours run 38-82 deg; carry/handling stays <=23)
    POUR_MIN_FRAMES = 3     # sustained tilted-while-lifted frames to call it a pour
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_bid = env.obj_body_id[self.PAN]
        self.veg_bid = env.obj_body_id[self.VEG]
        self.pan_geoms = env.objects[self.PAN].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        try:
            self.veg_lang = env.get_obj_lang(self.VEG)
        except Exception:
            self.veg_lang = "vegetable"
        self.extra_subtasks = {PICK_SUBTASK, TILT_SUBTASK, PLACE_SUBTASK}
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        q = np.array(env.sim.data.body_xquat[self.pan_bid])  # wxyz
        pan_up = T.quat2mat(q[[1, 2, 3, 0]])[:, 2].copy()
        try:
            on_plate = int(OU.check_obj_in_receptacle(env, self.VEG, "plate"))
        except Exception:
            on_plate = 0
        try:
            in_pan = int(OU.check_obj_in_receptacle(env, self.VEG, self.PAN))
        except Exception:
            in_pan = 0
        return {
            "pan_pos": env.sim.data.body_xpos[self.pan_bid].copy(),
            "pan_up": pan_up,
            "veg_pos": env.sim.data.body_xpos[self.veg_bid].copy(),
            "on_plate": on_plate,
            "in_pan": in_pan,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.pan_geoms)),
        }

    def _fw_open(self, fw, fw_open, grasp_t, hi):
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return None
        held = float(np.min(seg[: min(25, seg.size)]))
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return None

    def _release(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        fwo = self._fw_open(fw, fw_open, grasp_t, hi)
        gc = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
            if fwo is not None:
                rel = max(rel, fwo)
        elif fwo is not None:
            rel = fwo
        else:
            rel = setdown
        return int(min(max(rel, grasp_t + 1), hi))

    def resolve(self, history):
        T_ = len(history)
        pan_pos = np.stack([h["pan_pos"] for h in history])
        pan_up = np.stack([h["pan_up"] for h in history])
        veg_pos = np.stack([h["veg_pos"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())

        z0 = float(np.median(pan_pos[: self.REST_W, 2]))
        tilt = np.degrees(np.arccos(np.clip(pan_up @ pan_up[0], -1.0, 1.0)))
        lifted = (pan_pos[:, 2] - z0) > self.LIFT_TH
        on_plate = np.array([h["on_plate"] for h in history])
        in_pan = np.array([h["in_pan"] for h in history])

        # Pick -> Tilt = the pan's FIRST sustained lift-off from the stove. (Not
        # the last grounded frame before the pour: the pan often dips back down
        # while tilting low over the plate, which would push the boundary late and
        # collapse the Tilt.)
        lift_off = 0
        for t in range(T_ - 2):
            if lifted[t] and lifted[t + 1] and lifted[t + 2]:
                lift_off = t
                break

        # Tilt -> Place = when the vegetable has left the pan and landed on the
        # plate (the item is delivered). Do NOT gate on tilt angle -- a shallow
        # ~30 deg tilt can still dump the veg; the delivery is the true signal.
        delivered = np.where((on_plate[lift_off:] > 0) & (in_pan[lift_off:] == 0))[0]
        pour_detected = delivered.size > 0
        if pour_detected:
            pour_end = int(min(max(lift_off + int(delivered[0]), lift_off + 1), T_ - 1))
        else:
            # veg never registered delivered -> collapse so the demo is flagged.
            pour_end = lift_off

        # These demos often stop the instant the pan is down and the gripper has
        # just cleared it, so cap the release at T-1 to guarantee a terminal 'done'.
        release = min(self._release(grasp, pan_pos, fw, fw_open, max(pour_end, 1), T_), T_ - 1)

        segments = [
            (lift_off, PICK_SUBTASK, STAGE_PICK, PICK_LANG),
            (pour_end, TILT_SUBTASK, STAGE_TILT,
             f"tilt the pan to dump the {self.veg_lang} onto the plate"),
            (release, PLACE_SUBTASK, STAGE_PLACE, PLACE_LANG),
            (T_, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = 3

        veg_disp = float(np.linalg.norm(veg_pos[:, :2] - veg_pos[0, :2], axis=1).max())
        debug = {
            "lift_off": lift_off,
            "pour_end": pour_end,
            "release": release,
            "max_tilt": round(float(tilt.max()), 1),
            "pour_detected": pour_detected,
            "veg_disp": round(veg_disp, 2),
            "n_segments": len(segments),
        }
        return self.build_result(T_, segments, debug=debug)

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = set(str(v) for v in result.subtask_name[firsts])
        for req in (PICK_SUBTASK, TILT_SUBTASK, PLACE_SUBTASK):
            if req not in names:
                issues.append(f"missing {req} subtask")
        dbg = getattr(result, "debug", None) or {}
        if not dbg.get("pour_detected", True):
            issues.append("vegetable never registered on the plate (dump not detected)")
        if dbg.get("veg_disp", 1.0) < 0.1:
            issues.append("vegetable barely moved (dump not detected)")
        return issues
