import numpy as np

import robocasa.utils.object_utils as OU
from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# The atomic bank has no bare "pick up" or "wipe" skill, so both are declared
# extra_subtasks (non-bank names).
SUBTASK_PICK = "PickUpSponge"
SUBTASK_WIPE = "WipeTable"

PICK_LANG = "pick up the sponge from the counter"
WIPE_LANG = "wipe the table"
DONE_LANG = "task complete"


@register_labeler
class WipeTableLabeler(BaseSubtaskLabeler):
    """Labeler for the WipeTable composite task.

    Task (non-mobile): a sponge sits on the counter; the robot spawns at the sponge
    (`init_robot_here`). Two subtasks: pick up the sponge, then wipe the counter
    (press/rub it across the surface, then release). Success needs 5+ distinct
    counter contacts spread >= 0.1 m with the sponge finally released. The robot
    never drives -- no NavigateKitchen.

    Signals off privileged sim state: grasp onset = the first contact-grasp (fallback
    lift/displace); the wipe runs from the grasp until the sponge is released -- the
    later of the contact-grasp end (+pad) and the finger-width gripper-open. The
    sponge<->counter contact burst (OU.check_obj_any_counter_contact) marks the actual
    wiping (sanity-checked in validate).
    """

    ENV_NAME = "WipeTable"

    SPONGE = "sponge"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sponge_bid = env.obj_body_id[self.SPONGE]
        self.sponge_geoms = env.objects[self.SPONGE].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.extra_subtasks = {SUBTASK_PICK, SUBTASK_WIPE}
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        try:
            contact = int(OU.check_obj_any_counter_contact(env, self.SPONGE))
        except Exception:
            contact = 0
        return {
            "pos": env.sim.data.body_xpos[self.sponge_bid].copy(),
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.sponge_geoms)),
            "contact": contact,
        }

    def _grasp_onset(self, grasp_arr, obj):
        gi = np.where(grasp_arr > 0)[0]
        if gi.size:
            return int(gi[0])
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = np.where(obj[:, 2] > z0 + self.Z_LIFT)[0]
        if lifted.size:
            return int(lifted[0])
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1)
        w = np.where(disp > self.MOVE_TH)[0]
        return int(w[0]) if w.size else 0

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

    def _release_for(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
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
        T = len(history)
        pos = np.array([h["pos"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        contact = np.array([h["contact"] for h in history])

        grasp0 = self._grasp_onset(grasp, pos)
        release = self._release_for(grasp, pos, fw, fw_open, grasp0, T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(grasp0, SUBTASK_PICK, STAGE_PICK, PICK_LANG)
        emit(release, SUBTASK_WIPE, STAGE_EXECUTE, WIPE_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        return self.build_result(T, segments, debug={
            "grasp0": grasp0, "release": release,
            "contact_frames": int(contact.sum()), "n_segments": len(segments),
        })

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = set(str(v) for v in result.subtask_name[firsts])
        for req in (SUBTASK_PICK, SUBTASK_WIPE):
            if req not in names:
                issues.append(f"missing {req} subtask")
        if getattr(result, "debug", None) and result.debug.get("contact_frames", 0) < 3:
            issues.append("sponge barely contacted the counter (wipe not detected)")
        return issues
