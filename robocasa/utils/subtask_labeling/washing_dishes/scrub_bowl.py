import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Use the sponge to scrub the bowl inside the sink. The bank has no bare "pick up" or
# "scrub" skill, so both are extra_subtasks. Non-mobile (robot parked at the sink).
# Doc-canonical count = 2: PickUpSponge + ScrubBowl.
SUBTASK_PICK = "PickUpSponge"
SUBTASK_SCRUB = "ScrubBowl"

PICK_LANG = "pick up the sponge from the sink"
SCRUB_LANG = "scrub the bowl with the sponge"
DONE_LANG = "task complete"


@register_labeler
class ScrubBowlLabeler(BaseSubtaskLabeler):
    """Labeler for the ScrubBowl composite task.

    A sponge and a bowl sit in the sink; pick up the sponge, then scrub the bowl (rub the
    sponge over it, then release). Two subtasks: pick up the sponge -> scrub the bowl.

    Signals off privileged sim state (adapted from ScrubCuttingBoard): grasp onset is the
    sponge's first contact-grasp (fallback lift/displace); scrub runs from the grasp to the
    sponge's release (later of the contact-grasp end + pad and the finger-width open). The
    sponge<->bowl contact burst sanity-checks that scrubbing happened. Non-mobile -> no nav.
    """

    ENV_NAME = "ScrubBowl"

    SPONGE = "sponge"
    BOWL = "bowl"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    SCRUB_PAD = 15  # frames of linger after the last sponge<->bowl contact

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sponge_bid = env.obj_body_id[self.SPONGE]
        self.sponge_geoms = env.objects[self.SPONGE].contact_geoms
        self.bowl_geoms = env.objects[self.BOWL].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.extra_subtasks = {SUBTASK_PICK, SUBTASK_SCRUB}
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        try:
            touch = int(env.check_contact(self.sponge_geoms, self.bowl_geoms))
        except Exception:
            touch = 0
        return {
            "pos": env.sim.data.body_xpos[self.sponge_bid].copy(),
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.sponge_geoms)),
            "touch": touch,
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
        T = len(history)
        pos = np.array([h["pos"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        touch = np.array([h["touch"] for h in history])

        grasp0 = self._grasp_onset(grasp, pos)
        release = self._release(grasp, pos, fw, fw_open, grasp0, T)
        # Scrub ENDS at the last sponge<->bowl contact (+linger), NOT at the sponge release:
        # the robot often keeps holding the sponge after scrubbing, so a release-based end
        # would run to the last frame and leave no 'done'.
        tb = np.where(touch[grasp0:] > 0)[0]
        if tb.size:
            scrub_end = int(min(grasp0 + int(tb[-1]) + self.SCRUB_PAD, T - 1))
        else:
            scrub_end = int(min(release, T - 1))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(grasp0, SUBTASK_PICK, STAGE_PICK, PICK_LANG)
        emit(scrub_end, SUBTASK_SCRUB, STAGE_EXECUTE, SCRUB_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if int(touch.sum()) < 3:
            review.append("sponge barely contacted the bowl (scrub not detected)")
        result = self.build_result(T, segments, debug={
            "grasp0": grasp0, "scrub_end": scrub_end, "release": release,
            "touch_frames": int(touch.sum()), "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_PICK, SUBTASK_SCRUB}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = set(str(v) for v in result.subtask_name[firsts])
        for req in (SUBTASK_PICK, SUBTASK_SCRUB):
            if req not in names:
                issues.append(f"missing {req} subtask")
        dbg = getattr(result, "debug", None) or {}
        if dbg.get("touch_frames", 0) < 3:
            issues.append("sponge barely contacted the bowl (scrub not detected)")
        return issues
