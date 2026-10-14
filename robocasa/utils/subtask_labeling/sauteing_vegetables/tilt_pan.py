import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# No bank atomic covers grabbing/tilting a pan in place, so both subtasks are declared
# extensions. The base never drives and the pan is not relocated -- it is tilted over
# its burner, then set level again.
SUBTASK_GRASP = "Pick"
SUBTASK_TILT = "TiltPan"

DONE_LANG = "task complete"

TILT_START = 0.15   # pan orientation deviation (rad) from rest that marks the tilt onset
TILT_ACTIVE = 0.004  # per-frame orientation change (rad) counting as actively tilting


@register_labeler
class TiltPanLabeler(BaseSubtaskLabeler):
    """Labeler for the TiltPan composite task.

    Task (non-mobile): a pan holding a vegetable sits on a lit burner (already on).
    Grab the pan by its handle and tilt it (the env scores success at >0.3 rad, ~17
    deg, of orientation change from its initial pose for >=12 frames) to slide the
    vegetable, then set it level again. The pan is not relocated and the base never
    drives.

    Doc-canonical decomposition (2 subtasks): grasp, tilt. Emitted:
      1. Pick [pick]      -> grab the pan by its handle
      2. TiltPan [tilt]   -> tilt the pan, then set it level
    plus a terminal done. The split is the onset of pan reorientation: everything
    before is the reach + grasp, everything after (until the pan is level again) is
    the tilt.

    Detection: the pan's geodesic quaternion angle from its initial orientation (the
    same metric the env uses) is tracked; t_tilt_start = the first sustained deviation
    > TILT_START, and the tilt ends when the pan stops reorienting (last actively
    tilting frame + pad).
    """

    ENV_NAME = "TiltPan"

    PAN = "obj_container"
    VEG = "obj"
    REST_W = 15
    TILT_SUSTAIN = 2

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_body_id = env.obj_body_id[self.PAN]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.veg_lang = env.get_obj_lang(self.VEG)
        except Exception:
            self.veg_lang = "vegetable"
        self.extra_subtasks = {SUBTASK_GRASP, SUBTASK_TILT}
        self.expected_done_idx = None
        self.min_segment_frames = 5

    def per_step(self, env, t: int) -> dict:
        return {
            "quat": env.sim.data.body_xquat[self.pan_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
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
        quat = np.array([h["quat"] for h in history])
        q0 = np.median(quat[: self.REST_W], axis=0)
        q0 = q0 / (np.linalg.norm(q0) + 1e-9)
        qn = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9)
        # geodesic angle from the initial orientation (matches the env's metric)
        dot = np.clip(np.abs(qn @ q0), -1.0, 1.0)
        dev = 2.0 * np.arccos(dot)  # radians
        vel = np.concatenate([np.abs(np.diff(dev)), [0.0]])

        t_start = self._sustained_start(dev >= TILT_START, 0, self.TILT_SUSTAIN)
        if t_start is None:
            t_start = max(T - 1, 1)
        active = np.where(vel[t_start:] > TILT_ACTIVE)[0]
        t_end = min(t_start + int(active[-1]) + 3, T - 1) if active.size else T - 1
        t_end = max(t_end, t_start + 1)

        max_dev = float(dev.max())
        tilted = max_dev >= 0.30  # env threshold ~0.3 rad

        segments = [
            (t_start, SUBTASK_GRASP, STAGE_PICK, "grab the pan by its handle"),
            (t_end, SUBTASK_TILT, STAGE_TILT, f"lift the pan and tilt the pan to move the {self.veg_lang}"),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = len(segments) - 1

        review = []
        if not tilted:
            review.append(f"pan only tilted {np.degrees(max_dev):.0f} deg (< 17; tilt not detected)")
        result = self.build_result(T, segments, debug={
            "t_start": int(t_start), "t_end": int(t_end),
            "max_dev_deg": round(float(np.degrees(max_dev)), 1), "tilted": int(tilted),
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
        if int(np.sum(names == SUBTASK_GRASP)) != 1:
            issues.append("expected 1 Pick (grasp) subtask")
        if int(np.sum(names == SUBTASK_TILT)) != 1:
            issues.append("expected 1 TiltPan subtask")
        if not result.debug.get("tilted", 0):
            issues.append("pan tilt (>=17 deg) not detected")
        return issues
