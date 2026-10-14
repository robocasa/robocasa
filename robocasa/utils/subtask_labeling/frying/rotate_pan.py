import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# No bank atomic covers grabbing/rotating a pan in place, so both subtasks are
# declared extensions. The base never drives and the pan is not relocated -- it is
# rotated on its burner.
SUBTASK_GRASP = "Pick"
SUBTASK_ROTATE = "RotatePan"

DONE_LANG = "task complete"

ROT_START_DEG = 5.0    # yaw departure (deg) from rest that marks the rotation onset
ROT_ACTIVE_DEG = 0.15  # per-frame yaw change (deg) counting as actively rotating


@register_labeler
class RotatePanLabeler(BaseSubtaskLabeler):
    """Labeler for the RotatePan composite task.

    Task: a pan (holding meat/seafood) sits on a lit burner. Grab it by the handle
    and rotate it (>~15 deg yaw) to distribute the heat, then let go. The pan is NOT
    relocated -- it only turns in place -- and the base never drives.

    Doc-canonical decomposition (2 high-level subtasks: grasp, rotate). Emitted:
      1. Pick [pick]          -> grab the pan by its handle
      2. RotatePan [execute]  -> rotate the pan, then release
    plus a terminal done. The split is the onset of yaw rotation: everything before
    is the reach + grasp, everything after (until the pan stops turning) is the
    rotate.

    Detection: the pan's yaw (from its body rotation matrix) is tracked; t_rot_start
    is the first sustained departure > ROT_START_DEG from the initial yaw, and the
    rotate ends when the yaw stops changing (last actively-rotating frame + pad).
    """

    ENV_NAME = "RotatePan"

    PAN = "meat_container"
    REST_W = 15
    ROT_SUSTAIN = 2

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_body_id = env.obj_body_id[self.PAN]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.extra_subtasks = {SUBTASK_GRASP, SUBTASK_ROTATE}
        self.expected_done_idx = None
        # a genuine rotate can be a quick flick (the grab/approach is the long part),
        # so allow shorter segments than a pick-place labeler would.
        self.min_segment_frames = 5

    def per_step(self, env, t: int) -> dict:
        mat = env.sim.data.body_xmat[self.pan_body_id].reshape(3, 3)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        return {
            "yaw": yaw,
            "pan": env.sim.data.body_xpos[self.pan_body_id].copy(),
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
        yaw = np.array([h["yaw"] for h in history])
        yaw0 = float(np.median(yaw[: self.REST_W]))
        # wrapped yaw deviation from the initial orientation, in degrees
        dev = np.degrees(np.abs((yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi))
        # per-frame |yaw velocity| in degrees
        vel = np.abs(np.degrees((np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi))
        vel = np.concatenate([vel, vel[-1:]])

        # rotation onset = the FIRST time the pan turns past ROT_START_DEG from rest
        # (a short sustain filters single-frame jitter). The pan may twist, briefly
        # settle back, then rotate the rest of the way -- the whole thing is the
        # rotate, so the boundary is that first departure, not the final swing.
        t_rot_start = self._sustained_start(dev >= ROT_START_DEG, 0, self.ROT_SUSTAIN)
        if t_rot_start is None:
            t_rot_start = max(T - 1, 1)

        active = np.where(vel[t_rot_start:] > ROT_ACTIVE_DEG)[0]
        t_rot_end = min(t_rot_start + int(active[-1]) + 3, T - 1) if active.size else T - 1
        t_rot_end = max(t_rot_end, t_rot_start + 1)

        max_dev = float(dev.max())
        rotated = max_dev >= 15.0

        segments = [
            (t_rot_start, SUBTASK_GRASP, STAGE_PICK, "grab the pan by its handle"),
            (t_rot_end, SUBTASK_ROTATE, STAGE_EXECUTE, "rotate the pan to distribute the heat"),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = len(segments) - 1

        review = []
        if not rotated:
            review.append(f"pan only rotated {max_dev:.0f} deg (< 15; rotation not detected)")
        result = self.build_result(T, segments, debug={
            "t_rot_start": int(t_rot_start), "t_rot_end": int(t_rot_end),
            "max_dev_deg": round(max_dev, 1), "rotated": int(rotated),
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
        if int(np.sum(names == SUBTASK_ROTATE)) != 1:
            issues.append("expected 1 RotatePan subtask")
        if not result.debug.get("rotated", 0):
            issues.append("pan rotation (>=15 deg) not detected")
        return issues
