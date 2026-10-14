import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# No bank atomic covers grabbing/shaking a pan in place, so both subtasks are
# declared extensions. The base never drives and the pan is not relocated -- it is
# lifted and shaken over its burner, then set back down.
SUBTASK_GRASP = "Pick"
SUBTASK_SHAKE = "ShakePan"

DONE_LANG = "task complete"

SHAKE_START = 0.02    # pan displacement (m) from rest that marks the shake onset (lift)
SHAKE_ACTIVE = 0.001  # per-frame pan translation (m) counting as actively shaking


@register_labeler
class ShakePanLabeler(BaseSubtaskLabeler):
    """Labeler for the ShakePan composite task.

    Task (non-mobile): a pan holding a vegetable sits on a lit burner (the burner is
    already on). Grab the pan by its handle, lift and shake it to toss the vegetable,
    then set it back on the stove. The pan is not relocated and the base never drives.

    Doc-canonical decomposition (2 subtasks): grasp, shake. Emitted:
      1. Pick [pick]         -> grab the pan by its handle
      2. ShakePan [execute]  -> shake the pan, then set it back down
    plus a terminal done. The split is the onset of pan MOTION (the env scores the
    shake by pan translation > 1mm/frame): everything before is the reach + grasp,
    everything after (until the pan settles back) is the shake.

    Detection: the pan body position is tracked; t_shake_start = the first sustained
    displacement > SHAKE_START from rest (the lift), and the shake ends when the pan
    stops moving (last frame translating > SHAKE_ACTIVE + pad).
    """

    ENV_NAME = "ShakePan"

    PAN = "obj_container"
    VEG = "obj"
    REST_W = 15
    SHAKE_SUSTAIN = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_body_id = env.obj_body_id[self.PAN]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.veg_lang = env.get_obj_lang(self.VEG)
        except Exception:
            self.veg_lang = "vegetable"
        self.extra_subtasks = {SUBTASK_GRASP, SUBTASK_SHAKE}
        self.expected_done_idx = None
        self.min_segment_frames = 5

    def per_step(self, env, t: int) -> dict:
        return {
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
        pan = np.array([h["pan"] for h in history])
        pan0 = np.median(pan[: self.REST_W], axis=0)
        dev = np.linalg.norm(pan - pan0, axis=1)
        vel = np.concatenate([np.linalg.norm(np.diff(pan, axis=0), axis=1), [0.0]])

        t_start = self._sustained_start(dev >= SHAKE_START, 0, self.SHAKE_SUSTAIN)
        if t_start is None:
            t_start = max(T - 1, 1)
        active = np.where(vel[t_start:] > SHAKE_ACTIVE)[0]
        t_end = min(t_start + int(active[-1]) + 3, T - 1) if active.size else T - 1
        t_end = max(t_end, t_start + 1)

        # a genuine shake needs a burst of >1mm/frame motion (env wants >=25 frames)
        n_active = int(np.sum(vel[t_start:t_end] > SHAKE_ACTIVE))

        segments = [
            (t_start, SUBTASK_GRASP, STAGE_PICK, "grab the pan by its handle"),
            (t_end, SUBTASK_SHAKE, STAGE_EXECUTE, "lift the pan and lightly shake the pan"),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_active < 25:
            review.append(f"pan shake was brief ({n_active} active frames < 25) -- verify")
        result = self.build_result(T, segments, debug={
            "t_start": int(t_start), "t_end": int(t_end), "max_dev": round(float(dev.max()), 2),
            "n_active": n_active, "n_segments": len(segments),
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
        if int(np.sum(names == SUBTASK_SHAKE)) != 1:
            issues.append("expected 1 ShakePan subtask")
        return issues
