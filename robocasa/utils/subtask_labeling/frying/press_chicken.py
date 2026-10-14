import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PRESS,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# No bank atomic covers "press a spatula on the chicken", so both subtasks are
# declared extensions. The base never drives.
SUBTASK_PICK = "Pick"
SUBTASK_PRESS = "PressChicken"

DONE_LANG = "task complete"


@register_labeler
class PressChickenLabeler(BaseSubtaskLabeler):
    """Labeler for the PressChicken composite task.

    Task: a chicken drumstick fries in a pan on a lit burner; a spatula rests on a
    plate on the adjacent counter. Pick up the spatula and press it down on the
    chicken for a few seconds, then release. The base never drives.

    Doc-canonical decomposition (2 high-level subtasks: pick, press). Emitted:
      1. Pick [pick]           -> pick up the spatula from the plate
      2. PressChicken [press]  -> press the spatula against the chicken, then release
    plus a terminal done. The single grasp is split at the spatula's departure from
    the plate: everything after is the carry-to-chicken + press + lift-off.

    Detection: t_pick = the spatula's first sustained departure from its plate rest;
    the press is confirmed when the spatula comes within PRESS_TH of the chicken for
    at least PRESS_MIN sustained frames (else the demo is flagged).
    """

    ENV_NAME = "PressChicken"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    REST_W = 15
    PRESS_MIN = 10     # sustained spatula<->chicken contact frames to confirm the press

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.spatula_body_id = env.obj_body_id["spatula"]
        self.spatula_obj = env.objects["spatula"]
        self.chicken_obj = env.objects["chicken"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.extra_subtasks = {SUBTASK_PICK, SUBTASK_PRESS}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            contact = int(env.check_contact(self.spatula_obj, self.chicken_obj))
        except Exception:
            contact = 0
        return {
            "spatula": env.sim.data.body_xpos[self.spatula_body_id].copy(),
            "contact": contact,
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

    def _sustained_run(self, mask):
        run = best = 0
        for v in mask:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    def resolve(self, history):
        T = len(history)
        spat = np.array([h["spatula"] for h in history])
        contact = np.array([h["contact"] for h in history]).astype(bool)

        rest = np.median(spat[: self.REST_W], axis=0)
        disp = np.linalg.norm(spat - rest, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)

        press_run = self._sustained_run(contact[t_pick:]) if t_pick < T else 0
        press_detected = press_run >= self.PRESS_MIN

        # press ends when the spatula last lifts off the chicken (last contact +pad);
        # the short remainder is the terminal done.
        contact_idx = np.where(contact[t_pick:])[0]
        t_press_end = min(t_pick + int(contact_idx[-1]) + 3, T - 1) if contact_idx.size else T - 1
        t_press_end = max(t_press_end, t_pick + 1)

        segments = [
            (t_pick, SUBTASK_PICK, STAGE_PICK, "pick up the spatula from the plate"),
            (t_press_end, SUBTASK_PRESS, STAGE_PRESS, "press the spatula against the chicken on the pan"),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = len(segments) - 1

        review = []
        if not press_detected:
            review.append("spatula never pressed the chicken (contact not detected)")
        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "press_run": int(press_run),
            "press_detected": int(press_detected),
            "total_contact": int(contact.sum()), "n_segments": len(segments),
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
        if int(np.sum(names == SUBTASK_PICK)) != 1:
            issues.append("expected 1 Pick subtask")
        if int(np.sum(names == SUBTASK_PRESS)) != 1:
            issues.append("expected 1 PressChicken subtask")
        if not result.debug.get("press_detected", 0):
            issues.append("press contact never detected")
        return issues
