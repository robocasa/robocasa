import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PRESS,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# No bank atomic covers "press a fruit against a reamer to juice it", so both subtasks
# are declared extensions. The base never drives. Mirrors PressChicken.
SUBTASK_PICK = "Pick"
SUBTASK_PRESS = "JuiceFruitReamer"

DONE_LANG = "task complete"


@register_labeler
class JuiceFruitReamerLabeler(BaseSubtaskLabeler):
    """Labeler for the JuiceFruitReamer composite task.

    Fixed-base task: a sliced fruit (cut side down) and a reamer sit on the counter.
    Pick up the fruit and press it, cut-side down, against the reamer to juice it. The
    fruit is held (grasped) against the stationary reamer through the end; the base
    never drives.

    Doc-canonical decomposition (2 high-level subtasks: pick, press). Emitted:
      Pick [pick]              -> pick up the sliced {fruit} from the counter
      JuiceFruitReamer [press] -> press the {fruit} against the reamer to juice it
    plus a terminal done. The single grasp is split at the fruit's departure from its
    counter rest: everything after is the carry-to-reamer + press.

    Detection: t_pick = the fruit's first sustained departure from its rest; the press
    is confirmed when the fruit makes sustained contact with the reamer for at least
    PRESS_MIN frames (else the demo is flagged for review).
    """

    ENV_NAME = "JuiceFruitReamer"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    REST_W = 15
    PRESS_MIN = 8

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.fruit_body_id = env.obj_body_id["fruit"]
        self.fruit_obj = env.objects["fruit"]
        self.reamer_obj = env.objects["reamer"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fruit_lang = env.get_obj_lang("fruit")
        except Exception:
            self.fruit_lang = "fruit"
        self.extra_subtasks = {SUBTASK_PICK, SUBTASK_PRESS}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            contact = int(env.check_contact(self.fruit_obj, self.reamer_obj))
        except Exception:
            contact = 0
        return {
            "fruit": env.sim.data.body_xpos[self.fruit_body_id].copy(),
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
        fruit = np.array([h["fruit"] for h in history])
        contact = np.array([h["contact"] for h in history]).astype(bool)

        rest = np.median(fruit[: self.REST_W], axis=0)
        disp = np.linalg.norm(fruit - rest, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)

        press_run = self._sustained_run(contact[t_pick:]) if t_pick < T else 0
        press_detected = press_run >= self.PRESS_MIN

        contact_idx = np.where(contact[t_pick:])[0]
        t_press_end = min(t_pick + int(contact_idx[-1]) + 3, T - 1) if contact_idx.size else T - 1
        t_press_end = max(t_press_end, t_pick + 1)

        segments = [
            (t_pick, SUBTASK_PICK, STAGE_PICK, f"pick up the sliced {self.fruit_lang} from the counter"),
            (t_press_end, SUBTASK_PRESS, STAGE_PRESS, f"press the {self.fruit_lang} against the reamer to juice it"),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = len(segments) - 1

        review = []
        if not press_detected:
            review.append("fruit never pressed the reamer (contact not detected)")
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
            issues.append("expected 1 JuiceFruitReamer subtask")
        if not result.debug.get("press_detected", 0):
            issues.append("press contact never detected")
        return issues
