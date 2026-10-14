import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Close the electric kettle's lid, then press its lever to turn it on. Both are
# bank atomics; no object is carried and the base never drives.
SUBTASK_CLOSE = "CloseElectricKettleLid"
SUBTASK_TURNON = "TurnOnElectricKettle"

CLOSE_LANG = "close the electric kettle lid"
TURNON_LANG = "turn on the electric kettle"
DONE_LANG = "task complete"


@register_labeler
class StartElectricKettleLabeler(BaseSubtaskLabeler):
    """Labeler for the StartElectricKettle composite task.

    Task: the electric kettle's lid starts OPEN and its power lever is up. Close
    the lid, then press the lever to turn the kettle on.

    Doc-canonical count = 2 (`mobile: No`): CloseElectricKettleLid + TurnOnElectric-
    Kettle. 3 segments with done; the base never drives and nothing is carried.

    Both boundaries come from the kettle fixture's own joints (read from qpos, since
    the fixture's cached get_state() does not track a replay):
      - t_lid_closed: the lid joint (range [0, pi/2], starts near open ~1.4-1.57)
                  first drops below LID_CLOSED and stays -> the lid is shut.
                  CloseElectricKettleLid ends here.
      - t_switch_on: the power switch/lever joint (rests at ~0, slightly negative
                  from jitter) first rises above SWITCH_ON -> the lever is pressed
                  down. TurnOnElectricKettle ends here. Always at the very end.
    """

    ENV_NAME = "StartElectricKettle"

    LID_CLOSED = 0.10  # lid joint below this (of a ~1.57 open) == shut
    LID_SUSTAIN = 5
    SWITCH_ON = 0.001  # switch joint above this (jitter rests at/below 0) == pressed
    SWITCH_SUSTAIN = 2

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        ek = env.electric_kettle
        self.lid_adr = None
        self.switch_adr = None
        try:
            self.lid_adr = env.sim.model.get_joint_qpos_addr(ek._joint_names["lid"])
        except Exception:
            pass
        try:
            self.switch_adr = env.sim.model.get_joint_qpos_addr(ek._joint_names["switch"])
        except Exception:
            pass

        self.extra_subtasks = set()
        self.expected_done_idx = 2
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "lid": float(env.sim.data.qpos[self.lid_adr]) if self.lid_adr is not None else 0.0,
            "switch": float(env.sim.data.qpos[self.switch_adr]) if self.switch_adr is not None else 0.0,
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
        lid = np.array([h["lid"] for h in history])
        switch = np.array([h["switch"] for h in history])

        # close the lid: the lid joint drops to shut and stays
        t_lid_closed = self._sustained_start(lid < self.LID_CLOSED, 0, self.LID_SUSTAIN)
        if t_lid_closed is None:
            # never registered as shut -- fall back to the frame it is most closed
            t_lid_closed = int(np.argmin(lid))

        # turn on: the power lever is pressed (switch joint rises above rest), after
        # the lid is shut
        t_switch_on = self._sustained_start(switch > self.SWITCH_ON, t_lid_closed, self.SWITCH_SUSTAIN)
        if t_switch_on is None:
            hits = np.where(switch[t_lid_closed:] > self.SWITCH_ON)[0]
            t_switch_on = t_lid_closed + int(hits[0]) if hits.size else T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_lid_closed, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(t_switch_on, SUBTASK_TURNON, STAGE_EXECUTE, TURNON_LANG)
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not np.any(switch[t_lid_closed:] > self.SWITCH_ON):
            review.append("switch never registered as pressed -- check the turn-on")
        if not np.any(lid < self.LID_CLOSED):
            review.append("lid never registered as closed -- check the close")

        debug = {
            "t_lid_closed": t_lid_closed,
            "t_switch_on": t_switch_on,
            "lid_start": round(float(lid[0]), 3),
            "switch_peak": round(float(switch.max()), 4),
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

        def count(name):
            return int(np.sum(names == name))

        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseElectricKettleLid, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_TURNON) != 1:
            issues.append(f"expected 1 TurnOnElectricKettle, found {count(SUBTASK_TURNON)}")
        return issues
