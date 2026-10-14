import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# Both subtasks are bank atomics acting on the dishwasher fixture; no objects move
# and the base stays parked at the dishwasher.
SUBTASK_OPEN = "OpenDishwasher"
SUBTASK_RACK = "SlideDishwasherRack"

DONE_LANG = "task complete"


@register_labeler
class PrepareDishwasherLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareDishwasher composite task.

    Task: open the (closed) dishwasher door, then pull the top rack out. The robot
    stays parked at the dishwasher; nothing else moves.

    Doc-canonical decomposition (2 high-level subtasks). Emitted:
      1. OpenDishwasher        -> open the dishwasher door
      2. SlideDishwasherRack   -> pull out the top rack
    plus a terminal done. The door opens first (0 -> 1), then after a short reposition
    the rack slides out (0 -> 1); the reposition is folded into the rack subtask's
    approach.

    Detection: the dishwasher's door and rack fractions are read from
    `dishwasher.get_state` (qpos-backed, survives replay). OpenDishwasher ends when the
    door first reaches OPEN_TH and stays; SlideDishwasherRack ends when the rack first
    reaches RACK_TH and stays.
    """

    ENV_NAME = "PrepareDishwasher"

    OPEN_TH = 0.85    # door "open" cap; the actual threshold adapts to the demo's own
    OPEN_FRAC = 0.90  # door max (some demos only open to ~0.8, still past the 0.70 that
    OPEN_SUSTAIN = 5  # counts as open) so a fixed 0.85 doesn't miss the completion
    RACK_TH = 0.85
    RACK_FRAC = 0.90
    RACK_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.dishwasher = env.dishwasher
        self.extra_subtasks = set()
        self.expected_done_idx = 2
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        st = self.dishwasher.get_state(env)
        return {
            "door": float(st.get("door", 0.0)),
            "rack": float(st.get("rack", 0.0)),
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
        door = np.array([abs(h["door"]) for h in history])
        rack = np.array([abs(h["rack"]) for h in history])

        door_th = min(self.OPEN_TH, self.OPEN_FRAC * float(door.max())) if door.size else self.OPEN_TH
        rack_th = min(self.RACK_TH, self.RACK_FRAC * float(rack.max())) if rack.size else self.RACK_TH
        t_open = self._sustained_start(door >= door_th, 0, self.OPEN_SUSTAIN)
        if t_open is None:
            t_open = int(np.argmax(door))
        t_rack = self._sustained_start(rack >= rack_th, t_open, self.RACK_SUSTAIN)
        if t_rack is None:
            t_rack = max(int(np.argmax(rack)), t_open + 1)
        t_rack = max(t_rack, t_open + 1)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the dishwasher door")
        emit(t_rack, SUBTASK_RACK, STAGE_EXECUTE, "pull out the top rack")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        door_open_end = bool(door[-1] >= self.OPEN_TH * 0.8) if T else False
        rack_out_end = bool(rack[-1] >= 0.5) if T else False
        debug = {
            "t_open": int(t_open), "t_rack": int(t_rack),
            "door_end": round(float(door[-1]), 2), "rack_end": round(float(rack[-1]), 2),
            "door_open_end": int(door_open_end), "rack_out_end": int(rack_out_end),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        if int(np.sum(names == SUBTASK_OPEN)) != 1:
            issues.append("expected 1 OpenDishwasher")
        if int(np.sum(names == SUBTASK_RACK)) != 1:
            issues.append("expected 1 SlideDishwasherRack")
        if result.debug.get("door_open_end") == 0:
            issues.append("dishwasher door not open at end")
        if result.debug.get("rack_out_end") == 0:
            issues.append("rack not pulled out at end")
        return issues
