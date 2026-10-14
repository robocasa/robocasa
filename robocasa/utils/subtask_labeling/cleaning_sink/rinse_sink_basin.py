import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# Turn the sink water on, then sweep the spout across the basin (left / center /
# right) to rinse all locations. Both names are bank atomics.
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_SPOUT = "TurnSinkSpout"

ON_LANG = "turn on the sink"
SPOUT_LANG = "move the spout to wash all locations of the sink basin"
DONE_LANG = "task complete"


@register_labeler
class RinseSinkBasinLabeler(BaseSubtaskLabeler):
    """Labeler for the RinseSinkBasin composite task.

    Task (non-mobile): turn the sink water on, then maneuver the spout so the
    stream reaches every part of the basin (the spout's left / center / right
    orientations each get washed while the water is on). A plate sits on the
    counter as a distractor and is never touched. The robot stays parked at the
    sink.

    Doc-canonical decomposition (2 subtasks):
      1. TurnOnSinkFaucet [execute]  -> turn the handle until water flows
      2. TurnSinkSpout    [execute]  -> sweep the spout to rinse the whole basin
    plus a terminal done. No leading navigation and no pick-place.

    Events:
      - t_on:   first frame the water is on (handle turned). Subtask 1 spans
                [0, t_on].
      - t_wash: one frame after the spout stops moving (the sweep completes).
                Subtask 2 spans [t_on, t_wash] and absorbs the reach from the
                handle to the spout. Clamp to T-1 so a demo cut at the end of the
                sweep still leaves a terminal done frame.
    Validation is structural plus a check that all three basin locations (left,
    center, right) were washed (spout reached each orientation while water on).
    """

    ENV_NAME = "RinseSinkBasin"

    SPOUT_MOVE_EPS = 0.02

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sink = env.sink
        self.min_segment_frames = 30
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        hs = self.sink.get_handle_state(env=env)
        return {
            "water_on": int(bool(hs.get("water_on", False))),
            "spout_ori": hs.get("spout_ori", "center"),
            "spout_joint": float(hs.get("spout_joint", 0.0)),
        }

    def resolve(self, history):
        T = len(history)
        won = np.array([h["water_on"] for h in history])
        sori = [h["spout_ori"] for h in history]
        sj = np.array([h["spout_joint"] for h in history])

        onf = np.where(won > 0)[0]
        t_on = int(onf[0]) if onf.size else T - 1

        # The spout sweep ends one frame after the last spout motion (after the
        # water is on). The spout-joint wraps mod 2*pi, so measure motion on the
        # shortest angular step.
        dsj = np.abs(((np.diff(sj) + np.pi) % (2 * np.pi)) - np.pi)
        moving = np.where(dsj > self.SPOUT_MOVE_EPS)[0]
        moving = moving[moving >= t_on]
        if moving.size:
            t_wash = int(min(moving[-1] + 2, T - 1))
        else:
            t_wash = T - 1
        t_wash = max(t_wash, t_on)

        segments = []
        cursor = 0

        def emit(end, name, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, STAGE_EXECUTE, lang))
            cursor = end

        emit(t_on, SUBTASK_ON, ON_LANG)
        emit(t_wash, SUBTASK_SPOUT, SPOUT_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        washed = {sori[t] for t in range(T) if won[t]}
        debug = {
            "t_on": int(t_on),
            "t_wash": int(t_wash),
            "washed_locs": ",".join(sorted(washed)),
            "n_washed": len(washed & {"left", "center", "right"}),
            "water_on_at_end": int(bool(won[-1])) if T else 0,
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
        if int(np.sum(names == SUBTASK_ON)) != 1:
            issues.append("expected 1 TurnOnSinkFaucet")
        if int(np.sum(names == SUBTASK_SPOUT)) != 1:
            issues.append("expected 1 TurnSinkSpout")
        d = result.debug
        if d.get("n_washed", 0) < 3:
            issues.append(
                f"only {d.get('n_washed')}/3 basin locations washed "
                f"(washed: {d.get('washed_locs')})"
            )
        return issues
