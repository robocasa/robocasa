import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Turn the sink faucet on with hot water and hold the cutting board under the stream
# to rinse it, then turn the faucet off. Both names are bank atomics. Non-mobile
# (the robot stays at the sink); the cutting board sits in the basin and is at most
# nudged under the water -- there is no pick-place subtask.
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_OFF = "TurnOffSinkFaucet"

ON_LANG = "turn on the sink faucet with hot water and rinse the cutting board"
OFF_LANG = "turn off the sink faucet"
DONE_LANG = "task complete"


@register_labeler
class RinseCuttingBoardLabeler(BaseSubtaskLabeler):
    """Labeler for the RinseCuttingBoard composite task.

    Task (non-mobile): the cutting board sits in the sink. Turn the faucet on with hot
    water, rinse the board under the stream for ~10 seconds, then turn the faucet off.

    Doc-canonical count = 2 (`mobile: No`): TurnOnSinkFaucet + TurnOffSinkFaucet. The
    rinse hold (water running over the board) is absorbed into the turn-on subtask, so:
      TurnOnSinkFaucet [execute]  -> turn on (hot) and rinse the board  [0, t_off_start]
      TurnOffSinkFaucet [execute] -> turn the handle back off           [t_off_start, T]
    plus a terminal done. No pick-place and no navigation.

    Boundaries: the handle joint rises to an on-plateau (water flowing) and later falls
    back to off. t_off_start = one frame after the last on-plateau frame (the handle
    begins descending == the turn-off), so the whole rinse hold sits in subtask 1 and
    the descending handle sits in subtask 2. Review flags if the water was never hot or
    the board never registered under the stream.
    """

    ENV_NAME = "RinseCuttingBoard"

    HOT_TH = 0.55       # water_temp >= this == hot
    MIN_ON_FRAMES = 60  # water should run a while (task wants ~10s @ ~20fps)
    MIN_OFF = 20        # the turn-off subtask spans ~the last on-frames before shutoff

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sink = env.sink
        self.min_segment_frames = 12
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        hs = self.sink.get_handle_state(env=env)
        try:
            uw = 1.0 if self.sink.check_obj_under_water(env, "cutting_board") else 0.0
        except Exception:
            uw = 0.0
        return {
            "handle": abs(float(hs.get("handle_joint", 0.0))),
            "water_on": int(bool(hs.get("water_on", False))),
            "water_temp": float(hs.get("water_temp", 0.0)),
            "under_water": uw,
        }

    def resolve(self, history):
        T = len(history)
        handle = np.array([h["handle"] for h in history])
        won = np.array([h["water_on"] for h in history])
        temp = np.array([h["water_temp"] for h in history])
        uw = np.array([h["under_water"] for h in history])

        peak = float(handle.max()) if T else 0.0
        onf = np.where(won > 0)[0]
        has_on = peak > 1e-3 and onf.size > 0
        if has_on:
            t_on = int(onf[0])
            # water shuts off just after the last on-frame; the handle turn-off
            # motion spans roughly the last MIN_OFF on-frames -- start the OFF
            # subtask there so it is a real segment, not a 2-frame sliver.
            t_off_end = int(min(onf[-1] + 1, T - 1))
            t_off_start = max(t_on + 1, t_off_end - self.MIN_OFF)
            t_off_start = int(min(max(t_off_start, 1), max(t_off_end - 1, 1)))
        else:
            t_off_end = T - 1
            t_off_start = int(max(T - 1 - self.MIN_OFF, 1))

        segments = []
        cursor = 0

        def emit(end, name, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, STAGE_EXECUTE, lang))
            cursor = end

        emit(t_off_start, SUBTASK_ON, ON_LANG)
        emit(t_off_end, SUBTASK_OFF, OFF_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        # rinse quality checks (review, not hard structural)
        on_frames = int(won.sum())
        hot_during = bool((temp[won > 0] >= self.HOT_TH).any()) if on_frames else False
        board_rinsed = bool((uw[won > 0] > 0).any()) if on_frames else False
        review = []
        if not has_on:
            review.append("faucet never turned on -- verify")
        if on_frames and not hot_during:
            review.append("water was not hot during the rinse -- verify")
        if on_frames and not board_rinsed:
            review.append("cutting board never registered under the stream -- verify")
        if 0 < on_frames < self.MIN_ON_FRAMES:
            review.append(f"short rinse ({on_frames} frames water on) -- verify")
        if bool(won[-1]) if T else False:
            review.append("faucet still on at the end -- verify")

        result = self.build_result(T, segments, debug={
            "t_off_start": int(t_off_start), "handle_peak": round(peak, 2),
            "on_frames": on_frames, "hot": int(hot_during), "board_rinsed": int(board_rinsed),
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
        if int(np.sum(names == SUBTASK_ON)) != 1:
            issues.append(f"expected 1 TurnOnSinkFaucet, found {int(np.sum(names == SUBTASK_ON))}")
        if int(np.sum(names == SUBTASK_OFF)) != 1:
            issues.append(f"expected 1 TurnOffSinkFaucet, found {int(np.sum(names == SUBTASK_OFF))}")
        return issues
