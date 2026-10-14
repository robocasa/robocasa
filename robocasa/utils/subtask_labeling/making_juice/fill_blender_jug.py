import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# The sink faucet is turned on and the blender jug is held under the running water to
# fill it. Turning on the faucet is the bank atomic TurnOnSinkFaucet; the jug move has
# no bank name -> PickPlaceCounterToSink (extension). The jug is NOT released -- it is
# held under the water through the end of the episode.
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_MOVE = "PickPlaceCounterToSink"

ON_LANG = "turn on the sink faucet"
DONE_LANG = "task complete"


@register_labeler
class FillBlenderJugLabeler(BaseSubtaskLabeler):
    """Labeler for the FillBlenderJug composite task.

    Task: the blender jug sits on the counter by the sink. Turn on the faucet and hold
    the jug under the running water for a few seconds to fill it. The robot starts at
    the counter next to the sink (no real navigation).

    Doc-canonical decomposition (2 high-level subtasks: turn on faucet, fill jug).
    Emitted in the order they actually happen (usually faucet first, then jug):
      TurnOnSinkFaucet [execute]        -> turn on the sink faucet
      PickPlaceCounterToSink [pick]     -> pick up the blender jug from the counter
      PickPlaceCounterToSink [place]    -> hold the jug under the running water to fill it
    plus a terminal done. The jug is never set down, so the "place" is the hold under
    the water and runs to the end.

    Detection: t_faucet = first frame the sink water is on; t_pick = the jug's first
    sustained departure from its counter rest; the hold ends at the last frame the jug
    is under the water (+pad).
    """

    ENV_NAME = "FillBlenderJug"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    REST_W = 15
    HOLD_PAD = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.jug_body_id = env.obj_body_id["blender_jug"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = env.sink
        try:
            self.jug_lang = env.get_obj_lang("blender_jug")
        except Exception:
            self.jug_lang = "blender jug"
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            water_on = int(bool(self.sink.get_handle_state(env=env).get("water_on", False)))
        except Exception:
            water_on = 0
        try:
            under = int(bool(self.sink.check_obj_under_water(env, "blender_jug")))
        except Exception:
            under = 0
        return {
            "jug": env.sim.data.body_xpos[self.jug_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "water_on": water_on,
            "under": under,
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
        jug = np.array([h["jug"] for h in history])
        water_on = np.array([h["water_on"] for h in history])
        under = np.array([h["under"] for h in history])

        rest = np.median(jug[: self.REST_W], axis=0)
        disp = np.linalg.norm(jug - rest, axis=1)

        won = np.where(water_on > 0)[0]
        t_faucet = int(won[0]) if won.size else T - 1

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)

        uw = np.where(under > 0)[0]
        t_hold_end = int(uw[-1]) + self.HOLD_PAD if uw.size else T - 1
        t_hold_end = int(min(max(t_hold_end, t_pick + 1), T - 1))

        events = [
            (t_faucet, SUBTASK_ON, STAGE_EXECUTE, ON_LANG),
            (t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.jug_lang} from the counter"),
            (t_hold_end, SUBTASK_MOVE, STAGE_PLACE, f"hold the {self.jug_lang} under the running water to fill it"),
        ]
        events.sort(key=lambda e: e[0])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for end, name, stage, lang in events:
            emit(end, name, stage, lang)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if not won.size:
            review.append("faucet never turned on (water_on not detected)")
        if not uw.size:
            review.append("jug never registered under the water")
        result = self.build_result(T, segments, debug={
            "t_faucet": int(t_faucet), "t_pick": int(t_pick), "t_hold_end": int(t_hold_end),
            "water_on_frames": int(water_on.sum()), "under_frames": int(under.sum()),
            "faucet_first": int(t_faucet <= t_pick), "n_segments": len(segments),
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
        stages = result.subtask_stage[firsts]
        if int(np.sum(names == SUBTASK_ON)) != 1:
            issues.append("expected 1 TurnOnSinkFaucet")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 jug pick")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 jug place/hold")
        dbg = getattr(result, "debug", None) or {}
        if dbg.get("water_on_frames", 1) < 1:
            issues.append("water never turned on")
        return issues
