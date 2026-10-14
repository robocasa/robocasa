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


# Pick the washed fruit out of the sink and set it on the cutting board to air-dry, then turn
# off the sink faucet. Sink -> cutting board has no bank name -> PickPlaceSinkToCuttingBoard
# (extension); TurnOffSinkFaucet is bank. Non-mobile. Doc-canonical count = 3.
SUBTASK_MOVE = "PickPlaceSinkToCuttingBoard"
SUBTASK_OFF = "TurnOffSinkFaucet"

OFF_LANG = "turn off the sink faucet"
DONE_LANG = "task complete"

FRUIT = "fruit0"


@register_labeler
class AirDryFruitLabeler(BaseSubtaskLabeler):
    """Labeler for the AirDryFruit composite task.

    Task (non-mobile): a washed fruit sits in the sink; move it onto the cutting board to
    air-dry, then turn off the sink faucet. Doc-canonical count = 3:
    PickPlaceSinkToCuttingBoard[pick] -> [place] -> TurnOffSinkFaucet.

    pick/place split at the fruit's first sustained displacement; place = gripper release;
    TurnOffSinkFaucet spans from the place to the end (water off).
    """

    ENV_NAME = "AirDryFruit"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.fruit_bid = env.obj_body_id[FRUIT]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        try:
            self.fruit_lang = env.get_obj_lang(FRUIT)
        except Exception:
            self.fruit_lang = "fruit"
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        return {
            "pos": env.sim.data.body_xpos[self.fruit_bid].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "water_on": won,
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
        pos = np.array([h["pos"] for h in history])
        grip = np.array([h["grip"] for h in history])
        won = np.array([h["water_on"] for h in history])
        gd = np.linalg.norm(grip - pos, axis=1)

        rest = np.median(pos[: self.REST_W], axis=0)
        disp = np.linalg.norm(pos - rest, axis=1)
        t_move = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_move = int(mv[0]) if mv.size else 1
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        w = self.fruit_lang
        emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {w} from the sink")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {w} on the cutting board")

        review = []
        offf = np.where(won > 0.5)[0]
        water_ends_off = bool(won[-1] < 0.5) if T else False
        if water_ends_off and offf.size:
            t_off = int(min(max(int(offf[-1]) + 1, cursor + 1), T - 1))
            emit(t_off, SUBTASK_OFF, STAGE_EXECUTE, OFF_LANG)
        elif cursor < T - 1:
            emit(min(cursor + 1, T - 1), SUBTASK_OFF, STAGE_EXECUTE, OFF_LANG)
            review.append("sink water did not register off at the end -- verify TurnOffSinkFaucet")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_move": int(t_move), "t_placed": int(t_placed), "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 fruit pick")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 fruit place")
        if int(np.sum(names == SUBTASK_OFF)) != 1:
            issues.append("expected 1 TurnOffSinkFaucet")
        return issues
