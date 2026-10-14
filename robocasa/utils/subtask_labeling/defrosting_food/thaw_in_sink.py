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


# Take the food off the counter and set it in the sink, then turn on the water to
# thaw it. PickPlaceCounterToSink + TurnOnSinkFaucet are bank atomics. No navigation
# (the counter is beside the sink, mobile: No).
SUBTASK_MOVE = "PickPlaceCounterToSink"
SUBTASK_FAUCET = "TurnOnSinkFaucet"

DONE_LANG = "task complete"


@register_labeler
class ThawInSinkLabeler(BaseSubtaskLabeler):
    """Labeler for the ThawInSink composite task.

    Task: a frozen food sits on the counter; the sink is beside it (water off, robot
    starts at the sink). Put the food in the sink, then turn on the faucet.

    Doc-canonical count = 2 (`mobile: No`): PickPlaceCounterToSink + TurnOnSinkFaucet.
    food[pick] -> [place] -> TurnOnSinkFaucet -> done (4 segments; expected_done_idx = 3).

    Boundaries: t_move = the food first displaces MOVE_TOL from its counter rest;
    t_placed = the gripper RELEASE (dropped in the sink). TurnOnSinkFaucet spans from
    the food's placement to when the water first comes on (sink handle qpos ->
    water_on) and stays.
    """

    ENV_NAME = "ThawInSink"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    FAUCET_SUSTAIN = 5
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = env.sink
        try:
            self.obj_lang = env.get_obj_lang("obj")
        except Exception:
            self.obj_lang = "food"

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            water_on = bool(self.sink.get_handle_state(env)["water_on"])
        except Exception:
            water_on = False
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "water_on": water_on,
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
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        water = np.array([bool(h["water_on"]) for h in history])

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        t_move = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_move = int(mv[0]) if mv.size else max(T - 1, 0)

        gd = np.linalg.norm(grip - obj, axis=1)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.obj_lang} from the counter")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.obj_lang} in the sink")
        # TurnOnSinkFaucet: the water comes on after the food is placed
        has_faucet = False
        if cursor < T:
            t_on = self._sustained_start(water, cursor, self.FAUCET_SUSTAIN)
            if t_on is not None and t_on > cursor:
                emit(t_on, SUBTASK_FAUCET, STAGE_EXECUTE, "turn on the sink faucet")
                has_faucet = True
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_faucet:
            review.append("water never turned on -- verify")

        result = self.build_result(T, segments, debug={
            "t_move": t_move, "t_placed": t_placed,
            "has_faucet": has_faucet, "n_segments": len(segments),
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

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 food pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 food place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_FAUCET) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnSinkFaucet, found {count(SUBTASK_FAUCET)}")
        return issues
