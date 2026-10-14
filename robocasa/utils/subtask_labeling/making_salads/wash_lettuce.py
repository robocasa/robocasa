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

# The two actions are: move the lettuce (in its colander) from the counter into
# the sink, and turn the sink water on. Both are bank atomics. Their order varies
# per demo (some operators turn the water on first and then place the lettuce
# under the stream; others place the lettuce in the sink first and then turn the
# water on). Whichever happens first is labeled first.
SUBTASK_MOVE = "PickPlaceCounterToSink"
SUBTASK_ON = "TurnOnSinkFaucet"

# Order A turns the water on before the lettuce is in the sink; order B places
# the lettuce first, so turning the water on then runs it over the lettuce.
ON_LANG = "turn on the water in the sink"
ON_OVER_LETTUCE_LANG = "turn on the water in the sink and run it over the lettuce"
PICK_LANG = "pick up the lettuce from the counter"
PLACE_LANG = "place the lettuce in the sink"
DONE_LANG = "task complete"

WASH_TARGET = 25


@register_labeler
class WashLettuceLabeler(BaseSubtaskLabeler):
    """Labeler for the WashLettuce composite task.

    Task (non-mobile): the lettuce sits in a colander on the counter by the sink.
    Move it into the sink and run water over it until it is washed (the env
    requires WASH_TARGET frames with the lettuce under the running water). The
    robot stays parked at the sink.

    Doc-canonical decomposition (2 subtasks), emitted in the order they occur:
      - PickPlaceCounterToSink [pick/place]  -> move the lettuce into the sink
      - TurnOnSinkFaucet       [execute]     -> turn the water on
    plus a terminal done (the lettuce sits under the running water being washed).

    Two operator orderings, distinguished by whether the water turns on before or
    after the lettuce is lifted:
      A (water first):   TurnOnSinkFaucet, pick, place, done
      B (lettuce first): pick, place, TurnOnSinkFaucet, done

    The pick stage ends at the grasp (finger-width closing before the lift); the
    place ends at the lettuce's arrival in the sink (it settles near its final
    pose). t_on is the first frame the water is on.
    """

    ENV_NAME = "WashLettuce"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10
    FW_RANGE_MIN = 0.03
    # A grasp finger-close is only accepted if it lands within this many frames
    # before the lift; otherwise it is a stray close (e.g. on the sink handle) and
    # the lift itself is used as the pick->place boundary.
    GRASP_WINDOW = 80

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sink = env.sink
        self.lettuce_body_id = env.obj_body_id["lettuce"]
        try:
            self.lettuce_lang = env.get_obj_lang("lettuce")
        except Exception:
            self.lettuce_lang = "lettuce"
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j)
            for j in env.robots[0].gripper["right"].joints
        ]
        # The place can be a fast swing of the colander into the adjacent sink
        # (~18 frames for a real 0.5 m move), so the short-segment guard is set
        # lower than the usual 30.
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        hs = self.sink.get_handle_state(env=env)
        try:
            under = int(bool(self.sink.check_obj_under_water(env, "lettuce")))
        except Exception:
            under = 0
        return {
            "lettuce": env.sim.data.body_xpos[self.lettuce_body_id].copy(),
            "water_on": int(bool(hs.get("water_on", False))),
            "under_water": under,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
        }

    # ---- event helpers -------------------------------------------------

    def _lift_time(self, obj, T):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        cand = np.where(moved[:T])[0]
        return int(cand[0]) if cand.size else 0

    def _grasp_time(self, closes, lift, lo):
        # Grasp = the finger-width closing just before the lettuce is lifted,
        # within (lo, lift]. The lower bound excludes the handle grasp the robot
        # makes to turn the water on (in the water-first order).
        cand = closes[(closes > lo) & (closes <= lift)]
        if cand.size and (lift - int(cand[-1])) <= self.GRASP_WINDOW:
            return int(cand[-1])
        return int(lift)

    def _arrival_time(self, obj, lo, T):
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.PLACE_TOL
        idxs = np.where(far[:T])[0]
        idxs = idxs[idxs >= lo]
        return int(idxs[-1]) + 1 if idxs.size else lo + 1

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        lettuce = np.array([h["lettuce"] for h in history])
        won = np.array([h["water_on"] for h in history])
        under = np.array([h["under_water"] for h in history])
        fw = np.array([h["fw"] for h in history])

        lo_fw, hi_fw = float(np.min(fw)), float(np.max(fw))
        if hi_fw - lo_fw > self.FW_RANGE_MIN:
            mid = lo_fw + 0.5 * (hi_fw - lo_fw)
            closes = np.where((fw[:-1] >= mid) & (fw[1:] < mid))[0] + 1
        else:
            closes = np.array([], dtype=int)

        onf = np.where(won > 0)[0]
        t_on = int(onf[0]) if onf.size else T - 1
        t_lift = self._lift_time(lettuce, T)
        water_first = t_on < t_lift
        # In the water-first order the lettuce is grasped only after the water is
        # on, so exclude the earlier handle grasp from the grasp search.
        t_grasp = self._grasp_time(closes, t_lift, t_on if water_first else 0)
        t_arrive = self._arrival_time(lettuce, t_grasp, T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def emit_move(grasp_end, place_end):
            emit(grasp_end, SUBTASK_MOVE, STAGE_PICK, PICK_LANG)
            emit(place_end, SUBTASK_MOVE, STAGE_PLACE, PLACE_LANG)

        if water_first:
            # Order A: turn the water on, then move the lettuce under the stream.
            emit(t_on, SUBTASK_ON, STAGE_EXECUTE, ON_LANG)
            emit_move(max(t_grasp, t_on), t_arrive)
        else:
            # Order B: place the lettuce in the sink, then turn the water on over
            # it.
            emit_move(t_grasp, t_arrive)
            emit(max(t_on, t_arrive), SUBTASK_ON, STAGE_EXECUTE,
                 ON_OVER_LETTUCE_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": "A" if water_first else "B",
            "t_on": int(t_on),
            "t_grasp": int(t_grasp),
            "t_arrive": int(t_arrive),
            "water_turned_on": int(bool(onf.size)),
            "wash_frames": int(np.sum(under)),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        n_on = int(np.sum(names == SUBTASK_ON))
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_on != 1:
            issues.append(f"expected 1 TurnOnSinkFaucet, found {n_on}")
        d = result.debug
        if d.get("water_turned_on") == 0:
            issues.append("water was never turned on")
        if d.get("wash_frames", 0) < WASH_TARGET:
            issues.append(
                f"lettuce washed only {d.get('wash_frames')} frames (< {WASH_TARGET})"
            )
        return issues
