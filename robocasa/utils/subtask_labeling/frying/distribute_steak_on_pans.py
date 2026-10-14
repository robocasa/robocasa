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

# Turn on the empty pan's burner (TurnOnStove, bank), then move one steak from the
# full pan to the empty pan. Both pans sit on the same stove, so the steak move is
# stove -> stove, which no bank name covers -> PickPlaceStoveToStove (extension).
# The base never drives (both pans are within reach).
SUBTASK_TURN = "TurnOnStove"
SUBTASK_STEAK = "PickPlaceStoveToStove"

DONE_LANG = "task complete"
KNOB_ON = 0.35


@register_labeler
class DistributeSteakOnPansLabeler(BaseSubtaskLabeler):
    """Labeler for the DistributeSteakOnPans composite task.

    Task: two pans sit on the stove -- a full pan (two steaks, its burner already
    on) and an empty pan. Turn on the empty pan's burner, then move one steak from
    the full pan to the empty pan. The base never drives.

    Doc-canonical decomposition (2 high-level subtasks: turn, steak move). Emitted
    fine-grained:
      1. TurnOnStove                     -> turn on the empty pan's burner
      2. PickPlaceStoveToStove [pick]    -> pick a steak off the full pan
      3. PickPlaceStoveToStove [place]   -> set it in the empty pan
    plus a terminal done. The turn-on comes first (the burner heats while the steak
    is moved). Exactly one of the two steaks is relocated; the other stays.

    The empty pan's knob is read from ep_meta refs. The turn completes when that
    knob reaches its on value; the steak pick/place are the moving steak's first
    departure and the gripper release.
    """

    ENV_NAME = "DistributeSteakOnPans"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 30
    RELEASE_MARGIN = 0.06
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15

    STEAKS = ("steak1", "steak2")

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.steak_body_ids = {o: env.obj_body_id[o] for o in self.STEAKS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.stove = env.stove
        self.empty_knob = ep_meta.get("refs", {}).get("empty_pan_knob")
        self.extra_subtasks = {SUBTASK_STEAK}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        ks = self.stove.get_knobs_state(env=env)
        v = ks.get(self.empty_knob, 0.0)
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knob": float(min(abs(v), 2 * np.pi - abs(v))),
        }
        for o in self.STEAKS:
            d[o] = env.sim.data.body_xpos[self.steak_body_ids[o]].copy()
        return d

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

    def _turn_done(self, knob, T):
        final = float(np.median(knob[-self.REST_W:])) if T else 0.0
        if final >= KNOB_ON:
            below = np.where(knob[:T] < final - 0.05)[0]
            if below.size:
                return int(min(below[-1] + 2, T - 1))
        return 0

    def resolve(self, history):
        T = len(history)
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])
        opos = {o: np.array([h[o] for h in history]) for o in self.STEAKS}
        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.STEAKS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.STEAKS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.STEAKS}

        # the moving steak = the one displaced the most
        mover = max(self.STEAKS, key=lambda o: float(disp[o].max()))

        t_turn = self._turn_done(knob, T)

        t_pick = self._sustained_start(disp[mover] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = T - 1
        t_pick = max(t_pick, t_turn)
        hold_gd = float(np.median(gd[mover][t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd[mover] > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        burner = self.empty_knob.replace("_", "-") + " burner" if self.empty_knob else "burner"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_turn, SUBTASK_TURN, STAGE_EXECUTE, f"turn on the {burner} for the empty pan")
        emit(t_pick, SUBTASK_STEAK, STAGE_PICK, "pick up a steak from the full pan")
        emit(t_placed, SUBTASK_STEAK, STAGE_PLACE, "place the steak in the empty pan")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        burner_on = bool(knob[-1] >= KNOB_ON) if T else False
        debug = {
            "mover": mover, "t_turn": int(t_turn), "t_pick": int(t_pick),
            "t_placed": int(t_placed), "burner_on_at_end": int(burner_on),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

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
        n_turn = int(np.sum(names == SUBTASK_TURN))
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_turn != 1:
            issues.append(f"expected 1 TurnOnStove, found {n_turn}")
        if result.debug.get("burner_on_at_end") == 0:
            issues.append("empty pan's burner is not on at the end")
        return issues
