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


# Put two vegetables in the pan, then turn the burner on and set it to MEDIUM heat.
# Counter -> stove is the bank atomic PickPlaceCounterToStove; the stove has TurnOnStove
# but no "set to medium" atomic, so the heat-adjust is a coined extension (AdjustHeat).
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_HEAT = "AdjustHeat"

DONE_LANG = "task complete"

VEGS = ("veg1", "veg2")
KNOB_ON = 0.35             # |knob| >= this == burner on
MEDIUM_LO, MEDIUM_HI = np.pi / 4, 3 * np.pi / 4  # env's medium band


@register_labeler
class AdjustHeatLabeler(BaseSubtaskLabeler):
    """Labeler for the AdjustHeat composite task.

    Task (mobile, local): two vegetables sit on the counter; a pan sits on an OFF
    burner. Put both vegetables in the pan, then turn the burner on and set it to
    medium heat. The robot starts at the stove.

    Doc-canonical count = 3 (`mobile: Yes`): two PickPlaceCounterToStove (the
    vegetables, in grasp order) + the heat adjust. Structure:
      PickPlaceCounterToStove [pick/place] x2 -> put each vegetable in the pan
      AdjustHeat [execute]                    -> turn the burner to medium heat
    plus a terminal done. The heat adjust is last.

    Boundaries: per vegetable t_pick = first sustained displacement, t_place = the
    gripper RELEASE; the heat adjust runs from the last vegetable's placement to the
    turn-completion (the last frame the burner knob is below its settled value).
    """

    ENV_NAME = "AdjustHeat"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in VEGS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {}
        for o in self.names:
            try:
                self.word[o] = env.get_obj_lang(o)
            except Exception:
                self.word[o] = "vegetable"
        knob = ep_meta.get("refs", {}).get("knob") or getattr(env, "knob", None)
        self.burner_name = knob.replace("_", " ") if knob else ""
        self.knob_adr = None
        if knob is not None:
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{knob}_joint")
            except Exception:
                self.knob_adr = None
        self.extra_subtasks = {SUBTASK_HEAT}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        v = abs(float(env.sim.data.qpos[self.knob_adr])) if self.knob_adr is not None else 0.0
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knob": float(min(v, 2 * np.pi - v)),
        }
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
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

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        moved_words = [self.word[o] for o in order]
        same = len(order) == 2 and moved_words[0] == moved_words[1]

        def noun(o, k):
            w = self.word[o]
            if same:
                return f"the other {w}" if k == 1 else f"a {w}"
            return f"the {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for k, o in enumerate(order):
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            nn = noun(o, k)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} in the pan")
            n_moved += 1

        # AdjustHeat: the burner is turned to medium after the vegetables are in.
        review = []
        final = float(np.median(knob[-self.REST_W:])) if T else 0.0
        has_heat = final >= KNOB_ON
        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"
        if has_heat and cursor < T:
            below = np.where(knob[cursor:] < final - 0.05)[0]
            t_turn = int(min(cursor + int(below[-1]) + 2, T - 1)) if below.size else T - 1
            emit(max(t_turn, cursor + 1), SUBTASK_HEAT, STAGE_EXECUTE,
                 f"turn {burner} to medium heat")
            if not (MEDIUM_LO <= final <= MEDIUM_HI):
                review.append(f"final heat {final:.2f} outside the medium band -- verify")
        elif not has_heat:
            review.append("burner never turned on -- verify")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 vegetables moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "final_heat": round(final, 2),
            "has_heat": int(has_heat), "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_HEAT)) != 1:
            issues.append(f"expected 1 AdjustHeat, found {int(np.sum(names == SUBTASK_HEAT))}")
        return issues
