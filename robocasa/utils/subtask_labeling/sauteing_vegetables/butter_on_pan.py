import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Move the butter from its plate into the pan on the stove, then turn the pan's burner
# on. Plate -> pan is the bank atomic PickPlaceCounterToStove; TurnOnStove and
# NavigateKitchen are bank atomics. The pan starts on an OFF burner.
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_TURN = "TurnOnStove"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

KNOB_ON = 0.35


@register_labeler
class ButterOnPanLabeler(BaseSubtaskLabeler):
    """Labeler for the ButterOnPan composite task.

    Task (mobile, usually local): a butter stick sits on a plate on the counter; a pan
    sits on an OFF burner. Put the butter in the pan, then turn the burner on. The robot
    starts at the stove.

    Doc-canonical count = 2 (`mobile: Yes`): the butter's plate->pan move + TurnOnStove.
    Structure:
      PickPlaceCounterToStove [pick]   -> pick up the butter from the plate
      (NavigateKitchen [navigate])     -> carry the butter to the pan (adaptive)
      PickPlaceCounterToStove [place]  -> place the butter in the pan
      TurnOnStove [execute]            -> turn on the burner
    plus a terminal done.

    Boundaries: t_pick = the butter first displaces; the carry nav is split off the base
    motion; t_place = the gripper RELEASE; TurnOnStove runs from the place to the
    turn-completion (last frame the burner knob is below its settled value).
    """

    ENV_NAME = "ButterOnPan"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.25
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["butter"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.butter_lang = env.get_obj_lang("butter")
        except Exception:
            self.butter_lang = "butter"
        knob = ep_meta.get("refs", {}).get("knob") or getattr(env, "knob", None)
        self.burner_name = knob.replace("_", " ") if knob else ""
        self.knob_adr = None
        if knob is not None:
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{knob}_joint")
            except Exception:
                self.knob_adr = None
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = set()  # both moves are bank atomics
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        v = abs(float(env.sim.data.qpos[self.knob_adr])) if self.knob_adr is not None else 0.0
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "knob": float(min(v, 2 * np.pi - v)),
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
        knob = np.array([h["knob"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        gd = np.linalg.norm(grip - obj, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.butter_lang} from the plate")
        carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.butter_lang} to the pan")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.butter_lang} in the pan")

        review = []
        final = float(np.median(knob[-self.REST_W:])) if T else 0.0
        has_heat = final >= KNOB_ON
        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"
        if has_heat and cursor < T:
            below = np.where(knob[cursor:] < final - 0.05)[0]
            t_turn = int(min(cursor + int(below[-1]) + 2, T - 1)) if below.size else T - 1
            emit(max(t_turn, cursor + 1), SUBTASK_TURN, STAGE_EXECUTE, f"turn on {burner}")
        elif not has_heat:
            review.append("burner never turned on -- verify")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pick": t_pick, "t_placed": t_placed, "final_heat": round(final, 2),
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

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 butter pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 butter place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_TURN) != 1:
            issues.append(f"expected 1 TurnOnStove, found {count(SUBTASK_TURN)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
