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


# A fish sits on a plate on the counter; a pan (with soy sauce) sits on a stove burner.
# Put the fish on the pan and turn that burner on to start steaming. Counter -> pan (on the
# stove) is the bank atomic PickPlaceCounterToStove; TurnOnStove is a bank atomic. The robot
# starts at the counter with the fish (local).
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_ON = "TurnOnStove"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

KNOB_ON = 0.35


@register_labeler
class SteamFishLabeler(BaseSubtaskLabeler):
    """Labeler for the SteamFish composite task.

    Task (mobile, local): a fish sits on a plate on the counter beside the stove; a pan
    sits on a stove burner. Put the fish on the pan and turn that burner on.

    Doc-canonical count = 2 (`mobile: Yes`): the fish move + the burner turn-on.
      PickPlaceCounterToStove [pick]/[place] -> fish from the plate onto the pan
      TurnOnStove [execute]                  -> turn on the burner
    plus a terminal done. The fish is placed first, then the knob is turned.

    Boundaries: t_pick = the fish's first sustained displacement; t_place = the gripper
    RELEASE; TurnOnStove spans [t_place, done] (the reach + rotate). An optional carry nav
    is emitted if the base actually drives to the stove.
    """

    ENV_NAME = "SteamFish"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["fish"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fish = env.get_obj_lang("fish")
        except Exception:
            self.fish = "fish"
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
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_MOVE}
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
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                        t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        on = np.where(knob >= KNOB_ON)[0]
        t_on = int(on[0]) if on.size else None
        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.fish} from the plate")
        moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
        t_arrive = min(t_arrive, max(t_place - self.MIN_PICK_GAP, cursor))
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.fish} to the pan")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.fish} on the pan")

        if t_on is None:
            review.append("burner never turned on -- verify")
        emit(T - 1, SUBTASK_ON, STAGE_EXECUTE, f"turn on {burner}")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_place": int(t_place), "t_on": t_on,
            "n_segments": len(segments),
        })
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
            issues.append("expected 1 fish pick")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 fish place")
        if int(np.sum(names == SUBTASK_ON)) != 1:
            issues.append("expected 1 TurnOnStove")
        return issues
