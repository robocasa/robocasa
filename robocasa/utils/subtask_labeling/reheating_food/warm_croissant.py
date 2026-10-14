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


# Warm a croissant: move it from the counter onto the pan on the stove, then turn on the burner.
# PickPlaceCounterToStove + TurnOnStove are bank atomics; NavigateKitchen is bank. The counter is
# beside the stove, so the carry is adaptive. (QuickThaw pattern.)
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_KNOB = "TurnOnStove"

DONE_LANG = "task complete"


@register_labeler
class WarmCroissantLabeler(BaseSubtaskLabeler):
    """Labeler for the WarmCroissant composite task.

    Task (mobile, usually local): a croissant sits in a plate on the counter; an (off) pan sits on
    a burner. Put the croissant on the pan and turn on the burner. The robot starts at the stove.

    Doc-canonical count = 2 (`mobile: Yes`): PickPlaceCounterToStove + TurnOnStove.
      croissant[pick] -> (NavigateKitchen carry) -> [place on the pan] -> TurnOnStove -> done.
    The nav is adaptive (the counter is beside the stove).

    Boundaries: t_pick = the croissant first displaces; place = the gripper RELEASE; TurnOnStove =
    the chosen burner's knob rotating on (|knob| >= KNOB_ON), read from the knob joint qpos.
    """

    ENV_NAME = "WarmCroissant"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    KNOB_ON = 0.35
    KNOB_SUSTAIN = 5
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["croissant"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # Identify the burner knob(s). Most demos name the target burner in
        # ep_meta refs["knob"]; ~20% of this dataset stored refs=None, so fall
        # back to scanning every burner knob and detecting which one is turned
        # on during the trajectory (the success condition requires burner_on).
        knob = ep_meta.get("refs", {}).get("knob")
        try:
            burner_locs = [loc for loc in env.stove.knob_joints
                           if env.stove.knob_joints[loc] is not None]
        except Exception:
            burner_locs = []
        cand = [knob] if knob is not None else burner_locs
        self.knob_adrs = {}
        for loc in cand:
            try:
                self.knob_adrs[loc] = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{loc}_joint")
            except Exception:
                pass
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "knobs": {loc: float(env.sim.data.qpos[a]) for loc, a in self.knob_adrs.items()},
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
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 0)
        gd = np.linalg.norm(grip - obj, axis=1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                    if base_xy is not None and t_arrive > t_pick else 0.0)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the croissant from the counter")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the croissant to the stove")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the croissant on the pan")
        has_knob = False
        knob_peak = 0.0
        if cursor < T and self.knob_adrs:
            t_on, active_loc = None, None
            for loc in self.knob_adrs:
                vals = np.array([h["knobs"].get(loc, 0.0) for h in history])
                # Match the sim's is_burner_on: normalize to [0, 2pi) then
                # require 0.35 <= v <= 2pi - 0.35.
                norm = np.mod(vals, 2 * np.pi)
                on = (norm >= self.KNOB_ON) & (norm <= 2 * np.pi - self.KNOB_ON)
                knob_peak = max(knob_peak, float(np.abs(vals).max()))
                cand_on = self._sustained_start(on, cursor, self.KNOB_SUSTAIN)
                if cand_on is not None and cand_on > cursor and (t_on is None or cand_on < t_on):
                    t_on, active_loc = cand_on, loc
            if active_loc is not None:
                burner = f"the {active_loc.replace('_', ' ')} burner"
                emit(t_on, SUBTASK_KNOB, STAGE_EXECUTE, f"turn on {burner}")
                has_knob = True
        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_knob:
            review.append("burner never turned on -- verify")

        result = self.build_result(T, segments, debug={
            "t_pick": t_pick, "t_placed": t_placed, "nav_disp": round(nav_disp, 2),
            "knob_peak": round(knob_peak, 2), "has_knob": has_knob, "n_segments": len(segments),
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
            issues.append(f"expected 1 croissant pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 croissant place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_KNOB) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnStove, found {count(SUBTASK_KNOB)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
