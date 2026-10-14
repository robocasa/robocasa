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


# Take the frozen meat off the counter and drop it into the pot on the burner, then
# turn the burner on. PickPlaceCounterToStove + TurnOnStove are bank atomics;
# NavigateKitchen is bank. The counter is beside the stove, so the nav is adaptive.
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_KNOB = "TurnOnStove"

DONE_LANG = "task complete"


@register_labeler
class QuickThawLabeler(BaseSubtaskLabeler):
    """Labeler for the QuickThaw composite task.

    Task: frozen meat sits on the counter; a pot sits on a burner (robot starts at
    the stove, that burner off). Drop the meat into the pot, then turn the burner on.

    Doc-canonical count = 2 (`mobile: Yes`): PickPlaceCounterToStove + TurnOnStove.
    meat[pick] -> (nav) -> [place in pot] -> TurnOnStove -> done. The nav is ADAPTIVE
    (the counter is beside the stove).

    Boundaries: t_pick = meat first displaces from its counter rest; place ends at the
    gripper RELEASE. TurnOnStove spans from the placement to when the chosen burner's
    knob first rotates on (|knob angle| >= KNOB_ON) and stays -- read from the knob
    joint qpos (the stove's cached state does not track a state replay).
    """

    ENV_NAME = "QuickThaw"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    KNOB_ON = 0.35  # |knob angle| at/above this == burner on (matches is_burner_on)
    KNOB_SUSTAIN = 5
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.meat_body_id = env.obj_body_id["meat"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.meat_lang = env.get_obj_lang("meat")
        except Exception:
            self.meat_lang = "meat"

        knob = ep_meta.get("refs", {}).get("knob")
        # human-readable burner name from the knob location (e.g. "rear_left" ->
        # "rear left", "center" -> "center"); used to name which burner is turned on.
        self.burner_name = knob.replace("_", " ") if knob else ""
        self.knob_adr = None
        if knob is not None:
            jname = f"{env.stove.naming_prefix}knob_{knob}_joint"
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(jname)
            except Exception:
                self.knob_adr = None

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
            "meat": env.sim.data.body_xpos[self.meat_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "knob": float(env.sim.data.qpos[self.knob_adr]) if self.knob_adr is not None else 0.0,
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
        meat = np.array([h["meat"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knob = np.abs(np.array([h["knob"] for h in history]))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(meat[: self.REST_W], axis=0)
        disp = np.linalg.norm(meat - rest, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 0)

        gd = np.linalg.norm(grip - meat, axis=1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1

        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        if base_xy is not None and t_arrive > t_pick:
            nav_disp = float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
        else:
            nav_disp = 0.0

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.meat_lang} from the counter")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.meat_lang} to the stove")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.meat_lang} in the pot")
        # TurnOnStove: the burner's knob rotates on after the meat is placed
        has_knob = False
        if cursor < T:
            t_on = self._sustained_start(knob >= self.KNOB_ON, cursor, self.KNOB_SUSTAIN)
            if t_on is not None and t_on > cursor:
                burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"
                emit(t_on, SUBTASK_KNOB, STAGE_EXECUTE, f"turn on {burner}")
                has_knob = True
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_knob:
            review.append("burner never turned on -- verify")

        result = self.build_result(T, segments, debug={
            "t_pick": t_pick, "t_arrive": t_arrive, "t_placed": t_placed,
            "nav_disp": round(nav_disp, 2), "knob_peak": round(float(knob.max()), 2),
            "has_knob": has_knob, "n_segments": len(segments),
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
            issues.append(f"expected 1 meat pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 meat place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_KNOB) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnStove, found {count(SUBTASK_KNOB)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
