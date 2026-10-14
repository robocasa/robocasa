import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Take the frozen food out of the (already-open) fridge and set it on the plate on
# the counter. The fridge->counter move has no exact bank name so it is an
# extra_subtask; NavigateKitchen is a bank atomic.
SUBTASK_MOVE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class MoveToCounterLabeler(BaseSubtaskLabeler):
    """Labeler for the MoveToCounter composite task.

    Task: a frozen food (meat or vegetable) sits in an ALREADY-OPEN fridge (the robot
    starts at the fridge); a plate sits on the counter. Carry the food to the plate.

    Doc-canonical count = 3 (`mobile: Yes`): the food's pick-place, INTERRUPTED by a
    navigation from the fridge to the counter -- PickPlaceFridgeToCounter[pick] ->
    NavigateKitchen -> [place]. 3 subtasks + done. The nav is ADAPTIVE (gated on
    MIN_NAV_DIST).

    Boundaries (base-stop convention): t_pick = the food first displaces from its
    in-fridge rest; the nav is the carry (to the base stop); the place is the
    set-down + gripper RELEASE after the base parks.
    """

    ENV_NAME = "MoveToCounter"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.food_body_id = env.obj_body_id["frosted_food"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.food_lang = env.get_obj_lang("frosted_food")
        except Exception:
            self.food_lang = "food"
        try:
            self.plate_lang = env.get_obj_lang("plate")
        except Exception:
            self.plate_lang = "plate"

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "food": env.sim.data.body_xpos[self.food_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
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
        food = np.array([h["food"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(food[: self.REST_W], axis=0)
        disp = np.linalg.norm(food - rest, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 0)

        gd = np.linalg.norm(grip - food, axis=1)
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

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.food_lang} from the fridge")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE,
                 f"carry the {self.food_lang} to the counter with the plate")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.food_lang} on the {self.plate_lang}")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1
        return self.build_result(T, segments, debug={
            "t_pick": t_pick, "t_arrive": t_arrive, "t_placed": t_placed,
            "nav_disp": round(nav_disp, 2), "n_segments": len(segments),
        })

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
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
