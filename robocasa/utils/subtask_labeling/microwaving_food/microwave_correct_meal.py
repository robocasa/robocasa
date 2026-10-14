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


# Two bowls of food sit on the counter; ONE (the target, chosen at generation time and
# recorded in ep_meta refs) is placed in the (already-open) microwave, the door closed, and
# the microwave started. Counter -> microwave is the bank atomic PickPlaceCounterToMicrowave;
# NavigateKitchen + CloseMicrowave + TurnOnMicrowave are bank atomics too. No OpenMicrowave
# (it starts open). The other bowl is a distractor and never moves.
SUBTASK_MOVE = "PickPlaceCounterToMicrowave"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseMicrowave"
SUBTASK_ON = "TurnOnMicrowave"

DONE_LANG = "task complete"


@register_labeler
class MicrowaveCorrectMealLabeler(BaseSubtaskLabeler):
    """Labeler for the MicrowaveCorrectMeal composite task.

    Task (mobile, but the bowls sit on the counter by the microwave so nav is usually
    nil): two bowls each hold two foods; the TARGET bowl (ep_meta refs["target_bowl"] in
    {0,1}) must go into the ALREADY-OPEN microwave, then the door is closed and the start
    button pressed. The non-target bowl is a distractor.

    Doc-canonical count = 5 (`mobile: Yes`): the target bowl's (optionally nav-interrupted)
    counter->microwave move (pick / nav / place) + CloseMicrowave + TurnOnMicrowave.
    Structure: bowl[pick] -> (NavigateKitchen) -> [place] -> CloseMicrowave ->
    TurnOnMicrowave -> done. The target is identified from refs (deterministic across
    surfaces), NOT from motion, and named by its contents in the lang.

    Boundaries: t_pick = the target bowl first displaces from its rest; nav = the carry to
    the base stop (adaptive, usually absent); place ends at the gripper RELEASE (bowl set in
    the microwave). CloseMicrowave = the door swings shut after the place. TurnOnMicrowave =
    from the close to the end (reach + press start).
    """

    ENV_NAME = "MicrowaveCorrectMeal"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    CLOSE_FRAC = 0.10    # the door must be essentially LATCHED (< 10% of its open swing)
                         # before the turn-on -- ending the close at 30% left it visibly ajar.
    DOOR_SUSTAIN = 5
    OPEN_MIN = 0.05
    BUTTON_TH = 0.09
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        tb = int((ep_meta.get("refs", {}) or {}).get("target_bowl", getattr(env, "target_bowl", 0)))
        self.target_bowl = tb
        self.bowl_name = f"bowl{tb}"
        self.obj_body_id = env.obj_body_id[self.bowl_name]
        self.eef_site_id = env.robots[0].eef_site_id["right"]

        # describe the target bowl by its two foods
        foods = []
        for k in (f"food0_bowl{tb}", f"food1_bowl{tb}"):
            try:
                foods.append(env.get_obj_lang(k))
            except Exception:
                pass
        self.food_desc = " and ".join(foods) if foods else "food"

        mw = env.microwave
        self.door_adr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
        try:
            self.btn_gid = env.sim.model.geom_name2id(f"{mw.name}_start_button")
        except Exception:
            self.btn_gid = None

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
            "door": abs(float(env.sim.data.qpos[self.door_adr])),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "btn": (
                float(np.linalg.norm(env.sim.data.site_xpos[self.eef_site_id]
                                     - env.sim.data.geom_xpos[self.btn_gid]))
                if self.btn_gid is not None else 9.9
            ),
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
        door = np.array([h["door"] for h in history])
        btn = np.array([h["btn"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        door_peak = float(door.max())

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
        nav_disp = (
            float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
            if base_xy is not None and t_arrive > t_pick else 0.0
        )

        t_closed = (
            self._sustained_start(door < self.CLOSE_FRAC * door_peak, t_placed, self.DOOR_SUSTAIN)
            if door_peak >= self.OPEN_MIN else None
        )

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the bowl with the {self.food_desc}")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the bowl to the microwave")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the bowl in the microwave")
        has_close = False
        if t_closed is not None and t_closed > cursor:
            emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, "close the microwave door")
            has_close = True
        pressed = bool((btn[cursor:] < self.BUTTON_TH).any()) if cursor < T else False
        if cursor < T - 1:
            emit(T - 1, SUBTASK_ON, STAGE_EXECUTE, "turn on the microwave")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_close:
            review.append("microwave door never closed -- verify")
        if not pressed:
            review.append("gripper never neared the start button -- verify turn-on")

        result = self.build_result(T, segments, debug={
            "target_bowl": self.target_bowl, "t_pick": t_pick, "t_arrive": t_arrive,
            "t_placed": t_placed, "t_closed": t_closed if t_closed is not None else -1,
            "door_peak": round(door_peak, 2), "nav_disp": round(nav_disp, 2),
            "pressed": pressed, "n_segments": len(segments),
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
            issues.append(f"expected 1 bowl pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 bowl place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseMicrowave, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_ON) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnMicrowave, found {count(SUBTASK_ON)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
