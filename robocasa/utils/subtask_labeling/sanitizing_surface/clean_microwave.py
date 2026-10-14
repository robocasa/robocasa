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


# Open the (closed) microwave, then put the sponge from the counter inside it to
# prepare for sanitizing. OpenMicrowave, PickPlaceCounterToMicrowave, and
# NavigateKitchen are bank atomics. No close / turn-on (success only needs the
# sponge inside). The sponge object is named "obj" in the env.
SUBTASK_OPEN = "OpenMicrowave"
SUBTASK_MOVE = "PickPlaceCounterToMicrowave"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the microwave"
DONE_LANG = "task complete"


@register_labeler
class CleanMicrowaveLabeler(BaseSubtaskLabeler):
    """Labeler for the CleanMicrowave composite task.

    Task (mobile, usually local): a sponge sits on the counter and the microwave
    starts CLOSED. Open the microwave, then place the sponge inside it. The robot
    starts at the microwave.

    Doc-canonical count = 5 (`mobile: Yes`), action-level:
      OpenMicrowave [execute]              -> open the microwave
      PickPlaceCounterToMicrowave [pick]   -> pick up the sponge from the counter
      (NavigateKitchen [navigate])         -> carry the sponge to the microwave (adaptive)
      PickPlaceCounterToMicrowave [place]  -> place the sponge in the microwave
    plus a terminal done.

    Boundaries: OpenMicrowave ends when the door reaches its adaptive open level
    (capped before the sponge departs); t_pick = the sponge first displaces; the
    carry nav is split off the base motion; t_place = the gripper RELEASE.
    """

    ENV_NAME = "CleanMicrowave"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.25
    REST_W = 15
    OPEN_ABS = 0.60
    OPEN_MIN = 0.15
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        mw = env.microwave
        self.door_adr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
        jid = env.sim.model.joint_name2id(mw.door_joint_names[0])
        lo, hi = env.sim.model.jnt_range[jid]
        self.door_scale = max(abs(float(hi)), abs(float(lo)), 1e-6)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = set()  # OpenMicrowave + PickPlaceCounterToMicrowave are bank atomics
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "door": abs(float(env.sim.data.qpos[self.door_adr])) / self.door_scale,
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

        # OpenMicrowave: the door swings open (from closed) before the sponge is
        # taken to it. End it just before the sponge departs.
        peak = float(door.max())
        has_open = peak >= self.OPEN_MIN
        review = []
        t_open = 0
        if has_open:
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            fo = self._sustained_start(door >= lvl, 0, self.OPEN_SUSTAIN)
            if fo is None:
                fo = int(np.argmax(door))
            t_open = int(min(fo, max(t_pick - 1, 0)))
        else:
            review.append("microwave door never opened -- verify")

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        if has_open and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the sponge from the nearest counter")
        carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the sponge to the microwave")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the sponge in the microwave")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": int(t_open), "door_peak": round(peak, 2), "t_pick": t_pick,
            "t_placed": t_placed, "n_segments": len(segments),
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

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenMicrowave, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 sponge pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 sponge place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
