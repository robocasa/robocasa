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


# The microwave has finished; a bowl of food sits inside the (CLOSED) microwave. Open the
# door, take the bowl out, and carry it to the dining counter. OpenMicrowave +
# NavigateKitchen + PickPlaceMicrowaveToCounter are bank atomics.
SUBTASK_OPEN = "OpenMicrowave"
SUBTASK_MOVE = "PickPlaceMicrowaveToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class ReturnHeatedFoodLabeler(BaseSubtaskLabeler):
    """Labeler for the ReturnHeatedFood composite task.

    Task (mobile): a bowl of heated food sits inside a CLOSED microwave; a dining counter is
    across the kitchen (robot starts at the microwave). Open the microwave, take the bowl out,
    and place it on the dining counter.

    Doc-canonical count = 4 (`mobile: Yes`): OpenMicrowave + the bowl's nav-interrupted
    microwave->dining move (pick / nav / place). Structure:
      OpenMicrowave [execute] -> bowl[pick] -> NavigateKitchen -> [place] -> done.

    Boundaries: OpenMicrowave ends when the door reaches its adaptive open level
    min(OPEN_ABS, 0.9*peak) sustained. t_pick = the bowl first departs the microwave; nav =
    the carry to the base stop (adaptive); t_place = the gripper RELEASE on the dining counter.
    """

    ENV_NAME = "ReturnHeatedFood"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    OPEN_ABS = 0.60
    OPEN_MIN = 0.15
    OPEN_SUSTAIN = 5
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["bowl"]
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

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": abs(float(env.sim.data.qpos[self.door_adr])) / self.door_scale,
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
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_place = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        # OpenMicrowave: the single door swings open to its adaptive level.
        peak = float(door.max())
        has_open = peak >= self.OPEN_MIN
        if has_open:
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            t_open = self._sustained_start(door >= lvl, 0, self.OPEN_SUSTAIN)
            if t_open is None:
                t_open = int(np.argmax(door))
            t_open = int(min(t_open, max(t_pick - 1, 0)))
        else:
            t_open = 0

        # adaptive nav during the carry (pick -> place)
        moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
        nav_disp = (
            float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
            if base_xy is not None and t_arrive > t_pick else 0.0
        )

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if has_open and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the microwave door")
        elif not has_open:
            review.append("microwave door never opened -- verify")

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the bowl from the microwave")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the bowl to the dining counter")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, "place the bowl on the dining counter")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "door_peak": round(peak, 2), "t_pick": t_pick,
            "t_arrive": t_arrive, "t_place": t_place, "nav_disp": round(nav_disp, 2),
            "n_segments": len(segments),
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

        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenMicrowave, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 bowl pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 bowl place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
