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


# Open the (closed, double-door) cabinet, take the rolling pin out, and set it next to the
# meat on the cutting board (on the counter) to prep for tenderizing. OpenCabinet is a bank
# atomic; cabinet -> counter has no bank name -> PickPlaceCabinetToCounter (extension). The
# robot starts at the cabinet; the carry to the board is usually local, nav is adaptive.
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class PrepForTenderizingLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepForTenderizing composite task.

    Task (mobile, usually local): a rolling pin sits in a CLOSED double-door cabinet; the
    meat sits on a cutting board on the counter beside it. Open the cabinet, then place the
    rolling pin next to the meat on the cutting board. The robot starts at the cabinet.

    Doc-canonical decomposition (2 high-level: open, rolling-pin move). Emitted:
      OpenCabinet [execute]              -> open the cabinet door(s)
      PickPlaceCabinetToCounter [pick]   -> pick up the rolling pin from the cabinet
      (NavigateKitchen [navigate])       -> carry the rolling pin to the cutting board  (adaptive)
      PickPlaceCabinetToCounter [place]  -> place the rolling pin next to the meat on the cutting board
    plus a terminal done.

    Detection: the cabinet door(s) swing from shut toward open; OpenCabinet ends when the
    LAST opened leaf reaches its (adaptive) open threshold min(OPEN_ABS, 0.9*peak) -- a
    double-door cabinet where the second leaf only opens partway. The rolling-pin pick
    begins at its first sustained departure; an adaptive nav segment covers the carry when
    the base actually drives; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PrepForTenderizing"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    OPEN_ABS = 0.85
    OPEN_MIN = 0.25
    OPEN_SUSTAIN = 5
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.25

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["rolling_pin"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
            pref = cab.name
            joint_names = [n for n in env.sim.model.joint_names
                           if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": doors,
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
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
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

        # OpenCabinet: end when the LAST opened leaf reaches its adaptive open level.
        open_frames, n_open = [], 0
        for i in range(doorv.shape[1]):
            col = doorv[:, i]
            peak = float(col.max())
            if peak < self.OPEN_MIN:
                continue
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            fr = self._sustained_start(col >= lvl, 0, self.OPEN_SUSTAIN)
            if fr is not None:
                open_frames.append(fr)
                n_open += 1
        t_open = max(open_frames) if open_frames else 0
        t_open = int(min(t_open, max(t_pick - 1, 0)))
        door_word = "doors" if n_open >= 2 else "door"

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
        if t_open > 0 and n_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the cabinet {door_word}")
        elif len(self.door_adrs) > 0:
            review.append("cabinet has a door but it never opened -- verify")

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the rolling pin from the cabinet")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the rolling pin to the cutting board")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, "place the rolling pin next to the meat on the cutting board")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "n_open": n_open, "t_pick": t_pick,
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
            issues.append(f"expected 0-1 OpenCabinet, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 rolling-pin pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 rolling-pin place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
