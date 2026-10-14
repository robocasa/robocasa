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


# Take the mug off the dish rack, put it in the (already-open) cabinet, and close the cabinet.
# Dish-rack -> cabinet has no bank name -> PickPlaceDishRackToCabinet (extension); CloseCabinet
# is a bank atomic. The robot starts at the dish rack; the carry is usually local (adaptive nav).
SUBTASK_MOVE = "PickPlaceDishRackToCabinet"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"


@register_labeler
class EmptyDishRackLabeler(BaseSubtaskLabeler):
    """Labeler for the EmptyDishRack composite task.

    Task (mobile, usually local): a mug sits on the dish rack; an open cabinet holds two other
    mugs. Move the rack mug into the cabinet, then close the cabinet door(s). The robot starts
    at the dish rack.

    Doc-canonical decomposition (2 high-level: mug move + close). Emitted:
      PickPlaceDishRackToCabinet [pick]   -> pick up the mug from the dish rack
      (NavigateKitchen [navigate])        -> carry the mug to the cabinet  (adaptive)
      PickPlaceDishRackToCabinet [place]  -> place the mug in the cabinet
      CloseCabinet [execute]              -> close the cabinet door(s)
    plus a terminal done.

    Boundaries: t_pick = the mug first displaces from its rest; nav = the carry to the base stop
    (adaptive); t_place = the gripper RELEASE. CloseCabinet ends when the LAST door that
    transitions open->shut reaches closed (doors that stay open belong to a neighbouring
    cabinet); it pluralizes by how many doors close.
    """

    ENV_NAME = "EmptyDishRack"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    OPEN_MIN = 0.30
    CLOSE_ABS = 0.20
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["mug"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        cab = env.cabinet
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

    def _close_cabinet(self, doorv, cursor, T):
        frames, n = [], 0
        for i in range(doorv.shape[1]):
            col = doorv[:, i]
            started_open = float(col[: max(cursor, 1)].max()) >= self.OPEN_MIN
            ends_closed = float(np.median(col[-self.REST_W:])) < self.CLOSE_ABS
            if not (started_open and ends_closed):
                continue
            fr = self._sustained_start(col < self.CLOSE_ABS, cursor, self.CLOSE_SUSTAIN)
            if fr is not None:
                frames.append(fr)
                n += 1
        return (max(frames) if frames else None), n

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
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
        nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                    if base_xy is not None and t_arrive > t_pick else 0.0)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the mug from the dish rack")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the mug to the open cabinet")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, "place the mug in the cabinet")

        review = []
        t_close, n_close = self._close_cabinet(doorv, cursor, T)
        if t_close is not None and t_close > cursor:
            door_word = "doors" if n_close >= 2 else "door"
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the cabinet {door_word}")
        elif len(self.door_adrs) > 0:
            review.append("cabinet has a door but it never closed -- verify")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_arrive": int(t_arrive), "t_place": int(t_place),
            "nav_disp": round(nav_disp, 2), "n_close": n_close, "n_segments": len(segments),
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
            issues.append(f"expected 1 mug pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 mug place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseCabinet, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
