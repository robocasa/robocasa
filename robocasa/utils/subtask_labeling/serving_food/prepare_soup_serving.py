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


# Open the (closed) cabinet, move the ladle into the pot on the stove, then close the
# cabinet. OpenCabinet, CloseCabinet and NavigateKitchen are bank atomics; cabinet ->
# pot has no bank name -> PickPlaceCabinetToPot (extension). The robot starts at the
# cabinet (which sits next to the stove).
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToPot"
SUBTASK_CLOSE = "CloseCabinet"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# The cabinet is either an overhead cabinet or one at counter height; the open/close
# language says which, from the per-episode cabinet z height.
CAB_ABOVE_Z = 1.80


@register_labeler
class PrepareSoupServingLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareSoupServing composite task.

    Task (mobile, usually local): a ladle sits in a CLOSED cabinet next to the stove
    (where the robot starts); a pot sits on the stove. Open the cabinet, put the ladle
    in the pot, then close the cabinet.

    Doc-canonical count = 3 (`mobile: Yes`):
      OpenCabinet [execute]           -> open the cabinet door(s)
      PickPlaceCabinetToPot [pick/place] (+ nav) -> put the ladle in the pot
      CloseCabinet [execute]          -> close the cabinet door(s)
    plus a terminal done.

    Boundaries: OpenCabinet ends at OPEN_FRAC of the door peak (before the ladle
    departs); t_pick = ladle first move; t_place = the gripper RELEASE; CloseCabinet
    runs from the door's closing onset (after the ladle is placed) until it latches shut.
    """

    ENV_NAME = "PrepareSoupServing"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    REST_W = 15
    DOOR_MIN_PEAK = 0.5
    OPEN_FRAC = 0.85
    CLOSE_FRAC = 0.30
    OPEN_SUSTAIN = 5
    OPEN_DOOR_TH = 0.5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["ladle"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        cab = getattr(env, "cabinet", None) or getattr(env, "cab", None)
        nm = getattr(cab, "name", None)
        try:
            self.open_position = "above" if float(cab.pos[2]) >= CAB_ABOVE_Z else "in front"
        except Exception:
            self.open_position = "in front"
        self.door_addrs = []
        if nm is not None:
            for j in range(env.sim.model.njnt):
                try:
                    jn = env.sim.model.joint_id2name(j)
                except Exception:
                    jn = None
                if jn and jn.startswith(nm) and "doorhinge" in jn:
                    self.door_addrs.append(env.sim.model.jnt_qposadr[j])
        self.base_qpos_adrs = []
        for nm2 in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm2)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = (np.array([abs(float(env.sim.data.qpos[a])) for a in self.door_addrs])
                 if self.door_addrs else np.zeros(0))
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": doors,
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
        door = doorv.sum(axis=1) if doorv.shape[1] else np.zeros(T)
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
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        peak = float(door.max()) if T else 0.0
        has_open = peak >= self.DOOR_MIN_PEAK
        n_opened = int(sum(float(doorv[:, i].max()) >= self.OPEN_DOOR_TH
                           for i in range(doorv.shape[1]))) if doorv.shape[1] else 0
        door_word = "doors" if n_opened >= 2 else "door"

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

        # --- OpenCabinet (before the ladle departs) ---
        if has_open:
            lvl = self.OPEN_FRAC * peak
            fo = self._sustained_start(door[:max(t_pick, 1)] >= lvl, 0, self.OPEN_SUSTAIN)
            if fo is None:
                w = np.where(door >= lvl)[0]
                fo = int(w[0]) if w.size else 0
            t_open = int(min(fo, max(t_pick - 1, 0)))
            if t_open > 0:
                emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the cabinet {self.open_position}")
        else:
            review.append("cabinet door never opened -- verify")

        # --- ladle: cabinet -> pot ---
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the ladle from the cabinet")
        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the ladle to the pot on the stove")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the ladle in the pot")

        # --- CloseCabinet (door swings back shut after the ladle is placed) ---
        if has_open:
            close_onset = self._sustained_start(door[t_placed:] < self.OPEN_FRAC * peak, 0, self.OPEN_SUSTAIN)
            close_onset = (t_placed + close_onset) if close_onset is not None else t_placed
            # optional drive back to the cabinet before closing
            to_moving = np.where(binst[t_placed: close_onset] > self.BASE_MOVE_TH)[0]
            t_back = min(t_placed + int(to_moving[-1]) + 1, close_onset) if to_moving.size else t_placed
            if t_back > cursor and base_disp(t_placed, t_back) >= self.MIN_NAV_DIST:
                emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the open cabinet")
            t_closed = self._sustained_start(door[close_onset:] <= self.CLOSE_FRAC * peak, 0, self.OPEN_SUSTAIN)
            t_closed = (close_onset + t_closed) if t_closed is not None else T
            if t_closed > cursor:
                emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the cabinet {self.open_position}")
            else:
                review.append("cabinet never detected closing -- verify")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "door_peak": round(peak, 2), "t_pick": int(t_pick), "t_placed": int(t_placed),
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
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
            issues.append(f"expected 1 ladle pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 ladle place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseCabinet, found {count(SUBTASK_CLOSE)}")
        return issues
