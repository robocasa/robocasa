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


# Take meat from the fridge and lay it on the pan of marinade on the stove. The fridge starts
# closed, so the robot drives to it, opens it, takes the meat, and carries it back to the pan.
# OpenFridge is a bank atomic; fridge -> pan (on the stove) has no bank name ->
# PickPlaceFridgeToStove (extension); NavigateKitchen is a bank atomic.
SUBTASK_OPEN = "OpenFridge"
SUBTASK_MOVE = "PickPlaceFridgeToStove"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class PlaceMeatInMarinadeLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceMeatInMarinade composite task.

    Task (mobile): a meat sits in the closed fridge; a pan of marinade sits on the stove (where
    the robot starts). Open the fridge, take the meat, and place it on the pan.

    Doc-canonical count = 5 (`mobile: Yes`):
      (NavigateKitchen [navigate])       -> navigate to the fridge  (adaptive)
      OpenFridge [execute]               -> open the fridge door(s)
      PickPlaceFridgeToStove [pick]      -> pick up the {meat} from the fridge
      NavigateKitchen [navigate]         -> carry the {meat} to the pan on the stove  (adaptive)
      PickPlaceFridgeToStove [place]     -> place the {meat} on the pan of marinade
    plus a terminal done.

    Boundaries: OpenFridge ends when the fridge door(s) reach their adaptive open level (bounded
    before the meat is grabbed); t_pick = the meat's first sustained displacement; the carry nav
    ends at the last substantial base-motion frame; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PlaceMeatInMarinade"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    MIN_PICK_GAP = 12
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["meat"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.meat = env.get_obj_lang("meat")
        except Exception:
            self.meat = "meat"
        fr = getattr(env, "fridge", None)
        self.door_adrs, self.door_scales = [], []
        for n in env.sim.model.joint_names:
            if not n or fr is None or fr.name not in n:
                continue
            low = n.lower()
            if "door" not in low or "freezer" in low or "drawer" in low:
                continue
            try:
                adr = env.sim.model.get_joint_qpos_addr(n)
                jid = env.sim.model.joint_name2id(n)
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
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": np.array([abs(float(env.sim.data.qpos[a])) / s
                               for a, s in zip(self.door_adrs, self.door_scales)]),
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
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

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

        # --- OpenFridge (doors open before the meat is grabbed) ---
        t_open, n_doors = None, 0
        if doorv.shape[1]:
            frames = []
            for i in range(doorv.shape[1]):
                col = doorv[:, i]
                seg = col[:t_pick]
                if seg.size == 0:
                    continue
                peak = float(seg.max())
                if peak < self.OPEN_MIN or peak - float(col[0]) < 0.10:
                    continue
                lvl = min(self.OPEN_ABS, 0.9 * peak)
                fr = self._sustained_start(col[:t_pick] >= lvl, 0, self.OPEN_SUSTAIN)
                if fr is not None:
                    frames.append(fr)
                    n_doors += 1
            if frames:
                t_open = min(max(frames), max(t_pick - self.MIN_PICK_GAP, 1))
        if t_open is not None and t_open > cursor:
            # break out the drive to the fridge (before the doors move)
            door_start = None
            if doorv.shape[1]:
                anyopen = (doorv[:t_pick] > self.OPEN_MIN).any(axis=1)
                door_start = self._sustained_start(anyopen, 0, self.OPEN_SUSTAIN)
            nav_end = door_start if door_start is not None else t_open
            moving = np.where(binst[:nav_end] > self.BASE_MOVE_TH)[0]
            t_nav = min(int(moving[-1]) + 1, nav_end) if moving.size else 0
            if t_nav > cursor and base_disp(0, t_nav) >= self.MIN_NAV_DIST:
                emit(t_nav, SUBTASK_NAV, STAGE_NAVIGATE, "navigate to the fridge")
            door_word = "doors" if n_doors >= 2 else "door"
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the fridge {door_word}")
        elif doorv.shape[1]:
            review.append("fridge never detected opening -- verify")

        # --- meat: fridge -> pan on the stove ---
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.meat} from the fridge")
        moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
        t_arrive = min(t_arrive, t_place - self.MIN_PICK_GAP)
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.meat} to the pan on the stove")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.meat} on the pan of marinade")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "n_doors": n_doors, "t_pick": int(t_pick), "t_place": int(t_place),
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

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 meat pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 meat place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenFridge, found {count(SUBTASK_OPEN)}")
        return issues
