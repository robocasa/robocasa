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


# Carry the pan (with the steak in it) from the stove to the dining table, then take the
# steak out and set it on the plate. NavigateKitchen is a bank atomic; stove -> dining
# table and pan -> plate have no bank names -> PickPlaceStoveToDining / PickPlaceToPlate
# (extensions). The steak is "obj"; the pan it sits in is "obj_container".
MOVE_PAN = "PickPlaceStoveToDining"
MOVE_STEAK = "PickPlaceToPlate"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class ServeSteakLabeler(BaseSubtaskLabeler):
    """Labeler for the ServeSteak composite task.

    Task (mobile): a steak sits in a pan on the stove (where the robot starts); a plate
    sits on the dining table. Carry the pan to the dining table, then move the steak onto
    the plate.

    Doc-canonical count = 4 (`mobile: Yes`), action-level:
      PickPlaceStoveToDining [pick]   -> pick up the pan with the steak from the stove
      NavigateKitchen [navigate]      -> carry the pan to the dining table
      PickPlaceStoveToDining [place]  -> place the pan on the dining table
      PickPlaceToPlate [pick]         -> pick up the steak from the pan
      PickPlaceToPlate [place]        -> place the steak on the plate
    plus a terminal done.

    Boundaries: the pan carry is split off the base motion; the steak's rest is
    re-anchored once the pan is set down (it rides inside the pan until then), so its
    t_pick is the moment it leaves the pan. Places = the gripper RELEASE.
    """

    ENV_NAME = "ServeSteak"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    REST_W = 15
    SETTLE_TH = 0.004   # m/frame; below this the carried object is at rest
    SETTLE_W = 12       # sustained rest frames that mark the set-down

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.steak_id = env.obj_body_id["obj"]
        self.pan_id = env.obj_body_id.get("obj_container", self.steak_id)
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.steak = env.get_obj_lang("obj")
        except Exception:
            self.steak = "steak"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {MOVE_PAN, MOVE_STEAK}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "steak": env.sim.data.body_xpos[self.steak_id].copy(),
            "pan": env.sim.data.body_xpos[self.pan_id].copy(),
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

    def _release(self, gd, t_pick, T):
        hold = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t = self._sustained_start(gd > hold + self.RELEASE_MARGIN,
                                  t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        return t if t is not None else T - 1

    def _place_by_settle(self, pos, t_pick, T):
        # the carried object is "placed" when it first comes to sustained rest after the
        # carry -- robust to the gripper lingering (reaching into the pan for the steak).
        speed = np.concatenate([np.linalg.norm(np.diff(pos, axis=0), axis=1), [0.0]])
        run = 0
        for t in range(int(min(t_pick + self.MIN_CARRY, T - 1)), T):
            if speed[t] <= self.SETTLE_TH:
                run += 1
                if run >= self.SETTLE_W:
                    return t - self.SETTLE_W + 1
            else:
                run = 0
        return T - 1

    def resolve(self, history):
        T = len(history)
        steak = np.array([h["steak"] for h in history])
        pan = np.array([h["pan"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- pan (with steak) stove -> dining table ---
        rest = np.median(pan[: self.REST_W], axis=0)
        pdisp = np.linalg.norm(pan - rest, axis=1)
        t_pan_pick = self._sustained_start(pdisp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pan_pick is None:
            t_pan_pick = max(T - 1, 1)
        t_pan_placed = self._place_by_settle(pan, t_pan_pick, T)

        emit(t_pan_pick, MOVE_PAN, STAGE_PICK, f"pick up the pan with the {self.steak} from the stove")
        moving = np.where(binst[t_pan_pick: t_pan_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pan_pick + int(moving[-1]) + 1, t_pan_placed) if moving.size else t_pan_pick
        t_arrive = min(t_arrive, max(t_pan_placed - self.MIN_PICK_GAP, cursor))
        if t_arrive > cursor and base_disp(t_pan_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the pan to the plate on the dining table")
        emit(t_pan_placed, MOVE_PAN, STAGE_PLACE, "place the pan on the dining table")

        # --- steak pan -> plate (re-anchor FORWARD once the pan is down; the steak rode
        # inside the pan during the carry, so a backward window would sit mid-motion) ---
        srest = np.median(steak[t_pan_placed: min(t_pan_placed + self.REST_W, T)], axis=0)
        sdisp = np.linalg.norm(steak - srest, axis=1)
        t_steak_pick = self._sustained_start(sdisp >= self.MOVE_TOL, t_pan_placed, self.MOVE_SUSTAIN)
        if t_steak_pick is None:
            mv = np.where(sdisp[t_pan_placed:] >= self.MOVE_TOL)[0]
            t_steak_pick = t_pan_placed + int(mv[0]) if mv.size else min(t_pan_placed + 1, T - 1)
        t_steak_pick = max(t_steak_pick, cursor)
        gd_steak = np.linalg.norm(grip - steak, axis=1)
        t_steak_placed = self._release(gd_steak, t_steak_pick, T)

        emit(t_steak_pick, MOVE_STEAK, STAGE_PICK, f"pick up the {self.steak} from the pan")
        emit(t_steak_placed, MOVE_STEAK, STAGE_PLACE, f"place the {self.steak} on the plate")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pan_pick": int(t_pan_pick), "t_pan_placed": int(t_pan_placed),
            "t_steak_pick": int(t_steak_pick), "t_steak_placed": int(t_steak_placed),
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {MOVE_PAN, MOVE_STEAK}
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

        if count(MOVE_PAN, STAGE_PICK) != 1 or count(MOVE_PAN, STAGE_PLACE) != 1:
            issues.append("expected 1 pan pick + 1 pan place")
        if count(MOVE_STEAK, STAGE_PICK) != 1 or count(MOVE_STEAK, STAGE_PLACE) != 1:
            issues.append("expected 1 steak pick + 1 steak place")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
