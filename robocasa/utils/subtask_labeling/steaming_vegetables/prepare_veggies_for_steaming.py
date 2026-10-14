import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_WAIT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A colander of vegetables sits on the counter by the sink (where the robot starts); the
# sink faucet is already running. Lift the colander, rinse the vegetables under the stream,
# then carry the colander to the counter next to the stove. Counter -> counter has no bank
# name -> PickPlaceCounterToCounter (extension); the rinse hold is a coined RinseVegetables
# [wait]; NavigateKitchen is bank.
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_RINSE = "RinseVegetables"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class PrepareVeggiesForSteamingLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareVeggiesForSteaming composite task.

    Task (mobile): a colander holding two vegetables sits on the counter by the sink, and
    the faucet is already running. Lift the colander, rinse the vegetables under the stream,
    then set the colander on the counter next to the stove to prepare for steaming.

    Doc-canonical count = 4 (`mobile: Yes`):
      PickPlaceCounterToCounter [pick]  -> pick up the colander
      RinseVegetables [wait]            -> rinse the vegetables under the running water
      NavigateKitchen [navigate]        -> carry the colander to the counter by the stove
      PickPlaceCounterToCounter [place] -> place the colander on the counter by the stove
    plus a terminal done. The faucet is already on -- there is no TurnOnSink subtask.

    Boundaries: t_pick = the colander's first sustained displacement (the lift); the rinse
    is the hold under the faucet, ending when the base starts driving to the stove
    (carry onset); t_place = the gripper RELEASE.
    """

    ENV_NAME = "PrepareVeggiesForSteaming"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    MIN_RINSE_HOLD = 15   # rinse must run at least this long before a carry can start
    BASE_REST_TH = 0.003  # base speed (per-frame) at/below this == parked (rinsing)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["colander"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_MOVE, SUBTASK_RINSE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        uw = 0.0
        if self.sink is not None:
            try:
                uw = 1.0 if self.sink.check_obj_under_water(env, "colander") else 0.0
            except Exception:
                uw = 0.0
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "under_water": uw,
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
        uw = np.array([h["under_water"] for h in history])
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

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        # The rinse ends when the colander LEAVES the running water (lifted out of the
        # stream) -- that is the ground-truth "rinse" boundary. The base often stays parked
        # under the faucet a bit longer before it drives to the stove, so the base-park end
        # runs the rinse slightly too long; the under-water signal is tighter. Fall back to
        # the longest base-park end only if the colander never registered under water.
        on = np.where(uw > 0)[0]
        if on.size:
            t_carry = int(on[-1]) + 1
        else:
            rest = binst[t_pick: t_placed] < self.BASE_REST_TH
            best_len = 0
            best_end = None
            cur = 0
            for i, r in enumerate(rest):
                if r:
                    cur += 1
                    if cur > best_len:
                        best_len = cur
                        best_end = i + 1
                else:
                    cur = 0
            t_carry = (t_pick + best_end) if best_end is not None else min(t_pick + self.MIN_RINSE_HOLD, t_placed)
        t_carry = int(min(max(t_carry, t_pick + self.MIN_RINSE_HOLD),
                          max(t_placed - self.MIN_PICK_GAP, t_pick + 1)))
        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_placed
        has_carry = (t_arrive > t_carry and base_disp(t_carry, t_arrive) >= self.MIN_NAV_DIST)

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the colander")
        # the rinse ALWAYS ends when the colander leaves the water (t_carry); the carry nav
        # is only added when the base actually drives to the stove. When the stove counter
        # is adjacent (no carry drive) the rinse still ends at t_carry and the short move to
        # the counter is folded into the place -- the rinse must not run to the placement.
        emit(t_carry, SUBTASK_RINSE, STAGE_WAIT, "rinse the vegetables under the running water")
        if has_carry:
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the colander to the counter next to the stove")
        else:
            review.append("no carry drive detected to the stove counter -- verify")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the colander on the counter next to the stove")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if not bool((uw > 0).any()):
            review.append("colander never registered under the water -- verify")
        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_carry": (int(t_carry) if t_carry is not None else None),
            "t_placed": int(t_placed), "under_water_frames": int((uw > 0).sum()),
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE, SUBTASK_RINSE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 colander pick")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 colander place")
        if int(np.sum(names == SUBTASK_RINSE)) != 1:
            issues.append("expected 1 RinseVegetables")
        return issues
