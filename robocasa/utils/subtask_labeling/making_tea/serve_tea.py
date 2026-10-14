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

# The cup of tea is picked from the (open) microwave, carried across the kitchen to the
# dining table, and set on the saucer. Microwave -> dining table has no bank name ->
# PickPlaceMicrowaveToDiningTable (extension); NavigateKitchen is bank. The robot starts
# at the microwave.
SUBTASK_MOVE = "PickPlaceMicrowaveToDiningTable"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class ServeTeaLabeler(BaseSubtaskLabeler):
    """Labeler for the ServeTea composite task.

    Task (mobile): a cup of tea sits in the (open) microwave; a saucer sits on the
    dining table across the kitchen. Pick the cup, navigate to the dining table, and
    place it on the saucer. The robot starts at the microwave.

    Doc-canonical decomposition (3 high-level subtasks: pick, nav, place). Emitted:
      PickPlaceMicrowaveToDiningTable [pick]  -> pick up the cup of tea from the microwave
      NavigateKitchen                         -> carry the cup of tea to the dining table
      PickPlaceMicrowaveToDiningTable [place] -> place the cup of tea on the saucer
    plus a terminal done.

    Detection: t_pick = the cup's first sustained departure from its microwave rest; the
    nav/place split is the BASE STOP (the base plateaus at the dining table, then the
    cup is lowered onto the saucer -- same as DeliverBrewedCoffee); t_place = the gripper
    RELEASE.
    """

    ENV_NAME = "ServeTea"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.004
    BASE_SMOOTH_W = 7
    MIN_NAV_DIST = 0.40
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cup_body_id = env.obj_body_id["teacup"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        base_xy = (env.sim.data.body_xpos[self.base_body_id][:2].copy()
                   if self.base_body_id is not None else np.array([np.nan, np.nan]))
        return {
            "cup": env.sim.data.body_xpos[self.cup_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
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
        cup = np.array([h["cup"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.BASE_SMOOTH_W
            bspeed = np.convolve(inst, np.ones(W) / W, mode="same")
            bspeed = np.concatenate([bspeed, bspeed[-1:]])
        else:
            bspeed = np.zeros(T)
            base_xy = None

        rest = np.median(cup[: self.REST_W], axis=0)
        disp = np.linalg.norm(cup - rest, axis=1)
        gd = np.linalg.norm(grip - cup, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = min(1, T - 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_place = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        # nav/place split = the BASE STOP: last frame the base is still driving before
        # the cup is lowered onto the saucer.
        t_arrive = t_pick
        if base_xy is not None:
            moving = np.where(bspeed[t_pick: t_place] > self.BASE_MOVE_TH)[0]
            if moving.size:
                cand = min(t_pick + int(moving[-1]) + 1, t_place)
                disp_nav = float(np.linalg.norm(base_xy[min(cand, T - 1)] - base_xy[t_pick]))
                if disp_nav >= self.MIN_NAV_DIST:
                    t_arrive = cand

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the cup of tea from the microwave")
        if t_arrive > cursor:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the cup of tea to the dining table")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, "place the cup of tea on the saucer")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        return self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_arrive": int(t_arrive), "t_place": int(t_place),
            "n_segments": len(segments),
        })

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 pick stage")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 place stage")
        return issues
