import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A snack bar sits in an open drawer; a bowl sits on the counter. Put the bar in the bowl,
# then close the drawer. Drawer -> counter(bowl) is the bank atomic PickPlaceDrawerToCounter;
# CloseDrawer is a bank atomic. Non-mobile (drawer and bowl are right in front).
SUBTASK_MOVE = "PickPlaceDrawerToCounter"
SUBTASK_CLOSE = "CloseDrawer"

DONE_LANG = "task complete"


@register_labeler
class SnackSortingLabeler(BaseSubtaskLabeler):
    """Labeler for the SnackSorting composite task.

    Task (non-mobile): a snack bar sits in an open drawer; a bowl sits on the adjacent
    counter. Move the bar into the bowl, then close the drawer.

    Doc-canonical count = 2 (`mobile: No`):
      PickPlaceDrawerToCounter [pick]/[place] -> bar from the drawer into the bowl
      CloseDrawer [execute]                   -> close the drawer
    plus a terminal done.

    Boundaries: t_pick = the bar's first sustained displacement; t_place = the gripper
    RELEASE; CloseDrawer runs from the drawer's closing onset (after the place) until latched.
    """

    ENV_NAME = "SnackSorting"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    CLOSE_FRAC = 0.30     # drawer below this fraction of its open peak == closed
    CLOSE_SUSTAIN = 5
    DRAWER_MIN_PEAK = 0.20
    GRASP_TH = 0.09       # grip->bar dist below this == grasped (anchors the pick past the drawer-nudge)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["bar"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        dr = getattr(env, "drawer", None)
        jn = [n for n in env.sim.model.joint_names
              if n and dr is not None and dr.name in n and ("drawer" in n.lower() or "slide" in n.lower())]
        if not jn:
            jn = [n for n in env.sim.model.joint_names
                  if n and ("drawer" in n.lower() or "slide" in n.lower())]
        self.door_adrs, self.door_scales = [], []
        for j in jn:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "drawer": float(max((abs(float(env.sim.data.qpos[a])) / s
                                 for a, s in zip(self.door_adrs, self.door_scales)), default=0.0)),
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
        drawer = np.array([h["drawer"] for h in history])

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        gd = np.linalg.norm(grip - obj, axis=1)
        # the bar sits in an already-open drawer; the robot often pulls the drawer a little
        # further first, which slides the bar (disp rises) WHILE the gripper is still
        # approaching (gd large). That drawer-nudge is NOT the pick. Anchor the pick on the
        # grasp: find when the gripper closes on the bar (gd < GRASP_TH), then take the first
        # displacement from the bar's position at that moment -- when it is lifted out and
        # carried. ##demo_1 (drawer-nudge made the pick end ~8s early)
        t_grasp = self._sustained_start(gd < self.GRASP_TH, 0, self.MOVE_SUSTAIN)
        t_pick = None
        if t_grasp is not None:
            ref = np.median(obj[max(t_grasp - self.REST_W, 0): t_grasp + 1], axis=0)
            disp2 = np.linalg.norm(obj - ref, axis=1)
            t_pick = self._sustained_start(disp2 >= self.MOVE_TOL, t_grasp, self.MOVE_SUSTAIN)
        if t_pick is None:  # no grasp (or no post-grasp move) detected -> plain first move
            t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        peak = float(drawer.max()) if T else 0.0
        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the bar from the drawer")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the bar in the bowl")

        # CloseDrawer: the drawer slides shut after the bar is placed.
        if peak >= self.DRAWER_MIN_PEAK:
            t_closed = self._sustained_start(drawer[t_placed:] <= self.CLOSE_FRAC * peak, 0, self.CLOSE_SUSTAIN)
            t_closed = (t_placed + t_closed) if t_closed is not None else T
            if t_closed > cursor:
                emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, "close the drawer")
            else:
                review.append("drawer never detected closing -- verify")
        else:
            review.append("drawer never registered open -- verify")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_placed": int(t_placed), "drawer_peak": round(peak, 2),
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
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 bar pick")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 1:
            issues.append("expected 1 bar place")
        if int(np.sum(names == SUBTASK_CLOSE)) not in (0, 1):
            issues.append(f"expected 0-1 CloseDrawer, found {int(np.sum(names == SUBTASK_CLOSE))}")
        return issues
