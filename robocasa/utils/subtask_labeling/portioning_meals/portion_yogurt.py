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


# Portion four yogurt containers from the dining counter onto two plates (two per plate). All on
# the dining counter. Counter -> plate (counter -> counter) has no bank name ->
# PickPlaceCounterToCounter (extension). Fixed base / local.
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

YOGURTS = ("yogurt1", "yogurt2", "yogurt3", "yogurt4")
PLATES = ("plate1", "plate2")


@register_labeler
class PortionYogurtLabeler(BaseSubtaskLabeler):
    """Labeler for the PortionYogurt composite task.

    Task (mobile, but local): four yogurt containers sit on the dining counter; two plates sit
    beside them. Put two yogurts on each plate. The robot starts at the dining counter.

    Doc-canonical count = 4 (one move per yogurt). Each yogurt, in first-motion order:
      PickPlaceCounterToCounter [pick]  -> pick up {a|another} yogurt container from the counter
      (NavigateKitchen [navigate])      -> carry the yogurt container to {a plate|the other plate}  (adaptive)
      PickPlaceCounterToCounter [place] -> place the yogurt container on {a plate|the other plate}
    plus a terminal done. The destination plate is resolved from which one the yogurt ends up
    nearest to; the first used is "a plate", the other "the other plate".

    Boundaries: t_pick = first sustained displacement from a rest re-anchored at the segment start;
    a carry nav is emitted only if the base actually travels; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PortionYogurt"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in YOGURTS if o in env.obj_body_id]
        self.plates = [p for p in PLATES if p in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names + self.plates}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.names + self.plates:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

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
        opos = {o: np.array([h[o] for h in history]) for o in self.names + self.plates}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0
        seen = 0
        plate_phrase = {}

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            pick_art = "a" if seen == 0 else "another"
            seen += 1

            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            if self.plates:
                ypos = opos[o][min(t_placed, T - 1)]
                nearest = min(self.plates, key=lambda p: np.linalg.norm(ypos - opos[p][min(t_placed, T - 1)]))
                if nearest not in plate_phrase:
                    plate_phrase[nearest] = "a plate" if len(plate_phrase) == 0 else "the other plate"
                where = plate_phrase[nearest]
            else:
                where = "a plate"

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {pick_art} yogurt container from the counter")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                        if base_xy is not None and t_arrive > t_pick else 0.0)
            if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the yogurt container to {where}")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the yogurt container on {where}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 4:
            review.append(f"only {n_moved}/4 yogurts moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 4:
            issues.append(f"expected 4 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 4:
            issues.append(f"expected 4 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
