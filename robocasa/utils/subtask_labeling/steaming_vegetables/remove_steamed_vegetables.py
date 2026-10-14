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


# Two steamed vegetables sit in a saucepan on a stove burner; a serving plate sits on the
# adjacent counter. Take both vegetables out of the saucepan and put them on the plate.
# Saucepan (on the stove) -> counter is the bank atomic PickPlaceStoveToCounter. The robot
# starts at the stove (local).
SUBTASK_MOVE = "PickPlaceStoveToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

VEGS = ("vegetable1", "vegetable2")


@register_labeler
class RemoveSteamedVegetablesLabeler(BaseSubtaskLabeler):
    """Labeler for the RemoveSteamedVegetables composite task.

    Task (mobile, local): two steamed vegetables sit in a saucepan on a stove burner; a
    serving plate sits on the adjacent counter. Move both vegetables onto the plate.

    Doc-canonical count = 2 (`mobile: Yes`), action-level: one saucepan->plate move per
    vegetable, ORDER by first move; each splits pick / place with an optional carry nav.

    Per vegetable: t_pick = first sustained displacement re-anchored at the segment start
    (the vegetables are grabbed out of the saucepan, often off-center, so displacement --
    not grip distance -- drives the pick); t_place = the gripper RELEASE.
    """

    ENV_NAME = "RemoveSteamedVegetables"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    CARRY_SPEED_TH = 0.004   # object speed (m/frame) at/below this == at rest
    CARRY_REST_SUSTAIN = 8   # frames of rest that mark the end of an incidental drag
    SETTLE_TH = 0.004        # vegetable speed at/below this == settled
    SETTLE_W = 10            # frames of settle that mark a placement
    PLACE_MIN_DISP = 0.10    # vegetable must have been carried this far before a settle counts

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in VEGS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = "the vegetable"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.names:
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

    @staticmethod
    def _carry_start(speed, t_end, lo, th, sustain):
        """Walk back from t_end to the frame where motion resumes after the last
        sustained rest (>= lo). Anchors the pick to the carry that ends at the
        placement, skipping an incidental earlier DRAG of a bumped same-type
        vegetable (which would otherwise collapse the pick onto the previous
        segment's end)."""
        run = 0
        for t in range(int(t_end) - 1, int(lo) - 1, -1):
            if speed[t] < th:
                run += 1
                if run >= sustain:
                    return min(t + run, int(t_end))
            else:
                run = 0
        return int(lo)

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
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

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        n_moved = 0
        for o in order:
            nn = self.noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick0 = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick0 is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick0 = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick0 = max(t_pick0, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick0: min(t_pick0 + self.HOLD_W, T)]))
            spd = np.concatenate([np.linalg.norm(np.diff(opos[o], axis=0), axis=1), [0.0]])
            # place = the vegetable's FIRST sustained settle once it has been carried a real
            # distance. The gripper lingers over the plate (and the veg can be bumped again
            # later), so the object-settle boundary is more reliable than the gripper
            # release; fall back to the release, then to T-1.
            disp_pick = np.linalg.norm(opos[o] - opos[o][t_pick0], axis=1)
            t_placed = None
            run = 0
            for tt in range(min(t_pick0 + self.MIN_CARRY, T - 1), T):
                if spd[tt] < self.SETTLE_TH and disp_pick[tt] >= self.PLACE_MIN_DISP:
                    run += 1
                    if run >= self.SETTLE_W:
                        t_placed = tt - self.SETTLE_W + 1
                        break
                else:
                    run = 0
            if t_placed is None:
                t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                                 t_pick0 + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # pick = first-move; only when that collapses (a bumped same-type vegetable
            # dragged at/before the cursor) fall back to carry-start + floor.
            if t_pick0 >= cursor + self.MIN_PICK_GAP:
                t_pick = t_pick0
            else:
                t_pick = self._carry_start(spd, t_placed, cursor, self.CARRY_SPEED_TH, self.CARRY_REST_SUSTAIN)
                t_pick = max(t_pick, cursor + self.MIN_PICK_GAP)  # floor: no zero-length pick
            t_pick = int(min(max(t_pick, cursor), max(t_placed - self.MIN_PICK_GAP, cursor + 1)))
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"take {nn} out of the saucepan")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {nn} to the plate")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the plate")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 vegetables moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        picks = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK)))
        places = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE)))
        if picks != 2:
            issues.append(f"expected 2 picks, found {picks}")
        if places != 2:
            issues.append(f"expected 2 places, found {places}")
        return issues
