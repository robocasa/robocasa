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


# Gather four empty soda cans scattered around the kitchen and cluster them on the counter next
# to the stove. Each can is a nav-interrupted move (drive to the can, grab it, carry it to the
# stove, set it down in the cluster). Counter -> counter has no bank name ->
# PickPlaceCounterToCounter (extension); NavigateKitchen is a bank atomic.
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

CANS = ("can1", "can2", "can3", "can4")


@register_labeler
class RecycleSodaCansLabeler(BaseSubtaskLabeler):
    """Labeler for the RecycleSodaCans composite task.

    Task (mobile): four empty soda cans are scattered on counters around the kitchen (robot
    starts at one of them). Gather all four and cluster them on the counter next to the stove.

    Doc-canonical count = 9 (`mobile: Yes`): the four cans' nav-interrupted moves. Each can (in
    first-motion order) is a round trip:
      (NavigateKitchen to the can) -> can[pick] -> NavigateKitchen(carry to the stove) -> can[place]
    The first can has no lead-in nav (the robot starts by it; k>=1 gate). All cans are identical,
    so the lang is uniform and the segment index distinguishes them.

    Boundaries (MakeBananaMilkshake machinery): t_pick = first sustained displacement from a rest
    re-anchored at the segment start; nav OUT = base drive cursor->grasp (k>=1); nav BACK = base
    drive during the carry, ending at the last substantial-motion frame; place = gripper RELEASE.
    """

    ENV_NAME = "RecycleSodaCans"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.20
    MIN_PICK_GAP = 12    # a nav never runs to within this of the grasp/release it precedes, so
                         # the pick/place it bounds keeps a visible window (the base often drives
                         # right up to a can, which would otherwise collapse the pick).
    ARRIVE_RADIUS = 0.20 # base within this of its parked destination pose == in the vicinity
    ARRIVE_STOP_TH = 0.006  # ...and per-frame base speed below this == settled. A nav ends when
                         # the base is BOTH near its park pose AND stopped, so it lands at the real
                         # arrival, not the creep during a long arm-reach that follows.
    ARRIVE_SUSTAIN = 12
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [c for c in CANS if c in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # include YAW as well as forward/side: the base often translates to a can, then rotates
        # in place to face it -- a pure-position arrival would end the nav during that rotation.
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"):
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

        def arrival(start, end, park):
            # first frame in [start, end) the base is BOTH within ARRIVE_RADIUS of `park` (its
            # parked destination pose, incl. yaw) AND stopped, sustained -> the real settle. Falls
            # back to the last substantial-motion frame if the base drives straight into the grab.
            if base_xy is None or end <= start:
                return None
            near = np.linalg.norm(base_xy[start: end] - park, axis=1) < self.ARRIVE_RADIUS
            stopped = binst[start: end] < self.ARRIVE_STOP_TH
            fr = self._sustained_start(near & stopped, 0, self.ARRIVE_SUSTAIN)
            if fr is not None:
                return start + fr
            mv = np.where(binst[start: end] > self.BASE_MOVE_TH)[0]
            return start + int(mv[-1]) + 1 if mv.size else None

        n_moved = 0
        for k, o in enumerate(order):
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            if k >= 1 and base_xy is not None:
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, t_move - self.MIN_PICK_GAP)  # keep the pick a visible window
                nav_out_lang = ("navigate to the last soda can" if k == len(order) - 1
                                else "navigate to the next soda can")
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, nav_out_lang)
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, "pick up the soda can")
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)  # keep the place a visible window
            # the first gathered can joins the one(s) already by the stove ("the other soda can");
            # each later can joins the growing cluster ("the other soda cans", plural).
            other = "soda can" if k == 0 else "soda cans"
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the soda can next to the other {other} by the stove")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the soda can next to the other {other} next to the stove")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        # Cans that spawn already by the stove are the cluster ANCHORS -- the robot gathers only
        # the rest, so the number that MOVE varies (2-4: up to two anchors observed). Fewer than
        # two moved is worth a look.
        if n_moved < 2:
            review.append(f"only {n_moved} cans gathered -- verify")

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
        names = result.subtask_name[firsts]
        # 2-4 cans move; the rest spawn already clustered by the stove (anchors).
        npick = int(np.sum(stages == STAGE_PICK))
        nplace = int(np.sum(stages == STAGE_PLACE))
        if npick != nplace:
            issues.append(f"pick/place mismatch: {npick} picks, {nplace} places")
        if npick not in (2, 3, 4):
            issues.append(f"expected 2-4 can moves, found {npick}")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 8):
            issues.append(f"expected 0-8 NavigateKitchen, found {n_nav}")
        return issues
