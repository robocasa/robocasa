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


# Move the two mugs and the bowl off the counter by the sink to somewhere far away,
# to clear the sink for cleaning. Each dish is carried away and the robot returns for
# the next, so each move is nav-interrupted. Source and destination are both the
# counter, for which the bank has no name -> PickPlaceCounterToCounter is a declared
# extension; NavigateKitchen is a bank atomic.
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

DISHES = ("bowl", "mug1", "mug2")


@register_labeler
class ClearSinkAreaLabeler(BaseSubtaskLabeler):
    """Labeler for the ClearSinkArea composite task.

    Task: two mugs and a bowl sit on the counter next to the sink (a food item sits
    in the sink and stays put). Carry each dish far away from the sink to clear the
    area for cleaning. The robot starts at the sink.

    Doc-canonical count = 11 (`mobile: Yes`): each of the three dishes is a
    nav-interrupted move, and the robot drives BACK to the sink between them. Full
    decomposition per dish: [nav back to sink] -> pick -> nav(carry away) -> place;
    the first dish has no leading nav (robot starts at the sink):
        pick, nav, place, nav, pick, nav, place, nav, pick, nav, place = 11 subtasks
        + done (expected_done_idx = 11). Each nav is ADAPTIVE (gated on MIN_NAV_DIST).
    The dish ORDER varies (resolved from which moves first).

    Boundaries (ArrangeCuttingFruits base-stop convention): per dish a leading
    back-to-sink nav (base drive before the grasp), t_pick (first sustained
    displacement from rest), a carry-away nav (base drive while held, to the base
    stop), and the place ending at the gripper RELEASE.
    """

    ENV_NAME = "ClearSinkArea"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in DISHES}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in DISHES:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "bowl" if o == "bowl" else "mug"

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
        for o in DISHES:
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
        opos = {o: np.array([h[o] for h in history]) for o in DISHES}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in DISHES}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in DISHES}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in DISHES}

        first_move = {}
        for o in DISHES:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(DISHES, key=lambda o: first_move[o])

        # naming: the two mugs are indistinguishable -> "a mug" / "the other mug" by
        # move order; the bowl is "the bowl".
        moved_mugs = [o for o in order if o in ("mug1", "mug2") and first_move[o] < T]
        second_mug = moved_mugs[1] if len(moved_mugs) == 2 else None

        def noun_of(o):
            if o == "bowl":
                return f"the {self.obj_lang[o]}"
            word = self.obj_lang[o]  # "mug"
            return f"the other {word}" if o == second_mug else f"a {word}"

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for k, o in enumerate(order):
            if first_move[o] >= T:
                continue
            n_moved += 1
            noun = noun_of(o)

            t_pick = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav BACK to the sink -- only for dishes AFTER the first (the
            # return drive after dropping the previous dish far away). The first
            # dish's leading drive is just the robot's initial positioning at the
            # sink (it starts there); that is not a task nav, so it is absorbed into
            # the first pick rather than emitted as its own subtask.
            if k >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_at_sink = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_at_sink > cursor and base_disp(cursor, t_at_sink) >= self.MIN_NAV_DIST:
                    emit(t_at_sink, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the sink")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the counter")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} away from the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"set {noun} down away from the sink")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 dishes moved")

        debug = {
            "order": order,
            "first_move": {o: int(first_move[o]) for o in DISHES},
            "n_moved": n_moved, "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
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

        # The dish count VARIES 1-3: a dish that spawns already far from the sink is
        # left in place (success only requires every dish be far from the sink), so
        # only the dishes that start near it are relocated. Picks must match places
        # and there must be at least one move; the "only N/3 dishes moved" review note
        # routes the reduced-count demos to human confirmation.
        n_pick, n_place = count(SUBTASK_MOVE, STAGE_PICK), count(SUBTASK_MOVE, STAGE_PLACE)
        if n_pick != n_place:
            issues.append(f"dish picks ({n_pick}) != places ({n_place})")
        if not (1 <= n_pick <= 3):
            issues.append(f"expected 1-3 dish moves, found {n_pick}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 6):
            issues.append(f"expected 0-6 NavigateKitchen, found {n_nav}")
        return issues
