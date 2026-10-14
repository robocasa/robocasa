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


# Each drink is carried from the island to the (already-open) cabinet. Island->cabinet
# is a bank atomic (PickPlaceCounterToCabinet -- the island is a counter); the move is
# nav-interrupted, NavigateKitchen is bank.
SUBTASK_MOVE = "PickPlaceCounterToCabinet"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class DrinkwareConsolidationLabeler(BaseSubtaskLabeler):
    """Labeler for the DrinkwareConsolidation composite task.

    Task: 1-3 drinks sit on the island; a cabinet is ALREADY OPEN across from it (the
    robot starts at the island). Carry each drink to the cabinet.

    Doc-canonical count = 3 (`mobile: Yes`): each drink is a nav-interrupted
    island->cabinet move, and the robot drives BACK to the island for the next. Per
    drink: `[nav back to island] → pick → nav(carry to cabinet) → place`; the FIRST
    drink has no leading nav (the robot starts at the island). The count VARIES 1-3
    (`num_drinkware`), so expected_done_idx is set per demo. Each nav is ADAPTIVE. The
    ORDER varies (by first move).
    """

    ENV_NAME = "DrinkwareConsolidation"

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
        try:
            self.num = int(ep_meta["refs"]["num_drinkware"])
        except Exception:
            self.num = sum(1 for k in env.obj_body_id if k.startswith("obj_"))
        self.objs = tuple(f"obj_{i}" for i in range(self.num))
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.objs}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.objs:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "drink"

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.objs:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.objs}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.objs}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.objs}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.objs}

        first_move = {}
        for o in self.objs:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(self.objs, key=lambda o: first_move[o])
        words = [self.obj_lang[o] for o in order if first_move[o] < T]
        identical = len(set(words)) <= 1
        moved = [o for o in order if first_move[o] < T]
        n = len(moved)

        def noun_of(k, o):
            word = self.obj_lang[o]
            if not identical:
                return f"the {word}"
            if n == 1:
                return f"the {word}"
            if k == 0:
                return f"a {word}"
            if k == n - 1:
                return f"the last {word}" if n >= 3 else f"the other {word}"
            return f"another {word}"

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

        kk = 0
        for o in order:
            if first_move[o] >= T:
                continue
            noun = noun_of(kk, o)
            t_pick = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav back to the island -- only for drinks after the first
            if kk >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the island")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the island")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the open cabinet")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} in the open cabinet")
            kk += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n != self.num:
            review.append(f"only {n}/{self.num} drinks moved")

        result = self.build_result(T, segments, debug={
            "order": ",".join(order), "n_moved": n, "n_segments": len(segments),
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

        n_pick, n_place = count(SUBTASK_MOVE, STAGE_PICK), count(SUBTASK_MOVE, STAGE_PLACE)
        if n_pick != n_place:
            issues.append(f"drink picks ({n_pick}) != places ({n_place})")
        if not (1 <= n_pick <= self.num):
            issues.append(f"expected 1-{self.num} drink moves, found {n_pick}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 2 * self.num):
            issues.append(f"expected 0-{2*self.num} NavigateKitchen, found {n_nav}")
        return issues
