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


# Each meat (chicken drumstick, shrimp, steak) is carried from the (already-open)
# fridge to the serving tray on the dining counter. The fridge->counter move has no
# exact bank name so it is an extra_subtask; NavigateKitchen is bank.
SUBTASK_MOVE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

ITEMS = ("chicken", "shrimp", "beef")


@register_labeler
class DisplayMeatVarietyLabeler(BaseSubtaskLabeler):
    """Labeler for the DisplayMeatVariety composite task.

    Task: three meats (chicken, shrimp, beef) sit in an ALREADY-OPEN fridge (robot
    starts at the fridge); a serving tray sits on the dining counter. Carry all three
    to the tray.

    Doc-canonical count = 11 (`mobile: Yes`): each meat is a nav-interrupted fridge->
    counter move, and the robot drives BACK to the fridge between them. Per meat:
    `[nav back to fridge] -> pick -> nav(carry to counter) -> place`; the FIRST meat
    has no leading nav. 3 meats -> 11 subtasks + done. Navs are ADAPTIVE; ORDER varies.
    """

    ENV_NAME = "DisplayMeatVariety"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20
    CARRY_MIN = 0.50  # displacement (m) that marks a COMMITTED carry vs a brief touch

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in ITEMS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in ITEMS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = o

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
        for o in ITEMS:
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
        opos = {o: np.array([h[o] for h in history]) for o in ITEMS}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in ITEMS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in ITEMS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in ITEMS}
        # Order meats by their COMMITTED carry, not first contact: a brief touch/bump
        # trips a MOVE_TOL displacement early but never carries the meat across the
        # kitchen. t_far = first time the meat is displaced CARRY_MIN from its fridge
        # rest (a real fridge->counter carry reaches several metres; a touch stays well
        # under 0.5m). Ordering by t_far puts the real trips in sequence, and the
        # per-object loop's cursor then advances past any earlier touch.
        t_far = {}
        for o in ITEMS:
            fm = self._sustained_start(disp[o] >= self.CARRY_MIN, 0, self.MOVE_SUSTAIN)
            t_far[o] = fm if fm is not None else T
        order = sorted(ITEMS, key=lambda o: t_far[o])

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

        # ONE item per trip: the robot drives back to the fridge between items and
        # never carries two across the kitchen, so each meat is its own pick/nav/place.
        n_moved = 0
        for o in order:
            if t_far[o] >= T:
                continue
            word = self.obj_lang[o]
            # re-anchor this meat's rest to its SETTLED pose just BEFORE its turn (the
            # window ending at cursor) -- removing earlier meats shifts the ones still
            # in the fridge, so the global-rest displacement reads as already-moved and
            # collapses the pick.
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the fridge")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the fridge")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the dining counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} on the serving tray")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 meats moved")

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
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_MOVE, STAGE_PICK) != 3:
            issues.append(f"expected 3 meat picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 3:
            issues.append(f"expected 3 meat places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 6):
            issues.append(f"expected 0-6 NavigateKitchen, found {n_nav}")
        return issues
