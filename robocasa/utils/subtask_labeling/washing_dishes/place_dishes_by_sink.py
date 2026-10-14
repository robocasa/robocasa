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
from robocasa.utils.subtask_labeling.washing_dishes._fixtures import nearest_landmark


# A cup and a bowl sit on the counter; move both to the counter next to the sink. Counter ->
# counter has no bank name -> PickPlaceCounterToCounter (extension); NavigateKitchen is bank.
# Mobile (the two counters can be apart; the carry/return navs are conditional).
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

OBJS = ("dish0", "dish1")


@register_labeler
class PlaceDishesBySinkLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceDishesBySink composite task.

    Task (mobile): a cup and a bowl are on the counter; relocate both to the counter next to
    the sink. Doc-canonical count = 7 (action-level: two moves, each pick + carry nav + place,
    plus a return nav before the second). Order by first grasped move; distinct types read
    "the cup"/"the bowl", same-type reads "a X"/"the other X".

    pick = first displacement while grasped (cursor-relative rest); carry/return navs off base
    motion, emitted only when the base actually drives MIN_NAV_DIST; place = gripper RELEASE.
    """

    ENV_NAME = "PlaceDishesBySink"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.07
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base = {}
        self.src_landmark = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "dish"
            # the appliance next to each dish's source counter (for the return-nav landmark)
            xy = env.sim.data.body_xpos[self.obj_body_ids[o]][:2]
            self.src_landmark[o] = nearest_landmark(env, xy)
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
    def _last_sustained_end(mask, sustain):
        last_end, run = None, 0
        for i, v in enumerate(mask):
            if v:
                run += 1
            else:
                if run >= sustain:
                    last_end = i
                run = 0
        if run >= sustain:
            last_end = len(mask)
        return last_end

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
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), 0, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        from collections import Counter
        counts = Counter(self.base[o] for o in order)
        seen, noun = {}, {}
        for o in order:
            w = self.base[o]
            if counts[w] <= 1:
                noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for k, o in enumerate(order):
            nn = noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # return-to-source nav before a later dish (ends at the final sustained drive)
            if k >= 1:
                end = self._last_sustained_end(binst[cursor: t_pick] > self.BASE_MOVE_TH, self.MOVE_SUSTAIN)
                t_appr = cursor + end if end is not None else cursor
                t_appr = min(t_appr, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_appr > cursor and base_disp(cursor, t_appr) >= self.MIN_NAV_DIST:
                    lm = self.src_landmark.get(o)
                    by = f" by the {lm}" if lm else ""
                    emit(t_appr, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate to {nn} on the counter{by}")
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the counter")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {nn} to the counter next to the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the counter next to the sink")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 dishes moved")
        result = self.build_result(T, segments, debug={"order": order, "n_moved": n_moved,
                                                       "n_segments": len(segments)})
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
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 2:
            issues.append("expected 2 dish picks")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 2:
            issues.append("expected 2 dish places")
        return issues
