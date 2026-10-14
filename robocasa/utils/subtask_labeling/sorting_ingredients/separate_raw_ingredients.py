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


# Three ingredients sit in the (pre-opened) fridge; a cutting board and a bowl sit on the
# adjacent counter. To avoid cross-contamination, put meat/seafood on the cutting board and
# vegetables in the bowl. Fridge -> counter has no bank name -> PickPlaceFridgeToCounter
# (extension); NavigateKitchen is bank. The robot starts at the fridge (pre-opened -> no
# OpenFridge subtask).
SUBTASK_MOVE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# obj -> which receptacle it belongs in ("board" for meat/seafood, "bowl" for vegetables).
# random_ingredient's destination follows its per-episode type (ep_meta refs).
OBJS = ("meat_ingredient", "vegetable_ingredient", "random_ingredient")


@register_labeler
class SeparateRawIngredientsLabeler(BaseSubtaskLabeler):
    """Labeler for the SeparateRawIngredients composite task.

    Task (mobile): three ingredients (a meat, a vegetable, and one random of either type)
    sit in the pre-opened fridge; a cutting board and a bowl sit on the counter. Put
    meat/seafood on the cutting board and vegetables in the bowl. Robot starts at the fridge.

    Doc-canonical count = 11 (`mobile: Yes`), action-level: three fridge->counter moves,
    each pick + carry nav + place, plus a return-to-fridge nav before each item after the
    first (3*3 + 2 = 11). ORDER by first move; the fridge is pre-opened (no OpenFridge). The
    first item has no leading nav (the robot starts at the fridge).

    Boundaries: t_pick = first sustained displacement re-anchored at the segment start;
    navs off base motion (>=MIN_NAV_DIST); t_place = the gripper RELEASE.
    """

    ENV_NAME = "SeparateRawIngredients"

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
    CARRY_SPEED_TH = 0.004   # object speed (m/frame) at/below this == at rest
    CARRY_REST_SUSTAIN = 8   # frames of rest that mark the end of an incidental drag
    BASE_REST_TH = 0.003     # base speed (per-frame) at/below this == parked
    ARRIVE_SUSTAIN = 8       # frames of base-park that mark arrival back at the fridge
    GRASP_TH = 0.07          # grip->object dist below this == grasped (vs a bump while
                             # grabbing a neighbour, where the gripper is not on the object)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        random_type = ep_meta.get("refs", {}).get("random_ingredient_type", "meat")
        self.dest = {
            "meat_ingredient": "board",
            "vegetable_ingredient": "bowl",
            "random_ingredient": "board" if random_type == "meat" else "bowl",
        }
        self.noun = {}
        self.base = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "ingredient"
            self.noun[o] = f"the {self.base[o]}"
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
        neighbour (which would otherwise collapse the pick onto the previous
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

    def _recep(self, o):
        return "the cutting board" if self.dest[o] == "board" else "the bowl"

    def _place_prep(self, o):
        return "on the cutting board" if self.dest[o] == "board" else "in the bowl"

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
            gdo = np.linalg.norm(grip - opos[o], axis=1)
            # order by the first displacement WHILE GRASPED (a real carry). A bump from
            # grabbing a neighbour moves the object but the gripper is not on it (gd high),
            # so it must not count -- else the bumped item looks manipulated first.
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gdo < self.GRASP_TH), 0, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        # order-aware display nouns: when 2+ ingredients share a base name (e.g. two
        # steaks), the first handled is "a steak", the next "the other steak" -- not "the
        # steak" for both.
        from collections import Counter
        counts = Counter(self.base[o] for o in order)
        seen = {}
        disp_noun = {}
        for o in order:
            b = self.base[o]
            if counts[b] <= 1:
                disp_noun[o] = f"the {b}"
            else:
                seen[b] = seen.get(b, 0) + 1
                art = "an" if b[:1].lower() in "aeiou" else "a"
                disp_noun[o] = f"{art} {b}" if seen[b] == 1 else f"the other {b}"

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
        for k, o in enumerate(order):
            nn = disp_noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            # pick anchored on the first GRASPED displacement, so a bump from grabbing a
            # neighbour doesn't become the pick.
            t_pick0 = self._sustained_start((dsp >= self.MOVE_TOL) & (gd < self.GRASP_TH),
                                            cursor, self.MOVE_SUSTAIN)
            if t_pick0 is None:
                t_pick0 = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick0 is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick0 = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick0 = max(t_pick0, cursor)
            hold_gd = float(np.median(gd[t_pick0: min(t_pick0 + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick0 + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # pick = first-move (the grab); only when that COLLAPSES (a bumped same-type
            # neighbour dragged at/before the cursor so its first move lands on the previous
            # segment's end) fall back to carry-start + floor. Using carry-start
            # unconditionally pushed the pick onto the settle-before-release, swallowing the
            # carry nav.
            spd = np.concatenate([np.linalg.norm(np.diff(opos[o], axis=0), axis=1), [0.0]])
            if t_pick0 >= cursor + self.MIN_PICK_GAP:
                t_pick = t_pick0
            else:
                t_pick = self._carry_start(spd, t_placed, cursor, self.CARRY_SPEED_TH, self.CARRY_REST_SUSTAIN)
                t_pick = max(t_pick, cursor + self.MIN_PICK_GAP)  # floor: no zero-length pick
            t_pick = int(min(max(t_pick, cursor), max(t_placed - self.MIN_PICK_GAP, cursor + 1)))
            # return-to-fridge nav before each item after the first. It ends when the robot
            # FIRST arrives back at the fridge (the base parks), NOT at the last base twitch
            # before the grab -- the robot often reaches/repositions at the fridge before
            # grabbing, and that belongs to the pick (else a late reposition drags the nav
            # all the way to the grab).
            if k >= 1:
                drive0 = self._sustained_start(binst[cursor: t_pick] > self.BASE_MOVE_TH, 0, self.MOVE_SUSTAIN)
                if drive0 is not None:
                    arrive = self._sustained_start(binst[cursor + drive0: t_pick] < self.BASE_REST_TH,
                                                   0, self.ARRIVE_SUSTAIN)
                    if arrive is not None:
                        t_appr_end = cursor + drive0 + arrive
                    else:
                        appr = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                        t_appr_end = cursor + int(appr[-1]) + 1 if appr.size else cursor
                else:
                    t_appr_end = cursor
                t_appr_end = min(t_appr_end, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_appr_end > cursor and base_disp(cursor, t_appr_end) >= self.MIN_NAV_DIST:
                    emit(t_appr_end, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the fridge")
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the fridge")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {nn} to {self._recep(o)}")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} {self._place_prep(o)}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != len(self.names):
            review.append(f"only {n_moved}/{len(self.names)} ingredients moved")
        result = self.build_result(T, segments, debug={
            "order": order, "dest": self.dest, "n_moved": n_moved, "n_segments": len(segments),
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
        n = len(self.names)
        picks = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK)))
        places = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE)))
        if picks != n:
            issues.append(f"expected {n} picks, found {picks}")
        if places != n:
            issues.append(f"expected {n} places, found {places}")
        return issues
