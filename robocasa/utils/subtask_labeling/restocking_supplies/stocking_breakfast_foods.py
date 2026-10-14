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


# Each packaged food is moved counter -> its nearby cabinet (bank atomic, splits
# into pick/place). The two foods sit at two different counter/cabinet stations.
SUBTASK_FOOD = "PickPlaceCounterToCabinet"

# NavigateKitchen (bank atomic): drive between the two stations.
SUBTASK_NAV = "NavigateKitchen"
NAV_LANG = "navigate to the other open cabinet"

DONE_LANG = "task complete"


@register_labeler
class StockingBreakfastFoodsLabeler(BaseSubtaskLabeler):
    """Labeler for the StockingBreakfastFoods composite task.

    Task: two packaged foods sit on two different counters, each next to an open
    cabinet (obj1 by cab1, obj2 by cab2). Stock each food into the cabinet closest
    to it. The two stations can be adjacent or across the kitchen.

    Doc-canonical decomposition (3 subtasks): PickPlaceCounterToCabinet ->
    NavigateKitchen -> PickPlaceCounterToCabinet. Each food's counter is right by
    its cabinet, so no navigation interrupts a pick-place; the one nav is the
    drive between the two stations. The robot starts at cab1; if it stocks the
    far food first an extra leading nav appears (the realised count tracks the
    actual base motion).

    Boundaries (per food): t_pick = the food departs its counter rest while held;
    t_place = the food settles in its cabinet with the gripper released. Foods are
    ordered by placement. A NavigateKitchen is carved wherever the base drives at
    least MIN_NAV_DIST (capped so the reach+grasp keeps >= MIN_PICK_FRAMES).
    Validation is structural (two picks + two places; obj1 -> cab1, obj2 -> cab2).
    """

    ENV_NAME = "StockingBreakfastFoods"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65
    MIN_PICK_FRAMES = 20

    FOODS = ["obj1", "obj2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.food_body_ids = [env.obj_body_id[n] for n in self.FOODS]
        self.food_lang = []
        for n in self.FOODS:
            try:
                self.food_lang.append(env.get_obj_lang(n))
            except Exception:
                self.food_lang.append("food")
        # Each food's target cabinet (obj1 -> cab1, obj2 -> cab2), for the
        # structural end check.
        self.cab_xy = []
        for c in ("cab1", "cab2"):
            try:
                self.cab_xy.append(np.array(getattr(env, c).pos)[:2])
            except Exception:
                self.cab_xy.append(None)
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.expected_done_idx = None
        self.extra_subtasks = set()  # PickPlaceCounterToCabinet is a bank atomic

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "foods": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.food_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, food, grip, T):
        final = np.median(food[-self.REST_W:], axis=0)
        settled = np.linalg.norm(food - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(food - grip, axis=1) > self.RELEASE_TH
        ok = settled & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(settled)[0]
        return int(w[0]) if w.size else T - 1

    def _depart_time(self, food, grip, t_release):
        start = np.median(food[: self.REST_W], axis=0)
        moved = np.linalg.norm(food - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(food - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    def _nav_end(self, base_speed, base_xy, a, b):
        if b - a < 2 or base_xy is None:
            return a
        seg = base_speed[a:b]
        moving = np.where(seg > self.NAV_MOVING_TH)[0]
        if not moving.size:
            return a
        nav_end = min(a + int(moving[-1]) + 1, b)
        disp = float(np.linalg.norm(base_xy[min(nav_end, b - 1)] - base_xy[a]))
        return nav_end if disp >= self.MIN_NAV_DIST else a

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        foods = np.array([h["foods"] for h in history])  # (T, 2, 3)
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        has_base = np.isfinite(base_xy).all()

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        ops = []
        for i in range(len(self.FOODS)):
            food = foods[:, i, :]
            t_rel = self._release_time(food, grip, T)
            t_dep = self._depart_time(food, grip, t_rel)
            ops.append(dict(idx=i, lang=self.food_lang[i], dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang, cap=None):
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if cap is not None:
                nav_end = min(nav_end, max(cursor, cap))
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        for n, op in enumerate(order):
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            # Drive to this food's station (only if the base actually moves there;
            # capped so the reach+grasp keeps >= MIN_PICK_FRAMES).
            maybe_nav(t_dep, NAV_LANG, cap=t_dep - self.MIN_PICK_FRAMES)
            emit(t_dep, SUBTASK_FOOD, STAGE_PICK,
                 f"pick up the {op['lang']} from the counter")
            emit(t_rel, SUBTASK_FOOD, STAGE_PLACE,
                 f"place the {op['lang']} in the cabinet")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        # End check: obj1 near cab1, obj2 near cab2.
        ends_ok = True
        for i in range(len(self.FOODS)):
            if self.cab_xy[i] is None:
                continue
            fin = np.median(foods[-self.REST_W:, i, :], axis=0)[:2]
            if np.linalg.norm(fin - self.cab_xy[i]) > 0.6:
                ends_ok = False
        debug = {
            "order": np.array([o["idx"] for o in order]),
            "ends_ok": int(ends_ok),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        if result.debug.get("ends_ok") == 0:
            issues.append("a food did not end in its cabinet")
        return issues
