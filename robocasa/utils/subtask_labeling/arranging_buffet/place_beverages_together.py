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


# Each drink is moved from the counter to the dining counter, split across a
# navigation into pick / NavigateKitchen / place. No exact bank name covers
# counter -> dining-counter, so the PickPlace name is a declared extension; the
# name is unchanged across the pick/place stages.
SUBTASK_DRINK = "PickPlaceCounterToCounter"

# NavigateKitchen (bank atomic): carrying a drink to the dining counter, and the
# drive back to the counter for the next drink.
SUBTASK_NAV = "NavigateKitchen"
# The carry nav (holding a drink, heading to the cluster) names the destination
# ("navigate to the dining counter"). The return nav (after placing, going back
# to fetch the next drink) names that next drink ("navigate back to the counter
# with the {drink}").
NAV_TO_DINING_LANG = "navigate to the dining counter"

DONE_LANG = "task complete"


@register_labeler
class PlaceBeveragesTogetherLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceBeveragesTogether composite task.

    Task: move three drinks (an alcohol, a juice, a bottled water) from the
    counter to the dining counter and set them down in a tight cluster, one round
    trip each. A distractor tong/cup set on the dining counter is ignored.

    Doc-canonical decomposition (11 subtasks): for each of the three drinks, in
    performed order, PickPlaceCounterToCounter [pick] -> NavigateKitchen ->
    [place], with a NavigateKitchen back to the counter before each drink after
    the first. That is 3*(pick + nav + place)=9 + 2 returns = 11, plus a terminal
    done (12 segments).

    No door to open (the drinks sit on an open counter). The order the drinks are
    moved is auto-detected (varies). The first drink placed establishes the
    cluster ("place ... on the dining counter"); later drinks join it ("place ...
    next to the other drinks"). NavigateKitchen is adaptive (MIN_NAV_DIST); the
    counter<->dining-counter carries are long traversals. Validation is structural
    (three picks + three places).
    """

    ENV_NAME = "PlaceBeveragesTogether"

    DEPART_TOL = 0.10
    GRASP_TH = 0.08
    # Distance to a drink's final pose within which it counts as placed.
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65
    # Minimum frames reserved for the reach+grasp at the end of a pick phase.
    # The drinks sit on an open counter, so the robot often drives straight into
    # the grasp (no distinct "reach in" like a fridge); without this the leading
    # nav-back would extend to the grasp and leave a zero-length pick.
    MIN_PICK_FRAMES = 20

    DRINKS = ["alcohol", "juice", "bottled_water"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.drink_body_ids = [env.obj_body_id[n] for n in self.DRINKS]
        self.drink_lang = []
        for n in self.DRINKS:
            try:
                self.drink_lang.append(env.get_obj_lang(n))
            except Exception:
                self.drink_lang.append("drink")
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_DRINK}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "drinks": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.drink_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, drink, grip, T):
        final = np.median(drink[-self.REST_W:], axis=0)
        settled = np.linalg.norm(drink - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(drink - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, drink, grip, t_release):
        start = np.median(drink[: self.REST_W], axis=0)
        moved = np.linalg.norm(drink - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(drink - grip, axis=1) < self.GRASP_TH
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
        drinks = np.array([h["drinks"] for h in history])  # (T, 3, 3)
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
        for i in range(len(self.DRINKS)):
            drink = drinks[:, i, :]
            t_rel = self._release_time(drink, grip, T)
            t_dep = self._depart_time(drink, grip, t_rel)
            ops.append(dict(idx=i, lang=self.drink_lang[i], dep=t_dep, rel=t_rel))
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

        first_lang = order[0]["lang"]
        for n, op in enumerate(order):
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            pick_lang = f"pick up the {op['lang']} from the counter"
            if n == 0:
                place_lang = f"place the {op['lang']} on the dining counter"
            else:
                # Second drink names the first explicitly; later drinks join the
                # established cluster generically.
                if n == 1:
                    place_lang = f"place the {op['lang']} next to the {first_lang}"
                else:
                    place_lang = f"place the {op['lang']} next to the other drinks"
                # Leading drive back to the counter for the next drink -- names
                # that drink; capped so the reach+grasp (pick) keeps at least
                # MIN_PICK_FRAMES.
                maybe_nav(
                    t_dep,
                    f"navigate back to the counter with the {op['lang']}",
                    cap=t_dep - self.MIN_PICK_FRAMES,
                )
            emit(t_dep, SUBTASK_DRINK, STAGE_PICK, pick_lang)
            # Carry nav names the destination (the dining-counter cluster).
            maybe_nav(t_rel, NAV_TO_DINING_LANG)
            emit(t_rel, SUBTASK_DRINK, STAGE_PLACE, place_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": np.array([o["idx"] for o in order]),
            "rels": np.array([o["rel"] for o in order]),
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
        if n_pick != 3:
            issues.append(f"expected 3 pick stages, found {n_pick}")
        if n_place != 3:
            issues.append(f"expected 3 place stages, found {n_place}")
        return issues
