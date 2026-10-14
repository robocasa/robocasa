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


# Each drink is moved counter -> its cabinet, split across a navigation into
# pick / NavigateKitchen / place. No exact bank name covers counter -> cabinet,
# so the PickPlace name is a declared extension; the name is unchanged across the
# pick/place stages.
SUBTASK_DRINK = "PickPlaceCounterToCabinet"

# NavigateKitchen (bank atomic): carrying a drink to its cabinet, and the drive
# back to the counter for the next drink (which names the drink just placed).
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class BeverageSortingLabeler(BaseSubtaskLabeler):
    """Labeler for the BeverageSorting composite task.

    Task: sort four drinks from the counter into two open cabinets -- the two
    alcoholic drinks into one cabinet, the two non-alcoholic drinks into the
    other (one round trip each). Both cabinets start open (no door to open).

    Doc-canonical decomposition (9 subtasks): for each of the four drinks, in
    performed order, PickPlaceCounterToCabinet [pick] -> NavigateKitchen ->
    [place]; navigation is inserted only where the base genuinely drives (the two
    drinks bound for the near cabinet need no nav, the two for the far cabinet
    do, plus a drive back to the counter between them). A realised count is e.g.
    2 plain pick-places + 2*(pick + nav + place) + 1 return = 9, plus a terminal
    done.

    Auto-detected: the order the drinks are sorted (the two types do not have a
    fixed order; ordered by the frame each drink settles in a cabinet) and which
    cabinet each type goes to (the robot's choice -- success allows either
    arrangement). Each drink's lang uses its real name (via env.get_obj_lang).
    The cabinets are not pre-labelled, so the reference is dynamic: the first
    drink of the first type -> "an empty cabinet"; the first drink of the second
    type -> "the empty cabinet" (only one left); the second drink of a type ->
    "the cabinet with the {first drink of that type}". NavigateKitchen is adaptive
    (MIN_NAV_DIST). Validation is structural (four picks + four places; the two
    alcoholic drinks share one cabinet and the two non-alcoholic the other).
    """

    ENV_NAME = "BeverageSorting"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    # Distance to a drink's final pose within which it counts as placed.
    PLACE_TOL = 0.12
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65
    # Minimum frames reserved for the reach+grasp at the end of a pick phase. The
    # drinks sit on an open counter, so the robot often drives straight into the
    # grasp; without this the leading return-nav would extend to the grasp and
    # leave a zero-length pick.
    MIN_PICK_FRAMES = 20

    DRINKS = ["alcohol1", "alcohol2", "non_alcohol1", "non_alcohol2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.drink_body_ids = [env.obj_body_id[n] for n in self.DRINKS]
        self.drink_type = [
            "alcoholic" if n.startswith("alcohol") else "non-alcoholic"
            for n in self.DRINKS
        ]
        self.drink_lang = []
        for n in self.DRINKS:
            try:
                self.drink_lang.append(env.get_obj_lang(n))
            except Exception:
                self.drink_lang.append("drink")
        # These objects are all bottles; if get_obj_lang didn't already say so
        # (e.g. "liquor", "juice"), append "bottle" ("liquor" -> "liquor bottle",
        # "water bottle" left as is).
        self.drink_lang = [
            l if "bottle" in l else f"{l} bottle" for l in self.drink_lang
        ]
        # Cabinet reference positions (physical fixture locations from the model;
        # deterministic per demo). Only used to route each drink to its nearest
        # cabinet -- the lang is dynamic and identity-independent, so a cab1/cab2
        # relabel cannot change the output (cross-surface safe).
        self.cab_xy = None
        try:
            self.cab_xy = np.stack(
                [np.array(env.cab1.pos)[:2], np.array(env.cab2.pos)[:2]]
            )
        except Exception:
            self.cab_xy = None
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
        drinks = np.array([h["drinks"] for h in history])  # (T, 4, 3)
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
            final_xy = np.median(drink[-self.REST_W:], axis=0)[:2]
            if self.cab_xy is not None:
                tgt = int(np.argmin(np.linalg.norm(self.cab_xy - final_xy, axis=1)))
            else:
                tgt = 0 if self.drink_type[i] == "alcoholic" else 1
            t_rel = self._release_time(drink, grip, T)
            t_dep = self._depart_time(drink, grip, t_rel)
            ops.append(dict(
                idx=i, type=self.drink_type[i], lang=self.drink_lang[i],
                tgt=tgt, dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        # "a {drink}" when the two drinks of a type share a name (e.g. two juices);
        # "the {drink}" when they differ.
        type_langs = {}
        for o in ops:
            type_langs.setdefault(o["type"], []).append(o["lang"])
        type_dup = {t: len(set(ls)) == 1 for t, ls in type_langs.items()}

        # The cabinets are not pre-labelled by type: the FIRST drink of a type
        # goes to "an empty cabinet", and the SECOND references the cabinet the
        # first established.
        seen_type = {}  # type -> lang of the first drink of that type placed

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
            art = "a" if type_dup[op["type"]] else "the"
            if op["type"] not in seen_type:
                cab_ref = (
                    "an empty cabinet"
                    if len(seen_type) == 0
                    else "the empty cabinet"
                )
                place_ref = cab_ref
                nav_ref = cab_ref
                seen_type[op["type"]] = op["lang"]
            else:
                # Navigate to the cabinet that already holds the first drink of
                # this type; place the second one next to it. When the two drinks
                # share a name (e.g. two liquor bottles), say "the other ...".
                first = seen_type[op["type"]]
                other = "other " if type_dup[op["type"]] else ""
                nav_ref = f"the cabinet with the {other}{first}"
                place_ref = f"the cabinet next to the {other}{first}"
            pick_lang = f"pick up {art} {op['lang']} from the counter"
            place_lang = f"place the {op['lang']} in {place_ref}"
            nav_cab_lang = f"navigate to {nav_ref}"
            if n > 0:
                # Leading drive back to the counter -- names the drink about to be
                # fetched (what is left); capped so the reach+grasp (pick) keeps
                # >= MIN_PICK_FRAMES.
                back_lang = (
                    f"navigate back to the counter with the {op['lang']}"
                )
                maybe_nav(t_dep, back_lang, cap=t_dep - self.MIN_PICK_FRAMES)
            emit(t_dep, SUBTASK_DRINK, STAGE_PICK, pick_lang)
            maybe_nav(t_rel, nav_cab_lang)
            emit(t_rel, SUBTASK_DRINK, STAGE_PLACE, place_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": np.array([o["idx"] for o in order]),
            "order_types": np.array([o["type"] for o in order]),
            "targets": np.array([o["tgt"] for o in order]),
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
        if n_pick != 4:
            issues.append(f"expected 4 pick stages, found {n_pick}")
        if n_place != 4:
            issues.append(f"expected 4 place stages, found {n_place}")
        # the two alcoholic drinks share one cabinet and the two non-alcoholic
        # the other.
        tgts = result.debug.get("targets")
        types = result.debug.get("order_types")
        if tgts is not None and types is not None:
            at = {int(t) for t, ty in zip(tgts, types) if ty == "alcoholic"}
            nt = {int(t) for t, ty in zip(tgts, types) if ty == "non-alcoholic"}
            if len(at) != 1 or len(nt) != 1 or at == nt:
                issues.append(
                    f"alcoholic/non-alcoholic not cleanly split across cabinets "
                    f"(alcoholic {at}, non-alcoholic {nt})"
                )
        return issues
