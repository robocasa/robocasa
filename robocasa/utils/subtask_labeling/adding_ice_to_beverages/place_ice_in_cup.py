import numpy as np
from robosuite.utils import transform_utils as T

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each ice cube is one pick-and-place from the bowl into the single glass cup
# (both on the dining counter -> no exact bank name; declared bank extension,
# cf. AddIceCubes). Split into pick/place stages.
SUBTASK = "PickPlaceCounterToCounter"
PICK_LANG = "pick up an ice cube from the bowl"
PLACE_LANG = "place the ice cube in the glass cup"

# NavigateKitchen (bank atomic). The bowl and cup sit near the same stool, so the
# robot usually reaches both from one pose; a NavigateKitchen is inserted only if
# it genuinely drives between them. Destination-aware lang.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_CUP_LANG = "navigate to the glass cup"
NAV_TO_BOWL_LANG = "navigate back to the ice bowl"

# Bowl-carry / tilt-pour strategy (same skills as AddIceCubes): lift the whole
# bowl and tilt it over the cup to pour the cubes in. "Pick" (grab the bowl) and
# "Tilt" (pour) are bank-extension skills. A demo may pick one cube by hand and
# pour the rest (mixed).
SUBTASK_BOWL = "Pick"
BOWL_PICK_LANG = "pick up the bowl of ice cubes"
SUBTASK_TILT = "Tilt"
TILT_LANG = "tilt the bowl to pour the ice cubes into the glass cup"

DONE_LANG = "task complete"


@register_labeler
class PlaceIceInCupLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceIceInCup composite task.

    Task: two ice cubes start in one bowl; place both into a single glass cup
    (bowl and cup both on the dining counter). Doc-set count = 2 (two per-cube
    pick-and-places).

    Strategies auto-detected (and combinable in one demo):
      - Per-cube: pick each cube from the bowl, place it in the cup. pick/place
        stages; cubes co-grabbed within MULTIGRAB_WINDOW frames become one pick
        followed by a place per cube.
      - Bowl tilt-pour (same Pick/Tilt skills as AddIceCubes): lift the whole
        bowl and tilt it over the cup. Detected when the bowl's up-axis rotates
        past POUR_TILT_TH while lifted POUR_LIFT_TH off the counter. Labeled
        `Pick` (grab the bowl) -> `Tilt` (pour into the cup); a soft REVIEW note
        is attached.
      - Mixed: a cube placed by hand before the bowl is lifted is labeled
        per-cube; cubes still in the bowl when it is lifted are poured. The split
        is by whether a cube is released before the bowl pick-up.

    A NavigateKitchen segment is carved out wherever the base genuinely drives
    between the bowl and the cup (rare here -- they are near the same stool).
    Validation is structural rather than a fixed segment count.
    """

    ENV_NAME = "PlaceIceInCup"

    # Distance a cube must move from its bowl rest pose to count as departing.
    DEPART_TOL = 0.10
    GRASP_TH = 0.07
    # XY distance to the cup within which a cube counts as delivered.
    CUP_TOL = 0.09
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10
    # Cubes co-grabbed within this many frames = one multi-grab.
    MULTIGRAB_WINDOW = 50

    # --- bowl tilt-pour detection (mirrors AddIceCubes) ---
    POUR_TILT_TH = 45.0
    POUR_LIFT_TH = 0.05
    POUR_MIN_FRAMES = 3

    # --- navigation ---
    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    # Net base displacement a leading drive must reach to be a NavigateKitchen.
    MIN_NAV_DIST = 0.50

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cube_body_ids = [env.obj_body_id["ice_cube1"], env.obj_body_id["ice_cube2"]]
        self.bowl_body_id = env.obj_body_id["ice_bowl"]
        self.cup_body_id = env.obj_body_id["glass_cup"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK, SUBTASK_BOWL, SUBTASK_TILT}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        q = np.array(env.sim.data.body_xquat[self.bowl_body_id])
        bowl_up = T.quat2mat(q[[1, 2, 3, 0]])[:, 2].copy()
        return {
            "cubes": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.cube_body_ids]
            ),
            "cup": env.sim.data.body_xpos[self.cup_body_id].copy(),
            "bowl": env.sim.data.body_xpos[self.bowl_body_id].copy(),
            "bowl_up": bowl_up,
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, cube, grip, cup, T):
        in_cup = np.linalg.norm(cube[:, :2] - cup[:, :2], axis=1) < self.CUP_TOL
        released = np.linalg.norm(cube - grip, axis=1) > self.RELEASE_TH
        ok = in_cup & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(in_cup)[0]
        return int(w[-1]) if w.size else T - 1

    def _depart_time(self, cube, grip, t_release):
        start = np.median(cube[: self.REST_W], axis=0)
        moved = np.linalg.norm(cube - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(cube - grip, axis=1) < self.GRASP_TH
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

    def _group_multigrab(self, ops):
        groups = []
        for o in sorted(ops, key=lambda o: o["dep"]):
            if groups and (o["dep"] - groups[-1][-1]["dep"]) <= self.MULTIGRAB_WINDOW:
                groups[-1].append(o)
            else:
                groups.append([o])
        return groups

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        cubes = np.array([h["cubes"] for h in history])   # (T, 2, 3)
        cup = np.array([h["cup"] for h in history])        # (T, 3)
        bowl = np.array([h["bowl"] for h in history])      # (T, 3)
        bowl_up = np.array([h["bowl_up"] for h in history])
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

        ncubes = cubes.shape[1]
        bowl_rest = np.median(bowl[: self.REST_W, :2], axis=0)

        # Per-cube events (every cube targets the single cup).
        ops = []
        for c in range(ncubes):
            cube = cubes[:, c, :]
            t_rel = self._release_time(cube, grip, cup, T)
            t_dep = self._depart_time(cube, grip, t_rel)
            ops.append(dict(cube=c, dep=t_dep, rel=t_rel))

        # --- bowl tilt-pour detection ---
        bowl_z = bowl[:, 2]
        z0 = float(np.median(bowl_z[: self.REST_W]))
        tilt = np.degrees(np.arccos(np.clip(bowl_up @ bowl_up[0], -1.0, 1.0)))
        lifted = (bowl_z - z0) > self.POUR_LIFT_TH
        pour = (tilt > self.POUR_TILT_TH) & lifted
        episodes = []
        i = 0
        while i < T:
            if pour[i]:
                j = i
                while j < T and pour[j]:
                    j += 1
                if j - i >= self.POUR_MIN_FRAMES:
                    episodes.append([i, j])
                i = j
            else:
                i += 1

        # Bowl lift-off leading into the first pour, and the per-cube/poured split:
        # a cube placed before the bowl is lifted is per-cube; the rest are poured.
        poured_ops, percube_ops = [], list(ops)
        t_bowl_pick = None
        if episodes:
            first_pour = episodes[0][0]
            k = first_pour
            while k > 0 and lifted[k - 1]:
                k -= 1
            t_bowl_pick = k
            percube_ops = [o for o in ops if o["rel"] < t_bowl_pick]
            poured_ops = [o for o in ops if o["rel"] >= t_bowl_pick]
            # Spurious pour (e.g. fiddling with an emptied bowl): no cube poured ->
            # fall back to pure per-cube.
            if not poured_ops:
                episodes = []
                percube_ops = list(ops)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def nav_lang(a, b):
            if base_xy is None:
                return NAV_TO_CUP_LANG
            sb = base_xy[a]
            eb = base_xy[min(b - 1, len(base_xy) - 1)]
            cands = [
                (NAV_TO_BOWL_LANG, bowl_rest),
                (NAV_TO_CUP_LANG, cup[min(b - 1, T - 1), :2]),
            ]
            best = min(
                cands,
                key=lambda c: np.linalg.norm(eb - c[1]) - np.linalg.norm(sb - c[1]),
            )
            return best[0]

        def maybe_nav(end):
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, nav_lang(cursor, nav_end))

        def emit_percube(op_list):
            for grp in sorted(
                self._group_multigrab(op_list), key=lambda g: min(c["rel"] for c in g)
            ):
                pick_end = max(max(c["dep"] for c in grp), cursor)
                maybe_nav(pick_end)
                emit(pick_end, SUBTASK, STAGE_PICK, PICK_LANG)
                for c in sorted(grp, key=lambda c: c["rel"]):
                    maybe_nav(max(c["rel"], cursor))
                    emit(max(c["rel"], cursor), SUBTASK, STAGE_PLACE, PLACE_LANG)

        review = []
        if not episodes:
            emit_percube(percube_ops)
        else:
            review.append(
                f"bowl tilt-pour auto-detected (max tilt {tilt.max():.0f} deg) -- confirm"
            )
            emit_percube(percube_ops)
            # Pick up the bowl (lift-off), then one Tilt covering the pour.
            maybe_nav(t_bowl_pick)
            emit(t_bowl_pick, SUBTASK_BOWL, STAGE_EXECUTE, BOWL_PICK_LANG)
            maybe_nav(episodes[0][0])
            # Single cup -> all pour episodes are one pour into it; end at the last.
            t_end = min(episodes[-1][1], T - 1)
            emit(t_end, SUBTASK_TILT, STAGE_TILT, TILT_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "strategy": "tilt_pour" if episodes else "per_cube",
            "n_poured": len(poured_ops),
            "n_percube": len(percube_ops),
            "n_episodes": len(episodes),
            "bowl_tilt_max": float(tilt.max()),
            "bowl_lift_max": float((bowl_z - z0).max()),
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
        return result

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        n_tilt = int(np.sum(names == SUBTASK_TILT))
        n_bowlpick = int(np.sum(names == SUBTASK_BOWL))
        n_poured = int(result.debug.get("n_poured", 0))
        n_percube = int(result.debug.get("n_percube", 0))

        if n_percube + n_poured != 2:
            issues.append(f"expected 2 cubes total, found {n_percube + n_poured}")
        if n_place != n_percube:
            issues.append(f"expected {n_percube} place stage(s), found {n_place}")
        if n_poured > 0:
            if n_bowlpick != 1:
                issues.append(f"expected 1 bowl Pick, found {n_bowlpick}")
            if n_tilt != 1:
                issues.append(f"expected 1 Tilt, found {n_tilt}")
        else:
            if n_bowlpick or n_tilt:
                issues.append("unexpected bowl Pick/Tilt with no poured cubes")
        if n_pick > n_place:
            issues.append(f"more cube picks ({n_pick}) than places ({n_place})")
        return issues
