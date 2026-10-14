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


# Each ice cube is one pick-and-place from the bowl into its target glass. No
# exact bank name covers bowl-on-counter -> glass-on-counter, so the name is a
# declared bank extension (cf. AddIceCubes' PickPlaceCounterToBlender). Split
# into pick/place stages; the place lang names the target glass by its side
# (left/right), resolved per demo from the agentview camera (see setup()).
SUBTASK = "PickPlaceCounterToCounter"
PICK_LANG = "pick up an ice cube from the bowl"


def _place_lang(side):
    return f"place the ice cube in the {side} glass of water"


# NavigateKitchen (bank atomic). The two glasses sit at different stools along
# the dining counter; when they are far enough apart the robot drives between
# the bowl and a glass. Inserted adaptively (see MIN_NAV_DIST). Destination-aware
# lang: heading to a glass to place, or back to the bowl to pick the next cube.
SUBTASK_NAV = "NavigateKitchen"


def _nav_to_glass_lang(side):
    return f"navigate to the {side} glass of water"


NAV_TO_BOWL_LANG = "navigate back to the ice bowl"

# Bowl-carry / tilt-pour strategy (same skills as AddIceCubes): instead of
# moving cubes one at a time, the robot lifts the whole bowl and tilts it over a
# glass to pour cubes in. "Pick" (grab the bowl) and "Tilt" (pour) are
# bank-extension skills. A demo may use this for one glass and per-cube for the
# other (mixed), so both schemes can appear together.
SUBTASK_BOWL = "Pick"
BOWL_PICK_LANG = "pick up the bowl of ice cubes"
SUBTASK_TILT = "Tilt"


def _tilt_lang(side):
    return f"tilt the bowl to pour the ice cubes into the {side} glass of water"


# Camera used to resolve which glass is on the left vs the right (matches the
# rendered agentview the demos are reviewed in).
SIDE_CAMERA = "robot0_agentview_center"

DONE_LANG = "task complete"


@register_labeler
class PlaceEqualIceCubesLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceEqualIceCubes composite task.

    Task: four ice cubes start in one bowl; place two in each of two glasses of
    water that sit at different stools along the dining counter. Four per-cube
    pick-and-places (doc-set count = 4).

    Auto-detected per demo:
      - Each cube's target glass: the glass its final pose is closest to. The
        two cubes per glass are validated 2/2. Each glass is named by its side
        (left/right), resolved from the agentview camera so the labels match the
        rendered view (the left/right glass is not fixed to glass_cup1/2 -- it
        varies with the layout).
      - Handling order: cubes are placed in two glass-groups (both cubes for one
        glass, then both for the other); which glass is filled first varies.
        Ordered by the frame each cube settles in its glass.

    Segmentation is event-anchored with adaptive navigation, exactly like the
    MakeIceLemonade labeler:
      - pick  stage: reach the bowl + grasp + lift the cube, ending at t_depart
                     (the cube first leaves the bowl while held).
      - place stage: carry + lower + release into the target glass, ending at
                     t_release.
      - A NavigateKitchen segment is carved out of the leading part of any phase
        in which the base drives at least MIN_NAV_DIST: driving a cube out to the
        far glass (place phase) or driving back to the bowl for the next cube
        (pick phase). When both glasses are reachable from one base pose the
        robot never drives, so those demos are just 4 pick-places + done.

    Bowl-carry / tilt-pour variant (same skills as AddIceCubes): instead of
    moving cubes one at a time, the teleoperator may lift the whole bowl and tilt
    it over a glass to pour its cubes in. Detected geometrically -- the bowl's
    up-axis rotates past POUR_TILT_TH while it is lifted POUR_LIFT_TH off the
    counter. Such a glass is labeled "Pick" (grab the bowl, one per demo) ->
    "Tilt" (pour into that glass), with NavigateKitchen for the carry. A demo may
    be mixed: one glass filled cube-by-cube and the other poured (common here);
    each glass is labeled by the method actually used, in performed order
    (per-cube first, since those cubes are removed before the bowl is lifted).
    Tilt-pour demos carry a soft review note.

    The segment count varies with strategy/layout/route; validation is structural
    (two cubes per glass; each glass either two place stages or one Tilt; one bowl
    Pick when any pour occurs) rather than a fixed total.
    """

    ENV_NAME = "PlaceEqualIceCubes"

    # Distance a cube must move from its bowl rest pose to count as departing
    # (transport begun). Direction-agnostic displacement, not lift height.
    DEPART_TOL = 0.10
    # Gripper-to-cube distance below which the cube counts as held.
    GRASP_TH = 0.07
    # XY distance to a glass within which a cube counts as delivered.
    GLASS_TOL = 0.09
    # Gripper-to-cube distance above which the gripper counts as released.
    RELEASE_TH = 0.12
    # Sustained delivered-and-released frames to commit a release event.
    RELEASE_SUSTAIN = 5
    # Frames over which a cube's bowl rest pose is measured at the start.
    REST_W = 10
    # Two cubes whose departures fall within this many frames are treated as one
    # multi-grab (one pick stage, then a place stage per cube). Single grabs are
    # separated by ~100+ frames; a co-grab departs within a few frames.
    MULTIGRAB_WINDOW = 50

    # --- bowl-carry / tilt-pour detection (mirrors AddIceCubes) ---
    # Degrees the bowl's up-axis must rotate from rest to count as tilting.
    POUR_TILT_TH = 45.0
    # Height the bowl must rise above its rest pose to count as lifted.
    POUR_LIFT_TH = 0.05
    # Minimum sustained tilted-while-lifted frames to commit a pour (rejects a
    # one-frame orientation blip).
    POUR_MIN_FRAMES = 3
    # Two pour episodes over the same glass separated by fewer than this many
    # frames are merged (a single pour can dip below threshold momentarily).
    POUR_MERGE_GAP = 40

    # Mobile-base XY speed (m/frame, smoothed) above which the base is driving.
    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    # Net base displacement a leading drive must reach to be carved out as a
    # NavigateKitchen segment. The two glasses are only 0.4-1.1 m apart (closer
    # than MakeIceLemonade's fridge<->counter), so the threshold is lower here:
    # genuine inter-glass / bowl-return drives net 0.59-1.2 m, while local
    # positioning nudges cap at ~0.4 m, leaving a clean gap at ~0.5 m. Close-glass
    # layouts where the robot reaches both glasses from one base pose never drive
    # (base travel ~0), so they get no navs regardless.
    MIN_NAV_DIST = 0.50

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cube_body_ids = [env.obj_body_id[f"ice_cube{i}"] for i in range(1, 5)]
        self.bowl_body_id = env.obj_body_id["ice_bowl"]
        self.glass_body_ids = [
            env.obj_body_id["glass_cup1"],
            env.obj_body_id["glass_cup2"],
        ]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Resolve each glass's side (left/right) from the agentview camera at the
        # initial pose, so labels match the rendered view. The glasses are static,
        # so this fixed assignment holds for the whole demo. self.glass_side[i] is
        # the side of glass_cup{i+1}. Falls back to glass1=left / glass2=right if
        # the camera is unavailable.
        self.glass_side = ["left", "right"]
        try:
            cid = env.sim.model.camera_name2id(SIDE_CAMERA)
            cam_pos = env.sim.data.cam_xpos[cid].copy()
            cam_right = env.sim.data.cam_xmat[cid].reshape(3, 3)[:, 0]
            lat = [
                float(np.dot(env.sim.data.body_xpos[b] - cam_pos, cam_right))
                for b in self.glass_body_ids
            ]
            # Larger projection onto the camera's right axis = further right.
            if lat[0] <= lat[1]:
                self.glass_side = ["left", "right"]
            else:
                self.glass_side = ["right", "left"]
        except Exception:
            pass
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK, SUBTASK_BOWL, SUBTASK_TILT}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        # Bowl pose: full position + world up-axis (its local +z), for tilt-pour
        # detection. mujoco body_xquat is wxyz; transform_utils wants xyzw.
        q = np.array(env.sim.data.body_xquat[self.bowl_body_id])
        bowl_up = T.quat2mat(q[[1, 2, 3, 0]])[:, 2].copy()
        return {
            "cubes": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.cube_body_ids]
            ),
            "glasses": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.glass_body_ids]
            ),
            "bowl": env.sim.data.body_xpos[self.bowl_body_id].copy(),
            "bowl_up": bowl_up,
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
        }

    # ---- event helpers (shared shape with MakeIceLemonade) -------------

    def _release_time(self, cube, grip, glass, T):
        in_g = np.linalg.norm(cube[:, :2] - glass[:, :2], axis=1) < self.GLASS_TOL
        released = np.linalg.norm(cube - grip, axis=1) > self.RELEASE_TH
        ok = in_g & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(in_g)[0]
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
        """Cluster per-cube ops by departure time: cubes grabbed within
        MULTIGRAB_WINDOW frames of each other form one grab (one pick stage,
        then a place stage per cube)."""
        groups = []
        for o in sorted(ops, key=lambda o: o["dep"]):
            if groups and (o["dep"] - groups[-1][-1]["dep"]) <= self.MULTIGRAB_WINDOW:
                groups[-1].append(o)
            else:
                groups.append([o])
        return groups

    def _cube_events(self, cubes, glasses, grip, tgt, T, which):
        """Per-cube (dep, rel) events for the given cube indices."""
        ops = []
        for c in which:
            cube = cubes[:, c, :]
            glass = glasses[:, tgt[c] - 1, :]
            t_rel = self._release_time(cube, grip, glass, T)
            t_dep = self._depart_time(cube, grip, t_rel)
            ops.append(dict(cube=c, tgt=tgt[c], dep=t_dep, rel=t_rel))
        return ops

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        cubes = np.array([h["cubes"] for h in history])     # (T, 4, 3)
        glasses = np.array([h["glasses"] for h in history])  # (T, 2, 3)
        bowl = np.array([h["bowl"] for h in history])        # (T, 3)
        bowl_up = np.array([h["bowl_up"] for h in history])  # (T, 3)
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
        # Bowl resting position (where it sits on the counter). Used as the fixed
        # "bowl" nav landmark: once the robot lifts the bowl to pour, the bowl
        # travels with the base, so its live position would always read as
        # "nearby" -- the resting spot is what the robot navigates back to.
        bowl_rest = np.median(bowl[: self.REST_W, :2], axis=0)
        # Per-cube target glass (1 or 2), by final position.
        g_final = glasses[-1]
        tgt = [
            1 if np.linalg.norm(cubes[-1, c, :2] - g_final[0, :2])
            <= np.linalg.norm(cubes[-1, c, :2] - g_final[1, :2]) else 2
            for c in range(ncubes)
        ]

        # --- bowl tilt-pour detection (mirrors AddIceCubes) ---
        bowl_z = bowl[:, 2]
        z0 = float(np.median(bowl_z[: self.REST_W]))
        tilt = np.degrees(np.arccos(np.clip(bowl_up @ bowl_up[0], -1.0, 1.0)))
        lifted = (bowl_z - z0) > self.POUR_LIFT_TH
        pour = (tilt > self.POUR_TILT_TH) & lifted
        episodes = []  # [start, end, glass_idx] for each pour
        i = 0
        while i < T:
            if pour[i]:
                j = i
                while j < T and pour[j]:
                    j += 1
                if j - i >= self.POUR_MIN_FRAMES:
                    seg = bowl[i:j, :2]
                    dg = [
                        float(np.min(np.linalg.norm(seg - glasses[i:j, k, :2], axis=1)))
                        for k in range(2)
                    ]
                    episodes.append([i, j, int(np.argmin(dg))])
                i = j
            else:
                i += 1
        # One Tilt per poured glass: merge consecutive pour episodes over the
        # same glass into a single tilt span. The robot may dip the bowl more
        # than once into the same glass (with an un-tilt/reposition between);
        # that is still one "pour into glass X" subtask. (A different glass in
        # between breaks the run, yielding a separate Tilt.)
        merged = []
        for ep in episodes:
            if merged and merged[-1][2] == ep[2]:
                merged[-1][1] = ep[1]
            else:
                merged.append(list(ep))
        episodes = merged
        poured = sorted({ep[2] for ep in episodes})  # glass indices that are poured

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def nav_lang(a, b):
            # Destination = the landmark the base moves TOWARD the most over the
            # drive (largest distance decrease). The bowl sits centrally between
            # the glasses, so a phase-based guess is unreliable; the base motion
            # is not. Landmarks: the bowl and the two glasses (named by side).
            if base_xy is None:
                return NAV_TO_BOWL_LANG
            sb = base_xy[a]
            eb = base_xy[min(b - 1, len(base_xy) - 1)]
            cands = [
                (NAV_TO_BOWL_LANG, bowl_rest),
                (_nav_to_glass_lang(self.glass_side[0]), glasses[min(b - 1, T - 1), 0, :2]),
                (_nav_to_glass_lang(self.glass_side[1]), glasses[min(b - 1, T - 1), 1, :2]),
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

        def emit_percube(groups):
            for grp in sorted(groups, key=lambda g: min(c["rel"] for c in g)):
                pick_end = max(max(c["dep"] for c in grp), cursor)
                maybe_nav(pick_end)
                emit(pick_end, SUBTASK, STAGE_PICK, PICK_LANG)
                for c in sorted(grp, key=lambda c: c["rel"]):
                    side = self.glass_side[c["tgt"] - 1]
                    maybe_nav(max(c["rel"], cursor))
                    emit(max(c["rel"], cursor), SUBTASK, STAGE_PLACE, _place_lang(side))

        review = []
        if not poured:
            # Pure per-cube strategy: every cube picked and placed individually
            # (with multi-grab grouping for genuinely co-grabbed cubes).
            groups = self._group_multigrab(
                self._cube_events(cubes, glasses, grip, tgt, T, range(ncubes))
            )
            emit_percube(groups)
        else:
            review.append(
                f"bowl tilt-pour auto-detected (max tilt {tilt.max():.0f} deg) -- confirm"
            )
            # Cubes whose glass is NOT poured are still delivered one at a time;
            # this happens before the bowl is lifted.
            percube_idx = [c for c in range(ncubes) if (tgt[c] - 1) not in poured]
            groups = self._group_multigrab(
                self._cube_events(cubes, glasses, grip, tgt, T, percube_idx)
            )
            emit_percube(groups)
            # Pick up the whole bowl: lift-off leading into the first pour.
            first_pour = min(ep[0] for ep in episodes)
            k = first_pour
            while k > 0 and lifted[k - 1]:
                k -= 1
            maybe_nav(k)
            emit(k, SUBTASK_BOWL, STAGE_EXECUTE, BOWL_PICK_LANG)
            # One Tilt per poured glass, in pour order; carry (nav) before each.
            eps_sorted = sorted(episodes, key=lambda e: e[0])
            for n, (a, b, gi) in enumerate(eps_sorted):
                maybe_nav(a)
                last = n == len(eps_sorted) - 1
                t_end = min(b, T - 1) if last else b
                emit(t_end, SUBTASK_TILT, STAGE_TILT, _tilt_lang(self.glass_side[gi]))

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "strategy": "tilt_pour" if poured else "per_cube",
            "poured_sides": np.array([self.glass_side[g] for g in poured]),
            "n_poured": len(poured),
            "n_episodes": len(episodes),
            "bowl_tilt_max": float(tilt.max()),
            "bowl_lift_max": float((bowl_z - z0).max()),
            "n_glass1": int(sum(1 for t in tgt if t == 1)),
            "n_glass2": int(sum(1 for t in tgt if t == 2)),
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

        # Each of the two glasses is filled either per-cube (2 place stages) or by
        # a single Tilt; a tilt-pour run lifts the bowl once (one "Pick").
        if n_place != 2 * (2 - n_poured):
            issues.append(
                f"expected {2 * (2 - n_poured)} place stages "
                f"({2 - n_poured} per-cube glass(es)), found {n_place}"
            )
        if n_tilt != n_poured:
            issues.append(f"expected {n_poured} Tilt(s), found {n_tilt}")
        if n_bowlpick != (1 if n_poured else 0):
            issues.append(
                f"expected {1 if n_poured else 0} bowl Pick, found {n_bowlpick}"
            )
        if n_pick > n_place:
            issues.append(f"more cube picks ({n_pick}) than places ({n_place})")
        n1 = result.debug.get("n_glass1")
        n2 = result.debug.get("n_glass2")
        if n1 is not None and (n1 != 2 or n2 != 2):
            issues.append(f"expected 2 cubes per glass, found {n1}/{n2}")
        return issues
