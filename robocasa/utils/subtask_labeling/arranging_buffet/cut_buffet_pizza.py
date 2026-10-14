import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Subtask 1: OpenDrawer (bank atomic). Pull the drawer open. Single stage.
SUBTASK_OPEN = "OpenDrawer"

# The pizza-cutter transport, split across a navigation: pick (subtask 2) ->
# NavigateKitchen (subtask 3) -> place (subtask 4). PickPlaceDrawerToCounter is
# a bank atomic ("Pick the item from the drawer and place it on the counter").
SUBTASK_CUTTER = "PickPlaceDrawerToCounter"
CUTTER_PICK_LANG = "pick up the pizza cutter from the drawer"
CUTTER_PLACE_LANG = "place the pizza cutter on the dining counter"

# NavigateKitchen (bank atomic): carry the cutter to the dining counter (and an
# initial drive to the drawer if the robot does not start there).
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_COUNTER_LANG = "navigate to the dining counter"
NAV_TO_DRAWER_LANG = "navigate to the drawer"

DONE_LANG = "task complete"


@register_labeler
class CutBuffetPizzaLabeler(BaseSubtaskLabeler):
    """Labeler for the CutBuffetPizza composite task.

    Task: pull open a drawer, take the pizza cutter out, carry it to the dining
    counter, and set it down (next to a pizza that is already there). One object,
    fixed order. Distractor utensils/mugs/bread on the counter are ignored.

    Decomposition (4 subtasks; the doc lists 2, but the drawer-open and the
    navigation make it the OpenDoor -> pick -> navigate -> place pattern):
      1. OpenDrawer                          -> open the {left/right} drawer
      2. PickPlaceDrawerToCounter [pick]     -> take the cutter from the drawer
      3. NavigateKitchen                     -> carry it to the dining counter
      4. PickPlaceDrawerToCounter [place]    -> set it on the dining counter

    Boundaries:
      - t_open:  the summed drawer-slide displacement reaches DRAWER_OPEN_FRAC of
                 its per-demo peak. A drawer slides open (translation), so this
                 uses |joint| magnitude rather than a hinge angle; the joints
                 come from the registered drawer fixture (matched by name).
      - t_pick:  the cutter first moves DEPART_TOL from its drawer rest pose while
                 held (transport begun). Direction-agnostic.
      - t_place: the cutter settles within PLACE_TOL of its final pose with the
                 gripper released, sustained.
      - NavigateKitchen segments are carved out wherever the base drives at least
        MIN_NAV_DIST (the carry to the counter; an initial drive to the drawer if
        present). Validation is structural (one OpenDrawer, one pick + one
        place).
    """

    ENV_NAME = "CutBuffetPizza"

    # Height the cutter must rise above its in-drawer rest height to count as
    # lifted out (the pick->navigate boundary). The drawer slides HORIZONTALLY,
    # so the cutter's Z is unchanged while the drawer opens (it only moves with
    # the drawer in XY); a Z rise therefore marks the robot lifting it out, not
    # the drawer sliding. A displacement-from-start test would instead fire as
    # soon as the drawer opens, before the cutter is even grasped.
    LIFT_TOL = 0.06
    # Gripper-to-cutter distance below which the cutter counts as held (lenient:
    # the cutter is an elongated tool grasped by its handle).
    GRASP_TH = 0.12
    # Distance to the cutter's final pose within which it counts as placed.
    PLACE_TOL = 0.10
    # Gripper-to-cutter distance above which the gripper counts as released.
    RELEASE_TH = 0.15
    RELEASE_SUSTAIN = 5
    REST_W = 10

    # Drawer-open detection (summed |slide displacement|, fraction of peak). A
    # drawer opens ~0.3-0.4 m, so the thresholds are in metres, not radians.
    DRAWER_MIN_PEAK = 0.12
    DRAWER_OPEN_FRAC = 0.85
    DRAWER_START_RISE = 0.08

    # Navigation. The drawer<->dining-counter carry is a real traversal.
    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cutter_body_id = env.obj_body_id["pizza_cutter"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Drawer slide joints: joints belonging to the registered drawer fixture
        # (matched by the fixture's name prefix; layout-robust).
        self.drawer_qpos_addrs = []
        drawer = getattr(env, "drawer", None)
        dname = getattr(drawer, "name", None)
        if dname is not None:
            for j in range(env.sim.model.njnt):
                try:
                    nm = env.sim.model.joint_id2name(j)
                except Exception:
                    nm = None
                if nm and nm.startswith(dname):
                    self.drawer_qpos_addrs.append(env.sim.model.jnt_qposadr[j])
        # Drawer side (left/right) for the OpenDrawer lang. Read it from the
        # demo's recorded ep_meta lang ("...drawer on the {left/right}..."), which
        # is the ground truth and identical across the hdf5/LeRobot surfaces.
        # NOT env.drawer_side, which is re-randomized on each env reset and would
        # disagree between surfaces (and can be plain wrong).
        lang = (ep_meta or {}).get("lang", "") or ""
        if "on the left" in lang:
            side = "left"
        elif "on the right" in lang:
            side = "right"
        else:
            side = getattr(env, "drawer_side", None)
        self.open_lang = f"open the {side} drawer" if side else "open the drawer"
        self.expected_done_idx = None
        self.extra_subtasks = set()  # OpenDrawer + PickPlaceDrawerToCounter are bank

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        drawer = (
            sum(abs(env.sim.data.qpos[a]) for a in self.drawer_qpos_addrs)
            if self.drawer_qpos_addrs
            else 0.0
        )
        return {
            "cutter": env.sim.data.body_xpos[self.cutter_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
            "drawer": float(drawer),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, cutter, grip, T):
        final = np.median(cutter[-self.REST_W:], axis=0)
        settled = np.linalg.norm(cutter - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(cutter - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, cutter, grip, t_release):
        # Lift-out: the cutter rises LIFT_TOL above its in-drawer rest height
        # while held. The rest height is the cutter's Z in the drawer (constant
        # through the horizontal drawer slide), so this fires when the robot
        # lifts it out, not while the drawer opens.
        rest_z = float(np.median(cutter[: self.REST_W, 2]))
        lifted = cutter[:, 2] > rest_z + self.LIFT_TOL
        held = np.linalg.norm(cutter - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((lifted & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(lifted[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    def _open_time(self, drawer, a, b):
        b = max(b, a + 1)
        seg = drawer[a:b]
        peak = float(seg.max()) if seg.size else 0.0
        if peak < self.DRAWER_MIN_PEAK:
            return None, peak
        w = np.where(seg >= peak * self.DRAWER_OPEN_FRAC)[0]
        return (a + int(w[0]) if w.size else None), peak

    def _open_start_time(self, drawer, a, b):
        b = max(b, a + 1)
        base = float(drawer[a])
        w = np.where(drawer[a:b] > base + self.DRAWER_START_RISE)[0]
        return a + int(w[0]) if w.size else a

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
        cutter = np.array([h["cutter"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        drawer = np.array([h["drawer"] for h in history])
        has_base = np.isfinite(base_xy).all()

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        t_release = self._release_time(cutter, grip, T)
        t_depart = self._depart_time(cutter, grip, t_release)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang):
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        # --- OpenDrawer phase: optional drive to the drawer, then pull it open ---
        t_dep = max(t_depart, cursor)
        t_open, drawer_peak = self._open_time(drawer, cursor, max(t_dep, cursor + 1))
        open_ok = t_open is not None
        if t_open is None:
            t_open = cursor
        t_start = self._open_start_time(drawer, cursor, max(t_open, cursor + 1))
        maybe_nav(t_start, NAV_TO_DRAWER_LANG)
        if t_open > cursor:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, self.open_lang)
        # --- pick: reach + grasp + lift the cutter out of the drawer ---
        emit(t_dep, SUBTASK_CUTTER, STAGE_PICK, CUTTER_PICK_LANG)
        # --- carry to the dining counter, then place ---
        t_rel = max(t_release, cursor)
        maybe_nav(t_rel, NAV_TO_COUNTER_LANG)
        emit(t_rel, SUBTASK_CUTTER, STAGE_PLACE, CUTTER_PLACE_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_open": int(t_open),
            "t_depart": int(t_depart),
            "t_release": int(t_release),
            "drawer_peak": float(drawer_peak),
            "open_ok": int(open_ok),
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
        names = result.subtask_name[firsts]
        n_open = int(np.sum(names == SUBTASK_OPEN))
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_open != 1:
            issues.append(f"expected 1 OpenDrawer, found {n_open}")
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        return issues
