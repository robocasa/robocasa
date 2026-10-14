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


# Subtask 1: open the freezer door. The bank has OpenFridge but no freezer-
# specific skill, and the task explicitly opens the freezer compartment, so this
# is a declared bank extension. Single stage.
SUBTASK_OPEN = "OpenFreezer"
OPEN_LANG = "open the freezer door"

# The tray transport, split across a navigation: pick (subtask 2) ->
# NavigateKitchen (subtask 3) -> place (subtask 4). No exact bank name covers
# freezer -> dining counter, so the PickPlace name is a declared extension; the
# name is unchanged across the pick/place stages.
SUBTASK_TRAY = "PickPlaceFreezerToCounter"
TRAY_PICK_LANG = "pick up the ice cube tray from the freezer"
TRAY_PLACE_LANG = "place the ice cube tray on the dining counter"

# NavigateKitchen (bank atomic). The robot carries the tray from the freezer to
# the dining counter. A leading drive to the freezer is also possible if the
# robot does not start there.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_COUNTER_LANG = "navigate to the dining counter"
NAV_TO_FREEZER_LANG = "navigate to the freezer"

DONE_LANG = "task complete"


@register_labeler
class RetrieveIceTrayLabeler(BaseSubtaskLabeler):
    """Labeler for the RetrieveIceTray composite task.

    Task: open the freezer, take the ice cube tray out, carry it across the
    kitchen, and set it on the dining counter. One object, fixed order.

    Doc-canonical decomposition (4 subtasks):
      1. OpenFreezer                          -> open the freezer door
      2. PickPlaceFreezerToCounter [pick]     -> take the tray from the freezer
      3. NavigateKitchen                      -> carry it to the dining counter
      4. PickPlaceFreezerToCounter [place]    -> set it on the dining counter
    (The tray's pick and place are two subtasks because navigation separates
    them, mirroring MakeIceLemonade's lemon operation.)

    Boundaries (each detected from its own signal):
      - t_open:  the summed freezer-door angle reaches DOOR_OPEN_FRAC of its
                 per-demo peak -- the door has finished swinging and the robot
                 reaches for the tray. Door joints come from the registered
                 fridge fixture's freezer compartment (layout-robust).
      - t_pick:  the tray first moves DEPART_TOL from its freezer rest pose while
                 held (transport begun). Direction-agnostic (the tray comes down
                 from a high freezer shelf onto the lower counter).
      - t_place: the tray settles within PLACE_TOL of its final pose and the
                 gripper has released, sustained.
      - NavigateKitchen segments are carved out of the leading part of a phase
        whenever the base drives at least MIN_NAV_DIST: the carry to the dining
        counter (place phase) and, if present, an initial drive to the freezer.

    Validation is structural (one OpenFreezer, one tray pick + one place) rather
    than a fixed segment count, since a pre-opened freezer or a no-drive layout
    changes the total.
    """

    ENV_NAME = "RetrieveIceTray"

    # Distance the tray must move from its freezer rest pose to count as
    # departing (transport begun). Direction-agnostic displacement.
    DEPART_TOL = 0.10
    # Gripper-to-tray distance below which the tray counts as held.
    GRASP_TH = 0.10
    # Distance to the tray's final pose within which it counts as placed.
    PLACE_TOL = 0.10
    # Gripper-to-tray distance above which the gripper counts as released.
    RELEASE_TH = 0.15
    # Sustained placed-and-released frames to commit the place event.
    RELEASE_SUSTAIN = 5
    # Frames over which the tray's freezer rest pose is measured at the start.
    REST_W = 10

    # Minimum peak summed freezer-door angle (rad) for an open to be real.
    DOOR_MIN_PEAK = 0.30
    # Fraction of the summed door angle's in-window peak at which the door counts
    # as open (OpenFreezer -> pick boundary).
    DOOR_OPEN_FRAC = 0.85
    # Door-angle rise (rad) above its phase-start level marking the robot
    # beginning to open the door (nav -> OpenFreezer boundary).
    DOOR_START_RISE = 0.15

    # Mobile-base XY speed (m/frame, smoothed) above which the base is driving.
    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    # Net base displacement a leading drive must reach to be a NavigateKitchen.
    # The freezer<->dining-counter traversal is a real cross-kitchen drive.
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.tray_body_id = env.obj_body_id["ice_cube_tray"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Freezer door joints from the registered fridge fixture (layout-robust).
        self.door_qpos_addrs = []
        fridge = getattr(env, "fridge", None)
        for nm in list(getattr(fridge, "_freezer_door_joint_names", [])):
            try:
                jid = env.sim.model.joint_name2id(nm)
                self.door_qpos_addrs.append(env.sim.model.jnt_qposadr[jid])
            except Exception:
                continue
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_OPEN, SUBTASK_TRAY}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        door = (
            sum(abs(env.sim.data.qpos[a]) for a in self.door_qpos_addrs)
            if self.door_qpos_addrs
            else 0.0
        )
        return {
            "tray": env.sim.data.body_xpos[self.tray_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
            "door": float(door),
        }

    # ---- event helpers (shared shape with MakeIceLemonade) -------------

    def _release_time(self, tray, grip, T):
        """First frame the tray settles within PLACE_TOL of its final pose with
        the gripper released, sustained RELEASE_SUSTAIN frames."""
        final = np.median(tray[-self.REST_W:], axis=0)
        settled = np.linalg.norm(tray - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(tray - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, tray, grip, t_release):
        start = np.median(tray[: self.REST_W], axis=0)
        moved = np.linalg.norm(tray - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(tray - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    def _door_open_time(self, door, a, b):
        b = max(b, a + 1)
        seg = door[a:b]
        peak = float(seg.max()) if seg.size else 0.0
        if peak < self.DOOR_MIN_PEAK:
            return None, peak
        w = np.where(seg >= peak * self.DOOR_OPEN_FRAC)[0]
        return (a + int(w[0]) if w.size else None), peak

    def _door_start_time(self, door, a, b):
        b = max(b, a + 1)
        base = float(door[a])
        w = np.where(door[a:b] > base + self.DOOR_START_RISE)[0]
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
        tray = np.array([h["tray"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        door = np.array([h["door"] for h in history])
        has_base = np.isfinite(base_xy).all()

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        t_release = self._release_time(tray, grip, T)
        t_depart = self._depart_time(tray, grip, t_release)

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

        # --- OpenFreezer phase: optional drive to the freezer, then open ---
        t_dep = max(t_depart, cursor)
        t_open, door_peak = self._door_open_time(door, cursor, max(t_dep, cursor + 1))
        door_open_ok = t_open is not None
        if t_open is None:
            t_open = cursor
        t_start = self._door_start_time(door, cursor, max(t_open, cursor + 1))
        maybe_nav(t_start, NAV_TO_FREEZER_LANG)
        if t_open > cursor:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        # --- pick: reach + grasp + lift the tray out of the freezer ---
        emit(t_dep, SUBTASK_TRAY, STAGE_PICK, TRAY_PICK_LANG)
        # --- carry to the dining counter, then place ---
        t_rel = max(t_release, cursor)
        maybe_nav(t_rel, NAV_TO_COUNTER_LANG)
        emit(t_rel, SUBTASK_TRAY, STAGE_PLACE, TRAY_PLACE_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_open": int(t_open),
            "t_depart": int(t_depart),
            "t_release": int(t_release),
            "door_peak": float(door_peak),
            "door_open_ok": int(door_open_ok),
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
            issues.append(f"expected 1 OpenFreezer, found {n_open}")
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        return issues
