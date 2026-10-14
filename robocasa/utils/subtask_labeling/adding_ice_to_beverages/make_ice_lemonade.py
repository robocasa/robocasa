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


# Subtask 1: OpenFridge (bank atomic). The lemon wedge sits inside the fridge,
# so the robot must open the fridge door before it can be picked. Single stage.
SUBTASK_OPEN = "OpenFridge"
OPEN_LANG = "open the fridge door"

# The lemon transport, split across a navigation between the fridge and the
# glass: pick (subtask 2) -> NavigateKitchen (subtask 3) -> place (subtask 4).
# The PickPlace name is unchanged across the pick/place stages; no exact bank
# name covers fridge-shelf -> cup-on-counter, so it is a declared bank extension.
SUBTASK_LEMON = "PickPlaceFridgeToCounter"
LEMON_PICK_LANG = "pick up the lemon wedge from the fridge"
LEMON_PLACE_LANG = "place the lemon wedge in the glass of lemonade"

# Subtask 5: the ice cube, a single pick-and-place from the bowl into the glass
# (both on the counter -> no navigation). Split only into pick/place stages.
SUBTASK_ICE = "PickPlaceCounterToCounter"
ICE_PICK_LANG = "pick up an ice cube from the bowl"
ICE_PLACE_LANG = "place the ice cube in the glass of lemonade"

# NavigateKitchen (bank atomic). The lemon carry from the fridge to the glass is
# subtask 3. The same skill also covers the reposition drives a teleoperator
# makes when they do the ice first and then drive back to the fridge.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_FRIDGE_LANG = "navigate to the fridge"
NAV_TO_COUNTER_LANG = "navigate to the counter with the lemonade glass"

DONE_LANG = "task complete"


@register_labeler
class MakeIceLemonadeLabeler(BaseSubtaskLabeler):
    """Labeler for the MakeIceLemonade composite task.

    Task: open the fridge, move a lemon wedge from the fridge into the glass of
    lemonade, and move one ice cube from a bowl on the counter into the same
    glass.

    Doc-canonical decomposition (5 subtasks):
      1. OpenFridge                          -> open the fridge door
      2. PickPlaceFridgeToCounter [pick]     -> pick the lemon from the fridge
      3. NavigateKitchen                     -> carry the lemon to the glass
      4. PickPlaceFridgeToCounter [place]    -> place the lemon in the glass
      5. PickPlaceCounterToCounter           -> pick+place the ice in the glass
    (The lemon's pick and place are counted as two subtasks because a navigation
    separates them; the ice has no navigation so it is a single subtask, split
    only into pick/place stages.)

    Two things vary across teleoperated demos and are auto-detected:
      - Order. The teleoperator may do the lemon operation first or the ice
        first; the two operations never interleave. Ordered by the frame each
        object settles in the cup, so labels stay faithful to the video.
      - Which ice cube. Two cubes sit in the bowl; the used one is the cube that
        ends up in the cup (smallest final XY distance to the cup).

    The fridge door is found via the registered fridge fixture (layout-robust:
    single- or double-door fridges, named fridge_{left,right,main}_group), so
    the OpenFridge boundary does not depend on a hard-coded joint name. When the
    ice is done first, the OpenFridge / lemon subtasks simply occur later in the
    timeline, after a reposition NavigateKitchen back to the fridge.

    Navigation is inserted adaptively: wherever the mobile base drives at least
    MIN_NAV_DIST between phases (initial drive to a source, the lemon carry, or
    a reposition between the two operations) a NavigateKitchen segment is carved
    out. The segment count therefore varies with the route taken; validation is
    structural (exactly one OpenFridge, one lemon pick + place, one ice pick +
    place) rather than a fixed total.
    """

    ENV_NAME = "MakeIceLemonade"

    # Distance an object must move from its resting pose to count as departing
    # its source (transport has begun). Direction-agnostic: the lemon is pulled
    # DOWN from a high fridge shelf into the lower cup, so a height-based
    # lift-off test would miss it -- displacement from the start pose does not.
    DEPART_TOL = 0.10
    # Gripper-to-object distance below which the object counts as held.
    GRASP_TH = 0.07
    # XY distance to the cup within which an object counts as delivered.
    CUP_TOL = 0.08
    # Gripper-to-object distance above which the gripper counts as released.
    RELEASE_TH = 0.12
    # Sustained frames an object must sit delivered-and-released to commit the
    # release event (rejects a momentary pass through the cup region).
    RELEASE_SUSTAIN = 5
    # Frames over which an object's resting pose is measured at the start.
    REST_W = 10

    # Minimum peak summed door angle (rad) for an open to be considered real;
    # below this the robot never actually opened the fridge (flagged for review).
    DOOR_MIN_PEAK = 0.30
    # Fraction of the summed door angle's in-window peak at which the door counts
    # as open (the OpenFridge -> pick boundary: the door has finished swinging
    # and the robot reaches for the lemon). Summed over all door joints and
    # measured relative to the per-demo peak, so it is robust to single-door,
    # double-door, and partial opens alike -- unlike the fixture's binary
    # is_open(), which never fires when only one leaf of a french-door fridge is
    # opened.
    DOOR_OPEN_FRAC = 0.85
    # Door-angle rise (rad) above its level at the start of the phase that marks
    # the robot beginning to open the door -- the nav->OpenFridge boundary. The
    # robot often keeps nudging the base while opening, so the leading drive ends
    # when the door starts moving, not at the last base twitch.
    DOOR_START_RISE = 0.15

    # Mobile-base XY speed (m/frame, smoothed) above which the base counts as
    # actively driving.
    NAV_MOVING_TH = 0.004
    # Window over which base speed is smoothed before thresholding.
    NAV_SMOOTH_W = 7
    # Net base displacement (start-to-end of the drive) a leading move within a
    # phase must reach to be carved out as a NavigateKitchen segment. A genuine
    # fridge<->counter traversal nets ~0.70-1.70 m; local repositioning at a
    # station the robot is already at (aligning to the fridge, or stepping from
    # the glass to the ice bowl on the same counter) nets <0.60 m even when its
    # path wanders. 0.65 m sits in that gap, so local nudges fold into the
    # adjacent OpenFridge / pick stage instead of becoming spurious navigations.
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.lemon_body_id = env.obj_body_id["lemon_wedge"]
        self.cube_body_ids = [
            env.obj_body_id["ice_cube1"],
            env.obj_body_id["ice_cube2"],
        ]
        self.cup_body_id = env.obj_body_id["glass_cup"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # Mobile base body (PandaOmron), for drive detection.
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Fridge door joints from the registered fixture (layout-robust).
        self.door_qpos_addrs = []
        self.fridge_fixture = getattr(env, "fridge", None)
        fridge = self.fridge_fixture
        door_names = list(getattr(fridge, "_fridge_door_joint_names", []))
        for nm in door_names:
            try:
                jid = env.sim.model.joint_name2id(nm)
                self.door_qpos_addrs.append(env.sim.model.jnt_qposadr[jid])
            except Exception:
                continue
        # Variable per demo; structural validation replaces the fixed-count check.
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_LEMON, SUBTASK_ICE}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        # Summed opening angle across all fridge door leaves -- keeps rising as
        # the robot opens a second leaf, and reflects partial opens.
        door = (
            sum(abs(env.sim.data.qpos[a]) for a in self.door_qpos_addrs)
            if self.door_qpos_addrs
            else 0.0
        )
        return {
            "lemon": env.sim.data.body_xpos[self.lemon_body_id].copy(),
            "cubes": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.cube_body_ids]
            ),
            "cup": env.sim.data.body_xpos[self.cup_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
            "door": float(door),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, obj, grip, cup, T):
        """First frame the object is delivered into the cup (within CUP_TOL XY
        of the cup and the gripper released), sustained RELEASE_SUSTAIN frames."""
        in_cup = np.linalg.norm(obj[:, :2] - cup[:, :2], axis=1) < self.CUP_TOL
        released = np.linalg.norm(obj - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, obj, grip, t_release):
        """Pick->place boundary: first frame (before t_release) where the object
        has moved DEPART_TOL from its source pose while held -- transport has
        begun. Direction-agnostic (handles the lemon pulled down into the cup)."""
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(obj - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    def _door_open_time(self, door, a, b):
        """Frame within [a, b) at which the summed door angle first reaches
        DOOR_OPEN_FRAC of its in-window peak (the OpenFridge -> pick boundary:
        the door has finished swinging open and the robot reaches for the
        lemon). Returns (t_open, peak). t_open is None if the door never
        meaningfully opened in the window."""
        b = max(b, a + 1)
        seg = door[a:b]
        peak = float(seg.max()) if seg.size else 0.0
        if peak < self.DOOR_MIN_PEAK:
            return None, peak
        w = np.where(seg >= peak * self.DOOR_OPEN_FRAC)[0]
        return (a + int(w[0]) if w.size else None), peak

    def _door_start_time(self, door, a, b):
        """Frame within [a, b) at which the door first rises DOOR_START_RISE
        above its level at a -- the robot begins opening it (nav->OpenFridge
        boundary). Falls back to a if no rise is seen."""
        b = max(b, a + 1)
        base = float(door[a])
        w = np.where(door[a:b] > base + self.DOOR_START_RISE)[0]
        return a + int(w[0]) if w.size else a

    def _nav_end(self, base_speed, base_xy, a, b):
        """If the base drives at least MIN_NAV_DIST inside phase window [a, b),
        return the frame just after the last driving frame (the nav->manip
        handoff). Otherwise return a (no nav)."""
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
        lemon = np.array([h["lemon"] for h in history])
        cubes = np.array([h["cubes"] for h in history])  # (T, 2, 3)
        cup = np.array([h["cup"] for h in history])
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

        # Which ice cube was used? The one that ends up in the cup.
        final_d = np.linalg.norm(cubes[-1, :, :2] - cup[-1, :2], axis=1)
        used = int(np.argmin(final_d))
        ice = cubes[:, used, :]

        # Per-object pick (depart) + place (release) events.
        t_rel_lemon = self._release_time(lemon, grip, cup, T)
        t_rel_ice = self._release_time(ice, grip, cup, T)
        t_dep_lemon = self._depart_time(lemon, grip, t_rel_lemon)
        t_dep_ice = self._depart_time(ice, grip, t_rel_ice)

        order = ["lemon", "ice"] if t_rel_lemon <= t_rel_ice else ["ice", "lemon"]

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

        door_peak = 0.0
        door_open_ok = True
        for key in order:
            if key == "lemon":
                t_dep = max(t_dep_lemon, cursor)
                t_rel = max(t_rel_lemon, t_dep)
                # OpenFridge phase: leading drive to the fridge, then open the
                # door, ending when the door reaches its open angle.
                t_open, door_peak = self._door_open_time(
                    door, cursor, max(t_dep, cursor + 1))
                if t_open is None:
                    door_open_ok = False
                    t_open = cursor  # no detectable open; OpenFridge is empty
                # Leading drive to the fridge ends when the door starts moving
                # (not the last base nudge, which can overlap the door opening).
                t_start = self._door_start_time(door, cursor, max(t_open, cursor + 1))
                maybe_nav(t_start, NAV_TO_FRIDGE_LANG)
                if t_open > cursor:
                    emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
                # lemon pick (door is open, reach + grasp + depart)
                emit(t_dep, SUBTASK_LEMON, STAGE_PICK, LEMON_PICK_LANG)
                # lemon carry to the glass, then place
                maybe_nav(t_rel, NAV_TO_COUNTER_LANG)
                emit(t_rel, SUBTASK_LEMON, STAGE_PLACE, LEMON_PLACE_LANG)
            else:  # ice
                t_dep = max(t_dep_ice, cursor)
                t_rel = max(t_rel_ice, t_dep)
                maybe_nav(t_dep, NAV_TO_COUNTER_LANG)
                emit(t_dep, SUBTASK_ICE, STAGE_PICK, ICE_PICK_LANG)
                maybe_nav(t_rel, NAV_TO_COUNTER_LANG)
                emit(t_rel, SUBTASK_ICE, STAGE_PLACE, ICE_PLACE_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "used_cube": used,
            "order": np.array([0 if order[0] == "lemon" else 1]),
            "t_dep_lemon": t_dep_lemon,
            "t_rel_lemon": t_rel_lemon,
            "t_dep_ice": t_dep_ice,
            "t_rel_ice": t_rel_ice,
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
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        n_open = int(np.sum(names == SUBTASK_OPEN))
        if n_open != 1:
            issues.append(f"expected 1 OpenFridge subtask, found {n_open}")
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        present = set(str(n) for n in names)
        for required in (SUBTASK_LEMON, SUBTASK_ICE):
            if required not in present:
                issues.append(f"missing operation {required}")
        return issues
