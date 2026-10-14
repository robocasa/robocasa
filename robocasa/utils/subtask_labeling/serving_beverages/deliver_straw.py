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

SUBTASK_OPEN = "OpenDrawer"                 # bank atomic
SUBTASK_MOVE = "PickPlaceDrawerToCounter"   # bank atomic (drawer -> glass cup)
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the drawer in front of the robot"
PICK_LANG = "pick up the straw from the drawer"
PLACE_LANG = "place the straw in the glass cup"
NAV_LANG = "navigate to the cup on the dining counter"
DONE_LANG = "task complete"


@register_labeler
class DeliverStrawLabeler(BaseSubtaskLabeler):
    """Labeler for the DeliverStraw composite task.

    A straw sits in a (closed) top drawer; a glass cup sits on the dining counter
    across the kitchen. The robot opens the drawer, picks the straw out, drives to
    the dining counter, and places the straw in the cup. Four subtasks: OpenDrawer,
    PickPlaceDrawerToCounter[pick], NavigateKitchen, PickPlaceDrawerToCounter[place]
    -- the pick and place are split by the always-present drive.

    Signals off privileged sim state: the drawer slide joint runs from 0 (closed)
    to ~-0.45 (open) and stays open (it is never closed); the open subtask ends
    when it first reaches most of the way out. The straw's grasp onset is its first
    contact-grasp (fallback lift/displace) and its place ends fully-ungrasped
    (later of the contact-grasp end +pad and the finger-width open). The drive from
    the drawer to the dining counter is inserted as NavigateKitchen whenever the
    base moves (essentially always here).
    """

    ENV_NAME = "DeliverStraw"

    STRAW = "straw"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    OPEN_FRAC = 0.8      # drawer reaches this fraction of its open extreme -> opened
    OPEN_PICK_RESERVE = 20  # keep >= this many frames for the pick when a deep/slow
                         # drawer only finishes opening as the straw is grabbed

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.straw_bid = env.obj_body_id[self.STRAW]
        self.straw_geoms = env.objects[self.STRAW].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.drawer_adr = env.sim.model.get_joint_qpos_addr(env.drawer.door_joint_names[0])
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "pos": env.sim.data.body_xpos[self.straw_bid].copy(),
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.straw_geoms)),
            "drawer": float(env.sim.data.qpos[self.drawer_adr]),
        }

    # ---- event helpers -------------------------------------------------

    def _grasp_onset(self, grasp_arr, obj):
        gi = np.where(grasp_arr > 0)[0]
        if gi.size:
            return int(gi[0])
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = np.where(obj[:, 2] > z0 + self.Z_LIFT)[0]
        if lifted.size:
            return int(lifted[0])
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1)
        w = np.where(disp > self.MOVE_TH)[0]
        return int(w[0]) if w.size else 0

    def _fw_open(self, fw, fw_open, grasp_t, hi):
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return None
        held = float(np.min(seg[: min(25, seg.size)]))
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return None

    def _release_for(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        fwo = self._fw_open(fw, fw_open, grasp_t, hi)
        gc = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
            if fwo is not None:
                rel = max(rel, fwo)
        elif fwo is not None:
            rel = fwo
        else:
            rel = setdown
        return int(min(max(rel, grasp_t + 1), hi))

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
        pos = np.array([h["pos"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        drawer = np.array([h["drawer"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Drawer: 0 (closed) -> ~-0.45 (open), stays open. Open ends when it first
        # reaches most of the way out; use the median of the open-period as the
        # open level (robust to any brief overshoot).
        open_mask = drawer < -0.05
        drawer_level = float(np.median(drawer[open_mask])) if open_mask.any() else float(drawer.min())
        if drawer_level < -0.05:
            oo = np.where(drawer <= self.OPEN_FRAC * drawer_level)[0]
            open_done = int(oo[0]) if oo.size else 0
        else:
            open_done = 0  # drawer never opened (anomaly -> flagged)

        raw_grasp = self._grasp_onset(grasp, pos)
        # A deep/slow drawer can finish opening right as the straw is grabbed;
        # pull the OpenDrawer boundary back so the pick keeps a real window (only
        # affects near-collapse cases -- normally the drawer opens long before).
        open_done = min(open_done, max(0, raw_grasp - self.OPEN_PICK_RESERVE))
        grasp0 = max(raw_grasp, open_done)
        release = self._release_for(grasp, pos, fw, fw_open, grasp0, T)

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

        emit(open_done, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(grasp0, SUBTASK_MOVE, STAGE_PICK, PICK_LANG)
        maybe_nav(release, NAV_LANG)
        emit(release, SUBTASK_MOVE, STAGE_PLACE, PLACE_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "open_done": open_done,
            "grasp0": grasp0,
            "release": release,
            "drawer_level": round(drawer_level, 3),
            "base_max": round(float(np.linalg.norm(base_xy - base_xy[0], axis=1).max()), 2)
            if base_xy is not None else 0.0,
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        move = names == SUBTASK_MOVE
        n_pick = int(np.sum(move & (stages == STAGE_PICK)))
        n_place = int(np.sum(move & (stages == STAGE_PLACE)))
        if n_pick != n_place:
            issues.append(f"pick/place mismatch: {n_pick} picks, {n_place} places")
        if n_pick != 1:
            issues.append(f"expected 1 straw delivered, found {n_pick}")
        if SUBTASK_OPEN not in set(str(v) for v in names):
            issues.append(f"missing {SUBTASK_OPEN} subtask")
        return issues
