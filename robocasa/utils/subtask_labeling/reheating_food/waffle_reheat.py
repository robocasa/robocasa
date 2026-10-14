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

# All bank atomic tasks.
SUBTASK_OPEN = "OpenMicrowave"                 # open the microwave door
SUBTASK_MOVE = "PickPlaceCounterToMicrowave"   # bowl (with waffle) -> microwave
SUBTASK_CLOSE = "CloseMicrowave"               # close the door
SUBTASK_ON = "TurnOnMicrowave"                 # press the start button
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the microwave"
CLOSE_LANG = "close the microwave door"
ON_LANG = "turn on the microwave"
NAV_TO_COUNTER_LANG = "navigate to the counter with the waffle bowl"
NAV_TO_MICROWAVE_LANG = "navigate to the microwave"
DONE_LANG = "task complete"


@register_labeler
class WaffleReheatLabeler(BaseSubtaskLabeler):
    """Labeler for the WaffleReheat composite task.

    A waffle sits in a bowl (``waffle_container``) on the counter next to a
    microwave that spawns CLOSED. The robot opens the microwave, moves the bowl
    (waffle rides inside) into it, closes the door, and presses the start button
    -- the four subtasks are OpenMicrowave, PickPlaceCounterToMicrowave (pick +
    place), CloseMicrowave, TurnOnMicrowave.

    Signals off privileged sim state: the microwave door joint runs from 0
    (closed) to ~-1.4 (open) and back to 0; the open subtask ends when the door
    first reaches most of the way open, the close subtask ends when it returns to
    ~0. The bowl's grasp onset is its first contact-grasp (fallback lift/displace)
    and the place ends fully-ungrasped. The start button is a contact geom (the
    microwave's ``_turned_on`` is python-only state, not in the sim), so the
    turn-on ends at the first gripper/start-button contact. Some layouts put the
    bowl far enough to require a base drive, so NavigateKitchen is inserted (to the
    counter for the bowl, back to the microwave to place) only when the base moves.
    """

    ENV_NAME = "WaffleReheat"

    BOWL = "waffle_container"

    MOVE_TH = 0.10       # planar displacement above which the bowl counts as moved
    Z_LIFT = 0.03        # height the bowl rises to count as grasped
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    OPEN_FRAC = 0.8      # door reaches this fraction of its open extreme -> opened
    CLOSE_TH = -0.2      # door above this (toward 0) counts as closed again (some
                         # doors settle a little short of 0, ~-0.1/-0.15, but are
                         # "not open" enough for the microwave to start)
    NAV_PICK_RESERVE = 20  # leave >= this many frames for the pick after a drive
                         # to the counter (the robot may grab right as it arrives)

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.body_id = env.obj_body_id[self.BOWL]
        self.geoms = env.objects[self.BOWL].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        try:
            self.bowl_lang = env.get_obj_lang(self.BOWL)
        except Exception:
            self.bowl_lang = "bowl"
        mw = env.microwave
        self.mw_name = mw.name
        self.door_adr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
        self.start_geom = "{}_start_button".format(mw.name)
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Microwave ops (esp. the start-button press) can be quick; real
        # boundary collapses are 0-2 frames, so a 10-frame floor still catches
        # them while not flagging a legitimately fast press.
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        try:
            start = int(env.check_contact(self.gripper, self.start_geom))
        except Exception:
            start = 0
        return {
            "pos": env.sim.data.body_xpos[self.body_id].copy(),
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.geoms)),
            "door": float(env.sim.data.qpos[self.door_adr]),
            "start": start,
        }

    # ---- event helpers -------------------------------------------------

    def _grasp_onset(self, grasp_arr, obj):
        # The bowl only leaves the counter once grasped, so its lift is the most
        # reliable onset -- contact-grasp is sparse/late for some bowls (a brief
        # rim contact near release would fool a contact-first rule). Fall back to
        # planar displacement, then to the first contact.
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = np.where(obj[:, 2] > z0 + self.Z_LIFT)[0]
        if lifted.size:
            return int(lifted[0])
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1)
        w = np.where(disp > self.MOVE_TH)[0]
        if w.size:
            return int(w[0])
        gi = np.where(grasp_arr > 0)[0]
        return int(gi[0]) if gi.size else 0

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
        door = np.array([h["door"] for h in history])
        start = np.array([h["start"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Door: 0 (closed) -> ~-1.4 (open) -> 0 (closed). Open ends when the door
        # first reaches most of the way open; close ends when it returns to ~0.
        # Use the MEDIAN of the open-period as the open level (not the raw min):
        # some doors briefly dip deeper than their held-open plateau as the bowl
        # is set, and a 0.8*min threshold would then wait for that transient and
        # land the open boundary after the bowl is already placed.
        open_mask = door < -0.3
        door_level = float(np.median(door[open_mask])) if open_mask.any() else float(door.min())
        if door_level < -0.1:
            oo = np.where(door <= self.OPEN_FRAC * door_level)[0]
            open_done = int(oo[0]) if oo.size else 0
            pk = int(np.argmin(door))
            cc = np.where(door[pk:] > self.CLOSE_TH)[0]
            close_done = int(pk + int(cc[0])) if cc.size else T
        else:
            open_done, close_done = 0, T  # door never opened (anomaly -> flagged)

        # Bowl pick/place (grasp onset bounded to after the door opens; release
        # bounded to before the door closes).
        grasp0 = max(self._grasp_onset(grasp, pos), open_done)
        release = self._release_for(grasp, pos, fw, fw_open, grasp0, close_done)

        # Turn-on ends at the first gripper/start-button contact after the close.
        sc = np.where(start[close_done:] > 0)[0]
        turn_on_done = int(close_done + int(sc[0])) if sc.size else T

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang, reserve=0):
            # Stop the nav `reserve` frames short of `end` so the following
            # segment (the pick's reach + grab) isn't swallowed when the robot
            # grabs the instant it arrives at the counter.
            b = max(cursor, end - reserve)
            nav_end = self._nav_end(base_speed, base_xy, cursor, b)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        emit(open_done, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        maybe_nav(grasp0, NAV_TO_COUNTER_LANG, reserve=self.NAV_PICK_RESERVE)
        emit(grasp0, SUBTASK_MOVE, STAGE_PICK,
             f"pick up the {self.bowl_lang} with the waffle from the counter")
        maybe_nav(release, NAV_TO_MICROWAVE_LANG)
        emit(release, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.bowl_lang} with the waffle in the microwave")
        emit(close_done, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(turn_on_done, SUBTASK_ON, STAGE_EXECUTE, ON_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "open_done": open_done,
            "grasp0": grasp0,
            "release": release,
            "close_done": close_done,
            "turn_on_done": turn_on_done,
            "door_level": round(door_level, 3),
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
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != n_place:
            issues.append(f"pick/place mismatch: {n_pick} picks, {n_place} places")
        if n_pick != 1:
            issues.append(f"expected 1 bowl moved to the microwave, found {n_pick}")
        for req in (SUBTASK_OPEN, SUBTASK_CLOSE, SUBTASK_ON):
            if req not in names:
                issues.append(f"missing {req} subtask")
        return issues
