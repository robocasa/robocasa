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
SUBTASK_VEG = "PickPlaceSinkToCounter"        # vegetable sink -> bowl
SUBTASK_BOWL = "PickPlaceCounterToMicrowave"  # bowl counter -> microwave
SUBTASK_CLOSE = "CloseMicrowave"
SUBTASK_ON = "TurnOnMicrowave"
SUBTASK_NAV = "NavigateKitchen"

VEG_PLACE_LANG = "place the {} in the bowl"
BOWL_PICK_LANG = "pick up the bowl"
BOWL_PLACE_LANG = "place the bowl in the microwave"
CLOSE_LANG = "close the microwave door"
ON_LANG = "turn on the microwave"
NAV_TO_MICROWAVE_LANG = "navigate to the microwave"
DONE_LANG = "task complete"


@register_labeler
class SteamInMicrowaveLabeler(BaseSubtaskLabeler):
    """Labeler for the SteamInMicrowave composite task.

    A vegetable sits in the sink, a bowl on the counter, and a microwave that
    spawns OPEN. The robot moves the vegetable into the bowl, carries the bowl into
    the microwave, closes the door, and presses the start button. Subtasks (all
    bank atomics): PickPlaceSinkToCounter (veg -> bowl), PickPlaceCounterToMicrowave
    (bowl -> microwave), CloseMicrowave, TurnOnMicrowave. The microwave is opened at
    scene setup, so there is no open-microwave subtask.

    Signals off privileged sim state: each object's grasp onset is its first
    contact-grasp (fallback lift/displace) and its place ends fully-ungrasped
    (later of the contact-grasp end +pad and the finger-width open). The microwave
    door joint runs from ~-1.4 (open, at spawn) to 0 (closed); close ends when it
    returns to ~0 after the last clearly-open frame. The start button is a contact
    geom (`{name}_start_button`), so the turn-on ends at the first gripper/button
    contact. NavigateKitchen is inserted while the bowl is carried to the microwave.
    """

    ENV_NAME = "SteamInMicrowave"

    VEG = "vegetable"
    BOWL = "bowl"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    CLOSE_TH = -0.2      # door above this (toward 0) counts as closed

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.bids = {n: env.obj_body_id[n] for n in (self.VEG, self.BOWL)}
        self.geoms = {n: env.objects[n].contact_geoms for n in (self.VEG, self.BOWL)}
        mw = env.microwave
        self.door_adr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
        self.start_geom = "{}_start_button".format(mw.name)
        try:
            self.veg_lang = env.get_obj_lang(self.VEG)
        except Exception:
            self.veg_lang = "vegetable"
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
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
        d = {
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "door": float(env.sim.data.qpos[self.door_adr]),
            "start": start,
        }
        for n in (self.VEG, self.BOWL):
            d[f"pos_{n}"] = env.sim.data.body_xpos[self.bids[n]].copy()
            d[f"grasp_{n}"] = int(env._check_grasp(gripper=self.gripper, object_geoms=self.geoms[n]))
        return d

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
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        base_xy = np.array([h["base_xy"] for h in history])
        door = np.array([h["door"] for h in history])
        start = np.array([h["start"] for h in history])
        pos = {n: np.array([h[f"pos_{n}"] for h in history]) for n in (self.VEG, self.BOWL)}
        grasp = {n: np.array([h[f"grasp_{n}"] for h in history]) for n in (self.VEG, self.BOWL)}

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        veg_g = self._grasp_onset(grasp[self.VEG], pos[self.VEG])
        bowl_g_raw = self._grasp_onset(grasp[self.BOWL], pos[self.BOWL])
        veg_rel = self._release_for(
            grasp[self.VEG], pos[self.VEG], fw, fw_open, veg_g, max(bowl_g_raw, veg_g + 1)
        )
        bowl_g = max(bowl_g_raw, veg_rel + 1)

        # Door starts open (~-1.4) and is closed once. Close ends when the door
        # returns to ~0 after the LAST clearly-open frame (bowl placement can nudge
        # it a bit more open first). Bound the bowl's release before the close.
        open_frames = np.where(door < -0.5)[0]
        last_open = int(open_frames[-1]) if open_frames.size else 0
        cc = np.where(door[last_open:] > self.CLOSE_TH)[0]
        door_closed = int(last_open + int(cc[0])) if cc.size else T - 1

        bowl_rel = self._release_for(
            grasp[self.BOWL], pos[self.BOWL], fw, fw_open, bowl_g, max(last_open, bowl_g + 1)
        )
        door_closed = max(door_closed, bowl_rel + 1)

        sc = np.where(start[door_closed:] > 0)[0]
        turn_on_done = int(door_closed + int(sc[0])) if sc.size else T - 1
        turn_on_done = min(max(turn_on_done, door_closed + 1), T - 1)

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

        emit(veg_g, SUBTASK_VEG, STAGE_PICK, f"pick up the {self.veg_lang} from the sink")
        emit(veg_rel, SUBTASK_VEG, STAGE_PLACE, VEG_PLACE_LANG.format(self.veg_lang))
        emit(bowl_g, SUBTASK_BOWL, STAGE_PICK, BOWL_PICK_LANG)
        maybe_nav(bowl_rel, NAV_TO_MICROWAVE_LANG)
        emit(bowl_rel, SUBTASK_BOWL, STAGE_PLACE, BOWL_PLACE_LANG)
        emit(door_closed, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(turn_on_done, SUBTASK_ON, STAGE_EXECUTE, ON_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "veg": [veg_g, veg_rel],
            "bowl": [bowl_g, bowl_rel],
            "door_closed": door_closed,
            "turn_on_done": turn_on_done,
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
        for sub, label in ((SUBTASK_VEG, "vegetable"), (SUBTASK_BOWL, "bowl")):
            m = names == sub
            npi = int(np.sum(m & (stages == STAGE_PICK)))
            npl = int(np.sum(m & (stages == STAGE_PLACE)))
            if npi != 1 or npl != 1:
                issues.append(f"{label}: expected 1 pick + 1 place, found {npi} + {npl}")
        nameset = set(str(v) for v in names)
        for req in (SUBTASK_CLOSE, SUBTASK_ON):
            if req not in nameset:
                issues.append(f"missing {req} subtask")
        return issues
