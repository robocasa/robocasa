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

SUBTASK_MOVE = "PickPlaceCounterToStove"   # bank atomic (counter -> pot on stove)
SUBTASK_RETRIEVE = "PickUpSpatula"          # extra (no bank atomic for a bare pick)
SUBTASK_STIR = "StirVegetables"             # extra (no bank atomic for stirring)
SUBTASK_NAV = "NavigateKitchen"

RETRIEVE_LANG = "pick up the spatula from the counter"
STIR_LANG = "stir the vegetables in the pot"
NAV_TO_COUNTER_LANG = "navigate to the counter"
NAV_TO_STOVE_LANG = "navigate to the stove"
DONE_LANG = "task complete"


@register_labeler
class StirVegetablesLabeler(BaseSubtaskLabeler):
    """Labeler for the StirVegetables composite task.

    Two vegetables (`veg1`, `veg2`) and a spatula start on the counter; a pot sits
    on a lit stove burner. The robot puts both vegetables in the pot, retrieves the
    spatula, and stirs them. Four subtasks: two PickPlaceCounterToStove (the
    vegetables, in grasp order), PickUpSpatula, and StirVegetables.

    Signals off privileged sim state: each vegetable's grasp onset is its first
    contact-grasp (fallback lift/displace) and its place ends fully-ungrasped
    (later of the contact-grasp end +pad and the finger-width open). The spatula is
    grasped after both vegetables and held to the end, so its grasp onset ends the
    retrieve; the stir runs from there to the end, verified by the spatula<->veg
    contact bursts. Some layouts drive between the stove and the counter, so
    NavigateKitchen is inserted (to the counter for a pick, to the stove for a
    place / to stir) only when the base actually moves.
    """

    ENV_NAME = "StirVegetables"

    VEGS = ("veg1", "veg2")
    SPATULA = "spatula"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    STIR_PAD = 8

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65
    NAV_PICK_RESERVE = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.bids = {n: env.obj_body_id[n] for n in self.VEGS + (self.SPATULA,)}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.VEGS + (self.SPATULA,)}
        self.spatula_obj = env.objects[self.SPATULA]
        self.veg_objs = {n: env.objects[n] for n in self.VEGS}
        self.veg_lang = {}
        for n in self.VEGS:
            try:
                self.veg_lang[n] = env.get_obj_lang(n)
            except Exception:
                self.veg_lang[n] = "vegetable"
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.extra_subtasks = {SUBTASK_RETRIEVE, SUBTASK_STIR}
        # A successful stir needs only 5 stirring timesteps, so in a short demo
        # the stir can be brief; a 10-frame floor still catches real 0-2 frame
        # boundary collapses without flagging a legitimately quick stir.
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        try:
            stir = int(
                env.check_contact(self.spatula_obj, self.veg_objs["veg1"])
                or env.check_contact(self.spatula_obj, self.veg_objs["veg2"])
            )
        except Exception:
            stir = 0
        d = {
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "stir": stir,
        }
        for n in self.VEGS + (self.SPATULA,):
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
        stir = np.array([h["stir"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        pos = {n: np.array([h[f"pos_{n}"] for h in history]) for n in self.VEGS + (self.SPATULA,)}
        grasp = {n: np.array([h[f"grasp_{n}"] for h in history]) for n in self.VEGS + (self.SPATULA,)}

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Vegetables in grasp order.
        vg = {n: self._grasp_onset(grasp[n], pos[n]) for n in self.VEGS}
        order = sorted(self.VEGS, key=lambda n: vg[n])
        gs = self._grasp_onset(grasp[self.SPATULA], pos[self.SPATULA])  # spatula retrieve
        g_a, g_b = vg[order[0]], vg[order[1]]
        r_a = self._release_for(grasp[order[0]], pos[order[0]], fw, fw_open, g_a, max(g_b, g_a + 1))
        r_b = self._release_for(grasp[order[1]], pos[order[1]], fw, fw_open, g_b, max(gs, g_b + 1))

        # Stir: from the spatula grab to its last contact with a vegetable.
        sv = np.where(stir[gs:] > 0)[0]
        stir_touch0 = int(gs + int(sv[0])) if sv.size else gs
        stir_end = int(min(gs + int(sv[-1]) + self.STIR_PAD, T - 1)) if sv.size else T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang, reserve=0):
            b = max(cursor, end - reserve)
            nav_end = self._nav_end(base_speed, base_xy, cursor, b)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        for i, n in enumerate(order):
            gi = g_a if i == 0 else g_b
            ri = r_a if i == 0 else r_b
            maybe_nav(gi, NAV_TO_COUNTER_LANG, reserve=self.NAV_PICK_RESERVE)
            emit(gi, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.veg_lang[n]} from the counter")
            maybe_nav(ri, NAV_TO_STOVE_LANG)
            emit(ri, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.veg_lang[n]} in the pot")
        maybe_nav(gs, NAV_TO_COUNTER_LANG, reserve=self.NAV_PICK_RESERVE)
        emit(gs, SUBTASK_RETRIEVE, STAGE_PICK, RETRIEVE_LANG)
        maybe_nav(stir_touch0, NAV_TO_STOVE_LANG)
        emit(stir_end, SUBTASK_STIR, STAGE_EXECUTE, STIR_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "veg_grasp": [g_a, g_b],
            "veg_release": [r_a, r_b],
            "spatula_grasp": gs,
            "stir": [stir_touch0, stir_end],
            "stir_frames": int(stir.sum()),
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
        if n_pick != 2:
            issues.append(f"expected 2 vegetables moved to the pot, found {n_pick}")
        nameset = set(str(v) for v in names)
        for req in (SUBTASK_RETRIEVE, SUBTASK_STIR):
            if req not in nameset:
                issues.append(f"missing {req} subtask")
        if getattr(result, "debug", None) and result.debug.get("stir_frames", 0) < 3:
            issues.append("spatula barely contacted the vegetables (stir not detected)")
        return issues
