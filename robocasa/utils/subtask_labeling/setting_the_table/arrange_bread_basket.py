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

SUBTASK_OPEN = "OpenCabinet"                  # bank atomic
SUBTASK_BREAD = "PickPlaceCabinetToCounter"   # bank atomic (bread cabinet -> basket)
SUBTASK_BASKET = "PickPlaceCounterToCounter"  # extra (basket counter -> dining counter)
SUBTASK_NAV = "NavigateKitchen"

OPEN_ONE_LANG = "open the cabinet door above"
OPEN_MANY_LANG = "open the cabinet doors above"
BREAD_PLACE_LANG = "place the bread in the basket"
BASKET_PICK_LANG = "pick up the basket"
BASKET_PLACE_LANG = "place the basket on the dining counter"
NAV_LANG = "navigate to the dining counter"
DONE_LANG = "task complete"


@register_labeler
class ArrangeBreadBasketLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeBreadBasket composite task.

    Bread sits in a closed cabinet; a basket sits on a nearby counter; the dining
    counter is across the kitchen. The robot opens the cabinet, moves the bread
    into the basket, then carries the basket to the dining counter. Subtasks:
    OpenCabinet, PickPlaceCabinetToCounter[pick/place] (bread -> basket),
    PickPlaceCounterToCounter[pick] (basket), NavigateKitchen, [place] (basket ->
    dining counter).

    Signals off privileged sim state: the cabinet's door hinge(s) rotate open --
    a HingeCabinet has two hinges and only one opens, so the open amount is the MAX
    absolute angle across all of `cab.door_joint_names`. Each object's grasp onset
    is its first contact-grasp (fallback lift/displace) and its place ends
    fully-ungrasped (later of the contact-grasp end +pad and the finger-width open)
    -- robust to the basket's choppy contact-grasp (the fw-open dominates). The
    drive to the dining counter is NavigateKitchen, inserted while the basket is
    carried whenever the base moves.
    """

    ENV_NAME = "ArrangeBreadBasket"

    BREAD = "bread"
    BASKET = "basket"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10
    OPEN_FRAC = 0.9         # fallback: door reaches this fraction of its open angle
    DOOR_RISE_TH = 0.004    # per-frame hinge-angle increase that counts as "still
                            # swinging open" (used to find when a door stops opening)
    OPEN_PICK_RESERVE = 20  # keep a pick window if the door only finishes as the bread is grabbed
    BASKET_PICK_RESERVE = 12  # keep a basket-pick window for a fast bread->basket handoff

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.bread_bid = env.obj_body_id[self.BREAD]
        self.basket_bid = env.obj_body_id[self.BASKET]
        self.bread_geoms = env.objects[self.BREAD].contact_geoms
        self.basket_geoms = env.objects[self.BASKET].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.door_adrs = [
            env.sim.model.get_joint_qpos_addr(j) for j in env.cab.door_joint_names
        ]
        try:
            self.bread_lang = env.get_obj_lang(self.BREAD)
        except Exception:
            self.bread_lang = "bread"
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.extra_subtasks = {SUBTASK_BASKET}
        # Target demos are fast; the basket grab/set-down can be brief. A 10-frame
        # floor catches real 0-2 frame collapses without flagging a quick handoff.
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        door_vals = [abs(float(env.sim.data.qpos[a])) for a in self.door_adrs]
        return {
            "bread_pos": env.sim.data.body_xpos[self.bread_bid].copy(),
            "basket_pos": env.sim.data.body_xpos[self.basket_bid].copy(),
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "bread_grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.bread_geoms)),
            "basket_grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.basket_geoms)),
            "door_vals": door_vals,
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
        bread_pos = np.array([h["bread_pos"] for h in history])
        basket_pos = np.array([h["basket_pos"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        bread_grasp = np.array([h["bread_grasp"] for h in history])
        basket_grasp = np.array([h["basket_grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        # Per-hinge open angles -> open amount (max across hinges) + how many hinges
        # actually opened (a hinge cabinet has two doors, often only one is opened).
        door_vals = np.array([h["door_vals"] for h in history])
        n_joints = door_vals.shape[1] if door_vals.ndim == 2 else 0

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Cabinet door: 0 (closed) -> ~1.3-1.57 (open), stays open. Some layouts use
        # an OpenCabinet fixture (open shelf, no door joints) -> nothing to open, so
        # OpenCabinet is emitted only when a door actually opens. A hinge cabinet
        # opens its two doors ONE AFTER ANOTHER, so OpenCabinet ends when the LAST
        # door is open: take the per-hinge open times (each hinge reaching 0.8 of
        # its own open level) and use the max.
        n_opened = 0
        door_level = 0.0
        open_times = []
        for j in range(n_joints):
            col = door_vals[:, j]
            om = col > 0.2
            if not om.any():
                continue
            n_opened += 1
            level_j = float(np.median(col[om]))
            door_level = max(door_level, level_j)
            # This hinge is done opening at the LAST frame it is still swinging open
            # (its angle still rising). A door creeps the final few percent slowly,
            # so a fraction-of-open threshold ends the subtask a beat early.
            rising = np.where(np.diff(col) > self.DOOR_RISE_TH)[0]
            if rising.size:
                open_times.append(int(rising[-1]) + 2)
            else:
                ot = np.where(col >= self.OPEN_FRAC * level_j)[0]
                open_times.append(int(ot[0]) if ot.size else 0)
        has_open = n_opened > 0
        open_done = int(min(max(open_times), T - 1)) if open_times else 0

        bread_g_raw = self._grasp_onset(bread_grasp, bread_pos)  # bread contact is clean
        if has_open:
            # Reserve a pick window if the door only finishes opening as the bread
            # is grabbed (mirrors DeliverStraw).
            open_done = min(open_done, max(0, bread_g_raw - self.OPEN_PICK_RESERVE))
        bread_g = max(bread_g_raw, open_done)

        # Basket carry onset = its first SUSTAINED displacement from rest (3D). NOT
        # a first-lift rule: placing the bread transiently bumps the basket up a
        # little, which a lift rule reads as the grab and collapses the pick. A
        # sustained displacement rejects that bump and works whether the basket is
        # lifted or slid.
        basket_disp = np.linalg.norm(basket_pos - basket_pos[0], axis=1)
        bmoving = basket_disp > self.MOVE_TH
        basket_move = T
        for t in range(T - 2):
            if bmoving[t] and bmoving[t + 1] and bmoving[t + 2]:
                basket_move = t
                break
        # In some demos the basket never moves -- the operator carries the BREAD to
        # a basket that already sits by the dining counter instead of moving the
        # basket. Then there are no basket subtasks; the bread is carried over.
        basket_moves = basket_move < T

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

        if has_open:
            open_lang = OPEN_MANY_LANG if n_opened >= 2 else OPEN_ONE_LANG
            emit(open_done, SUBTASK_OPEN, STAGE_EXECUTE, open_lang)

        pick_bread_lang = f"pick up the {self.bread_lang} from the cabinet"
        if basket_moves:
            bread_rel = self._release_for(
                bread_grasp, bread_pos, fw, fw_open, bread_g, max(basket_move, bread_g + 1)
            )
            # Reserve a basket-pick window for a fast bread->basket handoff (the
            # gripper is already at the basket, so grab-and-carry looks instant).
            bread_rel = min(bread_rel, max(bread_g + 1, basket_move - self.BASKET_PICK_RESERVE))
            basket_g = max(basket_move, bread_rel + 1)
            basket_rel = self._release_for(basket_grasp, basket_pos, fw, fw_open, basket_g, T)
            emit(bread_g, SUBTASK_BREAD, STAGE_PICK, pick_bread_lang)
            emit(bread_rel, SUBTASK_BREAD, STAGE_PLACE, BREAD_PLACE_LANG)
            emit(basket_g, SUBTASK_BASKET, STAGE_PICK, BASKET_PICK_LANG)
            maybe_nav(basket_rel, NAV_LANG)
            emit(basket_rel, SUBTASK_BASKET, STAGE_PLACE, BASKET_PLACE_LANG)
        else:
            bread_rel = self._release_for(bread_grasp, bread_pos, fw, fw_open, bread_g, T)
            basket_g = basket_rel = None
            emit(bread_g, SUBTASK_BREAD, STAGE_PICK, pick_bread_lang)
            maybe_nav(bread_rel, NAV_LANG)   # carry the bread to the far basket
            emit(bread_rel, SUBTASK_BREAD, STAGE_PLACE, BREAD_PLACE_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "has_open": has_open,
            "n_opened": n_opened,
            "basket_moves": basket_moves,
            "open_done": open_done,
            "bread": [bread_g, bread_rel],
            "basket": [basket_g, basket_rel],
            "door_level": round(door_level, 2),
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
        dbg = getattr(result, "debug", None) or {}
        checks = [(SUBTASK_BREAD, "bread")]
        if dbg.get("basket_moves", True):
            checks.append((SUBTASK_BASKET, "basket"))
        for sub, label in checks:
            m = names == sub
            npi = int(np.sum(m & (stages == STAGE_PICK)))
            npl = int(np.sum(m & (stages == STAGE_PLACE)))
            if npi != 1 or npl != 1:
                issues.append(f"{label}: expected 1 pick + 1 place, found {npi} + {npl}")
        if dbg.get("has_open", True) and SUBTASK_OPEN not in set(str(v) for v in names):
            issues.append(f"missing {SUBTASK_OPEN} subtask")
        return issues
