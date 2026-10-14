import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

SUBTASK_FOOD = "PickPlaceCounterToCounter"   # extra (chicken/veg plate -> bowl)
SUBTASK_BOWL = "PickPlaceCounterToFridge"    # extra (bowl counter -> fridge)
SUBTASK_NAV = "NavigateKitchen"

BOWL_PICK_LANG = "pick up the bowl"
BOWL_PLACE_LANG = "place the bowl in the fridge"
NAV_LANG = "navigate to the fridge"
DONE_LANG = "task complete"


@register_labeler
class StoreLeftoversInBowlLabeler(BaseSubtaskLabeler):
    """Labeler for the StoreLeftoversInBowl composite task.

    A chicken drumstick and a vegetable sit on plates on the dining counter, with a
    bowl next to them (the robot spawns at the bowl). The robot moves both foods
    into the bowl, then carries the bowl into the (setup-opened) fridge. Subtasks:
    PickPlaceCounterToCounter for each food (plate -> bowl, in grasp order) and
    PickPlaceCounterToFridge for the bowl (counter -> fridge). The fridge is opened
    at scene setup and never closed, so there is no fridge open/close subtask.

    Detection off privileged sim state: each food's grasp onset is its first
    contact-grasp (fallback lift/displace) and its place ends fully-ungrasped. The
    bowl's carry onset is its first SUSTAINED displacement (from its post-fill rest
    position) -- robust to the foods bumping it and to its choppy rim contact-grasp.
    NavigateKitchen is inserted while the bowl is carried to the fridge.
    """

    ENV_NAME = "StoreLeftoversInBowl"

    CHICKEN = "chicken_drumstick"
    VEG = "vegetable"
    BOWL = "bowl"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.objs = (self.CHICKEN, self.VEG, self.BOWL)
        self.bids = {n: env.obj_body_id[n] for n in self.objs}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.objs}
        try:
            self.veg_lang = env.get_obj_lang(self.VEG)
        except Exception:
            self.veg_lang = "vegetable"
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.extra_subtasks = {SUBTASK_FOOD, SUBTASK_BOWL}
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        d = {
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
        }
        for n in self.objs:
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
        pos = {n: np.array([h[f"pos_{n}"] for h in history]) for n in self.objs}
        grasp = {n: np.array([h[f"grasp_{n}"] for h in history]) for n in self.objs}

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        food_name = {self.CHICKEN: "chicken drumstick", self.VEG: self.veg_lang}
        foods = (self.CHICKEN, self.VEG)
        g = {n: self._grasp_onset(grasp[n], pos[n]) for n in foods}
        order = sorted(foods, key=lambda n: g[n])
        g0, g1 = g[order[0]], g[order[1]]

        # Bowl carry onset: first SUSTAINED displacement from its resting spot,
        # measured after the second food is placed (the foods bump the bowl a
        # little, and its rim contact-grasp is choppy).
        bowl_g_raw = self._grasp_onset(grasp[self.BOWL], pos[self.BOWL])
        r0 = self._release_for(grasp[order[0]], pos[order[0]], fw, fw_open, g0, max(g1, g0 + 1))
        r1 = self._release_for(grasp[order[1]], pos[order[1]], fw, fw_open, g1, max(bowl_g_raw, g1 + 1))
        ref = pos[self.BOWL][min(r1, T - 1)]
        bowl_disp = np.linalg.norm(pos[self.BOWL] - ref, axis=1)
        bmoving = bowl_disp > self.MOVE_TH
        bowl_move = T
        for t in range(r1, T - 2):
            if bmoving[t] and bmoving[t + 1] and bmoving[t + 2]:
                bowl_move = t
                break
        bowl_g = max(bowl_move, r1 + 1)
        bowl_rel = self._release_for(grasp[self.BOWL], pos[self.BOWL], fw, fw_open, bowl_g, T)

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

        rels = {order[0]: r0, order[1]: r1}
        for i, n in enumerate(order):
            # The two foods sit on two different plates; the second pick is from
            # "the other plate" to distinguish it.
            plate = "the other plate" if i == 1 else "the plate"
            emit(g[n], SUBTASK_FOOD, STAGE_PICK, f"pick up the {food_name[n]} from {plate}")
            emit(rels[n], SUBTASK_FOOD, STAGE_PLACE, f"place the {food_name[n]} in the bowl")
        emit(bowl_g, SUBTASK_BOWL, STAGE_PICK, BOWL_PICK_LANG)
        maybe_nav(bowl_rel, NAV_LANG)
        emit(bowl_rel, SUBTASK_BOWL, STAGE_PLACE, BOWL_PLACE_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "food_grasp": [g0, g1],
            "food_release": [r0, r1],
            "bowl": [bowl_g, bowl_rel],
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
        food = names == SUBTASK_FOOD
        nfp = int(np.sum(food & (stages == STAGE_PICK)))
        nfl = int(np.sum(food & (stages == STAGE_PLACE)))
        if nfp != 2 or nfl != 2:
            issues.append(f"foods: expected 2 picks + 2 places, found {nfp} + {nfl}")
        bowl = names == SUBTASK_BOWL
        nbp = int(np.sum(bowl & (stages == STAGE_PICK)))
        nbl = int(np.sum(bowl & (stages == STAGE_PLACE)))
        if nbp != 1 or nbl != 1:
            issues.append(f"bowl: expected 1 pick + 1 place, found {nbp} + {nbl}")
        return issues
