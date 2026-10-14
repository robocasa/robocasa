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

# Each item is carried from the bowl to a plate (pick / navigate / place). Both
# sit on the dining counter, so this is the counter->counter bank extension.
SUBTASK_FOOD = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

NAV_BACK_LANG = "navigate back to the bowl"        # return between items
DONE_LANG = "task complete"


@register_labeler
class PortionHotDogsLabeler(BaseSubtaskLabeler):
    """Labeler for the PortionHotDogs composite task.

    Mobile task: two identical hot dog buns (`hotdog_bun1/2`) and two identical
    sausages (`sausage1/2`) start in a bowl on the dining counter; two plates sit
    on the same counter. Each plate must end with one bun and one sausage. The
    robot moves the four items one at a time; some layouts spread the plates far
    enough to require driving along the counter, others are within reach.

    Structure mirrors PackIdenticalLunches. Per item, in grasp order:
    PickPlaceCounterToCounter[pick] -> (NavigateKitchen carry, only when the base
    actually drives) -> PickPlaceCounterToCounter[place], with a NavigateKitchen
    back to the bowl before the next item, plus a terminal done.

    All four items share the bowl, so grabbing one nudges the others -- but a
    nudge leaves no grasp contact and no lift, so the grasp onset (which also
    fixes the handling order) is the item's first contact, falling back to its
    lift out of the bowl, then to its first real displacement. Displacements here
    are small (~0.2-2 m, plates can be close to the bowl), so a displacement-based
    order would be fooled by nudges; contact is the reliable anchor. The place
    ends when the gripper is fully ungrasped (the later of the contact-grasp end
    and the finger-width gripper-open, robust to a loose grasp's early contact).

    Placement is by TYPE, not pairing: an item may go on either plate, the only
    constraint being not to double a type on one plate. So the first bun (or
    sausage) is placed on "either plate" and the second on "the plate without a
    bun (or sausage)".
    """

    ENV_NAME = "PortionHotDogs"

    OBJS = ["hotdog_bun1", "hotdog_bun2", "sausage1", "sausage2"]
    PLATES = ["plate1", "plate2"]
    KIND = {"hotdog_bun1": "bun", "hotdog_bun2": "bun",
            "sausage1": "sausage", "sausage2": "sausage"}

    MOVE_TH = 0.10       # planar displacement above which an item counts as relocated
    Z_LIFT = 0.03        # height an item rises out of the bowl to count as grasped
    MOTION_TH = 0.004    # per-step item speed counting as "still moving"
    RELEASE_PAD = 8      # frames after contact ends for the fingers to fully open
    REL_FRAC = 0.5       # finger-width gripper-open level (release fallback)
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.body_ids = {n: env.obj_body_id[n] for n in self.OBJS + self.PLATES}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.OBJS}
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        # "bun" reads cleaner than the sim's "hotdog bun" (per user).
        self.bun_lang = "bun"
        try:
            self.sausage_lang = env.get_obj_lang("sausage1")
        except Exception:
            self.sausage_lang = "sausage"
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 15
        self.extra_subtasks = {SUBTASK_FOOD}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "pos": {n: env.sim.data.body_xpos[b].copy() for n, b in self.body_ids.items()},
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": {
                n: int(env._check_grasp(gripper=self.gripper, object_geoms=g))
                for n, g in self.geoms.items()
            },
        }

    # ---- event helpers -------------------------------------------------

    def _grasp_onset(self, grasp_arr, obj):
        """Frame the item is grasped, which also fixes the handling order.

        The item's first contact -- bump-proof, since nudging a bowl-mate leaves
        no grasp contact. Falls back to the lift out of the bowl (z rises past
        Z_LIFT) for a loose grasp with no contact, then to the first real
        displacement.
        """
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

    def _release(self, fw, fw_open, grasp_t, hi):
        """Finger-width gripper-open frame after the grasp, or None. The width is
        low while the item is held, then rises past the half-way point to fully
        open when released."""
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
        """Release (place end) = fully ungrasped: the later of the contact-grasp
        end (+ pad) and the finger-width gripper-open. Robust to a loose grasp's
        early-ending contact; the motion set-down is the last fallback."""
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        fwo = self._release(fw, fw_open, grasp_t, hi)
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
        pos = {n: np.array([h["pos"][n] for h in history]) for n in self.OBJS + self.PLATES}
        base_xy = np.array([h["base_xy"] for h in history])
        grasp = {n: np.array([h["grasp"][n] for h in history]) for n in self.OBJS}
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        plate_end = {pl: pos[pl][-1][:2] for pl in self.PLATES}
        info = {}
        for n in self.OBJS:
            p = pos[n]
            disp = float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max())
            if disp > self.MOVE_TH:
                de = {pl: float(np.linalg.norm(p[-1][:2] - plate_end[pl])) for pl in self.PLATES}
                dest = min(de, key=de.get)
                info[n] = dict(pos=p, kind=self.KIND[n], dest=dest,
                               grasp0=self._grasp_onset(grasp[n], p))
        order = sorted(info, key=lambda n: info[n]["grasp0"])

        prev_release = 0
        for i, n in enumerate(order):
            b = info[n]
            g = max(int(b["grasp0"]), prev_release)
            hi = info[order[i + 1]]["grasp0"] if i + 1 < len(order) else T
            hi = max(hi, g + 1)
            b["grasp"] = g
            b["release"] = self._release_for(grasp[n], b["pos"], fw, fw_open, g, hi)
            prev_release = b["release"]

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

        if order:
            # No leading nav: the robot spawns at the bowl (init_robot_here), so
            # the first segment is always the pick -- any initial base motion is
            # the reach for the first item, not a separate navigation.
            #
            # Placement is by TYPE, not pairing: an item may go on either plate;
            # the only constraint is not to double a type on one plate. So the
            # first bun (or sausage) goes on "either plate", and the second must
            # go on "the plate without a bun (or sausage)".
            placed = {"bun": 0, "sausage": 0}
            for i, n in enumerate(order):
                b = info[n]
                cat = self.bun_lang if b["kind"] == "bun" else self.sausage_lang
                if i > 0:
                    maybe_nav(b["grasp"], NAV_BACK_LANG)
                emit(b["grasp"], SUBTASK_FOOD, STAGE_PICK,
                     f"pick up a {cat} from the bowl")
                if placed[b["kind"]] == 0:
                    nav_lang = "navigate to either plate"
                    place_lang = f"place the {cat} on either plate"
                else:
                    nav_lang = f"navigate to the plate without a {cat}"
                    place_lang = f"place the {cat} on the plate without a {cat}"
                maybe_nav(b["release"], nav_lang)
                emit(b["release"], SUBTASK_FOOD, STAGE_PLACE, place_lang)
                placed[b["kind"]] += 1
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "dests": ",".join(f"{n}->{info[n]['dest']}" for n in order),
            "n_moved": len(order),
            "base_max": float(np.linalg.norm(base_xy - base_xy[0], axis=1).max())
            if base_xy is not None else 0.0,
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_FOOD}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != n_place:
            issues.append(f"pick/place mismatch: {n_pick} picks, {n_place} places")
        if not (1 <= n_pick <= 4):
            issues.append(f"expected 1-4 portioned items, found {n_pick}")
        return issues
