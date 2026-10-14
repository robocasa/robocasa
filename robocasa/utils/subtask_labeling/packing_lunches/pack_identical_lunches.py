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

# Each food item is carried from the fridge to a tupperware on the counter
# (pick / navigate / place). No bank name covers fridge-shelf -> counter, so it
# is a declared bank extension (matches the other fridge-pick labelers).
SUBTASK_FOOD = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_FRIDGE_LANG = "navigate to the fridge"      # leading drive (rarely fires)
NAV_BACK_LANG = "navigate back to the fridge"      # return drive between items
DONE_LANG = "task complete"


@register_labeler
class PackIdenticalLunchesLabeler(BaseSubtaskLabeler):
    """Labeler for the PackIdenticalLunches composite task.

    Mobile task: two identical vegetables (`vegetable0/1`) and two identical
    meats (`meat0/1`) sit in the (already-open) fridge; two tupperwares sit on a
    counter across the kitchen. Each tupperware must end with one vegetable and
    one meat. The robot starts at the fridge and packs the four items one at a
    time, driving between the fridge and the counter on every trip.

    All four items are relocated; the labeler emits, per item in handling order:
      PickPlaceFridgeToCounter[pick] -> NavigateKitchen (carry to the counter)
      -> PickPlaceFridgeToCounter[place], with a NavigateKitchen back to the
      fridge before the next item. Plus a terminal done. (No OpenFridge subtask:
      the env spawns the fridge already open.)

    Each item is grasped, carried, and placed; the handling order is the grasp
    order. The grasp onset is anchored on the item's *carry* (first displacement
    past BIG_TOL, which a shared-spot nudge never reaches) and refined to the
    first contact in a window before it -- or, for a loose grasp with no contact,
    the item's lift off the shelf. The place ends when the gripper is fully
    ungrasped: the later of the contact-grasp end and the finger-width
    gripper-open (robust to an early-ending contact), falling back to the motion
    set-down.

    Carry/place **language is relational**: each item's destination tupperware is
    the nearest one to its final resting position, and the wording describes that
    tupperware by what it already holds at that moment (a fresh tupperware, the
    one with the partner, the empty one, either one still holding only the
    partner, or the unique one holding only the partner) -- the "identical
    lunches" narrative. Genuine anomalies (a multi-grab of two items in one grasp,
    or a grasp/place the signals miss entirely) are hand-authored overrides.
    """

    ENV_NAME = "PackIdenticalLunches"

    OBJS = ["vegetable0", "vegetable1", "meat0", "meat1"]
    TUPS = ["tupperware0", "tupperware1"]
    KIND = {"vegetable0": "veg", "vegetable1": "veg", "meat0": "meat", "meat1": "meat"}

    MOVE_TH = 0.10       # planar displacement above which an item counts as relocated
    BIG_TOL = 0.5        # displacement marking a real carry (anchors the grasp; a
                         # bump of a shared-spot item never reaches this)
    GRASP_BACK = 100     # how far before the carry the grasp contact may begin
    Z_LIFT = 0.03        # height an item rises off the shelf to count as grasped
                         # (loose-grasp fallback when contact never fires)
    MOTION_TH = 0.004    # per-step item speed counting as "still moving"
    RELEASE_PAD = 8      # frames after contact ends for the fingers to fully open
    REL_FRAC = 0.5       # finger-width gripper-open level (release fallback)
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.body_ids = {n: env.obj_body_id[n] for n in self.OBJS + self.TUPS}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.OBJS}
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        try:
            self.veg_lang = env.get_obj_lang("vegetable0")
            self.meat_lang = env.get_obj_lang("meat0")
        except Exception:
            self.veg_lang, self.meat_lang = "vegetable", "meat"
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

        Anchored on the item's *carry* -- the first frame its displacement
        exceeds BIG_TOL. Two items share a fridge spot, so grabbing one nudges
        the other; a small-displacement departure would order the nudged item far
        too early, but a nudge never reaches BIG_TOL. The grasp is then the first
        contact in a window before the carry (the contact-grasp can fragment into
        blips, so take the first touch, not the first sustained run). Falls back
        to the carry onset itself when contact misses the item entirely.
        """
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1)
        w = np.where(disp > self.BIG_TOL)[0]
        depart = int(w[0]) if w.size else 0
        lo = max(0, depart - self.GRASP_BACK)
        gi = np.where(grasp_arr > 0)[0]
        gi = gi[gi >= lo]
        if gi.size:
            return int(gi[0])
        # Loose grasp -> no contact ever fires. The item is still lifted off the
        # shelf to be carried, so the first rise in height marks the grasp (a bump
        # never lifts it).
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = np.where(obj[lo:depart + 1, 2] > z0 + self.Z_LIFT)[0]
        return int(lo + lifted[0]) if lifted.size else depart

    def _release(self, fw, fw_open, grasp_t, hi, fallback):
        """Finger-width fallback for the release when the contact check misses.

        The finger width is low while the item is held, then rises back toward
        fully-open when released; the release is the first sustained rise past the
        half-way point between this item's held width and fully open. Used only
        when contact-grasp never fires for the item. Falls back to the motion
        set-down if the gripper never reopens inside the window.
        """
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return fallback
        held = float(np.min(seg[: min(25, seg.size)]))
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return fallback

    def _release_for(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
        """Release (place end) = the gripper fully ungrasped, bounded to [grasp_t,
        hi). Contact-grasp end + a pad is the primary signal; the finger-width
        gripper-open and then the motion set-down are fallbacks. Not pulled to the
        set-down, which a later bump of the placed item can corrupt.
        """
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        # Finger-width gripper-open frame -- the true "fully ungrasped" moment,
        # robust to both an early-ending contact (loose grasp) and to a later
        # jiggle/bump that would corrupt the motion set-down.
        fwo = self._release(fw, fw_open, grasp_t, hi, None)
        gc = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
            if fwo is not None:
                rel = max(rel, fwo)  # loose grasp: contact ends before the open
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
        pos = {n: np.array([h["pos"][n] for h in history]) for n in self.OBJS + self.TUPS}
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

        # Name the two tupperwares left/right by world-x; assign each item to the
        # tupperware nearest its final resting position.
        tup_x = {tp: float(np.mean(pos[tp][:, 0])) for tp in self.TUPS}
        lr = sorted(self.TUPS, key=lambda tp: tup_x[tp])
        tup_side = {lr[0]: "left", lr[1]: "right"}
        tup_end = {tp: pos[tp][-1][:2] for tp in self.TUPS}

        info = {}
        for n in self.OBJS:
            p = pos[n]
            disp = float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max())
            if disp > self.MOVE_TH:
                de = {tp: float(np.linalg.norm(p[-1][:2] - tup_end[tp])) for tp in self.TUPS}
                dest = min(de, key=de.get)
                info[n] = dict(pos=p, kind=self.KIND[n], dest=dest,
                               grasp0=self._grasp_onset(grasp[n], p))
        # Handling order = grasp order (bump-proof; see _grasp_onset).
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
            # The robot starts at the fridge, so the leading drive is usually
            # absent (collapses to nothing when the base doesn't move).
            maybe_nav(info[order[0]]["grasp"], NAV_TO_FRIDGE_LANG)
            # Carry/place language is relational -- each tupperware is described
            # by what it already holds at that moment (the "identical lunches"
            # narrative), not by a fixed left/right.
            contents = {tp: [] for tp in self.TUPS}
            for i, n in enumerate(order):
                b = info[n]
                cat = self.veg_lang if b["kind"] == "veg" else self.meat_lang
                dest = b["dest"]
                other = self.TUPS[1] if dest == self.TUPS[0] else self.TUPS[0]
                if i > 0:
                    maybe_nav(b["grasp"], NAV_BACK_LANG)
                emit(b["grasp"], SUBTASK_FOOD, STAGE_PICK,
                     f"pick up a {cat} from the fridge")

                if not contents[dest]:
                    # First item into this tupperware.
                    if not contents[other]:
                        nav_lang = "navigate to a tupperware on the counter"
                        place_lang = f"place the {cat} in a tupperware"
                    else:
                        nav_lang = "navigate to the empty tupperware on the counter"
                        place_lang = f"place the {cat} in the empty tupperware"
                else:
                    # Second item -> pairs with the partner already inside.
                    partner = contents[dest][0]
                    if partner not in contents[other]:
                        # Only this tupperware holds the partner.
                        nav_lang = f"navigate to the tupperware with the {partner}"
                        place_lang = f"place the {cat} in the same tupperware as the {partner}"
                    elif len(contents[other]) == 1:
                        # Both tupperwares hold ONLY the partner -> either is valid.
                        nav_lang = f"navigate to either tupperware with {partner} in it"
                        place_lang = f"place the {cat} in either tupperware with {partner} in it"
                    else:
                        # The other tupperware is already full, so this one is the
                        # unique tupperware holding only the partner.
                        nav_lang = f"navigate to the tupperware with only the {partner} in it"
                        place_lang = f"place the {cat} in the tupperware with only {partner} in it"

                maybe_nav(b["release"], nav_lang)
                emit(b["release"], SUBTASK_FOOD, STAGE_PLACE, place_lang)
                contents[dest].append(cat)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "dests": ",".join(f"{n}->{tup_side[info[n]['dest']]}" for n in order),
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
        # All four items should be packed; allow a short demo to pack fewer.
        if not (1 <= n_pick <= 4):
            issues.append(f"expected 1-4 packed items, found {n_pick}")
        return issues
