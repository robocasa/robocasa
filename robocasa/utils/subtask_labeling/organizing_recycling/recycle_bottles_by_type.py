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

# Each relocated bottle is carried along the dining counter to its type's
# cluster (pick / navigate / place). No bank name covers counter -> counter, so
# it is a declared extension.
SUBTASK_BOTTLE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_COUNTER_LANG = "navigate to the dining counter"
NAV_BACK_LANG = "navigate back to the middle"
DONE_LANG = "task complete"

# ep_meta obj_groups -> bottle type. "bottled_water" is plastic, "alcohol" glass.
GROUP_TYPE = {"bottled_water": "plastic", "alcohol": "glass"}


@register_labeler
class RecycleBottlesByTypeLabeler(BaseSubtaskLabeler):
    """Labeler for the RecycleBottlesByType composite task.

    Task (mobile): three bottles sit in the middle of the dining counter -- a
    plastic one (`bottle_plastic_middle`), a glass one (`bottle_glass_middle`),
    and a "mystery" bottle that is either. Each must end within the cluster of its
    type (the plastic cluster at one end of the counter, the glass cluster at the
    other). The robot starts at the middle.

    **A variable number of bottles is actually relocated, and not necessarily the
    middle ones.** The success check only requires each middle bottle be within
    0.30 of a same-type cluster sample, so in compact layouts a middle bottle can
    spawn already-clustered and is left alone -- and in a sizeable fraction of
    demos the robot *also* rearranges the end-cluster bottles
    (`bottle_plastic1/2`, `bottle_glass1/2`), e.g. consolidating a type. The
    labeler therefore watches **all seven** bottles and relocates whichever ones
    genuinely move (max planar displacement > `MOVE_TH`); an already-sorted bottle
    gets no segment. Tracking only the three middles silently mis-segments the
    end-cluster demos.

    Detection is by **displacement**, which is the only fully reliable signal
    here: every relocated bottle definitively changes position, whereas both the
    contact-grasp check and the gripper finger-width flicker or miss a bottle
    outright (in some demos each misses a *different* bottle, so neither alone is
    trustworthy). Each bottle's TYPE is read from the **ground-truth** object
    group in `ep_meta` (`bottled_water` -> plastic, `alcohol` -> glass), which
    also classifies the mystery bottle; nearest-cluster inference is unreliable
    because in compact layouts the two cluster centers can be ~0.1 apart.

    Emitted fine-grained, auto-ordered by which bottle departs first: per
    relocated bottle a PickPlaceCounterToCounter [pick] -> NavigateKitchen (carry
    to the cluster) -> PickPlaceCounterToCounter [place], with a NavigateKitchen
    back to the middle before the next bottle and a leading drive to the dining
    counter. Pick ends at the grasp (contact-grasp onset); place ends when the
    gripper is fully open again (the bottle is let go) -- detected from the
    finger width, which stays reliable even when the contact-grasp check misses
    the bottle. Boundaries fall back to the bottle's own motion (departure /
    set-down) when the grasp/release signals are unavailable.
    """

    ENV_NAME = "RecycleBottlesByType"

    MIDDLE = ["bottle_plastic_middle", "bottle_glass_middle", "mystery_middle"]
    ENDS = ["bottle_plastic1", "bottle_plastic2", "bottle_glass1", "bottle_glass2"]
    ALL = MIDDLE + ENDS

    MOVE_TH = 0.06       # max displacement above which a bottle counts as relocated
    DEPART_TOL = 0.04    # displacement at which a relocated bottle's move begins
    MOTION_TH = 0.004    # per-step bottle speed counting as "still moving"
    GRASP_BACK = 60      # how far before departure a grasp contact may begin
    RELEASE_PAD = 8      # frames after contact ends for the fingers to fully open
    REL_FRAC = 0.5       # finger-width gripper-open level (release fallback), as a
                         # fraction of the gap from the held width to fully open
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.body_ids = {n: env.obj_body_id[n] for n in self.ALL}
        self.geoms = {n: env.objects[n].contact_geoms for n in self.ALL}
        self.gripper = env.robots[0].gripper["right"]
        # Finger-width joint addresses: sum of |qpos| = how open the gripper is.
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        # Ground-truth type per bottle from its object group in ep_meta.
        groups = {c["name"]: c.get("obj_groups") for c in ep_meta.get("object_cfgs", [])}
        self.btype = {n: GROUP_TYPE.get(groups.get(n), "plastic") for n in self.ALL}
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 15
        self.extra_subtasks = {SUBTASK_BOTTLE}
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

    def _depart_time(self, obj, lo, T):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        cand = np.where(moved[lo:T])[0]
        return int(lo + int(cand[0])) if cand.size else lo

    def _release(self, fw, fw_open, grasp_t, hi, fallback):
        """Finger-width fallback for the release when the contact check misses.

        Within the grasp window the finger width is low while the bottle is held,
        then rises back toward fully-open when released; the release is the first
        sustained rise past the half-way point between this bottle's held width
        and fully open. Used only when contact-grasp never fires for the bottle
        (so the contact-end release is unavailable). Falls back to the motion
        set-down if the gripper never reopens inside the window. NOTE: finger
        width can lag well past the true release when the gripper does not fully
        reopen before reaching for the next bottle, which is why the contact-end
        is preferred whenever it exists.
        """
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return fallback
        held = float(np.min(seg[: min(25, seg.size)]))  # finger width while holding
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return fallback

    def _events(self, grasp, obj, fw, fw_open, depart, lo, hi):
        """Grasp (pick end) and release (place end) for one relocated bottle.

        The grasp onset uses the contact-grasp check (first contact near the
        bottle's departure), falling back to the departure when contact misses.
        The release ("fully ungrasped") is the contact-grasp end plus a small pad
        for the fingers to finish opening -- the most direct let-go signal. When
        contact never fires for the bottle it falls back to the finger-width
        gripper-open, then to the bottle's set-down.
        """
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[depart:hi] > self.MOTION_TH)[0]
        setdown = int(depart + int(mv[-1]) + 2) if mv.size else int(depart + 1)

        gi = np.where(grasp[lo:hi] > 0)[0] + lo
        gi = gi[gi >= depart - self.GRASP_BACK]
        grasp_t = int(gi[0]) if gi.size else int(depart)
        grasp_t = max(min(grasp_t, depart), lo)

        # The gripper holds the bottle through the set-down and lets go after, so
        # the contact-end is already past the set-down; do NOT pull the release to
        # the motion set-down, which a later bump of the placed bottle can corrupt.
        gc = np.where(grasp[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
        else:
            rel = self._release(fw, fw_open, grasp_t, hi, setdown)
        release_t = int(min(max(rel, grasp_t + 1), hi))
        return grasp_t, release_t

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
        pos = {n: np.array([h["pos"][n] for h in history]) for n in self.ALL}
        base_xy = np.array([h["base_xy"] for h in history])
        grasp = {n: np.array([h["grasp"][n] for h in history]) for n in self.ALL}
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())  # gripper fully-open finger width

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Which bottles were actually relocated (vs. left in place). Any of the
        # seven -- middle or end-cluster -- that moves more than MOVE_TH counts.
        info = {}
        for n in self.ALL:
            p = pos[n]
            disp = float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max())
            if disp > self.MOVE_TH:
                info[n] = dict(pos=p, type=self.btype[n], disp=disp,
                               depart=self._depart_time(p, 0, T))
        order = sorted(info, key=lambda n: info[n]["depart"])

        # Per-bottle grasp / release, in handling order. Each bottle's release
        # search is bounded by the next bottle's departure so a later bottle
        # bumping this one cannot extend its window.
        prev_release = 0
        for n, m in enumerate(order):
            b = info[m]
            hi = info[order[n + 1]]["depart"] if n + 1 < len(order) else T
            hi = max(hi, b["depart"] + 1)
            b["grasp"], b["release"] = self._events(
                grasp[m], b["pos"], fw, fw_open, b["depart"], prev_release, hi)
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

        # "a <type>" when more than one bottle of that type is relocated, else "the".
        type_n = {}
        for m in order:
            type_n[info[m]["type"]] = type_n.get(info[m]["type"], 0) + 1

        if order:
            # Leading drive to the dining counter (most demos start within reach).
            maybe_nav(info[order[0]]["grasp"], NAV_TO_COUNTER_LANG)
            for n, m in enumerate(order):
                b = info[m]
                art = "a" if type_n[b["type"]] > 1 else "the"
                if n > 0:
                    maybe_nav(b["grasp"], NAV_BACK_LANG)
                emit(b["grasp"], SUBTASK_BOTTLE, STAGE_PICK,
                     f"pick up {art} {b['type']} bottle from the middle")
                # Carry nav runs until the base stops; the place then covers the
                # set-down through the gripper-open release.
                maybe_nav(b["release"], f"navigate to the cluster of {b['type']} bottles")
                emit(b["release"], SUBTASK_BOTTLE, STAGE_PLACE,
                     f"place the {b['type']} bottle next to the other {b['type']} bottles")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "types": ",".join(info[m]["type"] for m in order),
            "n_moved": len(order),
            "base_max": float(np.linalg.norm(base_xy - base_xy[0], axis=1).max())
            if base_xy is not None else 0.0,
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        # Declare the counter->counter extension here too: a manual override
        # bypasses setup(), so without this the base bank-check would flag
        # PickPlaceCounterToCounter as an unknown subtask.
        self.extra_subtasks = {SUBTASK_BOTTLE}
        issues = super().validate(result)
        # Human-reviewed manual override: skip the structural bottle-count checks.
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
        # A variable number of bottles is relocated; at least one, and rarely more
        # than the scene's seven.
        if not (1 <= n_pick <= 5):
            issues.append(f"expected 1-5 relocated bottles, found {n_pick}")
        return issues
