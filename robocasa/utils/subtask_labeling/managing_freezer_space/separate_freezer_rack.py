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

# Each tupperware is carried counter -> freezer rack (pick / navigate / place),
# with a drive back to the counter between them. No bank name covers
# counter -> freezer, so it is a declared extension.
SUBTASK_TW = "PickPlaceCounterToFreezer"
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_FREEZER_LANG = "navigate to the freezer"
NAV_BACK_LANG = "navigate back to the counter"
DONE_LANG = "task complete"


@register_labeler
class SeparateFreezerRackLabeler(BaseSubtaskLabeler):
    """Labeler for the SeparateFreezerRack composite task.

    Task (mobile): two tupperware containers sit on the counter -- one holding the
    meat, one holding the vegetables -- and the freezer is already open. Carry the
    meat container to the SECOND-highest freezer rack and the vegetable container
    to the HIGHEST rack. The robot starts at the counter; the freezer is across the
    kitchen, so each container is a round trip.

    Doc-canonical decomposition (7 subtasks):
      1. PickPlaceCounterToFreezer [pick]   -> grab the first container
      2. NavigateKitchen                    -> carry it to the freezer
      3. PickPlaceCounterToFreezer [place]  -> set it on its rack
      4. NavigateKitchen                    -> drive back to the counter
      5. PickPlaceCounterToFreezer [pick]   -> grab the second container
      6. NavigateKitchen                    -> carry it to the freezer
      7. PickPlaceCounterToFreezer [place]  -> set it on its rack
    plus a terminal done. The two containers are auto-ordered by which is grabbed
    first (usually the meat). NavigateKitchen is adaptive (MIN_NAV_DIST).

    The containers are large and rim-grasped, so the contact-grasp check is
    unreliable; the gripper finger-width is clean. Pick ends at the grasp
    (finger-close before the lift); place ends at the release (first finger-open
    after the container arrives on its rack -- arrival-gated to skip a mid-carry
    regrasp).
    """

    ENV_NAME = "SeparateFreezerRack"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10
    FW_RANGE_MIN = 0.03
    GRASP_WINDOW = 90

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    # body key -> (lang noun, rack phrase)
    CONTAINERS = {
        "meat_tupperware": ("meat container", "second highest rack"),
        "veg_tupperware": ("vegetable container", "highest rack"),
    }

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.tw_keys = list(self.CONTAINERS)
        self.tw_body_ids = {k: env.obj_body_id[k] for k in self.tw_keys}
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j)
            for j in env.robots[0].gripper["right"].joints
        ]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # The set-down into the rack can be a fast ~20-frame motion once the
        # container has been carried right up to it, so the guard is below 30.
        self.min_segment_frames = 15
        self.extra_subtasks = {SUBTASK_TW}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "tw": {k: env.sim.data.body_xpos[b].copy() for k, b in self.tw_body_ids.items()},
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
        }

    # ---- event helpers -------------------------------------------------

    def _depart_time(self, obj, lo, hi):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        cand = np.where(moved[lo:hi])[0]
        return int(lo + int(cand[0])) if cand.size else lo

    def _grasp_time(self, closes, depart, lo):
        cand = closes[(closes > lo) & (closes <= depart)]
        if cand.size and (depart - int(cand[-1])) <= self.GRASP_WINDOW:
            return int(cand[-1])
        return int(depart)

    def _arrival_time(self, obj, lo, hi):
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.PLACE_TOL
        idxs = np.where(far)[0]
        idxs = idxs[(idxs >= lo) & (idxs < hi)]
        return int(idxs[-1]) + 1 if idxs.size else lo + 1

    def _release_time(self, opens, arrival, hi):
        cand = opens[(opens >= arrival) & (opens < hi)]
        return int(cand[0]) if cand.size else int(arrival)

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
        tw = {k: np.array([h["tw"][k] for h in history]) for k in self.tw_keys}
        base_xy = np.array([h["base_xy"] for h in history])
        fw = np.array([h["fw"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        lo_fw, hi_fw = float(np.min(fw)), float(np.max(fw))
        if hi_fw - lo_fw > self.FW_RANGE_MIN:
            mid = lo_fw + 0.5 * (hi_fw - lo_fw)
            opens = np.where((fw[:-1] < mid) & (fw[1:] >= mid))[0] + 1
            closes = np.where((fw[:-1] >= mid) & (fw[1:] < mid))[0] + 1
        else:
            opens = np.array([], dtype=int)
            closes = np.array([], dtype=int)

        info = {}
        for k in self.tw_keys:
            noun, rack = self.CONTAINERS[k]
            info[k] = dict(pos=tw[k], noun=noun, rack=rack,
                           depart=self._depart_time(tw[k], 0, T))
        order = sorted(self.tw_keys, key=lambda k: info[k]["depart"])
        first, second = info[order[0]], info[order[1]]

        # First container: handled before the second departs.
        first["grasp"] = self._grasp_time(closes, first["depart"], 0)
        first["arrival"] = self._arrival_time(first["pos"], first["grasp"], second["depart"])
        first["release"] = self._release_time(opens, first["arrival"], second["depart"])
        # Second container: after the first is placed.
        second["depart"] = max(second["depart"], first["release"])
        second["grasp"] = self._grasp_time(closes, second["depart"], first["release"])
        second["arrival"] = self._arrival_time(second["pos"], second["grasp"], T)
        second["release"] = self._release_time(opens, second["arrival"], T)

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

        # First container: pick, carry to freezer (the carry nav ends when the
        # container arrives at the rack, so base jitter while placing can't eat
        # into the place), place.
        emit(first["grasp"], SUBTASK_TW, STAGE_PICK,
             f"pick up the {first['noun']} from the counter")
        maybe_nav(first["arrival"], NAV_TO_FREEZER_LANG)
        emit(first["release"], SUBTASK_TW, STAGE_PLACE,
             f"place the {first['noun']} on the {first['rack']} of the freezer")
        # Drive back to the counter, then second container.
        maybe_nav(second["grasp"], NAV_BACK_LANG)
        emit(second["grasp"], SUBTASK_TW, STAGE_PICK,
             f"pick up the {second['noun']} from the counter")
        maybe_nav(second["arrival"], NAV_TO_FREEZER_LANG)
        emit(second["release"], SUBTASK_TW, STAGE_PLACE,
             f"place the {second['noun']} on the {second['rack']} of the freezer")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "first_grasp": int(first["grasp"]),
            "first_release": int(first["release"]),
            "second_grasp": int(second["grasp"]),
            "second_release": int(second["release"]),
            "base_max": float(np.linalg.norm(base_xy - base_xy[0], axis=1).max())
            if base_xy is not None else 0.0,
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
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        return issues
