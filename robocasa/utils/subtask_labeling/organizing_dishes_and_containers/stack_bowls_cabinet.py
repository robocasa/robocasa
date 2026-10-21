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

# Both bowls end stacked (smaller on top of larger) in the open cabinet, reached
# by one of two strategies the demos use interchangeably -- see the class
# docstring. PickPlaceCounterToCabinet is a bank atomic.
SUBTASK_BOWL = "PickPlaceCounterToCabinet"
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_CABINET_LANG = "navigate to the cabinet"
NAV_BACK_LANG = "navigate back to the counter"
DONE_LANG = "task complete"


@register_labeler
class StackBowlsCabinetLabeler(BaseSubtaskLabeler):
    """Labeler for the StackBowlsCabinet composite task.

    Task (mobile): two bowls sit on the counter and a cabinet is open. The end
    state is the smaller bowl stacked on top of the larger bowl inside the
    cabinet. Demos reach it by one of two strategies (detected per demo; both
    are supported and labeled on their own terms):

      A (independent): carry the larger bowl into the cabinet, then carry the
        smaller bowl and stack it on top IN the cabinet. Larger is grasped
        first; it stays put while the smaller is stacked (only a small nudge).
      B (nest-then-carry): nest the smaller bowl onto the larger one ON THE
        COUNTER, then carry the stacked pair into the cabinet as one unit.
        Smaller is grasped first; it is then carried a SECOND time (riding
        inside the larger bowl) -- this is the discriminator (NEST_LATE_TH).

    B is the common case in this dataset; A is rare. Both emit 2 bowl subtasks
    (PickPlaceCounterToCabinet [pick/place]) plus a terminal done, with adaptive
    NavigateKitchen segments where the base actually drives (the cabinet is
    reachable in most layouts, so most demos do not move -- and B never drives
    between the two picks, since the nest is local to the counter).

    The bowls are rim-grasped, so the contact-grasp check is unreliable; the
    gripper finger-width is clean. Pick ends at the grasp (finger-close before the
    lift); place ends at the release (first finger-open after the lift). Release is
    keyed off the lift rather than the bowl's arrival because stacking the second
    bowl nudges the first, which corrupts the first bowl's "final" pose.
    """

    ENV_NAME = "StackBowlsCabinet"

    DEPART_TOL = 0.10
    REST_W = 10
    FW_RANGE_MIN = 0.03
    GRASP_WINDOW = 90

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65
    # Strategy discriminator: the first-placed bowl's max displacement AFTER the
    # second bowl is lifted. A stacking nudge stays tiny (~0.02); a nested bowl
    # riding to the cabinet moves a lot (~0.5). The gap is wide, so 0.20 is safe.
    NEST_LATE_TH = 0.20

    BOWLS = ["bowl1", "bowl2"]
    # bowl1 is scaled 0.7 and bowl2 0.95 in the env, so bowl1 is the smaller bowl
    # (which is meant to end up stacked on top of the larger bowl2).
    SIZE = {"bowl1": "smaller", "bowl2": "larger"}

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.bowl_body_ids = {k: env.obj_body_id[k] for k in self.BOWLS}
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j)
            for j in env.robots[0].gripper["right"].joints
        ]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # In short demos the second bowl can be grabbed ~12 frames after the first
        # is released (the bowls sit right next to each other), so the guard is low.
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "bowls": {k: env.sim.data.body_xpos[b].copy() for k, b in self.bowl_body_ids.items()},
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

    def _release_time(self, opens, obj, depart, hi):
        cand = opens[(opens > depart) & (opens < hi)]
        if cand.size:
            return int(cand[0])
        # Fallback: the bowl's last significant motion (set-down) within bounds.
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[depart:hi] > 0.004)[0]
        return int(depart + int(mv[-1]) + 1) if mv.size else depart + 1

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
        bowls = {k: np.array([h["bowls"][k] for h in history]) for k in self.BOWLS}
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

        info = {k: dict(pos=bowls[k], size=self.SIZE[k],
                        depart=self._depart_time(bowls[k], 0, T))
                for k in self.BOWLS}
        order = sorted(self.BOWLS, key=lambda k: info[k]["depart"])
        first, second = info[order[0]], info[order[1]]

        first["grasp"] = self._grasp_time(closes, first["depart"], 0)
        first["release"] = self._release_time(opens, first["pos"], first["depart"],
                                              max(second["depart"], first["depart"] + 1))
        second["depart"] = max(second["depart"], first["release"])
        second["grasp"] = self._grasp_time(closes, second["depart"], first["release"])
        second["release"] = self._release_time(opens, second["pos"], second["depart"], T)

        # Strategy detection: does the first-placed bowl get carried a SECOND
        # time -- riding nested inside the other bowl into the cabinet? If so
        # this is the nest-on-the-counter-then-carry-the-pair strategy; else
        # each bowl is moved independently into the cabinet. Keyed on the first
        # bowl's displacement after the second bowl is lifted (bimodal: a
        # stacking nudge ~0.02 vs. a full carry ~0.5).
        sref = min(second["grasp"], T - 1)
        first_late = (
            float(np.max(np.linalg.norm(first["pos"][sref:] - first["pos"][sref], axis=1)))
            if sref < T - 1
            else 0.0
        )
        nest_then_carry = first_late > self.NEST_LATE_TH

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

        if nest_then_carry:
            # Strategy B: nest the first bowl onto the second ON THE COUNTER
            # (local, no nav), then carry the stacked pair into the cabinet as
            # one unit. The first bowl's "place" is the stack; the second bowl's
            # "place" is the whole pair arriving in the cabinet.
            emit(first["grasp"], SUBTASK_BOWL, STAGE_PICK,
                 f"pick up the {first['size']} bowl from the counter")
            emit(first["release"], SUBTASK_BOWL, STAGE_PLACE,
                 f"stack the {first['size']} bowl on top of the "
                 f"{second['size']} bowl")
            emit(second["grasp"], SUBTASK_BOWL, STAGE_PICK,
                 f"pick up the {second['size']} bowl from the counter")
            maybe_nav(second["release"], NAV_TO_CABINET_LANG)
            emit(second["release"], SUBTASK_BOWL, STAGE_PLACE,
                 "place the bowls in the cabinet")
        else:
            # Strategy A: carry the first bowl into the cabinet (it becomes the
            # bottom), drive back, then carry the second bowl and stack it on
            # top IN the cabinet.
            emit(first["grasp"], SUBTASK_BOWL, STAGE_PICK,
                 f"pick up the {first['size']} bowl from the counter")
            maybe_nav(first["release"], NAV_TO_CABINET_LANG)
            emit(first["release"], SUBTASK_BOWL, STAGE_PLACE,
                 f"place the {first['size']} bowl in the cabinet")
            maybe_nav(second["grasp"], NAV_BACK_LANG)
            emit(second["grasp"], SUBTASK_BOWL, STAGE_PICK,
                 f"pick up the {second['size']} bowl from the counter")
            maybe_nav(second["release"], NAV_TO_CABINET_LANG)
            emit(second["release"], SUBTASK_BOWL, STAGE_PLACE,
                 f"stack the {second['size']} bowl on top of the "
                 f"{first['size']} bowl in the cabinet")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "first_grasp": int(first["grasp"]),
            "first_release": int(first["release"]),
            "second_grasp": int(second["grasp"]),
            "second_release": int(second["release"]),
            "strategy": "B_nest_carry" if nest_then_carry else "A_independent",
            "first_late": round(first_late, 3),
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
