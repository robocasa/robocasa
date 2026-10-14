import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# The kettle is carried counter -> stove burner (pick/place), then the robot
# turns the burner knob on. Both names are bank atomics.
SUBTASK_KETTLE = "PickPlaceCounterToStove"
SUBTASK_TURN = "TurnOnStove"

DONE_LANG = "task complete"

# A stove knob counts as "on" once its (wrap-normalized) angle exceeds this; it
# matches Stove.is_burner_on's threshold.
KNOB_ON = 0.35


@register_labeler
class KettleBoilingLabeler(BaseSubtaskLabeler):
    """Labeler for the KettleBoiling composite task.

    Task (non-mobile): take the kettle from the counter, place it on a stove
    burner, then turn that burner on.

    Doc-canonical decomposition (2 subtasks):
      1. PickPlaceCounterToStove [pick/place]  -> kettle onto a stove burner
      2. TurnOnStove             [execute]     -> turn the burner knob on
    plus a terminal done. Because the base is parked at the stove the whole time,
    there is no leading navigation -- the demo opens directly on the pick.

    Events:
      - t_pick:  the kettle departs its rest pose while held. The kettle is
                 grasped loosely by its handle (gripper-to-body distance stays
                 large even while carried), so a displacement-only fallback backs
                 up the held-and-moved test, exactly as in ArrangeTea.
      - t_place: the kettle settles on the burner with the gripper released.
                 Release is measured RELATIVE to the tightest grasp distance
                 (handle-robust), since an absolute threshold would fire while
                 the kettle is still held by the handle.
      - t_turn:  the last frame any knob is still being turned, +1. The reach
                 from the released kettle over to the knob is part of TurnOnStove.
    Validation is structural (one pick, one place, one turn) plus a check that a
    burner ends on.
    """

    ENV_NAME = "KettleBoiling"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RETRACT_MARGIN = 0.10
    RELEASE_SUSTAIN = 5
    REST_W = 10
    KNOB_MOVE_EPS = 0.02
    # The finger-width signal is trusted as a release detector only if it spans
    # at least this range over the demo (a grasp clearly closed then opened).
    FW_RANGE_MIN = 0.03

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        try:
            self.kettle_lang = env.get_obj_lang("obj")
        except Exception:
            self.kettle_lang = "kettle"
        self.kettle_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.stove = env.stove
        self.knob_locs = list(self.stove.get_knobs_state(env=env).keys())
        # Cache the gripper finger joints (for the finger-width release signal)
        # and the world xy of every burner site (used to name the burner the
        # kettle ends up on).
        gripper = env.robots[0].gripper["right"]
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in gripper.joints
        ]
        # Flag suspiciously short subtasks for review (the demo-73 failure mode,
        # but near-zero instead of exactly zero).
        self.min_segment_frames = 30
        self.burner_xy = {}
        for loc, site in self.stove.burner_sites.items():
            if site is None:
                continue
            name = site.get("name")
            try:
                self.burner_xy[loc] = np.array(
                    env.sim.data.get_site_xpos(name)[:2], dtype=float
                )
            except Exception:
                pass
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        ks = self.stove.get_knobs_state(env=env)
        # Wrap-normalized "on-ness": how far the knob is turned from the off
        # (0 / 2pi) rest, regardless of turn direction.
        knob = max(
            (min(abs(ks[l]), 2 * np.pi - abs(ks[l])) for l in self.knob_locs),
            default=0.0,
        )
        fw = float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr))
        return {
            "obj": env.sim.data.body_xpos[self.kettle_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knob": float(knob),
            "fw": fw,
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, obj, grip, fw, t_dep, T):
        # Preferred signal: the gripper finger-width opening. The fingers stay
        # closed on the kettle through the whole carry and snap open when the
        # operator lets go, so the FIRST finger-open after the pickup (the kettle
        # can only depart its rest while grasped, so the gripper does not open in
        # between) pinpoints the release. This is robust to the loose handle
        # grasp -- where the gripper never gets "far" from the body and the
        # contact-based grasp check drops out mid-carry. The burner knob sits
        # right next to the kettle, so a gripper-to-body distance test would
        # otherwise fire only when the arm finally leaves, long after release.
        lo, hi = float(np.min(fw)), float(np.max(fw))
        if hi - lo > self.FW_RANGE_MIN:
            mid = lo + 0.5 * (hi - lo)
            opens = np.where((fw[:-1] < mid) & (fw[1:] >= mid))[0] + 1
            opens = opens[opens > t_dep]
            if opens.size:
                return int(opens[0])
        # Fallback (no clean finger-open, e.g. the gripper never opened cleanly):
        # release RELATIVE to the tightest grasp distance, sustained.
        final = np.median(obj[-self.REST_W:], axis=0)
        settled = np.linalg.norm(obj - final, axis=1) < self.PLACE_TOL
        gdist = np.linalg.norm(obj - grip, axis=1)
        grasp_dist = float(np.min(gdist))
        thr = max(self.RELEASE_TH, grasp_dist + self.RETRACT_MARGIN)
        released = gdist > thr
        ok = settled & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(settled)[0]
        return int(w[0]) if w.size else T - 1

    def _burner_location(self, obj):
        """Name of the burner the kettle ends up on (nearest burner site to the
        kettle's settled xy), formatted like 'rear-right'."""
        if not self.burner_xy:
            return None
        final_xy = np.median(obj[-self.REST_W:, :2], axis=0)
        loc = min(
            self.burner_xy, key=lambda l: np.linalg.norm(self.burner_xy[l] - final_xy)
        )
        return loc.replace("_", "-")

    def _depart_time(self, obj, end):
        # First frame the kettle leaves its rest pose. The kettle sits still on
        # the counter until it is picked up, and DEPART_TOL (0.10 m) is far larger
        # than any approach nudge, so the first displacement crossing is the
        # pickup. We deliberately do NOT also require the gripper to be close: the
        # kettle is grasped by its handle, so the gripper-to-body distance stays
        # large (~0.18 m) through the carry and only dips low at some unrelated
        # later moment -- gating on it would jump the pickup far past the true one
        # and collapse the place segment.
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        end = max(int(end), 1)
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _turn_done_time(self, knob, t_place, T):
        # The turn ends one frame after the knob stops moving (matches
        # ArrangeTea's close detection). Restrict to after the place so knob
        # jitter during the carry can't pull the boundary early. Many demos are
        # cut the instant the burner reaches "on" -- the operator is still
        # cranking the knob at the final frame, with no idle tail -- so clamp to
        # T-1 to always leave at least the terminal frame for the done segment.
        dk = np.abs(np.diff(knob))
        moving = np.where(dk > self.KNOB_MOVE_EPS)[0]
        moving = moving[moving >= t_place]
        if moving.size:
            return int(min(moving[-1] + 2, T - 1))
        # Knob never registered a clear turn: fall back to the first on-frame.
        on = np.where(knob >= KNOB_ON)[0]
        on = on[on >= t_place]
        return int(min(on[0], T - 1)) if on.size else T - 1

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])
        fw = np.array([h["fw"] for h in history])

        t_dep = self._depart_time(obj, T)
        t_rel = self._release_time(obj, grip, fw, t_dep, T)

        # The burner can only be turned once the kettle is down, so the place
        # must end no later than the first knob motion. This corrects demos where
        # the gripper lingers near the kettle (handle offset keeps the
        # gripper-to-body distance ambiguous) and only fully retracts after the
        # knob has already been turned -- without the cap the late retraction
        # would swallow the turn into the place segment.
        dk = np.abs(np.diff(knob))
        knob_moves = np.where(dk > self.KNOB_MOVE_EPS)[0]
        knob_moves = knob_moves[knob_moves > t_dep]
        t_knob_start = int(knob_moves[0]) if knob_moves.size else T
        t_rel = max(min(t_rel, t_knob_start), t_dep)

        t_turn = self._turn_done_time(knob, t_rel, T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        loc = self._burner_location(obj)
        burner = f"{loc} burner" if loc else "burner"
        turn_lang = f"turn on the {burner} where the {self.kettle_lang} is placed"

        emit(t_dep, SUBTASK_KETTLE, STAGE_PICK,
             f"pick up the {self.kettle_lang} from the counter")
        emit(t_rel, SUBTASK_KETTLE, STAGE_PLACE,
             f"place the {self.kettle_lang} on the stove burner")
        emit(t_turn, SUBTASK_TURN, STAGE_EXECUTE, turn_lang)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        burner_on = bool(knob[-1] >= KNOB_ON) if T else False
        debug = {
            "t_pick": int(t_dep),
            "t_place": int(t_rel),
            "t_turn": int(t_turn),
            "burner_on_at_end": int(burner_on),
            "n_segments": len(segments),
            "burner_loc": str(loc),
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
        names = result.subtask_name[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        n_turn = int(np.sum(names == SUBTASK_TURN))
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_turn != 1:
            issues.append(f"expected 1 TurnOnStove, found {n_turn}")
        if result.debug.get("burner_on_at_end") == 0:
            issues.append("no burner is on at the end of the demo")
        return issues
