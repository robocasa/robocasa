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

# Pan: cabinet -> stove burner (pick / navigate / place). No bank name covers
# cabinet -> stove, so it is a declared extension. Meat: counter/plate -> the pan
# on the stove, which is exactly PickPlaceCounterToStove ("pick from the plate,
# place in the pan"). Then turn the burner on.
SUBTASK_PAN = "PickPlaceCabinetToStove"
SUBTASK_MEAT = "PickPlaceCounterToStove"
SUBTASK_TURN = "TurnOnStove"
SUBTASK_NAV = "NavigateKitchen"

NAV_TO_STOVE_LANG = "navigate to the stove"
DONE_LANG = "task complete"

KNOB_ON = 0.35


@register_labeler
class SearingMeatLabeler(BaseSubtaskLabeler):
    """Labeler for the SearingMeat composite task.

    Task (mobile): grab the pan from the open cabinet, drive to the stove and set
    it on the specified burner, then put the meat (from a plate on the adjacent
    counter) into the pan and turn that burner on. The robot starts at the
    cabinet.

    Doc-canonical decomposition (3 high-level subtasks: pan, meat, turn). Emitted
    fine-grained:
      1. PickPlaceCabinetToStove [pick]   -> grab the pan from the cabinet
      2. NavigateKitchen                  -> carry the pan to the stove
      3. PickPlaceCabinetToStove [place]  -> set the pan on the burner
      4. PickPlaceCounterToStove [pick]   -> pick the meat off the plate
      5. PickPlaceCounterToStove [place]  -> place the meat in the pan
      6. TurnOnStove                      -> turn the burner on
    plus a terminal done. The meat sits on a counter adjacent to the stove, so its
    pick-place has no navigation. The cabinet->stove drive is short (~0.6-1.3 m),
    so the nav threshold is lowered; if a particular demo barely moves, the nav
    collapses and its place stage absorbs the carry.

    Releases are detected by object arrival (the object permanently reaches its
    final pose) rather than gripper-distance/finger-width -- the pan is
    handle-grasped and the meat is small, so neither of those is reliable. The
    knob is turned slowly over the final frames, so it is detected by value (it
    ends above KNOB_ON) rather than per-frame motion.
    """

    ENV_NAME = "SearingMeat"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10
    # Release = the gripper finger-width opening past its grasp level, once the
    # object has arrived near its final pose. Robust to the contact-based grasp
    # check dropping out mid-carry for the handle-grasped pan.
    ARRIVE_TOL = 0.15
    FW_OPEN_MARGIN = 0.008

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.40

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_body_id = env.obj_body_id["pan"]
        self.meat_body_id = env.obj_body_id["meat"]
        self.gripper = env.robots[0].gripper["right"]
        self.meat_geoms = env.objects["meat"].contact_geoms
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints
        ]
        try:
            self.meat_lang = env.get_obj_lang("meat")
        except Exception:
            self.meat_lang = "meat"
        self.knob = ep_meta.get("refs", {}).get("knob")
        self.burner = self.knob.replace("_", "-") if self.knob else None
        self.stove = env.stove
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 30
        self.extra_subtasks = {SUBTASK_PAN}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        kn = 0.0
        if self.knob is not None:
            ks = self.stove.get_knobs_state(env=env)
            a = ks.get(self.knob, 0.0)
            kn = float(min(abs(a), 2 * np.pi - abs(a)))
        return {
            "pan": env.sim.data.body_xpos[self.pan_body_id].copy(),
            "meat": env.sim.data.body_xpos[self.meat_body_id].copy(),
            "base_xy": base_xy,
            "knob": kn,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
            "grasp_meat": int(
                env._check_grasp(gripper=self.gripper, object_geoms=self.meat_geoms)
            ),
        }

    # ---- event helpers -------------------------------------------------

    def _pick_time(self, obj, end):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        end = max(int(end), 1)
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _release_pan(self, obj, fw, lo, T):
        # The pan is grasped by its handle, so the contact grasp check drops out
        # mid-carry -- detect the release from the gripper finger-width instead:
        # the first frame, after the pan has arrived near its final pose, where
        # the finger-width opens past the level it held during transport.
        final = np.median(obj[-self.REST_W:], axis=0)
        d2final = np.linalg.norm(obj - final, axis=1)
        near = np.where(d2final[:T] < self.ARRIVE_TOL)[0]
        near = near[near > lo]
        t_arrive = int(near[0]) if near.size else lo
        trans = fw[lo:max(t_arrive, lo + 1)]
        grasp_w = float(np.median(trans)) if trans.size else float(fw[lo])
        opened = fw > grasp_w + self.FW_OPEN_MARGIN
        cand = np.where(opened[:T])[0]
        cand = cand[cand >= t_arrive]
        if cand.size:
            return int(cand[0])
        return self._place_time(obj, lo, T)

    def _release_meat(self, grasp, meat, lo, T):
        # The meat is body-grasped (solid contact), so the contact grasp check is
        # reliable -- but the fingers barely close on the small meat, so the
        # finger-width is not. Release = the last frame the meat is grasped, +1.
        seg = np.where(grasp[lo:T] > 0)[0]
        if seg.size:
            return int(min(lo + int(seg[-1]) + 1, T))
        return self._place_time(meat, lo, T)

    def _place_time(self, obj, lo, T):
        # Fallback only: permanent arrival -- last frame the object is farther
        # than PLACE_TOL from its final resting pose, +1.
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.PLACE_TOL
        idxs = np.where(far[:T])[0]
        idxs = idxs[idxs >= lo]
        return int(idxs[-1]) + 1 if idxs.size else lo + 1

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

    def _turn_done_time(self, knob, lo, T):
        final = float(np.median(knob[-self.REST_W:])) if T else 0.0
        if final >= KNOB_ON:
            below = np.where(knob[:T] < final - 0.05)[0]
            below = below[below >= lo]
            if below.size:
                return int(min(below[-1] + 2, T - 1))
        return max(T - 1, lo)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        pan = np.array([h["pan"] for h in history])
        meat = np.array([h["meat"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        knob = np.array([h["knob"] for h in history])
        fw = np.array([h["fw"] for h in history])
        grasp_meat = np.array([h["grasp_meat"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        t_pan_pick = self._pick_time(pan, T)
        t_meat_pick_raw = self._pick_time(meat, T)
        t_pan_place = self._release_pan(pan, fw, t_pan_pick, T)
        t_meat_pick = max(t_meat_pick_raw, t_pan_place)
        t_meat_place = self._release_meat(grasp_meat, meat, t_meat_pick, T)
        t_turn = self._turn_done_time(knob, t_meat_place, T)

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

        burner = f"{self.burner} burner" if self.burner else "burner"

        emit(t_pan_pick, SUBTASK_PAN, STAGE_PICK, "pick up the pan from the cabinet")
        maybe_nav(t_pan_place, NAV_TO_STOVE_LANG)
        emit(t_pan_place, SUBTASK_PAN, STAGE_PLACE, f"place the pan on the {burner}")
        emit(t_meat_pick, SUBTASK_MEAT, STAGE_PICK,
             f"pick up the {self.meat_lang} from the counter")
        emit(t_meat_place, SUBTASK_MEAT, STAGE_PLACE,
             f"place the {self.meat_lang} on the pan")
        emit(t_turn, SUBTASK_TURN, STAGE_EXECUTE, f"turn on the {burner}")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        burner_on = bool(knob[-1] >= KNOB_ON) if T else False
        debug = {
            "t_pan_pick": int(t_pan_pick),
            "t_pan_place": int(t_pan_place),
            "t_meat_pick": int(t_meat_pick),
            "t_meat_place": int(t_meat_place),
            "t_turn": int(t_turn),
            "burner_on_at_end": int(burner_on),
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
        names = result.subtask_name[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        n_turn = int(np.sum(names == SUBTASK_TURN))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        if n_turn != 1:
            issues.append(f"expected 1 TurnOnStove, found {n_turn}")
        if result.debug.get("burner_on_at_end") == 0:
            issues.append("burner is not on at the end of the demo")
        return issues
