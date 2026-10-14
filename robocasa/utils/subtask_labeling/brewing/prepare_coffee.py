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

# The mug is carried cabinet -> under the coffee-machine dispenser (pick/place),
# then the robot presses the machine's start button. Both names are bank atomics.
SUBTASK_MUG = "CoffeeSetupMug"
SUBTASK_START = "StartCoffeeMachine"

START_LANG = "press the start button on the coffee machine"
DONE_LANG = "task complete"


@register_labeler
class PrepareCoffeeLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareCoffee composite task.

    Task (non-mobile): take the mug from the open cabinet, place it under the
    coffee-machine dispenser, then press the machine's start button. A distractor
    item shares the cabinet and is never moved. The robot stays parked at the
    cabinet/coffee machine.

    Doc-canonical decomposition (2 subtasks):
      1. CoffeeSetupMug    [pick/place]  -> mug under the dispenser
      2. StartCoffeeMachine [execute]    -> press the start button
    plus a terminal done. No leading navigation -- the demo opens on the pick.

    Events:
      - t_pick:  the mug departs its rest pose (first displacement crossing).
      - t_place: the gripper finger-width opens after the pickup (the release).
                 The mug is body-grasped, so this snaps cleanly; same robust
                 actuator signal used for the kettle in KettleBoiling.
      - t_start: one frame after the last gripper/start-button contact -- the
                 reach from the placed mug over to the button is part of
                 StartCoffeeMachine. Clamp to T-1 so a demo cut at the press still
                 leaves a terminal done frame.
    Validation is structural (one pick, one place, one StartCoffeeMachine) plus
    checks that the mug ends under the dispenser and the button was pressed.
    """

    ENV_NAME = "PrepareCoffee"

    DEPART_TOL = 0.10
    REST_W = 10
    FW_RANGE_MIN = 0.03
    # Fallback press detector: gripper-to-button distance under this counts as a
    # press when no explicit contact is recorded.
    PRESS_NEAR = 0.08
    # The mug counts as placed under the dispenser if its final xy is within this
    # of the receptacle site.
    RECEPT_TOL = 0.08

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        try:
            self.mug_lang = env.get_obj_lang("obj")
        except Exception:
            self.mug_lang = "mug"
        self.mug_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.gripper = env.robots[0].gripper["right"]
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints
        ]
        cm = env.coffee_machine
        pre = cm.naming_prefix
        self.button_names = [f"{pre}{b}" for b in cm._start_button_names]
        self.button_geom_ids = [env.sim.model.geom_name2id(b) for b in self.button_names]
        rid = env.sim.model.site_name2id(f"{pre}receptacle_place_site")
        self.receptacle_xy = np.array(env.sim.data.site_xpos[rid][:2], dtype=float)
        # Flag suspiciously short subtasks for review.
        self.min_segment_frames = 30
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        gp = env.sim.data.site_xpos[self.eef_site_id]
        button_dist = min(
            np.linalg.norm(gp - env.sim.data.geom_xpos[g]) for g in self.button_geom_ids
        )
        contact = int(
            any(env.check_contact(self.gripper, b) for b in self.button_names)
        )
        fw = float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr))
        return {
            "obj": env.sim.data.body_xpos[self.mug_body_id].copy(),
            "fw": fw,
            "button_dist": float(button_dist),
            "contact": contact,
        }

    # ---- event helpers -------------------------------------------------

    def _depart_time(self, obj, end):
        # First frame the mug leaves its rest pose -- the mug sits still in the
        # cabinet until picked up, and DEPART_TOL (0.10 m) dwarfs any approach
        # nudge, so the first displacement crossing is the pickup.
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        end = max(int(end), 1)
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _release_time(self, obj, fw, t_dep, T):
        # First gripper finger-open after the pickup = the release. The mug can
        # only depart while grasped, so the gripper does not open in between.
        lo, hi = float(np.min(fw)), float(np.max(fw))
        if hi - lo > self.FW_RANGE_MIN:
            mid = lo + 0.5 * (hi - lo)
            opens = np.where((fw[:-1] < mid) & (fw[1:] >= mid))[0] + 1
            opens = opens[opens > t_dep]
            if opens.size:
                return int(opens[0])
        # Fallback: the mug has permanently settled near its final pose.
        final = np.median(obj[-self.REST_W:], axis=0)
        settled = np.linalg.norm(obj - final, axis=1) < self.DEPART_TOL
        uns = np.where(~settled[:T])[0]
        return int(uns[-1]) + 1 if uns.size else max(t_dep + 1, 1)

    def _start_done_time(self, contact, button_dist, t_place, T):
        # One frame after the last button press. Prefer explicit gripper/button
        # contact; fall back to closest-approach. Clamp to T-1 so a demo cut
        # mid-press still leaves a terminal done frame.
        pressed = np.where(contact > 0)[0]
        pressed = pressed[pressed >= t_place]
        if pressed.size:
            return int(min(pressed[-1] + 2, T - 1))
        near = np.where(button_dist < self.PRESS_NEAR)[0]
        near = near[near >= t_place]
        if near.size:
            return int(min(near[-1] + 2, T - 1))
        return T - 1

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        obj = np.array([h["obj"] for h in history])
        fw = np.array([h["fw"] for h in history])
        button_dist = np.array([h["button_dist"] for h in history])
        contact = np.array([h["contact"] for h in history])

        t_dep = self._depart_time(obj, T)
        t_rel = self._release_time(obj, fw, t_dep, T)

        # The button can only be pressed once the mug is down, so the place must
        # end no later than the first button contact.
        pressed = np.where(contact > 0)[0]
        pressed = pressed[pressed > t_dep]
        t_press_start = int(pressed[0]) if pressed.size else T
        t_rel = max(min(t_rel, t_press_start), t_dep)

        t_start = self._start_done_time(contact, button_dist, t_rel, T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_dep, SUBTASK_MUG, STAGE_PICK,
             f"pick up the {self.mug_lang} from the cabinet")
        emit(t_rel, SUBTASK_MUG, STAGE_PLACE,
             f"place the {self.mug_lang} under the coffee machine dispenser")
        emit(t_start, SUBTASK_START, STAGE_EXECUTE, START_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        final_xy = np.median(obj[-self.REST_W:, :2], axis=0) if T else None
        under_disp = (
            bool(np.linalg.norm(final_xy - self.receptacle_xy) < self.RECEPT_TOL)
            if final_xy is not None else False
        )
        pressed_ok = bool(np.any(contact > 0))
        debug = {
            "t_pick": int(t_dep),
            "t_place": int(t_rel),
            "t_start": int(t_start),
            "mug_under_dispenser": int(under_disp),
            "button_pressed": int(pressed_ok),
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
        n_start = int(np.sum(names == SUBTASK_START))
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_start != 1:
            issues.append(f"expected 1 StartCoffeeMachine, found {n_start}")
        if result.debug.get("mug_under_dispenser") == 0:
            issues.append("mug did not end under the coffee machine dispenser")
        if result.debug.get("button_pressed") == 0:
            issues.append("start button was never contacted")
        return issues
