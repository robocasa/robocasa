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

# Move the pan from its current burner to another burner (stove -> stove, no bank
# name -> PickPlaceStoveToStove extension), then turn the new burner on
# (TurnOnStove, bank). The base never drives.
SUBTASK_PAN = "PickPlaceStoveToStove"
SUBTASK_TURN = "TurnOnStove"

DONE_LANG = "task complete"
KNOB_ON = 0.35


@register_labeler
class FryingPanAdjustmentLabeler(BaseSubtaskLabeler):
    """Labeler for the FryingPanAdjustment composite task.

    Task: a pan sits on a lit burner. Move it to a different burner and turn that
    burner on. The base never drives.

    Doc-canonical decomposition (2 high-level subtasks: pan move, turn). Emitted
    fine-grained:
      1. PickPlaceStoveToStove [pick]   -> pick the pan up off its burner
      2. PickPlaceStoveToStove [place]  -> set the pan on the other burner
      3. TurnOnStove                    -> turn the new burner on
    plus a terminal done. The pan is moved first, then its new burner is turned on.

    The new burner is the knob that ends turned on; its turn completes when it
    reaches the on value. The pan is short-hop relocated within the stove -- pick =
    first departure, place = the gripper release.
    """

    ENV_NAME = "FryingPanAdjustment"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    PLACE_TOL = 0.12       # pan distance from final pose counting as "still moving"
    ARRIVE_TOL = 0.15      # pan within this of its final pose == arrived
    REL_FRAC = 0.55        # finger-width must reopen this far into the grasp->open range
    REL_SUSTAIN = 6        # ...and stay open this many frames (skip a mid-settle spike)
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.gripper = env.robots[0].gripper["right"]
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints
        ]
        self.stove = env.stove
        self.extra_subtasks = {SUBTASK_PAN}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        ks = self.stove.get_knobs_state(env=env)
        knobs = {k: float(min(abs(v), 2 * np.pi - abs(v))) for k, v in ks.items()}
        return {
            "pan": env.sim.data.body_xpos[self.pan_body_id].copy(),
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
            "knobs": knobs,
        }

    @staticmethod
    def _sustained_start(mask, start, sustain):
        run = 0
        for t in range(int(start), len(mask)):
            if mask[t]:
                run += 1
                if run >= sustain:
                    return t - sustain + 1
            else:
                run = 0
        return None

    def _release_pan(self, pan, fw, lo, T):
        """Release = the first frame, after the pan has arrived near its final pose,
        where the finger-width reopens most of the way to its fully-open width AND stays
        open. The pan is handle-grasped and the gripper drifts to the adjacent knob after
        letting go, so a gripper-to-pan distance would fire far too late; the finger-width
        opening is position-independent. The high fraction + sustained window skip a brief
        mid-settle grip adjustment that momentarily nudges the fingers apart."""
        final = np.median(pan[-self.REST_W:], axis=0)
        d2final = np.linalg.norm(pan - final, axis=1)
        near = np.where(d2final[:T] < self.ARRIVE_TOL)[0]
        near = near[near > lo]
        t_arrive = int(near[0]) if near.size else lo
        trans = fw[lo:max(t_arrive, lo + 1)]
        grasp_w = float(np.median(trans)) if trans.size else float(fw[lo])
        fw_open = float(np.max(fw))
        level = grasp_w + self.REL_FRAC * (fw_open - grasp_w)
        opened = fw > level
        for t in range(max(t_arrive, lo + 1), max(T - self.REL_SUSTAIN, lo + 2)):
            if bool(np.all(opened[t: t + self.REL_SUSTAIN])):
                return t
        # fallback: last frame the pan is still far from its final pose
        far = np.where(d2final >= self.PLACE_TOL)[0]
        far = far[far >= lo]
        return int(far[-1]) + 1 if far.size else lo + 1

    def _on_knob(self, knob_seq, T):
        keys = list(knob_seq[0].keys()) if T else []
        best, best_val = None, KNOB_ON
        for k in keys:
            v_end = float(np.median([knob_seq[t][k] for t in range(max(T - self.REST_W, 0), T)]))
            if v_end >= best_val:
                best, best_val = k, v_end
        if best is None:
            return None, T - 1
        arr = np.array([knob_seq[t][best] for t in range(T)])
        final = float(np.median(arr[-self.REST_W:]))
        below = np.where(arr < final - 0.05)[0]
        t_done = int(min(below[-1] + 2, T - 1)) if below.size else max(T - 1, 0)
        return best, t_done

    def resolve(self, history):
        T = len(history)
        pan = np.array([h["pan"] for h in history])
        fw = np.array([h["fw"] for h in history])
        knob_seq = [h["knobs"] for h in history]

        rest = np.median(pan[: self.REST_W], axis=0)
        disp = np.linalg.norm(pan - rest, axis=1)

        # The burner to move TO is not dictated by the task (the robot picks any other
        # burner) -> the PLACE stays generic ("another burner"). The TURN-ON, though,
        # acts on the specific burner the pan now sits on, which is determinate -> name
        # its location.
        knob, t_turn = self._on_knob(knob_seq, T)
        burner = knob.replace("_", "-") + " burner" if knob else "burner"

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = T - 1
        # place ends at the gripper RELEASE (finger-width opens once the pan has
        # arrived) -- the full set-down, not just the moment the pan first reaches its
        # spot (it keeps settling under the gripper for a while after that).
        t_placed = self._release_pan(pan, fw, t_pick, T)
        t_turn = max(t_turn, t_placed)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_PAN, STAGE_PICK, "pick up the pan off its burner")
        emit(t_placed, SUBTASK_PAN, STAGE_PLACE, "place the pan on another burner")
        emit(t_turn, SUBTASK_TURN, STAGE_EXECUTE, f"turn on the {burner}")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        burner_on = bool(knob is not None)
        debug = {
            "t_pick": int(t_pick), "t_placed": int(t_placed), "t_turn": int(t_turn),
            "burner": knob, "burner_on_at_end": int(burner_on), "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

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
            issues.append("new burner is not on at the end")
        return issues
