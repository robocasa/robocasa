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
# cabinet -> stove, so it is a declared extension. Then turn the burner on
# (TurnOnStove, bank). The robot starts at the (closed) cabinet; opening it is
# part of the pan pick's approach (the pan does not move until it is grasped).
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_PAN = "PickPlaceCabinetToStove"
SUBTASK_TURN = "TurnOnStove"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the cabinet doors above"
NAV_TO_STOVE_LANG = "navigate to the stove"
DONE_LANG = "task complete"

KNOB_ON = 0.35


@register_labeler
class SetupFryingLabeler(BaseSubtaskLabeler):
    """Labeler for the SetupFrying composite task.

    Task (mobile): the pan sits in a CLOSED cabinet (robot starts at the cabinet).
    Open the cabinet, take the pan out, carry it to the stove and set it on a
    burner, then turn that burner on.

    Decomposition (OpenCabinet broken out at the user's request, ahead of the doc's
    2 high-level pan-move + turn). Emitted fine-grained:
      1. OpenCabinet                      -> open the cabinet doors above
      2. PickPlaceCabinetToStove [pick]   -> take the pan from the cabinet
      3. NavigateKitchen                  -> carry the pan to the stove (adaptive)
      4. PickPlaceCabinetToStove [place]  -> set the pan on the burner
      5. TurnOnStove                      -> turn the burner on
    plus a terminal done. The cabinet is closed at the start; OpenCabinet ends when
    the last door leaf that gets opened reaches its open threshold (double-door
    cabinet), and the pan pick begins after that. The stove is sometimes within arm
    reach of the cabinet, so the nav is adaptive and collapses when the base barely
    moves.

    The burner is not in ep_meta refs, so it is detected as the knob that ends
    turned on. Release is finger-width based (the pan is handle-grasped, so the
    contact-grasp check drops out mid-carry).
    """

    ENV_NAME = "SetupFrying"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10
    ARRIVE_TOL = 0.15
    FW_OPEN_MARGIN = 0.008

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.40

    OPEN_ABS = 0.85    # door fraction (of fully open) at/above which it counts as open
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pan_body_id = env.obj_body_id["pan"]
        self.gripper = env.robots[0].gripper["right"]
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints
        ]
        self.stove = env.stove
        # Cabinet door hinge joints (2 for the double-door cabinet), each tracked as a
        # fraction of its own fully-open range. Fall back to a model scan by the
        # cabinet's name prefix if the build-time dict is empty.
        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
            pref = cab.name
            joint_names = [
                n for n in env.sim.model.joint_names
                if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())
            ]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.has_door = len(self.door_adrs) > 0
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 20
        self.extra_subtasks = {SUBTASK_PAN}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        ks = self.stove.get_knobs_state(env=env)
        knobs = {k: float(min(abs(v), 2 * np.pi - abs(v))) for k, v in ks.items()}
        doors = np.array(
            [abs(float(env.sim.data.qpos[a])) / s
             for a, s in zip(self.door_adrs, self.door_scales)]
        )
        return {
            "pan": env.sim.data.body_xpos[self.pan_body_id].copy(),
            "base_xy": base_xy,
            "knobs": knobs,
            "doors": doors,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
        }

    # ---- event helpers -------------------------------------------------

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

    def _pick_time(self, obj, end):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        end = max(int(end), 1)
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _release_pan(self, obj, fw, lo, T):
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

    def _place_time(self, obj, lo, T):
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

    def _on_knob(self, knob_seq, T):
        """The knob that ends turned on, plus the frame its turn-on completes."""
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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        pan = np.array([h["pan"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        fw = np.array([h["fw"] for h in history])
        knob_seq = [h["knobs"] for h in history]
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # The burner to place ON is not dictated by the task (the robot picks any
        # burner) -> the PLACE stays generic ("a burner"). The TURN-ON acts on the
        # specific burner the pan now sits on, which is determinate -> name its location.
        knob, t_turn = self._on_knob(knob_seq, T)
        burner = knob.replace("_", "-") + " burner" if knob else "burner"

        t_pan_pick = self._pick_time(pan, T)
        t_pan_place = self._release_pan(pan, fw, t_pan_pick, T)
        t_turn = max(t_turn, t_pan_place)

        # OpenCabinet: the double-door cabinet swings open before the pan is grasped.
        # End when the LAST leaf that gets opened reaches its open threshold (a
        # max-over-doors signal would end at the first leaf and mislabel the second
        # swing as the pick). Bounded to end before the pan departs.
        t_open = 0
        if self.has_door and doorv.shape[1] > 0:
            open_frames = []
            for i in range(doorv.shape[1]):
                fr = self._sustained_start(doorv[:, i] >= self.OPEN_ABS, 0, self.OPEN_SUSTAIN)
                if fr is not None:
                    open_frames.append(fr)
            t_open = max(open_frames) if open_frames else int(np.argmax(doorv.max(axis=1)))
        t_open = int(min(t_open, max(t_pan_pick - 1, 0)))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if self.has_door and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(t_pan_pick, SUBTASK_PAN, STAGE_PICK, "take the pan from the cabinet")
        nav_end = self._nav_end(base_speed, base_xy, cursor, t_pan_place)
        if nav_end > cursor:
            emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, NAV_TO_STOVE_LANG)
        emit(t_pan_place, SUBTASK_PAN, STAGE_PLACE, "place the pan on a burner")
        emit(t_turn, SUBTASK_TURN, STAGE_EXECUTE, f"turn on the {burner}")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        burner_on = bool(knob is not None)
        debug = {
            "t_pan_pick": int(t_pan_pick),
            "t_pan_place": int(t_pan_place),
            "t_turn": int(t_turn),
            "burner": knob,
            "burner_on_at_end": int(burner_on),
            "n_segments": len(segments),
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
        n_open = int(np.sum(names == SUBTASK_OPEN))
        exp_open = 1 if getattr(self, "has_door", True) else 0
        if n_open != exp_open:
            issues.append(f"expected {exp_open} OpenCabinet, found {n_open}")
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_turn != 1:
            issues.append(f"expected 1 TurnOnStove, found {n_turn}")
        if result.debug.get("burner_on_at_end") == 0:
            issues.append("burner is not on at the end of the demo")
        return issues
