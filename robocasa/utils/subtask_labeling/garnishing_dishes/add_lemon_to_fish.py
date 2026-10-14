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

# Open the (closed) fridge, take the lemon wedge out, carry it across to the dining
# counter and set it on the plate that holds the fish. OpenFridge + NavigateKitchen
# are bank atomics; fridge -> dining counter has no bank name so
# PickPlaceFridgeToCounter is a declared extension. The robot starts at the fridge.
SUBTASK_OPEN = "OpenFridge"
SUBTASK_MOVE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class AddLemonToFishLabeler(BaseSubtaskLabeler):
    """Labeler for the AddLemonToFish composite task.

    Task (mobile): the lemon wedge sits in a CLOSED fridge (robot starts at the
    fridge); a plate with fish sits on the dining counter. Open the fridge, take the
    lemon out, carry it to the dining counter, and set it on the fish's plate.

    Doc-canonical decomposition (4 high-level subtasks). Emitted fine-grained:
      1. OpenFridge                       -> open the fridge door
      2. PickPlaceFridgeToCounter [pick]  -> pick up the lemon wedge from the fridge
      3. NavigateKitchen                  -> carry it to the dining counter
      4. PickPlaceFridgeToCounter [place] -> place it on the plate with the fish
    plus a terminal done.

    Detection: OpenFridge ends when the fridge door first reaches OPEN_ABS and stays;
    t_pick = the lemon's first sustained displacement from its in-fridge rest (after
    the door opens); the nav is the cross-kitchen carry (base drive to its stop); the
    place ends at the gripper RELEASE.
    """

    ENV_NAME = "AddLemonToFish"

    OPEN_ABS = 0.85
    OPEN_SUSTAIN = 5
    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.lemon_body_id = env.obj_body_id["lemon_wedge"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.lemon_lang = env.get_obj_lang("lemon_wedge")
        except Exception:
            self.lemon_lang = "lemon wedge"
        try:
            self.fish_lang = env.get_obj_lang("fish")
        except Exception:
            self.fish_lang = "fish"
        # fridge door hinge joint(s), each a fraction of its own open range
        fr = env.fridge
        self.door_adrs, self.door_scales = [], []
        for j in list(getattr(fr, "_fridge_door_joint_names", []) or []):
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.has_door = len(self.door_adrs) > 0
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        doors = np.array(
            [abs(float(env.sim.data.qpos[a])) / s
             for a, s in zip(self.door_adrs, self.door_scales)]
        )
        return {
            "lemon": env.sim.data.body_xpos[self.lemon_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": doors,
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

    def resolve(self, history):
        T = len(history)
        lemon = np.array([h["lemon"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(lemon[: self.REST_W], axis=0)
        disp = np.linalg.norm(lemon - rest, axis=1)
        gd = np.linalg.norm(grip - lemon, axis=1)

        # OpenFridge ends when the LAST door leaf that gets opened reaches its open
        # threshold (a french-door fridge is opened one leaf at a time; a max-over-doors
        # signal would end at the first leaf while the second is still swinging). Only
        # leaves that actually open count, so the reach-behind-one-door case stays 1.
        open_frames = []
        for i in range(doorv.shape[1]):
            fr = self._sustained_start(doorv[:, i] >= self.OPEN_ABS, 0, self.OPEN_SUSTAIN)
            if fr is not None:
                open_frames.append(fr)
        n_open = len(open_frames)
        if open_frames:
            t_open = max(open_frames)
        elif doorv.shape[1]:
            t_open = int(np.argmax(doorv.max(axis=1)))
        else:
            t_open = 0
        open_lang = "open the fridge doors" if n_open >= 2 else "open the fridge door"

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, t_open, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp[t_open:] >= self.MOVE_TOL)[0]
            t_pick = t_open + int(mv[0]) if mv.size else min(t_open + 1, T - 1)
        t_pick = max(t_pick, t_open)

        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if self.has_door and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, open_lang)
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.lemon_lang} from the fridge")
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE,
                 f"carry the {self.lemon_lang} to the dining counter")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.lemon_lang} on the plate with the {self.fish_lang}")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_open": int(t_open), "t_pick": int(t_pick), "t_arrive": int(t_arrive),
            "t_placed": int(t_placed), "n_doors_opened": int(n_open),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        exp_open = 1 if getattr(self, "has_door", True) else 0
        if int(np.sum(names == SUBTASK_OPEN)) != exp_open:
            issues.append(f"expected {exp_open} OpenFridge")
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 pick stage")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 place stage")
        return issues
