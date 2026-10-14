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

# The newer meat is carried from the counter into the (already-open) fridge and set
# behind the older meat, then the fridge is closed. Counter -> fridge has no bank
# name -> PickPlaceCounterToFridge (extension); NavigateKitchen + CloseFridge are bank.
# The robot starts at the counter.
SUBTASK_MOVE = "PickPlaceCounterToFridge"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseFridge"

DONE_LANG = "task complete"


@register_labeler
class LoadFridgeFifoLabeler(BaseSubtaskLabeler):
    """Labeler for the LoadFridgeFifo composite task.

    Task (mobile): an older meat sits on a fridge rack; a newer meat of the same type
    sits on the counter. Carry the newer meat into the (already-open) fridge and place
    it behind the older meat on the same rack, then close the fridge door. The robot
    starts at the counter.

    Doc-canonical decomposition (4 high-level subtasks). Emitted fine-grained:
      1. PickPlaceCounterToFridge [pick]   -> pick up the newer meat from the counter
      2. NavigateKitchen                   -> carry it to the fridge
      3. PickPlaceCounterToFridge [place]  -> place it behind the older meat
      4. CloseFridge                       -> close the fridge door
    plus a terminal done.

    Detection: t_pick = the meat's first sustained displacement from its counter rest;
    the nav is the cross-kitchen carry (base drive to its stop); the place ends at the
    gripper RELEASE; CloseFridge ends when the door fraction drops back to closed.
    """

    ENV_NAME = "LoadFridgeFifo"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15
    CLOSE_TH = 0.20    # door fraction below this == closed
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.meat_body_id = env.obj_body_id["meat_new"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.meat_lang = env.get_obj_lang("meat_new")
        except Exception:
            self.meat_lang = "meat"
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
            "meat": env.sim.data.body_xpos[self.meat_body_id].copy(),
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
        meat = np.array([h["meat"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(meat[: self.REST_W], axis=0)
        disp = np.linalg.norm(meat - rest, axis=1)
        gd = np.linalg.norm(grip - meat, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = min(1, T - 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # CloseFridge: after the meat is placed, each opened door swings back to closed.
        # End when the LAST door that was open finishes closing (french-door fridges close
        # two leaves); pluralize "doors" vs "door" by how many actually close.
        close_frames = []
        for i in range(doorv.shape[1]):
            if float(doorv[:, i].max()) < 0.5:   # this leaf was never opened
                continue
            fr = self._sustained_start(doorv[:, i] < self.CLOSE_TH, t_placed, self.CLOSE_SUSTAIN)
            if fr is not None:
                close_frames.append(fr)
        n_closed = len(close_frames)
        t_close = max(close_frames) if close_frames else T - 1
        t_close = max(t_close, t_placed + 1)
        close_lang = "close the fridge doors" if n_closed >= 2 else "close the fridge door"

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

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the newer {self.meat_lang} from the counter")
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.meat_lang} to the fridge")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.meat_lang} behind the older {self.meat_lang} in the fridge")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, close_lang)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        closed = bool(doorv[-1].max() < self.CLOSE_TH) if doorv.shape[1] else False
        debug = {
            "t_pick": int(t_pick), "t_arrive": int(t_arrive), "t_placed": int(t_placed),
            "t_close": int(t_close), "closed_at_end": int(closed), "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 pick stage")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 place stage")
        if int(np.sum(names == SUBTASK_CLOSE)) != 1:
            issues.append("expected 1 CloseFridge")
        if result.debug.get("closed_at_end") == 0:
            issues.append("fridge not closed at end")
        return issues
