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

# A refrigerated item is moved from the fridge compartment to the freezer compartment of
# the same fixture, then both the fridge and freezer doors are closed. Fridge -> freezer
# has no bank name -> PickPlaceFridgeToFreezer (extension); CloseFridge is bank. The move
# is largely local (both compartments of one fixture). Robot starts at the fridge.
SUBTASK_MOVE = "PickPlaceFridgeToFreezer"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseFridge"

DONE_LANG = "task complete"


@register_labeler
class MoveFridgeToFreezerLabeler(BaseSubtaskLabeler):
    """Labeler for the MoveFridgeToFreezer composite task.

    Task (mobile): a refrigerated item sits in the (open) fridge; move it to any shelf of
    the (open) freezer, then close both the fridge and freezer doors. The two
    compartments belong to one fixture, so the move is largely local. Robot starts at the
    fridge.

    Doc-canonical decomposition (2 high-level: item move, close). Emitted:
      PickPlaceFridgeToFreezer [pick] -> [nav if the base drives] -> [place] ->
      CloseFridge -> done. CloseFridge covers both doors and ends when the LAST of the
      fridge / freezer door groups reaches closed.

    Detection: t_pick = the item first departs its fridge rest; place = the gripper
    RELEASE; the close ends when every opened door leaf swings back below CLOSE_TH.
    """

    ENV_NAME = "MoveFridgeToFreezer"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.07
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.40
    REST_W = 15
    CLOSE_TH = 0.20
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.item_body_id = env.obj_body_id["refrigerated_item"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.item_lang = env.get_obj_lang("refrigerated_item")
        except Exception:
            self.item_lang = "item"
        fr = env.fridge
        joint_names = list(getattr(fr, "_fridge_door_joint_names", []) or []) + \
            list(getattr(fr, "_freezer_door_joint_names", []) or [])
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
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        return {
            "item": env.sim.data.body_xpos[self.item_body_id].copy(),
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
        item = np.array([h["item"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(item[: self.REST_W], axis=0)
        disp = np.linalg.norm(item - rest, axis=1)
        gd = np.linalg.norm(grip - item, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = min(1, T - 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        close_frames = []
        for i in range(doorv.shape[1]):
            if float(doorv[:, i].max()) < 0.5:
                continue
            fr = self._sustained_start(doorv[:, i] < self.CLOSE_TH, t_placed, self.CLOSE_SUSTAIN)
            if fr is not None:
                close_frames.append(fr)
        t_close = max(close_frames) if close_frames else T - 1
        t_close = max(t_close, t_placed + 1)

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

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.item_lang} from the fridge")
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.item_lang} to the freezer")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.item_lang} on a shelf in the freezer")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, "close the fridge and freezer doors")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        closed = bool(doorv[-1].max() < self.CLOSE_TH) if doorv.shape[1] else False
        return self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_placed": int(t_placed), "t_close": int(t_close),
            "closed_at_end": int(closed), "n_segments": len(segments),
        })

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
            issues.append("doors not closed at end")
        return issues
