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


# A fruit or vegetable is carried from the counter onto the correct (fruit vs veg)
# fridge rack, then the fridge is closed. The fridge starts OPEN. Counter -> fridge
# has no bank name -> PickPlaceCounterToFridge (extension); NavigateKitchen and
# CloseFridge are bank atomics. Two fridge distractors (a fruit and a vegetable
# already on their racks) stay put.
SUBTASK_MOVE = "PickPlaceCounterToFridge"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseFridge"

DONE_LANG = "task complete"


@register_labeler
class FreshProduceOrganizationLabeler(BaseSubtaskLabeler):
    """Labeler for the FreshProduceOrganization composite task.

    Task (mobile): a fruit-or-vegetable food sits on the counter (where the robot
    starts) and the fridge is already open, holding a fruit and a vegetable on their
    racks. Place the food on the rack matching its category, then close the fridge.

    Doc-canonical count = 4 (`mobile: Yes`): pick + carry nav + place + CloseFridge.
    Structure:
      PickPlaceCounterToFridge [pick]  -> pick up the {food} from the counter
      (NavigateKitchen [navigate])     -> carry the {food} to the fridge  (adaptive)
      PickPlaceCounterToFridge [place] -> place the {food} on the {rack} of the fridge
      CloseFridge [execute]            -> close the fridge door(s)
    plus a terminal done. The target rack (top / second-highest shelf) comes from
    ep_meta refs by category.

    Boundaries: t_pick = the food first displaces from its counter rest; the carry
    nav is split off the base motion (adaptive); t_place = the gripper RELEASE;
    CloseFridge ends when the opened door(s) swing back below the closed fraction.
    """

    ENV_NAME = "FreshProduceOrganization"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15
    CLOSE_TH = 0.20
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["food"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.food_lang = env.get_obj_lang("food")
        except Exception:
            self.food_lang = "food"
        # Target rack from ep_meta refs (category -> rack index; -1 top, -2 second).
        refs = ep_meta.get("refs", {})
        cat = refs.get("restock_obj_cat", "fruit")
        idx = refs.get("fruit_rack_index", -1) if cat == "fruit" else refs.get("veg_rack_index", -2)
        self.rack = "top shelf" if idx == -1 else "second highest shelf"
        # Fridge door joints (french-door fridges have two leaves).
        fr = env.fridge
        self.door_adrs, self.door_scales = [], []
        for n in list(getattr(fr, "_fridge_door_joint_names", []) or []):
            try:
                adr = env.sim.model.get_joint_qpos_addr(n)
                jid = env.sim.model.joint_name2id(n)
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
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
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
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        gd = np.linalg.norm(grip - obj, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # CloseFridge: after the food is placed, each opened leaf swings back to closed.
        close_frames = []
        for i in range(doorv.shape[1]):
            if float(doorv[:, i].max()) < 0.5:
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

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.food_lang} from the counter")
        carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.food_lang} to the fridge")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.food_lang} on the {self.rack} of the fridge")
        has_close = n_closed > 0 or t_close < T
        if has_close:
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, close_lang)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        closed = bool(doorv[-1].max() < self.CLOSE_TH) if doorv.shape[1] else False
        if doorv.shape[1] and not closed:
            review.append("fridge not closed at end -- verify")

        result = self.build_result(T, segments, debug={
            "t_pick": t_pick, "t_placed": t_placed, "t_close": int(t_close),
            "n_closed": n_closed, "closed_at_end": int(closed), "rack": self.rack,
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append(f"expected 1 pick, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append(f"expected 1 place, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_CLOSE)) != 1:
            issues.append(f"expected 1 CloseFridge, found {int(np.sum(names == SUBTASK_CLOSE))}")
        if int(np.sum(names == SUBTASK_NAV)) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {int(np.sum(names == SUBTASK_NAV))}")
        return issues
