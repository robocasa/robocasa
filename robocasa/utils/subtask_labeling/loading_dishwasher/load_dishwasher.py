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

# Each dish is moved counter -> dishwasher rack (pick/place); then the dishwasher
# is closed (the rack slides in and the door shuts together). No bank name covers
# counter -> dishwasher, so PickPlaceCounterToDishwasher is a declared extension;
# CloseDishwasher is a bank atomic.
SUBTASK_DISH = "PickPlaceCounterToDishwasher"
SUBTASK_CLOSE = "CloseDishwasher"

CLOSE_LANG = "close the dishwasher"
DONE_LANG = "task complete"


@register_labeler
class LoadDishwasherLabeler(BaseSubtaskLabeler):
    """Labeler for the LoadDishwasher composite task.

    Task (non-mobile): the dishwasher starts open with its top rack slid out. Put
    the two dishes (a cup and a bowl) from the counter onto the rack, then close
    the dishwasher (the rack slides in and the door shuts together). The robot
    stays parked at the dishwasher.

    Doc-canonical decomposition (3 subtasks):
      1. PickPlaceCounterToDishwasher [pick/place]  -> first dish onto the rack
      2. PickPlaceCounterToDishwasher [pick/place]  -> second dish onto the rack
      3. CloseDishwasher                            -> close the dishwasher
    plus a terminal done. The two dishes are auto-ordered by which is placed first
    (teleop order varies). No leading navigation.

    Releases use the contact grasp check (the dishes are solid -- cup and bowl --
    so it is reliable and gives a clean per-dish ungrasp). The close is detected
    from the door joint reaching its closed range.
    """

    ENV_NAME = "LoadDishwasher"

    DEPART_TOL = 0.10
    REST_W = 10
    CLOSE_TH = 0.05
    FW_RANGE_MIN = 0.03

    DISHES = ["dish0", "dish1"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.dish_body_ids = [env.obj_body_id[d] for d in self.DISHES]
        self.dish_lang = []
        for d in self.DISHES:
            try:
                self.dish_lang.append(env.get_obj_lang(d))
            except Exception:
                self.dish_lang.append(d)
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j)
            for j in env.robots[0].gripper["right"].joints
        ]
        dw = env.dishwasher
        self.door_qadr = env.sim.model.get_joint_qpos_addr(dw._joint_names["door"])
        self.min_segment_frames = 30
        self.extra_subtasks = {SUBTASK_DISH}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        return {
            "dishes": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.dish_body_ids]
            ),
            "door": float(abs(env.sim.data.qpos[self.door_qadr])),
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
        }

    # ---- event helpers -------------------------------------------------

    def _pick_time(self, obj, end):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        end = max(int(end), 1)
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _release_time(self, opens, obj, lo, hi, T):
        # Release = the first gripper finger-open after the dish is lifted, within
        # (lo, hi]. The dishes are rim-grasped, so the contact grasp check is
        # unreliable (it often never fires), but the finger-width opening is clean.
        hi = min(int(hi), T)
        cand = opens[(opens > lo) & (opens <= hi)]
        if cand.size:
            return int(cand[0])
        # Fallback: the dish's permanent arrival near its final pose.
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.DEPART_TOL
        idxs = np.where(far[:T])[0]
        idxs = idxs[(idxs > lo) & (idxs < hi)]
        return int(idxs[-1]) + 1 if idxs.size else min(lo + 1, T)

    def _grasp_time(self, closes, lift, after):
        # The grasp = the finger-width closing just before the dish is lifted (the
        # last close in (after, lift]). The pick stage ends here -- once the dish
        # is grasped the label switches to place. Falls back to the lift.
        cand = closes[(closes > after) & (closes <= lift)]
        return int(cand[-1]) if cand.size else int(lift)

    def _close_time(self, door, lo, T):
        above = np.where(door[:T] > self.CLOSE_TH)[0]
        above = above[above >= lo]
        return int(min(int(above[-1]) + 1, T)) if above.size else lo

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        dishes = np.array([h["dishes"] for h in history])  # (T, 2, 3)
        door = np.array([h["door"] for h in history])
        fw = np.array([h["fw"] for h in history])

        # Finger-width crossings: opens = releasing, closes = grasping.
        lo_fw, hi_fw = float(np.min(fw)), float(np.max(fw))
        if hi_fw - lo_fw > self.FW_RANGE_MIN:
            mid = lo_fw + 0.5 * (hi_fw - lo_fw)
            opens = np.where((fw[:-1] < mid) & (fw[1:] >= mid))[0] + 1
            closes = np.where((fw[:-1] >= mid) & (fw[1:] < mid))[0] + 1
        else:
            opens = np.array([], dtype=int)
            closes = np.array([], dtype=int)

        lifts = [self._pick_time(dishes[:, i, :], T) for i in range(2)]
        # The dish lifted first is placed first (pick -> place -> pick -> place).
        first, second = sorted(range(2), key=lambda i: lifts[i])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # Each dish: pick ends at the grasp (finger close before its lift); place
        # ends at the release (finger open).
        t_grasp_a = self._grasp_time(closes, lifts[first], 0)
        t_rel_a = self._release_time(
            opens, dishes[:, first, :], t_grasp_a, lifts[second], T
        )
        t_grasp_b = self._grasp_time(closes, lifts[second], t_rel_a)
        t_rel_b = self._release_time(opens, dishes[:, second, :], t_grasp_b, T, T)
        t_close = self._close_time(door, t_rel_b, T)

        for idx, t_grasp, t_rel in (
            (first, t_grasp_a, t_rel_a),
            (second, t_grasp_b, t_rel_b),
        ):
            lang = self.dish_lang[idx]
            emit(t_grasp, SUBTASK_DISH, STAGE_PICK,
                 f"pick up the {lang} from the counter")
            emit(t_rel, SUBTASK_DISH, STAGE_PLACE,
                 f"place the {lang} in the dishwasher")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        closed = bool(door[-1] < self.CLOSE_TH) if T else True
        debug = {
            "order": np.array([first, second]),
            "t_close": int(t_close),
            "closed_at_end": int(closed),
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
        n_close = int(np.sum(names == SUBTASK_CLOSE))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        if n_close != 1:
            issues.append(f"expected 1 CloseDishwasher, found {n_close}")
        if result.debug.get("closed_at_end") == 0:
            issues.append("dishwasher did not end closed")
        return issues
