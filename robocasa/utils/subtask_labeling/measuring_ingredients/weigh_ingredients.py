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

# The packaged food is moved cabinet -> the digital scale on the counter
# (pick/place), then the (double-door) cabinet is closed. PickPlaceCabinetToCounter
# is a bank atomic; the bank has OpenCabinet but no close, so CloseCabinet is a
# declared extension.
SUBTASK_FOOD = "PickPlaceCabinetToCounter"
SUBTASK_CLOSE = "CloseCabinet"

CLOSE_LANG = "close the cabinet"
DONE_LANG = "task complete"


@register_labeler
class WeighIngredientsLabeler(BaseSubtaskLabeler):
    """Labeler for the WeighIngredients composite task.

    Task (non-mobile): a packaged food sits in the open cabinet (a second packaged
    food is a distractor) and a digital scale sits on the counter. Take the food,
    place it on the scale to be weighed, then close the cabinet. The robot stays
    parked at the cabinet.

    Doc-canonical decomposition (2 subtasks):
      1. PickPlaceCabinetToCounter [pick/place]  -> food onto the digital scale
      2. CloseCabinet                            -> close the cabinet
    plus a terminal done.

    The food is a solid packaged item, so the contact-grasp check is reliable: pick
    ends at the grasp onset, place ends at the release (the later of the
    contact-grasp end and the food's arrival on the scale). The close is the
    cabinet's last open frame, +1.
    """

    ENV_NAME = "WeighIngredients"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10
    CLOSE_TH = 0.05

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        try:
            self.food_lang = env.get_obj_lang("obj")
        except Exception:
            self.food_lang = "item"
        self.food_body_id = env.obj_body_id["obj"]
        self.gripper = env.robots[0].gripper["right"]
        self.food_geoms = env.objects["obj"].contact_geoms
        cab = env.cab
        nm = getattr(cab, "name", None)
        self.door_addrs = []
        if nm is not None:
            for j in range(env.sim.model.njnt):
                try:
                    jn = env.sim.model.joint_id2name(j)
                except Exception:
                    jn = None
                if jn and jn.startswith(nm) and "doorhinge" in jn:
                    self.door_addrs.append(env.sim.model.jnt_qposadr[j])
        self.min_segment_frames = 15
        self.extra_subtasks = {SUBTASK_CLOSE}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        door = (
            sum(abs(env.sim.data.qpos[a]) for a in self.door_addrs)
            if self.door_addrs else 0.0
        )
        return {
            "food": env.sim.data.body_xpos[self.food_body_id].copy(),
            "door": float(door),
            "grasp": int(
                env._check_grasp(gripper=self.gripper, object_geoms=self.food_geoms)
            ),
        }

    # ---- event helpers -------------------------------------------------

    def _lift_time(self, obj, T):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        cand = np.where(moved[:T])[0]
        return int(cand[0]) if cand.size else 0

    def _grasp_time(self, grasp, lift, T):
        seg = np.where(grasp[:T] > 0)[0]
        return int(seg[0]) if seg.size else int(lift)

    def _release_time(self, grasp, obj, lo, T):
        # Release = the later of the contact-grasp end and the food's arrival on
        # the scale (a flaky short grasp falls back to arrival).
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.PLACE_TOL
        idxs = np.where(far[:T])[0]
        idxs = idxs[idxs >= lo]
        t_arrival = int(idxs[-1]) + 1 if idxs.size else lo + 1
        seg = np.where(grasp[lo:T] > 0)[0]
        t_grasp_end = int(lo + int(seg[-1]) + 1) if seg.size else lo
        return int(min(max(t_arrival, t_grasp_end), T))

    def _close_time(self, door, lo, T):
        above = np.where(door[:T] > self.CLOSE_TH)[0]
        above = above[above >= lo]
        return int(min(int(above[-1]) + 1, T)) if above.size else lo

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        food = np.array([h["food"] for h in history])
        door = np.array([h["door"] for h in history])
        grasp = np.array([h["grasp"] for h in history])

        t_lift = self._lift_time(food, T)
        t_grasp = self._grasp_time(grasp, t_lift, T)
        t_release = self._release_time(grasp, food, t_grasp, T)
        t_close = self._close_time(door, t_release, T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_grasp, SUBTASK_FOOD, STAGE_PICK,
             f"pick up the {self.food_lang} from the cabinet")
        emit(t_release, SUBTASK_FOOD, STAGE_PLACE,
             f"place the {self.food_lang} on the digital scale")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        closed = bool(door[-1] < self.CLOSE_TH) if T else True
        debug = {
            "t_grasp": int(t_grasp),
            "t_release": int(t_release),
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
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_close != 1:
            issues.append(f"expected 1 CloseCabinet, found {n_close}")
        if result.debug.get("closed_at_end") == 0:
            issues.append("cabinet did not end closed")
        return issues
