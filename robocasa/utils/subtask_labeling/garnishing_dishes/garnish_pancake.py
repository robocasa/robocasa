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

# Open the (closed) fridge, take the strawberry, drive to the dining counter,
# place it on the pancake. OpenFridge and NavigateKitchen are bank atomics; no
# exact bank name covers fridge -> counter, so PickPlaceFridgeToCounter is a
# declared extension (same as CookieDoughPrep).
SUBTASK_OPEN = "OpenFridge"
SUBTASK_STRAWBERRY = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the fridge"
NAV_LANG = "navigate to the pancake on the dining counter"
DONE_LANG = "task complete"


@register_labeler
class GarnishPancakeLabeler(BaseSubtaskLabeler):
    """Labeler for the GarnishPancake composite task.

    Task (mobile): the strawberry is in the closed fridge; a pancake (on a plate)
    sits on the dining counter across the kitchen. Open the fridge, take the
    strawberry, drive to the dining counter, and place it on the pancake. A fridge
    distractor is ignored. The robot starts at the fridge.

    Doc-canonical decomposition (4 subtasks):
      1. OpenFridge                        -> open the fridge door(s)
      2. PickPlaceFridgeToCounter [pick]   -> take the strawberry
      3. NavigateKitchen                   -> drive to the dining counter
      4. PickPlaceFridgeToCounter [place]  -> set it on the pancake
    plus a terminal done.

    Events: t_open = the fridge door stops moving (fully open); t_pick = the
    strawberry departs the fridge; t_release = the strawberry is let go (contact
    grasp end -- it is small and body-grasped, so the contact check is reliable
    and the finger-width is noisy from the door handle). The fridge->dining-counter
    drive is several metres, so NavigateKitchen is adaptive (MIN_NAV_DIST).
    """

    ENV_NAME = "GarnishPancake"

    DEPART_TOL = 0.10
    PLACE_TOL = 0.12
    REST_W = 10
    DOOR_MOVE_EPS = 0.02

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        try:
            self.berry_lang = env.get_obj_lang("strawberry")
        except Exception:
            self.berry_lang = "strawberry"
        self.berry_body_id = env.obj_body_id["strawberry"]
        self.gripper = env.robots[0].gripper["right"]
        self.berry_geoms = env.objects["strawberry"].contact_geoms
        fr = env.fridge
        self.door_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in fr._fridge_door_joint_names
        ]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.min_segment_frames = 30
        self.extra_subtasks = {SUBTASK_STRAWBERRY}
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "berry": env.sim.data.body_xpos[self.berry_body_id].copy(),
            "base_xy": base_xy,
            "door": float(sum(abs(env.sim.data.qpos[a]) for a in self.door_qadr)),
            "grasp": int(
                env._check_grasp(gripper=self.gripper, object_geoms=self.berry_geoms)
            ),
        }

    # ---- event helpers -------------------------------------------------

    def _pick_time(self, obj, end):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        end = max(int(end), 1)
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _release_time(self, grasp, obj, lo, T):
        seg = np.where(grasp[lo:T] > 0)[0]
        if seg.size:
            return int(min(lo + int(seg[-1]) + 1, T))
        final = np.median(obj[-self.REST_W:], axis=0)
        far = np.linalg.norm(obj - final, axis=1) >= self.PLACE_TOL
        idxs = np.where(far[:T])[0]
        idxs = idxs[idxs >= lo]
        return int(idxs[-1]) + 1 if idxs.size else lo + 1

    def _open_time(self, door, end, T):
        dmove = np.where(np.abs(np.diff(door)) > self.DOOR_MOVE_EPS)[0]
        dmove = dmove[dmove < end]
        return int(min(int(dmove[-1]) + 2, T)) if dmove.size else 1

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        berry = np.array([h["berry"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        door = np.array([h["door"] for h in history])
        grasp = np.array([h["grasp"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        t_pick = self._pick_time(berry, T)
        t_open = self._open_time(door, t_pick, T)
        t_release = self._release_time(grasp, berry, t_pick, T)

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

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(t_pick, SUBTASK_STRAWBERRY, STAGE_PICK,
             f"pick up the {self.berry_lang} from the fridge")
        maybe_nav(t_release, NAV_LANG)
        emit(t_release, SUBTASK_STRAWBERRY, STAGE_PLACE,
             f"place the {self.berry_lang} on the pancake")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        # The fridge is opened during the demo (it may end closed -- the operator
        # sometimes shuts it before leaving, and the robot is metres away at the
        # dining counter by the end, so the final door state is irrelevant).
        door_opened = bool(np.max(door) > 0.5) if T else False
        debug = {
            "t_open": int(t_open),
            "t_pick": int(t_pick),
            "t_release": int(t_release),
            "door_opened": int(door_opened),
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
        n_open = int(np.sum(names == SUBTASK_OPEN))
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if n_open != 1:
            issues.append(f"expected 1 OpenFridge, found {n_open}")
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        if n_nav != 1:
            issues.append(f"expected 1 NavigateKitchen, found {n_nav}")
        if result.debug.get("door_opened") == 0:
            issues.append("fridge was never opened")
        return issues
