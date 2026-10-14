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

# First the blender lid is lifted off and set on the counter, then the non-rotten fruit
# (always fruit1; fruit0 is textured rotten and stays put) is picked off the counter and
# dropped into the open blender. Removing the lid has no bank name -> RemoveBlenderLid;
# counter -> blender has no bank name -> PickPlaceCounterToBlender (both extensions). The
# base never drives.
SUBTASK_LID = "RemoveBlenderLid"
SUBTASK_MOVE = "PickPlaceCounterToBlender"

LID_LANG = "take the lid off the blender"
DONE_LANG = "task complete"


@register_labeler
class ChooseRipeFruitLabeler(BaseSubtaskLabeler):
    """Labeler for the ChooseRipeFruit composite task.

    Fixed-base task: two fruit of the same kind sit on the counter, one rotten (fruit0,
    recolored brown) and one ripe (fruit1). Take the lid off the blender, then put the
    ripe/non-rotten fruit in the blender. The robot starts at the blender; the rotten
    fruit is never touched.

    Decomposition (3 high-level subtasks: remove lid, pick, place). Emitted:
      RemoveBlenderLid [execute]        -> take the lid off the blender
      PickPlaceCounterToBlender [pick]  -> pick up the non-rotten {fruit} from the counter
      PickPlaceCounterToBlender [place] -> place the non-rotten {fruit} in the blender
    plus a terminal done.

    Detection: the lid is grasped, lifted off the blender and set on the counter -- the
    remove-lid subtask ends when the gripper RELEASES the set-down lid; then only fruit1
    (the ripe one) moves. t_pick = its first sustained departure from the counter rest
    (after the lid is off); t_place = the gripper RELEASE.
    """

    ENV_NAME = "ChooseRipeFruit"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    LID_GRASP_TH = 0.06     # grip<->lid dist marking the gripper is on the lid
    LID_MOVE_MIN = 0.15     # lid displacement confirming it was actually removed
    LID_RELEASE_MARGIN = 0.08
    LID_MIN_CARRY = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.ripe = "fruit1"       # the ripe fruit is hard-coded fruit1 in the task
        self.rotten = "fruit0"
        self.ripe_body_id = env.obj_body_id[self.ripe]
        self.rotten_body_id = env.obj_body_id[self.rotten]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fruit_lang = env.get_obj_lang(self.ripe)
        except Exception:
            self.fruit_lang = "fruit"
        # the blender lid is a free-body auxiliary fixture; track its body if present
        self.lid_body_id = None
        try:
            lid = env.blender.blender_lid
            if lid is not None:
                self.lid_body_id = env.sim.model.body_name2id(f"{lid.name}_main")
        except Exception:
            self.lid_body_id = None
        self.extra_subtasks = {SUBTASK_LID, SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "ripe": env.sim.data.body_xpos[self.ripe_body_id].copy(),
            "rotten": env.sim.data.body_xpos[self.rotten_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }
        if self.lid_body_id is not None:
            d["lid"] = env.sim.data.body_xpos[self.lid_body_id].copy()
        return d

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

    def _lid_off(self, history, T):
        """Frame the blender lid is set down and released. None if no lid removal."""
        if self.lid_body_id is None or "lid" not in history[0]:
            return None, 0.0
        lid = np.array([h["lid"] for h in history])
        grip = np.array([h["grip"] for h in history])
        lid_disp = np.linalg.norm(lid - lid[0], axis=1)
        lid_gd = np.linalg.norm(grip - lid, axis=1)
        if float(lid_disp.max()) < self.LID_MOVE_MIN:
            return None, float(lid_disp.max())
        grasp_t = self._sustained_start(lid_gd < self.LID_GRASP_TH, 0, 3)
        if grasp_t is None:
            grasp_t = int(np.argmin(lid_gd))
        hold_gd = float(np.median(lid_gd[grasp_t: min(grasp_t + self.HOLD_W, T)]))
        released = lid_gd > hold_gd + self.LID_RELEASE_MARGIN
        t_off = self._sustained_start(released, grasp_t + self.LID_MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_off is None:
            # fall back to the frame the lid finishes moving (its displacement plateau)
            peak = int(np.argmax(lid_disp))
            t_off = peak
        return int(min(max(t_off, grasp_t + 1), T - 1)), float(lid_disp.max())

    def resolve(self, history):
        T = len(history)
        ripe = np.array([h["ripe"] for h in history])
        rotten = np.array([h["rotten"] for h in history])
        grip = np.array([h["grip"] for h in history])

        t_lid_off, lid_disp_max = self._lid_off(history, T)
        anchor = t_lid_off if t_lid_off is not None else 0

        # re-anchor the fruit rest after the lid is off (fruit1 has not moved yet)
        rest = np.median(ripe[: max(self.REST_W, anchor + 1)][-self.REST_W:], axis=0)
        disp = np.linalg.norm(ripe - rest, axis=1)
        gd = np.linalg.norm(grip - ripe, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, anchor, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, anchor + 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1
        t_placed = max(t_placed, t_pick + 1)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if t_lid_off is not None:
            emit(t_lid_off, SUBTASK_LID, STAGE_EXECUTE, LID_LANG)
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the non-rotten {self.fruit_lang} from the counter")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the non-rotten {self.fruit_lang} in the blender")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        rotten_moved = float(np.linalg.norm(rotten - rotten[0], axis=1).max())
        review = []
        if rotten_moved > 0.10:
            review.append(f"rotten fruit moved ({rotten_moved:.2f}m) -- check the wrong fruit wasn't chosen")
        if t_lid_off is None:
            review.append(f"blender lid never removed (lid moved only {lid_disp_max:.2f}m)")
        result = self.build_result(T, segments, debug={
            "t_lid_off": t_lid_off, "lid_disp_max": lid_disp_max,
            "t_pick": int(t_pick), "t_placed": int(t_placed),
            "ripe_disp": float(disp.max()), "rotten_moved": rotten_moved,
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
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 pick stage")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 place stage")
        if result.debug.get("t_lid_off") is not None and int(np.sum(names == SUBTASK_LID)) != 1:
            issues.append("expected 1 RemoveBlenderLid subtask")
        return issues
