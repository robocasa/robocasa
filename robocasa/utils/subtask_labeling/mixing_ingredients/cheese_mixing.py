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


# Put the cheese in the pot of milk on the (lit) stove, then stir it with the spatula. The
# cheese move is the bank atomic PickPlaceCounterToStove (plate -> pot on the stove); the
# stir has no bank name -> StirPot (extension, an EXECUTE action that includes picking up the
# spatula and stirring the cheese in the pot).
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_STIR = "StirPot"

DONE_LANG = "task complete"


@register_labeler
class CheeseMixingLabeler(BaseSubtaskLabeler):
    """Labeler for the CheeseMixing composite task.

    Task (mobile, but local): a pot of milk sits on a LIT burner; cheese sits on a plate and a
    spatula on the counter. Put the cheese in the pot, then stir it with the spatula. The base
    only makes a short initial approach (absorbed into the cheese pick, not a subtask).

    Doc-canonical decomposition (cheese move + stir); the spatula pick was broken out per
    review. Emitted:
      PickPlaceCounterToStove [pick]   -> pick up the cheese from the plate
      PickPlaceCounterToStove [place]  -> place the cheese in the pot
      StirPot [pick]                   -> pick up the spatula
      StirPot [execute]                -> stir the cheese in the pot with the spatula
    plus a terminal done.

    Boundaries: t_pick = the cheese first displaces from its rest; t_place = the gripper
    RELEASE (cheese dropped in the pot). The spatula pick runs from the cheese release to the
    frame the spatula is first firmly in hand (grip<->spatula below the grasp threshold); the
    stir runs from there to the end. A review fires if the spatula is never grasped.
    """

    ENV_NAME = "CheeseMixing"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 30
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    SPATULA_GRASP_TH = 0.12   # gd to the spatula below this == the spatula is in hand
    SPATULA_SUSTAIN = 3       # frames the spatula grasp must hold to end the spatula pick

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cheese_body_id = env.obj_body_id["cheese"]
        self.spatula_body_id = env.obj_body_id["spatula"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cheese_lang = env.get_obj_lang("cheese")
        except Exception:
            self.cheese_lang = "cheese"
        self.extra_subtasks = {SUBTASK_STIR}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "cheese": env.sim.data.body_xpos[self.cheese_body_id].copy(),
            "spatula": env.sim.data.body_xpos[self.spatula_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
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
        cheese = np.array([h["cheese"] for h in history])
        spatula = np.array([h["spatula"] for h in history])
        grip = np.array([h["grip"] for h in history])

        rest = np.median(cheese[: self.REST_W], axis=0)
        disp = np.linalg.norm(cheese - rest, axis=1)
        cgd = np.linalg.norm(grip - cheese, axis=1)
        sgd = np.linalg.norm(grip - spatula, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(cgd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = cgd > hold_gd + self.RELEASE_MARGIN
        t_place = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.cheese_lang} from the plate")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.cheese_lang} in the pot")
        # pick up the spatula (reach + grasp), then stir. The spatula pick ends when the
        # spatula is first firmly in hand (grip<->spatula below the grasp threshold).
        t_spatula = self._sustained_start(sgd < self.SPATULA_GRASP_TH, t_place, self.SPATULA_SUSTAIN)
        spatula_grasped = t_spatula is not None
        if spatula_grasped and int(t_spatula) > cursor:
            emit(int(t_spatula), SUBTASK_STIR, STAGE_PICK, "pick up the spatula")
        if cursor < T - 1:
            emit(T - 1, SUBTASK_STIR, STAGE_EXECUTE, f"stir the {self.cheese_lang} in the pot with the spatula")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if not spatula_grasped:
            review.append("spatula never grasped after the cheese drop -- verify the stir")

        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_place": int(t_place),
            "spatula_grasped": spatula_grasped, "n_segments": len(segments),
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

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 cheese pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 cheese place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_STIR, STAGE_PICK) not in (0, 1):
            issues.append(f"expected 0-1 spatula pick, found {count(SUBTASK_STIR, STAGE_PICK)}")
        if count(SUBTASK_STIR, STAGE_EXECUTE) != 1:
            issues.append(f"expected 1 StirPot execute, found {count(SUBTASK_STIR, STAGE_EXECUTE)}")
        return issues
