import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Take the meat out of its container onto the cutting board, then take the condiment out of
# the (already-open) cabinet and set it next to the cutting board. Meat container->board has
# no bank name -> PickPlaceCounterToCounter (extension); condiment cabinet->counter is the
# bank atomic PickPlaceCabinetToCounter. The cabinet starts open (no OpenCabinet). The base
# barely moves (everything sits by the cabinet), so there is no navigation.
SUBTASK_MEAT = "PickPlaceCounterToCounter"
SUBTASK_COND = "PickPlaceCabinetToCounter"

DONE_LANG = "task complete"


@register_labeler
class PrepMarinatingMeatLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepMarinatingMeat composite task.

    Task (mobile, but local): the meat sits in a container on the counter; a condiment sits
    in an ALREADY-OPEN cabinet; a cutting board sits on the counter. Move the meat onto the
    cutting board, then set the condiment next to the cutting board. A cabinet distractor
    never moves.

    Doc-canonical decomposition (2 high-level: meat move + condiment move). Emitted in
    first-motion order (meat first):
      PickPlaceCounterToCounter [pick]   -> pick up the {meat} from the {container}
      PickPlaceCounterToCounter [place]  -> place the {meat} on the cutting board
      PickPlaceCabinetToCounter [pick]   -> pick up the {condiment} from the cabinet
      PickPlaceCabinetToCounter [place]  -> place the {condiment} next to the cutting board
    plus a terminal done. No OpenCabinet (the cabinet starts open).

    Per object: t_pick = first sustained displacement from a rest re-anchored at the segment
    start; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PrepMarinatingMeat"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.body = {o: env.obj_body_id[o] for o in ("meat", "condiment")}
        try:
            self.meat_lang = env.get_obj_lang("meat")
        except Exception:
            self.meat_lang = "meat"
        try:
            self.cond_lang = env.get_obj_lang("condiment")
        except Exception:
            self.cond_lang = "condiment"
        try:
            self.cont_lang = env.get_obj_lang("meat_container")
        except Exception:
            self.cont_lang = "container"

        self.extra_subtasks = {SUBTASK_MEAT}
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in ("meat", "condiment"):
            d[o] = env.sim.data.body_xpos[self.body[o]].copy()
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

    def _move(self, opos, gd, cursor, T):
        r = np.median(opos[max(cursor - self.REST_W, 0): cursor + 1], axis=0)
        dsp = np.linalg.norm(opos - r, axis=1)
        t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
            t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        t_pick = max(t_pick, cursor)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1
        return t_pick, t_place

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in ("meat", "condiment")}
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in ("meat", "condiment")}

        first_move = {}
        for o in ("meat", "condiment"):
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in ("meat", "condiment") if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            t_pick, t_place = self._move(opos[o], gd[o], cursor, T)
            if o == "meat":
                emit(t_pick, SUBTASK_MEAT, STAGE_PICK, f"pick up the {self.meat_lang} from the {self.cont_lang}")
                emit(t_place, SUBTASK_MEAT, STAGE_PLACE, f"place the {self.meat_lang} on the cutting board")
            else:
                emit(t_pick, SUBTASK_COND, STAGE_PICK, f"pick up the {self.cond_lang} from the cabinet")
                emit(t_place, SUBTASK_COND, STAGE_PLACE, f"place the {self.cond_lang} next to the cutting board")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved (meat + condiment)")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
