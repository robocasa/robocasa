import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Pick a can and stack it on top of another can (both in the cabinet). No exact
# bank name covers a stack, so StackCan is a declared extension; it splits into
# pick/place stages like a pick-place.
SUBTASK_STACK = "StackCan"

DONE_LANG = "task complete"


@register_labeler
class StackCansLabeler(BaseSubtaskLabeler):
    """Labeler for the StackCans composite task.

    Task (non-mobile): four cans sit in the open cabinet; the robot stacks two of
    them on top of the other two, making two stacks of two. The robot stays at the
    cabinet.

    Doc-canonical decomposition (2 subtasks):
      1. StackCan [pick/place] -> lift one can onto a base can
      2. StackCan [pick/place] -> lift another can onto the other base can
    plus a terminal done. No open/close (the cabinet stays open) and no
    navigation.

    The two moved cans are auto-detected (the two with the largest displacement;
    they are grasped and lifted ~one can-height onto a base, the other two barely
    move). Per moved can: t_pick = it departs its rest while held; t_place = it
    settles on the base with the gripper released. Ordered by stack completion.
    Validation is structural (two picks + two places; exactly two cans moved).
    """

    ENV_NAME = "StackCans"

    DEPART_TOL = 0.06
    GRASP_TH = 0.10
    PLACE_TOL = 0.06
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10
    # A moved can must be displaced past MOVE_MIN, and clearly more than the most-
    # displaced base (a base can gets nudged up to ~0.07 when a can is stacked on
    # it, so an absolute base threshold is unreliable -- require a 2x gap instead).
    MOVE_MIN = 0.08
    BASE_GAP_FRAC = 0.5

    CANS = ["can1", "can2", "can3", "can4"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.can_body_ids = [env.obj_body_id[n] for n in self.CANS]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_STACK}

    def per_step(self, env, t: int) -> dict:
        return {
            "cans": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.can_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, can, grip, T):
        final = np.median(can[-self.REST_W:], axis=0)
        settled = np.linalg.norm(can - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(can - grip, axis=1) > self.RELEASE_TH
        ok = settled & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(settled)[0]
        return int(w[0]) if w.size else T - 1

    def _depart_time(self, can, grip, t_release):
        start = np.median(can[: self.REST_W], axis=0)
        moved = np.linalg.norm(can - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(can - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        cans = np.array([h["cans"] for h in history])  # (T, 4, 3)
        grip = np.array([h["grip"] for h in history])

        # The two moved cans = the two with the largest start->end displacement.
        starts = np.median(cans[: self.REST_W], axis=0)   # (4,3)
        finals = np.median(cans[-self.REST_W:], axis=0)   # (4,3)
        disp = np.linalg.norm(finals - starts, axis=1)    # (4,)
        movers = list(np.argsort(disp)[-2:])              # indices of top-2

        ops = []
        for i in movers:
            can = cans[:, i, :]
            t_rel = self._release_time(can, grip, T)
            t_dep = self._depart_time(can, grip, t_rel)
            ops.append(dict(idx=int(i), dep=t_dep, rel=t_rel, disp=float(disp[i])))
        order = sorted(ops, key=lambda o: o["rel"])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for n, op in enumerate(order):
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            if n == 0:
                pick_lang = "pick up a can in the open cabinet"
                place_lang = "stack the can on top of another can"
            else:
                # By the second stack one pair already exists; this can is one of
                # the still-unstacked cans and goes on the other base.
                pick_lang = "pick up an unstacked can"
                place_lang = "stack the can on top of another unstacked can"
            emit(t_dep, SUBTASK_STACK, STAGE_PICK, pick_lang)
            emit(t_rel, SUBTASK_STACK, STAGE_PLACE, place_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        # Exactly two clear movers: the smaller mover is a genuine move and is
        # clearly larger than the most-displaced base.
        min_mover = min(o["disp"] for o in ops) if ops else 0.0
        max_base = max(sorted(disp)[:-2]) if len(disp) > 2 else 0.0
        clean = bool(min_mover > self.MOVE_MIN and max_base < self.BASE_GAP_FRAC * min_mover)
        debug = {
            "movers": np.array([o["idx"] for o in order]),
            "disp": disp,
            "clean_split": int(clean),
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
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        if result.debug.get("clean_split") == 0:
            issues.append("could not cleanly identify exactly two moved cans")
        return issues
