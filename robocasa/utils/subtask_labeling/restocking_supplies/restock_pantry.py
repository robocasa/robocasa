import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each can is moved counter -> cabinet (bank atomic, splits into pick/place).
# Non-mobile task, so no navigation interrupts the pick-place; the cabinet is
# already open (no open/close subtask).
SUBTASK_CAN = "PickPlaceCounterToCabinet"

DONE_LANG = "task complete"


@register_labeler
class RestockPantryLabeler(BaseSubtaskLabeler):
    """Labeler for the RestockPantry composite task.

    Task (non-mobile): two cans sit on the counter; the cabinet is already open
    and has one can on one side (plus a non-can distractor on the other). Move the
    two counter cans into the cabinet next to the existing can.

    Doc-canonical decomposition (2 subtasks):
      1. PickPlaceCounterToCabinet [pick/place] -> stock the first can
      2. PickPlaceCounterToCabinet [pick/place] -> stock the second can
    plus a terminal done. No OpenCabinet/CloseCabinet (the cabinet stays open) and
    no navigation (the robot stays at the cabinet).

    Boundaries (per can): t_pick = the can departs its counter rest while held;
    t_place = the can settles in the cabinet with the gripper released. Cans are
    ordered by placement. Validation is structural (two picks + two places; both
    cans end in the cabinet near the existing can).
    """

    ENV_NAME = "RestockPantry"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    CANS = ["obj1", "obj2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.can_body_ids = [env.obj_body_id[n] for n in self.CANS]
        self.can_lang = []
        for n in self.CANS:
            try:
                lang = env.get_obj_lang(n)
            except Exception:
                lang = "can"
            # "canned food" reads awkwardly with an article; the task calls them
            # cans.
            self.can_lang.append("can" if lang == "canned food" else lang)
        # Existing can in the cabinet (the side the counter cans join), for the
        # structural end check only.
        try:
            self.ref_can_id = env.obj_body_id["cab_obj1"]
        except Exception:
            self.ref_can_id = None
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.expected_done_idx = None
        self.extra_subtasks = set()  # PickPlaceCounterToCabinet is a bank atomic

    def per_step(self, env, t: int) -> dict:
        ref = (
            env.sim.data.body_xpos[self.ref_can_id].copy()
            if self.ref_can_id is not None
            else np.array([np.nan, np.nan, np.nan])
        )
        return {
            "cans": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.can_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "ref": ref,
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
        cans = np.array([h["cans"] for h in history])  # (T, 2, 3)
        grip = np.array([h["grip"] for h in history])
        ref = np.array([h["ref"] for h in history])

        ops = []
        for i in range(len(self.CANS)):
            can = cans[:, i, :]
            t_rel = self._release_time(can, grip, T)
            t_dep = self._depart_time(can, grip, t_rel)
            ops.append(dict(idx=i, lang=self.can_lang[i], dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        same = len({o["lang"] for o in ops}) == 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for n, op in enumerate(order):
            # Two identical cans: the first is "a can" (one of two), the second is
            # "the can" (the remaining one). Distinct names are always "the".
            art = "a" if (same and n == 0) else "the"
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            emit(t_dep, SUBTASK_CAN, STAGE_PICK,
                 f"pick up {art} {op['lang']} from the counter")
            emit(t_rel, SUBTASK_CAN, STAGE_PLACE,
                 f"place the {op['lang']} in the cabinet")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        # End check: both cans near the existing cabinet can.
        ends_ok = True
        if np.isfinite(ref).all():
            ref_xy = np.median(ref[-self.REST_W:], axis=0)[:2]
            for i in range(len(self.CANS)):
                fin = np.median(cans[-self.REST_W:, i, :], axis=0)[:2]
                if np.linalg.norm(fin - ref_xy) > 0.4:
                    ends_ok = False
        debug = {
            "order": np.array([o["idx"] for o in order]),
            "ends_ok": int(ends_ok),
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
        if result.debug.get("ends_ok") == 0:
            issues.append("a can did not end near the existing cabinet can")
        return issues
