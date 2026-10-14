import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each condiment is moved counter -> cabinet, next to its matching counterpart
# (bank atomic, splits into pick/place). Non-mobile, cabinet already open: no
# nav, no open/close.
SUBTASK_OBJ = "PickPlaceCounterToCabinet"

DONE_LANG = "task complete"


@register_labeler
class CategorizeCondimentsLabeler(BaseSubtaskLabeler):
    """Labeler for the CategorizeCondiments composite task.

    Task (non-mobile): a condiment bottle (obj1) and a shaker (obj2) sit on the
    counter; the open cabinet already holds a matching bottle (cab_obj1) and
    shaker (cab_obj2). Place each counter item next to its counterpart in the
    cabinet (bottle by the bottle, shaker by the shaker).

    Doc-canonical decomposition (2 subtasks):
      1. PickPlaceCounterToCabinet [pick/place] -> stock the first condiment
      2. PickPlaceCounterToCabinet [pick/place] -> stock the second condiment
    plus a terminal done. The robot stays at the cabinet.

    Per item: t_pick = it departs its counter rest while held; t_place = it settles
    in the cabinet (by its counterpart) with the gripper released. Ordered by
    placement. Validation is structural (two picks + two places; each item ends
    next to its counterpart).
    """

    ENV_NAME = "CategorizeCondiments"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    OBJS = ["obj1", "obj2"]
    COUNTERPARTS = ["cab_obj1", "cab_obj2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = [env.obj_body_id[n] for n in self.OBJS]
        self.cp_body_ids = [env.obj_body_id[n] for n in self.COUNTERPARTS]

        def lang(n, fallback):
            try:
                return env.get_obj_lang(n)
            except Exception:
                return fallback

        self.obj_lang = [lang(n, "condiment") for n in self.OBJS]
        self.cp_lang = [lang(n, "condiment") for n in self.COUNTERPARTS]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.expected_done_idx = None
        self.extra_subtasks = set()  # PickPlaceCounterToCabinet is a bank atomic

    def per_step(self, env, t: int) -> dict:
        return {
            "objs": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.obj_body_ids]
            ),
            "cps": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.cp_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, obj, grip, T):
        final = np.median(obj[-self.REST_W:], axis=0)
        settled = np.linalg.norm(obj - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(obj - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, obj, grip, t_release):
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(obj - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        objs = np.array([h["objs"] for h in history])  # (T, 2, 3)
        cps = np.array([h["cps"] for h in history])     # (T, 2, 3)
        grip = np.array([h["grip"] for h in history])

        ops = []
        for i in range(len(self.OBJS)):
            obj = objs[:, i, :]
            t_rel = self._release_time(obj, grip, T)
            t_dep = self._depart_time(obj, grip, t_rel)
            ops.append(dict(idx=i, dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for op in order:
            i = op["idx"]
            on = self.obj_lang[i]
            cp = self.cp_lang[i]
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            # Name the counterpart it joins; "the other {x}" when they share a name.
            if on == cp:
                place_lang = f"place the {on} next to the other {on} in the cabinet"
            else:
                place_lang = f"place the {on} next to the {cp} in the cabinet"
            emit(t_dep, SUBTASK_OBJ, STAGE_PICK, f"pick up the {on} from the counter")
            emit(t_rel, SUBTASK_OBJ, STAGE_PLACE, place_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        # End check: each item ends next to its counterpart.
        ends_ok = True
        for i in range(len(self.OBJS)):
            of = np.median(objs[-self.REST_W:, i, :], axis=0)[:2]
            cf = np.median(cps[-self.REST_W:, i, :], axis=0)[:2]
            if np.linalg.norm(of - cf) > 0.30:
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
            issues.append("an item did not end next to its counterpart")
        return issues
