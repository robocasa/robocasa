from collections import Counter

import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each condiment is moved counter -> cabinet (bank atomic, splits into pick/place).
# Non-mobile, cabinet already open: no nav, no open/close.
SUBTASK_COND = "PickPlaceCounterToCabinet"

DONE_LANG = "task complete"


@register_labeler
class OrganizeCondimentsLabeler(BaseSubtaskLabeler):
    """Labeler for the OrganizeCondiments composite task.

    Task (non-mobile): three condiments and one non-condiment distractor sit on
    the counter; the open cabinet is empty. Move only the three condiments into
    the cabinet, leaving the distractor on the counter.

    Doc-canonical decomposition (3 subtasks):
      1-3. PickPlaceCounterToCabinet [pick/place] -> stock each condiment
    plus a terminal done. The robot stays at the cabinet.

    Per condiment: t_pick = it departs its counter rest while held; t_place = it
    settles in the cabinet with the gripper released. Ordered by placement; the
    distractor is ignored. Validation is structural (three picks + three places).
    """

    ENV_NAME = "OrganizeCondiments"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    CONDIMENTS = ["condiment1", "condiment2", "condiment3"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cond_body_ids = [env.obj_body_id[n] for n in self.CONDIMENTS]
        self.cond_lang = []
        for n in self.CONDIMENTS:
            try:
                self.cond_lang.append(env.get_obj_lang(n))
            except Exception:
                self.cond_lang.append("condiment")
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.expected_done_idx = None
        self.extra_subtasks = set()  # PickPlaceCounterToCabinet is a bank atomic

    def per_step(self, env, t: int) -> dict:
        return {
            "conds": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.cond_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, cond, grip, T):
        final = np.median(cond[-self.REST_W:], axis=0)
        settled = np.linalg.norm(cond - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(cond - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, cond, grip, t_release):
        start = np.median(cond[: self.REST_W], axis=0)
        moved = np.linalg.norm(cond - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(cond - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        conds = np.array([h["conds"] for h in history])  # (T, 3, 3)
        grip = np.array([h["grip"] for h in history])

        ops = []
        for i in range(len(self.CONDIMENTS)):
            cond = conds[:, i, :]
            t_rel = self._release_time(cond, grip, T)
            t_dep = self._depart_time(cond, grip, t_rel)
            ops.append(dict(idx=i, lang=self.cond_lang[i], dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        # "a {name}" when more than one condiment shares that name (e.g. two
        # shakers), else "the {name}".
        counts = Counter(self.cond_lang)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for op in order:
            art = "a" if counts[op["lang"]] > 1 else "the"
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            emit(t_dep, SUBTASK_COND, STAGE_PICK,
                 f"pick up {art} {op['lang']} from the counter")
            emit(t_rel, SUBTASK_COND, STAGE_PLACE,
                 f"place {art} {op['lang']} in the cabinet")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": np.array([o["idx"] for o in order]),
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
        if n_pick != 3:
            issues.append(f"expected 3 pick stages, found {n_pick}")
        if n_place != 3:
            issues.append(f"expected 3 place stages, found {n_place}")
        return issues
