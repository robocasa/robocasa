import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each condiment is moved cabinet -> counter near the stove, split across a
# navigation into pick / NavigateKitchen / place. No exact bank name covers
# cabinet -> counter, so the PickPlace name is a declared extension; the name is
# unchanged across the pick/place stages.
SUBTASK_COND = "PickPlaceCabinetToCounter"

# NavigateKitchen (bank atomic): carry a condiment to the stove counter, and the
# drive back to the cabinet for the second condiment.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_CAB_LANG = "navigate back to the cabinet"
NAV_TO_COUNTER_LANG = "navigate to the counter near the stove"

DONE_LANG = "task complete"


@register_labeler
class LineUpCondimentsLabeler(BaseSubtaskLabeler):
    """Labeler for the LineUpCondiments composite task.

    Task: take two condiments from the open cabinet and set them on the counter
    near the stove, lined up next to each other (one round trip each).

    Doc-canonical decomposition (7 subtasks):
      1. PickPlaceCabinetToCounter [pick]   -> take the first condiment
      2. NavigateKitchen                    -> carry it to the stove counter
      3. PickPlaceCabinetToCounter [place]  -> set it down
      4. NavigateKitchen                    -> drive back to the cabinet
      5. PickPlaceCabinetToCounter [pick]   -> take the second condiment
      6. NavigateKitchen                    -> carry it to the stove counter
      7. PickPlaceCabinetToCounter [place]  -> set it next to the first
    plus a terminal done. No OpenCabinet (the cabinet starts open). The robot
    starts at the cabinet, so the first condiment has no leading nav.

    Auto-detected: which condiment is moved first (ordered by the frame it
    settles). NavigateKitchen is adaptive (MIN_NAV_DIST) -- big when the cabinet
    is across the kitchen from the stove, short/absent when they are adjacent.
    Validation is structural (two picks + two places).
    """

    ENV_NAME = "LineUpCondiments"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    CONDIMENTS = ["condiment1", "condiment2"]

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
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_COND}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "conds": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.cond_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
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
        conds = np.array([h["conds"] for h in history])  # (T, 2, 3)
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        has_base = np.isfinite(base_xy).all()

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        ops = []
        for i in range(len(self.CONDIMENTS)):
            cond = conds[:, i, :]
            t_rel = self._release_time(cond, grip, T)
            t_dep = self._depart_time(cond, grip, t_rel)
            ops.append(dict(idx=i, lang=self.cond_lang[i], dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])
        first_lang = order[0]["lang"]

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

        for n, op in enumerate(order):
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            pick_lang = f"pick up the {op['lang']} from the cabinet"
            if n == 0:
                place_lang = f"place the {op['lang']} on the counter near the stove"
                carry_lang = NAV_TO_COUNTER_LANG
            else:
                # Drive back to the cabinet for the second condiment, which then
                # joins the first where it was set down.
                maybe_nav(t_dep, NAV_TO_CAB_LANG)
                place_lang = f"place the {op['lang']} next to the {first_lang}"
                carry_lang = f"navigate to the counter where the {first_lang} was placed"
            emit(t_dep, SUBTASK_COND, STAGE_PICK, pick_lang)
            # Carry to the stove counter, then place.
            maybe_nav(t_rel, carry_lang)
            emit(t_rel, SUBTASK_COND, STAGE_PLACE, place_lang)

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
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        return issues
