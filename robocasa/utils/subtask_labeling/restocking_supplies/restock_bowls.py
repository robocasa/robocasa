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


# Subtask 1: open the cabinet (bank atomic). The cabinet starts closed; the robot
# opens it (both doors of the double-door cabinet) before stocking the bowls.
SUBTASK_OPEN = "OpenCabinet"
OPEN_LANG = "open the cabinet"

# Subtasks 2-3: each bowl is moved counter -> cabinet (bank atomic, splits into
# pick/place). Non-mobile task, so no navigation interrupts the pick-place.
SUBTASK_BOWL = "PickPlaceCounterToCabinet"

# Subtask 4: close the cabinet. The bank has OpenCabinet but no close variant, so
# CloseCabinet is a declared extension. Single stage.
SUBTASK_CLOSE = "CloseCabinet"
CLOSE_LANG = "close the cabinet"

DONE_LANG = "task complete"


@register_labeler
class RestockBowlsLabeler(BaseSubtaskLabeler):
    """Labeler for the RestockBowls composite task.

    Task (non-mobile): open the cabinet, move two bowls from the counter into the
    cabinet, then close the cabinet. The cabinet starts closed.

    Doc-canonical decomposition (4 subtasks):
      1. OpenCabinet                          -> open the cabinet
      2. PickPlaceCounterToCabinet [pick/place] -> stock the first bowl
      3. PickPlaceCounterToCabinet [pick/place] -> stock the second bowl
      4. CloseCabinet                         -> close the cabinet
    plus a terminal done. No navigation (the robot stays at the cabinet).

    Boundaries:
      - t_open:  the summed door angle first reaches OPEN_FRAC of its per-demo peak
                 (door open, stocking can begin).
      - per bowl: t_pick = the bowl departs its counter rest while held; t_place =
                 the bowl settles in the cabinet with the gripper released. Bowls
                 are ordered by placement; the cabinet stays open throughout.
      - t_close: the door's last open frame, +1 (it closes after both bowls are in).
        Validation is structural (one OpenCabinet, two picks + two places, one
        CloseCabinet; the cabinet ends closed).
    """

    ENV_NAME = "RestockBowls"

    DEPART_TOL = 0.10
    GRASP_TH = 0.10
    PLACE_TOL = 0.10
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    # Door (summed |hinge angle|): open when it reaches OPEN_FRAC of its peak,
    # closed when below CLOSE_TH.
    DOOR_MIN_PEAK = 0.5
    OPEN_FRAC = 0.85
    CLOSE_TH = 0.05

    BOWLS = ["obj1", "obj2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.bowl_body_ids = [env.obj_body_id[n] for n in self.BOWLS]
        self.bowl_lang = []
        for n in self.BOWLS:
            try:
                self.bowl_lang.append(env.get_obj_lang(n))
            except Exception:
                self.bowl_lang.append("bowl")
        # Cabinet door hinges (model joints under the cabinet fixture's name).
        cab = getattr(env, "cab", None)
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
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_CLOSE}

    def per_step(self, env, t: int) -> dict:
        door = (
            sum(abs(env.sim.data.qpos[a]) for a in self.door_addrs)
            if self.door_addrs
            else 0.0
        )
        return {
            "bowls": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.bowl_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(door),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, bowl, grip, T):
        final = np.median(bowl[-self.REST_W:], axis=0)
        settled = np.linalg.norm(bowl - final, axis=1) < self.PLACE_TOL
        released = np.linalg.norm(bowl - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, bowl, grip, t_release):
        start = np.median(bowl[: self.REST_W], axis=0)
        moved = np.linalg.norm(bowl - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(bowl - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        bowls = np.array([h["bowls"] for h in history])  # (T, 2, 3)
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])

        peak = float(door.max()) if T else 0.0
        open_ok = peak >= self.DOOR_MIN_PEAK
        w = np.where(door >= self.OPEN_FRAC * peak)[0] if open_ok else np.array([])
        t_open = int(w[0]) if w.size else 0
        # Close = last open frame, +1.
        above = np.where(door > self.CLOSE_TH)[0]
        t_close = int(above[-1]) + 1 if above.size else T
        t_close = min(t_close, T)

        ops = []
        for i in range(len(self.BOWLS)):
            bowl = bowls[:, i, :]
            t_rel = self._release_time(bowl, grip, T)
            t_dep = self._depart_time(bowl, grip, t_rel)
            ops.append(dict(idx=i, lang=self.bowl_lang[i], dep=t_dep, rel=t_rel))
        order = sorted(ops, key=lambda o: o["rel"])

        # "a bowl" when the two bowls share a name (they usually do), else "the".
        same = len({o["lang"] for o in ops}) == 1

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        for n, op in enumerate(order):
            # Two identical bowls: the first is "a bowl" (one of two), the second
            # is "the bowl" (the remaining one). Distinct names are always "the".
            art = "a" if (same and n == 0) else "the"
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            emit(t_dep, SUBTASK_BOWL, STAGE_PICK,
                 f"pick up {art} {op['lang']} from the counter")
            emit(t_rel, SUBTASK_BOWL, STAGE_PLACE,
                 f"place the {op['lang']} in the cabinet")
        emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        closed = bool(door[-1] < self.CLOSE_TH) if T else True
        debug = {
            "t_open": int(t_open),
            "t_close": int(t_close),
            "door_peak": float(peak),
            "open_ok": int(open_ok),
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
        n_open = int(np.sum(names == SUBTASK_OPEN))
        n_close = int(np.sum(names == SUBTASK_CLOSE))
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_open != 1:
            issues.append(f"expected 1 OpenCabinet, found {n_open}")
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        if n_close != 1:
            issues.append(f"expected 1 CloseCabinet, found {n_close}")
        if result.debug.get("closed_at_end") == 0:
            issues.append("cabinet did not end closed")
        return issues
