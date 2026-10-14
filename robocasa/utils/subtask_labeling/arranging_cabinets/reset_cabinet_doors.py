import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Close a cabinet's open door(s). The bank has OpenCabinet but no close variant,
# so CloseCabinet is a declared extension. Single stage. One CloseCabinet per
# cabinet -- a double-door cabinet's two doors are one "close the cabinet doors"
# action (even when shut a few hundred frames apart); a single-door cabinet is
# "close the cabinet door".
SUBTASK_CLOSE = "CloseCabinet"
CLOSE_ONE_LANG = "close the cabinet door"
CLOSE_BOTH_LANG = "close the cabinet doors"

# NavigateKitchen (bank atomic): drive between cabinets (and the initial drive
# from the counter to the first cabinet).
SUBTASK_NAV = "NavigateKitchen"
NAV_FIRST_LANG = "navigate to an open cabinet"
NAV_NEXT_LANG = "navigate to the next open cabinet"

DONE_LANG = "task complete"


@register_labeler
class ResetCabinetDoorsLabeler(BaseSubtaskLabeler):
    """Labeler for the ResetCabinetDoors composite task.

    Task: close all the open cabinet doors. Three cabinet references are opened
    (`cab`, `cab2`, `cab3`), but they may not be distinct (the env can sample the
    same cabinet twice), so the scene has 2-3 distinct cabinets; each is single-
    or double-doored. A distractor object on the counter is ignored.

    Doc-canonical subtask count: **5** -- the common case is two double-door
    cabinets: close door, close door, NavigateKitchen, close door, close door
    (4 closes + 1 nav). The realised count tracks the actual scene (one
    CloseCabinet per open door; a NavigateKitchen wherever the base drives between
    cabinets), so single-door cabinets and 3-cabinet scenes vary.

    Boundaries:
      - close_time:  the door's |hinge angle| last exceeds CLOSE_TH, +1 (the frame
                     it becomes and stays closed; robust to momentary dips).
      - close_start: the last frame the door was still substantially open
                     (>= CLOSE_HI_FRAC of its peak) before close_time -- bounds the
                     preceding nav so a drive can't swallow the closing motion.
      - NavigateKitchen segments are carved where the base drives at least
        MIN_NAV_DIST (between cabinets; not between the two doors of one cabinet,
        where the base is stationary). Doors are handled in the order they close.
        Validation is structural (one close per open door; all doors end closed).
    """

    ENV_NAME = "ResetCabinetDoors"

    # A door counts as closed when |hinge angle| < CLOSE_TH (radians).
    CLOSE_TH = 0.05
    CLOSE_HI_FRAC = 0.7

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        # Distinct cabinets (cab/cab2/cab3 may repeat), each with its door hinges.
        cabs = [getattr(env, "cab", None), getattr(env, "cab2", None),
                getattr(env, "cab3", None)]
        jnames = [env.sim.model.joint_id2name(j) for j in range(env.sim.model.njnt)]
        self.doors = []  # list of (cabinet_name, qpos_addr)
        seen = set()
        for cab in cabs:
            nm = getattr(cab, "name", None)
            if nm is None or nm in seen:
                continue
            seen.add(nm)
            for jn in jnames:
                if jn and jn.startswith(nm) and "doorhinge" in jn:
                    addr = env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(jn)]
                    self.doors.append((nm, int(addr)))
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_CLOSE}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        return {
            "doors": np.array([abs(env.sim.data.qpos[a]) for _, a in self.doors]),
            "base_xy": base_xy,
        }

    # ---- helpers -------------------------------------------------------

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
        doors = np.array([h["doors"] for h in history])  # (T, n_doors)
        base_xy = np.array([h["base_xy"] for h in history])
        has_base = np.isfinite(base_xy).all()
        n_doors = doors.shape[1] if doors.ndim == 2 else 0

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Per-door close time = last frame still open, +1.
        per_door = []
        for d in range(n_doors):
            ang = doors[:, d]
            above = np.where(ang > self.CLOSE_TH)[0]
            close_time = int(above[-1]) + 1 if above.size else 0
            per_door.append((min(close_time, T), d))
        per_door.sort()

        # Group doors by cabinet: each cabinet is one close action covering all its
        # doors (a double-door cabinet -> "close the cabinet doors"). Doors of a
        # cabinet close consecutively, so group consecutive same-cabinet doors in
        # close order.
        clusters = []  # each: {cab, close, doors: [idx]}
        for ct, d in per_door:
            cab = self.doors[d][0]
            if clusters and clusters[-1]["cab"] == cab:
                clusters[-1]["doors"].append(d)
                clusters[-1]["close"] = max(clusters[-1]["close"], ct)
            else:
                clusters.append({"cab": cab, "close": ct, "doors": [d]})

        segments = []
        cursor = 0
        n_navs = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang):
            nonlocal n_navs
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)
                n_navs += 1

        for cl in clusters:
            t_close = max(cl["close"], cursor)
            # close_start: last frame any door in the cluster was substantially
            # open (bounds the preceding nav so it can't swallow the closing).
            csum = doors[:, cl["doors"]].sum(axis=1)
            seg = csum[cursor:max(t_close, cursor + 1)]
            peak = float(seg.max()) if seg.size else 0.0
            hi = np.where(seg >= self.CLOSE_HI_FRAC * peak)[0] if peak > 0 else np.array([])
            close_start = cursor + int(hi[-1]) if hi.size else cursor
            nav_lang = NAV_FIRST_LANG if n_navs == 0 else NAV_NEXT_LANG
            maybe_nav(close_start, nav_lang)
            close_lang = CLOSE_BOTH_LANG if len(cl["doors"]) > 1 else CLOSE_ONE_LANG
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, close_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        all_closed = bool(np.all(doors[-1] < self.CLOSE_TH)) if n_doors else True
        debug = {
            "n_doors": int(n_doors),
            "n_closes": int(len(clusters)),
            "n_navs": int(n_navs),
            "all_closed": int(all_closed),
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
        names = result.subtask_name[firsts]
        n_close = int(np.sum(names == SUBTASK_CLOSE))
        n_closes = result.debug.get("n_closes", 0)
        if n_closes and n_close != n_closes:
            issues.append(f"expected {n_closes} CloseCabinet, found {n_close}")
        if result.debug.get("all_closed") == 0:
            issues.append("not all doors ended closed")
        return issues
