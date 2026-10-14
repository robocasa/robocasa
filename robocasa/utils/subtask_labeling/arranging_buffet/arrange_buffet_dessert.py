import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Subtask 1: open the fridge door (bank atomic). Opened once; both sweets are
# inside.
SUBTASK_OPEN = "OpenFridge"
OPEN_LANG = "open the fridge door"

# Each sweet is moved fridge -> tray, split across a navigation into pick /
# NavigateKitchen / place. No exact bank name covers fridge -> tray-on-counter,
# so the PickPlace name is a declared extension; the name is unchanged across
# the pick/place stages.
SUBTASK_SWEET = "PickPlaceFridgeToCounter"

# NavigateKitchen (bank atomic): carrying a sweet to the dining counter, and the
# drive back to the fridge for the second sweet.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_FRIDGE_LANG = "navigate to the fridge"
NAV_TO_COUNTER_LANG = "navigate to the empty tray on the dining counter"

DONE_LANG = "task complete"


@register_labeler
class ArrangeBuffetDessertLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeBuffetDessert composite task.

    Task: open the fridge, then move two sweets from the fridge onto the empty
    tray on the dining counter (one round trip each). Several distractor items
    on the tray/counter are ignored.

    Doc-canonical decomposition (8 subtasks):
      1. OpenFridge                          -> open the fridge door
      2. PickPlaceFridgeToCounter [pick]     -> take the first sweet
      3. NavigateKitchen                     -> carry it to the dining counter
      4. PickPlaceFridgeToCounter [place]    -> set it on the tray
      5. NavigateKitchen                     -> drive back to the fridge
      6. PickPlaceFridgeToCounter [pick]     -> take the second sweet
      7. NavigateKitchen                     -> carry it to the dining counter
      8. PickPlaceFridgeToCounter [place]    -> set it on the tray
    plus a terminal done (9 segments).

    Auto-detected: which sweet is fetched first (ordered by the frame it settles
    on the tray; the order varies). The fridge is opened once, before the first
    sweet. NavigateKitchen segments are inserted adaptively wherever the base
    drives at least MIN_NAV_DIST (the carry to the counter and the drive back to
    the fridge are ~3-5 m; small positioning nudges are <0.6 m), so the segment
    count stays faithful to the route. Validation is structural (one OpenFridge,
    two sweet picks + two places).
    """

    ENV_NAME = "ArrangeBuffetDessert"

    # Distance a sweet must move from its fridge rest pose to count as departing.
    DEPART_TOL = 0.10
    GRASP_TH = 0.08
    # XY distance to the tray within which a sweet counts as placed.
    TRAY_TOL = 0.18
    # Gripper-to-sweet distance above which the gripper counts as released.
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    # Fridge-door open detection (summed angle, fraction of per-demo peak).
    DOOR_MIN_PEAK = 0.30
    DOOR_OPEN_FRAC = 0.85
    DOOR_START_RISE = 0.15

    # Navigation. Carry/return drives are ~3-5 m; nudges <0.6 m.
    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sweet_body_ids = [env.obj_body_id["sweet1"], env.obj_body_id["sweet2"]]
        self.tray_body_id = env.obj_body_id["tray"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Fridge door joints from the registered fixture (layout-robust).
        self.door_qpos_addrs = []
        fridge = getattr(env, "fridge", None)
        for nm in list(getattr(fridge, "_fridge_door_joint_names", [])):
            try:
                jid = env.sim.model.joint_name2id(nm)
                self.door_qpos_addrs.append(env.sim.model.jnt_qposadr[jid])
            except Exception:
                continue
        # Human-readable sweet names for the lang (e.g. "cupcake"); fall back to
        # a generic word if unavailable.
        self.sweet_lang = []
        for nm in ("sweet1", "sweet2"):
            try:
                self.sweet_lang.append(env.get_obj_lang(nm))
            except Exception:
                self.sweet_lang.append("dessert")
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_SWEET}

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        door = (
            sum(abs(env.sim.data.qpos[a]) for a in self.door_qpos_addrs)
            if self.door_qpos_addrs
            else 0.0
        )
        return {
            "sweets": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.sweet_body_ids]
            ),
            "tray": env.sim.data.body_xpos[self.tray_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
            "door": float(door),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, sweet, grip, tray, T):
        on_tray = np.linalg.norm(sweet[:, :2] - tray[:, :2], axis=1) < self.TRAY_TOL
        released = np.linalg.norm(sweet - grip, axis=1) > self.RELEASE_TH
        ok = on_tray & released
        run = 0
        for t in range(T):
            if ok[t]:
                run += 1
                if run >= self.RELEASE_SUSTAIN:
                    return t - self.RELEASE_SUSTAIN + 1
            else:
                run = 0
        w = np.where(on_tray)[0]
        return int(w[-1]) if w.size else T - 1

    def _depart_time(self, sweet, grip, t_release):
        start = np.median(sweet[: self.REST_W], axis=0)
        moved = np.linalg.norm(sweet - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(sweet - grip, axis=1) < self.GRASP_TH
        end = max(t_release, 1)
        cand = np.where((moved & held)[:end])[0]
        if cand.size:
            return int(cand[0])
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else int(t_release)

    def _door_open_time(self, door, a, b):
        b = max(b, a + 1)
        seg = door[a:b]
        peak = float(seg.max()) if seg.size else 0.0
        if peak < self.DOOR_MIN_PEAK:
            return None, peak
        w = np.where(seg >= peak * self.DOOR_OPEN_FRAC)[0]
        return (a + int(w[0]) if w.size else None), peak

    def _door_start_time(self, door, a, b):
        b = max(b, a + 1)
        base = float(door[a])
        w = np.where(door[a:b] > base + self.DOOR_START_RISE)[0]
        return a + int(w[0]) if w.size else a

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
        sweets = np.array([h["sweets"] for h in history])  # (T, 2, 3)
        tray = np.array([h["tray"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        door = np.array([h["door"] for h in history])
        has_base = np.isfinite(base_xy).all()

        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # Per-sweet pick (depart) + place (release-on-tray) events.
        ops = []
        for s in range(2):
            sweet = sweets[:, s, :]
            t_rel = self._release_time(sweet, grip, tray, T)
            t_dep = self._depart_time(sweet, grip, t_rel)
            ops.append(dict(sweet=s, dep=t_dep, rel=t_rel,
                            lang=self.sweet_lang[s] if s < len(self.sweet_lang) else "dessert"))
        order = sorted(ops, key=lambda o: o["rel"])

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

        door_peak = 0.0
        door_open_ok = True
        for n, op in enumerate(order):
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            pick_lang = f"pick up the {op['lang']} from the fridge"
            place_lang = f"place the {op['lang']} on the tray"
            if n == 0:
                # First sweet: open the fridge (once), then pick.
                t_open, door_peak = self._door_open_time(door, cursor, max(t_dep, cursor + 1))
                if t_open is None:
                    door_open_ok = False
                    t_open = cursor
                t_start = self._door_start_time(door, cursor, max(t_open, cursor + 1))
                maybe_nav(t_start, NAV_TO_FRIDGE_LANG)
                if t_open > cursor:
                    emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
                emit(t_dep, SUBTASK_SWEET, STAGE_PICK, pick_lang)
            else:
                # Second sweet: drive back to the (already-open) fridge, then pick.
                maybe_nav(t_dep, NAV_TO_FRIDGE_LANG)
                emit(t_dep, SUBTASK_SWEET, STAGE_PICK, pick_lang)
            # carry to the dining counter, then place on the tray
            maybe_nav(t_rel, NAV_TO_COUNTER_LANG)
            emit(t_rel, SUBTASK_SWEET, STAGE_PLACE, place_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": np.array([o["sweet"] for o in order]),
            "deps": np.array([o["dep"] for o in order]),
            "rels": np.array([o["rel"] for o in order]),
            "door_peak": float(door_peak),
            "door_open_ok": int(door_open_ok),
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
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_open != 1:
            issues.append(f"expected 1 OpenFridge, found {n_open}")
        if n_pick != 2:
            issues.append(f"expected 2 pick stages, found {n_pick}")
        if n_place != 2:
            issues.append(f"expected 2 place stages, found {n_place}")
        return issues
