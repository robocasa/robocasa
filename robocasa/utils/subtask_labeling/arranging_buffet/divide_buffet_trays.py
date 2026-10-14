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


# Subtask 1: open the fridge door (bank atomic). Opened once; all four foods are
# inside.
SUBTASK_OPEN = "OpenFridge"
OPEN_LANG = "open the fridge door"

# Each food is moved fridge -> its tray, split across a navigation into pick /
# NavigateKitchen / place. No exact bank name covers fridge -> tray-on-counter,
# so the PickPlace name is a declared extension; the name is unchanged across the
# pick/place stages.
SUBTASK_FOOD = "PickPlaceFridgeToCounter"

# NavigateKitchen (bank atomic): carrying a food to its tray, and the drive back
# to the fridge for the next food.
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_FRIDGE_LANG = "navigate to the fridge"

DONE_LANG = "task complete"


@register_labeler
class DivideBuffetTraysLabeler(BaseSubtaskLabeler):
    """Labeler for the DivideBuffetTrays composite task.

    Task: open the fridge, then move four foods from the fridge onto two trays on
    the dining counter -- the two vegetables onto one tray, the two meats onto
    the other (one round trip each). Two distractor cups are ignored.

    Doc-canonical decomposition (16 subtasks):
      1.  OpenFridge                          -> open the fridge door
      then, for each of the four foods in performed order:
          PickPlaceFridgeToCounter [pick]     -> take the food
          NavigateKitchen                     -> carry it to its tray
          PickPlaceFridgeToCounter [place]    -> set it on the tray
      with a NavigateKitchen (back to the fridge) before each food after the
      first. That is OpenFridge(1) + 4*(pick + nav + place)=12 + 3 returns = 16,
      plus a terminal done (17 segments).

    Auto-detected: the order the foods are fetched (they go in two type-groups --
    both vegetables then both meats, or vice versa -- but which group is first
    varies; ordered by the frame each food settles on a tray). The fridge is
    opened once, before the first food. Each food's lang uses its real name (via
    env.get_obj_lang) and its tray is named by food type ("the vegetable tray" /
    "the meat tray"). NavigateKitchen is adaptive (MIN_NAV_DIST); the
    fridge<->tray carries are long traversals. Validation is structural (one
    OpenFridge, four picks + four places, two vegetables + two meats).
    """

    ENV_NAME = "DivideBuffetTrays"

    DEPART_TOL = 0.10
    GRASP_TH = 0.08
    # XY distance to a tray within which a food counts as placed (trays are wide).
    TRAY_TOL = 0.20
    RELEASE_TH = 0.12
    RELEASE_SUSTAIN = 5
    REST_W = 10

    DOOR_MIN_PEAK = 0.30
    DOOR_OPEN_FRAC = 0.85
    DOOR_START_RISE = 0.15

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    FOODS = ["vegetable1", "vegetable2", "meat1", "meat2"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.food_body_ids = [env.obj_body_id[n] for n in self.FOODS]
        self.food_type = ["vegetable" if n.startswith("vegetable") else "meat"
                          for n in self.FOODS]
        self.food_lang = []
        for n in self.FOODS:
            try:
                self.food_lang.append(env.get_obj_lang(n))
            except Exception:
                self.food_lang.append(n.rstrip("12"))
        self.tray_body_ids = [
            env.obj_body_id["tray1_container"],
            env.obj_body_id["tray2_container"],
        ]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.door_qpos_addrs = []
        fridge = getattr(env, "fridge", None)
        for nm in list(getattr(fridge, "_fridge_door_joint_names", [])):
            try:
                jid = env.sim.model.joint_name2id(nm)
                self.door_qpos_addrs.append(env.sim.model.jnt_qposadr[jid])
            except Exception:
                continue
        self.expected_done_idx = None
        self.extra_subtasks = {SUBTASK_FOOD}

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
            "foods": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.food_body_ids]
            ),
            "trays": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.tray_body_ids]
            ),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": base_xy,
            "door": float(door),
        }

    # ---- event helpers -------------------------------------------------

    def _release_time(self, food, grip, tray, T):
        on_tray = np.linalg.norm(food[:, :2] - tray[:, :2], axis=1) < self.TRAY_TOL
        released = np.linalg.norm(food - grip, axis=1) > self.RELEASE_TH
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

    def _depart_time(self, food, grip, t_release):
        start = np.median(food[: self.REST_W], axis=0)
        moved = np.linalg.norm(food - start, axis=1) > self.DEPART_TOL
        held = np.linalg.norm(food - grip, axis=1) < self.GRASP_TH
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
        foods = np.array([h["foods"] for h in history])    # (T, 4, 3)
        trays = np.array([h["trays"] for h in history])    # (T, 2, 3)
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

        # Per-food pick (depart) + place (release on nearest tray) events.
        t_final = trays[-1]  # (2, 3)
        ops = []
        for i in range(len(self.FOODS)):
            food = foods[:, i, :]
            tgt = 0 if (np.linalg.norm(food[-1, :2] - t_final[0, :2])
                        <= np.linalg.norm(food[-1, :2] - t_final[1, :2])) else 1
            tray = trays[:, tgt, :]
            t_rel = self._release_time(food, grip, tray, T)
            t_dep = self._depart_time(food, grip, t_rel)
            ops.append(dict(
                idx=i, type=self.food_type[i], lang=self.food_lang[i],
                tgt=tgt, dep=t_dep, rel=t_rel))
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

        # "a {food}" when the two foods of a type are the same (e.g. two fish);
        # "the {food}" when they differ (and are individually identifiable).
        type_langs = {}
        for o in ops:
            type_langs.setdefault(o["type"], []).append(o["lang"])
        type_dup = {t: len(set(ls)) == 1 for t, ls in type_langs.items()}

        # The trays are not pre-labeled by type: the FIRST food of a type goes to
        # "an empty tray", and the SECOND references the tray the first established.
        seen_type = {}  # type -> lang of the first food of that type placed

        door_peak = 0.0
        door_open_ok = True
        for n, op in enumerate(order):
            t_dep = max(op["dep"], cursor)
            t_rel = max(op["rel"], t_dep)
            art = "a" if type_dup[op["type"]] else "the"
            if op["type"] not in seen_type:
                # First food of this type. Both trays are empty for the first type
                # handled ("an empty tray"); only one remains empty for the second
                # ("the empty tray").
                tray_ref = (
                    "an empty tray on the dining counter"
                    if len(seen_type) == 0
                    else "the empty tray on the dining counter"
                )
                seen_type[op["type"]] = op["lang"]
            else:
                tray_ref = f"the tray where the {seen_type[op['type']]} was placed"
            # Pick is indefinite when the two foods of a type match ("pick up a
            # fish" -- one of two); once it is in hand the place is definite
            # ("place the fish").
            pick_lang = f"pick up {art} {op['lang']} from the fridge"
            place_lang = f"place the {op['lang']} on {tray_ref}"
            nav_tray_lang = f"navigate to {tray_ref}"
            if n == 0:
                t_open, door_peak = self._door_open_time(door, cursor, max(t_dep, cursor + 1))
                if t_open is None:
                    door_open_ok = False
                    t_open = cursor
                t_start = self._door_start_time(door, cursor, max(t_open, cursor + 1))
                maybe_nav(t_start, NAV_TO_FRIDGE_LANG)
                if t_open > cursor:
                    emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
                emit(t_dep, SUBTASK_FOOD, STAGE_PICK, pick_lang)
            else:
                maybe_nav(t_dep, NAV_TO_FRIDGE_LANG)
                emit(t_dep, SUBTASK_FOOD, STAGE_PICK, pick_lang)
            maybe_nav(t_rel, nav_tray_lang)
            emit(t_rel, SUBTASK_FOOD, STAGE_PLACE, place_lang)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        tgts = [o["tgt"] for o in order]
        debug = {
            "order": np.array([o["idx"] for o in order]),
            "order_types": np.array([o["type"] for o in order]),
            "targets": np.array(tgts),
            "n_veg": int(sum(1 for o in order if o["type"] == "vegetable")),
            "n_meat": int(sum(1 for o in order if o["type"] == "meat")),
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
        if n_pick != 4:
            issues.append(f"expected 4 pick stages, found {n_pick}")
        if n_place != 4:
            issues.append(f"expected 4 place stages, found {n_place}")
        n_veg = result.debug.get("n_veg")
        n_meat = result.debug.get("n_meat")
        if n_veg is not None and (n_veg != 2 or n_meat != 2):
            issues.append(f"expected 2 vegetables + 2 meats, found {n_veg}/{n_meat}")
        # the two vegetables share a tray and the two meats share the other
        tgts = result.debug.get("targets")
        types = result.debug.get("order_types")
        if tgts is not None and types is not None:
            vt = {int(t) for t, ty in zip(tgts, types) if ty == "vegetable"}
            mt = {int(t) for t, ty in zip(tgts, types) if ty == "meat"}
            if len(vt) != 1 or len(mt) != 1 or vt == mt:
                issues.append(f"vegetables/meats not cleanly split across trays (veg {vt}, meat {mt})")
        return issues
