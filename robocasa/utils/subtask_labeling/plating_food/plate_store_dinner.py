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


# Plate and store a dinner: put one steak on the plate, the other in the bowl, then open the
# fridge and store the bowl inside. Steak -> plate/bowl is the bank atomic PickPlaceStoveToCounter;
# OpenFridge is a bank atomic; bowl -> fridge (counter -> fridge) has no bank name ->
# PickPlaceCounterToFridge (extension). The bowl carries its steak into the fridge.
SUBTASK_STEAK = "PickPlaceStoveToCounter"
SUBTASK_OPEN = "OpenFridge"
SUBTASK_BOWL = "PickPlaceCounterToFridge"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

STEAKS = ("meat1", "meat2")


@register_labeler
class PlateStoreDinnerLabeler(BaseSubtaskLabeler):
    """Labeler for the PlateStoreDinner composite task.

    Task (mobile): two steaks sit in pans on the stove; a plate and a bowl sit on the counter; the
    fridge starts closed. Put one steak on the plate and the other in the bowl, then open the
    fridge and store the bowl (with its steak) inside. The robot starts at the stove.

    Doc-canonical decomposition (8, action-level). Emitted in first-motion order:
      steak_A: PickPlaceStoveToCounter [pick] -> (carry) -> [place on the plate|in the bowl]
      steak_B: PickPlaceStoveToCounter [pick] -> (carry) -> [place in the bowl|on the plate]
      (NavigateKitchen -> the fridge) -> OpenFridge [execute]
      (NavigateKitchen -> back to the counter) -> PickPlaceCounterToFridge [pick the bowl]
        -> NavigateKitchen(carry the bowl to the fridge) -> [place the bowl in the fridge]
      -> done.
    The steak that goes in the bowl rides it into the fridge, but its own move ends at its release
    (well before the bowl is carried), so it is counted once. Each steak's destination is resolved
    from the receptacle it is nearest to at release. All navs are adaptive (emitted only when the
    base actually travels).

    Boundaries: t_pick/t_move = first sustained displacement; place = the gripper RELEASE; navs end
    at the position arrival; OpenFridge ends when the fridge door(s) reach their adaptive open level.
    """

    ENV_NAME = "PlateStoreDinner"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    MIN_PICK_GAP = 12
    ARRIVE_RADIUS = 0.20
    ARRIVE_STOP_TH = 0.006
    ARRIVE_SUSTAIN = 12
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.steaks = [o for o in STEAKS if o in env.obj_body_id]
        recept = [r for r in ("plate", "bowl") if r in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.steaks + recept}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # fridge COMPARTMENT doors only (exclude freezer + drawer joints)
        fr = getattr(env, "fridge", None)
        self.door_adrs, self.door_scales = [], []
        for n in env.sim.model.joint_names:
            if not n or fr is None or fr.name not in n:
                continue
            low = n.lower()
            if "door" not in low or "freezer" in low or "drawer" in low:
                continue
            try:
                adr = env.sim.model.get_joint_qpos_addr(n)
                jid = env.sim.model.joint_name2id(n)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_BOWL}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": np.array([abs(float(env.sim.data.qpos[a])) / s
                               for a, s in zip(self.door_adrs, self.door_scales)]),
        }
        for o in self.obj_body_ids:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
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

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.obj_body_ids}
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        def arrival(start, end, park):
            if base_xy is None or end <= start:
                return None
            near = np.linalg.norm(base_xy[start: end] - park, axis=1) < self.ARRIVE_RADIUS
            stopped = binst[start: end] < self.ARRIVE_STOP_TH
            fr = self._sustained_start(near & stopped, 0, self.ARRIVE_SUSTAIN)
            if fr is not None:
                return start + fr
            mv = np.where(binst[start: end] > self.BASE_MOVE_TH)[0]
            return start + int(mv[-1]) + 1 if mv.size else None

        def move_window(o, cur):
            r = np.median(opos[o][max(cur - self.REST_W, 0): cur + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            tm = self._sustained_start(dsp >= self.MOVE_TOL, cur, self.MOVE_SUSTAIN)
            if tm is None:
                mv = np.where(dsp[cur:] >= self.MOVE_TOL)[0]
                tm = cur + int(mv[0]) if mv.size else min(cur + 1, T - 1)
            tm = max(tm, cur)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold = float(np.median(gd[tm: min(tm + self.HOLD_W, T)]))
            tp = self._sustained_start(gd > hold + self.RELEASE_MARGIN, tm + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            return tm, (tp if tp is not None else T - 1)

        review = []

        # --- the two steaks (first-motion order), each stove -> plate/bowl ---
        first_move = {}
        for o in self.steaks:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        steak_order = sorted([o for o in self.steaks if first_move[o] < T], key=lambda o: first_move[o])

        seen_steaks = 0
        for o in steak_order:
            t_move, t_placed = move_window(o, cursor)
            # destination: nearest receptacle at release
            here = opos[o][min(t_placed, T - 1)]
            dest = "the plate"
            if "plate" in opos and "bowl" in opos:
                dpl = np.linalg.norm(here - opos["plate"][min(t_placed, T - 1)])
                dbw = np.linalg.norm(here - opos["bowl"][min(t_placed, T - 1)])
                dest = "the plate" if dpl < dbw else "the bowl"
            elif "bowl" in opos:
                dest = "the bowl"
            prep = "on" if dest == "the plate" else "in"
            art = "a" if seen_steaks == 0 else "the other"
            emit(t_move, SUBTASK_STEAK, STAGE_PICK, f"pick up {art} steak from the stove")
            a = arrival(t_move, t_placed, np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)) if base_xy is not None else None
            t_arr = min(a if a is not None else t_move, t_placed - self.MIN_PICK_GAP)
            if t_arr > cursor and base_disp(t_move, t_arr) >= self.MIN_NAV_DIST:
                emit(t_arr, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the steak to {dest}")
            emit(t_placed, SUBTASK_STEAK, STAGE_PLACE, f"place the steak {prep} {dest}")
            seen_steaks += 1
        if seen_steaks != 2:
            review.append(f"only {seen_steaks}/2 steaks moved")

        # --- the bowl is carried to the fridge AFTER it is open, so resolve the bowl's move first
        #     and bound the OpenFridge detection to end before the bowl is grabbed. This keeps the
        #     door swing that happens WHEN the bowl is pushed in from stretching OpenFridge over the
        #     bowl pick. ---
        bowl_tm = bowl_tp = None
        if "bowl" in opos:
            bowl_tm, bowl_tp = move_window("bowl", cursor)
        open_hi = bowl_tm if bowl_tm is not None else T

        # --- OpenFridge: doors slide from shut to their adaptive open level (within [cursor, bowl grab]) ---
        t_open, n_doors = None, 0
        if doorv.shape[1]:
            frames = []
            for i in range(doorv.shape[1]):
                col = doorv[:, i]
                seg = col[cursor: open_hi]
                if seg.size == 0:
                    continue
                peak = float(seg.max())
                if peak < self.OPEN_MIN or peak - float(col[min(cursor, T - 1)]) < 0.10:
                    continue
                lvl = min(self.OPEN_ABS, 0.9 * peak)
                fr = self._sustained_start(col[: open_hi] >= lvl, cursor, self.OPEN_SUSTAIN)
                if fr is not None:
                    frames.append(fr)
                    n_doors += 1
            t_open = max(frames) if frames else None
            if t_open is not None:
                # extend OpenFridge to when the robot DEPARTS the fridge (the base begins its
                # drive back). The robot stays parked at the fridge finishing the open well after
                # the door first swings wide, and that stationary time belongs to OpenFridge, not
                # to the nav back.
                if base_xy is not None:
                    dep = self._sustained_start(binst > self.BASE_MOVE_TH, t_open, 5)
                    if dep is not None and dep > t_open:
                        t_open = dep
                t_open = min(t_open, max(open_hi - self.MIN_PICK_GAP, cursor + 1))
        if t_open is not None and t_open > cursor:
            # break out the drive to the fridge (before the doors move)
            begin = cursor
            door_start = None
            if doorv.shape[1]:
                anyopen = (doorv[: open_hi] > self.OPEN_MIN).any(axis=1)
                door_start = self._sustained_start(anyopen, cursor, self.OPEN_SUSTAIN)
            nav_end = door_start if door_start is not None else t_open
            if base_xy is not None:
                park = np.median(base_xy[max(nav_end - self.REST_W, 0): nav_end + 1], axis=0)
                a = arrival(begin, nav_end, park)
                t_nav = min(a if a is not None else begin, max(nav_end - self.MIN_PICK_GAP, begin))
                if t_nav > cursor and base_disp(begin, t_nav) >= self.MIN_NAV_DIST:
                    emit(t_nav, SUBTASK_NAV, STAGE_NAVIGATE, "navigate to the fridge")
            door_word = "doors" if n_doors >= 2 else "door"
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the fridge {door_word}")
        elif doorv.shape[1]:
            review.append("fridge never detected opening -- verify")

        # --- bowl -> fridge (with its steak); break out the drive back to the counter ---
        if bowl_tm is not None:
            t_move, t_placed = bowl_tm, bowl_tp
            if base_xy is not None:
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = min(a if a is not None else cursor, t_move - self.MIN_PICK_GAP)
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the counter with the bowl")
            emit(t_move, SUBTASK_BOWL, STAGE_PICK, "pick up the bowl from the counter")
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arr = min(a if a is not None else t_move, t_placed - self.MIN_PICK_GAP)
                if t_arr > cursor and base_disp(t_move, t_arr) >= self.MIN_NAV_DIST:
                    emit(t_arr, SUBTASK_NAV, STAGE_NAVIGATE, "carry the bowl to the fridge")
            emit(t_placed, SUBTASK_BOWL, STAGE_PLACE, "place the bowl in the fridge")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "steak_order": steak_order, "t_open": t_open, "n_doors": n_doors,
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_STEAK, STAGE_PICK) != 2:
            issues.append(f"expected 2 steak picks, found {count(SUBTASK_STEAK, STAGE_PICK)}")
        if count(SUBTASK_STEAK, STAGE_PLACE) != 2:
            issues.append(f"expected 2 steak places, found {count(SUBTASK_STEAK, STAGE_PLACE)}")
        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenFridge, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_BOWL, STAGE_PICK) != 1:
            issues.append(f"expected 1 bowl pick, found {count(SUBTASK_BOWL, STAGE_PICK)}")
        if count(SUBTASK_BOWL, STAGE_PLACE) != 1:
            issues.append(f"expected 1 bowl place, found {count(SUBTASK_BOWL, STAGE_PLACE)}")
        return issues
