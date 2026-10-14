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


# Set up a hot dog on the dining table: bun from the counter onto the plate, condiment from the
# (open) cabinet next to the plate, and sausage from the (closed) fridge onto the plate -- so the
# sausage move is preceded by opening the fridge. Source-appropriate atomics: counter ->
# PickPlaceCounterToCounter (ext), cabinet -> PickPlaceCabinetToCounter (bank), fridge ->
# PickPlaceFridgeToCounter (ext); OpenFridge + NavigateKitchen are bank atomics.
SUBTASK_COUNTER = "PickPlaceCounterToCounter"
SUBTASK_CABINET = "PickPlaceCabinetToCounter"
SUBTASK_FRIDGE = "PickPlaceFridgeToCounter"
SUBTASK_OPEN = "OpenFridge"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

PLATE_LM = "the plate on the dining table"


@register_labeler
class HotDogSetupLabeler(BaseSubtaskLabeler):
    """Labeler for the HotDogSetup composite task.

    Task (mobile): a hot dog bun sits on the counter (robot starts there), a condiment in the open
    cabinet, a sausage in the closed fridge, and a plate on the dining table. Put the bun and
    sausage on the plate and the condiment beside it.

    Doc-canonical count = 12 (`mobile: Yes`): the three items' nav-interrupted moves, plus opening
    the fridge before the sausage. Each item, in first-motion order:
      (NavigateKitchen -> its source) -> [OpenFridge if the fridge] -> item[pick]
      -> NavigateKitchen(carry to the dining table) -> item[place]
    plus a terminal done. A lead-in nav is emitted whenever the base actually travels (the bun is
    at the start counter, so it gets none).

    Boundaries: t_pick = first sustained displacement; navs end at the position arrival; OpenFridge
    ends when the fridge door(s) reach their adaptive open level (bounded before the sausage grab);
    place = the gripper RELEASE.
    """

    ENV_NAME = "HotDogSetup"

    # object -> (pick_from, place prep, subtask, needs_open_fridge, lead-in nav phrase)
    ITEMS = {
        "hotdog_bun": ("the plate on the counter", "on", SUBTASK_COUNTER, False, "navigate to the counter"),
        "condiment": ("the cabinet", "next to", SUBTASK_CABINET, False, "navigate back to the open cabinet"),
        "sausage": ("the fridge", "on", SUBTASK_FRIDGE, True, "navigate to the fridge"),
    }

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.20
    MIN_PICK_GAP = 12
    ARRIVE_RADIUS = 0.20
    ARRIVE_STOP_TH = 0.006
    ARRIVE_SUSTAIN = 12
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in self.ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: (env.get_obj_lang(o) if o == "condiment" else
                         ("hot dog bun" if o == "hotdog_bun" else o)) for o in self.names}
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
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_COUNTER, SUBTASK_FRIDGE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": np.array([abs(float(env.sim.data.qpos[a])) / s
                               for a, s in zip(self.door_adrs, self.door_scales)]),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.names:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0
        review = []

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

        def open_fridge_before(t_pick):
            # detect the fridge doors reaching their adaptive open level within [cursor, t_pick).
            # Returns (t_open, n_doors, door_start): door_start is the first frame any door starts
            # opening -- the drive to the fridge ends there, so OpenFridge is not swallowed by the nav.
            if not doorv.shape[1]:
                return None, 0, None
            frames, n = [], 0
            for i in range(doorv.shape[1]):
                col = doorv[:, i]
                seg = col[cursor: t_pick]
                if seg.size == 0:
                    continue
                peak = float(seg.max())
                if peak < self.OPEN_MIN or peak - float(col[min(cursor, T - 1)]) < 0.10:
                    continue
                lvl = min(self.OPEN_ABS, 0.9 * peak)
                fr = self._sustained_start(col[:t_pick] >= lvl, cursor, self.OPEN_SUSTAIN)
                if fr is not None:
                    frames.append(fr)
                    n += 1
            if not frames:
                return None, 0, None
            anyopen = (doorv[:t_pick] > self.OPEN_MIN).any(axis=1)
            door_start = self._sustained_start(anyopen, cursor, self.OPEN_SUSTAIN)
            t_open = int(min(max(frames), max(t_pick - self.MIN_PICK_GAP, cursor + 1)))
            return t_open, n, door_start

        n_moved = 0
        for o in order:
            pick_from, prep, sub, needs_open, nav_phrase = self.ITEMS[o]
            word = self.word[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            # for a fridge item, find the open first so the lead-in nav ends where the door starts
            # opening (the robot has arrived), not at the later grab.
            t_open, n_doors, door_start = (None, 0, None)
            if needs_open:
                t_open, n_doors, door_start = open_fridge_before(t_move)
            nav_target = door_start if (needs_open and door_start is not None) else t_move
            # lead-in nav to the source (whenever the base actually travels there)
            if base_xy is not None and nav_target > cursor:
                src_park = np.median(base_xy[max(nav_target - self.REST_W, 0): nav_target + 1], axis=0)
                a = arrival(cursor, nav_target, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, nav_target - 1, t_move - self.MIN_PICK_GAP)
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, nav_phrase)
            # open the fridge (for the fridge item), after arriving and before the grab
            if needs_open:
                if t_open is not None and t_open > cursor:
                    door_word = "doors" if n_doors >= 2 else "door"
                    emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the fridge {door_word}")
                elif doorv.shape[1]:
                    review.append("fridge never detected opening -- verify")
            emit(t_move, sub, STAGE_PICK, f"pick up the {word} from {pick_from}")
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to {PLATE_LM}")
            emit(t_placed, sub, STAGE_PLACE, f"place the {word} {prep} the plate")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 places, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_OPEN)) not in (0, 1):
            issues.append(f"expected 0-1 OpenFridge, found {int(np.sum(names == SUBTASK_OPEN))}")
        return issues
