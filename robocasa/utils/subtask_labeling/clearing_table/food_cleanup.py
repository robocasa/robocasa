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


# Put the food items from the counter into the (already-open) cabinet, then close the
# cabinet. PickPlaceCounterToCabinet + CloseCabinet are bank atomics. No navigation
# (the counter is beside the cabinet, mobile: No).
SUBTASK_MOVE = "PickPlaceCounterToCabinet"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"


@register_labeler
class FoodCleanupLabeler(BaseSubtaskLabeler):
    """Labeler for the FoodCleanup composite task.

    Task: 1-3 food items sit on the counter; a cabinet is ALREADY OPEN beside it.
    Put every food item into the cabinet, then (for a door cabinet) close it. Base
    never drives (mobile: No).

    Doc-canonical count = 2 (`mobile: No`): one PickPlaceCounterToCabinet per food +
    CloseCabinet. The food count VARIES 1-3, so expected_done_idx is set per demo;
    validation is structural. The ORDER varies (by first move). CloseCabinet is
    emitted only when the cabinet actually closes (some cabinet types have no door).

    Boundaries: per food, first-move / relative-release split (t_move = first
    displacement from rest; t_placed = the gripper RELEASE). CloseCabinet spans from
    the last food's placement to when the door returns (near) shut.
    """

    ENV_NAME = "FoodCleanup"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20
    CLOSE_ABS = 0.30  # door fraction (of its open range) below this == (near) shut
    CLOSE_SUSTAIN = 5
    OPEN_MIN = 0.25  # door open peak below this == no real door to close

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.foods = tuple(
            n for n in env.obj_body_id if n.startswith("food")
        )
        self.foods = tuple(sorted(self.foods, key=lambda s: int(s[4:]) if s[4:].isdigit() else 0))
        self.num_food = len(self.foods)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.foods}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.foods:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "food item"

        # cabinet door joints (for CloseCabinet); track each as a fraction of its
        # own open range (see FillKettle).
        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
            pref = cab.name
            joint_names = [
                n for n in env.sim.model.joint_names
                if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower() or "slide" in n.lower())
            ]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_adrs.append(adr)
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": np.array(
                [abs(float(env.sim.data.qpos[a])) / s
                 for a, s in zip(self.door_adrs, self.door_scales)]
            ),
        }
        for o in self.foods:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.foods}
        grip = np.array([h["grip"] for h in history])
        doorv = np.array([h["doors"] for h in history]) if self.door_adrs else np.zeros((T, 0))

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.foods}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.foods}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.foods}

        first_move = {}
        for o in self.foods:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(self.foods, key=lambda o: first_move[o])

        words = [self.obj_lang[o] for o in order if first_move[o] < T]
        identical = len(set(words)) <= 1
        moved = [o for o in order if first_move[o] < T]
        n = len(moved)

        def food_noun(k, o):
            word = self.obj_lang[o]
            if not identical:
                return f"the {word}"
            if n == 1:
                return f"the {word}"
            if k == 0:
                return f"a {word}"
            if k == n - 1:
                return f"the last {word}" if n >= 3 else f"the other {word}"
            return f"another {word}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            if first_move[o] >= T:
                continue
            t_move = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            noun = food_noun(n_moved, o)
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} in the open cabinet above")
            n_moved += 1

        # CloseCabinet: the door swings shut after the last food is placed. Emitted
        # only if there is a door that actually opens (peak >= OPEN_MIN) and closes.
        door_peak = float(doorv.max()) if doorv.size else 0.0
        has_close = False
        if door_peak >= self.OPEN_MIN and cursor < T:
            dmax = doorv.max(axis=1)  # most-open leaf per frame
            t_closed = self._sustained_start(dmax < self.CLOSE_ABS, cursor, self.CLOSE_SUSTAIN)
            if t_closed is not None and t_closed > cursor:
                emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, "close the cabinet")
                has_close = True

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != self.num_food:
            review.append(f"only {n_moved}/{self.num_food} foods moved")
        # a doorless cabinet (open shelves) legitimately has no CloseCabinet -- only
        # flag when there IS a door (it opened) but it was never closed.
        if door_peak >= self.OPEN_MIN and not has_close:
            review.append("cabinet has a door but was never closed -- verify")

        result = self.build_result(T, segments, debug={
            "order": ",".join(order), "n_moved": n_moved,
            "door_peak": round(door_peak, 2), "has_close": has_close,
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

        n_pick, n_place = count(SUBTASK_MOVE, STAGE_PICK), count(SUBTASK_MOVE, STAGE_PLACE)
        if n_pick != n_place:
            issues.append(f"food picks ({n_pick}) != places ({n_place})")
        if not (1 <= n_pick <= self.num_food):
            issues.append(f"expected 1-{self.num_food} food moves, found {n_pick}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseCabinet, found {count(SUBTASK_CLOSE)}")
        return issues
