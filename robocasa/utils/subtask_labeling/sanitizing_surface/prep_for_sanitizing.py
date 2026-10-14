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


# Open the (closed) cabinet, then take two cleaners out onto the counter. OpenCabinet
# and PickPlaceCabinetToCounter are bank atomics. Non-mobile task -- no navigation and
# no close (success only needs both cleaners on the counter).
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToCounter"

OPEN_LANG = "open the cabinet"
DONE_LANG = "task complete"

OBJS = ("obj1", "obj2")


@register_labeler
class PrepForSanitizingLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepForSanitizing composite task.

    Task (non-mobile): two cleaners sit in a CLOSED cabinet; the robot starts at the
    cabinet. Open the cabinet, then move both cleaners onto the counter.

    Doc-canonical count = 3 (`mobile: No`), action-level: OpenCabinet + two
    cabinet->counter moves. Structure:
      OpenCabinet [execute]                     -> open the cabinet
      PickPlaceCabinetToCounter [pick/place] x2 -> take each cleaner out
    plus a terminal done. No navigation (the robot stays at the cabinet).

    Boundaries: OpenCabinet ends when the summed door angle reaches OPEN_FRAC of its
    per-demo peak (capped before the first cleaner departs); per cleaner t_pick = the
    first sustained displacement from a rest re-anchored at the segment start; t_place
    = the gripper RELEASE. Cleaners are ordered by first move.
    """

    ENV_NAME = "PrepForSanitizing"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    DOOR_MIN_PEAK = 0.5
    OPEN_FRAC = 0.85
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {}
        for o in self.names:
            try:
                self.word[o] = env.get_obj_lang(o)
            except Exception:
                self.word[o] = "cleaner"
        # This task's cabinet is a full-height cabinet in front (uniform across the
        # dataset, cab_z ~1.84), so the open language is fixed.
        self.open_lang = "open the cabinet in front"
        # Cabinet door hinges.
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
        self.extra_subtasks = set()  # both names are bank atomics
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        door = (sum(abs(env.sim.data.qpos[a]) for a in self.door_addrs)
                if self.door_addrs else 0.0)
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(door),
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
        door = np.array([h["door"] for h in history])

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])
        first_pick = first_move[order[0]] if order else T

        # OpenCabinet: door reaches OPEN_FRAC of its peak, before the first cleaner departs.
        peak = float(door.max()) if T else 0.0
        has_open = peak >= self.DOOR_MIN_PEAK
        t_open = 0
        review = []
        if has_open:
            lvl = self.OPEN_FRAC * peak
            fo = self._sustained_start(door >= lvl, 0, self.OPEN_SUSTAIN)
            if fo is None:
                w = np.where(door >= lvl)[0]
                fo = int(w[0]) if w.size else 0
            t_open = int(min(fo, max(first_pick - 1, 0)))
        else:
            review.append("cabinet door never opened -- verify")

        # two cleaners share a type -> "a"/"the other"; distinct -> "the {word}"
        moved_words = [self.word[o] for o in order]
        same = len(order) == 2 and moved_words[0] == moved_words[1]

        def noun(o, k):
            w = self.word[o]
            if same:
                return f"the other {w}" if k == 1 else f"a {w}"
            return f"the {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if has_open and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, self.open_lang)

        n_moved = 0
        for k, o in enumerate(order):
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            nn = noun(o, k)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the cabinet")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the counter")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 cleaners moved")

        result = self.build_result(T, segments, debug={
            "order": order, "t_open": int(t_open), "n_moved": n_moved,
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
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        if int(np.sum(names == SUBTASK_OPEN)) != 1:
            issues.append(f"expected 1 OpenCabinet, found {int(np.sum(names == SUBTASK_OPEN))}")
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
