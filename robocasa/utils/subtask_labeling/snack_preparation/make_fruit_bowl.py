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


# Two fruit sit in a cabinet (the robot spawns at the cabinet); a bowl sits on the counter.
# Open the cabinet, put both fruit into the bowl, then close the cabinet. OpenCabinet and
# CloseCabinet are bank atomics; cabinet -> bowl has no bank name -> PickPlaceCabinetToBowl
# (extension). Non-mobile. The cabinet's starting state is not forced by the env, so the
# open/close subtasks are emitted ONLY when the door actually swings (adaptive).
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToBowl"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"

CAB_ABOVE_Z = 1.80
OBJS = ("fruit1", "fruit2")


@register_labeler
class MakeFruitBowlLabeler(BaseSubtaskLabeler):
    """Labeler for the MakeFruitBowl composite task.

    Task (non-mobile): two fruit sit in a cabinet (the robot spawns at the cabinet); a bowl
    sits on the counter. Open the cabinet, put both fruit into the bowl, then close the
    cabinet.

    Doc-canonical count = 3 (`mobile: No`): the two fruit moves cover the picking/placing;
    OpenCabinet / CloseCabinet bracket them. The env does not force the cabinet's starting
    state, so OpenCabinet is emitted only when the door STARTS closed and opens, and
    CloseCabinet only when the door ENDS closed.

    Boundaries: OpenCabinet ends at OPEN_FRAC of the door peak (before the first fruit
    departs); per fruit t_pick = first sustained displacement, t_place = the gripper
    RELEASE; CloseCabinet runs from the closing onset (after the last place) until latched.
    """

    ENV_NAME = "MakeFruitBowl"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    DOOR_MIN_PEAK = 0.5
    OPEN_FRAC = 0.85
    CLOSE_FRAC = 0.30
    OPEN_SUSTAIN = 5
    OPEN_DOOR_TH = 0.5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = "the fruit"
        cab = getattr(env, "cab", None) or getattr(env, "cabinet", None)
        nm = getattr(cab, "name", None)
        try:
            self.open_position = "above" if float(cab.pos[2]) >= CAB_ABOVE_Z else "in front"
        except Exception:
            self.open_position = "in front"
        self.door_addrs = []
        if nm is not None:
            for j in range(env.sim.model.njnt):
                try:
                    jn = env.sim.model.joint_id2name(j)
                except Exception:
                    jn = None
                if jn and jn.startswith(nm) and "doorhinge" in jn:
                    self.door_addrs.append(env.sim.model.jnt_qposadr[j])
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = (np.array([abs(float(env.sim.data.qpos[a])) for a in self.door_addrs])
                 if self.door_addrs else np.zeros(0))
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy(), "doors": doors}
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
        door = doorv.sum(axis=1) if doorv.shape[1] else np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])
        t_first_pick = first_move[order[0]] if order else max(T - 1, 1)

        peak = float(door.max()) if T else 0.0
        has_cab = peak >= self.DOOR_MIN_PEAK
        started_open = has_cab and float(door[0]) >= self.OPEN_FRAC * peak
        ended_closed = has_cab and float(door[-1]) <= self.CLOSE_FRAC * peak
        n_opened = int(sum(float(doorv[:, i].max()) >= self.OPEN_DOOR_TH
                           for i in range(doorv.shape[1]))) if doorv.shape[1] else 0

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- OpenCabinet (only if the door started closed and opens before the first pick) ---
        if has_cab and not started_open:
            lvl = self.OPEN_FRAC * peak
            fo = self._sustained_start(door[:max(t_first_pick, 1)] >= lvl, 0, self.OPEN_SUSTAIN)
            if fo is None:
                w = np.where(door >= lvl)[0]
                fo = int(w[0]) if w.size else 0
            t_open = int(min(fo, max(t_first_pick - 1, 0)))
            if t_open > 0:
                emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the cabinet {self.open_position}")

        # --- fruit: cabinet -> bowl ---
        n_moved = 0
        for o in order:
            nn = self.noun[o]
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
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the cabinet")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} in the bowl")
            n_moved += 1

        # --- CloseCabinet (only if the door ends closed) ---
        if has_cab and ended_closed:
            close_onset = self._sustained_start(door[cursor:] < self.OPEN_FRAC * peak, 0, self.OPEN_SUSTAIN)
            close_onset = (cursor + close_onset) if close_onset is not None else cursor
            t_closed = self._sustained_start(door[close_onset:] <= self.CLOSE_FRAC * peak, 0, self.OPEN_SUSTAIN)
            t_closed = (close_onset + t_closed) if t_closed is not None else T
            if t_closed > cursor:
                emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the cabinet {self.open_position}")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 fruit moved")
        if not has_cab:
            review.append("cabinet door never opened -- verify")
        result = self.build_result(T, segments, debug={
            "order": order, "door_peak": round(peak, 2), "started_open": bool(started_open),
            "ended_closed": bool(ended_closed), "n_opened": n_opened,
            "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
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

        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenCabinet, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 2:
            issues.append(f"expected 2 picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 2:
            issues.append(f"expected 2 places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseCabinet, found {count(SUBTASK_CLOSE)}")
        return issues
