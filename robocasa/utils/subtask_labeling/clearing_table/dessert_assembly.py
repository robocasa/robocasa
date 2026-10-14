import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each item (a bowl holding a dessert, and a cupcake) is carried from the counter
# onto the tray. Source and destination are both the counter (the tray sits on it),
# and the bank has no PickPlaceCounterToCounter, so the name is a declared extension.
# It splits pick / place.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"

CONTAINER = "dessert1_container"  # the bowl holding dessert1
CUPCAKE = "dessert2"
OBJS = (CONTAINER, CUPCAKE)


@register_labeler
class DessertAssemblyLabeler(BaseSubtaskLabeler):
    """Labeler for the DessertAssembly composite task.

    Task: a bowl holding a dessert (donut/cake) and a cupcake sit on the counter;
    place both onto the tray (`receptacle`). The tray is the stationary target; only
    the bowl and the cupcake move. Everything is within arm reach on one counter, so
    the base never drives.

    Doc-canonical count = 2 (`mobile: Yes`, but no actual nav): one
    PickPlaceCounterToCounter per item. Both are always relocated; the ORDER varies.
    2 PickPlace subtasks + done = 5 segments; expected_done_idx = 4.

    The two are distinct bodies, tracked by their own gripper distance (no sticky
    attribution -- the bowl is grabbed at the rim, so an absolute grasp gate would
    miss it; the relative-release split needs no such gate). Per object, first-move /
    relative-release split: t_move = first sustained displacement from rest; t_placed
    = the gripper RELEASE (grip->obj rising above the object's own carry hold level).
    """

    ENV_NAME = "DessertAssembly"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in OBJS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            cont = env.get_obj_lang(CONTAINER)
        except Exception:
            cont = "bowl"
        try:
            dessert = env.get_obj_lang("dessert1")
        except Exception:
            dessert = "dessert"
        try:
            cup = env.get_obj_lang(CUPCAKE)
        except Exception:
            cup = "cupcake"
        self.noun = {
            CONTAINER: f"the {cont} with the {dessert}",
            CUPCAKE: f"the {cup}",
        }

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = 4
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in OBJS:
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
        opos = {o: np.array([h[o] for h in history]) for o in OBJS}
        grip = np.array([h["grip"] for h in history])

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in OBJS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in OBJS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in OBJS}

        first_move = {}
        for o in OBJS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(OBJS, key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for o in order:
            # re-anchor rest to this object's settled pose at the start of its turn
            # (the other item's handling can jostle it)
            r = np.median(opos[o][cursor: min(cursor + self.REST_W, T)], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)

            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {self.noun[o]} from the counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {self.noun[o]} on the tray")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1
        return self.build_result(T, segments, debug={"order": order, "n_segments": len(segments)})

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(stage):
            return int(np.sum((names == SUBTASK_MOVE) & (stages == stage)))

        if count(STAGE_PICK) != 2:
            issues.append(f"expected 2 dessert picks, found {count(STAGE_PICK)}")
        if count(STAGE_PLACE) != 2:
            issues.append(f"expected 2 dessert places, found {count(STAGE_PLACE)}")
        return issues
