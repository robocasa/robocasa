import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A yogurt and a VARIABLE number (1-3) of fruit sit in an ALREADY-OPEN cabinet (the setup
# opens it, and the language never mentions opening/closing); move each onto the counter.
# Cabinet -> counter has no bank name -> PickPlaceCabinetToCounter (extension). Non-mobile.
SUBTASK_MOVE = "PickPlaceCabinetToCounter"

DONE_LANG = "task complete"


@register_labeler
class YogurtDelightPrepLabeler(BaseSubtaskLabeler):
    """Labeler for the YogurtDelightPrep composite task.

    Task (non-mobile): a yogurt and `num_fruit` (1-3) fruit sit in a pre-opened cabinet
    (the robot spawns at the cabinet). Place all of them onto the counter.

    Doc-canonical count = 2 (`mobile: No`) but the fruit count is VARIABLE, so the real
    coverage is (1 yogurt + num_fruit) cabinet->counter moves. There is NO OpenCabinet /
    CloseCabinet subtask -- the cabinet is already open and the language never mentions it.
    ORDER by first move; each move splits pick / place.

    Per item: t_pick = first sustained displacement re-anchored at the segment start;
    t_place = the gripper RELEASE.
    """

    ENV_NAME = "YogurtDelightPrep"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.num_fruit = int(ep_meta.get("refs", {}).get("num_fruit", 0))
        cand = ["yogurt"] + [f"fruit_{i}" for i in range(max(self.num_fruit, 0))]
        # fall back to whatever fruit bodies actually exist if refs is missing/short
        if self.num_fruit <= 0:
            cand = ["yogurt"] + [f"fruit_{i}" for i in range(3)]
        self.names = [o for o in cand if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.n_expected = len(self.names)
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = "the yogurt" if o == "yogurt" else "the fruit"
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
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

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

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
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the counter")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != self.n_expected:
            review.append(f"only {n_moved}/{self.n_expected} items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_expected": self.n_expected,
            "num_fruit": self.num_fruit, "n_segments": len(segments),
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
        picks = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK)))
        places = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE)))
        n = self.n_expected
        if picks != n:
            issues.append(f"expected {n} picks, found {picks}")
        if places != n:
            issues.append(f"expected {n} places, found {places}")
        return issues
