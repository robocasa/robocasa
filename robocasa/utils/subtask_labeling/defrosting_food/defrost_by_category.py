import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Sort four frozen foods: the two fruits go into the sink, the two vegetables go into
# a bowl on the counter. Fruits->sink is PickPlaceCounterToSink (bank); vegetables->
# bowl is counter->counter (the bowl sits on the counter), for which the bank has no
# name -> PickPlaceCounterToCounter is a declared extension. No navigation (mobile: No).
SUBTASK_SINK = "PickPlaceCounterToSink"      # fruit -> sink
SUBTASK_BOWL = "PickPlaceCounterToCounter"   # vegetable -> bowl

DONE_LANG = "task complete"

FRUITS = ("obj0", "obj1")
VEGGIES = ("obj2", "obj3")
OBJS = FRUITS + VEGGIES


@register_labeler
class DefrostByCategoryLabeler(BaseSubtaskLabeler):
    """Labeler for the DefrostByCategory composite task.

    Task: two frozen fruits (obj0, obj1) and two frozen vegetables (obj2, obj3) sit
    on the counter, plus a bowl. Put both fruits in the sink and both vegetables in
    the bowl. The base never drives (mobile: No).

    Doc-canonical count = 4 (`mobile: No`): one pick-place per food (2 fruits ->
    sink, 2 vegetables -> bowl). All four are relocated; the ORDER varies (resolved
    from which moves first). 4 pick-places (pick/place each) + done = 9 segments;
    expected_done_idx = 8.

    The four are distinct bodies, tracked by their own gripper distance (no sticky
    attribution). Per object, first-move / relative-release split: t_move = first
    sustained displacement from rest; t_placed = the gripper RELEASE.
    """

    ENV_NAME = "DefrostByCategory"

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
        self.obj_lang = {}
        for o in OBJS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "fruit" if o in FRUITS else "vegetable"

        self.extra_subtasks = {SUBTASK_BOWL}
        self.expected_done_idx = 8
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

        # a/other naming within each category (fruits, veggies) when types repeat
        def noun_for(o, moved_in_cat):
            word = self.obj_lang[o]
            same = [x for x in moved_in_cat if self.obj_lang[x] == word]
            if len(same) >= 2:
                return f"the other {word}" if o == same[1] else f"a {word}"
            return f"the {word}"

        moved = [o for o in order if first_move[o] < T]
        moved_fruits = [o for o in order if o in FRUITS and first_move[o] < T]
        moved_veg = [o for o in order if o in VEGGIES and first_move[o] < T]

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for o in order:
            if first_move[o] >= T:
                continue
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

            if o in FRUITS:
                noun = noun_for(o, moved_fruits)
                emit(t_move, SUBTASK_SINK, STAGE_PICK, f"pick up {noun} from the counter")
                emit(t_placed, SUBTASK_SINK, STAGE_PLACE, f"place {noun} in the sink")
            else:
                noun = noun_for(o, moved_veg)
                emit(t_move, SUBTASK_BOWL, STAGE_PICK, f"pick up {noun} from the counter")
                emit(t_placed, SUBTASK_BOWL, STAGE_PLACE, f"place {noun} in the bowl")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if len(moved) != 4:
            review.append(f"only {len(moved)}/4 foods moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": len(moved), "n_segments": len(segments),
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

        def count(name, stage):
            return int(np.sum((names == name) & (stages == stage)))

        if count(SUBTASK_SINK, STAGE_PICK) != 2:
            issues.append(f"expected 2 fruit picks, found {count(SUBTASK_SINK, STAGE_PICK)}")
        if count(SUBTASK_SINK, STAGE_PLACE) != 2:
            issues.append(f"expected 2 fruit places, found {count(SUBTASK_SINK, STAGE_PLACE)}")
        if count(SUBTASK_BOWL, STAGE_PICK) != 2:
            issues.append(f"expected 2 vegetable picks, found {count(SUBTASK_BOWL, STAGE_PICK)}")
        if count(SUBTASK_BOWL, STAGE_PLACE) != 2:
            issues.append(f"expected 2 vegetable places, found {count(SUBTASK_BOWL, STAGE_PLACE)}")
        return issues
