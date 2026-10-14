import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Each kebab skewer is moved from the plate to the oven tray (both sit on the dining
# counter). Source and destination are both the counter, so the bank has no name ->
# PickPlaceCounterToCounter is a declared extension. No navigation (mobile: No).
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"

OBJS = ("skewer1", "skewer2")


@register_labeler
class MeatSkewerAssemblyLabeler(BaseSubtaskLabeler):
    """Labeler for the MeatSkewerAssembly composite task.

    Task: two kebab skewers sit on a plate; an oven tray sits on the same dining
    counter. Move both skewers from the plate onto the oven tray. The base never
    drives (mobile: No).

    Doc-canonical count = 2 (`mobile: No`): one PickPlaceCounterToCounter per skewer.
    Both are always relocated; the ORDER varies. 2 pick-places + done = 5 segments;
    expected_done_idx = 4.

    The two skewers are distinct bodies, tracked by their own gripper distance (NO
    sticky attribution -- a skewer is long and grabbed off-centre, so an absolute
    grasp gate would miss it; the relative-release split needs no such gate). Per
    skewer, first-move / relative-release: t_move = first sustained displacement from
    rest (re-anchored at the start of its turn); t_placed = the gripper RELEASE.
    """

    ENV_NAME = "MeatSkewerAssembly"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20
    SPEED_TH = 0.003  # per-frame object speed above this == it is being moved
    # both skewers MOVING at once this many consecutive frames == grabbed & carried
    # TOGETHER. A genuine two-in-one-hand carry co-runs for 50-70+ frames; a settle of
    # one skewer overlapping the next pick only grazes ~8, so the bar is set well above
    # that noise floor to avoid false merges.
    MIN_CO_RUN = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in OBJS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in OBJS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "kebab skewer"

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
        # per-frame object speed (the two skewers sit close on the plate, so gripper
        # proximity can't tell "both grabbed" from "one grabbed, other adjacent" --
        # use MOTION instead: they were carried together iff they MOVE at the same time)
        moving = {}
        for o in OBJS:
            sp = np.linalg.norm(np.diff(opos[o], axis=0), axis=1)
            moving[o] = np.concatenate([sp, sp[-1:]]) > self.SPEED_TH

        first_move = {}
        for o in OBJS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T

        # co-move grouping: two skewers carried TOGETHER (both MOVING at once for
        # MIN_CO_RUN frames) are one physical action -> one combined move.
        groups = self._comove_groups(OBJS, moving, first_move, T)
        groups.sort(key=lambda g: min(first_move[o] for o in g))

        same_type = self.obj_lang[OBJS[0]] == self.obj_lang[OBJS[1]]

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_sep = 0  # count of single-object (separate) moves emitted, for a/other naming
        for g in groups:
            # group t_move: earliest member to start moving (rest re-anchored at cursor)
            tmoves, releases = [], []
            for o in g:
                r = np.median(opos[o][cursor: min(cursor + self.REST_W, T)], axis=0)
                dsp = np.linalg.norm(opos[o] - r, axis=1)
                tm = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
                if tm is None:
                    mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                    tm = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
                tmoves.append(max(tm, cursor))
            t_move = min(tmoves)
            for o in g:
                hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
                rel = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                            t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
                releases.append(T - 1 if rel is None else rel)
            t_placed = max(releases)

            word = self.obj_lang[g[0]]
            if len(g) >= 2:
                noun = f"the {word}s"  # both skewers moved together
            elif same_type:
                noun = "the other " + word if n_sep == 1 else "a " + word
                n_sep += 1
            else:
                noun = "the " + word
                n_sep += 1
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the plate")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} on the oven tray")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1
        review = []
        if any(len(g) >= 2 for g in groups):
            review.append("skewers moved together -> combined into one move")
        result = self.build_result(T, segments, debug={
            "n_groups": len(groups), "combined": any(len(g) >= 2 for g in groups),
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def _comove_groups(self, objs, moving, first_move, T):
        """Union objects that are carried together (both MOVING for >= MIN_CO_RUN
        consecutive frames) -- one physical action. Objects that never move (first_move
        == T) are dropped."""
        moved = [o for o in objs if first_move[o] < T]
        parent = {o: o for o in moved}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(moved):
            for b in moved[i + 1:]:
                both = moving[a] & moving[b]
                run = best = 0
                for v in both:
                    run = run + 1 if v else 0
                    best = max(best, run)
                if best >= self.MIN_CO_RUN:
                    parent[find(a)] = find(b)
        groups = {}
        for o in moved:
            groups.setdefault(find(o), []).append(o)
        return list(groups.values())

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

        # 2 separate skewer moves, OR 1 combined move (both grabbed together); picks
        # must match places.
        n_pick, n_place = count(STAGE_PICK), count(STAGE_PLACE)
        if n_pick != n_place:
            issues.append(f"skewer picks ({n_pick}) != places ({n_place})")
        if n_pick not in (1, 2):
            issues.append(f"expected 1 (combined) or 2 (separate) skewer moves, found {n_pick}")
        return issues
