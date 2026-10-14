import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Two toasted bread slices sit in the pop-up toaster; a bowl is on the counter beside it; a
# peanut-butter bottle and a jelly jar sit on the far dining counter. Drop both slices into
# the bowl, carry the bowl to the dining counter, and set it next to the pb+jelly. The
# toaster->bowl moves have no exact bank name (toaster slot -> counter bowl) -> use the bank
# "PickPlaceToasterToCounter"; the bowl carry is counter -> dining (extension
# PickPlaceCounterToDining); NavigateKitchen is bank. Mobile; doc count = 5.
SUBTASK_BREAD = "PickPlaceToasterToCounter"
SUBTASK_BOWL = "PickPlaceCounterToDining"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

BREADS = ("bread_slice_1", "bread_slice_2")


@register_labeler
class PJSandwichPrepLabeler(BaseSubtaskLabeler):
    """Labeler for the PJSandwichPrep composite task.

    Task (mobile): two toasted bread slices are in the pop-up toaster, a bowl sits on the
    counter beside it, and a peanut-butter bottle + jelly jar are on the far dining counter.
    Put both slices in the bowl, carry the bowl to the dining counter, and place it next to
    the peanut butter and jelly.

    Doc-canonical count = 5 (`mobile: Yes`), action-level: two toaster->bowl bread moves
    (pick/place each) + the bowl move (pick + carry nav + place). Breads in first-grasped-move
    order; same-type slices read "a ..."/"the other ...".

    Boundaries: pick = first displacement while grasped; the bowl carry nav comes off base
    motion (ends at the last substantial base-drive); place = the gripper RELEASE.
    """

    ENV_NAME = "PJSandwichPrep"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.06
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    GRASP_TH = 0.08
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 10

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.breads = [o for o in BREADS if o in env.obj_body_id]
        self.bowl_id = env.obj_body_id["bowl"]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.breads}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_BREAD, SUBTASK_BOWL}
        self.expected_done_idx = None
        self.min_segment_frames = 8

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "bowl": env.sim.data.body_xpos[self.bowl_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.breads:
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

    @staticmethod
    def _last_sustained_end(mask, sustain):
        last_end, run = None, 0
        for i, v in enumerate(mask):
            if v:
                run += 1
            else:
                if run >= sustain:
                    last_end = i
                run = 0
        if run >= sustain:
            last_end = len(mask)
        return last_end

    def resolve(self, history):
        T = len(history)
        bpos = {o: np.array([h[o] for h in history]) for o in self.breads}
        bowl = np.array([h["bowl"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        gd = {o: np.linalg.norm(grip - bpos[o], axis=1) for o in self.breads}
        # order the two slices by first grasped move
        first_move = {}
        for o in self.breads:
            rest = np.median(bpos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(bpos[o] - rest, axis=1)
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), 0, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.breads if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def pick_place(pos, cursor_ref, grasp):
            r = np.median(pos[max(cursor_ref - self.REST_W, 0): cursor_ref + 1], axis=0)
            dsp = np.linalg.norm(pos - r, axis=1)
            tp = self._sustained_start((dsp >= self.MOVE_TOL) & (grasp < self.GRASP_TH), cursor_ref, self.MOVE_SUSTAIN)
            if tp is None:
                tp = self._sustained_start(dsp >= self.MOVE_TOL, cursor_ref, self.MOVE_SUSTAIN)
            if tp is None:
                mv = np.where(dsp[cursor_ref:] >= self.MOVE_TOL)[0]
                tp = cursor_ref + int(mv[0]) if mv.size else min(cursor_ref + 1, T - 1)
            tp = max(tp, cursor_ref)
            hold = float(np.median(grasp[tp: min(tp + self.HOLD_W, T)]))
            tpl = self._sustained_start(grasp > hold + self.RELEASE_MARGIN, tp + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if tpl is None:
                tpl = T - 1
            return tp, tpl

        # two bread slices -> bowl
        n_bread = len(order)
        for i, o in enumerate(order):
            nn = ("a toasted bread slice" if i == 0 else "the other toasted bread slice")
            pl = ("the bread slice" if i == 0 else "the other bread slice")
            t_pick, t_place = pick_place(bpos[o], cursor, gd[o])
            t_pick = int(min(max(t_pick, cursor + self.MIN_PICK_GAP), max(t_place - self.MIN_PICK_GAP, cursor + 1)))
            emit(t_pick, SUBTASK_BREAD, STAGE_PICK, f"pick up {nn} from the toaster")
            emit(t_place, SUBTASK_BREAD, STAGE_PLACE, f"place {pl} in the bowl")

        # bowl -> dining counter (pick, carry nav, place)
        gdb = np.linalg.norm(grip - bowl, axis=1)
        t_bpick, t_bplace = pick_place(bowl, cursor, gdb)
        t_bpick = int(min(max(t_bpick, cursor + self.MIN_PICK_GAP), max(t_bplace - self.MIN_PICK_GAP, cursor + 1)))
        emit(t_bpick, SUBTASK_BOWL, STAGE_PICK, "pick up the bowl")
        end = self._last_sustained_end(binst[t_bpick: t_bplace] > self.BASE_MOVE_TH, self.MOVE_SUSTAIN)
        t_arrive = t_bpick + end if end is not None else t_bpick
        t_arrive = min(t_arrive, max(t_bplace - self.MIN_PICK_GAP, cursor))
        if t_arrive > cursor and base_disp(t_bpick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE,
                 "carry the bowl to the peanut butter and jelly on the dining counter")
        emit(t_bplace, SUBTASK_BOWL, STAGE_PLACE,
             "place the bowl next to the peanut butter and jelly")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_bread != 2:
            review.append(f"only {n_bread}/2 bread slices moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_bread": n_bread, "t_bpick": t_bpick, "t_bplace": t_bplace,
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_BREAD, SUBTASK_BOWL}
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

        if count(SUBTASK_BREAD, STAGE_PICK) != 2:
            issues.append(f"expected 2 bread picks, found {count(SUBTASK_BREAD, STAGE_PICK)}")
        if count(SUBTASK_BREAD, STAGE_PLACE) != 2:
            issues.append(f"expected 2 bread places, found {count(SUBTASK_BREAD, STAGE_PLACE)}")
        if count(SUBTASK_BOWL, STAGE_PICK) != 1:
            issues.append("expected 1 bowl pick")
        if count(SUBTASK_BOWL, STAGE_PLACE) != 1:
            issues.append("expected 1 bowl place")
        return issues
