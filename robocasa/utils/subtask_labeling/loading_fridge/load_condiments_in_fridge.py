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

# Two condiments are carried from the counter onto the fridge's top shelf; any
# distractor already on the top shelf is first relocated to a lower shelf to make
# room. Counter -> fridge has no bank name -> PickPlaceCounterToFridge; the in-fridge
# distractor shuffle is PickPlaceFridgeShelfToShelf (also an extension);
# NavigateKitchen is bank. The robot starts at the fridge.
SUBTASK_COND = "PickPlaceCounterToFridge"
SUBTASK_DISTR = "PickPlaceFridgeShelfToShelf"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

CONDIMENTS = ("condiment1", "condiment2")
DISTRACTORS = ("distractor1", "distractor2")
ITEMS = CONDIMENTS + DISTRACTORS


@register_labeler
class LoadCondimentsInFridgeLabeler(BaseSubtaskLabeler):
    """Labeler for the LoadCondimentsInFridge composite task.

    Task (mobile): two condiments sit on the counter; two distractor items are already
    inside the (open) fridge. Put both condiments on the top shelf; if a distractor is
    on the top shelf, move it to a lower shelf first. The robot starts at the fridge.

    VARIABLE count: the two condiments always move (counter -> top shelf, each
    nav-interrupted); 0-2 distractors are relocated within the fridge (top -> lower
    shelf, a local shuffle with no nav). Doc-canonical 9 counts the full case. ORDER
    is taken from the data -- distractors are usually cleared first.

    Per object: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; navs split off the base motion (a
    distractor shuffle drives no base motion, so it emits no nav).
    """

    ENV_NAME = "LoadCondimentsInFridge"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.07
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in ITEMS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in ITEMS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = o
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_COND, SUBTASK_DISTR}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in ITEMS:
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
        opos = {o: np.array([h[o] for h in history]) for o in ITEMS}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in ITEMS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in ITEMS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in ITEMS}

        first_move = {}
        for o in ITEMS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in ITEMS if first_move[o] < T], key=lambda o: first_move[o])

        same_cond = self.obj_lang[CONDIMENTS[0]] == self.obj_lang[CONDIMENTS[1]]

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_cond = 0
        n_distr = 0
        for o in order:
            is_cond = o in CONDIMENTS
            word = self.obj_lang[o]
            if is_cond:
                subtask = SUBTASK_COND
                noun = ("the other " + word if n_cond == 1 else "a " + word) if same_cond else "the " + word
            else:
                subtask = SUBTASK_DISTR
                noun = "the " + word

            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav to the counter (only condiments require it; the base drive is
            # what marks it, so a distractor shuffle -- no base motion -- emits nothing).
            to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
            t_go = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
            if t_go > cursor and base_disp(cursor, t_go) >= self.MIN_NAV_DIST:
                emit(t_go, SUBTASK_NAV, STAGE_NAVIGATE, "navigate to the counter with condiments")

            src = "the counter" if is_cond else "the top shelf"
            emit(t_pick, subtask, STAGE_PICK, f"pick up {noun} from {src}")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if is_cond and t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the fridge")

            if is_cond:
                emit(t_placed, subtask, STAGE_PLACE, f"place {noun} on the top shelf of the fridge")
                n_cond += 1
            else:
                emit(t_placed, subtask, STAGE_PLACE, f"move {noun} to a lower shelf of the fridge")
                n_distr += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_cond != 2:
            review.append(f"only {n_cond}/2 condiments moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_cond": n_cond, "n_distr": n_distr, "n_segments": len(segments),
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
        n_cond_pick = int(np.sum((names == SUBTASK_COND) & (stages == STAGE_PICK)))
        n_cond_place = int(np.sum((names == SUBTASK_COND) & (stages == STAGE_PLACE)))
        if n_cond_pick != 2:
            issues.append(f"expected 2 condiment picks, found {n_cond_pick}")
        if n_cond_place != 2:
            issues.append(f"expected 2 condiment places, found {n_cond_place}")
        return issues
