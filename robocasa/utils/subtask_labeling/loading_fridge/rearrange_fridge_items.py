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

# Two items are shuffled from a source shelf to a destination shelf inside the fridge
# (in-fridge, no nav -> PickPlaceFridgeShelfToShelf extension), then a pitcher is
# carried from the counter onto the now-cleared source shelf (counter -> fridge, nav
# -> PickPlaceCounterToFridge extension). NavigateKitchen is bank. Robot starts at the
# fridge.
SUBTASK_SHUFFLE = "PickPlaceFridgeShelfToShelf"
SUBTASK_PITCHER = "PickPlaceCounterToFridge"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

ITEMS = ("item1_source", "item2_source")


@register_labeler
class RearrangeFridgeItemsLabeler(BaseSubtaskLabeler):
    """Labeler for the RearrangeFridgeItems composite task.

    Task (mobile): two items sit on one fridge shelf; a pitcher sits on the counter.
    Move the two items to the other shelf to make space, then place the pitcher on the
    cleared shelf. The robot starts at the fridge.

    Doc-canonical decomposition (3 high-level subtasks: 2 item shuffles + pitcher).
    The two items are shuffled within the fridge (no nav); the pitcher is a
    nav-interrupted counter -> fridge move. ORDER is taken from the data (items first,
    then the pitcher).

    Per object: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; navs split off the base motion (an
    in-fridge shuffle drives no base motion, so it emits no nav). Shelf names from refs.
    """

    ENV_NAME = "RearrangeFridgeItems"

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
        self.all_objs = ITEMS + ("pitcher",)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.all_objs}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in self.all_objs:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "item" if o != "pitcher" else "pitcher"
        refs = ep_meta.get("refs", {})
        self.src_shelf = "top shelf" if refs.get("source_rack_index", -1) == -1 else "second highest shelf"
        self.dst_shelf = "top shelf" if refs.get("dest_rack_index", -1) == -1 else "second highest shelf"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_SHUFFLE, SUBTASK_PITCHER}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.all_objs:
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
        objs = self.all_objs
        opos = {o: np.array([h[o] for h in history]) for o in objs}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in objs}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in objs}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in objs}

        first_move = {}
        for o in objs:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in objs if first_move[o] < T], key=lambda o: first_move[o])

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

        n_item = 0
        for o in order:
            is_pitcher = o == "pitcher"
            word = self.obj_lang[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav to the counter (only the pitcher needs it; a shuffle drives no base)
            to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
            t_go = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
            if is_pitcher and t_go > cursor and base_disp(cursor, t_go) >= self.MIN_NAV_DIST:
                emit(t_go, SUBTASK_NAV, STAGE_NAVIGATE, "navigate to the counter for the pitcher")

            if is_pitcher:
                emit(t_pick, SUBTASK_PITCHER, STAGE_PICK, "pick up the pitcher from the counter")
            else:
                emit(t_pick, SUBTASK_SHUFFLE, STAGE_PICK, f"pick up the {word} from the {self.src_shelf}")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            if is_pitcher:
                carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
                t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
                if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                    emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the pitcher to the fridge")
                emit(t_placed, SUBTASK_PITCHER, STAGE_PLACE,
                     f"place the pitcher on the cleared {self.src_shelf}")
            else:
                emit(t_placed, SUBTASK_SHUFFLE, STAGE_PLACE,
                     f"move the {word} to the {self.dst_shelf}")
                n_item += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_item != 2:
            review.append(f"only {n_item}/2 items shuffled")
        result = self.build_result(T, segments, debug={
            "order": order, "n_item": n_item, "n_segments": len(segments),
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
        if int(np.sum((names == SUBTASK_SHUFFLE) & (stages == STAGE_PICK))) != 2:
            issues.append("expected 2 shelf-shuffle picks")
        if int(np.sum((names == SUBTASK_PITCHER) & (stages == STAGE_PICK))) != 1:
            issues.append("expected 1 pitcher pick")
        return issues
