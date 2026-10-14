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

# Three items are carried from the counter into the (already-open) fridge: alcohol
# onto the top shelf (out of a child's reach), fruit/veg onto a lower shelf. Counter
# -> fridge has no bank name -> PickPlaceCounterToFridge (extension); NavigateKitchen
# is bank. The robot starts at the counter and returns to it between items.
SUBTASK_MOVE = "PickPlaceCounterToFridge"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

ITEMS = ("item1", "item2", "item3")


@register_labeler
class CreateChildFriendlyFridgeLabeler(BaseSubtaskLabeler):
    """Labeler for the CreateChildFriendlyFridge composite task.

    Task (mobile): three items sit on the counter -- item1 (fruit/veg), item2
    (alcohol), item3 (alcohol OR fruit/veg, from ep_meta refs). Load them into the
    (already-open) fridge: alcohol on the TOP shelf (out of reach), fruit/veg on a
    LOWER shelf. The robot starts at the counter and drives back between items.

    Doc-canonical decomposition (11 high-level subtasks): per item `[nav back to the
    counter] -> pick -> nav(carry to the fridge) -> place`; the first item has no
    leading nav. 3 items -> 11 subtasks + done. ORDER varies (by first move).

    Per item: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; navs split off the base motion. The
    destination shelf is by type: alcohol -> "the top shelf", fruit/veg -> "a lower
    shelf".
    """

    ENV_NAME = "CreateChildFriendlyFridge"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
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
        third = ep_meta.get("refs", {}).get("third_item_type", "fruit_vegetable")
        # item1 = fruit/veg (lower), item2 = alcohol (top), item3 = alcohol|fruit_veg
        self.shelf = {
            "item1": "the lower shelf",
            "item2": "the top shelf",
            "item3": "the top shelf" if third == "alcohol" else "the lower shelf",
        }
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_MOVE}
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
        order = sorted(ITEMS, key=lambda o: first_move[o])

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

        n_moved = 0
        for o in order:
            if first_move[o] >= T:
                continue
            word, dst = self.obj_lang[o], self.shelf[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the counter")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the counter")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the fridge")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} on {dst} of the fridge")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
