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

# Meat: counter -> the pan on the stove == PickPlaceCounterToStove (bank).
# Condiment and vegetable: (already-open) cabinet -> the counter by the plate ==
# PickPlaceCabinetToCounter (bank). NavigateKitchen is bank; the moves may be
# nav-interrupted (adaptive). The robot starts at the stove.
SUBTASK_MEAT = "PickPlaceCounterToStove"
SUBTASK_CAB = "PickPlaceCabinetToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# name -> (subtask, source phrase, dest phrase). The meat sits in a plate on the
# counter (try_to_place_in="container"), so its source names the plate.
SPECS = {
    "meat": (SUBTASK_MEAT, "the plate on the counter", "the pan"),
    "condiment": (SUBTASK_CAB, "the open cabinet", "the counter"),
    "vegetable": (SUBTASK_CAB, "the open cabinet", "the counter"),
}
ITEMS = ("meat", "condiment", "vegetable")


@register_labeler
class AssembleCookingArrayLabeler(BaseSubtaskLabeler):
    """Labeler for the AssembleCookingArray composite task.

    Task (mobile): move the meat from the counter onto the pan on the stove, then
    move the condiment and vegetable from the ALREADY-OPEN cabinet to the counter
    by the plate. The robot starts at the stove.

    Doc-canonical decomposition (4 high-level subtasks). Emitted fine-grained as,
    per object, `[nav to source] -> pick -> [carry nav] -> place`: the meat move
    (counter -> pan, no nav -- both are at the stove) then the two cabinet items
    (cabinet -> counter, each possibly nav-interrupted). Object ORDER is the meat
    first (always), then the condiment/vegetable in whichever order they move.

    Per object: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE (finger-distance opens past its hold
    level); navs are split off the base motion (adaptive, MIN_NAV_DIST).
    """

    ENV_NAME = "AssembleCookingArray"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.06
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    BASE_MOVE_TH = 0.004
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
        self.extra_subtasks = set()
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
        # meat always first (it is moved before the cabinet items); then by first move.
        order = sorted(ITEMS, key=lambda o: (o != "meat", first_move[o]))

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
            subtask, src, dst = SPECS[o]
            word = self.obj_lang[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav to the source (base drive before the grasp), except the
            # first object (robot starts at the stove where the meat is).
            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_go = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_go > cursor and base_disp(cursor, t_go) >= self.MIN_NAV_DIST:
                    emit(t_go, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate to {src}")

            emit(t_pick, subtask, STAGE_PICK, f"pick up the {word} from {src}")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to {dst}")
            emit(t_placed, subtask, STAGE_PLACE, f"place the {word} on {dst}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
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
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 3:
            issues.append(f"expected 3 pick stages, found {n_pick}")
        if n_place != 3:
            issues.append(f"expected 3 place stages, found {n_place}")
        return issues
