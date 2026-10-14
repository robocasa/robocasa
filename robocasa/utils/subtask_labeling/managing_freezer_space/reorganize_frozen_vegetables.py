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

# Two frozen vegetables come out of the freezer into the tupperware on the counter, then
# the tupperware (with the veg) goes back into the freezer. Freezer -> counter and counter
# -> freezer have no bank names -> PickPlaceFreezerToCounter / PickPlaceCounterToFreezer
# (extensions); NavigateKitchen is bank. The robot starts at the fridge and drives between
# the freezer and the counter.
SUBTASK_VEG = "PickPlaceFreezerToCounter"
SUBTASK_TUP = "PickPlaceCounterToFreezer"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

VEG = ("vegetable1", "vegetable2")
TUP = "tupperware"
ITEMS = VEG + (TUP,)


@register_labeler
class ReorganizeFrozenVegetablesLabeler(BaseSubtaskLabeler):
    """Labeler for the ReorganizeFrozenVegetables composite task.

    Task (mobile): take two vegetables out of the (open) freezer and put them in the
    tupperware on the counter, then put the tupperware back in the freezer. The robot
    starts at the fridge and drives between the freezer and the counter.

    Doc-canonical decomposition (3 high-level: veg1 + veg2 moves + tupperware move).
    Emitted per object in grasp order (the two veg first, tupperware last):
      veg:  [nav back to freezer] -> pick (freezer) -> nav(carry to counter) -> place (tupperware)
      tup:  [nav back to counter] -> pick (counter) -> nav(carry to freezer) -> place (freezer)
    plus a terminal done. The first object (a veg) has no leading nav; the tupperware is
    usually grasped where the robot just placed the veg (no leading nav).

    Per object: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; navs split off the base motion.
    """

    ENV_NAME = "ReorganizeFrozenVegetables"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.40
    REST_W = 15
    MIN_PICK_GAP = 12   # the gripper lingers in the tupperware it just filled before it
                        # grabs it, so the last veg's place can run into the tupperware
                        # pick -- shorten the prior place to keep the pick its own frames

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in ITEMS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in ITEMS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "vegetable" if o in VEG else "tupperware"
        self.same_veg = self.obj_lang[VEG[0]] == self.obj_lang[VEG[1]]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_VEG, SUBTASK_TUP}
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

    def _veg_noun(self, o, n_veg_done):
        word = self.obj_lang[o]
        if self.same_veg:
            return "the other " + word if n_veg_done == 1 else "a " + word
        return "the " + word

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
        n_veg = 0
        for o in order:
            is_tup = o == TUP
            subtask = SUBTASK_TUP if is_tup else SUBTASK_VEG
            src = "the counter" if is_tup else "the freezer"
            noun = "the tupperware" if is_tup else self._veg_noun(o, n_veg)

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
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate back to {src} for {noun}")

            # if this pick would collapse into the previous PLACE (the gripper lingered on
            # the just-placed object before grabbing this one), pull that place back so the
            # pick keeps its own frames.
            if segments and segments[-1][2] == STAGE_PLACE and t_pick - cursor < self.MIN_PICK_GAP:
                prev_start = segments[-2][0] if len(segments) >= 2 else 0
                new_end = max(prev_start + 1, t_pick - self.MIN_PICK_GAP)
                if new_end < cursor:
                    pe = segments[-1]
                    segments[-1] = (new_end, pe[1], pe[2], pe[3])
                    cursor = new_end

            emit(t_pick, subtask, STAGE_PICK, f"pick up {noun} from {src}")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                if is_tup:
                    emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the tupperware to the freezer")
                else:
                    emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the counter with the tupperware")
            if is_tup:
                emit(t_placed, subtask, STAGE_PLACE, "place the tupperware in the freezer")
            else:
                emit(t_placed, subtask, STAGE_PLACE, f"place {noun} in the tupperware")
                n_veg += 1
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 objects moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_veg": n_veg, "n_segments": len(segments),
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
