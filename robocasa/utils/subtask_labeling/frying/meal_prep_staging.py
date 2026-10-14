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

# Both pans (counter -> a stove burner) and both foods (counter -> a pan) are
# PickPlaceCounterToStove (bank). NavigateKitchen is bank; moves may be
# nav-interrupted (adaptive). The robot starts at the stove.
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

PANS = ("pan1", "pan2")
FOODS = ("vegetable", "meat")
ITEMS = PANS + FOODS


@register_labeler
class MealPrepStagingLabeler(BaseSubtaskLabeler):
    """Labeler for the MealPrepStaging composite task.

    Task (mobile): two pans and a vegetable and a meat sit on the counter. Place
    both pans onto different burners, then place the vegetable and the meat on
    different pans. The robot starts at the stove.

    Doc-canonical decomposition (4 high-level subtasks: 2 pan placements + 2 food
    placements). Emitted fine-grained as, per object, `[nav] -> pick -> [carry nav]
    -> place`. The two pans go first in the task description, but demos often
    INTERLEAVE (place a pan, put a food on it, place the other pan, ...), so the
    object ORDER is taken from the data (by first move), not assumed.

    Per object: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; navs split off the base motion.
    Pans are identical -> "a pan" / "the other pan"; the two burners / pans a food
    lands on are "a burner"/"the other burner" and "a pan"/"the other pan" by order.
    """

    ENV_NAME = "MealPrepStaging"

    MOVE_TOL = 0.08        # food departure threshold (kept high: a small tol catches an
                           # adjacent food being jostled when its neighbour is grabbed)
    PAN_MOVE_TOL = 0.06    # pans use a smaller tol so their pick/place boundary fires as
                           # the (heavy) pan first lifts, not a touch later
    MOVE_SUSTAIN = 5
    PLACE_TOL = 0.12       # object distance from final pose counting as "still moving"
    ARRIVE_TOL = 0.15      # object within this of its final pose == arrived
    REL_FRAC = 0.55        # finger-width must reopen this far into the grasp->open range
    REL_SUSTAIN = 6        # ...and stay open this many frames (skip a mid-settle spike)
    BASE_MOVE_TH = 0.004
    # This is a LOCAL task -- the counter is adjacent to the stove, so the base only
    # repositions a short distance (~0.35-0.65 m) between reaches; those are not real
    # navigation and are not labeled. The threshold sits well above that band so only a
    # genuine cross-kitchen walk (if one ever occurred) would emit a NavigateKitchen.
    MIN_NAV_DIST = 1.0
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in ITEMS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.gripper = env.robots[0].gripper["right"]
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints
        ]
        self.obj_lang = {}
        for o in FOODS:
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
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr)),
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

    def _release(self, obj, fw, lo, T):
        """Place ends at the gripper RELEASE: the first frame, after the object has
        arrived near its final pose, where the finger-width reopens most of the way to
        its fully-open width AND stays open. Position-independent (not fooled by the
        gripper lingering by the pan it set down, which a gripper-to-object distance
        would be) and later than the object's first arrival (it keeps settling under the
        gripper before it lets go). The high fraction + sustained window skip a brief
        mid-settle grip adjustment that momentarily nudges the fingers apart."""
        final = np.median(obj[-self.REST_W:], axis=0)
        d2final = np.linalg.norm(obj - final, axis=1)
        near = np.where(d2final[:T] < self.ARRIVE_TOL)[0]
        near = near[near > lo]
        t_arrive = int(near[0]) if near.size else lo
        trans = fw[lo:max(t_arrive, lo + 1)]
        grasp_w = float(np.median(trans)) if trans.size else float(fw[lo])
        fw_open = float(np.max(fw))
        level = grasp_w + self.REL_FRAC * (fw_open - grasp_w)
        opened = fw > level
        for t in range(max(t_arrive, lo + 1), max(T - self.REL_SUSTAIN, lo + 2)):
            if bool(np.all(opened[t: t + self.REL_SUSTAIN])):
                return t
        far = np.where(d2final >= self.PLACE_TOL)[0]
        far = far[far >= lo]
        return int(far[-1]) + 1 if far.size else lo + 1

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in ITEMS}
        grip = np.array([h["grip"] for h in history])
        fw = np.array([h["fw"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in ITEMS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in ITEMS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in ITEMS}

        def tol_for(o):
            return self.PAN_MOVE_TOL if o in PANS else self.MOVE_TOL

        first_move = {}
        for o in ITEMS:
            fm = self._sustained_start(disp[o] >= tol_for(o), 0, self.MOVE_SUSTAIN)
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

        n_pan = 0
        n_food = 0
        n_moved = 0
        for o in order:
            if first_move[o] >= T:
                continue
            is_pan = o in PANS
            if is_pan:
                # pick "a pan" / place "the pan" (the one in hand); second pan is "the
                # other pan". Burner is not dictated by the task -> generic.
                if n_pan == 0:
                    pick_noun, place_noun, dst = "a pan", "the pan", "a burner"
                else:
                    pick_noun = place_noun = "the other pan"
                    dst = "another burner"
                nav_lang = "navigate to the empty pan on the counter"
            else:
                word = self.obj_lang[o]
                pick_noun = place_noun = "the " + word
                # a food always goes on a pan that is currently empty. "an empty pan"
                # only when two empty pans are on burners (a choice of two); otherwise
                # "the empty pan" (the single remaining one).
                dst = "an empty pan" if (n_pan - n_food) >= 2 else "the empty pan"
                nav_lang = "navigate to the counter"

            tol = tol_for(o)
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= tol, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= tol)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_go = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_go > cursor and base_disp(cursor, t_go) >= self.MIN_NAV_DIST:
                    emit(t_go, SUBTASK_NAV, STAGE_NAVIGATE, nav_lang)

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {pick_noun} from the counter")

            # place ends at the gripper RELEASE (finger-width opens once the object has
            # arrived) -- the full set-down, not the moment the object first reaches its
            # spot. Robust to the gripper lingering by the pan it just placed (to set a
            # food on it), which a gripper-to-object distance would misread as still held.
            t_placed = self._release(opos[o], fw, t_pick, T)

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {place_noun} to the stove")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {place_noun} on {dst}")
            if is_pan:
                n_pan += 1
            else:
                n_food += 1
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 4:
            review.append(f"only {n_moved}/4 items moved")
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
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 4:
            issues.append(f"expected 4 pick stages, found {n_pick}")
        if n_place != 4:
            issues.append(f"expected 4 place stages, found {n_place}")
        return issues
