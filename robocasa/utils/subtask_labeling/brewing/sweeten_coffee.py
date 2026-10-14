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


# Sweeten the coffee: drop a sugar cube from the saucer into the coffee mug (a
# local pick-place), then bring milk from the (already-open) fridge and set it on
# the counter next to the coffee (a nav-interrupted pick-place). Neither move has an
# exact bank name, so both are extra_subtasks; NavigateKitchen is a bank atomic.
SUBTASK_SUGAR = "PickPlaceSaucerToCoffee"    # sugar cube: saucer -> coffee mug
SUBTASK_MILK = "PickPlaceFridgeToCounter"    # milk: fridge -> counter
SUBTASK_NAV = "NavigateKitchen"

TO_FRIDGE_LANG = "navigate to the fridge"
DONE_LANG = "task complete"


@register_labeler
class SweetenCoffeeLabeler(BaseSubtaskLabeler):
    """Labeler for the SweetenCoffee composite task.

    Task: a mug of coffee and a saucer with a sugar cube sit on the counter (where
    the robot starts); milk sits in an ALREADY-OPEN fridge across the kitchen. Drop
    the sugar cube into the coffee, then fetch the milk and set it on the counter
    next to the coffee.

    Doc-canonical count = 5 (`mobile: Yes`): the sugar's local pick-place
    (PickPlaceSaucerToCoffee[pick]->[place], no nav -- the saucer and mug are both on
    the counter) + the milk's pick-place, which is INTERRUPTED by a navigation back
    from the fridge (PickPlaceFridgeToCounter[pick] -> NavigateKitchen -> [place]).
    2 + 3 = 5 segments (+ done = 6; expected_done_idx = 5). The sugar always leads;
    order is resolved from which object moves first.

    Boundaries:
      - sugar (first-move split, no nav): pick = reach + grasp; place = lift + drop
                 into the mug + release. t_move = sugar displaces MOVE_TOL from the
                 saucer; t_placed = the gripper releases it (grip->sugar rises).
      - milk (base-stop split): the milk pick includes the drive OUT to the fridge
                 (the reach) + grasp; t_move = milk displaces from the fridge. The nav
                 is the carry back -- it ends when the base stops driving (t_arrive =
                 last base-moving frame). The place is the set-down + release after
                 the base parks (same convention as DeliverBrewedCoffee). A nav is
                 emitted iff the base's PEAK displacement over the carry >= MIN_NAV_DIST.
    """

    ENV_NAME = "SweetenCoffee"

    MOVE_TOL = 0.10  # object displaced this far from rest == grasped & being moved
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_move used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.08  # grip->obj rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20  # a carry lasts at least this many frames before a release counts
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    MIN_NAV_DIST = 0.20  # peak base displacement over the milk carry to emit a nav
    REST_W = 20

    SUGAR = "sugar_cube"
    MILK = "milk"

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sugar_body_id = env.obj_body_id[self.SUGAR]
        self.milk_body_id = env.obj_body_id[self.MILK]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.sugar_lang = env.get_obj_lang(self.SUGAR)
        except Exception:
            self.sugar_lang = "sugar cube"
        try:
            self.milk_lang = env.get_obj_lang(self.MILK)
        except Exception:
            self.milk_lang = "milk"

        self.base_qpos_adrs = []
        for nm in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
        ):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_SUGAR, SUBTASK_MILK}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "sugar": env.sim.data.body_xpos[self.sugar_body_id].copy(),
            "milk": env.sim.data.body_xpos[self.milk_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    # ---- helpers -------------------------------------------------------

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

    def _move_release(self, opos, grip, disp, cursor, T):
        """(t_move, t_placed) for one object: t_move = first sustained displacement
        after cursor; t_placed = the gripper release after the carry."""
        t_move = self._sustained_start(disp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp[cursor:] >= self.MOVE_TOL)[0]
            t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        t_move = max(t_move, cursor)
        gd = np.linalg.norm(grip - opos, axis=1)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1
        return t_move, t_placed

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        sugar = np.array([h["sugar"] for h in history])
        milk = np.array([h["milk"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        pos = {self.SUGAR: sugar, self.MILK: milk}
        disp = {o: np.linalg.norm(pos[o] - np.median(pos[o][: self.REST_W], axis=0), axis=1)
                for o in pos}
        first_move = {}
        for o in pos:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(pos, key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for o in order:
            t_move, t_placed = self._move_release(pos[o], grip, disp[o], cursor, T)
            if o == self.SUGAR:
                emit(t_move, SUBTASK_SUGAR, STAGE_PICK,
                     f"pick up the {self.sugar_lang} from the saucer")
                emit(t_placed, SUBTASK_SUGAR, STAGE_PLACE,
                     f"place the {self.sugar_lang} in the coffee")
            else:  # milk: [nav to fridge] / pick / [nav back] / place
                # nav OUT to the fridge: the base drive from the counter (after the
                # sugar) to the fridge, before the milk is grasped. Emitted unless
                # it is a very short hop (peak base disp < MIN_NAV_DIST).
                to_moving = np.where(binst[cursor: t_move] > self.BASE_MOVE_TH)[0]
                t_fridge = min(cursor + int(to_moving[-1]) + 1, t_move) if to_moving.size else cursor
                to_disp = (
                    float(np.max(np.linalg.norm(base_xy[cursor: t_fridge] - base_xy[cursor], axis=1)))
                    if base_xy is not None and t_fridge > cursor else 0.0
                )
                if t_fridge > cursor and to_disp >= self.MIN_NAV_DIST:
                    emit(t_fridge, SUBTASK_NAV, STAGE_NAVIGATE, TO_FRIDGE_LANG)
                emit(t_move, SUBTASK_MILK, STAGE_PICK,
                     f"pick up the {self.milk_lang} from the fridge")
                # nav BACK to the counter: the base drive while carrying the milk.
                back_moving = np.where(binst[t_move: t_placed] > self.BASE_MOVE_TH)[0]
                t_arrive = min(t_move + int(back_moving[-1]) + 1, t_placed) if back_moving.size else t_move
                back_disp = (
                    float(np.max(np.linalg.norm(base_xy[t_move: t_arrive] - base_xy[t_move], axis=1)))
                    if base_xy is not None and t_arrive > t_move else 0.0
                )
                if t_arrive > cursor and back_disp >= self.MIN_NAV_DIST:
                    emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE,
                         f"carry the {self.milk_lang} to the counter by the coffee maker")
                emit(t_placed, SUBTASK_MILK, STAGE_PLACE,
                     f"place the {self.milk_lang} next to the coffee")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": order,
            "first_move": {o: int(first_move[o]) for o in pos},
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
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

        for nm, label in ((SUBTASK_SUGAR, "sugar"), (SUBTASK_MILK, "milk")):
            if count(nm, STAGE_PICK) != 1:
                issues.append(f"expected 1 {label} pick, found {count(nm, STAGE_PICK)}")
            if count(nm, STAGE_PLACE) != 1:
                issues.append(f"expected 1 {label} place, found {count(nm, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1, 2):
            issues.append(f"expected 0-2 NavigateKitchen, found {n_nav}")
        return issues
