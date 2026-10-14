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


# Each of the two fruits is carried from the (already-open) fridge to the cutting
# board by the sink. The fridge->cutting-board move has no exact bank name (the
# bank has no PickPlaceFridgeToCounter), so it is an extra_subtask; NavigateKitchen
# is a bank atomic. The two non-fruit items stay in the fridge and are ignored.
SUBTASK_MOVE = "PickPlaceFridgeToCounter"   # fruit: fridge -> cutting board
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

OBJS = ("fruit1", "fruit2")


@register_labeler
class ArrangeCuttingFruitsLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeCuttingFruits composite task.

    Task: two fruits and two non-fruit items sit in an ALREADY-OPEN fridge (the
    robot starts at the fridge). Carry each fruit across the kitchen to the cutting
    board by the sink; leave the non-fruit items in the fridge.

    Doc-canonical count = 7 (`mobile: Yes`): each fruit is a nav-interrupted
    fridge->board pick-place, and between the two the robot drives BACK to the
    fridge for the second fruit. Full decomposition:
        fruit_a[pick] -> nav(carry to board) -> fruit_a[place]
        -> nav(back to fridge) -> fruit_b[pick] -> nav(carry to board) -> fruit_b[place]
    = 7 subtasks + done (expected_done_idx = 7). The robot starts AT the fridge, so
    the FIRST fruit has no leading nav; the SECOND fruit is preceded by the return
    drive. Each nav is ADAPTIVE (gated on MIN_NAV_DIST) -- a demo that reaches the
    board without really driving drops that nav. The ORDER of the two fruits varies
    and is resolved from which moves first.

    Boundaries per fruit (SweetenCoffee's milk-fetch convention, base-stop splits):
      - a leading return-to-fridge nav: the base drive from the board back to the
        fridge BEFORE this fruit is grasped (only the second fruit has one).
      - t_pick = the fruit first displaces MOVE_TOL from its rest (in-fridge) pose.
      - the carry nav = the base drive while the fruit is held, ending at the base
        stop; the place is the set-down + gripper RELEASE after the base parks.
    """

    ENV_NAME = "ArrangeCuttingFruits"

    MOVE_TOL = 0.10  # fruit displaced this far from rest == grasped & being moved
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_pick used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.08  # grip->obj rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20  # a carry lasts at least this many frames before a release counts
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    MIN_NAV_DIST = 0.20  # peak base displacement over a leg to emit a nav
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
                self.obj_lang[o] = "fruit"

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
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }
        for o in OBJS:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in OBJS}
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in OBJS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in OBJS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in OBJS}

        first_move = {}
        for o in OBJS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(OBJS, key=lambda o: first_move[o])
        same_type = self.obj_lang[OBJS[0]] == self.obj_lang[OBJS[1]]

        def fruit_noun(k, o):
            word = self.obj_lang[o]
            if same_type:
                return f"the other {word}" if k == 1 else f"a {word}"
            return f"the {word}"

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

        for k, o in enumerate(order):
            noun = fruit_noun(k, o)
            # t_pick: this fruit first displaces from the fridge (after cursor)
            t_pick = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav: the base drive from the board BACK to the fridge, before
            # this fruit is grasped (only the second fruit has one -- the robot
            # starts at the fridge). Split at the base stop.
            to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
            t_at_fridge = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
            if t_at_fridge > cursor and base_disp(cursor, t_at_fridge) >= self.MIN_NAV_DIST:
                nav_lang = "navigate back to the fridge" if k >= 1 else "navigate to the fridge"
                emit(t_at_fridge, SUBTASK_NAV, STAGE_NAVIGATE, nav_lang)

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the fridge")

            # place end = gripper release after the carry
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            # carry nav: base drive while the fruit is held, split at the base stop
            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE,
                     f"carry {noun} to the cutting board next to the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
                 f"place {noun} on the cutting board next to the sink")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": order,
            "first_move": {o: int(first_move[o]) for o in OBJS},
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

        if count(SUBTASK_MOVE, STAGE_PICK) != 2:
            issues.append(f"expected 2 fruit picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 2:
            issues.append(f"expected 2 fruit places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 4):
            issues.append(f"expected 0-4 NavigateKitchen, found {n_nav}")
        return issues
