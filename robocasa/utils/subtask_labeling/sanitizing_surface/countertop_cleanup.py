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


# Clear the countertop: a utensil goes to the (open) drawer, a receptacle to the
# (open) cabinet, and a bowl of food to the (open) fridge. Counter -> drawer/cabinet
# are bank atomics; counter -> fridge has no bank name -> PickPlaceCounterToFridge
# (extension). NavigateKitchen is a bank atomic. The two foods ride inside the bowl.
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# object -> (subtask name, destination phrase)
DEST = {
    "utensil": ("PickPlaceCounterToDrawer", "the drawer"),
    "receptacle": ("PickPlaceCounterToCabinet", "the cabinet"),
    "food_bowl": ("PickPlaceCounterToFridge", "the fridge"),
}


@register_labeler
class CountertopCleanupLabeler(BaseSubtaskLabeler):
    """Labeler for the CountertopCleanup composite task.

    Task (mobile): a utensil, a receptacle, and a bowl (holding two foods) sit on the
    counter; the drawer, cabinet, and fridge all start open. Put the utensil in the
    drawer, the receptacle in the cabinet, and the bowl in the fridge. The robot starts
    at the counter.

    Doc-canonical count = 5 (`mobile: Yes`), action-level: three moves to three
    destinations, each nav-interrupted. Per object, in first-move order:
      [nav back to the counter] -> pick -> nav(carry to dest) -> place
    the first object has no leading nav. The two foods ride inside the bowl.

    Per object: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; the return/carry navs are split off
    the base motion (MIN_NAV_DIST).
    """

    ENV_NAME = "CountertopCleanup"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15
    CAB_ABOVE_Z = 1.80    # cabinet z at/above this is a high wall cabinet ("above")
    DRAWER_SIDE_TH = 0.35  # drawer |y| in base frame beyond this == to a side (else in front)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in DEST if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            if o == "food_bowl":
                # "the bowl of food" -- the receptacle can also be a bowl, so name
                # the food-holding one distinctly to avoid two "the bowl"s.
                self.noun[o] = "the bowl of food"
            else:
                try:
                    self.noun[o] = f"the {env.get_obj_lang(o)}"
                except Exception:
                    self.noun[o] = "the item"
        # Per-episode destination phrases; the cabinet is "above" (high wall cabinet)
        # or "in front" (low full-height) depending on its height.
        try:
            cab_z = float(getattr(env, "cab").pos[2])
        except Exception:
            cab_z = self.CAB_ABOVE_Z
        cab_phrase = "the cabinet above" if cab_z >= self.CAB_ABOVE_Z else "the cabinet in front"
        self.dest_phrase = {o: DEST[o][1] for o in self.names}
        if "receptacle" in self.dest_phrase:
            self.dest_phrase["receptacle"] = cab_phrase
        # drawer direction from geometry: drawer.pos in the robot base frame (+y=left,
        # -y=right); env.drawer_side is re-randomized on replay so it is not trusted.
        drawer_dir = "the drawer"
        try:
            bid = env.sim.model.body_name2id("mobilebase0_base")
            bp = env.sim.data.body_xpos[bid].copy()
            bm = env.sim.data.body_xmat[bid].reshape(3, 3)
            y = float((bm.T @ (np.array(env.drawer.pos) - bp))[1])
            if y >= self.DRAWER_SIDE_TH:
                drawer_dir = "the drawer to the left"
            elif y <= -self.DRAWER_SIDE_TH:
                drawer_dir = "the drawer to the right"
            else:
                drawer_dir = "the drawer in front"
        except Exception:
            pass
        if "utensil" in self.dest_phrase:
            self.dest_phrase["utensil"] = drawer_dir
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {"PickPlaceCounterToFridge"}  # the other two are bank atomics
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.names:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

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
            sub = DEST[o][0]
            dest = self.dest_phrase[o]
            noun = self.noun[o]
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

            emit(t_pick, sub, STAGE_PICK, f"pick up {noun} from the counter")

            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to {dest}")
            emit(t_placed, sub, STAGE_PLACE, f"place {noun} in {dest}")
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
        # setup() is skipped for manual overrides, so re-declare the extension name
        # before the bank check, then short-circuit the structural checks for them.
        self.extra_subtasks = {"PickPlaceCounterToFridge"}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
