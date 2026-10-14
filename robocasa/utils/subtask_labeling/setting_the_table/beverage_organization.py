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


# A VARIABLE number of drinks (2-4) sit on one counter (where the robot spawns); carry each
# to the dining counter. NavigateKitchen is a bank atomic; counter -> dining table has no
# bank name -> PickPlaceCounterToDining (extension). The robot drives back to the source
# counter between drinks.
SUBTASK_MOVE = "PickPlaceCounterToDining"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class BeverageOrganizationLabeler(BaseSubtaskLabeler):
    """Labeler for the BeverageOrganization composite task.

    Task (mobile): a VARIABLE number of drinks (2-4) sit on one counter (where the robot
    starts). Move all of them to the dining counter.

    Doc-canonical count = 7 (`mobile: Yes`, action-level for the typical 2-drink run):
    per drink `[nav back] -> pick -> nav(carry) -> place`; the first has no leading nav.
    The number of drinks VARIES, so validate() checks coverage (equal picks/places, one
    per drink) rather than a fixed total. ORDER by first move.

    Boundaries: per drink t_pick = first sustained displacement re-anchored at the segment
    start; navs split off the base motion (MIN_NAV_DIST); t_place = the gripper RELEASE.
    """

    ENV_NAME = "BeverageOrganization"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    MIN_PICK_GAP = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.num_bev = int(ep_meta.get("refs", {}).get("num_bev", 0))
        self.names = [f"obj_{i}" for i in range(max(self.num_bev, 0)) if f"obj_{i}" in env.obj_body_id]
        if not self.names:
            self.names = [n for n in (f"obj_{i}" for i in range(8)) if n in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = "the drink"
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
            nn = self.noun[o]
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
                t_back = min(t_back, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate back to the counter with {nn}")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the counter")

            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {nn} to the dining counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the dining counter")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        n_exp = self.num_bev if self.num_bev else len(self.names)
        if n_moved != n_exp:
            review.append(f"only {n_moved}/{n_exp} drinks moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "num_bev": self.num_bev,
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        npick = int(np.sum(stages == STAGE_PICK))
        nplace = int(np.sum(stages == STAGE_PLACE))
        if npick != nplace:
            issues.append(f"pick/place mismatch: {npick} picks, {nplace} places")
        if npick not in (2, 3, 4):
            issues.append(f"expected 2-4 drink moves, found {npick}")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 8):
            issues.append(f"expected 0-8 NavigateKitchen, found {n_nav}")
        return issues
