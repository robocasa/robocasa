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


# A tupperware and a roll of aluminum foil sit in an (already-open) cabinet (robot starts
# there); a plate of leftovers sits on the dining counter. Carry both storage items to the
# dining counter near the plate. Cabinet -> dining counter has no bank name ->
# PickPlaceCabinetToDining (extension); NavigateKitchen is bank. Cabinet pre-opened.
SUBTASK_MOVE = "PickPlaceCabinetToDining"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

OBJS = ("tupperware", "aluminum_foil")


@register_labeler
class PrepareStoringLeftoversLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareStoringLeftovers composite task.

    Task (mobile): a tupperware and aluminum foil sit in a pre-opened cabinet (robot starts
    there); a plate of leftover food is on the dining counter. Bring both items to the
    dining counter next to the plate.

    Doc-canonical count = 7 (`mobile: Yes`), action-level: two cabinet->dining moves, each
    pick + carry nav + place, plus one return-to-cabinet nav before the second item. ORDER
    by first grasped move; cabinet pre-opened (no OpenCabinet).

    Boundaries: pick = first displacement while grasped; navs off base motion (return ends
    at arrival); place = the gripper RELEASE.
    """

    ENV_NAME = "PrepareStoringLeftovers"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    GRASP_TH = 0.07
    BASE_REST_TH = 0.003
    ARRIVE_SUSTAIN = 8

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = "the tupperware" if o == "tupperware" else "the aluminum foil"
        try:
            f1, f2 = env.get_obj_lang("food1"), env.get_obj_lang("food2")
            self.food_text = f"{f1}s" if f1 == f2 else f"{f1} and {f2}"
        except Exception:
            self.food_text = "leftovers"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
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

    @staticmethod
    def _last_sustained_end(mask, sustain):
        """End index (exclusive) of the LAST run of >= sustain consecutive True values, or
        None -- where the final sustained base-drive stops (skips a brief mid-drive pause and
        ignores post-arrival twitches)."""
        last_end, run = None, 0
        for i, v in enumerate(mask):
            if v:
                run += 1
            else:
                if run >= sustain:
                    last_end = i
                run = 0
        if run >= sustain:
            last_end = len(mask)
        return last_end

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
            gdo = np.linalg.norm(grip - opos[o], axis=1)
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gdo < self.GRASP_TH), 0, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        n_moved = 0
        for k, o in enumerate(order):
            nn = self.noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            t_pick = self._sustained_start((dsp >= self.MOVE_TOL) & (gd < self.GRASP_TH), cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            if k >= 1:
                # the return drive is often stop-and-go; end the nav at the final sustained
                # base-drive, not the first brief pause. ##demo_15 (ended ~5s early)
                end = self._last_sustained_end(binst[cursor: t_pick] > self.BASE_MOVE_TH, self.MOVE_SUSTAIN)
                t_appr_end = cursor + end if end is not None else cursor
                t_appr_end = min(t_appr_end, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_appr_end > cursor and base_disp(cursor, t_appr_end) >= self.MIN_NAV_DIST:
                    emit(t_appr_end, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the cabinet")
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the cabinet")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {nn} to the plate on the dining counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
                 f"place {nn} on the dining counter near the plate with {self.food_text}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        picks = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK)))
        places = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE)))
        if picks != 2:
            issues.append(f"expected 2 picks, found {picks}")
        if places != 2:
            issues.append(f"expected 2 places, found {places}")
        return issues
