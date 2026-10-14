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


# Gather the two sanitizing tools next to the cutting board: the spray bottle from the
# (already-open) cabinet and the dish brush from the counter beside the sink. The
# cabinet source is a bank atomic (PickPlaceCabinetToCounter); the counter->counter
# brush move has no bank name -> PickPlaceCounterToCounter (extension). NavigateKitchen
# is a bank atomic; the robot drives from the cutting board over to the sink counter
# for the brush.
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# object -> (word, pick_from phrase, nav-to-source phrase, subtask name)
ITEMS = {
    "spray": ("the spray bottle", "the cabinet", "the cabinet", "PickPlaceCabinetToCounter"),
    "dish_brush": ("the dish brush", "the counter next to the sink",
                   "the counter next to the sink", "PickPlaceCounterToCounter"),
}


@register_labeler
class SanitizePrepCuttingBoardLabeler(BaseSubtaskLabeler):
    """Labeler for the SanitizePrepCuttingBoard composite task.

    Task (mobile): a cutting board sits on the counter (where the robot starts); a
    spray bottle is in the already-open cabinet and a dish brush is on the counter by
    the sink. Bring both next to the cutting board for sanitizing.

    Doc-canonical count = 5 (`mobile: Yes`): a multi-source gather -- per tool,
    `[nav to the source] -> pick -> [carry nav to the board] -> place`. The first tool
    (nearest the start) has no leading nav. ORDER is resolved from when each tool moves.

    Per tool: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE next to the board; the navs are split
    off the base motion (MIN_NAV_DIST). The leading approach nav is only emitted for
    the 2nd+ tool (k>=1), never for the first.
    """

    ENV_NAME = "SanitizePrepCuttingBoard"

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
        self.names = [o for o in ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {"PickPlaceCounterToCounter"}  # cabinet move is a bank atomic
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
            word, pick_from, nav_to, sub = ITEMS[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading approach nav to the source (2nd+ tool only)
            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_to = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_to > cursor and base_disp(cursor, t_to) >= self.MIN_NAV_DIST:
                    emit(t_to, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate to {nav_to}")

            emit(t_pick, sub, STAGE_PICK, f"pick up {word} from {pick_from}")

            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {word} to the cutting board")
            emit(t_placed, sub, STAGE_PLACE, f"place {word} next to the cutting board")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        # Some demos gather only one tool (the other is never touched) -- a
        # legitimate short demo, not an error. Only 0 moved is worth a flag.
        if n_moved == 0:
            review.append("no tools moved -- verify")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {"PickPlaceCounterToCounter"}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        # 1 or 2 tools may be gathered; picks and places must pair up.
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick not in (1, 2):
            issues.append(f"expected 1-2 picks, found {n_pick}")
        if n_place != n_pick:
            issues.append(f"pick/place mismatch: {n_pick} picks, {n_place} places")
        return issues
