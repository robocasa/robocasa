import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# Two bottled waters are carried from the counter into the (open) freezer, then the
# freezer door is closed. Counter -> freezer has no bank name -> PickPlaceCounterToFreezer
# (extension); NavigateKitchen and CloseFridge are bank. The robot starts at the counter
# and drives back to it between the two bottles.
SUBTASK_MOVE = "PickPlaceCounterToFreezer"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseFridge"

DONE_LANG = "task complete"

BOTTLES = ("bottle1", "bottle2")


@register_labeler
class FreezeBottledWatersLabeler(BaseSubtaskLabeler):
    """Labeler for the FreezeBottledWaters composite task.

    Task (mobile): two bottled waters sit on the counter; put both in the (open) freezer
    to freeze, then close the freezer door. The robot starts at the counter and drives
    back to it between the two bottles.

    Doc-canonical decomposition (8 high-level): per bottle `[nav back to the counter] ->
    pick -> nav(carry to the freezer) -> place`; the first bottle has no leading nav (2
    bottles -> 7 subtasks) + CloseFridge -> 8 subtasks + done. ORDER is taken from the
    data.

    Per bottle: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE; navs split off the base motion. The
    close ends when the freezer door swings back below CLOSE_TH.
    """

    ENV_NAME = "FreezeBottledWaters"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15
    CLOSE_TH = 0.20
    CLOSE_SUSTAIN = 5
    MIN_PICK_GAP = 12   # the base sometimes creeps until the grasp; cap the return nav
                        # so the reach/grasp keeps its own frames (else the pick collapses)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in BOTTLES}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in BOTTLES:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "bottle of water"
        self.same = self.obj_lang[BOTTLES[0]] == self.obj_lang[BOTTLES[1]]
        fr = env.fridge
        joint_names = list(getattr(fr, "_freezer_door_joint_names", []) or [])
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
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
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": doors,
        }
        for o in BOTTLES:
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
        opos = {o: np.array([h[o] for h in history]) for o in BOTTLES}
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in BOTTLES}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in BOTTLES}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in BOTTLES}

        first_move = {}
        for o in BOTTLES:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in BOTTLES if first_move[o] < T], key=lambda o: first_move[o])

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
        last_place = 0
        for o in order:
            word = self.obj_lang[o]
            noun = ("the other " + word if n_moved == 1 else "a " + word) if self.same else "the " + word
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
                # cap so the reach/grasp keeps its own frames (base can creep to the grab)
                t_back = min(t_back, max(cursor + 1, t_pick - self.MIN_PICK_GAP))
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate back to the counter for {noun}")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the counter")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the freezer")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} in the freezer")
            last_place = t_placed
            n_moved += 1

        close_frames = []
        for i in range(doorv.shape[1]):
            if float(doorv[:, i].max()) < 0.5:
                continue
            fr = self._sustained_start(doorv[:, i] < self.CLOSE_TH, last_place, self.CLOSE_SUSTAIN)
            if fr is not None:
                close_frames.append(fr)
        n_doors = int(sum(1 for i in range(doorv.shape[1]) if float(doorv[:, i].max()) >= 0.5))
        review = []
        if close_frames:
            t_close = max(max(close_frames), cursor + 1)
            door_word = "doors" if n_doors >= 2 else "door"
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the freezer {door_word}")
        elif doorv.shape[1] and float(doorv.max()) >= 0.5:
            review.append("freezer door never closed -- verify")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 bottles moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_close": len(close_frames),
            "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
