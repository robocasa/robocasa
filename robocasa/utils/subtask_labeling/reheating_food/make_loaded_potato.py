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


# Assemble a loaded potato: open the microwave, take the reheated potato out and put it on the
# cutting board, then add the condiment and the cheese from the counter beside it. OpenMicrowave
# and PickPlaceMicrowaveToCounter are bank atomics; counter -> board (counter -> counter) has no
# bank name -> PickPlaceCounterToCounter (extension); NavigateKitchen is a bank atomic.
SUBTASK_OPEN = "OpenMicrowave"
SUBTASK_MW = "PickPlaceMicrowaveToCounter"
SUBTASK_COUNTER = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the microwave"
DONE_LANG = "task complete"


@register_labeler
class MakeLoadedPotatoLabeler(BaseSubtaskLabeler):
    """Labeler for the MakeLoadedPotato composite task.

    Task (mobile, usually local): a reheated potato sits in a CLOSED microwave; a cutting board, a
    condiment, and a cheese sit on the counter. Open the microwave, move the potato onto the board,
    and add the condiment and cheese. The robot starts at the microwave.

    Doc-canonical count = 6 (three items, pick + place each), plus OpenMicrowave at the front:
      OpenMicrowave [execute]                              -> open the microwave
      PickPlaceMicrowaveToCounter/CounterToCounter [pick]  -> pick up the {item} from the {microwave|counter}
      (NavigateKitchen [navigate])                         -> carry the {item} to the cutting board (adaptive)
      PickPlaceMicrowaveToCounter/CounterToCounter [place] -> place the {item} on the cutting board
    plus a terminal done. `condiment` is named by get_obj_lang.

    Boundaries: OpenMicrowave ends when the door reaches its adaptive open level
    min(OPEN_ABS, 0.9*peak), capped just before the potato departs; t_pick = first sustained
    displacement from a rest re-anchored at the segment start; a carry nav is emitted only if the
    base actually travels; t_place = the gripper RELEASE.
    """

    ENV_NAME = "MakeLoadedPotato"

    # object -> (pick_from phrase, subtask)
    ITEMS = {
        "potato": ("the microwave", SUBTASK_MW),
        "condiment": ("the counter", SUBTASK_COUNTER),
        "cheese": ("the counter", SUBTASK_COUNTER),
    }

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    OPEN_ABS = 0.60
    OPEN_MIN = 0.15
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in self.ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # Microwave door (the potato starts inside a closed microwave).
        mw = env.microwave
        self.door_adr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
        jid = env.sim.model.joint_name2id(mw.door_joint_names[0])
        lo, hi = env.sim.model.jnt_range[jid]
        self.door_scale = max(abs(float(hi)), abs(float(lo)), 1e-6)
        self.word = {}
        for o in self.names:
            if o == "condiment":
                try:
                    self.word[o] = env.get_obj_lang(o)
                except Exception:
                    self.word[o] = "condiment"
            else:
                self.word[o] = o
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_COUNTER}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "door": abs(float(env.sim.data.qpos[self.door_adr])) / self.door_scale,
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
        door = np.array([h["door"] for h in history])
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

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # OpenMicrowave first: the door swings open (from closed) before the
        # potato can be taken out. End it just before the potato departs.
        review = []
        potato_fm = first_move.get("potato", T)
        peak = float(door.max())
        has_open = peak >= self.OPEN_MIN
        if has_open:
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            t_open = self._sustained_start(door >= lvl, 0, self.OPEN_SUSTAIN)
            if t_open is None:
                t_open = int(np.argmax(door))
            t_open = int(min(t_open, max(potato_fm - 1, 0)))
            if t_open > 0:
                emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        else:
            review.append("microwave door never opened -- verify")

        n_moved = 0
        for o in order:
            pick_from, sub = self.ITEMS[o]
            word = self.word[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_pick, sub, STAGE_PICK, f"pick up the {word} from {pick_from}")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                        if base_xy is not None and t_arrive > t_pick else 0.0)
            if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the cutting board")
            emit(t_placed, sub, STAGE_PLACE, f"place the {word} on the cutting board")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "has_open": has_open,
            "door_peak": round(peak, 2), "n_segments": len(segments),
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
        names = result.subtask_name[firsts]
        n_open = int(np.sum(names == SUBTASK_OPEN))
        if n_open not in (0, 1):
            issues.append(f"expected 0-1 OpenMicrowave, found {n_open}")
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
