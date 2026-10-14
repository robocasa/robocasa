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


# A bowl on the counter holds a microwavable meat and a non-microwavable fruit. Remove the
# fruit onto the small plate, then carry the bowl (now meat-only) to the ALREADY-OPEN
# microwave, close the door, and press start. Fruit bowl->plate has no bank name ->
# PickPlaceCounterToCounter (extension); bowl counter->microwave is the bank atomic
# PickPlaceCounterToMicrowave; NavigateKitchen + CloseMicrowave + TurnOnMicrowave are bank.
SUBTASK_FRUIT = "PickPlaceCounterToCounter"
SUBTASK_BOWL = "PickPlaceCounterToMicrowave"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseMicrowave"
SUBTASK_ON = "TurnOnMicrowave"

DONE_LANG = "task complete"


@register_labeler
class FilterMicrowavableItemLabeler(BaseSubtaskLabeler):
    """Labeler for the FilterMicrowavableItem composite task.

    Task (mobile, but everything sits by the microwave so nav is nil): a bowl holds a meat
    (microwavable) and a fruit (not). Move the fruit onto the small plate, then place the
    bowl in the ALREADY-OPEN microwave, close the door, and press start.

    Doc-canonical count = 6 (`mobile: Yes`): fruit move (pick/place) + bowl move
    (pick/nav/place) + CloseMicrowave + TurnOnMicrowave. Structure:
      fruit[pick] -> fruit[place] -> bowl[pick] -> (NavigateKitchen) -> bowl[place] ->
      CloseMicrowave -> TurnOnMicrowave -> done.
    No OpenMicrowave (it starts open). The order is fixed (fruit out before the bowl goes
    in) but resolved from first-motion to be safe. The fruit move is local (bowl->plate on
    the counter); the bowl is CARRIED to the microwave, so its leg gets an adaptive nav.

    Boundaries: per object, t_pick = first sustained displacement from a rest re-anchored at
    the segment start; t_place = the gripper RELEASE. The bowl's nav ends at the base stop
    during the carry (emitted only when the base actually drives MIN_NAV_DIST). CloseMicrowave
    = the door swings shut after the bowl place; TurnOnMicrowave = the close to the end (reach
    + press start; the microwave's cached turned_on is unusable on a state replay).
    """

    ENV_NAME = "FilterMicrowavableItem"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    CLOSE_FRAC = 0.10    # the door must be essentially LATCHED (< 10% of its open swing)
                         # before the turn-on -- ending the close at 30% left it visibly ajar.
    DOOR_SUSTAIN = 5
    OPEN_MIN = 0.05
    BUTTON_TH = 0.09
    REST_W = 15
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.body = {o: env.obj_body_id[o] for o in ("bowl", "fruit")}
        try:
            self.fruit_lang = env.get_obj_lang("fruit")
        except Exception:
            self.fruit_lang = "fruit"
        try:
            self.meat_lang = env.get_obj_lang("meat")
        except Exception:
            self.meat_lang = "meat"

        mw = env.microwave
        self.door_adr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
        try:
            self.btn_gid = env.sim.model.geom_name2id(f"{mw.name}_start_button")
        except Exception:
            self.btn_gid = None

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_FRUIT}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": abs(float(env.sim.data.qpos[self.door_adr])),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "btn": (
                float(np.linalg.norm(env.sim.data.site_xpos[self.eef_site_id]
                                     - env.sim.data.geom_xpos[self.btn_gid]))
                if self.btn_gid is not None else 9.9
            ),
        }
        for o in ("bowl", "fruit"):
            d[o] = env.sim.data.body_xpos[self.body[o]].copy()
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

    def _move(self, opos, gd, cursor, T):
        """(t_pick, t_place) for an object grasped/carried starting at/after cursor."""
        r = np.median(opos[max(cursor - self.REST_W, 0): cursor + 1], axis=0)
        dsp = np.linalg.norm(opos - r, axis=1)
        t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
            t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        t_pick = max(t_pick, cursor)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1
        return t_pick, t_place

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in ("bowl", "fruit")}
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        btn = np.array([h["btn"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in ("bowl", "fruit")}
        door_peak = float(door.max())
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # order by first sustained displacement (fruit out first, then the bowl)
        first_move = {}
        for o in ("bowl", "fruit"):
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in ("bowl", "fruit") if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        nav_disp_dbg = 0.0
        for o in order:
            t_pick, t_place = self._move(opos[o], gd[o], cursor, T)
            if o == "fruit":
                emit(t_pick, SUBTASK_FRUIT, STAGE_PICK, f"pick up the {self.fruit_lang} from the bowl")
                emit(t_place, SUBTASK_FRUIT, STAGE_PLACE, f"place the {self.fruit_lang} on the small plate")
            else:
                emit(t_pick, SUBTASK_BOWL, STAGE_PICK, f"pick up the bowl with the {self.meat_lang}")
                # the bowl is carried across to the microwave: adaptive nav during the carry
                moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
                t_arrive = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
                nav_disp = (
                    float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                    if base_xy is not None and t_arrive > t_pick else 0.0
                )
                nav_disp_dbg = round(nav_disp, 2)
                if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                    emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the bowl to the microwave")
                emit(t_place, SUBTASK_BOWL, STAGE_PLACE, "place the bowl in the microwave")
            n_moved += 1

        t_closed = (
            self._sustained_start(door < self.CLOSE_FRAC * door_peak, cursor, self.DOOR_SUSTAIN)
            if door_peak >= self.OPEN_MIN else None
        )
        has_close = False
        if t_closed is not None and t_closed > cursor:
            emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, "close the microwave door")
            has_close = True
        pressed = bool((btn[cursor:] < self.BUTTON_TH).any()) if cursor < T else False
        if cursor < T - 1:
            emit(T - 1, SUBTASK_ON, STAGE_EXECUTE, "turn on the microwave")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved (fruit + bowl)")
        if not has_close:
            review.append("microwave door never closed -- verify")
        if not pressed:
            review.append("gripper never neared the start button -- verify turn-on")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "nav_disp": nav_disp_dbg,
            "t_closed": t_closed if t_closed is not None else -1,
            "door_peak": round(door_peak, 2), "pressed": pressed, "n_segments": len(segments),
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

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseMicrowave, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_ON) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnMicrowave, found {count(SUBTASK_ON)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
