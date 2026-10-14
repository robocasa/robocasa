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


# Open the (closed) microwave, put the food from the counter inside, close the door,
# and turn it on. OpenMicrowave + PickPlaceCounterToMicrowave + CloseMicrowave +
# TurnOnMicrowave are bank atomics; NavigateKitchen is bank (adaptive -- the counter
# is beside the microwave).
SUBTASK_OPEN = "OpenMicrowave"
SUBTASK_MOVE = "PickPlaceCounterToMicrowave"
SUBTASK_CLOSE = "CloseMicrowave"
SUBTASK_ON = "TurnOnMicrowave"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class MicrowaveThawingLabeler(BaseSubtaskLabeler):
    """Labeler for the MicrowaveThawing composite task.

    Task: a frozen food sits on the counter; the microwave is CLOSED (robot starts at
    the microwave). Open the microwave, put the food inside, close the door, and turn
    it on.

    Doc-canonical count = 4 (`mobile: Yes`): OpenMicrowave + PickPlaceCounterToMicrowave
    + CloseMicrowave + TurnOnMicrowave. Structure: OpenMicrowave -> food[pick] -> (nav)
    -> [place] -> CloseMicrowave -> TurnOnMicrowave -> done. The nav is ADAPTIVE.

    Boundaries (door joint `{name}_microjoint`; a state replay, so the microwave's
    cached turned_on is unusable -- the turn-on is read from the gripper pressing the
    start button geom):
      - t_open: the door swings open (reaches OPEN_FRAC of its peak). OpenMicrowave ends.
      - food pick/place: first displacement -> gripper RELEASE (food set in microwave).
      - t_closed: the door swings back shut (< CLOSE_FRAC of its peak) after the place.
      - TurnOnMicrowave: from the close to the end (the robot reaches for and presses
        the start button); flagged for review if the gripper never nears the button.
    """

    ENV_NAME = "MicrowaveThawing"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    OPEN_FRAC = 0.80
    CLOSE_FRAC = 0.30
    DOOR_SUSTAIN = 5
    OPEN_MIN = 0.05  # door joint peak below this (rad) == it never opened
    BUTTON_TH = 0.09  # gripper within this of the start button == pressing it
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.obj_lang = env.get_obj_lang("obj")
        except Exception:
            self.obj_lang = "food"

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

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": abs(float(env.sim.data.qpos[self.door_adr])),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        d["btn"] = (
            float(np.linalg.norm(env.sim.data.site_xpos[self.eef_site_id]
                                 - env.sim.data.geom_xpos[self.btn_gid]))
            if self.btn_gid is not None else 9.9
        )
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
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        btn = np.array([h["btn"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        door_peak = float(door.max())
        has_open = door_peak >= self.OPEN_MIN

        # OpenMicrowave: the door first swings open
        if has_open:
            t_open = self._sustained_start(door >= self.OPEN_FRAC * door_peak, 0, self.DOOR_SUSTAIN)
            if t_open is None:
                t_open = int(np.argmax(door))
        else:
            t_open = 0

        # food: pick + place (into the microwave), after the door is open
        rest = np.median(obj[max(t_open, 0): max(t_open, 0) + self.REST_W], axis=0) if t_open < T else np.median(obj[:self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, t_open, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp[t_open:] >= self.MOVE_TOL)[0]
            t_pick = t_open + int(mv[0]) if mv.size else max(t_open, T - 1)
        t_pick = max(t_pick, t_open)
        gd = np.linalg.norm(grip - obj, axis=1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        nav_disp = (
            float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
            if base_xy is not None and t_arrive > t_pick else 0.0
        )

        # CloseMicrowave: the door swings shut after the food is placed
        t_closed = self._sustained_start(door < self.CLOSE_FRAC * door_peak, t_placed, self.DOOR_SUSTAIN) if has_open else None

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if has_open:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the microwave")
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.obj_lang} from the counter")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.obj_lang} to the microwave")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.obj_lang} on the plate in the microwave")
        has_close = False
        if t_closed is not None and t_closed > cursor:
            emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, "close the microwave door")
            has_close = True
        # TurnOnMicrowave: reach for and press the start button, to the end
        pressed = bool((btn[cursor:] < self.BUTTON_TH).any()) if cursor < T else False
        if cursor < T - 1:
            emit(T - 1, SUBTASK_ON, STAGE_EXECUTE, "turn on the microwave")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_open:
            review.append("microwave door never opened -- verify")
        if not has_close:
            review.append("microwave door never closed -- verify")
        if not pressed:
            review.append("gripper never neared the start button -- verify turn-on")

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "t_pick": t_pick, "t_placed": t_placed,
            "t_closed": t_closed if t_closed is not None else -1,
            "door_peak": round(door_peak, 2), "pressed": pressed,
            "nav_disp": round(nav_disp, 2), "n_segments": len(segments),
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

        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenMicrowave, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 food pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 food place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseMicrowave, found {count(SUBTASK_CLOSE)}")
        if count(SUBTASK_ON) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnMicrowave, found {count(SUBTASK_ON)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
