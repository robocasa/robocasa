import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# The saucepan simmers on a low burner with its lid on. Turn the burner off and lift the
# lid off the saucepan onto the counter. TurnOffStove and PickPlaceStoveToCounter are bank
# atomics. Robot starts at the stove (local, no drive).
SUBTASK_TURNOFF = "TurnOffStove"
SUBTASK_LID = "PickPlaceStoveToCounter"  # lid: saucepan -> counter

DONE_LANG = "task complete"

KNOB_ON = 0.35


@register_labeler
class StopSlowCookingLabeler(BaseSubtaskLabeler):
    """Labeler for the StopSlowCooking composite task.

    Task (mobile, local): a saucepan simmers on a low front burner (knob LOW at the start)
    with its lid on. Turn the burner off and move the lid onto the counter. Robot starts
    at the stove.

    Doc-canonical count = 2 (`mobile: Yes`): the burner turn-off + the lid move.
      TurnOffStove [execute]                    -> turn off the burner
      PickPlaceStoveToCounter [pick]/[place]    -> lid off the saucepan onto the counter
    plus a terminal done. Emitted in temporal order (usually turn-off first).

    Boundaries: t_knob_off = first frame the knob leaves the on-range; the lid's t_pick =
    its first sustained displacement, t_place = the gripper RELEASE.
    """

    ENV_NAME = "StopSlowCooking"
    LID = "saucepan_auxiliary"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.lid_body_id = env.obj_body_id[self.LID]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        knob = ep_meta.get("refs", {}).get("knob") or getattr(env, "knob", None)
        self.burner_name = knob.replace("_", " ") if knob else ""
        self.knob_adr = None
        if knob is not None:
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{knob}_joint")
            except Exception:
                self.knob_adr = None
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        v = abs(float(env.sim.data.qpos[self.knob_adr])) if self.knob_adr is not None else 0.0
        return {
            "lid": env.sim.data.body_xpos[self.lid_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knob": float(min(v, 2 * np.pi - v)),
        }

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
        lid = np.array([h["lid"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])

        off = np.where(knob < KNOB_ON)[0]
        t_knob_off = int(off[0]) if off.size else None

        rest = np.median(lid[: self.REST_W], axis=0)
        disp = np.linalg.norm(lid - rest, axis=1)
        gd = np.linalg.norm(grip - lid, axis=1)
        t_lid_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_lid_pick is None:
            t_lid_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_lid_pick: min(t_lid_pick + self.HOLD_W, T)]))
        t_lid_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                            t_lid_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_lid_place is None:
            t_lid_place = T - 1

        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"
        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def emit_turnoff(end):
            emit(end, SUBTASK_TURNOFF, STAGE_EXECUTE, f"turn off {burner}")

        def emit_lid():
            emit(t_lid_pick, SUBTASK_LID, STAGE_PICK, "pick up the lid from the saucepan")
            emit(t_lid_place, SUBTASK_LID, STAGE_PLACE, "place the lid on the counter")

        if t_knob_off is None:
            review.append("burner never turned off -- verify")
            emit_lid()
        elif t_knob_off <= t_lid_pick:
            emit_turnoff(t_knob_off)  # turn off first, then lift the lid
            emit_lid()
        else:
            emit_lid()               # lid removed first, then turn off
            emit_turnoff(T)

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1
        result = self.build_result(T, segments, debug={
            "t_knob_off": t_knob_off, "t_lid_pick": int(t_lid_pick),
            "t_lid_place": int(t_lid_place), "n_segments": len(segments),
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

        if count(SUBTASK_TURNOFF, STAGE_EXECUTE) != 1:
            issues.append(f"expected 1 TurnOffStove[execute], found {count(SUBTASK_TURNOFF, STAGE_EXECUTE)}")
        if count(SUBTASK_LID, STAGE_PICK) != 1 or count(SUBTASK_LID, STAGE_PLACE) != 1:
            issues.append("expected 1 lid pick + 1 lid place")
        return issues
