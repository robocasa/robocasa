import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_WAIT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

OFF_TH = 0.05      # folded |knob| at/below this == fully off
DESC_DROP = 0.30   # knob dropping this far below its on-plateau == the robot is rotating it off


# The saucepan simmers on a lit burner (knob ON at the start). Wait for the sauce to
# simmer to a deep red, then turn the burner off. TurnOffStove is a bank atomic; the
# waiting is a coined SimmerSauce[wait]. The robot spawns at the stove and does not move.
SUBTASK_WAIT = "SimmerSauce"      # coined: the idle wait while the sauce simmers
SUBTASK_TURNOFF = "TurnOffStove"  # bank atomic

WAIT_LANG = "wait for the sauce to simmer"
DONE_LANG = "task complete"

KNOB_ON = 0.35  # folded |knob| >= this == burner on


@register_labeler
class TurnOffSimmeredSauceHeatLabeler(BaseSubtaskLabeler):
    """Labeler for the TurnOffSimmeredSauceHeat composite task.

    Task (mobile, local): a saucepan of marinara simmers on a lit burner (its knob is ON
    at the start). Wait for the sauce to simmer to a deep red (~450-500 sim steps), then
    turn the burner off. Robot starts at the stove; no navigation, no pick-place.

    Doc-canonical count = 2 (`mobile: Yes`): the simmer WAIT + the burner turn-off.
      SimmerSauce [wait]     -> wait for the sauce to simmer
      TurnOffStove [execute] -> turn off the burner
    plus a terminal done.

    Boundary: the WAIT ends when the sauce is fully simmered. The sim colors the sauce by
    task_timestep/simmer_timesteps (interpolating start->finished), and task_timestep equals
    the frame index, so the sauce reaches "done" at frame == simmer_timesteps (in ep_meta
    refs). The color itself is a model geom_rgba written during the rollout and is NOT in the
    replayed state, but simmer_timesteps fully determines it -- a far more reliable, uniform
    boundary than the gripper/knob motion (which the robot does at wildly varying, sometimes
    much-later, times). So SimmerSauce[wait] = [0, simmer_timesteps] and TurnOffStove =
    [simmer_timesteps, done] (the reach + rotate), completing when the knob reaches off.
    """

    ENV_NAME = "TurnOffSimmeredSauceHeat"

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        knob = ep_meta.get("refs", {}).get("knob") or getattr(env, "knob", None)
        self.burner_name = knob.replace("_", " ") if knob else ""
        # frames of simmering the demo waited before the burner is turned off; the sim
        # colors the sauce by task_timestep/simmer_timesteps and task_timestep == frame, so
        # the sauce is fully simmered ("done") at frame == simmer_timesteps.
        try:
            self.simmer_timesteps = float(ep_meta["refs"]["simmer_timesteps"])
        except Exception:
            self.simmer_timesteps = None
        self.knob_adr = None
        if knob is not None:
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{knob}_joint")
            except Exception:
                self.knob_adr = None
        self.extra_subtasks = {SUBTASK_WAIT}
        self.expected_done_idx = None
        # the turn-off itself is legitimately brief; don't flag it as too-short
        self.min_segment_frames = None

    def per_step(self, env, t: int) -> dict:
        v = abs(float(env.sim.data.qpos[self.knob_adr])) if self.knob_adr is not None else 0.0
        return {"knob": float(min(v, 2 * np.pi - v))}

    def resolve(self, history):
        T = len(history)
        knob = np.array([h["knob"] for h in history])
        below = np.where(knob < KNOB_ON)[0]

        # the WAIT ends when the sauce is fully simmered -- at frame == simmer_timesteps (the
        # sim's sauce-color progress, in ep_meta). But if the robot begins rotating the knob
        # slightly before that frame, start the turn-off at the rotation onset instead, so a
        # quick turn-off right at "done" still gets a full segment. Rotation onset = the first
        # frame the knob drops well below its on-plateau.
        plateau = float(np.median(knob[: max(int(0.3 * T), 1)]))
        desc = np.where(knob < plateau - DESC_DROP)[0]
        desc_onset = int(desc[0]) if desc.size else (T - 1)
        if self.simmer_timesteps is not None:
            t_turn_start = min(int(round(self.simmer_timesteps)), desc_onset)
        else:
            t_turn_start = desc_onset
        t_turn_start = int(min(max(t_turn_start, 1), max(T - 2, 1)))

        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if not below.size:
            review.append("burner never turned off -- verify")
        # turn-off runs to the end (reach/rotate + withdrawal); demos are cut right after
        # the burner goes off, so 'done' is just the final frame.
        emit(t_turn_start, SUBTASK_WAIT, STAGE_WAIT, WAIT_LANG)
        emit(T - 1, SUBTASK_TURNOFF, STAGE_EXECUTE, f"turn off {burner}")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1
        result = self.build_result(T, segments, debug={
            "t_turn_start": t_turn_start, "desc_onset": desc_onset, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_WAIT}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_WAIT) & (stages == STAGE_WAIT))) != 1:
            issues.append("expected 1 SimmerSauce[wait]")
        if int(np.sum((names == SUBTASK_TURNOFF) & (stages == STAGE_EXECUTE))) != 1:
            issues.append("expected 1 TurnOffStove[execute]")
        return issues
