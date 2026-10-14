import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_WAIT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Turn the pot's burner on, wait for the pot to heat up (~30s), then add the vegetable.
# TurnOnStove and PickPlaceCounterToStove are bank atomics; the heat-up wait has no bank
# name so it is a coined extension (PreheatPot). The burner starts OFF.
SUBTASK_TURN = "TurnOnStove"
SUBTASK_WAIT = "PreheatPot"
SUBTASK_MOVE = "PickPlaceCounterToStove"

DONE_LANG = "task complete"

KNOB_ON = 0.35


@register_labeler
class PreheatPotLabeler(BaseSubtaskLabeler):
    """Labeler for the PreheatPot composite task.

    Task (mobile, usually local): a pot sits on an OFF burner; a vegetable sits on the
    counter. Turn the burner on, let the pot heat up (the env requires ~500 steps of
    burner-on before the vegetable goes in), then add the vegetable. Robot at the stove.

    Doc-canonical count = 2 (`mobile: Yes`), action-level (turn-on + add-veg), with the
    heat-up wait broken out:
      TurnOnStove [execute]            -> turn on the burner
      PreheatPot [wait]                -> wait for the pot to heat up
      PickPlaceCounterToStove [pick]   -> pick up the vegetable from the counter
      PickPlaceCounterToStove [place]  -> place the vegetable in the pot
    plus a terminal done.

    Boundaries: TurnOnStove ends at the turn-completion (last frame the knob is below
    its settled on-value); the wait runs until the gripper leaves the knob to reach the
    vegetable (first sustained gripper motion after the turn); t_pick = the vegetable
    first displaces; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PreheatPot"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    GRIP_SUSTAIN = 5
    REACH_MIN = 15       # floor for the reach/grab (pick) segment so it isn't flagged short
    APPROACH_MIN = 0.05  # min gripper->veg closing (m) to count as a real reach
    APPROACH_FRAC = 0.75  # reach begins where gd leaves this fraction of the hold->grab span

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["vegetable"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.veg_lang = env.get_obj_lang("vegetable")
        except Exception:
            self.veg_lang = "vegetable"
        knob = ep_meta.get("refs", {}).get("knob") or getattr(env, "knob", None)
        self.burner_name = knob.replace("_", " ") if knob else ""
        self.knob_adr = None
        if knob is not None:
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{knob}_joint")
            except Exception:
                self.knob_adr = None
        self.extra_subtasks = {SUBTASK_WAIT}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        v = abs(float(env.sim.data.qpos[self.knob_adr])) if self.knob_adr is not None else 0.0
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
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
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        gd = np.linalg.norm(grip - obj, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # TurnOnStove first: the knob rises to on, before the pot heats and the veg moves.
        peak = float(knob.max()) if T else 0.0
        has_heat = peak >= KNOB_ON
        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if has_heat:
            t_on = self._sustained_start(knob >= min(0.9 * peak, peak - 1e-6), 0, 5)
            if t_on is None:
                t_on = int(np.argmax(knob >= KNOB_ON)) if (knob >= KNOB_ON).any() else 0
            t_on = int(min(max(t_on, 1), max(t_pick - 1, 1)))
            emit(t_on, SUBTASK_TURN, STAGE_EXECUTE, f"turn on {burner}")
            # wait ends "when the robot goes for the item": the reach onset is where the
            # gripper->veg distance leaves its heat-up plateau and descends into the grab
            # at t_pick. Using distance (not gripper speed) captures a slow approach that a
            # speed threshold would bury in the wait, which is what collapsed seg2 before.
            gd_pick = float(np.min(gd[max(t_pick - 5, 0): t_pick + 1]))
            span = gd[t_on:t_pick]
            gd_hold = float(np.percentile(span, 75)) if span.size else gd_pick
            if gd_hold - gd_pick > self.APPROACH_MIN:
                thr = gd_pick + self.APPROACH_FRAC * (gd_hold - gd_pick)
                # walk back through the final below-threshold descent to its onset
                tt = int(t_pick) - 1
                while tt > t_on and gd[tt] < thr:
                    tt -= 1
                t_reach = tt
            else:
                # gripper stayed by the veg through the heat-up: only a short final grab
                t_reach = int(t_pick) - self.REACH_MIN
            # keep the wait non-empty and the reach/grab at least REACH_MIN frames
            t_reach = int(min(max(t_reach, cursor + 1), max(t_pick - self.REACH_MIN, cursor + 1)))
            # Fixed canonical duration: the env requires ~500 burner-on steps of heat-up
            # (per-demo waits average ~500); the label stays fixed even when the actual
            # seg is trimmed so seg2 captures the reach.
            emit(t_reach, SUBTASK_WAIT, STAGE_WAIT,
                 "wait about 500 timesteps for the pot to heat up")
        else:
            review.append("burner never turned on -- verify")

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.veg_lang} from the counter")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.veg_lang} in the pot")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_pick": t_pick, "t_placed": t_placed, "knob_peak": round(peak, 2),
            "has_heat": int(has_heat), "n_segments": len(segments),
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

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_TURN) != 1:
            issues.append(f"expected 1 TurnOnStove, found {count(SUBTASK_TURN)}")
        if count(SUBTASK_WAIT) not in (0, 1):
            issues.append(f"expected 0-1 PreheatPot wait, found {count(SUBTASK_WAIT)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 vegetable pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 vegetable place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        return issues
