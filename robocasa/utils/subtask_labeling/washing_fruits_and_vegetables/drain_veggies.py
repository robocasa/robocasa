import numpy as np
from robosuite.utils import transform_utils as TU

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_TILT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Get the vegetable out of the pot into the sink, turn the faucet on to wash it, turn it off,
# then put the vegetable back in the pot. Two cycles bracket the faucet on/off. The FIRST cycle
# comes in two flavours (the overwhelming majority of demos POUR):
#   POUR (~90%): pick up the pot, tilt it to pour the veg into the sink, put the pot down.
#     -> Pick / Tilt / Place extensions (like DumpLeftovers); the pot is grasped the whole cycle.
#   LIFT (~10%): lift the veg straight out of the pot and set it in the sink (pot stays put).
#     -> PickPlacePotToSink (ext) pick + place.
# The SECOND cycle (after washing) is always a veg lift: pick the veg out of the sink and put it
# back in the pot -> PickPlaceSinkToPot (ext). TurnOn/OffSinkFaucet are bank. Doc count = 4.
SUBTASK_IN = "PickPlacePotToSink"     # lift-flavour cycle 1 (veg)
SUBTASK_OUT = "PickPlaceSinkToPot"    # cycle 2 (veg)
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_OFF = "TurnOffSinkFaucet"
POUR_PICK = "Pick"                    # pour-flavour cycle 1 (pot) -- coined, like DumpLeftovers
POUR_TILT = "Tilt"
POUR_PLACE = "Place"

DONE_LANG = "task complete"
VEG = "obj"
POT = "obj_container"


@register_labeler
class DrainVeggiesLabeler(BaseSubtaskLabeler):
    """Labeler for the DrainVeggies composite task.

    Task: drain/wash the vegetable. Cycle 1 empties the pot into the sink -- by POURING (pick
    up the pot, tilt it over the sink, set it back down) in the large majority of demos, or by
    LIFTING the veg straight out in a minority. Then TurnOnSinkFaucet -> TurnOffSinkFaucet, then
    cycle 2 lifts the washed veg from the sink back into the pot.

    Pour vs lift is detected from the pot's tilt (max tilt > POUR_TILT_TH => pour). Pour cycle:
      Pick(pot) [grab+lift] -> Tilt(pour) [lift..peak tilt] -> Place(pot) [peak tilt..release].
    Lift cycle: PickPlacePotToSink pick [veg first move] + place [veg release in the sink].
    """

    ENV_NAME = "DrainVeggies"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.09
    REGRASP_SUSTAIN = 5
    DEPART_TOL = 0.08   # gripper displacement from the faucet handle that ends the turn-off
    OFF_MIN = 12        # floor the turn-off span (a handle flick is a real, if brief, action)
    # pour vs lift: genuine pours tilt the pot 90-160 deg; the ~10 lifts keep it < 20 deg.
    # 60 cleanly splits them (a lone 38-deg borderline demo falls to the lift path).
    POUR_TILT_TH = 60.0

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_bid = env.obj_body_id[VEG]
        self.pot_bid = env.obj_body_id.get(POT)
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        try:
            self.obj_lang = env.get_obj_lang(VEG)
        except Exception:
            self.obj_lang = "vegetable"
        self.extra_subtasks = {SUBTASK_IN, SUBTASK_OUT, POUR_PICK, POUR_TILT, POUR_PLACE}
        self.expected_done_idx = None
        self.min_segment_frames = 10
        self.pour = False

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        d = {
            "pos": env.sim.data.body_xpos[self.obj_bid].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "water_on": won,
        }
        if self.pot_bid is not None:
            d["pot_pos"] = env.sim.data.body_xpos[self.pot_bid].copy()
            d["pot_quat"] = np.array(env.sim.data.body_xquat[self.pot_bid])
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

    def _first_move(self, pos, cursor, T):
        rest = np.median(pos[max(cursor - self.REST_W, 0): cursor + 1], axis=0)
        dsp = np.linalg.norm(pos - rest, axis=1)
        t = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t is None:
            mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
            t = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        return max(int(t), cursor)

    def _release(self, gd, t_move, T, after=None):
        hold = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        start = (after if after is not None else t_move + self.MIN_CARRY)
        t = self._sustained_start(gd > hold + self.RELEASE_MARGIN, start, self.RELEASE_SUSTAIN)
        return int(t) if t is not None else T - 1

    def resolve(self, history):
        T = len(history)
        pos = np.array([h["pos"] for h in history])
        grip = np.array([h["grip"] for h in history])
        won = np.array([h["water_on"] for h in history])
        gd = np.linalg.norm(grip - pos, axis=1)
        has_pot = self.pot_bid is not None and "pot_pos" in history[0]
        if has_pot:
            pot_pos = np.array([h["pot_pos"] for h in history])
            pot_q = np.array([h["pot_quat"] for h in history])
            pot_up = np.array([TU.quat2mat(q[[1, 2, 3, 0]])[:, 2] for q in pot_q])
            pot_tilt = np.degrees(np.arccos(np.clip(pot_up @ pot_up[0], -1.0, 1.0)))
            gd_pot = np.linalg.norm(grip - pot_pos, axis=1)
        else:
            pot_tilt = np.zeros(T)
        self.pour = bool(pot_tilt.max() > self.POUR_TILT_TH)

        w = self.obj_lang
        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # ---- cycle 1: empty the pot into the sink ----
        if self.pour:
            t_pot_move = self._first_move(pot_pos, 0, T)
            t_peak = t_pot_move + int(np.argmax(pot_tilt[t_pot_move:]))
            t_pot_place = self._release(gd_pot, t_pot_move, T, after=t_peak)
            # fallback: pot comes back to rest (untilted, back down) if no clean release
            if t_pot_place >= T - 1:
                rest = np.median(pot_pos[: self.REST_W], axis=0)
                back = (np.linalg.norm(pot_pos - rest, axis=1) < self.MOVE_TOL) & (pot_tilt < 10.0)
                s = self._sustained_start(back, t_peak, self.MOVE_SUSTAIN)
                if s is not None:
                    t_pot_place = s
            emit(t_pot_move, POUR_PICK, STAGE_PICK, "pick up the pot")
            emit(t_peak, POUR_TILT, STAGE_TILT, f"pour the {w} into the sink")
            emit(t_pot_place, POUR_PLACE, STAGE_PLACE, "put the pot down")
        else:
            t_move1 = self._first_move(pos, 0, T)
            t_place1 = self._release(gd, t_move1, T)
            emit(t_move1, SUBTASK_IN, STAGE_PICK, f"pick up the {w} from the pot")
            emit(t_place1, SUBTASK_IN, STAGE_PLACE, f"place the {w} in the sink")

        # ---- faucet on (wash) then off ----
        onf = np.where(won > 0.5)[0]
        onf = onf[onf >= cursor] if onf.size else onf
        has_water = onf.size > 0
        t_off_start = int(min(max(int(onf[-1]) + 1, cursor + 1), T - 1)) if has_water \
            else min(cursor + 1, T - 1)
        emit(t_off_start, SUBTASK_ON, STAGE_EXECUTE, f"turn on the sink faucet to wash the {w}")
        # The turn-off is quick: it ends when the gripper leaves the faucet handle. The travel
        # back to the veg in the sink belongs to the following cycle-2 pick, not the faucet.
        faucet = np.median(grip[max(t_off_start - 3, 0): t_off_start + 1], axis=0)
        dep = np.linalg.norm(grip - faucet, axis=1)
        t_off_end = self._sustained_start(dep >= self.DEPART_TOL, t_off_start, self.MOVE_SUSTAIN)
        if t_off_end is None:
            t_off_end = self._sustained_start(gd < self.GRASP_TH, t_off_start, self.REGRASP_SUSTAIN)
        if t_off_end is None:
            t_off_end = min(t_off_start + 1, T - 1)
        t_off_end = min(max(t_off_end, t_off_start + self.OFF_MIN), T - 1)
        emit(t_off_end, SUBTASK_OFF, STAGE_EXECUTE, "turn off the sink faucet")

        # ---- cycle 2: lift the veg from the sink back into the pot ----
        t_move2 = self._first_move(pos, cursor, T)
        t_place2 = self._release(gd, t_move2, T)
        emit(t_move2, SUBTASK_OUT, STAGE_PICK, f"pick up the {w} from the sink")
        emit(t_place2, SUBTASK_OUT, STAGE_PLACE, f"put the {w} back in the pot")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if not has_water:
            review.append("sink water never registered on -- verify the wash")
        result = self.build_result(T, segments, debug={
            "pour": self.pour, "max_pot_tilt": float(pot_tilt.max()),
            "t_off_start": int(t_off_start), "t_off_end": int(t_off_end),
            "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_IN, SUBTASK_OUT, POUR_PICK, POUR_TILT, POUR_PLACE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]

        def n(name):
            return int(np.sum(names == name))

        if self.pour:
            for req in (POUR_PICK, POUR_TILT, POUR_PLACE):
                if n(req) != 1:
                    issues.append(f"expected 1 {req}, found {n(req)}")
            if n(SUBTASK_OUT) != 2:
                issues.append(f"expected 2 {SUBTASK_OUT}, found {n(SUBTASK_OUT)}")
        else:
            if n(SUBTASK_IN) != 2:
                issues.append(f"expected 2 {SUBTASK_IN}, found {n(SUBTASK_IN)}")
            if n(SUBTASK_OUT) != 2:
                issues.append(f"expected 2 {SUBTASK_OUT}, found {n(SUBTASK_OUT)}")
        for req in (SUBTASK_ON, SUBTASK_OFF):
            if n(req) != 1:
                issues.append(f"expected 1 {req}, found {n(req)}")
        return issues
