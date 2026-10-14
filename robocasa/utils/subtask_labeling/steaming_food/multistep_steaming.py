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


# Turn on the sink faucet, rinse a vegetable in the sink (counter -> sink), turn the faucet
# off, move the vegetable into a pot sitting by the stove (sink -> pot), then move the pot
# onto a stove burner (counter -> stove). TurnOnSinkFaucet / TurnOffSinkFaucet are bank
# atomics; PickPlaceCounterToSink and PickPlaceCounterToStove are extensions; sink -> pot
# is PickPlaceSinkToPot (extension). The robot starts at the sink.
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_OFF = "TurnOffSinkFaucet"
MOVE_TO_SINK = "PickPlaceCounterToSink"
MOVE_TO_POT = "PickPlaceSinkToPot"
MOVE_POT = "PickPlaceCounterToStove"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class MultistepSteamingLabeler(BaseSubtaskLabeler):
    """Labeler for the MultistepSteaming composite task.

    Task (mobile): turn the sink faucet on, move a vegetable from the counter into the sink
    (to rinse), turn the faucet off, move the vegetable from the sink into a pot standing by
    the stove, then move the pot onto a stove burner.

    Doc-canonical count = 7 (`mobile: Yes`), action-level: TurnOnSinkFaucet + veg->sink +
    TurnOffSinkFaucet + veg->pot + pot->burner, plus the carry navs. Each pick-place move is
    split pick/place, so the per-frame timeline is:
      TurnOnSinkFaucet [execute]                -> [0, faucet on]
      PickPlaceCounterToSink [pick]/[place]     -> vegetable from the counter into the sink
      TurnOffSinkFaucet [execute]               -> [veg in sink, faucet off]
      PickPlaceSinkToPot [pick]/(nav)/[place]   -> vegetable from the sink into the pot
      PickPlaceCounterToStove [pick]/(nav)/[place] -> pot onto the burner
    plus a terminal done.

    Boundaries: faucet on/off from the sink's water_on flag; the vegetable is handled TWICE
    (its two carries are read from its displacement re-anchored at the sink after the first
    drop); pot pick = its first displacement; every place = the gripper RELEASE.
    """

    ENV_NAME = "MultistepSteaming"

    MOVE_TOL = 0.06
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.veg_body_id = env.obj_body_id["vegetable1"]
        self.pot_body_id = env.obj_body_id["pot"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        try:
            self.veg = env.get_obj_lang("vegetable1")
        except Exception:
            self.veg = "vegetable"
        knob = ep_meta.get("refs", {}).get("knob") or getattr(env, "knob", None)
        self.burner_name = knob.replace("_", " ") if knob else ""
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {MOVE_TO_SINK, MOVE_TO_POT}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        won = 0
        if self.sink is not None:
            try:
                won = int(bool(self.sink.get_handle_state(env=env).get("water_on", False)))
            except Exception:
                won = 0
        return {
            "veg": env.sim.data.body_xpos[self.veg_body_id].copy(),
            "pot": env.sim.data.body_xpos[self.pot_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "water_on": won,
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

    def _release(self, gd, t_pick, T):
        hold = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        r = self._sustained_start(gd > hold + self.RELEASE_MARGIN,
                                  t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        return r if r is not None else T - 1

    def resolve(self, history):
        T = len(history)
        veg = np.array([h["veg"] for h in history])
        pot = np.array([h["pot"] for h in history])
        grip = np.array([h["grip"] for h in history])
        won = np.array([h["water_on"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        gd_v = np.linalg.norm(grip - veg, axis=1)
        gd_p = np.linalg.norm(grip - pot, axis=1)

        review = []

        # --- faucet on (start) ---
        onf = np.where(won > 0)[0]
        t_on = int(onf[0]) if onf.size else max(T // 20, 1)
        if not onf.size:
            review.append("faucet never registered on -- verify")

        # --- veg carry #1: counter -> sink ---
        rest1 = np.median(veg[: self.REST_W], axis=0)
        d1 = np.linalg.norm(veg - rest1, axis=1)
        t_v1_pick = self._sustained_start(d1 >= self.MOVE_TOL, t_on, self.MOVE_SUSTAIN)
        if t_v1_pick is None:
            t_v1_pick = min(t_on + 1, T - 1)
        t_v1_pick = max(t_v1_pick, t_on)
        t_v1_place = self._release(gd_v, t_v1_pick, T)

        # --- faucet off: first off-frame after the veg is dropped in the sink ---
        offf = np.where(won[t_v1_place:] == 0)[0]
        t_off = int(t_v1_place + offf[0]) if offf.size else min(t_v1_place + 1, T - 1)
        if not offf.size:
            review.append("faucet never registered off after rinse -- verify")

        # --- veg carry #2: sink -> pot (re-anchor rest in the sink) ---
        a0 = max(t_v1_place, min(t_off, T - 1) - self.REST_W)
        rest2 = np.median(veg[a0: min(t_off, T - 1) + 1], axis=0) if min(t_off, T - 1) + 1 > a0 else veg[t_v1_place]
        d2 = np.linalg.norm(veg - rest2, axis=1)
        t_v2_pick = self._sustained_start(d2 >= self.MOVE_TOL, t_off, self.MOVE_SUSTAIN)
        if t_v2_pick is None:
            mv = np.where(d2[t_off:] >= self.MOVE_TOL)[0]
            t_v2_pick = t_off + int(mv[0]) if mv.size else min(t_off + 1, T - 1)
        t_v2_pick = max(t_v2_pick, t_off)
        t_v2_place = self._release(gd_v, t_v2_pick, T)

        # --- pot -> burner ---
        rest_p = np.median(pot[: self.REST_W], axis=0)
        dp = np.linalg.norm(pot - rest_p, axis=1)
        t_pot_pick = self._sustained_start(dp >= self.MOVE_TOL, t_v2_place, self.MOVE_SUSTAIN)
        if t_pot_pick is None:
            mv = np.where(dp[t_v2_place:] >= self.MOVE_TOL)[0]
            t_pot_pick = t_v2_place + int(mv[0]) if mv.size else min(t_v2_place + 1, T - 1)
        t_pot_pick = max(t_pot_pick, t_v2_place)
        t_pot_place = self._release(gd_p, t_pot_pick, T)

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

        def carry_nav(t_pick, t_place, lang):
            moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
            t_arr = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
            t_arr = min(t_arr, max(t_place - self.MIN_PICK_GAP, cursor))
            if t_arr > cursor and base_disp(t_pick, t_arr) >= self.MIN_NAV_DIST:
                emit(t_arr, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        burner = f"the {self.burner_name} burner" if self.burner_name else "the burner"

        # floor the pick boundaries so a pick never collapses -- e.g. the pot is lifted
        # straight after the vegetable is dropped INTO it, with no separate approach, so
        # its raw pick lands on the place boundary (zero-length).
        t_v1_pick = min(max(t_v1_pick, t_on + self.MIN_PICK_GAP),
                        max(t_v1_place - self.MIN_PICK_GAP, t_on + 1))
        t_v2_pick = min(max(t_v2_pick, t_off + self.MIN_PICK_GAP),
                        max(t_v2_place - self.MIN_PICK_GAP, t_off + 1))
        t_pot_pick = min(max(t_pot_pick, t_v2_place + self.MIN_PICK_GAP),
                         max(t_pot_place - self.MIN_PICK_GAP, t_v2_place + 1))

        emit(t_on, SUBTASK_ON, STAGE_EXECUTE, "turn on the sink faucet")
        emit(t_v1_pick, MOVE_TO_SINK, STAGE_PICK, f"pick up the {self.veg} from the counter")
        emit(t_v1_place, MOVE_TO_SINK, STAGE_PLACE, f"place the {self.veg} in the sink")
        emit(t_off, SUBTASK_OFF, STAGE_EXECUTE, "turn off the sink faucet")
        emit(t_v2_pick, MOVE_TO_POT, STAGE_PICK, f"pick up the {self.veg} from the sink")
        carry_nav(t_v2_pick, t_v2_place, f"carry the {self.veg} to the pot by the stove")
        emit(t_v2_place, MOVE_TO_POT, STAGE_PLACE, f"place the {self.veg} in the pot")
        emit(t_pot_pick, MOVE_POT, STAGE_PICK, "pick up the pot from the counter")
        carry_nav(t_pot_pick, t_pot_place, "carry the pot to the stove")
        emit(t_pot_place, MOVE_POT, STAGE_PLACE, f"place the pot on {burner}")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_on": int(t_on), "t_v1_pick": int(t_v1_pick), "t_v1_place": int(t_v1_place),
            "t_off": int(t_off), "t_v2_pick": int(t_v2_pick), "t_v2_place": int(t_v2_place),
            "t_pot_pick": int(t_pot_pick), "t_pot_place": int(t_pot_place),
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {MOVE_TO_SINK, MOVE_TO_POT}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def cnt(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if cnt(SUBTASK_ON) != 1:
            issues.append(f"expected 1 TurnOnSinkFaucet, found {cnt(SUBTASK_ON)}")
        if cnt(SUBTASK_OFF) != 1:
            issues.append(f"expected 1 TurnOffSinkFaucet, found {cnt(SUBTASK_OFF)}")
        for mv in (MOVE_TO_SINK, MOVE_TO_POT, MOVE_POT):
            if cnt(mv, STAGE_PICK) != 1 or cnt(mv, STAGE_PLACE) != 1:
                issues.append(f"expected 1 pick + 1 place for {mv}")
        return issues
