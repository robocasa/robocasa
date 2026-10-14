import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_WAIT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Turn off the burner under the (already-boiled) kettle, wait for it to cool, then
# carry the kettle from the stove to the dining table.
SUBTASK_TURNOFF = "TurnOffStove"          # bank atomic; splits execute (turn) + wait (cool)
SUBTASK_MOVE = "PickPlaceStoveToCounter"  # bank atomic: stove -> counter (dining table)
SUBTASK_NAV = "NavigateKitchen"

TURNOFF_LANG = "turn off the stove burner"
WAIT_LANG = "wait for the kettle to cool"
NAV_LANG = "carry the kettle to the dining table"
DONE_LANG = "task complete"


@register_labeler
class CoolKettleLabeler(BaseSubtaskLabeler):
    """Labeler for the CoolKettle composite task.

    Task: the kettle is done boiling on the stove (its burner knob is on at the
    start). Turn the burner off, wait ~150 steps for the kettle to cool, then
    carry it from the stove to the dining table.

    Doc-canonical count = 4 (`mobile: Yes`): TurnOffStove (one subtask, split into
    the turn + the cooling wait -- like TurnOnToaster's press/wait in
    GetToastedBread) + the kettle's pick-place, which is INTERRUPTED by a
    navigation to the dining table -> PickPlaceStoveToCounter[pick] ->
    NavigateKitchen -> PickPlaceStoveToCounter[place]. 6 segments with the nav
    (expected_done_idx = 5); the nav is adaptive but the base drove on every demo.

    Boundaries:
      - t_knob_off: first frame the target burner's knob leaves the "on" range
                (|angle| in [0.35, 2*pi-0.35]) -- the knob starts on. The burner is
                identified from ep_meta refs.knob. TurnOffStove[execute] ends here.
      - t_pick_start: last frame in [t_knob_off, t_lift] the gripper is still FAR
                (> REACH_TH) from the kettle -- i.e. the cooling wait ends and the
                reach-to-grasp begins. TurnOffStove[wait] ends here. (The kettle is
                a large, handle-grabbed object, so the reach is read from the
                gripper->kettle distance, not a tight grasp test.)
      - t_lift:  kettle Z rises LIFT_TOL above its rest (stove) height -> pick ->
                carry boundary.
      - nav_end: frame after the last base-motion burst in [t_lift, t_placed].
      - t_placed: kettle first settles within PLACE_TOL of its final (on-table)
                pose.
    """

    ENV_NAME = "CoolKettle"

    KNOB_ON = 0.35
    REACH_TH = 0.28  # gripper->kettle beyond this == not yet reaching (still waiting)
    COOLDOWN = 150  # the cooling wait is >= this many steps (the sim's cooldown timer),
    # so the kettle provably stays on the stove at least this long after the knob-off
    LIFT_TOL = 0.06
    LIFT_SUSTAIN = 10
    ARRIVE_TOL = 0.20  # kettle within this of its final pose == arrived at the table
    ARRIVE_SUSTAIN = 8
    RELEASE_MARGIN = 0.05  # grip->kettle rising this far above the carry level == let go
    RELEASE_SUSTAIN = 5
    REST_W = 20
    MIN_NAV_DIST = 0.40

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.kettle_body_id = env.obj_body_id["kettle"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.kettle_lang = env.get_obj_lang("kettle")
        except Exception:
            self.kettle_lang = "kettle"

        # the burner whose knob is on and gets turned off (from ep_meta)
        knob_loc = ep_meta.get("refs", {}).get("knob")
        self.knob_adr = None
        if knob_loc is not None:
            try:
                self.knob_adr = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{knob_loc}_joint"
                )
            except Exception:
                self.knob_adr = None

        self.base_qpos_adrs = []
        for nm in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
        ):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "kettle": env.sim.data.body_xpos[self.kettle_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knob": (
                float(env.sim.data.qpos[self.knob_adr])
                if self.knob_adr is not None
                else 0.0
            ),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    # ---- helpers -------------------------------------------------------

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        kettle = np.array([h["kettle"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])

        base_xy = (
            np.array([h["base_xy"] for h in history])
            if self.base_qpos_adrs
            else None
        )

        # turn-off: knob leaves the on-range (it starts on)
        nk = np.mod(knob, 2 * np.pi)
        on = (nk >= self.KNOB_ON) & (nk <= 2 * np.pi - self.KNOB_ON)
        off = np.where(~on)[0]
        t_knob_off = int(off[0]) if off.size else 0

        # pick: kettle lifts off the stove
        rest_z = float(np.median(kettle[: self.REST_W, 2]))
        lifted = kettle[:, 2] > rest_z + self.LIFT_TOL
        t_lift = self._sustained_start(lifted, 0, self.LIFT_SUSTAIN)
        if t_lift is None:
            lf = np.where(lifted)[0]
            t_lift = int(lf[0]) if lf.size else T - 1
        t_lift = max(t_lift, t_knob_off)

        # wait -> pick boundary: the last frame the gripper is still far from the
        # kettle (the cooling wait) before the reach-to-grasp that ends in the lift,
        # but never earlier than the fixed cooldown (some demos keep the hand near
        # the kettle through the wait, so the "far" test alone can miss it).
        gd = np.linalg.norm(grip - kettle, axis=1)
        far = np.where(gd[t_knob_off:t_lift] > self.REACH_TH)[0]
        far_based = t_knob_off + int(far[-1]) + 1 if far.size else t_knob_off
        t_pick_start = max(far_based, t_knob_off + self.COOLDOWN)
        t_pick_start = min(max(t_pick_start, t_knob_off), max(t_lift - 1, t_knob_off))

        # carry -> place: split on the kettle ARRIVING near the table (its final
        # pose), not on base motion -- the base often drives right up to the
        # placement, which would otherwise collapse the place stage. The nav is the
        # transport up to the arrival; the place runs until the gripper RELEASES the
        # kettle. The kettle reaches its resting spot well before the robot lets go
        # (25-82 frames early), so a first-settle test ended the place too soon and
        # mislabeled the set-down/release as "done" -- so "placed" = the gripper->
        # kettle distance rising above the carry (hold) level (same fix as BoilPot).
        final = np.median(kettle[-self.REST_W:], axis=0)
        dist = np.linalg.norm(kettle - final, axis=1)
        t_arrive = self._sustained_start(dist < self.ARRIVE_TOL, t_lift, self.ARRIVE_SUSTAIN)
        if t_arrive is None:
            fa = np.where(dist[t_lift:] > self.ARRIVE_TOL)[0]
            t_arrive = t_lift + int(fa[-1]) + 1 if fa.size else t_lift
        hold_gd = float(np.median(gd[t_lift: max(t_arrive, t_lift + 1)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_arrive, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_knob_off, SUBTASK_TURNOFF, STAGE_EXECUTE, TURNOFF_LANG)
        emit(t_pick_start, SUBTASK_TURNOFF, STAGE_WAIT, WAIT_LANG)
        emit(t_lift, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.kettle_lang} from the stove")
        # nav: the transport from the lift to the table arrival, if the base drove
        nav_disp = (
            float(np.linalg.norm(base_xy[min(t_arrive, T - 1)] - base_xy[min(t_lift, T - 1)]))
            if base_xy is not None
            else 0.0
        )
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.kettle_lang} on the dining table")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_knob_off": t_knob_off,
            "t_pick_start": t_pick_start,
            "t_lift": t_lift,
            "t_arrive": t_arrive,
            "t_placed": t_placed,
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

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
        if count(SUBTASK_TURNOFF, STAGE_WAIT) != 1:
            issues.append(f"expected 1 TurnOffStove[wait], found {count(SUBTASK_TURNOFF, STAGE_WAIT)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 kettle pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 kettle place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
