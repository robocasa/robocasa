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


# Carry the water-filled pot from the counter by the sink to the stove (a
# nav-interrupted pick-place), then turn on the burner under it. All three names
# are bank atomics.
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_TURNON = "TurnOnStove"

DONE_LANG = "task complete"
NAV_LANG = "carry the pot to the stove"


@register_labeler
class BoilPotLabeler(BaseSubtaskLabeler):
    """Labeler for the BoilPot composite task.

    Task: pick up the pot (filled with water) from the counter by the sink, carry
    it to the stove and set it on a burner, then turn that burner on.

    Doc-canonical count = 4 (`mobile: Yes`): the pot's pick-place is INTERRUPTED by
    a navigation (the pot spawns by the sink, the stove is across the kitchen), so
    it expands into PickPlaceCounterToStove[pick] -> NavigateKitchen ->
    PickPlaceCounterToStove[place]; then TurnOnStove. 5 segments with the nav
    (expected_done_idx = 4). The nav is ADAPTIVE (MIN_NAV_DIST) in case a scene
    ever places the pot within reach of the stove; the base drove on every demo
    observed, so the nav is essentially always present.

    Boundaries (each from its own signal):
      - t_lift:  pot Z first rises LIFT_TOL above its rest (counter) height and
                 stays up -> the pick -> carry boundary.
      - nav_end: frame after the last base-motion burst in [t_lift, t_placed],
                 gated on MIN_NAV_DIST net displacement (the pot is a heavy handle-
                 grabbed object, so the carry is read from the pot/base motion, not
                 the gripper distance).
      - t_placed: the pot first settles within PLACE_TOL of its final (on-stove)
                 pose and stays -> the place completes.
      - t_knob_on: first frame the burner the pot ends on has its knob turned to
                 the "on" range (|angle| in [0.35, 2*pi-0.35], matching the stove
                 fixture's is_burner_on). The turned burner is auto-detected.
    """

    ENV_NAME = "BoilPot"

    LIFT_TOL = 0.06
    LIFT_SUSTAIN = 10
    ARRIVE_TOL = 0.20  # pot within this of its final on-stove pose == arrived
    ARRIVE_SUSTAIN = 8
    RELEASE_MARGIN = 0.05  # grip->pot rising this far above the carry level == let go
    RELEASE_SUSTAIN = 5
    KNOB_ON = 0.35  # |knob angle| in [KNOB_ON, 2*pi-KNOB_ON] == burner on
    REST_W = 20
    NAV_MOVING_TH = 0.005
    NAV_SMOOTH_W = 5
    # Every observed demo drives the base at least ~0.37 m from the pick to the
    # placement (the pot always spawns across the kitchen from the stove); the
    # displacement distribution is continuous from 0.37 up with NOTHING near zero,
    # so there is no genuine "pot already in reach" case. Gate the nav in the gap
    # between 0 and 0.37 -- 0.30 keeps a floor above gripper jitter while giving the
    # two shortest carries (demo_39 0.369, demo_93 0.377) their NavigateKitchen.
    MIN_NAV_DIST = 0.30

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.pot_body_id = env.obj_body_id["pot"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.pot_lang = env.get_obj_lang("pot")
        except Exception:
            self.pot_lang = "pot"

        # Stove knob joints (one per real burner). Track them all and let resolve
        # pick the one that gets turned on.
        stove = env.stove
        prefix = stove.naming_prefix
        self.knob_locs, self.knob_adrs = [], []
        for loc, joint in stove.knob_joints.items():
            if joint is None:
                continue
            try:
                adr = env.sim.model.get_joint_qpos_addr(f"{prefix}knob_{loc}_joint")
            except Exception:
                continue
            self.knob_locs.append(loc)
            self.knob_adrs.append(adr)

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
            "pot": env.sim.data.body_xpos[self.pot_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knobs": np.array(
                [float(env.sim.data.qpos[a]) for a in self.knob_adrs]
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

    def _nav_end(self, base_speed, base_xy, a, b):
        if base_xy is None or b - a < 2:
            return a
        seg = base_speed[a:b]
        moving = np.where(seg > self.NAV_MOVING_TH)[0]
        if not moving.size:
            return a
        nav_end = min(a + int(moving[-1]) + 1, b)
        disp = float(
            np.linalg.norm(base_xy[min(nav_end, len(base_xy) - 1)] - base_xy[a])
        )
        return nav_end if disp >= self.MIN_NAV_DIST else a

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        pot = np.array([h["pot"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knobs = np.array([h["knobs"] for h in history])  # (T, n_knob)

        base_xy = (
            np.array([h["base_xy"] for h in history])
            if self.base_qpos_adrs
            else None
        )
        if base_xy is not None and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # pick -> carry: the pot lifts off the counter
        rest_z = float(np.median(pot[: self.REST_W, 2]))
        lifted = pot[:, 2] > rest_z + self.LIFT_TOL
        t_lift = self._sustained_start(lifted, 0, self.LIFT_SUSTAIN)
        if t_lift is None:
            lf = np.where(lifted)[0]
            t_lift = int(lf[0]) if lf.size else 0

        # place ends when the gripper RELEASES the pot (lets go and moves off to the
        # knob), after the pot has ARRIVED near its final on-stove pose. The pot is
        # large / handle-grabbed, so "released" = the gripper->pot distance rising
        # above the carry (hold) level -- the pot reaches its spot well before the
        # robot lets go, so a first-settle test ended the place too early (place too
        # short / turn-on too early: demo_72, demo_93).
        final = np.median(pot[-self.REST_W:], axis=0)
        dist = np.linalg.norm(pot - final, axis=1)
        t_arrive = self._sustained_start(dist < self.ARRIVE_TOL, t_lift, self.ARRIVE_SUSTAIN)
        if t_arrive is None:
            t_arrive = t_lift
        gd = np.linalg.norm(grip - pot, axis=1)
        hold_gd = float(np.median(gd[t_lift: max(t_arrive, t_lift + 1)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_arrive, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # turn-on: the burner the pot ends on has its knob in the "on" range
        norm = np.mod(knobs, 2 * np.pi)  # [0, 2pi)
        on = (norm >= self.KNOB_ON) & (norm <= 2 * np.pi - self.KNOB_ON)  # (T, n)
        # the burner whose knob enters the "on" range -- if more than one ever
        # reads on, take the one turned LATEST (the deliberate turn-on). In
        # practice exactly one front burner is turned.
        candidates = []
        for j, loc in enumerate(self.knob_locs):
            w = np.where(on[:, j])[0]
            if w.size:
                candidates.append((int(w[0]), loc))
        if candidates:
            t_knob_on, burner_loc = max(candidates)
        else:
            t_knob_on, burner_loc = T - 1, None

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_lift, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.pot_lang} from the counter")
        nav_end = self._nav_end(base_speed, base_xy, cursor, int(min(max(t_placed, cursor), T)))
        if nav_end > cursor:
            emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.pot_lang} on a stove burner")
        loc_txt = burner_loc.replace("_", " ") if burner_loc else "stove"
        emit(t_knob_on, SUBTASK_TURNON, STAGE_EXECUTE, f"turn on the {loc_txt} burner")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if burner_loc is None:
            review.append("no burner detected as turned on -- check the knob")

        debug = {
            "t_lift": t_lift,
            "nav_end": nav_end,
            "t_placed": t_placed,
            "burner": burner_loc,
            "t_knob_on": t_knob_on,
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
        return result

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

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 pot pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 pot place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_TURNON) != 1:
            issues.append(f"expected 1 TurnOnStove, found {count(SUBTASK_TURNON)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
