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


# Start a simmering sauce: set the pan on the chosen burner, put the tomato and onion in the pan,
# and turn the burner on. Each move is the bank atomic PickPlaceCounterToStove; TurnOnStove is a
# bank atomic; NavigateKitchen is bank. Everything is by the stove where the robot starts, so navs
# are adaptive.
SUBTASK_MOVE = "PickPlaceCounterToStove"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_KNOB = "TurnOnStove"

DONE_LANG = "task complete"

MOVERS = ("pan", "tomato", "onion")


@register_labeler
class SimmeringSauceLabeler(BaseSubtaskLabeler):
    """Labeler for the SimmeringSauce composite task.

    Task (mobile, usually local): a pan, a tomato, and an onion sit on the counter beside the
    stove. Put the pan on the chosen burner, add the tomato and onion to it, and turn the burner
    on. The robot starts at the stove.

    Doc-canonical count = 4 (`mobile: Yes`): three moves + TurnOnStove. Emitted in first-motion
    order (pan first):
      PickPlaceCounterToStove [pick]  -> pick up the {pan|tomato|onion} from the counter
      (NavigateKitchen [navigate])    -> carry the {item} to the stove  (adaptive)
      PickPlaceCounterToStove [place] -> place the pan on the {burner} burner / the {veg} in the pan
      ... (repeat) ... -> TurnOnStove -> done.

    Boundaries: t_pick = first sustained displacement from a rest re-anchored at the segment start;
    place = the gripper RELEASE; TurnOnStove = the chosen burner's knob rotating on (|knob| >=
    KNOB_ON), read from the knob joint qpos.
    """

    ENV_NAME = "SimmeringSauce"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    KNOB_ON = 0.35
    KNOB_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in MOVERS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        knob = ep_meta.get("refs", {}).get("knob")
        self.burner_name = knob.replace("_", " ") if knob else ""
        # Candidate burner knobs: the named one when present, else all burners
        # (a few datasets store refs=None) -- the active burner is then detected
        # from the knob that turns on.
        try:
            burner_locs = [loc for loc in env.stove.knob_joints
                           if env.stove.knob_joints[loc] is not None]
        except Exception:
            burner_locs = []
        cand = [knob] if knob is not None else burner_locs
        self.knob_adrs = {}
        for loc in cand:
            try:
                self.knob_adrs[loc] = env.sim.model.get_joint_qpos_addr(
                    f"{env.stove.naming_prefix}knob_{loc}_joint")
            except Exception:
                pass
        self.finger_adrs = []
        for nm in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2"):
            try:
                jid = env.sim.model.joint_name2id(nm)
                self.finger_adrs.append(env.sim.model.jnt_qposadr[jid])
            except Exception:
                pass
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
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "knobs": {loc: float(env.sim.data.qpos[a]) for loc, a in self.knob_adrs.items()},
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_adrs)),
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
        fw = np.array([h["fw"] for h in history]) if self.finger_adrs else np.zeros(T)
        # Fingers reopen to (near) fully-open on release. The grasped width is
        # object-dependent (the pan closes far more than a tomato), but the
        # open width is not, so key the release off the open level.
        fw_open = 0.90 * float(np.percentile(fw, 95)) if fw.size else 0.0
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # Global first-move gives unambiguous, well-separated pick times (the
        # objects are moved one at a time). Using these directly -- rather than
        # re-anchoring per segment -- avoids pick boundaries collapsing when a
        # place boundary lands late.
        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        # Which burner turns on (sim's normalized definition: 0.35 <= v <= 2pi-0.35).
        active_loc, on_masks = None, {}
        for loc in self.knob_adrs:
            vals = np.array([h["knobs"].get(loc, 0.0) for h in history])
            norm = np.mod(vals, 2 * np.pi)
            on = (norm >= self.KNOB_ON) & (norm <= 2 * np.pi - self.KNOB_ON)
            on_masks[loc] = on
            if on.any() and active_loc is None:
                active_loc = loc
        burner_name = self.burner_name or (active_loc.replace("_", " ") if active_loc else "")
        burner = f"the {burner_name} burner" if burner_name else "the burner"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for i, o in enumerate(order):
            t_pick = max(first_move[o], cursor)
            nxt = first_move[order[i + 1]] if i + 1 < len(order) else T
            # Release = fingers reopen. gd-to-object-center is unreliable here
            # (the pan is grasped by its handle, so the center stays ~0.24 away
            # even after release, and the gripper keeps returning over the pan to
            # drop the vegetables in). Bound the place at the next object's pick.
            t_placed = self._sustained_start(fw >= fw_open, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None or t_placed > nxt:
                t_placed = nxt
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {o} from the counter")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                        if base_xy is not None and t_arrive > t_pick else 0.0)
            if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {o} to the stove")
            dest = f"on {burner}" if o == "pan" else "in the pan"
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {o} {dest}")
            n_moved += 1

        review = []
        has_knob = False
        if cursor < T and active_loc is not None:
            t_on = self._sustained_start(on_masks[active_loc], cursor, self.KNOB_SUSTAIN)
            if t_on is not None and t_on > cursor:
                emit(t_on, SUBTASK_KNOB, STAGE_EXECUTE, f"turn on {burner}")
                has_knob = True
        if not has_knob:
            review.append("burner never turned on -- verify")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "has_knob": has_knob,
            "fw_open": round(fw_open, 3), "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 places, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_KNOB)) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnStove, found {int(np.sum(names == SUBTASK_KNOB))}")
        return issues
