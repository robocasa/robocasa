import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Clean up after slicing: move the knife and the cutting board from the counter into the
# sink, and bring the spray down from the (open) cabinet onto the counter. Counter -> sink
# has no bank name -> PickPlaceCounterToSink (extension); cabinet -> counter is the bank
# atomic PickPlaceCabinetToCounter. NavigateKitchen is bank. Robot starts at the counter.
MOVE_SINK = "PickPlaceCounterToSink"
MOVE_CAB = "PickPlaceCabinetToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# object -> (move name, pick, carry, place, nav-back). The cabinet is pre-opened in the
# scene, so there is no OpenCabinet subtask.
SPEC = {
    "knife": dict(
        move=MOVE_SINK,
        pick="pick up the knife from the counter",
        carry="carry the knife to the sink",
        place="place the knife in the sink",
        back="navigate back to the counter with the knife",
    ),
    "receptacle": dict(
        move=MOVE_SINK,
        pick="pick up the cutting board from the counter",
        carry="carry the cutting board to the sink",
        place="place the cutting board in the sink",
        back="navigate back to the counter with the cutting board",
    ),
    "spray": dict(
        move=MOVE_CAB,
        pick="pick up the spray from the cabinet",
        carry="carry the spray to the counter",
        place="place the spray on the counter",
        back="navigate to the open cabinet",
    ),
}


@register_labeler
class CleanBoardLabeler(BaseSubtaskLabeler):
    """Labeler for the CleanBoard composite task.

    Task (mobile): after slicing meat, move the knife and the cutting board from the
    counter into the sink, and bring the spray down from the OPEN cabinet above the counter
    onto the counter. Robot starts at the counter.

    Doc-canonical count = 7 (`mobile: Yes`), action-level: three moves (knife->sink,
    board->sink, spray cabinet->counter), each pick/place with adaptive navs between; the
    first-moved object has no leading nav. ORDER by first move; each object carries its own
    source/dest lang.

    Boundaries: per object t_pick = first sustained displacement re-anchored at the segment
    start; navs split off the base motion (MIN_NAV_DIST); t_place = the gripper RELEASE.
    """

    ENV_NAME = "CleanBoard"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    LIFT_TOL = 0.08      # z rise above rest that marks a real lift off the surface (not a
                         # small bump/slide -- the board is genuinely raised ~0.2-0.4m to
                         # clear the sink edge, so 0.08 ignores incidental nudges)
    LIFT_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    MIN_PICK_GAP = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in SPEC if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {MOVE_SINK}  # PickPlaceCabinetToCounter is a bank atomic
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
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
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            spec = SPEC[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_disp = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_disp is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_disp = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            # prefer the LIFT onset (the object leaving its surface = the real carry) over an
            # early horizontal move: the cutting board is SLID along the counter to grab it
            # before being lifted, which would otherwise trigger a spurious pick on the slide.
            rest_z = float(np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1, 2]))
            t_lift = self._sustained_start(opos[o][:, 2] > rest_z + self.LIFT_TOL, cursor, self.LIFT_SUSTAIN)
            t_pick = t_lift if t_lift is not None else t_disp
            t_pick = max(t_pick, cursor)

            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                t_back = min(t_back, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, spec["back"])

            emit(t_pick, spec["move"], STAGE_PICK, spec["pick"])

            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, spec["carry"])
            emit(t_placed, spec["move"], STAGE_PLACE, spec["place"])
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != len(self.names):
            review.append(f"only {n_moved}/{len(self.names)} objects moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {MOVE_SINK}
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
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 5):
            issues.append(f"expected 0-5 NavigateKitchen, found {n_nav}")
        return issues
