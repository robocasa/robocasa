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


# Sort two empty bottles from the dining counter by size: the large one goes next to the other
# large bottle on the counter beside the sink; the small one goes into the sink next to the
# other small bottle. Each is a nav-interrupted move (the sink is across the kitchen). Large ->
# counter has no bank name -> PickPlaceCounterToCounter (extension); small -> sink is the bank
# atomic PickPlaceCounterToSink; NavigateKitchen is a bank atomic.
SUBTASK_LARGE = "PickPlaceCounterToCounter"
SUBTASK_SMALL = "PickPlaceCounterToSink"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class RecycleBottlesBySizeLabeler(BaseSubtaskLabeler):
    """Labeler for the RecycleBottlesBySize composite task.

    Task (mobile): a large and a small empty bottle sit on the dining counter (where the robot
    starts); matching sample bottles already sit by the sink (large on the counter, small in the
    sink). Carry the large bottle next to the large sample (on the counter, outside the sink) and
    the small bottle into the sink next to the small sample.

    Doc-canonical count = 7 (`mobile: Yes`): the two bottles' nav-interrupted moves (the second
    also has a lead-in nav back to the dining counter). Structure (order resolved from motion):
      large[pick] -> NavigateKitchen(carry) -> large[place]
      -> NavigateKitchen(to dining) -> small[pick] -> NavigateKitchen(carry) -> small[place]
      -> done.
    The first bottle has no lead-in nav (the robot starts at the dining counter; k>=1 gate).

    Boundaries (MakeBananaMilkshake machinery): t_pick = first sustained displacement; nav OUT =
    base drive cursor->grasp (k>=1); nav BACK = base drive during the carry, ending at the last
    substantial-motion frame; place = the gripper RELEASE. The sample bottles never move.
    """

    ENV_NAME = "RecycleBottlesBySize"

    # object -> (word, dest phrase, subtask)
    ITEMS = {
        "bottle_large1": ("large bottle", "next to the other large bottle by the sink", SUBTASK_LARGE),
        "bottle_small1": ("small bottle", "in the sink", SUBTASK_SMALL),
    }

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.20
    MIN_PICK_GAP = 12    # a nav never runs to within this of the grasp/release it precedes, so
                         # the pick/place it bounds keeps a visible window (the base often drives
                         # right up to the object, which would otherwise collapse the pick).
    ARRIVE_RADIUS = 0.20 # base within this of its parked destination pose == in the vicinity
    ARRIVE_STOP_TH = 0.006  # ...and per-frame speed below this == settled. The nav ends when the
                         # base is BOTH near its park pose AND stopped, so it lands at the real
                         # settle (not while still decelerating into the zone, nor a brief settle
                         # before a final creep).
    ARRIVE_SUSTAIN = 12
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = list(self.ITEMS.keys())
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # include YAW as well as forward/side: the base often translates to the counter, then
        # rotates in place to face the object -- a pure-position arrival would end the nav during
        # that rotation. Yaw (rad) and translation (m) are comparable enough at these scales.
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_LARGE}
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

        def arrival(start, end, park):
            # first frame in [start, end) the base is BOTH within ARRIVE_RADIUS of `park` (its
            # parked destination pose) AND stopped (speed < ARRIVE_STOP_TH), sustained. So it ends
            # at the real settle -- ignoring the base creep during a long arm-reach that follows,
            # and surviving a brief mid-drive pause (still far from `park`), and not firing while
            # the base is still decelerating into the zone.
            if base_xy is None or end <= start:
                return None
            near = np.linalg.norm(base_xy[start: end] - park, axis=1) < self.ARRIVE_RADIUS
            stopped = binst[start: end] < self.ARRIVE_STOP_TH
            fr = self._sustained_start(near & stopped, 0, self.ARRIVE_SUSTAIN)
            if fr is not None:
                return start + fr
            # no clean settle (base drove right into the grab) -> last substantial-motion frame
            mv = np.where(binst[start: end] > self.BASE_MOVE_TH)[0]
            return start + int(mv[-1]) + 1 if mv.size else None

        n_moved = 0
        for k, o in enumerate(order):
            word, dst, sub = self.ITEMS[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            if k >= 1 and base_xy is not None:
                # nav back to the source: ends when the base arrives at where it parks to grab it.
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, t_move - self.MIN_PICK_GAP)  # keep the pick a visible window
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate back to the {word} on the dining counter")
            emit(t_move, sub, STAGE_PICK, f"pick up the {word} from the dining counter")
            # carry to the sink: ends when the base arrives at where it parks to set the bottle down.
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)  # keep the place a visible window
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the sink")
            emit(t_placed, sub, STAGE_PLACE, f"place the {word} {dst}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 2:
            review.append(f"only {n_moved}/2 bottles moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 3):
            issues.append(f"expected 0-3 NavigateKitchen, found {n_nav}")
        return issues
