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


# Each receptacle is carried from the dining table to the sink, then the water is
# turned on. Dining-table->sink is a bank atomic (PickPlaceCounterToSink);
# NavigateKitchen and TurnOnSinkFaucet are bank atomics.
SUBTASK_MOVE = "PickPlaceCounterToSink"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_FAUCET = "TurnOnSinkFaucet"

DONE_LANG = "task complete"

OBJS = ("receptacle1", "receptacle2")


@register_labeler
class ClearReceptaclesForCleaningLabeler(BaseSubtaskLabeler):
    """Labeler for the ClearReceptaclesForCleaning composite task.

    Task: two receptacles sit on the dining table (robot starts there); a sink is
    across the kitchen (water off). Carry each receptacle to the sink, then turn the
    water on.

    Doc-canonical count = 8 (`mobile: Yes`): each receptacle is a nav-interrupted
    dining->sink move, the robot drives BACK for the second, then turns on the water:
    `rec_a[pick] → nav(carry to sink) → [place] → nav(back to table) → rec_b[pick] →
    nav(carry to sink) → [place] → TurnOnSinkFaucet` = 8 subtasks + done. The FIRST
    receptacle has no leading nav (robot starts at the table). Navs are ADAPTIVE;
    ORDER varies (by first move).
    """

    ENV_NAME = "ClearReceptaclesForCleaning"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20
    FAUCET_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in OBJS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = env.sink
        self.obj_lang = {}
        for o in OBJS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "receptacle"

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
        try:
            water_on = bool(self.sink.get_handle_state(env)["water_on"])
        except Exception:
            water_on = False
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "water_on": water_on,
        }
        for o in OBJS:
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
        opos = {o: np.array([h[o] for h in history]) for o in OBJS}
        grip = np.array([h["grip"] for h in history])
        water = np.array([bool(h["water_on"]) for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in OBJS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in OBJS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in OBJS}

        first_move = {}
        for o in OBJS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(OBJS, key=lambda o: first_move[o])
        same_type = self.obj_lang[OBJS[0]] == self.obj_lang[OBJS[1]]

        def noun_of(k, o):
            word = self.obj_lang[o]
            if same_type:
                return f"the other {word}" if k == 1 else f"a {word}"
            return f"the {word}"

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

        for k, o in enumerate(order):
            noun = noun_of(k, o)
            t_pick = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            if k >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the dining table")

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the dining table")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} in the sink")

        # TurnOnSinkFaucet: the water turns on after both receptacles are placed
        has_faucet = False
        if cursor < T:
            t_on = self._sustained_start(water, cursor, self.FAUCET_SUSTAIN)
            if t_on is not None and t_on > cursor:
                emit(t_on, SUBTASK_FAUCET, STAGE_EXECUTE, "turn on the water")
                has_faucet = True

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_faucet:
            review.append("water never turned on -- verify")

        result = self.build_result(T, segments, debug={
            "order": order, "water_ever_on": bool(water.any()),
            "has_faucet": has_faucet, "n_segments": len(segments),
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

        if count(SUBTASK_MOVE, STAGE_PICK) != 2:
            issues.append(f"expected 2 receptacle picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 2:
            issues.append(f"expected 2 receptacle places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_FAUCET) not in (0, 1):
            issues.append(f"expected 0-1 TurnOnSinkFaucet, found {count(SUBTASK_FAUCET)}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 4):
            issues.append(f"expected 0-4 NavigateKitchen, found {n_nav}")
        return issues
