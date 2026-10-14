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


# Clear the sink area for faucet cleaning: carry the cup from the counter to the
# (open) cabinet, and carry the bowl (holding a food item) to the (open) fridge. Each
# move is nav-interrupted and the two go to DIFFERENT destinations. Cup->cabinet is
# PickPlaceCounterToCabinet (bank); the bowl->fridge move has no bank name so it is an
# extra_subtask. NavigateKitchen is bank.
SUBTASK_CUP = "PickPlaceCounterToCabinet"    # cup: counter -> cabinet
SUBTASK_BOWL = "PickPlaceCounterToFridge"    # bowl (with food): counter -> fridge
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

CUP = "cup"
BOWL = "food_item_container"  # the bowl that holds the food item


@register_labeler
class PrepSinkForCleaningLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepSinkForCleaning composite task.

    Task: a cup and a bowl (holding a food item) sit on the counter next to the sink
    (where the robot starts); a cabinet and a fridge are ALREADY OPEN across the
    kitchen. Carry the cup to the cabinet and the bowl to the fridge to clear the
    sink area for faucet cleaning.

    Doc-canonical count = 7 (`mobile: Yes`): each object is a nav-interrupted move,
    and between the two the robot drives BACK to the sink for the second object.
    Full decomposition:
        obj_a[pick] -> nav(carry to dest) -> obj_a[place]
        -> nav(back to sink) -> obj_b[pick] -> nav(carry to dest) -> obj_b[place]
    = 7 subtasks + done (expected_done_idx = 7). The robot starts at the sink, so the
    FIRST object has no leading nav. Each nav is ADAPTIVE (gated on MIN_NAV_DIST). The
    ORDER varies (resolved from which moves first); each object's DESTINATION is fixed
    by identity (cup -> cabinet, bowl -> fridge).

    Boundaries (SweetenCoffee / ArrangeCuttingFruits base-stop convention): per object
    a leading back-to-sink nav (base drive before the grasp), t_pick (first sustained
    displacement from rest), a carry nav (base drive while held, to the base stop),
    and the place ending at the gripper RELEASE.
    """

    ENV_NAME = "PrepSinkForCleaning"

    OBJS = (CUP, BOWL)
    # per object: (subtask name, destination noun, object noun)
    DEST = {
        CUP: (SUBTASK_CUP, "cabinet"),
        BOWL: (SUBTASK_BOWL, "fridge"),
    }

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {}
        for o in self.OBJS:
            try:
                self.obj_body_ids[o] = env.obj_body_id[o]
            except Exception:
                # bowl may be registered only under its container name; fall back to
                # the food item it holds as a proxy for the bowl's motion
                self.obj_body_ids[o] = env.obj_body_id["food_item"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cup_lang = env.get_obj_lang(CUP)
        except Exception:
            self.cup_lang = "cup"
        self.obj_noun = {CUP: self.cup_lang, BOWL: "bowl"}

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_BOWL}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.OBJS:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.OBJS}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.OBJS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.OBJS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.OBJS}

        first_move = {}
        for o in self.OBJS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(self.OBJS, key=lambda o: first_move[o])

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
            name, dest = self.DEST[o]
            noun = self.obj_noun[o]

            t_pick = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav BACK to the sink -- only for the 2nd object (the return
            # drive after dropping the first at its distant destination). The first
            # object's leading drive is just the robot's initial positioning at the
            # sink (it starts there); that is not a task nav, so it is absorbed into
            # the first pick rather than emitted as its own subtask.
            if k >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_at_sink = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_at_sink > cursor and base_disp(cursor, t_at_sink) >= self.MIN_NAV_DIST:
                    emit(t_at_sink, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the sink")

            emit(t_pick, name, STAGE_PICK, f"pick up the {noun} from the counter")

            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {noun} to the {dest}")
            emit(t_placed, name, STAGE_PLACE, f"place the {noun} in the {dest}")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": order,
            "first_move": {o: int(first_move[o]) for o in self.OBJS},
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

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

        for nm, label in ((SUBTASK_CUP, "cup"), (SUBTASK_BOWL, "bowl")):
            if count(nm, STAGE_PICK) != 1:
                issues.append(f"expected 1 {label} pick, found {count(nm, STAGE_PICK)}")
            if count(nm, STAGE_PLACE) != 1:
                issues.append(f"expected 1 {label} place, found {count(nm, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 3):
            issues.append(f"expected 0-3 NavigateKitchen, found {n_nav}")
        return issues
