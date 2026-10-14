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


# Open the fridge, then take the three perishables (a fruit, a vegetable, a piece of
# meat) out and set each on the counter. OpenFridge + NavigateKitchen are bank
# atomics; the fridge->counter move has no exact bank name so it is an extra_subtask.
SUBTASK_OPEN = "OpenFridge"
SUBTASK_MOVE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the fridge"
DONE_LANG = "task complete"

ITEMS = ("fruit", "vegetable", "meat")


@register_labeler
class PrepFridgeForCleaningLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepFridgeForCleaning composite task.

    Task: a fruit, a vegetable, and a piece of meat sit in a CLOSED fridge (the robot
    starts at the fridge). Open the fridge and set all three on the counter to clear
    the fridge for cleaning.

    Doc-canonical count = 4 (`mobile: Yes`): OpenFridge + one fridge->counter move per
    item. The items go on the adjacent counter within arm reach, so the per-item nav
    is ADAPTIVE (gated on MIN_NAV_DIST) and usually absent. OpenFridge + 3 items
    (pick/place stages each) + done -> expected_done_idx set per demo (7 with no navs).
    The item ORDER varies (resolved from which moves first).

    Boundaries:
      - t_open: the fridge door(s) reach their open plateau (OPEN_FRAC of each leaf's
        own peak; a double door is opened one leaf at a time). The items sit on
        shelves and do NOT ride the door, so their rest pose is their initial pose.
      - per item (first-move / relative-release split): t_move = the item first
        displaces MOVE_TOL from its in-fridge rest; t_placed = the gripper RELEASES it.
        An adaptive carry-nav is emitted between them if the base actually drove.
    """

    ENV_NAME = "PrepFridgeForCleaning"

    OPEN_FRAC = 0.90  # a leaf is open at this fraction of its own peak
    OPEN_SUSTAIN = 5
    OPEN_MIN = 0.15  # peak door fraction below this == the leaf never opened
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
        self.obj_body_ids = {o: env.obj_body_id[o] for o in ITEMS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in ITEMS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = o

        # fridge door hinge joints (a fridge has 1-2 door leaves); track each as a
        # fraction of its own fully-open range.
        fridge = env.fridge
        joint_names = list(getattr(fridge, "_fridge_door_joint_names", []) or [])
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_adrs.append(adr)
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = np.array(
            [abs(float(env.sim.data.qpos[a])) / s
             for a, s in zip(self.door_adrs, self.door_scales)]
        )
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": doors,
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in ITEMS:
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
        opos = {o: np.array([h[o] for h in history]) for o in ITEMS}
        grip = np.array([h["grip"] for h in history])
        doorv = np.array([h["doors"] for h in history]) if self.door_adrs else np.zeros((T, 0))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # open the fridge: the last leaf that opens reaches its own-peak plateau
        door_peak = float(doorv.max()) if doorv.size else 0.0
        has_open = door_peak >= self.OPEN_MIN
        if has_open and doorv.shape[1]:
            open_frames = []
            for i in range(doorv.shape[1]):
                peak_i = float(doorv[:, i].max())
                if peak_i < self.OPEN_MIN:
                    continue
                fr = self._sustained_start(doorv[:, i] >= self.OPEN_FRAC * peak_i, 0, self.OPEN_SUSTAIN)
                if fr is not None:
                    open_frames.append(fr)
            t_open = max(open_frames) if open_frames else 0
        else:
            t_open = 0

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in ITEMS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in ITEMS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in ITEMS}

        first_move = {}
        for o in ITEMS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, t_open, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(ITEMS, key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)

        n_moved = 0
        for o in order:
            if first_move[o] >= T:
                continue
            n_moved += 1
            t_move = self._sustained_start(disp[o] >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(disp[o][cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)

            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            word = self.obj_lang[o]
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the fridge")
            # adaptive carry-nav (usually absent -- the counter is by the fridge)
            carry_moving = np.where(binst[t_move: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_move + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_move
            if base_xy is not None and t_arrive > t_move:
                nav_disp = float(np.max(np.linalg.norm(base_xy[t_move: t_arrive] - base_xy[t_move], axis=1)))
            else:
                nav_disp = 0.0
            if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} on the counter")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_open:
            review.append(f"fridge door never opened (peak {door_peak:.2f}) -- verify")
        if n_moved != 3:
            review.append(f"only {n_moved}/3 items removed")

        debug = {
            "t_open": t_open, "door_peak": round(door_peak, 2),
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
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

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenFridge, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 3:
            issues.append(f"expected 3 item picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 3:
            issues.append(f"expected 3 item places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        return issues
