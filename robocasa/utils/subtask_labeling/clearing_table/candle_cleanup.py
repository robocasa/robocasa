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


# Each decoration is carried from the dining table to the (already-open) cabinet, then
# the cabinet is closed. Dining-table->cabinet is a bank atomic
# (PickPlaceCounterToCabinet -- the dining table is a counter); NavigateKitchen and
# CloseCabinet are bank atomics.
SUBTASK_MOVE = "PickPlaceCounterToCabinet"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"

OBJS = ("obj1", "obj2")


@register_labeler
class CandleCleanupLabeler(BaseSubtaskLabeler):
    """Labeler for the CandleCleanup composite task.

    Task: two decorations sit on the dining table (robot starts there); a cabinet is
    ALREADY OPEN across the kitchen. Carry each decoration to the cabinet and then
    close it.

    Doc-canonical count = 8 (`mobile: Yes`): each decoration is a nav-interrupted
    dining->cabinet move, the robot drives BACK for the second, and finally closes
    the cabinet: `deco_a[pick] → nav(carry to cabinet) → [place] → nav(back to table)
    → deco_b[pick] → nav(carry to cabinet) → [place] → CloseCabinet` = 8 subtasks +
    done. The FIRST decoration has no leading nav (robot starts at the table). Navs
    are ADAPTIVE; ORDER varies (by first move).
    """

    ENV_NAME = "CandleCleanup"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20
    CLOSE_ABS = 0.30
    CLOSE_SUSTAIN = 5
    OPEN_MIN = 0.25

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in OBJS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in OBJS:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "decoration"

        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
            pref = cab.name
            joint_names = [
                n for n in env.sim.model.joint_names
                if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())
            ]
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

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": np.array(
                [abs(float(env.sim.data.qpos[a])) / s
                 for a, s in zip(self.door_adrs, self.door_scales)]
            ),
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
        doorv = np.array([h["doors"] for h in history]) if self.door_adrs else np.zeros((T, 0))
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
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the cabinet")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {noun} in the cabinet")

        # CloseCabinet after both decorations are placed
        door_peak = float(doorv.max()) if doorv.size else 0.0
        has_close = False
        if door_peak >= self.OPEN_MIN and cursor < T:
            dmax = doorv.max(axis=1)
            t_closed = self._sustained_start(dmax < self.CLOSE_ABS, cursor, self.CLOSE_SUSTAIN)
            if t_closed is not None and t_closed > cursor:
                emit(t_closed, SUBTASK_CLOSE, STAGE_EXECUTE, "close the cabinet")
                has_close = True

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_close:
            review.append("no CloseCabinet detected (never closed) -- verify")

        result = self.build_result(T, segments, debug={
            "order": order, "door_peak": round(door_peak, 2),
            "has_close": has_close, "n_segments": len(segments),
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
            issues.append(f"expected 2 deco picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 2:
            issues.append(f"expected 2 deco places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_CLOSE) not in (0, 1):
            issues.append(f"expected 0-1 CloseCabinet, found {count(SUBTASK_CLOSE)}")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 4):
            issues.append(f"expected 0-4 NavigateKitchen, found {n_nav}")
        return issues
