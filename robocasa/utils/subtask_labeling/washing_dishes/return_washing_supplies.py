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
from robocasa.utils.subtask_labeling.washing_dishes._fixtures import nearest_landmark


# The REVERSE of CollectWashingSupplies: two cleaning supplies sit in the sink; carry both
# back to the (open) cabinet and close its doors. Sink -> cabinet has no bank name ->
# PickPlaceSinkToCabinet (extension); NavigateKitchen + CloseCabinet are bank. Mobile.
SUBTASK_MOVE = "PickPlaceSinkToCabinet"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"

OBJS = ("supply1", "supply2")


@register_labeler
class ReturnWashingSuppliesLabeler(BaseSubtaskLabeler):
    """Labeler for the ReturnWashingSupplies composite task.

    Task (mobile): two cleaning supplies are in the sink; return both to the open cabinet and
    close the cabinet door(s). Doc-canonical count = 8 (action-level: two sink->cabinet moves,
    each pick + carry nav + place, a return-to-sink nav before the second, plus the close).
    Order by first grasped move; same-type supplies read "a X"/"the other X".

    pick = first displacement while grasped (cursor-relative rest); carry nav off base motion
    (ends at arrival); place = gripper RELEASE; CloseCabinet from the door-close onset.
    """

    ENV_NAME = "ReturnWashingSupplies"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.07
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    OPEN_MIN = 0.30
    CLOSE_ABS = 0.20
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "cleaner"
        # cabinet doors (like EmptyDishRack)
        cab = getattr(env, "cabinet", None)
        try:
            self.cab_landmark = nearest_landmark(env, cab.pos) if cab is not None else None
        except Exception:
            self.cab_landmark = None
        joint_names = list(getattr(cab, "door_joint_names", []) or []) if cab is not None else []
        if not joint_names and cab is not None:
            pref = cab.name
            joint_names = [n for n in env.sim.model.joint_names
                           if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": np.array([abs(float(env.sim.data.qpos[a])) / s
                               for a, s in zip(self.door_adrs, self.door_scales)]),
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

    @staticmethod
    def _last_sustained_end(mask, sustain):
        last_end, run = None, 0
        for i, v in enumerate(mask):
            if v:
                run += 1
            else:
                if run >= sustain:
                    last_end = i
                run = 0
        if run >= sustain:
            last_end = len(mask)
        return last_end

    def _close_cabinet(self, doorv, cursor, T):
        frames, n = [], 0
        for i in range(doorv.shape[1]):
            col = doorv[:, i]
            started_open = float(col[: max(cursor, 1)].max()) >= self.OPEN_MIN
            ends_closed = float(np.median(col[-self.REST_W:])) < self.CLOSE_ABS
            if not (started_open and ends_closed):
                continue
            fr = self._sustained_start(col < self.CLOSE_ABS, cursor, self.CLOSE_SUSTAIN)
            if fr is not None:
                frames.append(fr)
                n += 1
        return (max(frames) if frames else None), n

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), 0, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        from collections import Counter
        counts = Counter(self.base[o] for o in order)
        seen, noun = {}, {}
        for o in order:
            w = self.base[o]
            if counts[w] <= 1:
                noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for k, o in enumerate(order):
            nn = noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # return-to-sink nav before a later supply
            if k >= 1:
                end = self._last_sustained_end(binst[cursor: t_pick] > self.BASE_MOVE_TH, self.MOVE_SUSTAIN)
                t_appr = cursor + end if end is not None else cursor
                t_appr = min(t_appr, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_appr > cursor and base_disp(cursor, t_appr) >= self.MIN_NAV_DIST:
                    emit(t_appr, SUBTASK_NAV, STAGE_NAVIGATE, "navigate to the sink")
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the sink")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                by = f" by the {self.cab_landmark}" if self.cab_landmark else ""
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate to the open cabinet{by}")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} in the cabinet")
            n_moved += 1

        review = []
        t_close, n_close = self._close_cabinet(doorv, cursor, T)
        if t_close is not None and t_close > cursor:
            door_word = "doors" if n_close >= 2 else "door"
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the cabinet {door_word}")
        elif len(self.door_adrs) > 0:
            review.append("cabinet has a door but it never closed -- verify")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 supplies moved")
        result = self.build_result(T, segments, debug={"order": order, "n_moved": n_moved,
                                                       "n_close": n_close, "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK))) != 2:
            issues.append("expected 2 supply picks")
        if int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE))) != 2:
            issues.append("expected 2 supply places")
        if int(np.sum(names == SUBTASK_CLOSE)) not in (0, 1):
            issues.append("expected 0-1 CloseCabinet")
        return issues
