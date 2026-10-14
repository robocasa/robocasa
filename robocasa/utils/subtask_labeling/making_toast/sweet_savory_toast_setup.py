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

# The avocado and bread on the counter go onto the plate, the jam comes out of the (open,
# double-door) cabinet next to the plate, and then the cabinet is closed. Counter ->
# counter and cabinet -> counter have no bank names -> PickPlaceCounterToCounter /
# PickPlaceCabinetToCounter (extensions); NavigateKitchen and CloseCabinet are bank. The
# robot starts at the counter; the cabinet is adjacent in some layouts (no nav) and metres
# away in others (a real navigation before/after the jam).
SUBTASK_COUNTER = "PickPlaceCounterToCounter"
SUBTASK_CABINET = "PickPlaceCabinetToCounter"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"

# object key -> (subtask, natural word, source phrase, destination phrase)
ITEMS = {
    "avocado": (SUBTASK_COUNTER, "avocado", "the counter", "on the plate"),
    "bread": (SUBTASK_COUNTER, "bread", "the counter", "on the plate"),
    "jam": (SUBTASK_CABINET, "jam", "the cabinet", "next to the plate"),
}


@register_labeler
class SweetSavoryToastSetupLabeler(BaseSubtaskLabeler):
    """Labeler for the SweetSavoryToastSetup composite task.

    Task (mobile, but no real nav): put the avocado and bread from the counter onto the
    plate, take the jam from the (open, double-door) cabinet and set it next to the
    plate, then close the cabinet. The robot starts at the counter.

    Doc-canonical decomposition (4 high-level: avocado + bread + jam moves + close).
    Emitted per item in grasp order (avocado/bread order varies; jam is always last,
    just before the close), then the close:
      pick -> place (avocado/bread on the plate, jam next to the plate)
      CloseCabinet [execute] -> close the cabinet doors
    plus a terminal done.

    Per item: t_pick = first sustained displacement from a rest re-anchored at the segment
    start; t_place = the gripper RELEASE. CloseCabinet ends when the LAST door that
    actually transitions from open to shut reaches closed; it pluralizes by that count
    (usually "doors" for the double-door cabinet).
    """

    ENV_NAME = "SweetSavoryToastSetup"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    OPEN_MIN = 0.30
    CLOSE_ABS = 0.20
    CLOSE_SUSTAIN = 5
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.35   # the cabinet is adjacent in some layouts (no nav) and metres
                          # away in others (demo_20 drives ~0.65 m) -> adaptive nav

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = list(ITEMS.keys())
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        cab = env.cab
        joint_names = list(getattr(cab, "door_joint_names", []) or [])
        if not joint_names:
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
        self.extra_subtasks = {SUBTASK_COUNTER, SUBTASK_CABINET}
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": doors,
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
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.names}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.names}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        first_move = {}
        for o in self.names:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
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
            subtask, word, src, dst = ITEMS[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # Leading approach-nav to a far source (the jam's cabinet is metres away in
            # some layouts). Suppressed for the first object (the robot starts by it).
            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_go = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                if t_go > cursor and base_disp(cursor, t_go) >= self.MIN_NAV_DIST:
                    # the cabinet is open the whole task (opened at scene setup), so name it so
                    nav_dest = "the open cabinet" if src == "the cabinet" else src
                    emit(t_go, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate to {nav_dest}")

            emit(t_pick, subtask, STAGE_PICK, f"pick up the {word} from {src}")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            # Carry-nav back to the plate on the counter (only when the base drives).
            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the counter with the plate")
            emit(t_placed, subtask, STAGE_PLACE, f"place the {word} {dst}")
            n_moved += 1

        t_close, n_close = self._close_cabinet(doorv, cursor, T)
        review = []
        if t_close is not None and t_close > cursor:
            door_word = "doors" if n_close >= 2 else "door"
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the cabinet {door_word}")
        else:
            review.append("cabinet never detected closing")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_close": n_close,
            "closed": int(t_close is not None), "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        if result.debug.get("closed") == 1 and int(np.sum(names == SUBTASK_CLOSE)) != 1:
            issues.append("expected 1 CloseCabinet")
        return issues
