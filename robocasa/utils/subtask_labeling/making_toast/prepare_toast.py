import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# From the (already-open) cabinet, the bread goes onto the cutting board and the jam onto
# the counter; then the cabinet is closed. Cabinet -> counter has no bank name ->
# PickPlaceCabinetToCounter (extension); CloseCabinet is bank. Fixed base (no nav).
SUBTASK_MOVE = "PickPlaceCabinetToCounter"
SUBTASK_CLOSE = "CloseCabinet"

DONE_LANG = "task complete"

# object key -> (natural word, destination phrase)
ITEMS = {
    "obj": ("bread", "on the cutting board"),
    "obj2": ("jam", "on the counter"),
}


@register_labeler
class PrepareToastLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareToast composite task.

    Task (fixed-base): from the (open) cabinet, put the bread on the cutting board and
    the jam on the counter, then close the cabinet. A knife on the counter is a
    distractor. The base never drives.

    Doc-canonical decomposition (3 high-level: bread move + jam move + close). Emitted
    per item in grasp order (ORDER varies -- often jam first), then the close:
      PickPlaceCabinetToCounter [pick]  -> pick up the {bread|jam} from the cabinet
      PickPlaceCabinetToCounter [place] -> place the bread on the cutting board / jam on the counter
      CloseCabinet [execute]            -> close the cabinet door(s)
    plus a terminal done.

    Per item: t_pick = first sustained displacement from a rest re-anchored at the segment
    start; t_place = the gripper RELEASE. CloseCabinet ends when the LAST door that
    actually transitions from open to shut reaches closed (doors that stay open belong to
    a neighbouring cabinet and are ignored); it pluralizes by how many doors close.
    """

    ENV_NAME = "PrepareToast"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    OPEN_MIN = 0.30    # a door must have been at least this open to count as "closed by us"
    CLOSE_ABS = 0.20   # door fraction below this == shut
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = list(ITEMS.keys())
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
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
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 12

    def per_step(self, env, t: int) -> dict:
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy(), "doors": doors}
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
        """(end_frame, n_doors_closed) for the doors that transition open -> shut."""
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

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in self.names}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in self.names}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        first_move = {}
        for o in self.names:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            word, dst = ITEMS[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the cabinet")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} {dst}")
            n_moved += 1

        t_close, n_close = self._close_cabinet(doorv, cursor, T)
        review = []
        if t_close is not None and t_close > cursor:
            door_word = "doors" if n_close >= 2 else "door"
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the cabinet {door_word}")
        elif len(self.door_adrs) > 0:
            # a cabinet WITH a door that never closed -- worth a look
            review.append("cabinet has a door but it never closed -- verify")
        # else: doorless open-shelf cabinet in this layout -> no CloseCabinet (legitimate)

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved")
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
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        if result.debug.get("closed") == 1 and int(np.sum(names == SUBTASK_CLOSE)) != 1:
            issues.append("expected 1 CloseCabinet")
        return issues
