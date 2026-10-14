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

# Two items are moved between the top two freezer racks (an in-freezer shuffle), then the
# freezer door is closed. Rack -> rack has no bank name -> PickPlaceFreezerRackToRack
# (extension); CloseFridge is bank. No navigation. Robot starts at the fridge.
SUBTASK_MOVE = "PickPlaceFreezerRackToRack"
SUBTASK_CLOSE = "CloseFridge"

DONE_LANG = "task complete"

MOVED = ("item1", "item2")   # the two items that are relocated (item3/item4 stay)


@register_labeler
class MaximizeFreezerSpaceLabeler(BaseSubtaskLabeler):
    """Labeler for the MaximizeFreezerSpace composite task.

    Task (mobile, no real nav): two items sit on one of the top two freezer racks; move
    them onto the other rack (where two other items already are) to pack the freezer,
    then close the freezer door. Direction ("up"/"down") is in ep_meta refs. Robot starts
    at the fridge.

    Doc-canonical decomposition (3 high-level: item1 + item2 moves + close). Emitted per
    item in grasp order (ORDER varies), then the close:
      PickPlaceFreezerRackToRack [pick]  -> pick up the {item} from the {source rack}
      PickPlaceFreezerRackToRack [place] -> move the {item} to the {target rack}
      CloseFridge [execute]              -> close the freezer door
    plus a terminal done.

    Per item: t_pick = first sustained displacement from a rest re-anchored at the segment
    start; t_place = the gripper RELEASE. The shuffle drives no base motion (no nav). The
    close ends when the freezer door swings back below CLOSE_TH. Rack names from refs.
    """

    ENV_NAME = "MaximizeFreezerSpace"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.07
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    CLOSE_TH = 0.20
    CLOSE_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in MOVED}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o in MOVED:
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = "item"
        direction = ep_meta.get("refs", {}).get("move_direction", "up")
        if direction == "up":
            self.src_rack, self.dst_rack = "second highest rack", "highest rack"
        else:
            self.src_rack, self.dst_rack = "highest rack", "second highest rack"
        self.same = self.obj_lang[MOVED[0]] == self.obj_lang[MOVED[1]]
        fr = env.fridge
        joint_names = list(getattr(fr, "_freezer_door_joint_names", []) or [])
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
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = np.array([abs(float(env.sim.data.qpos[a])) / s
                          for a, s in zip(self.door_adrs, self.door_scales)])
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy(), "doors": doors}
        for o in MOVED:
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
        opos = {o: np.array([h[o] for h in history]) for o in MOVED}
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))

        rest = {o: np.median(opos[o][: self.REST_W], axis=0) for o in MOVED}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in MOVED}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in MOVED}

        first_move = {}
        for o in MOVED:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in MOVED if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for o in order:
            word = self.obj_lang[o]
            noun = ("the other " + word if n_moved == 1 else "a " + word) if self.same else "the " + word
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {noun} from the {self.src_rack}")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"move {noun} to the {self.dst_rack}")
            n_moved += 1

        # CloseFridge: the freezer door swings shut after the last item is placed.
        close_frames = []
        for i in range(doorv.shape[1]):
            if float(doorv[:, i].max()) < 0.5:
                continue
            fr = self._sustained_start(doorv[:, i] < self.CLOSE_TH, cursor, self.CLOSE_SUSTAIN)
            if fr is not None:
                close_frames.append(fr)
        n_doors = int(sum(1 for i in range(doorv.shape[1]) if float(doorv[:, i].max()) >= 0.5))
        review = []
        if close_frames:
            t_close = max(max(close_frames), cursor + 1)
            door_word = "doors" if n_doors >= 2 else "door"
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, f"close the freezer {door_word}")
        elif doorv.shape[1] and float(doorv.max()) >= 0.5:
            review.append("freezer door never closed -- verify")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_close": len(close_frames),
            "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 pick stages, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 place stages, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
