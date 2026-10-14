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

# Open the (closed) cabinet, take the strainer out, and stand it upright in the bowl on
# the counter. OpenCabinet is a bank atomic; cabinet -> counter/bowl has no bank name ->
# PickPlaceCabinetToCounter (extension). The robot starts at the cabinet; the strainer
# move is local (no navigation).
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToCounter"

DONE_LANG = "task complete"


@register_labeler
class StrainerSetupLabeler(BaseSubtaskLabeler):
    """Labeler for the StrainerSetup composite task.

    Task (mobile, but no real nav): the strainer sits in a CLOSED cabinet; a bowl sits
    on the counter by it. Open the cabinet, then place the strainer upright in the bowl.
    The robot starts at the cabinet.

    Doc-canonical decomposition (2 high-level: open, strainer move). Emitted:
      OpenCabinet [execute]              -> open the cabinet door(s)
      PickPlaceCabinetToCounter [pick]   -> pick up the strainer from the cabinet
      PickPlaceCabinetToCounter [place]  -> place the strainer upright in the bowl
    plus a terminal done.

    Detection: the cabinet door(s) swing from shut toward open; OpenCabinet ends when the
    LAST opened leaf reaches its (adaptive) open threshold -- a double-door cabinet where
    the second leaf only opens partway needs min(OPEN_ABS, 0.9*peak). The strainer pick
    begins at its first sustained departure; t_place = the gripper RELEASE.
    """

    ENV_NAME = "StrainerSetup"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    OPEN_ABS = 0.85
    OPEN_MIN = 0.25    # a door whose peak stays below this never really opened
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.strainer_body_id = env.obj_body_id["strainer"]
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
        return {
            "strainer": env.sim.data.body_xpos[self.strainer_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": doors,
        }

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
        strainer = np.array([h["strainer"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))

        rest = np.median(strainer[: self.REST_W], axis=0)
        disp = np.linalg.norm(strainer - rest, axis=1)
        gd = np.linalg.norm(grip - strainer, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_place = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        # OpenCabinet: end when the LAST opened leaf reaches its adaptive open level.
        open_frames, n_open = [], 0
        for i in range(doorv.shape[1]):
            col = doorv[:, i]
            peak = float(col.max())
            if peak < self.OPEN_MIN:
                continue
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            fr = self._sustained_start(col >= lvl, 0, self.OPEN_SUSTAIN)
            if fr is not None:
                open_frames.append(fr)
                n_open += 1
        t_open = max(open_frames) if open_frames else 0
        t_open = int(min(t_open, max(t_pick - 1, 0)))
        door_word = "doors" if n_open >= 2 else "door"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if n_open > 0 and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the cabinet {door_word}")
        elif len(self.door_adrs) > 0:
            # a cabinet WITH a door that we never saw open -- worth a look
            review.append("cabinet has a door but it never opened -- verify")
        # else: doorless open-shelf cabinet in this layout -> no OpenCabinet (legitimate)
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, "pick up the strainer from the cabinet")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, "place the strainer upright in the bowl")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": int(t_open), "n_open": n_open, "t_pick": int(t_pick),
            "t_place": int(t_place), "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 pick stage")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 place stage")
        if result.debug.get("n_open", 0) >= 1 and int(np.sum(names == SUBTASK_OPEN)) != 1:
            issues.append("expected 1 OpenCabinet")
        return issues
