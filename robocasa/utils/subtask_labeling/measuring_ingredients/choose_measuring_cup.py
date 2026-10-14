import re

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

# Open the (closed) drawer and take out the requested measuring cup (the larger or the
# smaller of the two) onto the counter. OpenDrawer and PickPlaceDrawerToCounter are bank
# atomics. The base never drives; the OTHER cup stays in the drawer.
SUBTASK_OPEN = "OpenDrawer"
SUBTASK_MOVE = "PickPlaceDrawerToCounter"

DONE_LANG = "task complete"


@register_labeler
class ChooseMeasuringCupLabeler(BaseSubtaskLabeler):
    """Labeler for the ChooseMeasuringCup composite task.

    Task (fixed-base): two measuring cups (a large + a small, same instance) sit in a
    CLOSED drawer. Open the drawer and put the requested cup (larger if `use_tablespoon`,
    else smaller) on the counter; leave the other. Robot starts at the drawer.

    Doc-canonical decomposition (2 high-level: open, cup move). Emitted:
      OpenDrawer [execute]              -> open the {side} drawer
      PickPlaceDrawerToCounter [pick]   -> pick up the {larger|smaller} measuring cup from the drawer
      PickPlaceDrawerToCounter [place]  -> place the {larger|smaller} measuring cup on the counter
    plus a terminal done.

    Detection: both cups RIDE the drawer as it slides open, so the chosen cup's rest is
    RE-ANCHORED at the drawer-open frame; its pick is the first departure from THAT rest
    (the lift-out onto the counter). OpenDrawer ends when the drawer reaches its adaptive
    open level; t_place = the gripper RELEASE.
    """

    ENV_NAME = "ChooseMeasuringCup"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        use_tbsp = ep_meta.get("refs", {}).get("use_tablespoon", False)
        self.chosen = "measuring_cup_large" if use_tbsp else "measuring_cup_small"
        self.size_word = "larger" if use_tbsp else "smaller"
        self.cup_body_id = env.obj_body_id[self.chosen]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # drawer_side is re-sampled on replay (random_index in ManipulateDrawer), so it is
        # unreliable -- take the side from the demo's own recorded lang ("Open the {side}
        # drawer ...") which describes the drawer the robot actually opened.
        m = re.search(r"[Oo]pen the (left|right) drawer", ep_meta.get("lang", ""))
        self.side = m.group(1) if m else (getattr(env, "drawer_side", "") or "")
        # drawer slide joint(s)
        dr = getattr(env, "drawer", None)
        jn = []
        if dr is not None:
            jn = [n for n in env.sim.model.joint_names
                  if n and dr.name in n and ("drawer" in n.lower() or "slide" in n.lower())]
        if not jn:
            jn = [n for n in env.sim.model.joint_names
                  if n and ("drawer" in n.lower() or "slide" in n.lower())]
        self.door_adrs, self.door_scales = [], []
        for j in jn:
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
        return {
            "cup": env.sim.data.body_xpos[self.cup_body_id].copy(),
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
        cup = np.array([h["cup"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))

        # OpenDrawer: the drawer slides from shut to its (adaptive) open level.
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

        # chosen cup rest re-anchored AFTER the drawer is open (it rode the drawer out);
        # its pick is the first departure from that rest -- the lift onto the counter.
        anchor = max(t_open, 0)
        rest = np.median(cup[anchor: anchor + self.REST_W], axis=0) if anchor + self.REST_W <= T \
            else np.median(cup[max(anchor, T - self.REST_W):], axis=0)
        disp = np.linalg.norm(cup - rest, axis=1)
        gd = np.linalg.norm(grip - cup, axis=1)

        t_pick = self._sustained_start(disp >= self.MOVE_TOL, anchor, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = min(anchor + 1, T - 1)
        t_open = int(min(t_open, max(t_pick - 1, 0)))
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_place = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        side = f"{self.side} " if self.side else ""
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if n_open > 0 and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the {side}drawer")
        else:
            review.append("drawer never detected opening")
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.size_word} measuring cup from the drawer")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.size_word} measuring cup on the counter")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": int(t_open), "n_open": n_open, "t_pick": int(t_pick),
            "t_place": int(t_place), "chosen": self.chosen, "n_segments": len(segments),
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
            issues.append("expected 1 OpenDrawer")
        return issues
