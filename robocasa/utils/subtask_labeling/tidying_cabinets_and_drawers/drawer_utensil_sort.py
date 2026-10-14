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


# Pull open the (closed) drawer, then put two counter utensils inside it. OpenDrawer and
# PickPlaceCounterToDrawer are bank atomics. The robot starts at the counter in front of
# the drawer (local).
SUBTASK_OPEN = "OpenDrawer"
SUBTASK_MOVE = "PickPlaceCounterToDrawer"

DONE_LANG = "task complete"

OBJS = ("utensil1", "utensil2")


@register_labeler
class DrawerUtensilSortLabeler(BaseSubtaskLabeler):
    """Labeler for the DrawerUtensilSort composite task.

    Task (mobile, local): two utensils sit on the counter; a drawer in front starts closed.
    Open the drawer and put both utensils inside.

    Doc-canonical count = 3 (`mobile: Yes`): OpenDrawer + two counter->drawer utensil moves.
      OpenDrawer [execute]              -> open the {side} drawer
      PickPlaceCounterToDrawer [pick]   -> pick up the {utensil} from the counter
      PickPlaceCounterToDrawer [place]  -> place the {utensil} in the drawer
    plus a terminal done. Utensils in first-move order; same-type utensils are "a X"/"the
    other X".

    Boundaries: OpenDrawer ends when the drawer slide reaches its adaptive open level; per
    utensil t_pick = first sustained displacement, t_place = the gripper RELEASE. The drawer
    side comes from the ep_meta language (env.drawer_side is re-randomized on replay).
    """

    ENV_NAME = "DrawerUtensilSort"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    OPEN_INIT = 0.35      # the INITIAL drawer pull reaches this before the first utensil
    OPEN_MIN = 0.20
    OPEN_RISE_MIN = 0.06
    OPEN_SUSTAIN = 5
    GRASP_TH = 0.07       # grip->utensil dist below this == grasped (filters bumps)
    MIN_PICK_GAP = 12
    COMMIT_W = 30         # a real carry keeps the object displaced this long...
    COMMIT_FRAC = 0.5     # ...staying above MOVE_TOL*this; a transient bump settles back below

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
                self.base[o] = "utensil"
        # drawer side from the ep_meta lang ("... the {side} drawer ...") -- env.drawer_side
        # is re-randomized on replay and unreliable.
        lang = ep_meta.get("lang", "")
        m = re.search(r"the (\w+) drawer", lang)
        self.drawer_side = m.group(1) if m else ""
        dr = getattr(env, "drawer", None)
        jn = [n for n in env.sim.model.joint_names
              if n and dr is not None and dr.name in n and ("drawer" in n.lower() or "slide" in n.lower())]
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
        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        d["drawer"] = float(max((abs(float(env.sim.data.qpos[a])) / s
                                 for a, s in zip(self.door_adrs, self.door_scales)), default=0.0))
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

    def _committed_onset(self, dsp, start):
        """First frame >= start where the object leaves rest (dsp >= MOVE_TOL) AND stays
        displaced for COMMIT_W frames (never settling back toward rest). This is the start of
        the committed carry; it ignores a transient bump (which rises then settles) without
        relying on the grip->object distance, which for a long utensil held at one end stays
        offset (~0.08) through the real carry and only dips low spuriously."""
        T = len(dsp)
        floor = self.MOVE_TOL * self.COMMIT_FRAC
        for t in range(int(start), T):
            if dsp[t] >= self.MOVE_TOL and bool(np.all(dsp[t: min(t + self.COMMIT_W, T)] >= floor)):
                return t
        return None

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}
        drawer = np.array([h["drawer"] for h in history])

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- utensils: order by COMMITTED carry onset. A long utensil (whisk/ladle) is held
        # at one end, so grip->center stays ~0.08 (above GRASP_TH) through its real carry while
        # the gripper only dips <GRASP_TH transiently -- often near the OTHER utensil as it goes
        # into the same drawer. Keying order/pick on "grasped" therefore mis-orders same-type
        # pairs (the early carry looks late) and absorbs one utensil's whole trip into the
        # other's PICK. The committed onset -- the first displacement the object SUSTAINS,
        # never settling back -- ignores a bump without needing the grasp signal.
        # ##demo_16/34/50/85 (two whisks or two ladles).
        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._committed_onset(dsp, 0)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])
        first_util = min((first_move[o] for o in order), default=T)

        # --- OpenDrawer: the INITIAL pull, which reaches a modest level before the first
        # utensil is grabbed. The drawer is often opened further between utensils (two-stage
        # open) -- that re-open folds into the next pick, not a second OpenDrawer. Cap the
        # open before the first grab so it never swallows the first utensil's pick.
        peak = float(drawer.max()) if T else 0.0
        rose = peak - float(drawer[0]) if T else 0.0
        t_open = None
        if peak >= self.OPEN_MIN and rose >= self.OPEN_RISE_MIN:
            t_open = self._sustained_start(drawer >= self.OPEN_INIT, 0, self.OPEN_SUSTAIN)
            if t_open is not None:
                t_open = min(t_open, max(first_util - self.MIN_PICK_GAP, 1))
        side = f"{self.drawer_side} " if self.drawer_side else ""
        if t_open is not None and t_open > cursor:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the {side}drawer")
        elif rose >= self.OPEN_RISE_MIN:
            review.append("drawer moved but open not cleanly detected -- verify")

        from collections import Counter
        counts = Counter(self.base[o] for o in order)
        seen = {}
        disp = {}
        for o in order:
            b = self.base[o]
            if counts[b] <= 1:
                disp[o] = f"the {b}"
            else:
                seen[b] = seen.get(b, 0) + 1
                art = "an" if b[:1].lower() in "aeiou" else "a"
                disp[o] = f"{art} {b}" if seen[b] == 1 else f"the other {b}"

        n_moved = 0
        for o in order:
            nn = disp[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._committed_onset(dsp, cursor)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # floor the pick: the second utensil is often grabbed straight after the first
            # is placed (or across the second drawer re-open), collapsing its pick to ~0.
            t_pick = int(min(max(t_pick, cursor + self.MIN_PICK_GAP),
                             max(t_placed - self.MIN_PICK_GAP, cursor + 1)))
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the counter")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} in the drawer")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 utensils moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "t_open": t_open, "peak": round(peak, 3),
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
        names = result.subtask_name[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_OPEN)) not in (0, 1):
            issues.append(f"expected 0-1 OpenDrawer, found {int(np.sum(names == SUBTASK_OPEN))}")
        return issues
