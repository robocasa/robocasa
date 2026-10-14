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


# Fully pull open the drawer in front, then cluster the utensils by type inside it: the counter
# fork goes on the fork side, the counter spoon on the spoon side (a fork and a spoon already sit
# in the drawer on those sides). OpenDrawer and PickPlaceCounterToDrawer are bank atomics. Fixed
# base (the drawer is right in front of where the robot starts).
SUBTASK_OPEN = "OpenDrawer"
SUBTASK_MOVE = "PickPlaceCounterToDrawer"

DONE_LANG = "task complete"


@register_labeler
class ClusterUtensilsInDrawerLabeler(BaseSubtaskLabeler):
    """Labeler for the ClusterUtensilsInDrawer composite task.

    Task (mobile, but local): a fork and a spoon sit on the counter; the drawer in front starts
    only part-way open and already holds a fork on one side and a spoon on the other. Pull the
    drawer fully open, then put the counter fork on the fork side and the counter spoon on the
    spoon side. The robot starts at the counter.

    Doc-canonical decomposition (2 utensil moves) plus the drawer open the robot performs first:
      OpenDrawer [execute]             -> fully open the drawer in front
      PickPlaceCounterToDrawer [pick]  -> pick up the {fork|spoon} from the counter
      PickPlaceCounterToDrawer [place] -> place the {fork|spoon} on the {side} side of the drawer
    (the two utensil moves in first-motion order, fork first) plus a terminal done. The sides come
    from ep_meta (fork_side / spoon_side).

    Boundaries: OpenDrawer ends when the drawer slide reaches its adaptive open level (0.9*peak,
    sustained). Per utensil: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE.
    """

    ENV_NAME = "ClusterUtensilsInDrawer"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    # drawer-open detection (normalized slide value in [0, 1])
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20       # peak must exceed this for an open to count
    OPEN_RISE_MIN = 0.06  # ...and the drawer must slide at least this far from its start (a real
                          # pull; below this the drawer is already open enough and the robot just
                          # places -- a legitimate no-open episode, not an error)
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = ["counter_fork", "counter_spoon"]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {"counter_fork": "fork", "counter_spoon": "spoon"}
        self.side = {
            "counter_fork": ep_meta.get("fork_side", "correct"),
            "counter_spoon": ep_meta.get("spoon_side", "correct"),
        }
        # drawer slide joint(s) -- resolved per demo (the layout/model varies episode to episode)
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

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}
        drawer = np.array([h["drawer"] for h in history])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        # --- OpenDrawer: the robot pulls the (part-open) drawer fully open before placing ---
        peak = float(drawer.max()) if T else 0.0
        rose = peak - float(drawer[0]) if T else 0.0
        t_open = None
        if peak >= self.OPEN_MIN and rose >= self.OPEN_RISE_MIN:
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            t_open = self._sustained_start(drawer >= lvl, 0, self.OPEN_SUSTAIN)
        if t_open is not None and t_open > cursor:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "fully open the drawer in front")
        elif rose >= self.OPEN_RISE_MIN:
            # the drawer moved but the open level was not cleanly reached -- worth a look
            review.append("drawer moved but open not cleanly detected -- verify")
        # else: the drawer starts open enough and the robot never pulls it -- a legitimate
        # no-open episode; emit no OpenDrawer and do not flag.

        # --- the two utensil moves, in first-motion order (after the drawer is open) ---
        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][cursor: cursor + self.REST_W], axis=0) if T > cursor else opos[o][0]
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        n_moved = 0
        for o in order:
            word, side = self.word[o], self.side[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the counter")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} on the {side} side of the drawer")
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
        # OpenDrawer is present when the robot pulls the drawer open; a few episodes start with
        # the drawer already open enough and legitimately have none.
        if int(np.sum(names == SUBTASK_OPEN)) not in (0, 1):
            issues.append(f"expected 0-1 OpenDrawer, found {int(np.sum(names == SUBTASK_OPEN))}")
        return issues
