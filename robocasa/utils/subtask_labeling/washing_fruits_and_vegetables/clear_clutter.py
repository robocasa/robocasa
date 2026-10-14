from collections import Counter

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


# Move the produce from the counter into the sink, turn the faucet on to wash it, turn it off,
# then move the produce into the tray. Counter->sink is PickPlaceCounterToSink (bank);
# sink->tray has no bank name -> PickPlaceSinkToTray (ext); TurnOn/OffSinkFaucet are bank.
# There are 1 OR 2 produce items (env samples num_food in {1, 2}); each distractor unwashable
# object is ignored. Doc count = 6 (for a single produce).
SUBTASK_IN = "PickPlaceCounterToSink"
SUBTASK_OUT = "PickPlaceSinkToTray"
SUBTASK_ON = "TurnOnSinkFaucet"
SUBTASK_OFF = "TurnOffSinkFaucet"

DONE_LANG = "task complete"
PRODUCE = ("obj_0", "obj_1", "obj_2")


@register_labeler
class ClearClutterLabeler(BaseSubtaskLabeler):
    """Labeler for the ClearClutter composite task.

    Task: move every produce item (1 or 2 of them) from the counter into the sink, turn the
    faucet on to wash them and off, then move every produce item from the sink into the tray.
    Two grasp cycles bracket the faucet on/off:
      [counter->sink pick/place for each produce] -> TurnOnSinkFaucet -> TurnOffSinkFaucet ->
      [sink->tray pick/place for each produce].
    Doc count = 6 for a single produce; with two produce the pick/place pairs repeat (the gate
    checks coverage, not a fixed total). The unwashable distractor(s) are never moved.

    The produce order within each cycle is by first grasped move (the second cycle is typically
    LIFO). The base does not navigate in this task, so there are no nav subtasks. Picks split at
    first sustained displacement (cursor-relative rest); places = gripper RELEASE; the wash spans
    from the last counter->sink place to the last water-on frame; the turn-off spans to the first
    re-grasp.
    """

    ENV_NAME = "ClearClutter"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.09
    REGRASP_SUSTAIN = 5
    DEPART_TOL = 0.08   # gripper displacement from the faucet handle that ends the turn-off
    OFF_MIN = 12        # floor the turn-off span (a handle flick is a real, if brief, action)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.produce = [o for o in PRODUCE if o in env.obj_body_id]
        self.body = {o: env.obj_body_id[o] for o in self.produce}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.base = {}
        for o in self.produce:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "produce"
        self.extra_subtasks = {SUBTASK_OUT}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "water_on": won,
        }
        for o in self.produce:
            d[o] = env.sim.data.body_xpos[self.body[o]].copy()
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

    def _grasped_move(self, pos, gd, cursor, T):
        """First sustained displacement (from a cursor-relative rest) of a grasped object."""
        rest = np.median(pos[max(cursor - self.REST_W, 0): cursor + 1], axis=0)
        dsp = np.linalg.norm(pos - rest, axis=1)
        t = self._sustained_start((dsp >= self.MOVE_TOL) & (gd < self.GRASP_TH),
                                  cursor, self.MOVE_SUSTAIN)
        if t is None:
            t = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t is None:
            mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
            t = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        return max(int(t), cursor)

    def _release(self, gd, t_move, T):
        hold = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        t = self._sustained_start(gd > hold + self.RELEASE_MARGIN,
                                  t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        return int(t) if t is not None else T - 1

    def resolve(self, history):
        T = len(history)
        pos = {o: np.array([h[o] for h in history]) for o in self.produce}
        grip = np.array([h["grip"] for h in history])
        won = np.array([h["water_on"] for h in history])
        gd = {o: np.linalg.norm(grip - pos[o], axis=1) for o in self.produce}
        N = len(self.produce)

        # per-type naming: distinct types -> "the X"; a same-type pair -> pick "a X"/"the other
        # X", place always "the X". Order of first appearance in cycle 1.
        def first_move_all(start):
            fm = {}
            for o in self.produce:
                rest = np.median(pos[o][max(start - self.REST_W, 0): start + 1], axis=0)
                dsp = np.linalg.norm(pos[o] - rest, axis=1)
                t = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH),
                                          start, self.MOVE_SUSTAIN)
                if t is None:
                    t = self._sustained_start(dsp >= self.MOVE_TOL, start, self.MOVE_SUSTAIN)
                fm[o] = int(t) if t is not None else T
            return fm

        fm1 = first_move_all(0)
        order1 = sorted([o for o in self.produce if fm1[o] < T], key=lambda o: fm1[o])
        # produce that never registered a move still get processed in name order
        order1 += [o for o in self.produce if o not in order1]

        counts = Counter(self.base[o] for o in order1)
        seen, art_noun, the_noun = {}, {}, {}
        for o in order1:
            w = self.base[o]
            the_noun[o] = f"the {w}"
            if counts[w] <= 1:
                art_noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                art_noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- cycle 1: every produce counter -> sink ---
        for o in order1:
            t_move = self._grasped_move(pos[o], gd[o], cursor, T)
            t_place = self._release(gd[o], t_move, T)
            emit(t_move, SUBTASK_IN, STAGE_PICK, f"pick up {art_noun[o]} from the counter")
            emit(t_place, SUBTASK_IN, STAGE_PLACE, f"place {the_noun[o]} in the sink")

        # --- faucet on (wash), then off ---
        onf = np.where(won > 0.5)[0]
        onf = onf[onf >= cursor] if onf.size else onf
        has_water = onf.size > 0
        t_off_start = int(min(max(int(onf[-1]) + 1, cursor + 1), T - 1)) if has_water \
            else min(cursor + 1, T - 1)
        if N == 1:
            wash_lang = f"turn on the sink faucet to wash the {self.base[order1[0]]}"
        else:
            wash_lang = "turn on the sink faucet to wash them"
        emit(t_off_start, SUBTASK_ON, STAGE_EXECUTE, wash_lang)

        # The turn-off is quick: it ends when the gripper leaves the faucet handle. The long
        # travel across to the produce in the sink belongs to the following cycle-2 pick (a pick
        # spans approach -> object first move), not to the faucet subtask.
        faucet = np.median(grip[max(t_off_start - 3, 0): t_off_start + 1], axis=0)
        dep = np.linalg.norm(grip - faucet, axis=1)
        t_off_end = self._sustained_start(dep >= self.DEPART_TOL, t_off_start, self.MOVE_SUSTAIN)
        if t_off_end is None:
            # fallback: gripper never clearly departs -> first re-grasp of any produce
            regrab = [self._sustained_start(gd[o] < self.GRASP_TH, t_off_start,
                                            self.REGRASP_SUSTAIN) for o in self.produce]
            regrab = [r for r in regrab if r is not None]
            t_off_end = int(min(regrab)) if regrab else min(t_off_start + 1, T - 1)
        t_off_end = min(max(t_off_end, t_off_start + self.OFF_MIN), T - 1)
        emit(t_off_end, SUBTASK_OFF, STAGE_EXECUTE, "turn off the sink faucet")

        # --- cycle 2: every produce sink -> tray (grasped order after the wash) ---
        fm2 = first_move_all(cursor)
        order2 = sorted([o for o in self.produce if fm2[o] < T], key=lambda o: fm2[o])
        order2 += [o for o in self.produce if o not in order2]
        for o in order2:
            t_move = self._grasped_move(pos[o], gd[o], cursor, T)
            t_place = self._release(gd[o], t_move, T)
            emit(t_move, SUBTASK_OUT, STAGE_PICK, f"pick up {the_noun[o]} from the sink")
            emit(t_place, SUBTASK_OUT, STAGE_PLACE, f"place {the_noun[o]} in the tray")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_water:
            review.append("sink water never registered on -- verify the wash")
        result = self.build_result(T, segments, debug={
            "N": N, "order1": order1, "order2": order2, "t_off_start": t_off_start,
            "t_off_end": t_off_end, "has_water": has_water, "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_OUT}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        N = len(self.produce)
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage):
            return int(np.sum((names == name) & (stages == stage)))

        for nm, st, exp in ((SUBTASK_IN, STAGE_PICK, N), (SUBTASK_IN, STAGE_PLACE, N),
                            (SUBTASK_OUT, STAGE_PICK, N), (SUBTASK_OUT, STAGE_PLACE, N)):
            if count(nm, st) != exp:
                issues.append(f"expected {exp} {nm}/{st}, found {count(nm, st)}")
        for nm in (SUBTASK_ON, SUBTASK_OFF):
            if int(np.sum(names == nm)) != 1:
                issues.append(f"expected 1 {nm}, found {int(np.sum(names == nm))}")
        return issues
