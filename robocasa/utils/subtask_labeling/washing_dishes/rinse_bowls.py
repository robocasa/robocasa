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


# Turn on the sink water, then rinse each bowl under the running faucet for a few seconds and
# set it back down. The faucet turn-on is the bank atomic TurnOnSinkFaucet. Each bowl is
# handled in place (picked up, held under the water, set down) -- not relocated -- so there is
# no bank pick-place: RinseBowl is a bank-extension skill spanning the pick / rinse(execute) /
# place of one bowl. Doc-canonical count = 3 (the per-bowl template); this task repeats it.
SUBTASK = "RinseBowl"
SUBTASK_FAUCET = "TurnOnSinkFaucet"
FAUCET_LANG = "turn on the sink faucet"

DONE_LANG = "task complete"

BOWLS = ("dish0", "dish1")


@register_labeler
class RinseBowlsLabeler(BaseSubtaskLabeler):
    """Labeler for the RinseBowls composite task.

    Task (non-mobile): rinse each bowl under the running water and place it back down in the
    sink. Per bowl: RinseBowl[pick] (grab + bring under the water) -> RinseBowl[execute]
    (hold under the running water) -> RinseBowl[place] (lift back out + set down + release).
    Bowls ordered by first grasped move; same-type bowls read "a bowl"/"the other bowl".

    Boundaries per bowl:
      - pick -> rinse:  the bowl first goes under the running water.
      - rinse -> place: the bowl is lifted back out after the last under-water frame.
      - place end:      the gripper releases the bowl.
    """

    ENV_NAME = "RinseBowls"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.10
    LIFT_TOL = 0.08
    MIN_PLACE = 10          # floor duration for the place when the bowl is already at rest
    PLACE_MOTION_TH = 0.004  # bowl speed above which the set-down is still in progress
    PLACE_PAD = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in BOWLS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.base = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "bowl"
        self.extra_subtasks = {SUBTASK}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy(), "water_on": won}
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
            try:
                d[o + "_uw"] = 1.0 if self.sink.check_obj_under_water(env, o) else 0.0
            except Exception:
                d[o + "_uw"] = 0.0
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
        uw = {o: np.array([h[o + "_uw"] for h in history]) for o in self.names}
        won = np.array([h.get("water_on", 0.0) for h in history])
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

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
        seen, art_noun, the_noun = {}, {}, {}
        for o in order:
            w = self.base[o]
            if counts[w] <= 1:
                art_noun[o] = the_noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                if seen[w] == 1:
                    art_noun[o], the_noun[o] = f"{art} {w}", f"the {w}"
                else:
                    # "the other bowl" only in the pick; rinse/place just say "the bowl"
                    art_noun[o], the_noun[o] = f"the other {w}", f"the {w}"

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        # Subtask 0: turn the sink water on (water starts off, handle turned early).
        on_idx = np.where(won > 0.5)[0]
        if on_idx.size:
            t_on = int(on_idx[0])
            # end the turn-on at the water-on frame, but never past the first bowl's grasp
            first_grasp = min((first_move[o] for o in order), default=T)
            t_on = int(min(max(t_on, 1), first_grasp if first_grasp < T else T))
            emit(t_on, SUBTASK_FAUCET, STAGE_EXECUTE, FAUCET_LANG)
        else:
            review.append("sink water never registered on -- check TurnOnSinkFaucet")

        n_rinse = 0
        for k, o in enumerate(order):
            # bound this bowl's place by where the next bowl starts (or the end)
            next_bound = int(first_move[order[k + 1]]) if k + 1 < len(order) else T
            sp = np.concatenate([np.linalg.norm(np.diff(opos[o], axis=0), axis=1), [0.0]])

            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_grab = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), cursor, self.MOVE_SUSTAIN)
            if t_grab is None:
                t_grab = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_grab is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_grab = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_grab = max(t_grab, cursor)

            uwi = np.where(uw[o] > 0.5)[0]
            uwi = uwi[(uwi >= t_grab) & (uwi < next_bound)] if uwi.size else uwi
            has_rinse = uwi.size > 0
            if has_rinse:
                first_uw, last_uw = int(uwi[0]), int(uwi[-1])
                pick_end = int(min(max(first_uw, t_grab + 1), next_bound - 1))
                rinse_end_nat = int(max(last_uw + 1, pick_end + 1))
                n_rinse += 1
            else:
                pick_end = int(min(t_grab + self.MIN_CARRY, next_bound - 1))
                rinse_end_nat = pick_end
                review.append(f"{o} never registered under running water -- check the rinse")

            # place = set the bowl back down (in the sink or on the counter). ALWAYS emitted.
            # It runs from the end of the rinse to where the bowl comes to rest (last frame of
            # bowl motion before the next bowl / the end). When the bowl is released right at
            # the water (no post-rinse room before the next bowl), carve the place out of the
            # tail of the rinse so "place the bowl down" is still a real segment, not a collapse.
            win_end = next_bound if next_bound < T else T
            mv_tail = np.where(sp[rinse_end_nat:win_end] > self.PLACE_MOTION_TH)[0]
            motion_end = (rinse_end_nat + int(mv_tail[-1]) + 1 + self.PLACE_PAD
                          if mv_tail.size else rinse_end_nat + self.MIN_PLACE)
            bowl_end = int(min(motion_end, win_end - 1))
            place_start = int(max(min(rinse_end_nat, bowl_end - self.MIN_PLACE), pick_end + 1))
            bowl_end = int(max(bowl_end, place_start + 1))

            emit(pick_end, SUBTASK, STAGE_PICK, f"pick up {art_noun[o]} from the sink")
            if has_rinse and place_start > cursor:
                emit(place_start, SUBTASK, STAGE_EXECUTE, f"rinse {the_noun[o]} under the running water")
            emit(bowl_end, SUBTASK, STAGE_PLACE, f"place {the_noun[o]} down")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if len(order) == 0:
            review.append("no bowls moved")
        result = self.build_result(T, segments, debug={"order": order, "n_rinse": n_rinse,
                                                       "n_bowls": len(order), "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        npick = int(np.sum((names == SUBTASK) & (stages == STAGE_PICK)))
        nplace = int(np.sum((names == SUBTASK) & (stages == STAGE_PLACE)))
        # every bowl gets a pick and a place ("place the bowl down" in the sink or on counter)
        if npick != nplace:
            issues.append(f"pick/place mismatch ({npick} picks vs {nplace} places)")
        if npick < 1:
            issues.append("expected at least 1 bowl rinsed")
        if int(np.sum(names == SUBTASK_FAUCET)) != 1:
            issues.append("expected 1 TurnOnSinkFaucet")
        return issues
