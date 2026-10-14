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


# Pick the washed foods out of the sink and sort them into two bowls (lemon wedges in one,
# broccoli in the other), then turn off the sink faucet. Sink -> bowl has no bank name ->
# PickPlaceSinkToBowl (extension); TurnOffSinkFaucet is bank. Non-mobile. Doc count = 4.
SUBTASK_MOVE = "PickPlaceSinkToBowl"
SUBTASK_OFF = "TurnOffSinkFaucet"

OFF_LANG = "turn off the sink faucet"
DONE_LANG = "task complete"

FOODS = ("food1", "food2", "food3")
BOWLS = ("bowl1", "bowl2")


@register_labeler
class AfterwashSortingLabeler(BaseSubtaskLabeler):
    """Labeler for the AfterwashSorting composite task.

    Task (non-mobile): several washed foods sit in the sink; sort them into two bowls (same-
    type foods go together), then turn off the sink faucet. Doc-canonical count = 4. Foods
    ordered by first move; each food's destination bowl is detected (nearest bowl at the
    place), labeled "one bowl" / "the other bowl" by order of first use. Same-type foods read
    "a X"/"the other X".

    pick/place split at each food's first sustained displacement (cursor-relative rest); place
    = gripper release; TurnOffSinkFaucet spans from the last place to the end (water off).
    """

    ENV_NAME = "AfterwashSorting"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.foods = [o for o in FOODS if o in env.obj_body_id]
        self.bowls = [o for o in BOWLS if o in env.obj_body_id]
        self.body = {o: env.obj_body_id[o] for o in self.foods + self.bowls}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.base = {}
        for o in self.foods:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "item"
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy(), "water_on": won}
        for o in self.foods + self.bowls:
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

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.foods + self.bowls}
        won = np.array([h["water_on"] for h in history])
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.foods}

        first_move = {}
        for o in self.foods:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.foods if first_move[o] < T], key=lambda o: first_move[o])

        from collections import Counter
        counts = Counter(self.base[o] for o in order)
        seen, noun = {}, {}
        art_noun, the_noun = {}, {}
        for o in order:
            w = self.base[o]
            if counts[w] <= 1:
                art_noun[o] = the_noun[o] = f"the {w}"
            else:
                seen[w] = seen.get(w, 0) + 1
                art = "an" if w[:1].lower() in "aeiou" else "a"
                # pick uses "a"/"the other"; place always uses "the {w}" (no "other")
                art_noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"
                the_noun[o] = f"the {w}"

        # bowl destination phrasing: first placement -> "a bowl"; a NEW, different bowl ->
        # "the other bowl"; a bowl that already holds a same-type food -> "the same bowl as the
        # other {food}" (references the item already there); a bowl with a different-type food ->
        # "the same bowl as the {existing food}".
        used = []
        bowl_contents = {}  # bname -> list of food-type strings placed there

        def bowl_phrase(bname, w):
            if bname not in bowl_contents:
                phrase = "a bowl" if not used else "the other bowl"
                used.append(bname)
                bowl_contents[bname] = [w]
                return phrase
            existing = bowl_contents[bname]
            phrase = (f"the same bowl as the other {w}" if w in existing
                      else f"the same bowl as the {existing[0]}")
            bowl_contents[bname].append(w)
            return phrase

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n = 0
        for o in order:
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # destination bowl = nearest bowl to the food at the place
            fp = opos[o][min(t_placed, T - 1)][:2]
            if self.bowls:
                bname = min(self.bowls, key=lambda b: float(np.linalg.norm(opos[b][min(t_placed, T - 1)][:2] - fp)))
                dest = bowl_phrase(bname, self.base[o])
            else:
                dest = "the bowl"
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {art_noun[o]} from the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {the_noun[o]} in {dest}")
            n += 1

        # TurnOffSinkFaucet: water starts on and is turned off after the foods are sorted.
        review = []
        offf = np.where(won > 0.5)[0]
        water_ends_off = bool(won[-1] < 0.5) if T else False
        if water_ends_off and offf.size:
            t_off = int(min(max(int(offf[-1]) + 1, cursor + 1), T - 1))
            emit(t_off, SUBTASK_OFF, STAGE_EXECUTE, OFF_LANG)
        elif cursor < T - 1:
            # water stays on at the end -> still emit a turn-off tail, flag it
            emit(min(cursor + 1, T - 1), SUBTASK_OFF, STAGE_EXECUTE, OFF_LANG)
            review.append("sink water did not register off at the end -- verify TurnOffSinkFaucet")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n != len(self.foods):
            review.append(f"only {n}/{len(self.foods)} foods moved")
        result = self.build_result(T, segments, debug={"order": order, "n": n,
                                                       "bowls_used": used, "n_segments": len(segments)})
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
        npick = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK)))
        if npick != len(self.foods):
            issues.append(f"expected {len(self.foods)} food picks, found {npick}")
        if int(np.sum(names == SUBTASK_OFF)) != 1:
            issues.append("expected 1 TurnOffSinkFaucet")
        return issues
