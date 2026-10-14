import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Pack four foods into two tupperwares on the dining counter, sorted by temperature: the two warm
# items (one from a pan on the stove, one from a plate on the counter by the stove) go in one
# tupperware; the two cold items (from the fridge) go in the other. Each food is a nav-interrupted
# move to the dining counter. Source-appropriate atomics: stove -> PickPlaceStoveToCounter (bank);
# counter -> PickPlaceCounterToCounter (ext); fridge -> PickPlaceFridgeToCounter (ext).
SUBTASK_STOVE = "PickPlaceStoveToCounter"
SUBTASK_COUNTER = "PickPlaceCounterToCounter"
SUBTASK_FRIDGE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class PackFoodByTempLabeler(BaseSubtaskLabeler):
    """Labeler for the PackFoodByTemp composite task.

    Task (mobile): two warm foods (hot0 in a pan on the stove, hot1 on a plate on the counter by
    the stove) and two cold foods (cold0/cold1 in the open fridge). Pack the warm pair into one
    tupperware and the cold pair into the other, both on the dining counter. The robot starts at
    the fridge.

    Doc-canonical count = 15 (`mobile: Yes`, action-level): the four foods' nav-interrupted moves
    (pick + carry-nav + place each, plus the lead-in navs between sources). Each food, in
    first-motion order:
      (NavigateKitchen to its source) -> food[pick]
      -> NavigateKitchen(carry to the dining counter) -> food[place]
    A lead-in nav is emitted whenever the base actually travels to reach the food (so the very
    first food gets one iff it is not the fridge item the robot already stands at). Warm items
    land in "the tupperware for the warm items", cold in "the tupperware for the cold items".

    Boundaries (RecycleBottlesBySize machinery): t_pick = first sustained displacement; navs end
    at the position+yaw arrival; place = the gripper RELEASE.
    """

    ENV_NAME = "PackFoodByTemp"

    # object -> (source_landmark, pick_from phrase, dest phrase, subtask)
    # object -> (source_landmark, pick_from phrase, dest phrase, subtask, temperature group)
    ITEMS = {
        "hot0": ("the stove", "the stove", "the tupperware for the warm items", SUBTASK_STOVE, "warm"),
        "hot1": ("the counter by the stove", "the counter by the stove", "the tupperware for the warm items", SUBTASK_COUNTER, "warm"),
        "cold0": ("the fridge", "the fridge", "the tupperware for the cold items", SUBTASK_FRIDGE, "cold"),
        "cold1": ("the fridge", "the fridge", "the tupperware for the cold items", SUBTASK_FRIDGE, "cold"),
    }

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.20
    MIN_PICK_GAP = 12
    ARRIVE_RADIUS = 0.20
    ARRIVE_STOP_TH = 0.006
    ARRIVE_SUSTAIN = 12
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in self.ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: env.get_obj_lang(o) for o in self.names}
        from collections import Counter
        counts = Counter(self.word.values())
        self.same_type = {o: counts[self.word[o]] > 1 for o in self.names}
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_COUNTER, SUBTASK_FRIDGE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
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

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0
        seen = {}
        # first item placed in each temperature group -> its sibling gets a more specific carry
        # ("...to the tupperware ... with the {first item}") since that tupperware now holds it.
        group_first_word = {}
        # sources the robot has already stood at -> a return trip reads "navigate back to ...".
        # The robot starts at the fridge, so any drive to the fridge is a return.
        visited_sources = {"the fridge"}

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        def arrival(start, end, park):
            if base_xy is None or end <= start:
                return None
            near = np.linalg.norm(base_xy[start: end] - park, axis=1) < self.ARRIVE_RADIUS
            stopped = binst[start: end] < self.ARRIVE_STOP_TH
            fr = self._sustained_start(near & stopped, 0, self.ARRIVE_SUSTAIN)
            if fr is not None:
                return start + fr
            mv = np.where(binst[start: end] > self.BASE_MOVE_TH)[0]
            return start + int(mv[-1]) + 1 if mv.size else None

        n_moved = 0
        for k, o in enumerate(order):
            src_land, pick_from, dest, sub, group = self.ITEMS[o]
            word = self.word[o]
            first_of_group = group not in group_first_word
            if first_of_group:
                group_first_word[group] = word
            m = seen.get(word, 0)
            seen[word] = m + 1
            if self.same_type[o]:
                pick_art = "a" if m == 0 else "the other"
                place_art = "the" if m == 0 else "the other"
            else:
                pick_art = place_art = "the"

            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            # lead-in nav to the food's source -- emitted whenever the base actually travels there
            # (so the first food gets one iff the robot is not already standing at its source).
            if base_xy is not None:
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, t_move - self.MIN_PICK_GAP)
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    back = src_land in visited_sources
                    nav_out_lang = (f"navigate back to {src_land}" if back
                                    else f"navigate to {src_land}")
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, nav_out_lang)
            visited_sources.add(src_land)
            emit(t_move, sub, STAGE_PICK, f"pick up {pick_art} {word} from {pick_from}")
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                if first_of_group:
                    carry_lang = f"carry the {word} to the dining counter with the tupperwares"
                else:
                    # the sibling of this item is already in its tupperware -> point at it
                    carry_lang = (f"carry the {word} to the tupperware on the dining counter "
                                  f"with the {group_first_word[group]}")
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, carry_lang)
            # the first item of a group chooses one of the two empty tupperwares ("a tupperware");
            # the second item names the tupperware by its temperature (dest = "...warm/cold items").
            place_dest = "a tupperware" if first_of_group else dest
            emit(t_placed, sub, STAGE_PLACE, f"place {place_art} {word} in {place_dest}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 4:
            review.append(f"only {n_moved}/4 foods moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 4:
            issues.append(f"expected 4 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 4:
            issues.append(f"expected 4 places, found {int(np.sum(stages == STAGE_PLACE))}")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 8):
            issues.append(f"expected 0-8 NavigateKitchen, found {n_nav}")
        return issues
