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


# Pick the metallic utensils off the dining counter and drop them in the (open) drawer, leaving
# the wooden utensils behind (distractors). Two or three metallic utensils move depending on the
# episode. Each is a nav-interrupted move (the drawer is across the kitchen from the dining
# counter). Dining-counter -> drawer is the bank atomic PickPlaceCounterToDrawer; NavigateKitchen
# is a bank atomic.
SUBTASK_MOVE = "PickPlaceCounterToDrawer"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# candidate metallic objects (some episodes have only the first two); the wooden spoons never move.
METALLIC = ("metallic_utensil", "metallic_utensil_2", "third_metallic_utensil")


@register_labeler
class OrganizeMetallicUtensilsLabeler(BaseSubtaskLabeler):
    """Labeler for the OrganizeMetallicUtensils composite task.

    Task (mobile): a mix of metallic and wooden utensils sits on the dining counter (robot starts
    there); an open drawer is across the kitchen. Move only the metallic utensils into the drawer;
    leave the wooden ones. Two or three metallic utensils are present (variable episode-to-episode).

    Doc-canonical count = 7 (`mobile: Yes`, action-level): the metallic utensils' nav-interrupted
    moves. Each (in first-motion order) is a round trip:
      (NavigateKitchen back to the dining counter) -> metallic[pick]
      -> NavigateKitchen(carry to the open drawer) -> metallic[place in the open drawer]
    The first utensil has no lead-in nav (the robot starts at the dining counter; k>=1 gate).
    Utensils are named "metallic {type}" (get_obj_lang); two of the same type get a/the-other.

    Boundaries (RecycleBottlesBySize machinery): t_pick = first sustained displacement; nav OUT =
    base drive cursor->grasp (k>=1) ending at the position+yaw arrival; nav BACK = base drive
    during the carry, ending at the arrival; place = the gripper RELEASE.
    """

    ENV_NAME = "OrganizeMetallicUtensils"

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
        self.names = [o for o in METALLIC if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: f"metallic {env.get_obj_lang(o)}" for o in self.names}
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
        self.extra_subtasks = set()
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
            word = self.word[o]
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

            if k >= 1 and base_xy is not None:
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, t_move - self.MIN_PICK_GAP)
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the dining counter")
            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up {pick_art} {word} from the dining counter")
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the open drawer")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {place_art} {word} in the open drawer")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        # 2 or 3 metallic utensils move (variable); fewer than 2 is worth a look.
        if n_moved < 2:
            review.append(f"only {n_moved} metallic utensils moved -- verify")

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
        npick = int(np.sum(stages == STAGE_PICK))
        nplace = int(np.sum(stages == STAGE_PLACE))
        if npick != nplace:
            issues.append(f"pick/place mismatch: {npick} picks, {nplace} places")
        if npick not in (2, 3):
            issues.append(f"expected 2-3 metallic moves, found {npick}")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 6):
            issues.append(f"expected 0-6 NavigateKitchen, found {n_nav}")
        return issues
