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


# Sort four counter utensils by material: the two wooden spoons go into the open cabinet, the two
# metallic utensils into the open drawer. Each move is a bank atomic -- wooden ->
# PickPlaceCounterToCabinet, metallic -> PickPlaceCounterToDrawer. The cabinet/drawer sit away
# from the counter, so each utensil is a nav-interrupted move (carry out, then drive back to the
# counter for the next). NavigateKitchen is a bank atomic.
SUBTASK_CABINET = "PickPlaceCounterToCabinet"
SUBTASK_DRAWER = "PickPlaceCounterToDrawer"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class ArrangeUtensilsByTypeLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeUtensilsByType composite task.

    Task (mobile): two wooden spoons and two metallic utensils sit on the counter (robot starts
    there); an open cabinet and an open drawer are alongside. Put the wooden spoons in the open
    cabinet and the metallic utensils in the open drawer.

    Doc-canonical decomposition (4 high-level: one move per utensil), emitted as nav-interrupted
    moves in first-motion order. Each utensil:
      (NavigateKitchen [navigate])             -> navigate back to the counter with the utensils (k>=1)
      PickPlaceCounterToCabinet/Drawer [pick]  -> pick up {a|the other} {word} from the counter
      (NavigateKitchen [navigate])             -> carry the {word} to the open {cabinet|drawer}
      PickPlaceCounterToCabinet/Drawer [place] -> place {the|the other} {word} in the open {cabinet|drawer}
    plus a terminal done. `word` is "wooden spoon" for the wooden pair and "metallic utensil" for
    the metallic pair; the two of a kind are disambiguated a/the-other by move order. The first
    utensil has no lead-in nav (the robot starts at the counter; k>=1 gate). Both navs are adaptive
    -- emitted only when the base actually travels (local layouts suppress them).

    Boundaries (RecycleBottlesBySize machinery): t_pick = first sustained displacement from a rest
    re-anchored at the segment start; nav OUT = base drive cursor->grasp ending at the position+yaw
    arrival; carry nav ends at the arrival; place = the gripper RELEASE.
    """

    ENV_NAME = "ArrangeUtensilsByType"

    # object -> (word, dest word, subtask)
    ITEMS = {
        "wooden_spoon": ("wooden spoon", "open cabinet", SUBTASK_CABINET),
        "wooden_spoon_2": ("wooden spoon", "open cabinet", SUBTASK_CABINET),
        "metallic_utensil": ("metallic utensil", "open drawer", SUBTASK_DRAWER),
        "metallic_utensil_2": ("metallic utensil", "open drawer", SUBTASK_DRAWER),
    }

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.20
    MIN_PICK_GAP = 12
    ARRIVE_RADIUS = 0.20
    ARRIVE_STOP_TH = 0.006
    ARRIVE_SUSTAIN = 12
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in self.ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
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
            # first frame in [start, end) the base is BOTH within ARRIVE_RADIUS of `park` (its
            # parked destination pose, incl. yaw) AND stopped, sustained -> the real settle. Falls
            # back to the last substantial-motion frame if the base drives straight into the grab.
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
            word, dest, sub = self.ITEMS[o]
            m = seen.get(word, 0)
            seen[word] = m + 1
            pick_art = "a" if m == 0 else "the other"
            place_art = "the" if m == 0 else "the other"

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

            # lead-in nav: the drive back to the counter to fetch this utensil (not the first --
            # the robot starts at the counter).
            if k >= 1 and base_xy is not None:
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, t_move - self.MIN_PICK_GAP)
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the counter with the utensils")
            emit(t_move, sub, STAGE_PICK, f"pick up {pick_art} {word} from the counter")
            # carry to the cabinet/drawer
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the {dest}")
            emit(t_placed, sub, STAGE_PLACE, f"place {place_art} {word} in the {dest}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 4:
            review.append(f"only {n_moved}/4 utensils moved")

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
        if not (0 <= n_nav <= 7):
            issues.append(f"expected 0-7 NavigateKitchen, found {n_nav}")
        return issues
