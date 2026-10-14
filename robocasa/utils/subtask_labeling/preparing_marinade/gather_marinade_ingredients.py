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


# Gather marinade ingredients around a mixing bowl: bring the oil/vinegar bottle and the shaker
# from the (open) cabinet to the counter beside the bowl, and add garlic from the (open) fridge
# into the bowl. Cabinet -> counter is the bank atomic PickPlaceCabinetToCounter; fridge -> bowl
# (counter) has no bank name -> PickPlaceFridgeToCounter (extension); NavigateKitchen is a bank atomic.
SUBTASK_CABINET = "PickPlaceCabinetToCounter"
SUBTASK_FRIDGE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

BOWL_LM = "the counter with the mixing bowl"


@register_labeler
class GatherMarinadeIngredientsLabeler(BaseSubtaskLabeler):
    """Labeler for the GatherMarinadeIngredients composite task.

    Task (mobile): an oil/vinegar bottle and a shaker sit in an open cabinet; garlic sits in the
    open fridge; a mixing bowl sits on the counter. Set the bottle and the shaker beside the bowl
    and add the garlic into it. The robot starts at the cabinet.

    Doc-canonical count = 6 (three items, pick + place each). Each item, in first-motion order:
      (NavigateKitchen [navigate])                     -> navigate to {the cabinet|the fridge}  (adaptive)
      PickPlaceCabinetToCounter/FridgeToCounter [pick] -> pick up the {item} from the {cabinet|fridge}
      (NavigateKitchen [navigate])                     -> carry the {item} to the counter with the mixing bowl (adaptive)
      PickPlaceCabinetToCounter/FridgeToCounter [place]-> place the {item} {next to|in} the mixing bowl
    plus a terminal done. Navs are adaptive (emitted only when the base actually travels), so the
    bottle/shaker (local at the cabinet) usually have none and the garlic (across at the fridge) does.

    Boundaries (PackFoodByTemp machinery): t_pick = first sustained displacement; navs end at the
    base arrival; place = the gripper RELEASE.
    """

    ENV_NAME = "GatherMarinadeIngredients"

    # object -> (place phrase, subtask). The cabinet phrase ("the cabinet above" vs "the cabinet in
    # front") is resolved per episode in setup() from the cabinet's height; the fridge phrase is fixed.
    ITEMS = {
        "oil_or_vinegar_bottle": ("next to the mixing bowl", SUBTASK_CABINET),
        "shaker": ("next to the mixing bowl", SUBTASK_CABINET),
        "garlic": ("in the mixing bowl", SUBTASK_FRIDGE),
    }

    # a wall cabinet (mounted above the counter) sits high; a full-height / front cabinet sits low.
    CAB_ABOVE_Z = 1.80

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
        # "the cabinet above" for a high wall cabinet, "the cabinet in front" for a low full-height
        # cabinet (this varies with the layout, e.g. demo_10 is a front cabinet).
        try:
            cab_z = float(getattr(env, "cabinet").pos[2])
        except Exception:
            cab_z = self.CAB_ABOVE_Z
        cab_phrase = "the cabinet above" if cab_z >= self.CAB_ABOVE_Z else "the cabinet in front"
        # object -> (lead-in nav phrase, pick_from phrase, place phrase, subtask). The above/in-front
        # distinction describes the PICK location only; the return nav just says "the open cabinet".
        self.resolved = {}
        for o, (dest, sub) in self.ITEMS.items():
            if sub == SUBTASK_CABINET:
                self.resolved[o] = ("navigate back to the open cabinet", cab_phrase, dest, sub)
            else:
                self.resolved[o] = ("navigate to the fridge", "the fridge", dest, sub)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_FRIDGE}
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
        for o in order:
            nav_phrase, pick_from, dest, sub = self.resolved[o]
            word = self.word[o]
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

            # lead-in nav to the item's source, emitted only if the base actually travels there
            if base_xy is not None:
                src_park = np.median(base_xy[max(t_move - self.REST_W, 0): t_move + 1], axis=0)
                a = arrival(cursor, t_move, src_park)
                t_src = a if a is not None else cursor
                t_src = min(t_src, t_move - self.MIN_PICK_GAP)
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, nav_phrase)
            emit(t_move, sub, STAGE_PICK, f"pick up the {word} from {pick_from}")
            if base_xy is not None:
                dst_park = np.median(base_xy[max(t_placed - self.REST_W, 0): t_placed + 1], axis=0)
                a = arrival(t_move, t_placed, dst_park)
                t_arrive = a if a is not None else t_move
            else:
                t_arrive = t_move
            t_arrive = min(t_arrive, t_placed - self.MIN_PICK_GAP)
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to {BOWL_LM}")
            emit(t_placed, sub, STAGE_PLACE, f"place the {word} {dest}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")

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
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
