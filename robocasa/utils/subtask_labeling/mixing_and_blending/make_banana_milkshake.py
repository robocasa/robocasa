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


# Prep a banana milkshake: bring the honey from the (open) cabinet to next to the blender
# (local), then make two round trips to the (open) fridge -- one for the milk (placed next to
# the blender) and one for the banana (dropped in the blender jug). Cabinet -> counter is the
# bank atomic PickPlaceCabinetToCounter; fridge -> counter / fridge -> blender have no bank
# names -> extensions; NavigateKitchen is a bank atomic.
SUBTASK_HONEY = "PickPlaceCabinetToCounter"    # honey: cabinet -> next to blender
SUBTASK_MILK = "PickPlaceFridgeToCounter"      # milk:  fridge  -> next to blender
SUBTASK_BANANA = "PickPlaceFridgeToBlender"    # banana: fridge -> blender jug
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class MakeBananaMilkshakeLabeler(BaseSubtaskLabeler):
    """Labeler for the MakeBananaMilkshake composite task.

    Task (mobile): honey sits in an ALREADY-OPEN cabinet (where the robot starts); milk and a
    banana sit in an ALREADY-OPEN fridge across the kitchen; a blender sits by the cabinet.
    Put the honey next to the blender, then fetch the milk (next to the blender) and the
    banana (into the blender jug).

    Doc-canonical count = 11 (`mobile: Yes`): the honey's pick + carry + place (the carry-back
    nav fires when the blender is away from the cabinet, which is most layouts) + the milk's and
    banana's nav-interrupted fetches, each a round trip to the fridge (NavigateKitchen out /
    pick / NavigateKitchen back / place). Structure (milk/banana order varies):
      honey[pick] -> (NavigateKitchen carry) -> honey[place]
      -> NavigateKitchen(to fridge) -> milk[pick] -> NavigateKitchen(carry) -> milk[place]
      -> NavigateKitchen(to fridge) -> banana[pick] -> NavigateKitchen(carry) -> banana[place]
      -> done.
    The honey has NO lead-in nav (it is the leading object; its initial positioning is not a
    subtask, k>=1 gate).

    Boundaries (SweetenCoffee machinery, applied per object): t_move = the object first
    displaces MOVE_TOL from a rest re-anchored at the segment start; t_placed = the gripper
    RELEASE. The nav OUT to the fridge is the base drive from `cursor` to the grasp (emitted
    only for the milk/banana fetches, k>=1 -- the honey is the leading object so its initial
    positioning is not a subtask); the nav BACK is the base drive during the carry. Each nav is
    emitted only when the base's peak displacement over that leg >= MIN_NAV_DIST.
    """

    ENV_NAME = "MakeBananaMilkshake"

    # object -> (source phrase, dest phrase, subtask name)
    ITEMS = {
        "honey": ("the cabinet", "next to the blender", SUBTASK_HONEY),
        "milk": ("the fridge", "next to the blender", SUBTASK_MILK),
        "banana": ("the fridge", "in the blender jug", SUBTASK_BANANA),
    }

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.008  # per-frame base speed above this == real driving. Sits between the
                          # post-arrival placing-creep (<=~0.003) which must be EXCLUDED so a nav
                          # ends at the true arrival, and the slow final approach to a fixture
                          # (~0.014-0.025) which must be INCLUDED. A nav ends at the LAST such
                          # frame, so it also survives a brief mid-drive pause (the base resumes
                          # and drives on to the real arrival).
    MIN_NAV_DIST = 0.20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = list(self.ITEMS.keys())
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {}
        for o in self.names:
            try:
                self.word[o] = env.get_obj_lang(o)
            except Exception:
                self.word[o] = o

        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MILK, SUBTASK_BANANA}
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
            return (
                float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                if base_xy is not None and b > a else 0.0
            )

        n_moved = 0
        for k, o in enumerate(order):
            src, dst, sub = self.ITEMS[o]
            word = self.word[o]
            # re-anchor rest at the segment start (an item can drift as the base drives)
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

            # nav OUT to the source (only for non-leading fetches): the base drive cursor ->
            # grasp, ending at the LAST substantial-motion frame (the arrival at the fridge;
            # survives a brief mid-drive pause since we take the last such frame).
            if k >= 1:
                to_moving = np.where(binst[cursor: t_move] > self.BASE_MOVE_TH)[0]
                t_src = min(cursor + int(to_moving[-1]) + 1, t_move) if to_moving.size else cursor
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate to {src}")

            emit(t_move, sub, STAGE_PICK, f"pick up the {word} from {src}")

            # nav BACK to the blender: the base drive during the carry, ending at the last
            # substantial-motion frame -- the placing-creep (below BASE_MOVE_TH) is excluded,
            # so the lift-into-the-jug/onto-the-counter that follows is part of the place.
            back_moving = np.where(binst[t_move: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_move + int(back_moving[-1]) + 1, t_placed) if back_moving.size else t_move
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to the blender")

            emit(t_placed, sub, STAGE_PLACE, f"place the {word} {dst}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved (honey, milk, banana)")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved,
            "first_move": {o: int(first_move[o]) for o in self.names},
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
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        for nm, label in ((SUBTASK_HONEY, "honey"), (SUBTASK_MILK, "milk"), (SUBTASK_BANANA, "banana")):
            if count(nm, STAGE_PICK) != 1:
                issues.append(f"expected 1 {label} pick, found {count(nm, STAGE_PICK)}")
            if count(nm, STAGE_PLACE) != 1:
                issues.append(f"expected 1 {label} place, found {count(nm, STAGE_PLACE)}")
        # up to 5 navs: honey's carry-back (its lead-in nav is gated off) + a round trip
        # (out + back) for each of milk and banana.
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 6):
            issues.append(f"expected 0-6 NavigateKitchen, found {n_nav}")
        return issues
