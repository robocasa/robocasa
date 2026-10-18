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


# Portion food from a plate into two tupperware containers: one cooked food and one vegetable in
# each. All on the dining counter. Plate -> tupperware (counter -> counter) has no bank name ->
# PickPlaceCounterToCounter (extension). Fixed base / local.
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

FOODS = ("cooked_food1", "cooked_food2", "vegetable1", "vegetable2")
TUPS = ("tupperware1", "tupperware2")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


@register_labeler
class PortionInTupperwareLabeler(BaseSubtaskLabeler):
    """Labeler for the PortionInTupperware composite task.

    Task (mobile, but local): a plate on the dining counter holds two cooked foods and two
    vegetables; two tupperware containers sit beside it. Move the items from the plate into the
    tupperwares (one cooked food and one vegetable per tupperware). The robot starts at the
    dining counter.

    Doc-canonical count = 4 (one move per item). Each item, in first-motion order:
      PickPlaceCounterToCounter [pick]  -> pick up {a|another} {item} from the plate
      (NavigateKitchen [navigate])      -> carry the {item} to the {left|right} tupperware  (adaptive)
      PickPlaceCounterToCounter [place] -> place the {item} in the {left|right} tupperware
    plus a terminal done. Items are named by type (get_obj_lang; repeats get "another"). The
    destination tupperware is resolved from which one the item ends up nearest to, named by its
    geometric placement in the robot base frame -- "the left/right tupperware" when the two are
    separated laterally (the usual case), or "the near/far tupperware" when they are arranged
    front-to-back (a few mobile demos where the robot drives between them).

    Boundaries: t_pick = first sustained displacement from a rest re-anchored at the segment start;
    a carry nav is emitted only if the base actually travels; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PortionInTupperware"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    LATERAL_MIN = 0.15   # |Δy| below this (and dominated by Δx) -> name tups near/far, not left/right

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in FOODS if o in env.obj_body_id]
        self.tups = [t for t in TUPS if t in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names + self.tups}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: env.get_obj_lang(o) for o in self.names}
        # Name each destination tupperware by its geometric placement in the robot
        # base frame at the demo's initial pose: (x_fwd, y_left). When the two are
        # separated laterally (the usual case) they are "the left/right tupperware"
        # (+y=left, -y=right); when they are arranged front-to-back (a few mobile
        # demos where the robot drives between a near and a far one) they are "the
        # near/far tupperware". env side fields are re-randomized on replay, so take
        # this from body geometry.
        bbid = env.sim.model.body_name2id("mobilebase0_base")
        bp = env.sim.data.body_xpos[bbid].copy()
        bm = env.sim.data.body_xmat[bbid].reshape(3, 3)
        tup_rel = {t: (bm.T @ (env.sim.data.body_xpos[self.obj_body_ids[t]].copy() - bp))[:2]
                   for t in self.tups}
        self.tup_side = {}
        if len(self.tups) == 2:
            a, b = self.tups
            dy = abs(float(tup_rel[a][1] - tup_rel[b][1]))
            dx = abs(float(tup_rel[a][0] - tup_rel[b][0]))
            if dy >= self.LATERAL_MIN or dy >= dx:
                self.tup_side[max(self.tups, key=lambda t: tup_rel[t][1])] = "the left tupperware"
                self.tup_side[min(self.tups, key=lambda t: tup_rel[t][1])] = "the right tupperware"
            else:
                self.tup_side[min(self.tups, key=lambda t: tup_rel[t][0])] = "the near tupperware"
                self.tup_side[max(self.tups, key=lambda t: tup_rel[t][0])] = "the far tupperware"
        elif len(self.tups) == 1:
            self.tup_side[self.tups[0]] = "the tupperware"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.names + self.tups:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.names + self.tups}
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

        n_moved = 0
        for o in order:
            word = self.word[o]
            m = seen.get(word, 0)
            seen[word] = m + 1
            pick_phrase = (f"{_article(word)} {word}" if m == 0 else f"another {word}")

            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            if self.tups:
                fpos = opos[o][min(t_placed, T - 1)]
                nearest = min(self.tups, key=lambda b: np.linalg.norm(fpos - opos[b][min(t_placed, T - 1)]))
                where = self.tup_side.get(nearest, "a tupperware")
            else:
                where = "a tupperware"

            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {pick_phrase} from the plate")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                        if base_xy is not None and t_arrive > t_pick else 0.0)
            if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {word} to {where}")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} in {where}")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 4:
            review.append(f"only {n_moved}/4 items moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        # PickPlaceCounterToCounter is a declared extension. Re-declare it here so a
        # manual override (which bypasses setup()) still passes the atomic-bank check,
        # and keep the manual short-circuit so a hand-reviewed demo skips the fixed
        # 4-pick/4-place count checks.
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 4:
            issues.append(f"expected 4 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 4:
            issues.append(f"expected 4 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
