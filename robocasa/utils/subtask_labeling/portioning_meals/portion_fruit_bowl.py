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


# Portion four fruits from a plate into two bowls (two fruits per bowl), all on the dining counter.
# Plate -> bowl (both on the counter) has no bank name -> PickPlaceCounterToCounter (extension).
# Fixed base / local (plate and bowls sit side by side).
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

FRUITS = ("fruit1", "fruit2", "fruit3", "fruit4")
BOWLS = ("bowl1", "bowl2")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


@register_labeler
class PortionFruitBowlLabeler(BaseSubtaskLabeler):
    """Labeler for the PortionFruitBowl composite task.

    Task (mobile, but local): a plate on the dining counter holds four fruits; two bowls sit
    beside it. Move the fruits from the plate into the bowls, two per bowl. The robot starts at
    the dining counter.

    Doc-canonical count = 4 (one move per fruit). Each fruit, in first-motion order:
      PickPlaceCounterToCounter [pick]  -> pick up {a|another} {fruit} from the plate
      (NavigateKitchen [navigate])      -> carry the {fruit} to {a bowl|the other bowl}  (adaptive)
      PickPlaceCounterToCounter [place] -> place the {fruit} in {a bowl|the other bowl}
    plus a terminal done. Fruits are named by type (get_obj_lang; repeats get "another"). The
    destination bowl is resolved from which bowl the fruit ends up nearest to; the first bowl used
    is "a bowl", the other "the other bowl".

    Boundaries: t_pick = first sustained displacement from a rest re-anchored at the segment start;
    a carry nav is emitted only if the base actually travels; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PortionFruitBowl"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in FRUITS if o in env.obj_body_id]
        self.bowls = [b for b in BOWLS if b in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names + self.bowls}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {o: env.get_obj_lang(o) for o in self.names}
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
        for o in self.names + self.bowls:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.names + self.bowls}
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
        bowl_phrase = {}  # bowl name -> "a bowl" / "the other bowl" (by first use)

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

            # which bowl did this fruit end up in? (nearest bowl at release)
            if self.bowls:
                fpos = opos[o][min(t_placed, T - 1)]
                nearest = min(self.bowls, key=lambda b: np.linalg.norm(fpos - opos[b][min(t_placed, T - 1)]))
                if nearest not in bowl_phrase:
                    bowl_phrase[nearest] = "a bowl" if len(bowl_phrase) == 0 else "the other bowl"
                where = bowl_phrase[nearest]
            else:
                where = "a bowl"

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
            review.append(f"only {n_moved}/4 fruits moved")

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
        if int(np.sum(stages == STAGE_PICK)) != 4:
            issues.append(f"expected 4 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 4:
            issues.append(f"expected 4 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
