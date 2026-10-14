import numpy as np

import robocasa.utils.object_utils as OU
from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Sort foods into two bowls: the squashes (from the cabinet) into one bowl, the carrot (from
# the sink) into the other. Cabinet/sink -> bowl have no bank names -> PickPlaceCabinetToBowl /
# PickPlaceSinkToBowl (extensions); NavigateKitchen is bank. Mobile. Doc count = 8.
SUBTASK_CAB = "PickPlaceCabinetToBowl"
SUBTASK_SINK = "PickPlaceSinkToBowl"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

FOODS = ("food1", "food2", "food3")
BOWLS = ("bowl1", "bowl2")


@register_labeler
class PrewashFoodSortingLabeler(BaseSubtaskLabeler):
    """Labeler for the PrewashFoodSorting composite task.

    Task (mobile): move each food from its source (the cabinet for the squashes, the sink for
    the carrot) into a bowl, sorting by type (same-type foods share a bowl). Doc count = 8.
    Order by first grasped move; each food's source is detected (sink basin vs cabinet); the
    destination bowl is detected (nearest bowl at the place) and labeled "one bowl" / "the
    other bowl" / "the same bowl as the other {food}".
    """

    ENV_NAME = "PrewashFoodSorting"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    GRASP_TH = 0.07
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.foods = [o for o in FOODS if o in env.obj_body_id]
        self.bowls = [o for o in BOWLS if o in env.obj_body_id]
        self.body = {o: env.obj_body_id[o] for o in self.foods + self.bowls}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.base = {}
        self.src = {}
        for o in self.foods:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = "item"
            # source = the sink if the food sits in the sink at the start, else the cabinet.
            # (The sink's basin-geom lookup is unavailable on these sink models, so detect by
            # membership instead -- matches the ep_meta lang "... from the sink/cabinet".)
            in_sink = False
            if self.sink is not None:
                try:
                    in_sink = bool(OU.obj_inside_of(env, o, self.sink))
                except Exception:
                    in_sink = False
            self.src[o] = "the sink" if in_sink else "the cabinet"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {SUBTASK_CAB, SUBTASK_SINK}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
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

    @staticmethod
    def _last_sustained_end(mask, sustain):
        last_end, run = None, 0
        for i, v in enumerate(mask):
            if v:
                run += 1
            else:
                if run >= sustain:
                    last_end = i
                run = 0
        if run >= sustain:
            last_end = len(mask)
        return last_end

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.foods + self.bowls}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.foods}

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        first_move = {}
        for o in self.foods:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), 0, self.MOVE_SUSTAIN)
            if fm is None:
                fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.foods if first_move[o] < T], key=lambda o: first_move[o])

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
                art_noun[o] = f"{art} {w}" if seen[w] == 1 else f"the other {w}"
                the_noun[o] = f"the {w}"

        used, bowl_contents = [], {}

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

        visited = set()
        n = 0
        for k, o in enumerate(order):
            w = self.base[o]
            src = self.src[o]
            bank = SUBTASK_SINK if src == "the sink" else SUBTASK_CAB
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start((dsp >= self.MOVE_TOL) & (gd[o] < self.GRASP_TH), cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd[o] > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # return nav to this food's source before picking it (k>=1)
            if k >= 1:
                end = self._last_sustained_end(binst[cursor: t_pick] > self.BASE_MOVE_TH, self.MOVE_SUSTAIN)
                t_appr = cursor + end if end is not None else cursor
                t_appr = min(t_appr, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_appr > cursor and base_disp(cursor, t_appr) >= self.MIN_NAV_DIST:
                    # the squashes come from a cabinet that is open from the start, so name it
                    # "the open cabinet"; say "back" when this source was already visited earlier.
                    loc = "the open cabinet" if src == "the cabinet" else src
                    verb = "navigate back to" if src in visited else "navigate to"
                    emit(t_appr, SUBTASK_NAV, STAGE_NAVIGATE, f"{verb} {loc}")
            emit(t_pick, bank, STAGE_PICK, f"pick up {art_noun[o]} from {src}")
            # dest bowl = nearest bowl to the food at the place
            fp = opos[o][min(t_placed, T - 1)][:2]
            if self.bowls:
                bname = min(self.bowls, key=lambda b: float(np.linalg.norm(opos[b][min(t_placed, T - 1)][:2] - fp)))
                dest = bowl_phrase(bname, w)
            else:
                dest = "the bowl"
            # the carry-nav names the destination bowl: a freshly-introduced bowl is located by
            # the sink ("a bowl next to the sink"); a back-reference ("the same bowl as ...") is
            # already unambiguous, so it takes no location.
            carry_dest = dest if dest.startswith("the same bowl") else f"{dest} next to the sink"
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            carried = t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST
            if carried:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {the_noun[o]} to {carry_dest}")
            # if the food was carried to a named bowl, the place just drops it in "the bowl";
            # with no carry-nav the place still names which bowl.
            emit(t_placed, bank, STAGE_PLACE, f"place {the_noun[o]} in {'the bowl' if carried else dest}")
            visited.add(src)
            n += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n != len(self.foods):
            review.append(f"only {n}/{len(self.foods)} foods moved")
        result = self.build_result(T, segments, debug={"order": order, "n": n,
                                                       "src": self.src, "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_CAB, SUBTASK_SINK}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        npick = int(np.sum(stages == STAGE_PICK))
        nplace = int(np.sum(stages == STAGE_PLACE))
        if npick != len(self.foods):
            issues.append(f"expected {len(self.foods)} picks, found {npick}")
        if nplace != len(self.foods):
            issues.append(f"expected {len(self.foods)} places, found {nplace}")
        return issues
