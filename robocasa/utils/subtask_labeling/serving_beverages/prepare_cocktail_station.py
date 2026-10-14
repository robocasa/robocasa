import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# A bowl sits on the dining counter. Grab a lemon wedge from the (closed) fridge and drop
# it in the bowl, then take a liquor bottle and a glass cup from the (open) cabinet and
# set them next to the bowl. OpenFridge and NavigateKitchen are bank atomics; fridge/
# cabinet -> dining table have no bank names -> PickPlaceFridgeToDining /
# PickPlaceCabinetToDining (extensions). The robot starts at the fridge.
SUBTASK_OPEN = "OpenFridge"
SUBTASK_NAV = "NavigateKitchen"
MOVE_FRIDGE = "PickPlaceFridgeToDining"
MOVE_CABINET = "PickPlaceCabinetToDining"

DONE_LANG = "task complete"

# object -> (source word, move-subtask name, place destination phrase, from_fridge)
SPEC = {
    "lemon_wedge": ("fridge", MOVE_FRIDGE, "in the bowl", True),
    "liquor": ("cabinet", MOVE_CABINET, "next to the bowl", False),
    "glass_cup": ("cabinet", MOVE_CABINET, "next to the bowl", False),
}


@register_labeler
class PrepareCocktailStationLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareCocktailStation composite task.

    Task (mobile): a bowl sits on the dining counter; a lemon wedge is in the CLOSED
    fridge (where the robot starts) and a liquor bottle + glass cup are in the OPEN
    cabinet. Put the lemon wedge in the bowl, then set the liquor and cup next to it.

    Doc-canonical count = 12 (`mobile: Yes`): OpenFridge + three moves, each `[nav] ->
    pick -> nav(carry) -> place`; the first (lemon, from the fridge) has no leading nav.
    ORDER by first move; each item is picked from its own source (fridge or cabinet).

    Boundaries: OpenFridge ends when the fridge door(s) reach their adaptive open level
    (before the lemon departs); per item t_pick = first sustained displacement re-anchored
    at the segment start; navs split off the base motion; t_place = the gripper RELEASE.
    """

    ENV_NAME = "PrepareCocktailStation"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    MIN_PICK_GAP = 15  # floor so a nav that runs into the grab/release doesn't collapse the pick/place
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    REST_W = 20
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in SPEC if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun, self.word = {}, {}
        for o in self.names:
            try:
                self.word[o] = env.get_obj_lang(o)
            except Exception:
                self.word[o] = o.replace("_", " ")
            self.noun[o] = f"the {self.word[o]}"
        fr = getattr(env, "fridge", None)
        self.door_adrs, self.door_scales = [], []
        for n in env.sim.model.joint_names:
            if not n or fr is None or fr.name not in n:
                continue
            low = n.lower()
            # include BOTH the fridge and freezer doors of the unit (a side-by-side
            # fridge opens two); the per-door "actually opened" filter in resolve()
            # keeps only the ones the robot really opened. Exclude drawers.
            if "door" not in low or "drawer" in low:
                continue
            try:
                adr = env.sim.model.get_joint_qpos_addr(n)
                jid = env.sim.model.joint_name2id(n)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {MOVE_FRIDGE, MOVE_CABINET}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": np.array([abs(float(env.sim.data.qpos[a])) / s
                               for a, s in zip(self.door_adrs, self.door_scales)]),
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
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
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
        fridge_picks = [first_move[o] for o in order if SPEC[o][3]]
        first_fridge_pick = min(fridge_picks, default=T)

        # --- OpenFridge boundary (before the first fridge item departs) ---
        t_open, n_doors = None, 0
        if doorv.shape[1] and first_fridge_pick < T:
            frames = []
            for i in range(doorv.shape[1]):
                col = doorv[:, i]
                seg = col[:first_fridge_pick]
                if seg.size == 0:
                    continue
                peak = float(seg.max())
                if peak < self.OPEN_MIN or peak - float(col[0]) < 0.10:
                    continue
                lvl = min(self.OPEN_ABS, 0.9 * peak)
                fo = self._sustained_start(col[:first_fridge_pick] >= lvl, 0, self.OPEN_SUSTAIN)
                if fo is not None:
                    frames.append(fo)
                    n_doors += 1
            if frames:
                t_open = min(max(frames), max(first_fridge_pick - 1, 1))

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0
        review = []
        open_emitted = False
        visited = set()

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # cabinet items in pick order -- the nav names the open cabinet by whatever is
        # still inside it at that trip (both on the first trip, the last one on the second).
        cabinet_order = [o for o in order if SPEC[o][0] == "cabinet"]

        def list_phrase(words):
            words = list(words)
            if len(words) == 1:
                return f"the {words[0]}"
            if len(words) == 2:
                return f"the {words[0]} and {words[1]}"
            return "the " + ", ".join(words[:-1]) + ", and " + words[-1]

        n_moved = 0
        for o in order:
            src_word, move_name, dest, from_fridge = SPEC[o]
            noun = self.noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            # leading nav back to this item's source (not for the very first item)
            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                t_back = min(t_back, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    if src_word == "cabinet":
                        remaining = [self.word[x] for x in cabinet_order
                                     if cabinet_order.index(x) >= cabinet_order.index(o)]
                        emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE,
                             f"navigate to the open cabinet with {list_phrase(remaining)}")
                    else:
                        word = "back to" if src_word in visited else "to"
                        emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, f"navigate {word} the {src_word}")

            # OpenFridge right before the first fridge item is grabbed
            if from_fridge and not open_emitted and t_open is not None and t_open > cursor:
                door_word = "doors" if n_doors >= 2 else "door"
                emit(min(t_open, max(t_pick - 1, cursor + 1)), SUBTASK_OPEN, STAGE_EXECUTE,
                     f"open the fridge {door_word}")
                open_emitted = True

            emit(t_pick, move_name, STAGE_PICK, f"pick up {noun} from the {src_word}")
            visited.add(src_word)

            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {noun} to the dining counter with the bowl")
            emit(t_placed, move_name, STAGE_PLACE, f"place {noun} {dest}")
            n_moved += 1

        if not open_emitted and doorv.shape[1]:
            review.append("fridge never detected opening -- verify")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "t_open": t_open, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {MOVE_FRIDGE, MOVE_CABINET}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenFridge, found {count(SUBTASK_OPEN)}")
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 places, found {int(np.sum(stages == STAGE_PLACE))}")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 5):
            issues.append(f"expected 0-5 NavigateKitchen, found {n_nav}")
        return issues
