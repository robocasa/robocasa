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


# Open the (closed) fridge and set the butter on a plate on the dining table, then fetch the
# butter knife from the counter by the fridge and set it next to the plate. OpenFridge and
# NavigateKitchen are bank atomics; the two moves come from different sources ->
# PickPlaceFridgeToDining (butter) and PickPlaceCounterToDining (knife), both extensions.
# The robot starts at the fridge.
SUBTASK_OPEN = "OpenFridge"
MOVE_BUTTER = "PickPlaceFridgeToDining"
MOVE_KNIFE = "PickPlaceCounterToDining"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

# object -> (move name, pick lang, carry lang, place lang, nav-back lang). The butter noun
# is filled per-episode from get_obj_lang; the knife is always "the butter knife".
SPEC = {
    "butter": dict(
        move=MOVE_BUTTER,
        pick="pick up {noun} from the fridge",
        carry="carry {noun} to the dining counter",
        place="place {noun} on the plate",
        back="navigate back to the open fridge",
    ),
    "butter_knife": dict(
        move=MOVE_KNIFE,
        pick="pick up the butter knife from the counter by the fridge",
        carry="carry the butter knife to the plate on the dining counter",
        place="place the butter knife next to the plate",
        back="navigate back to the counter by the fridge",
    ),
}


@register_labeler
class SetupButterPlateLabeler(BaseSubtaskLabeler):
    """Labeler for the SetupButterPlate composite task.

    Task (mobile): butter sits in the CLOSED fridge (where the robot starts) and a butter
    knife sits on the counter by the fridge; a plate waits on the dining table. Open the
    fridge, put the butter on the plate, then set the knife next to the plate.

    Doc-canonical count = 8 (`mobile: Yes`): OpenFridge + butter `pick -> nav(carry) ->
    place` + knife `nav back -> pick -> nav(carry) -> place`. 8 subtasks + done. The two
    items come from DIFFERENT sources, so each carries its own source/nav-back lang. ORDER
    by first move (butter leads).

    Boundaries: OpenFridge ends when the opened door(s) reach their adaptive open level
    (bounded before the butter is grabbed); per item t_pick = first sustained displacement
    re-anchored at the segment start; navs split off the base motion (MIN_NAV_DIST); t_place
    = the gripper RELEASE.
    """

    ENV_NAME = "SetupButterPlate"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    MIN_PICK_GAP = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    REST_W = 15
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in SPEC if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = "the butter" if o == "butter" else "the butter knife"
        fr = getattr(env, "fridge", None)
        self.door_adrs, self.door_scales = [], []
        for n in env.sim.model.joint_names:
            if not n or fr is None or fr.name not in n:
                continue
            low = n.lower()
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
        self.extra_subtasks = {MOVE_BUTTER, MOVE_KNIFE}
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
        first_pick = min([first_move[o] for o in order], default=T)

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- OpenFridge (doors open before the butter is grabbed) ---
        t_open, n_doors = None, 0
        if doorv.shape[1]:
            frames = []
            for i in range(doorv.shape[1]):
                col = doorv[:, i]
                seg = col[:first_pick]
                if seg.size == 0:
                    continue
                peak = float(seg.max())
                if peak < self.OPEN_MIN or peak - float(col[0]) < 0.10:
                    continue
                lvl = min(self.OPEN_ABS, 0.9 * peak)
                fo = self._sustained_start(col[:first_pick] >= lvl, 0, self.OPEN_SUSTAIN)
                if fo is not None:
                    frames.append(fo)
                    n_doors += 1
            if frames:
                t_open = min(max(frames), max(first_pick - self.MIN_PICK_GAP, 1))
        if t_open is not None and t_open > cursor:
            door_word = "doors" if n_doors >= 2 else "door"
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the fridge {door_word}")
        elif doorv.shape[1]:
            review.append("fridge never detected opening -- verify")

        n_moved = 0
        for o in order:
            spec = SPEC[o]
            nn = self.noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)

            if n_moved >= 1:
                to_moving = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
                t_back = min(cursor + int(to_moving[-1]) + 1, t_pick) if to_moving.size else cursor
                t_back = min(t_back, max(t_pick - self.MIN_PICK_GAP, cursor))
                if t_back > cursor and base_disp(cursor, t_back) >= self.MIN_NAV_DIST:
                    emit(t_back, SUBTASK_NAV, STAGE_NAVIGATE, spec["back"])

            emit(t_pick, spec["move"], STAGE_PICK, spec["pick"].format(noun=nn))

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
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, spec["carry"].format(noun=nn))
            emit(t_placed, spec["move"], STAGE_PLACE, spec["place"].format(noun=nn))
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "t_open": t_open, "n_doors": n_doors,
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {MOVE_BUTTER, MOVE_KNIFE}
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
        if count(MOVE_BUTTER, STAGE_PICK) != 1 or count(MOVE_BUTTER, STAGE_PLACE) != 1:
            issues.append("expected 1 butter pick + 1 butter place")
        if count(MOVE_KNIFE, STAGE_PICK) != 1 or count(MOVE_KNIFE, STAGE_PLACE) != 1:
            issues.append("expected 1 knife pick + 1 knife place")
        n_nav = int(np.sum(names == SUBTASK_NAV))
        if not (0 <= n_nav <= 3):
            issues.append(f"expected 0-3 NavigateKitchen, found {n_nav}")
        return issues
