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


# Blend a marinade: bring two vegetables from the (open) fridge to the blender, put the lid on, and
# start it. Each fridge -> blender fetch is nav-interrupted (a round trip) -> PickPlaceFridgeToBlender
# (extension); NavigateKitchen + CloseBlenderLid + TurnOnBlender are bank atomics. The blender lid
# starts OFF beside the blender and is placed ON. (Clone of PrepareVeggieDip with two veg.)
SUBTASK_VEG = "PickPlaceFridgeToBlender"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseBlenderLid"
SUBTASK_START = "TurnOnBlender"

DONE_LANG = "task complete"


@register_labeler
class BlendMarinadeLabeler(BaseSubtaskLabeler):
    """Labeler for the BlendMarinade composite task.

    Task (mobile): two marinade vegetables sit in the (already-open) fridge (where the robot
    starts); a blender sits across the kitchen with its lid off beside it. Bring both to the
    blender, put the lid on, and start it.

    Doc-canonical count = 9 (`mobile: Yes`): the two vegetables' nav-interrupted fridge->blender
    fetches (pick / nav / place; the second also has a lead-in nav back to the fridge) +
    CloseBlenderLid + TurnOnBlender. Structure (order resolved from motion):
      veg[pick] -> NavigateKitchen(carry) -> veg[place]
      -> NavigateKitchen(to fridge) -> veg[pick] -> NavigateKitchen(carry) -> veg[place]
      -> CloseBlenderLid -> TurnOnBlender -> done.
    The first vegetable has no lead-in nav (the robot starts at the fridge; k>=1 gate). The two are
    named by type (get_obj_lang); if the same type they are disambiguated a/the-other.

    Boundaries (PrepareVeggieDip machinery): t_pick = first sustained displacement; navs from base
    motion; place = the gripper RELEASE; CloseBlenderLid = the lid reaches its ON displacement then
    the gripper releases it; TurnOnBlender = the blender turning on.
    """

    ENV_NAME = "BlendMarinade"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20
    LID_ON_TH = 0.25
    LID_SUSTAIN = 3
    LID_HOLD_W = 20
    LID_RELEASE_MARGIN = 0.08

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in ("marinade_veg1", "marinade_veg2") if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.word = {}
        for o in self.names:
            try:
                self.word[o] = env.get_obj_lang(o)
            except Exception:
                self.word[o] = "vegetable"
        from collections import Counter
        counts = Counter(self.word.values())
        self.same_type = {o: counts[self.word[o]] > 1 for o in self.names}
        self.blender = env.blender
        self.lid_body_id = None
        try:
            lid = env.blender.blender_lid
            if lid is not None:
                self.lid_body_id = env.sim.model.body_name2id(f"{lid.name}_main")
        except Exception:
            self.lid_body_id = None
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_VEG}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        if self.lid_body_id is not None:
            d["lid"] = env.sim.data.body_xpos[self.lid_body_id].copy()
        try:
            d["on"] = int(bool(self.blender.get_state().get("turned_on", False)))
        except Exception:
            d["on"] = 0
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
        on = np.array([h.get("on", 0) for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        has_lid = "lid" in history[0]
        if has_lid:
            lid_arr = np.array([h["lid"] for h in history])
            lid_disp = np.linalg.norm(lid_arr - history[0]["lid"], axis=1)
            lid_gd = np.linalg.norm(grip - lid_arr, axis=1)
        else:
            lid_disp = np.zeros(T)
            lid_gd = np.zeros(T)

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

        n_moved = 0
        last_place = 0
        for k, o in enumerate(order):
            base_word = self.word[o]
            m = seen.get(base_word, 0)
            seen[base_word] = m + 1
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

            if k >= 1:
                to_moving = np.where(binst[cursor: t_move] > self.BASE_MOVE_TH)[0]
                t_src = min(cursor + int(to_moving[-1]) + 1, t_move) if to_moving.size else cursor
                if t_src > cursor and base_disp(cursor, t_src) >= self.MIN_NAV_DIST:
                    emit(t_src, SUBTASK_NAV, STAGE_NAVIGATE, "navigate back to the fridge")
            emit(t_move, SUBTASK_VEG, STAGE_PICK, f"pick up {pick_art} {base_word} from the fridge")
            back_moving = np.where(binst[t_move: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_move + int(back_moving[-1]) + 1, t_placed) if back_moving.size else t_move
            if t_arrive > cursor and base_disp(t_move, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {base_word} to the blender")
            emit(t_placed, SUBTASK_VEG, STAGE_PLACE, f"place {place_art} {base_word} in the blender")
            last_place = t_placed
            n_moved += 1

        review = []
        has_close = False
        if has_lid:
            lid_on = self._sustained_start(lid_disp >= self.LID_ON_TH, last_place + 1, self.LID_SUSTAIN)
            if lid_on is not None:
                hold = float(np.median(lid_gd[lid_on: min(lid_on + self.LID_HOLD_W, T)]))
                rel = self._sustained_start(lid_gd > hold + self.LID_RELEASE_MARGIN, lid_on, self.LID_SUSTAIN)
                t_close = rel if rel is not None else lid_on
                if t_close > cursor:
                    emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, "close the blender lid")
                    has_close = True
        if not has_close:
            review.append("blender lid never placed on -- verify")

        won = np.where(on[cursor:] > 0)[0]
        t_start = int(cursor + int(won[0])) if won.size else T - 1
        if t_start > cursor:
            emit(t_start, SUBTASK_START, STAGE_EXECUTE, "start the blender")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 vegetables moved")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "has_close": has_close,
            "lid_disp_max": float(lid_disp.max()), "n_segments": len(segments),
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

        if count(SUBTASK_VEG, STAGE_PICK) != 2:
            issues.append(f"expected 2 picks, found {count(SUBTASK_VEG, STAGE_PICK)}")
        if count(SUBTASK_VEG, STAGE_PLACE) != 2:
            issues.append(f"expected 2 places, found {count(SUBTASK_VEG, STAGE_PLACE)}")
        if count(SUBTASK_START) != 1:
            issues.append("expected 1 TurnOnBlender")
        n_nav = count(SUBTASK_NAV)
        if not (0 <= n_nav <= 4):
            issues.append(f"expected 0-4 NavigateKitchen, found {n_nav}")
        return issues
