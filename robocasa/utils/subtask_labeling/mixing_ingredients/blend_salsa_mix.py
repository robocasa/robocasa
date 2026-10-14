import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Open the blender lid, put the three vegetables (tomato, chili pepper, onion) in, close the
# lid, and start the blender. All bank atomics: OpenBlenderLid, PickPlaceCounterToBlender,
# CloseBlenderLid, TurnOnBlender. The base never drives (the plate is by the blender).
SUBTASK_OPEN = "OpenBlenderLid"
SUBTASK_MOVE = "PickPlaceCounterToBlender"
SUBTASK_CLOSE = "CloseBlenderLid"
SUBTASK_START = "TurnOnBlender"

DONE_LANG = "task complete"

VEG = ("tomato", "chili_pepper", "onion")


@register_labeler
class BlendSalsaMixLabeler(BaseSubtaskLabeler):
    """Labeler for the BlendSalsaMix composite task.

    Task: three vegetables sit on a plate by the blender (capped with a removable lid).
    Open the lid, put all three in the blender, close the lid, and start it. The robot
    starts at the blender; the base never drives.

    Doc-canonical count = 6 (`mobile: Yes`): OpenBlenderLid + 3 vegetable moves +
    CloseBlenderLid + TurnOnBlender. Emitted:
      OpenBlenderLid [execute]
      PickPlaceCounterToBlender [pick]/[place]  x3 (order varies, resolved from motion)
      CloseBlenderLid [execute]
      TurnOnBlender [execute]
    plus a terminal done.

    Detection (BlendIngredients machinery): the lid is a free-body auxiliary fixture; open
    ends when the lid is off (displacement >= LID_OFF_TH) and the gripper let go; each
    vegetable's pick/place runs while the lid is off (first-move / release, rest re-anchored
    at the segment start); close ends when the lid returns to its seat; start runs to the
    blender turning on (soft -- `turned_on` latches unreliably on replay) or the end.
    """

    ENV_NAME = "BlendSalsaMix"

    MOVE_TOL = 0.05      # motion-onset tol WITHIN a grasp window (the lift off the plate)
    MOVE_SUSTAIN = 3
    MOVE_TOL_STD = 0.15  # displacement that reliably marks a big move on its own
    MOVE_SUSTAIN_STD = 5
    GRASP_TH = 0.08      # gd below this == the veg is in hand. A veg starting near the blender
                         # translates < MOVE_TOL_STD (chili ~0.08) so grasp is the primary
                         # signal; kept tight so an adjacent veg (nudged, gd ~0.11 while the
                         # gripper grabs its neighbour) is not counted.
    GRASP_SUSTAIN = 4
    MIN_PICK_GAP = 8     # a place ends this far before the next grasp (keeps every pick visible)
    HOLD_W = 30
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    LID_OFF_TH = 0.30
    LID_CLOSED_TH = 0.12
    LID_RELEASE = 0.15
    LID_SUSTAIN = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = list(VEG)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.veg_lang = {}
        for o in self.names:
            try:
                self.veg_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.veg_lang[o] = o.replace("_", " ")
        self.blender = env.blender
        self.lid_body_id = None
        try:
            lid = env.blender.blender_lid
            if lid is not None:
                self.lid_body_id = env.sim.model.body_name2id(f"{lid.name}_main")
        except Exception:
            self.lid_body_id = None
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
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

    @staticmethod
    def _longest_run(mask, min_len):
        best, best_len, rs = None, 0, None
        for t in range(len(mask)):
            if mask[t]:
                if rs is None:
                    rs = t
            else:
                if rs is not None:
                    L = t - rs
                    if L >= min_len and L > best_len:
                        best_len, best = L, (rs, t)
                    rs = None
        if rs is not None:
            L = len(mask) - rs
            if L >= min_len and L > best_len:
                best = (rs, len(mask))
        return best

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        on = np.array([h.get("on", 0) for h in history])
        has_lid = "lid" in history[0]
        if has_lid:
            lid = np.array([h["lid"] for h in history])
            lid_disp = np.linalg.norm(lid - lid[0], axis=1)
            lid_gd = np.linalg.norm(grip - lid, axis=1)
        else:
            lid_disp = np.zeros(T)
            lid_gd = np.zeros(T)

        # --- open lid ---
        off = self._sustained_start(lid_disp >= self.LID_OFF_TH, 0, self.LID_SUSTAIN)
        if off is not None:
            rel = self._sustained_start(lid_gd > self.LID_RELEASE, off, self.LID_SUSTAIN)
            t_open = rel if rel is not None else off
        else:
            t_open = None
        anchor = t_open if t_open is not None else 0

        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}
        # HYBRID manipulation-start per veg (after the lid is off): earliest of a big
        # displacement OR a tight grasp. A veg starting near the blender barely translates
        # (chili ~0.08), so grasp is the primary signal; the grasp is kept tight (GRASP_TH)
        # so an adjacent veg nudged while its neighbour is grabbed is not mistaken for a move.
        manip = {}
        for o in self.names:
            rest = np.median(opos[o][max(anchor - self.REST_W, 0): anchor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            mask = np.zeros(T, dtype=bool)
            mask[anchor:] = True
            dfm = self._sustained_start((dsp >= self.MOVE_TOL_STD) & mask, anchor, self.MOVE_SUSTAIN_STD)
            grun = self._longest_run((gd[o] < self.GRASP_TH) & mask, self.GRASP_SUSTAIN)
            gst = grun[0] if grun is not None else None
            cands = [x for x in (dfm, gst) if x is not None]
            manip[o] = min(cands) if cands else None
        order = sorted([o for o in self.names if manip[o] is not None], key=lambda o: manip[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if t_open is not None:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, "open the blender lid")
        else:
            review.append("blender lid never detected off")

        n_moved = 0
        last_place = cursor
        for i, o in enumerate(order):
            word = self.veg_lang[o]
            g0 = max(int(manip[o]), cursor)
            win_end = int(manip[order[i + 1]]) if i + 1 < len(order) else T
            # t_pick = the veg's motion onset (lift off the plate) within its grasp window
            r = np.median(opos[o][max(g0 - self.REST_W, 0): g0 + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp[:win_end] >= self.MOVE_TOL, g0, self.MOVE_SUSTAIN)
            if t_pick is None:
                t_pick = g0
            t_pick = int(min(max(t_pick, cursor), win_end - 1))
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {word} from the counter")
            hold_gd = float(np.median(gd[o][t_pick: min(t_pick + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = win_end
            if i + 1 < len(order):
                t_placed = min(t_placed, win_end - self.MIN_PICK_GAP)
            t_placed = int(min(max(t_placed, t_pick + 1), T))
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {word} in the blender")
            last_place = t_placed
            n_moved += 1

        # --- close lid (after the last vegetable) ---
        if has_lid:
            cl = self._sustained_start(lid_disp < self.LID_CLOSED_TH, last_place + 1, self.LID_SUSTAIN)
            t_close = cl if cl is not None else T - 1
            t_close = max(t_close, cursor + 1)
            if t_close < T:
                emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, "close the blender lid")

        # --- start blender (soft) ---
        won = np.where(on[cursor:] > 0)[0]
        t_start = int(cursor + int(won[0])) if won.size else T - 1
        if t_start > cursor:
            emit(t_start, SUBTASK_START, STAGE_EXECUTE, "start the blender")

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if n_moved != 3:
            review.append(f"only {n_moved}/3 vegetables moved")

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "order": order, "n_moved": n_moved,
            "lid_disp_max": float(lid_disp.max()), "turned_on_frames": int(on.sum()),
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
        if int(np.sum(stages == STAGE_PICK)) != 3:
            issues.append(f"expected 3 vegetable picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 3:
            issues.append(f"expected 3 vegetable places, found {int(np.sum(stages == STAGE_PLACE))}")
        if int(np.sum(names == SUBTASK_START)) != 1:
            issues.append("expected 1 TurnOnBlender")
        return issues
