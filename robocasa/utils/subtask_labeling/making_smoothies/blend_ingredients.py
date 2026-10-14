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

# Open the blender lid, put the fruit in, close the lid, and start the blender. All four
# are bank atomics: OpenBlenderLid, PickPlaceCounterToBlender, CloseBlenderLid, and
# TurnOnBlender ("Turn on the blender by pressing the power button"). The base never
# drives (fruit is on the counter right by the blender).
SUBTASK_OPEN = "OpenBlenderLid"
SUBTASK_MOVE = "PickPlaceCounterToBlender"
SUBTASK_CLOSE = "CloseBlenderLid"
SUBTASK_START = "TurnOnBlender"

DONE_LANG = "task complete"


@register_labeler
class BlendIngredientsLabeler(BaseSubtaskLabeler):
    """Labeler for the BlendIngredients composite task.

    Task: a fruit sits on the counter by the blender (capped with a removable lid).
    Open the blender lid, put the fruit in, close the lid, and start the blender. The
    robot starts at the blender; the base never drives.

    Doc-canonical decomposition (open lid, fruit move, close lid, start). Emitted:
      OpenBlenderLid [execute]          -> open the blender lid
      PickPlaceCounterToBlender [pick]  -> pick up the {fruit} from the counter
      PickPlaceCounterToBlender [place] -> place the {fruit} in the blender
      CloseBlenderLid [execute]         -> close the blender lid
      StartBlender [execute]            -> start the blender
    plus a terminal done.

    Detection: the lid is a free-body auxiliary fixture. Open ends when the lid is off
    (displacement ≥ LID_OFF_TH) and the gripper has let go; the fruit pick/place run
    while the lid is off; close ends when the lid returns to its seat (displacement <
    LID_CLOSED_TH); the start runs to the blender turning on (or the end of the demo --
    `turned_on` latches unreliably on frame-by-frame replay, so it is a soft signal).
    """

    ENV_NAME = "BlendIngredients"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 30
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    REST_W = 15
    LID_OFF_TH = 0.30       # lid displacement (m) counting as "off the blender"
    LID_CLOSED_TH = 0.12    # lid displacement (m) counting as "back on the blender"
    LID_RELEASE = 0.15      # grip<->lid dist above which the gripper has let go of the lid
    LID_SUSTAIN = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.fruit_body_id = env.obj_body_id["fruit"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fruit_lang = env.get_obj_lang("fruit")
        except Exception:
            self.fruit_lang = "fruit"
        self.blender = env.blender
        self.lid_body_id = None
        try:
            lid = env.blender.blender_lid
            if lid is not None:
                self.lid_body_id = env.sim.model.body_name2id(f"{lid.name}_main")
        except Exception:
            self.lid_body_id = None
        self.extra_subtasks = {SUBTASK_OPEN, SUBTASK_MOVE, SUBTASK_CLOSE, SUBTASK_START}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "fruit": env.sim.data.body_xpos[self.fruit_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }
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
        fruit = np.array([h["fruit"] for h in history])
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

        # --- open lid: lid off + gripper let go ---
        off = self._sustained_start(lid_disp >= self.LID_OFF_TH, 0, self.LID_SUSTAIN)
        if off is not None:
            rel = self._sustained_start(lid_gd > self.LID_RELEASE, off, self.LID_SUSTAIN)
            t_open = rel if rel is not None else off
        else:
            t_open = None

        anchor = t_open if t_open is not None else 0

        # --- fruit pick/place (after the lid is off) ---
        rest = np.median(fruit[max(anchor - self.REST_W, 0): anchor + 1], axis=0)
        fdisp = np.linalg.norm(fruit - rest, axis=1)
        fgd = np.linalg.norm(grip - fruit, axis=1)
        t_pick = self._sustained_start(fdisp >= self.MOVE_TOL, anchor, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = min(anchor + 1, T - 1)
        hold_gd = float(np.median(fgd[t_pick: min(t_pick + self.HOLD_W, T)]))
        released = fgd > hold_gd + self.RELEASE_MARGIN
        t_place = self._sustained_start(released, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        # --- close lid: lid returns to its seat (after fruit placed) ---
        if has_lid:
            cl = self._sustained_start(lid_disp < self.LID_CLOSED_TH, t_place + 1, self.LID_SUSTAIN)
            t_close = cl if cl is not None else T - 1
        else:
            t_close = t_place + 1
        t_close = max(t_close, t_place + 1)

        # --- start: blender turns on (soft) else run to end ---
        won = np.where(on[t_close:] > 0)[0]
        t_start = int(t_close + int(won[0])) if won.size else T - 1
        t_start = max(t_start, t_close + 1)

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
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.fruit_lang} from the counter")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.fruit_lang} in the blender")
        if has_lid:
            emit(t_close, SUBTASK_CLOSE, STAGE_EXECUTE, "close the blender lid")
        emit(t_start, SUBTASK_START, STAGE_EXECUTE, "start the blender")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "t_pick": int(t_pick), "t_place": int(t_place),
            "t_close": int(t_close), "t_start": int(t_start),
            "lid_disp_max": float(lid_disp.max()), "turned_on_frames": int(on.sum()),
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues  # human-reviewed override: skip the structural checks
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 fruit pick")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 fruit place")
        if int(np.sum(names == SUBTASK_START)) != 1:
            issues.append("expected 1 StartBlender")
        if result.debug.get("t_open") is not None and int(np.sum(names == SUBTASK_OPEN)) != 1:
            issues.append("expected 1 OpenBlenderLid")
        return issues
