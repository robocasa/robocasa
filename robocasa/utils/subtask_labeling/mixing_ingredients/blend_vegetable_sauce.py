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


# Turn off the burner under the saucepan, carry the boiled vegetable from the saucepan to the
# blender, put the lid on, and start it. TurnOffStove + NavigateKitchen + CloseBlenderLid +
# TurnOnBlender are bank atomics; the saucepan(stove) -> blender move has no bank name ->
# PickPlaceStoveToBlender (extension). The blender lid starts OFF (beside the blender).
SUBTASK_OFF = "TurnOffStove"
SUBTASK_MOVE = "PickPlaceStoveToBlender"
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_CLOSE = "CloseBlenderLid"
SUBTASK_START = "TurnOnBlender"

DONE_LANG = "task complete"


@register_labeler
class BlendVegetableSauceLabeler(BaseSubtaskLabeler):
    """Labeler for the BlendVegetableSauce composite task.

    Task (mobile): a boiled vegetable sits in a saucepan on a LIT burner (robot starts at the
    stove); a blender is across the kitchen with its lid off beside it. Turn off the burner,
    carry the vegetable to the blender, put the lid on, and start it.

    Doc-canonical count = 6 (`mobile: Yes`): TurnOffStove + the vegetable's nav-interrupted
    stove->blender move (pick / nav / place) + CloseBlenderLid + TurnOnBlender. Structure:
      TurnOffStove [execute] -> veg[pick] -> NavigateKitchen(carry) -> veg[place]
      -> CloseBlenderLid -> TurnOnBlender -> done.
    The vegetable pick has NO lead-in nav (it is in the saucepan on the stove where the robot
    already stands).

    Boundaries: TurnOffStove ends when the target burner's knob leaves the "on" range (drops
    below OFF_FRAC of its peak), capped before the veg pick. t_pick = the veg first displaces
    from the saucepan; nav = the carry to the base stop; t_place = the gripper RELEASE.
    CloseBlenderLid = the lid reaches its ON displacement; TurnOnBlender = the blender on (soft)
    or the end.
    """

    ENV_NAME = "BlendVegetableSauce"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    GRASP_GATE = 0.15    # the veg pick only counts while the gripper holds it (gd < this) --
                         # the saucepan/veg gets jostled during the knob turn-off, which would
                         # otherwise fire a false pick ~1000 frames before the real carry.
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20
    OFF_FRAC = 0.30       # knob below this fraction of its peak == turned off
    OFF_SUSTAIN = 5
    LID_ON_TH = 0.30
    LID_SUSTAIN = 3
    LID_HOLD_W = 20
    LID_RELEASE_MARGIN = 0.08  # CloseBlenderLid ends at the gripper RELEASE of the lid, not
                          # when the lid first touches the blender (the gripper seats it first).

    _LOC_TXT = {
        "front_left": "front left", "front_right": "front right",
        "rear_left": "rear left", "rear_right": "rear right",
        "center": "center",
    }

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.veg_body_id = env.obj_body_id["vegetable"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.veg_lang = env.get_obj_lang("vegetable")
        except Exception:
            self.veg_lang = "vegetable"
        self.knob = (ep_meta.get("refs", {}) or {}).get("knob_id", getattr(env, "knob", None))
        self.knob_adr = None
        try:
            self.knob_adr = env.sim.model.get_joint_qpos_addr(
                f"{env.stove.naming_prefix}knob_{self.knob}_joint")
        except Exception:
            self.knob_adr = None
        self.loc_txt = self._LOC_TXT.get(self.knob, "stove")

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
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "veg": env.sim.data.body_xpos[self.veg_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "knob": abs(float(env.sim.data.qpos[self.knob_adr])) if self.knob_adr is not None else 0.0,
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
        veg = np.array([h["veg"] for h in history])
        grip = np.array([h["grip"] for h in history])
        knob = np.array([h["knob"] for h in history])
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

        # veg pick / carry / place
        rest = np.median(veg[: self.REST_W], axis=0)
        disp = np.linalg.norm(veg - rest, axis=1)
        gd = np.linalg.norm(grip - veg, axis=1)
        t_pick = self._sustained_start((disp >= self.MOVE_TOL) & (gd < self.GRASP_GATE), 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN, t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # TurnOffStove: the knob leaves its on range (before the veg pick)
        knob_peak = float(knob.max())
        has_off = knob_peak > 0.2 and self.knob_adr is not None
        if has_off:
            off = self._sustained_start(knob < self.OFF_FRAC * knob_peak, 0, self.OFF_SUSTAIN)
            t_off = off if off is not None else 0
            t_off = int(min(max(t_off, 1), max(t_pick - 1, 1)))
        else:
            t_off = 0

        # nav during the carry
        moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
        nav_disp = (float(np.max(np.linalg.norm(base_xy[t_pick: t_arrive] - base_xy[t_pick], axis=1)))
                    if base_xy is not None and t_arrive > t_pick else 0.0)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if has_off and t_off > 0:
            emit(t_off, SUBTASK_OFF, STAGE_EXECUTE, f"turn off the {self.loc_txt} burner")
        elif not has_off:
            review.append("stove knob never seen on -- verify TurnOffStove")

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.veg_lang} from the saucepan")
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.veg_lang} to the blender")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.veg_lang} in the blender")

        has_close = False
        if has_lid:
            lid_on = self._sustained_start(lid_disp >= self.LID_ON_TH, t_placed + 1, self.LID_SUSTAIN)
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

        result = self.build_result(T, segments, debug={
            "t_off": t_off, "knob_peak": round(knob_peak, 2), "t_pick": int(t_pick),
            "t_arrive": int(t_arrive), "t_placed": int(t_placed), "nav_disp": round(nav_disp, 2),
            "has_close": has_close, "n_segments": len(segments),
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

        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 veg pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 veg place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_OFF) not in (0, 1):
            issues.append(f"expected 0-1 TurnOffStove, found {count(SUBTASK_OFF)}")
        if count(SUBTASK_START) != 1:
            issues.append("expected 1 TurnOnBlender")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
