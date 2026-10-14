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


# Open the (closed) fridge, take the lemon out, and set it on the cutting board that
# already holds the fish. OpenFridge and NavigateKitchen are bank atomics; fridge ->
# counter has no bank name -> PickPlaceFridgeToCounter (extension). The robot starts at
# the fridge and the cutting board is on the adjacent counter, so the task is usually
# local (no drive).
SUBTASK_OPEN = "OpenFridge"
SUBTASK_MOVE = "PickPlaceFridgeToCounter"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class LemonSeasoningFishLabeler(BaseSubtaskLabeler):
    """Labeler for the LemonSeasoningFish composite task.

    Task (mobile, usually local): a lemon sits in the CLOSED fridge (where the robot
    starts); a fish sits on a cutting board on the adjacent counter. Open the fridge,
    take the lemon, and place it on the cutting board with the fish.

    Doc-canonical count = 2 (`mobile: Yes`), action-level. Structure:
      OpenFridge [execute]              -> open the fridge door(s)
      PickPlaceFridgeToCounter [pick]   -> pick up the lemon from the fridge
      (NavigateKitchen [navigate])      -> carry the lemon to the cutting board (adaptive)
      PickPlaceFridgeToCounter [place]  -> place the lemon on the cutting board with the fish
    plus a terminal done. The robot spawns at the fridge, so there is no leading nav.

    Boundaries: OpenFridge ends when the fridge door(s) reach their adaptive open level
    (bounded before the lemon is grabbed); t_pick = the lemon's first sustained
    displacement; the carry nav (if any) ends at the last substantial base-motion frame;
    t_place = the gripper RELEASE.
    """

    ENV_NAME = "LemonSeasoningFish"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.25
    MIN_PICK_GAP = 12
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["lemon"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.lemon = env.get_obj_lang("lemon")
        except Exception:
            self.lemon = "lemon"
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
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "doors": np.array([abs(float(env.sim.data.qpos[a])) / s
                               for a, s in zip(self.door_adrs, self.door_scales)]),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
        }

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
        obj = np.array([h["obj"] for h in history])
        grip = np.array([h["grip"] for h in history])
        doorv = (np.array([h["doors"] for h in history])
                 if T and len(history[0]["doors"]) else np.zeros((T, 0)))
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        rest = np.median(obj[: self.REST_W], axis=0)
        disp = np.linalg.norm(obj - rest, axis=1)
        gd = np.linalg.norm(grip - obj, axis=1)
        t_pick = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_pick is None:
            t_pick = max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_place = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                        t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_place is None:
            t_place = T - 1

        segments = []
        cursor = 0
        review = []

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        # --- OpenFridge (doors open before the lemon is grabbed) ---
        t_open, n_doors = None, 0
        if doorv.shape[1]:
            frames = []
            for i in range(doorv.shape[1]):
                col = doorv[:, i]
                seg = col[:t_pick]
                if seg.size == 0:
                    continue
                peak = float(seg.max())
                if peak < self.OPEN_MIN or peak - float(col[0]) < 0.10:
                    continue
                lvl = min(self.OPEN_ABS, 0.9 * peak)
                fo = self._sustained_start(col[:t_pick] >= lvl, 0, self.OPEN_SUSTAIN)
                if fo is not None:
                    frames.append(fo)
                    n_doors += 1
            if frames:
                t_open = min(max(frames), max(t_pick - self.MIN_PICK_GAP, 1))
        if t_open is not None and t_open > cursor:
            door_word = "doors" if n_doors >= 2 else "door"
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the fridge {door_word}")
        elif doorv.shape[1]:
            review.append("fridge never detected opening -- verify")

        # --- lemon: fridge -> cutting board on the counter ---
        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.lemon} from the fridge")
        moving = np.where(binst[t_pick: t_place] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(moving[-1]) + 1, t_place) if moving.size else t_pick
        t_arrive = min(t_arrive, t_place - self.MIN_PICK_GAP)
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.lemon} to the cutting board")
        emit(t_place, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {self.lemon} on the cutting board with the fish")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": t_open, "n_doors": n_doors, "t_pick": int(t_pick),
            "t_place": int(t_place), "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
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
            issues.append(f"expected 1 lemon pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 lemon place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_OPEN) not in (0, 1):
            issues.append(f"expected 0-1 OpenFridge, found {count(SUBTASK_OPEN)}")
        return issues
