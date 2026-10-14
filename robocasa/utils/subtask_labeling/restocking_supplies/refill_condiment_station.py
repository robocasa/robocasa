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


# Refill the dining-table condiment station: open the (closed) cabinet, take the
# condiment out, and carry it across the kitchen to the dining table. OpenCabinet and
# NavigateKitchen are bank atomics; cabinet -> dining table has no bank name ->
# PickPlaceCabinetToDining (extension). A cabinet distractor and a dining-table
# distractor stay put.
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToDining"
SUBTASK_NAV = "NavigateKitchen"

# The cabinet may have one door or two; the open-language pluralizes to match how
# many leaves actually swing open in the demo.
OPEN_LANG_ONE = "open the cabinet door"
OPEN_LANG_MANY = "open the cabinet doors"
DONE_LANG = "task complete"


@register_labeler
class RefillCondimentStationLabeler(BaseSubtaskLabeler):
    """Labeler for the RefillCondimentStation composite task.

    Task (mobile): a condiment sits in a CLOSED cabinet (where the robot starts); the
    dining table across the kitchen holds the same condiment as its station. Open the
    cabinet, take the condiment, and place it on the dining table.

    Doc-canonical count = 4 (`mobile: Yes`):
      OpenCabinet [execute]                -> open the cabinet
      PickPlaceCabinetToDining [pick]      -> pick up the {condiment} from the cabinet
      NavigateKitchen [navigate]           -> carry the {condiment} to the dining table
      PickPlaceCabinetToDining [place]     -> place the {condiment} on the dining table
    plus a terminal done.

    Boundaries: OpenCabinet ends when the summed door angle reaches OPEN_FRAC of its
    per-demo peak (capped before the condiment departs); t_pick = the condiment first
    displaces from its cabinet rest; the carry nav is split off the base motion; t_place
    = the gripper RELEASE on the dining table.
    """

    ENV_NAME = "RefillCondimentStation"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.30
    REST_W = 15
    DOOR_MIN_PEAK = 0.5
    OPEN_FRAC = 0.85
    OPEN_SUSTAIN = 5
    OPEN_DOOR_TH = 0.5  # a single leaf swinging past this (rad) counts as opened

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_id = env.obj_body_id["condiment"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cond_lang = env.get_obj_lang("condiment")
        except Exception:
            self.cond_lang = "condiment"
        # Cabinet door hinges (model joints under the cabinet fixture's name).
        cab = getattr(env, "cab", None)
        nm = getattr(cab, "name", None)
        self.door_addrs = []
        if nm is not None:
            for j in range(env.sim.model.njnt):
                try:
                    jn = env.sim.model.joint_id2name(j)
                except Exception:
                    jn = None
                if jn and jn.startswith(nm) and "doorhinge" in jn:
                    self.door_addrs.append(env.sim.model.jnt_qposadr[j])
        self.base_qpos_adrs = []
        for nm2 in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm2)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = (np.array([abs(float(env.sim.data.qpos[a])) for a in self.door_addrs])
                 if self.door_addrs else np.zeros(0))
        return {
            "obj": env.sim.data.body_xpos[self.obj_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "doors": doors,
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
        door = doorv.sum(axis=1) if doorv.shape[1] else np.zeros(T)
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
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_pick = int(mv[0]) if mv.size else max(T - 1, 1)
        hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                         t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # OpenCabinet: door reaches OPEN_FRAC of its peak, before the condiment departs.
        peak = float(door.max()) if T else 0.0
        has_open = peak >= self.DOOR_MIN_PEAK
        # pluralize "door" vs "doors" by how many leaves actually swung open
        n_opened = int(sum(float(doorv[:, i].max()) >= self.OPEN_DOOR_TH
                           for i in range(doorv.shape[1]))) if doorv.shape[1] else 0
        open_lang = OPEN_LANG_MANY if n_opened >= 2 else OPEN_LANG_ONE
        t_open = 0
        if has_open:
            lvl = self.OPEN_FRAC * peak
            fo = self._sustained_start(door[:max(t_pick, 1)] >= lvl, 0, self.OPEN_SUSTAIN)
            if fo is None:
                w = np.where(door >= lvl)[0]
                fo = int(w[0]) if w.size else 0
            t_open = int(min(fo, max(t_pick - 1, 0)))

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if has_open and t_open > 0:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, open_lang)
        elif not has_open:
            review.append("cabinet door never opened -- verify")

        emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.cond_lang} from the cabinet")
        carry_moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_pick + int(carry_moving[-1]) + 1, t_placed) if carry_moving.size else t_pick
        if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {self.cond_lang} to the dining table")
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.cond_lang} on the dining table")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_open": int(t_open), "door_peak": round(peak, 2), "t_pick": t_pick,
            "t_placed": t_placed, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        # setup() is skipped for manual overrides, so re-declare the extension name
        # before the bank check, then short-circuit the structural checks for them.
        self.extra_subtasks = {SUBTASK_MOVE}
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

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenCabinet, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 condiment pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 condiment place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
