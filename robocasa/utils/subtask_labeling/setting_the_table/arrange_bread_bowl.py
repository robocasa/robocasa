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


# Pull the toaster oven rack out, take the heated bread off it and drop it in the bowl on
# the counter, then carry the bread bowl to the dining counter. SlideToasterOvenRack and
# NavigateKitchen are bank atomics; toaster oven -> bowl and counter -> dining table have
# no bank names -> PickPlaceToasterOvenToBowl / PickPlaceCounterToDining (extensions).
SUBTASK_RACK = "SlideToasterOvenRack"
MOVE_BREAD = "PickPlaceToasterOvenToBowl"
MOVE_BOWL = "PickPlaceCounterToDining"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"


@register_labeler
class ArrangeBreadBowlLabeler(BaseSubtaskLabeler):
    """Labeler for the ArrangeBreadBowl composite task.

    Task (mobile): a heated bread sits in the OPEN toaster oven (where the robot starts);
    a bowl (already holding a bread) sits on the adjacent counter. Put the heated bread
    in the bowl, then carry the bread bowl to the dining counter.

    Doc-canonical count = 5 (`mobile: Yes`), action-level:
      PickPlaceToasterOvenToBowl [pick]  -> pick up the {bread} from the toaster oven
      PickPlaceToasterOvenToBowl [place] -> place the {bread} in the bowl
      PickPlaceCounterToDining [pick]    -> pick up the bread bowl from the counter
      NavigateKitchen [navigate]         -> carry the bread bowl to the dining counter
      PickPlaceCounterToDining [place]   -> place the bread bowl on the dining counter
    plus a terminal done.

    Boundaries: the bread's t_pick is its first move out of the toaster oven; the bowl's
    rest is re-anchored once the bread is dropped in (it sits still until the robot lifts
    it), so its t_pick is the moment it leaves the counter; the carry is split off the
    base motion; places = the gripper RELEASE.
    """

    ENV_NAME = "ArrangeBreadBowl"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    REST_W = 15
    SETTLE_TH = 0.004   # m/frame; below this the moved object is at rest
    SETTLE_W = 12       # sustained rest frames that mark the set-down
    RACK_MIN_TRAVEL = 0.05    # min rack extension (m) for a real pull-out
    RACK_MOVE_SPEED = 0.0015  # per-frame rack extension speed that counts as "moving"
    RACK_SUSTAIN = 5
    RACK_SETTLE_W = 8         # sustained still frames = the rack came to rest (pull done)

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.bread_id = env.obj_body_id["toaster_oven_bread"]
        self.bowl_id = env.obj_body_id["bowl"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # the toaster oven's slide-out rack joints (the bread rides out on one). A
        # multi-level toaster oven has rack0/rack1 -- collect ALL of them; resolve() picks
        # the one the robot actually pulls (the joint with the most travel).
        to = getattr(env, "toaster_oven", None)
        prefix = getattr(to, "naming_prefix", None)
        self.rack_adrs = []
        if prefix:
            for n in env.sim.model.joint_names:
                if n and n.startswith(prefix) and "rack" in n and n.endswith("_joint"):
                    try:
                        self.rack_adrs.append(env.sim.model.get_joint_qpos_addr(n))
                    except Exception:
                        pass
        try:
            self.bread = env.get_obj_lang("toaster_oven_bread")
        except Exception:
            self.bread = "bread"
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        self.extra_subtasks = {MOVE_BREAD, MOVE_BOWL}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "bread": env.sim.data.body_xpos[self.bread_id].copy(),
            "bowl": env.sim.data.body_xpos[self.bowl_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "racks": (np.array([float(env.sim.data.qpos[a]) for a in self.rack_adrs])
                      if self.rack_adrs else np.zeros(0)),
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

    def _release(self, gd, t_pick, T):
        hold = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
        t = self._sustained_start(gd > hold + self.RELEASE_MARGIN,
                                  t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        return t if t is not None else T - 1

    def _place_by_settle(self, pos, t_pick, T):
        # the bread is "placed" when it first comes to sustained rest in the bowl --
        # robust to the gripper lingering (reaching for the bowl the bread now sits in).
        speed = np.concatenate([np.linalg.norm(np.diff(pos, axis=0), axis=1), [0.0]])
        run = 0
        for t in range(int(min(t_pick + self.MIN_CARRY, T - 1)), T):
            if speed[t] <= self.SETTLE_TH:
                run += 1
                if run >= self.SETTLE_W:
                    return t - self.SETTLE_W + 1
            else:
                run = 0
        return T - 1

    def resolve(self, history):
        T = len(history)
        bread = np.array([h["bread"] for h in history])
        bowl = np.array([h["bowl"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # --- SlideToasterOvenRack: the robot pulls the rack out (the bread rides out on
        # it). The rack-out frame is where the rack reaches its extended plateau. ---
        racks = (np.abs(np.array([h["racks"] for h in history]))
                 if T and len(history[0]["racks"]) else np.zeros((T, 0)))
        # the rack the robot actually pulled = the joint with the most travel
        rack = racks[:, int(np.argmax(racks.max(axis=0) - racks.min(axis=0)))] if racks.shape[1] else np.zeros(T)
        anchor = 0
        if rack.size and float(rack.max()) >= self.RACK_MIN_TRAVEL:
            # end of the FIRST pull-out: the rack starts moving, then first comes to rest.
            # (the robot may hold it there while grabbing the bread and drag it the rest of
            # the way as it lifts -- that later motion belongs to the pick, not the pull.)
            speed = np.concatenate([np.abs(np.diff(rack)), [0.0]])
            rack_start = self._sustained_start(speed > self.RACK_MOVE_SPEED, 0, self.RACK_SUSTAIN)
            if rack_start is not None:
                t_rack_out = self._sustained_start(speed <= self.RACK_MOVE_SPEED,
                                                   rack_start + self.RACK_SUSTAIN, self.RACK_SETTLE_W)
                if t_rack_out is not None and t_rack_out > cursor:
                    t_rack_out = int(min(t_rack_out, T - 1))
                    emit(t_rack_out, SUBTASK_RACK, STAGE_EXECUTE, "pull out the toaster oven rack")
                    anchor = t_rack_out

        # --- bread: off the (extended) rack -> bowl. Re-anchor at the rack-out frame; the
        # bread rode out with the rack, so its t_pick is the lift AFTER the rack is out. ---
        rest = np.median(bread[anchor: min(anchor + self.REST_W, T)], axis=0)
        bdisp = np.linalg.norm(bread - rest, axis=1)
        t_bread_pick = self._sustained_start(bdisp >= self.MOVE_TOL, anchor, self.MOVE_SUSTAIN)
        if t_bread_pick is None:
            t_bread_pick = max(T - 1, 1)
        t_bread_pick = max(t_bread_pick, cursor)
        t_bread_placed = self._place_by_settle(bread, t_bread_pick, T)
        emit(t_bread_pick, MOVE_BREAD, STAGE_PICK, f"pick up the {self.bread} from the toaster oven")
        emit(t_bread_placed, MOVE_BREAD, STAGE_PLACE, f"place the {self.bread} in the bowl")

        # --- bowl: counter -> dining counter (re-anchor FORWARD once the bread is in) ---
        brest = np.median(bowl[t_bread_placed: min(t_bread_placed + self.REST_W, T)], axis=0)
        bwdisp = np.linalg.norm(bowl - brest, axis=1)
        t_bowl_pick = self._sustained_start(bwdisp >= self.MOVE_TOL, t_bread_placed, self.MOVE_SUSTAIN)
        if t_bowl_pick is None:
            mv = np.where(bwdisp[t_bread_placed:] >= self.MOVE_TOL)[0]
            t_bowl_pick = t_bread_placed + int(mv[0]) if mv.size else min(t_bread_placed + 1, T - 1)
        t_bowl_pick = max(t_bowl_pick, cursor)
        gd_bowl = np.linalg.norm(grip - bowl, axis=1)
        t_bowl_placed = self._release(gd_bowl, t_bowl_pick, T)

        emit(t_bowl_pick, MOVE_BOWL, STAGE_PICK, "pick up the bread bowl from the counter")
        moving = np.where(binst[t_bowl_pick: t_bowl_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_bowl_pick + int(moving[-1]) + 1, t_bowl_placed) if moving.size else t_bowl_pick
        t_arrive = min(t_arrive, max(t_bowl_placed - self.MIN_PICK_GAP, cursor))
        if t_arrive > cursor and base_disp(t_bowl_pick, t_arrive) >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the bread bowl to the dining counter")
        emit(t_bowl_placed, MOVE_BOWL, STAGE_PLACE, "place the bread bowl on the dining counter")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_bread_pick": int(t_bread_pick), "t_bread_placed": int(t_bread_placed),
            "t_bowl_pick": int(t_bowl_pick), "t_bowl_placed": int(t_bowl_placed),
            "n_segments": len(segments),
        })
        result.review = []
        return result

    def validate(self, result):
        self.extra_subtasks = {MOVE_BREAD, MOVE_BOWL}
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

        if count(SUBTASK_RACK) not in (0, 1):
            issues.append(f"expected 0-1 SlideToasterOvenRack, found {count(SUBTASK_RACK)}")
        if count(MOVE_BREAD, STAGE_PICK) != 1 or count(MOVE_BREAD, STAGE_PLACE) != 1:
            issues.append("expected 1 bread pick + 1 bread place")
        if count(MOVE_BOWL, STAGE_PICK) != 1 or count(MOVE_BOWL, STAGE_PLACE) != 1:
            issues.append("expected 1 bowl pick + 1 bowl place")
        if count(SUBTASK_NAV) not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {count(SUBTASK_NAV)}")
        return issues
