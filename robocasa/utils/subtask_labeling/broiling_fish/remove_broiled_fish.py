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


# Open the oven, slide the rack out, take the (broiled) fish off the rack and carry
# it to the plate on the counter. OpenOven, SlideOvenRack and NavigateKitchen are
# bank atomics; the oven->counter fish move has no exact bank name so it is an
# extra_subtask.
SUBTASK_OPEN = "OpenOven"
SUBTASK_RACK = "SlideOvenRack"
SUBTASK_MOVE = "PickPlaceOvenToCounter"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the oven door"
RACK_LANG = "slide the oven rack out"
NAV_LANG = "carry the fish to the counter with the plate"
DONE_LANG = "task complete"


@register_labeler
class RemoveBroiledFishLabeler(BaseSubtaskLabeler):
    """Labeler for the RemoveBroiledFish composite task.

    Task: a broiled fish is inside a CLOSED oven (rack slid in). Open the oven door,
    slide the rack out, take the fish off the rack and place it on the plate on the
    counter.

    Doc-canonical count = 5 (`mobile: Yes`): OpenOven + SlideOvenRack (out) + the
    fish's pick-place, which is INTERRUPTED by a navigation to the counter ->
    PickPlaceOvenToCounter[pick] -> NavigateKitchen -> [place]. 6 segments with the
    nav (expected_done_idx = 5); the nav is adaptive.

    Boundaries:
      - t_open:   the door joint reaches (near) fully open -- it starts closed (~0)
                  and swings to ~1.15. OpenOven ends here.
      - t_rack_out: the moving rack reaches its slid-OUT plateau (it starts in ~0 and
                  is pulled out; the fish rides out with it). SlideOvenRack ends here.
      - t_carry:  the fish is LIFTED off the (slid-out) rack -- its displacement from
                  its on-rack pose grows -> the pick ends. (Measured from the on-rack
                  pose, not the in-oven start, so the rack-ride out is not counted.)
      - t_arrive: the base stops driving -> nav ends, the arm's set-down is the place.
      - t_placed: the gripper releases the fish onto the plate.
    """

    ENV_NAME = "RemoveBroiledFish"

    OPEN_FRAC = 0.90  # door/rack >= this fraction of its peak == (near) fully open/out
    OPEN_SUSTAIN = 5
    MOVE_TOL = 0.12  # fish displaced this far off the rack == being lifted / carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08  # grip->fish rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    BASE_MOVE_TH = 0.003
    MIN_NAV_DIST = 0.20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.fish_body_id = env.obj_body_id["fish"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.fish_lang = env.get_obj_lang("fish")
        except Exception:
            self.fish_lang = "fish"

        oven = env.oven
        self.door_adr = env.sim.model.get_joint_qpos_addr(oven._joint_names["door"])
        prefix = oven.naming_prefix
        rack_names = [
            n for n in env.sim.model.joint_names
            if n.startswith(prefix) and "rack" in n and n.endswith("_joint")
        ]
        self.rack_adrs = [env.sim.model.get_joint_qpos_addr(n) for n in rack_names]

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
            "fish": env.sim.data.body_xpos[self.fish_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "door": float(env.sim.data.qpos[self.door_adr]),
            "racks": np.array([float(env.sim.data.qpos[a]) for a in self.rack_adrs]),
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
        fish = np.array([h["fish"] for h in history])
        grip = np.array([h["grip"] for h in history])
        door = np.array([h["door"] for h in history])
        racks = np.array([h["racks"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # open the oven: door swings to (near) fully open
        peak_door = float(door.max())
        t_open = self._sustained_start(door >= self.OPEN_FRAC * peak_door, 0, self.OPEN_SUSTAIN)
        if t_open is None:
            t_open = int(np.argmax(door))

        # slide the rack out: the moving rack reaches its slid-out plateau
        rrange = racks.max(axis=0) - racks.min(axis=0)
        mover = int(np.argmax(rrange)) if racks.shape[1] else 0
        rack = racks[:, mover] if racks.shape[1] else np.zeros(T)
        peak_rack = float(rack.max())
        t_rack_out = self._sustained_start(rack >= self.OPEN_FRAC * peak_rack, t_open, self.OPEN_SUSTAIN)
        if t_rack_out is None:
            t_rack_out = t_open

        # pick: the fish is lifted OFF the slid-out rack. Measure displacement from
        # its on-rack pose (median around t_rack_out) so the rack-ride out doesn't
        # count as the carry.
        w0 = min(t_rack_out, T - self.REST_W) if T > self.REST_W else 0
        on_rack = np.median(fish[max(w0, 0): max(w0, 0) + self.REST_W], axis=0)
        fdisp = np.linalg.norm(fish - on_rack, axis=1)
        t_carry = self._sustained_start(fdisp >= self.MOVE_TOL, t_rack_out, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(fdisp[t_rack_out:] >= self.MOVE_TOL)[0]
            t_carry = t_rack_out + int(mv[0]) if mv.size else min(t_rack_out + 1, T - 1)

        # place ends at the gripper release
        gd = np.linalg.norm(grip - fish, axis=1)
        hold_gd = float(np.median(gd[t_carry: min(t_carry + self.HOLD_W, T)]))
        t_placed = self._sustained_start(
            gd > hold_gd + self.RELEASE_MARGIN, t_carry + self.MIN_CARRY, self.RELEASE_SUSTAIN
        )
        if t_placed is None:
            t_placed = T - 1

        # nav/place split: the base stops driving (carry) then the arm sets down
        moving = np.where(binst[t_carry: t_placed] > self.BASE_MOVE_TH)[0]
        t_arrive = min(t_carry + int(moving[-1]) + 1, t_placed) if moving.size else t_carry
        t_arrive = int(min(max(t_arrive, t_carry), t_placed))

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(t_rack_out, SUBTASK_RACK, STAGE_EXECUTE, RACK_LANG)
        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.fish_lang} from the oven")
        if base_xy is not None and t_arrive > t_carry:
            off = base_xy[t_carry: t_arrive] - base_xy[t_carry]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.fish_lang} on the plate")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if peak_door < 0.3:
            review.append("door never opened")
        if float(rrange[mover]) < 0.05:
            review.append("rack never slid out")

        debug = {
            "t_open": t_open, "t_rack_out": t_rack_out, "t_carry": t_carry,
            "t_arrive": t_arrive, "t_placed": t_placed, "nav_disp": round(nav_disp, 2),
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
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

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenOven, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_RACK) != 1:
            issues.append(f"expected 1 SlideOvenRack, found {count(SUBTASK_RACK)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 fish pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 fish place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
