import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Take the saucepan lid out of the (already-open) cabinet and carry it across to
# the saucepan sitting on a lit stove burner, then set it on top. A single
# nav-interrupted pick-place. NavigateKitchen is a bank atomic; the cabinet->stove
# lid move has no exact bank name (the bank has CabinetToCounter) so it is an
# extra_subtask.
SUBTASK_MOVE = "PickPlaceCabinetToStove"
SUBTASK_NAV = "NavigateKitchen"

NAV_LANG = "carry the lid to the stove"
DONE_LANG = "task complete"


@register_labeler
class PlaceLidToBoilLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceLidToBoil composite task.

    Task: the saucepan is already on a lit burner (its knob starts on) with water
    in it; its lid sits in an ALREADY-OPEN cabinet. Take the lid out of the cabinet,
    carry it to the stove and set it on the saucepan to boil faster.

    Doc-canonical count = 3 (`mobile: Yes`): the lid's pick-place is INTERRUPTED by a
    navigation from the cabinet to the stove -> PickPlaceCabinetToStove[pick] ->
    NavigateKitchen -> PickPlaceCabinetToStove[place]. 4 segments with the nav
    (expected_done_idx = 3). The nav is ADAPTIVE (gated on MIN_NAV_DIST); the cabinet
    is across the kitchen from the stove, so the base drives on every demo observed.
    (Same shape as FillKettle without the OpenCabinet prefix -- the cabinet is
    already open and there is no lit-burner turn-on, the burner starts on.)

    The lid is extracted horizontally (its Z barely changes until the set-down), so
    boundaries use the lid's DISPLACEMENT and the gripper RELEASE:
      - t_carry (pick end): the lid first displaces MOVE_TOL from its rest (in-
                 cabinet) pose -> grasped and being pulled out / carried.
      - t_arrive (nav end / place start): the DESCENT ONSET -- the first frame, after
                 the base parks during the carry, where the lid starts dropping onto
                 the saucepan (same as FillKettle).
      - t_placed (place end): the gripper RELEASES the lid (grip->lid distance rises
                 above the carry hold level), after it arrives.
    """

    ENV_NAME = "PlaceLidToBoil"

    MOVE_TOL = 0.12  # lid displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_carry used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.05  # grip->lid rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    DESCEND_DZ = 0.03  # lid Z dropping this far below its carry height == descending
    DESCEND_SUSTAIN = 5
    MIN_NAV_DIST = 0.20  # peak base displacement over the carry to emit a nav
    REST_W = 20

    LID = "saucepan_auxiliary"

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.lid_body_id = env.obj_body_id[self.LID]
        self.eef_site_id = env.robots[0].eef_site_id["right"]

        self.base_qpos_adrs = []
        for nm in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
        ):
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
            "lid": env.sim.data.body_xpos[self.lid_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    # ---- helpers -------------------------------------------------------

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        lid = np.array([h["lid"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # pick -> carry: the lid is grasped and pulled out of the cabinet
        rest = np.median(lid[: self.REST_W], axis=0)
        disp = np.linalg.norm(lid - rest, axis=1)
        t_carry = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_carry = int(mv[0]) if mv.size else max(T - 1, 0)

        # place ends at the gripper RELEASE, after the lid arrives
        gd = np.linalg.norm(grip - lid, axis=1)
        hold_gd = float(np.median(gd[t_carry: min(t_carry + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_carry, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # carry -> place: the DESCENT ONSET -- the lid drops onto the saucepan after
        # the base parks during the carry (a lid-3D-arrival split would end the nav
        # only once the descent was nearly done).
        oz = lid[:, 2]
        moving = np.where(binst[t_carry: t_placed] > self.BASE_MOVE_TH)[0]
        t_base_stop = min(t_carry + int(moving[-1]) + 1, t_placed) if moving.size else t_carry
        w0 = max(t_carry, t_base_stop - self.REST_W)
        hold_z = float(np.median(oz[w0: t_base_stop + 1])) if t_base_stop > w0 else float(oz[t_base_stop])
        t_arrive = self._sustained_start(oz < hold_z - self.DESCEND_DZ, t_base_stop, self.DESCEND_SUSTAIN)
        if t_arrive is None:
            t_arrive = t_base_stop
        t_arrive = int(min(max(t_arrive, t_carry), t_placed))

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, "pick up the lid from the cabinet")
        if base_xy is not None and t_arrive > t_carry:
            off = base_xy[t_carry: t_arrive] - base_xy[t_carry]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, "place the lid on the saucepan")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "t_carry": t_carry,
            "t_arrive": t_arrive,
            "t_placed": t_placed,
            "nav_disp": round(nav_disp, 2),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

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
            issues.append(f"expected 1 lid pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 lid place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
