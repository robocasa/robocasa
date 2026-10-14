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


# Carry the tongs from the counter across to the counter by the oven and set them
# down next to the plate. A single nav-interrupted pick-place. The counter->counter
# move has no exact bank name, so it is an extra_subtask; NavigateKitchen is a bank
# atomic.
SUBTASK_MOVE = "PickPlaceCounterToCounter"
SUBTASK_NAV = "NavigateKitchen"

NAV_LANG = "carry the tongs to the counter by the oven"
DONE_LANG = "task complete"


@register_labeler
class PrepareBroilingStationLabeler(BaseSubtaskLabeler):
    """Labeler for the PrepareBroilingStation composite task.

    Task: a pair of tongs sits on the counter (where the robot starts); a plate is
    on the counter next to the oven across the kitchen. Pick up the tongs and set
    them down next to the plate.

    Doc-canonical count = 3 (`mobile: Yes`): the tongs' pick-place is INTERRUPTED by
    a navigation to the oven counter -> PickPlaceCounterToCounter[pick] ->
    NavigateKitchen -> [place]. 4 segments with the nav (expected_done_idx = 3). The
    nav is ADAPTIVE (gated on MIN_NAV_DIST); the plate counter is across the kitchen,
    so the base drives on every demo observed. Same single-move shape as
    DeliverBrewedCoffee (base-stop nav/place split).

    Boundaries:
      - t_carry (pick end): the tongs first displace MOVE_TOL from their rest pose.
      - t_arrive (nav end / place start): the frame the BASE stops driving. The nav
                 is the drive; once the base parks, the arm's final positioning +
                 lowering + release is the place.
      - t_placed (place end): the gripper RELEASES the tongs (grip->tongs distance
                 rises above the carry hold level).
    """

    ENV_NAME = "PrepareBroilingStation"

    MOVE_TOL = 0.12  # tongs displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_carry used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.05  # grip->tongs rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    MIN_NAV_DIST = 0.20  # peak base displacement over the carry to emit a nav
    REST_W = 20

    TONGS = "tongs"

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.tongs_body_id = env.obj_body_id[self.TONGS]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.tongs_lang = env.get_obj_lang(self.TONGS)
        except Exception:
            self.tongs_lang = "tongs"

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
            "tongs": env.sim.data.body_xpos[self.tongs_body_id].copy(),
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
        tongs = np.array([h["tongs"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # pick -> carry: the tongs are grasped and lifted off the counter
        rest = np.median(tongs[: self.REST_W], axis=0)
        disp = np.linalg.norm(tongs - rest, axis=1)
        t_carry = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_carry = int(mv[0]) if mv.size else max(T - 1, 0)

        # place ends at the gripper RELEASE, after the tongs arrive
        gd = np.linalg.norm(grip - tongs, axis=1)
        hold_gd = float(np.median(gd[t_carry: min(t_carry + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_carry, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # carry -> place: split when the BASE stops driving (the drive is the nav;
        # the arm's positioning + set-down + release once parked is the place).
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

        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.tongs_lang} from the counter")
        if base_xy is not None and t_arrive > t_carry:
            off = base_xy[t_carry: t_arrive] - base_xy[t_carry]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.tongs_lang} next to the plate")
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
            issues.append(f"expected 1 tongs pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 tongs place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
