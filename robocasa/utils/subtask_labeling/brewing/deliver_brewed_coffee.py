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


# Take the brewed coffee (a mug) from the coffee machine and carry it across the
# kitchen to the dining counter, setting it next to the breakfast plate. A single
# nav-interrupted pick-place. CoffeeServeMug ("pick the mug from under the coffee
# machine dispenser and place it on the counter") and NavigateKitchen are bank
# atomics -- no extra_subtasks.
SUBTASK_MOVE = "CoffeeServeMug"
SUBTASK_NAV = "NavigateKitchen"

COFFEE_LANG = "brewed coffee"  # get_obj_lang gives "mug"; the task calls it this
NAV_LANG = "carry the brewed coffee to the dining counter"
DONE_LANG = "task complete"


@register_labeler
class DeliverBrewedCoffeeLabeler(BaseSubtaskLabeler):
    """Labeler for the DeliverBrewedCoffee composite task.

    Task: a mug of brewed coffee sits on the coffee machine; a breakfast plate is
    set on the dining counter across the kitchen. Pick up the mug and place it next
    to the breakfast plate on the dining counter.

    Doc-canonical count = 3 (`mobile: Yes`): the mug's pick-place is INTERRUPTED by a
    navigation from the coffee machine to the dining counter -> CoffeeServeMug[pick]
    -> NavigateKitchen -> CoffeeServeMug[place]. 4 segments with the nav
    (expected_done_idx = 3). The nav is ADAPTIVE (gated on MIN_NAV_DIST); the dining
    counter is across the kitchen, so the base drives on every demo observed. Same
    single-move shape as PlaceLidToBoil.

    The mug is lifted then set down, but boundaries use its DISPLACEMENT (robust to
    lift direction) and the gripper RELEASE:
      - t_carry (pick end): the mug first displaces MOVE_TOL from its rest (on-
                 machine) pose -> grasped and being lifted / carried.
      - t_arrive (nav end / place start): the frame the BASE stops driving. The nav
                 is the drive to the dining counter; once the base parks, the arm's
                 final positioning + lowering + release is the place (the base often
                 parks a couple seconds before the mug finishes descending).
      - t_placed (place end): the gripper RELEASES the mug (grip->mug distance rises
                 above the carry hold level), after it arrives.
    """

    ENV_NAME = "DeliverBrewedCoffee"

    MOVE_TOL = 0.12  # mug displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after t_carry used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.05  # grip->mug rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    MIN_NAV_DIST = 0.20  # peak base displacement over the carry to emit a nav
    REST_W = 20

    MUG = "brewed_coffee"

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.mug_body_id = env.obj_body_id[self.MUG]
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

        self.extra_subtasks = set()
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "mug": env.sim.data.body_xpos[self.mug_body_id].copy(),
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
        mug = np.array([h["mug"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # pick -> carry: the mug is grasped and lifted off the coffee machine
        rest = np.median(mug[: self.REST_W], axis=0)
        disp = np.linalg.norm(mug - rest, axis=1)
        t_carry = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(disp >= self.MOVE_TOL)[0]
            t_carry = int(mv[0]) if mv.size else max(T - 1, 0)

        # place ends at the gripper RELEASE, after the mug arrives
        gd = np.linalg.norm(grip - mug, axis=1)
        hold_gd = float(np.median(gd[t_carry: min(t_carry + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_carry, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # carry -> place: split when the BASE stops driving. The nav is the drive to
        # the dining counter; once the base parks, the arm does the final positioning
        # + lowering + release, which is the place. (Per user review: the base often
        # parks a couple seconds before the mug finishes descending -- demo_71 base
        # stops ~0:16 while the arm keeps moving the mug to ~0:19 -- and that whole
        # tail reads as the place, so end the nav at the base stop, not the descent.)
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

        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, f"pick up the {COFFEE_LANG} from the coffee machine")
        if base_xy is not None and t_arrive > t_carry:
            off = base_xy[t_carry: t_arrive] - base_xy[t_carry]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
             f"place the {COFFEE_LANG} next to the breakfast plate on the dining counter")
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
            issues.append(f"expected 1 mug pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 mug place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
