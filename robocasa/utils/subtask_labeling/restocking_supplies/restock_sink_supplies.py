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


# Restock the sink supplies: the drawer starts part-open; the robot first pulls it
# fully open, then moves a sponge and a soap dispenser onto the counter beside the
# sink (the soap dispenser ends upright). OpenDrawer and PickPlaceDrawerToCounter are
# bank atomics; NavigateKitchen is a bank atomic (the carry is adaptive).
SUBTASK_OPEN = "OpenDrawer"
SUBTASK_MOVE = "PickPlaceDrawerToCounter"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "fully open the drawer"
DONE_LANG = "task complete"


@register_labeler
class RestockSinkSuppliesLabeler(BaseSubtaskLabeler):
    """Labeler for the RestockSinkSupplies composite task.

    Task (mobile, usually local): a sponge and a soap dispenser sit in an open drawer.
    Move both onto the counter next to the sink (the soap dispenser upright). The robot
    starts at the drawer.

    Doc-canonical count = 2 (`mobile: Yes`): the two pick-places (the drawer is
    pre-opened, so no open/close). ORDER is resolved from when each item starts moving.
    Each splits pick / place, with an adaptive carry nav emitted only if the base
    actually travels. 2 pick-places + done.

    Per item: t_pick = first sustained displacement from a rest re-anchored at the
    segment start; t_place = the gripper RELEASE.
    """

    ENV_NAME = "RestockSinkSupplies"

    # object -> (word, upright_prefix)
    ITEMS = {
        "sponge": ("the sponge", ""),
        "soap_dispenser": ("the soap dispenser", "upright "),
    }

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_TH = 0.12        # gripper-object distance below this == holding it
    GRASP_SUSTAIN = 8     # a real grasp is held at least this many frames
    VEL_TH = 0.004        # object speed above this == still being carried
    BASE_MOVE_TH = 0.008
    CARRY_SUSTAIN = 6     # base moving this many frames == the carry has begun
    MIN_NAV_DIST = 0.30
    REST_W = 15
    # drawer-open detection (normalized slide value in [0, 1])
    OPEN_ABS = 0.70
    OPEN_MIN = 0.20       # peak must exceed this for an open to count
    OPEN_RISE_MIN = 0.06  # ...and the drawer must slide at least this far from its start
    OPEN_SUSTAIN = 5

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in self.ITEMS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])
        # drawer slide joint(s) -- resolved per demo (layout/model varies per episode)
        dr = getattr(env, "drawer", None)
        jn = [n for n in env.sim.model.joint_names
              if n and dr is not None and dr.name in n and ("drawer" in n.lower() or "slide" in n.lower())]
        if not jn:
            jn = [n for n in env.sim.model.joint_names
                  if n and ("drawer" in n.lower() or "slide" in n.lower())]
        self.door_adrs, self.door_scales = [], []
        for j in jn:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            self.door_scales.append(max(abs(float(hi)), abs(float(lo)), 1e-6))
            self.door_adrs.append(adr)
        self.extra_subtasks = set()  # OpenDrawer + PickPlaceDrawerToCounter are bank atomics
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "drawer": float(max((abs(float(env.sim.data.qpos[a])) / s
                                 for a, s in zip(self.door_adrs, self.door_scales)), default=0.0)),
        }
        for o in self.names:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
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

    def _grasp_interval(self, held, disp, T):
        """Among sustained held runs (gripper on the object), return the one during
        which the object is actually carried (max displacement > MOVE_TOL, largest
        wins). Returns (start, end) or None. Filters out the brief 'gripper passes
        near' blips and the collateral bump from grabbing the other item."""
        best, best_disp = None, self.MOVE_TOL
        t = 0
        while t < T:
            if held[t]:
                a = t
                while t < T and held[t]:
                    t += 1
                b = t - 1
                if b - a + 1 >= self.GRASP_SUSTAIN:
                    md = float(disp[a: b + 1].max())
                    if md > best_disp:
                        best, best_disp = (a, b), md
            else:
                t += 1
        return best

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        drawer = np.array([h["drawer"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        def base_disp(a, b):
            if base_xy is None or b <= a:
                return 0.0
            return float(np.max(np.linalg.norm(base_xy[a:b] - base_xy[a], axis=1)))

        # Attribute each supply by GRIPPER PROXIMITY, not displacement: the two items
        # sit together in the drawer, so grabbing one nudges the other (a false
        # first-move at the same frame). The carried item is the one the gripper holds.
        plans = {}  # o -> (t_pick, t_arrive|None, t_placed)
        for o in self.names:
            gd = np.linalg.norm(grip - opos[o], axis=1)
            rest = np.median(opos[o][: self.REST_W], axis=0)
            disp = np.linalg.norm(opos[o] - rest, axis=1)
            vel = np.concatenate([[0.0], np.linalg.norm(np.diff(opos[o], axis=0), axis=1)])
            g = self._grasp_interval(gd < self.HOLD_TH, disp, T)
            if g is None:
                continue
            g0, g1 = g
            # real lift = re-anchor at the grasp, so a pre-grasp nudge (an item slid
            # aside inside the drawer while the other is grabbed) does not count.
            rest_g0 = np.median(opos[o][max(g0 - 5, 0): g0 + 5], axis=0)
            disp_g0 = np.linalg.norm(opos[o] - rest_g0, axis=1)
            t_lift = self._sustained_start(disp_g0 >= self.MOVE_TOL, g0, self.MOVE_SUSTAIN)
            t_lift = t_lift if t_lift is not None else g0
            # place = the object settles (last carried motion) or the gripper releases,
            # whichever comes first (the gripper sometimes lingers after setting down).
            rel = self._sustained_start(gd > self.HOLD_TH, t_lift + 1, 3)
            mv = np.where(vel[t_lift: g1 + 1] > self.VEL_TH)[0]
            settle = (t_lift + int(mv[-1]) + 1) if mv.size else g1
            t_placed = min(rel if rel is not None else T - 1, settle)
            t_placed = max(t_placed, t_lift + 1)
            # If the base carries the object, the pick ends when the DRIVE begins --
            # the lift-out of the drawer is part of the pick, not the carry.
            t_arrive = None
            t_pick = t_lift
            cs_rel = self._sustained_start(binst[t_lift: t_placed] > self.BASE_MOVE_TH,
                                           0, self.CARRY_SUSTAIN)
            if cs_rel is not None:
                cs = t_lift + cs_rel
                moving = np.where(binst[cs: t_placed] > self.BASE_MOVE_TH)[0]
                arrive = min(cs + int(moving[-1]) + 1, t_placed) if moving.size else cs
                if arrive > cs and base_disp(cs, arrive) >= self.MIN_NAV_DIST:
                    t_pick, t_arrive = cs, arrive
            plans[o] = (t_pick, t_arrive, t_placed, g0)

        order = sorted(plans.keys(), key=lambda o: plans[o][0])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        # OpenDrawer first: the drawer starts part-open and the robot pulls it the
        # rest of the way before taking anything out. Emit it if the drawer really
        # slid (rose >= OPEN_RISE_MIN), ending when the robot reaches the first item
        # (the drawer may keep drifting open during the grasp -- that's not the open).
        first_grasp = plans[order[0]][3] if order else T
        peak = float(drawer.max()) if T else 0.0
        rose = peak - float(drawer[0]) if T else 0.0
        if peak >= self.OPEN_MIN and rose >= self.OPEN_RISE_MIN:
            lvl = min(self.OPEN_ABS, 0.9 * peak)
            t_open = self._sustained_start(drawer >= lvl, 0, self.OPEN_SUSTAIN)
            if t_open is None:
                t_open = int(np.argmax(drawer))
            t_open = int(min(t_open, max(first_grasp - 1, 0)))
            if t_open > 0:
                emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        elif rose >= self.OPEN_RISE_MIN:
            review.append("drawer moved but open not cleanly detected -- verify")

        n_moved = 0
        for o in order:
            word, upright = self.ITEMS[o]
            t_pick, t_arrive, t_placed, _ = plans[o]
            t_pick = max(t_pick, cursor)
            t_placed = max(t_placed, t_pick + 1)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {word} from the drawer")
            if t_arrive is not None and t_arrive > cursor:
                emit(min(t_arrive, t_placed), SUBTASK_NAV, STAGE_NAVIGATE,
                     f"carry {word} to the counter next to the sink")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE,
                 f"place {word} {upright}on the counter next to the sink")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        if n_moved != 2:
            review.append(f"only {n_moved}/2 supplies moved (gripper-proximity grasp)")

        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved,
            "plans": {o: [int(plans[o][0]), plans[o][1], int(plans[o][2])] for o in plans},
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        # OpenDrawer + PickPlaceDrawerToCounter are bank atomics; keep the manual
        # short-circuit so a hand-reviewed override skips the structural checks.
        self.extra_subtasks = set()
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        names = result.subtask_name[firsts]
        n_open = int(np.sum(names == SUBTASK_OPEN))
        if n_open not in (0, 1):
            issues.append(f"expected 0-1 OpenDrawer, found {n_open}")
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append(f"expected 2 picks, found {int(np.sum(stages == STAGE_PICK))}")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append(f"expected 2 places, found {int(np.sum(stages == STAGE_PLACE))}")
        return issues
