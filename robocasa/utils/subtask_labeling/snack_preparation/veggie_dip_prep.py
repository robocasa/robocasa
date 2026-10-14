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


# Two vegetables and a bowl sit on the counter; move all three onto a tray (where the robot
# spawns). Counter -> tray has no bank name -> PickPlaceToTray (extension). Mobile, but in
# practice local (everything on one counter); an optional NavigateKitchen guards the rare
# base drive.
SUBTASK_MOVE = "PickPlaceToTray"
SUBTASK_NAV = "NavigateKitchen"

DONE_LANG = "task complete"

OBJS = ("cucumber", "carrot", "bowl")


@register_labeler
class VeggieDipPrepLabeler(BaseSubtaskLabeler):
    """Labeler for the VeggieDipPrep composite task.

    Task (mobile): a cucumber, a carrot and a bowl sit on the counter, spread away from a
    tray (where the robot spawns). Fetch each and place it on the tray -- the robot drives
    OUT to each item and back, so most items get both a leading approach nav and a carry
    nav.

    Doc-canonical count = 3 (`mobile: Yes`), action-level: one counter->tray move per item.
    ORDER by first move; per item, up to four segments:
      NavigateKitchen [navigate]  -> navigate to the item (approach; if base drives >=MIN_NAV_DIST)
      PickPlaceToTray [pick]      -> pick up the item from the counter
      NavigateKitchen [navigate]  -> carry the item to the tray (if base drives >=MIN_NAV_DIST)
      PickPlaceToTray [place]     -> place the item on the tray
    plus a terminal done.

    The approach nav is gated on DISTANCE, not the object index: the robot starts at the
    TRAY (the destination, not the first item), so even the first item can be a real fetch
    (demo_70 cucumber ~2m -> nav) or already adjacent (demo_90 bowl ~0m -> no nav).

    Per item: t_pick = first sustained displacement re-anchored at the segment start; the
    approach nav ends at the last base-motion frame before t_pick; t_place = the gripper
    RELEASE.
    """

    ENV_NAME = "VeggieDipPrep"

    MOVE_TOL = 0.08
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 12
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in OBJS if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.noun = {}
        for o in self.names:
            try:
                self.noun[o] = f"the {env.get_obj_lang(o)}"
            except Exception:
                self.noun[o] = f"the {o}"
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
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
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

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        first_move = {}
        for o in self.names:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            dsp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted([o for o in self.names if first_move[o] < T], key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if base_xy is not None and b > a else 0.0)

        n_moved = 0
        for o in order:
            nn = self.noun[o]
            r = np.median(opos[o][max(cursor - self.REST_W, 0): cursor + 1], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_pick = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_pick is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_pick = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_pick = max(t_pick, cursor)
            gd = np.linalg.norm(grip - opos[o], axis=1)
            hold_gd = float(np.median(gd[t_pick: min(t_pick + self.HOLD_W, T)]))
            t_placed = self._sustained_start(gd > hold_gd + self.RELEASE_MARGIN,
                                             t_pick + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1
            # leading approach nav: the base drives from the current spot out to the
            # item before the grab (t_pick = first object motion), so that drive is
            # currently absorbed into the pick. Split it out as its own NavigateKitchen
            # when it is a real trip (>=MIN_NAV_DIST). The robot starts at the TRAY (the
            # destination), so even the FIRST item can be a real fetch (demo_70 cucumber
            # ~2m) or already adjacent (demo_90 bowl ~0m) -- the distance gate, not the
            # object index, decides (so no k>=1 suppression here).
            appr = np.where(binst[cursor: t_pick] > self.BASE_MOVE_TH)[0]
            t_appr_end = cursor + int(appr[-1]) + 1 if appr.size else cursor
            t_appr_end = min(t_appr_end, max(t_pick - self.MIN_PICK_GAP, cursor))
            # qualify the approach with a landmark (the bowl) so "the counter" is never
            # bare; the bowl itself is the landmark, so it gets no redundant qualifier.
            appr_lang = f"navigate to {nn}" if o == "bowl" else f"navigate to {nn} by the bowl"
            if t_appr_end > cursor and base_disp(cursor, t_appr_end) >= self.MIN_NAV_DIST:
                emit(t_appr_end, SUBTASK_NAV, STAGE_NAVIGATE, appr_lang)
            emit(t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up {nn} from the counter")
            moving = np.where(binst[t_pick: t_placed] > self.BASE_MOVE_TH)[0]
            t_arrive = min(t_pick + int(moving[-1]) + 1, t_placed) if moving.size else t_pick
            t_arrive = min(t_arrive, max(t_placed - self.MIN_PICK_GAP, cursor))
            if t_arrive > cursor and base_disp(t_pick, t_arrive) >= self.MIN_NAV_DIST:
                emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, f"carry {nn} to the tray")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place {nn} on the tray")
            n_moved += 1

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 items moved")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved, "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_MOVE}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        picks = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PICK)))
        places = int(np.sum((names == SUBTASK_MOVE) & (stages == STAGE_PLACE)))
        if picks != 3:
            issues.append(f"expected 3 picks, found {picks}")
        if places != 3:
            issues.append(f"expected 3 places, found {places}")
        return issues
